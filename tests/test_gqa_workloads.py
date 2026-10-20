"""Independent shape/identity oracles for grouped LH workload lowering."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class GqaWorkloadsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'config.json'
        self.raw = dict(model_type='qwen2', hidden_size=192,
                        intermediate_size=384, num_attention_heads=12,
                        num_key_value_heads=4, num_hidden_layers=2,
                        max_position_embeddings=4096)

    def model(self, **changes):
        from hbnpu_model import ModelConfig
        self.path.write_text(json.dumps(dict(self.raw, **changes)))
        return ModelConfig.from_json(self.path)

    def test_projection_widths_and_cache_size(self):
        from hbnpu_gqa import build_workloads
        model = self.model()
        ops = {r['operation']: r for r in build_workloads(model, 'decode', 5, layer=0)}
        self.assertEqual(model.kv_cache_bytes(5), 2560)
        self.assertEqual(model.head_dim, 16)
        self.assertEqual(ops['q_proj']['shape'], (1,192,192))
        self.assertEqual(ops['k_proj']['shape'], (1,192,64))
        self.assertEqual(ops['v_proj']['shape'], (1,192,64))
        self.assertEqual(ops['gate_proj']['shape'], (1,192,384))
        self.assertEqual(ops['up_proj']['shape'], (1,192,384))
        self.assertEqual(ops['down_proj']['shape'], (1,384,192))

    def test_group_identity_mha_mqa_and_odd_groups(self):
        from hbnpu_gqa import build_workloads
        for heads, kv, group in [(4,4,1),(12,4,3),(16,4,4),(28,4,7),(32,4,8),(36,4,9),(7,1,7)]:
            model = self.model(hidden_size=heads*128, num_attention_heads=heads,
                               num_key_value_heads=kv)
            records = [r for r in build_workloads(model,'decode',143,layer=0) if r['operation']=='qk']
            self.assertEqual(len(records), kv)
            self.assertEqual([h for r in records for h in r['query_heads']], list(range(heads)))
            self.assertEqual(len({r['stream_tensor'] for r in records}),kv)
            self.assertTrue(all(r['shape']==(group,128,143) for r in records))

    def test_prefill_causal_prefix_and_batches(self):
        from hbnpu_gqa import build_workloads
        records = [r for r in build_workloads(self.model(),'prefill',3,batch=2,layer=0) if r['operation']=='qk']
        self.assertEqual(len(records),24)
        for r in records:
            self.assertEqual(r['shape'], (3,16,r['token']+1))
            self.assertEqual(r['valid_kv_range'], (0,r['token']+1))
        self.assertNotEqual(records[0]['stream_tensor'],records[12]['stream_tensor'])

    def test_invalid_configs_and_partitioning(self):
        from hbnpu_model import ModelConfig
        for changes in [dict(num_attention_heads=13), dict(num_key_value_heads=5),
                        dict(num_key_value_heads=0),dict(hidden_size=True),
                        dict(model_type='qwen2_moe'),dict(num_local_experts=8),
                        dict(use_sliding_window=True),dict(hidden_act='gelu')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.model(**changes)
        self.model()
        sharded = ModelConfig.from_json(self.path,tp=2,pp=2)
        self.assertEqual((sharded.local_query_heads,sharded.local_kv_heads,sharded.local_layers),(6,2,1))
        with self.assertRaises(ValueError): ModelConfig.from_json(self.path,tp=3)
        with self.assertRaises(ValueError): ModelConfig.from_json(self.path,pp=3)

    def test_lh_padding_chunks_and_disjoint_regions(self):
        from hbnpu_gqa import build_workloads, lower_lh, RowAllocator
        model = self.model(hidden_size=1152,num_attention_heads=9,num_key_value_heads=1)
        record = next(r for r in build_workloads(model,'decode',143,layer=0) if r['operation']=='qk')
        alloc = RowAllocator(Path(__file__).resolve().parents[1]/'configs/HBM2_8Gb_x128_pim.ini')
        lowered = lower_lh(record,alloc)
        self.assertEqual([p['valid_shape'] for p in lowered],[(8,128,143),(1,128,143)])
        self.assertEqual([p['query_heads'] for p in lowered],[list(range(8)),[8]])
        self.assertEqual([(p['config'].m,p['config'].n,p['config'].k) for p in lowered],[(16,8,128),(16,1,128)])
        self.assertEqual(lowered[0]['config'].input_row,lowered[1]['config'].input_row)
        self.assertNotEqual(lowered[0]['config'].output_row,lowered[1]['config'].output_row)
        for p in lowered:
            c = p['config']
            self.assertEqual((c.df,c.mcf,c.ucf),(1,16,1))
            self.assertEqual(p['padded_shape'],(8,128,256))
            self.assertEqual(p['output_vectors_per_channel'],2)
        regions = list(alloc.regions.values())
        for i,a in enumerate(regions):
            for b in regions[i+1:]:
                self.assertTrue(a['row']+a['rows']<=b['row'] or b['row']+b['rows']<=a['row'])
        with self.assertRaises(ValueError): alloc.allocate('too_large',10**20)

    def test_grouped_algebra_matches_individual_heads(self):
        from hbnpu_gqa import build_workloads
        model = self.model(hidden_size=12,num_attention_heads=6,num_key_value_heads=2)
        q = [[h+1,2-h] for h in range(6)]
        keys = [[[1,2],[3,1],[2,-1]], [[-1,2],[0,3],[4,1]]]
        values = [[[1,0],[2,3],[4,1]], [[2,1],[1,0],[3,2]]]
        records = list(build_workloads(model,'decode',3,layer=0))
        def mul(a,b): return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
        for r in (r for r in records if r['operation']=='qk'):
            kv = r['kv_head']
            scores = mul([q[h] for h in r['query_heads']],list(zip(*keys[kv])))
            # No softmax simulation: deterministic score values test SV algebra only.
            output = mul(scores,values[kv])
            for idx,h in enumerate(r['query_heads']):
                single = mul(mul([q[h]],list(zip(*keys[h//3]))),values[h//3])
                self.assertEqual(output[idx],single[0])
            sv = next(s for s in records if s['operation']=='sv' and s['kv_head']==kv)
            self.assertEqual(sv['shape'],(3,3,2))
            self.assertEqual(sv['query_heads'],r['query_heads'])

    def test_sv_binds_normalized_qk_output_and_stream_orientation(self):
        from hbnpu_gqa import build_workloads
        records = list(build_workloads(self.model(),'decode',3,layer=0))
        qk = next(r for r in records if r['operation']=='qk')
        sv = next(r for r in records if r['operation']=='sv')
        self.assertEqual(qk['stream_orientation'],'K_transposed')
        self.assertEqual(sv['stream_orientation'],'V')
        self.assertEqual(sv['resident_from'],dict(tensor=qk['output_tensor'],operation='softmax',executed=False))

    def test_partial_reduction_padding_preserves_valid_results(self):
        from hbnpu_gqa import build_workloads, lower_lh, RowAllocator
        model = self.model(hidden_size=387,num_attention_heads=3,num_key_value_heads=1)
        record = next(r for r in build_workloads(model,'decode',143,layer=0) if r['operation']=='qk')
        allocator = RowAllocator(Path(__file__).resolve().parents[1]/'configs/HBM2_8Gb_x128_pim.ini')
        program = lower_lh(record,allocator)[0]
        c = program['config']
        self.assertEqual((c.m,c.n,c.k),(16,3,256))
        self.assertEqual(program['valid_shape'],(3,129,143))
        a = [[(i+j)%5-2 for j in range(129)] for i in range(3)]
        b = [[(i-j)%7-3 for j in range(143)] for i in range(129)]
        padded_a = [row+[0]*(c.k-129) for row in a]
        padded_b = [row+[0]*(16*c.m-143) for row in b]+[[0]*(16*c.m) for _ in range(c.k-129)]
        for i in range(c.n):
            for j in range(16*c.m):
                actual = sum(padded_a[i][k]*padded_b[k][j] for k in range(c.k))
                expected = sum(a[i][k]*b[k][j] for k in range(129)) if j<143 else 0
                self.assertEqual(actual,expected)


if __name__ == '__main__': unittest.main()
