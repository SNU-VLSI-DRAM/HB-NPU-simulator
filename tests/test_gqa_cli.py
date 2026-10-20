"""Exercise artifact generation and failure-before-write guarantees."""
import configparser
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hbnpu_instruction import decode_packet
from hbnpu_model import ModelConfig


class GqaCliTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.output = self.base/'generated'
        self.model = ROOT/'tests/fixtures/models/qwen2.5-7b.json'
        self.hardware = ROOT/'configs/HBM2_8Gb_x128_pim.ini'

    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-B',str(ROOT/'gen_gqa_workload.py'),
                               '--model-config',str(self.model),'--hardware',str(self.hardware),
                               '--phase','decode','--sequence-length','143',
                               '--output',str(self.output),*args],capture_output=True,text=True)

    def test_manifest_trace_identity_and_model_fixtures(self):
        result = self.run_cli('--operations','qk','sv')
        self.assertEqual(result.returncode,0,result.stderr)
        manifest = json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(len(manifest['programs']),2)
        self.assertEqual(manifest['modeled_layers'],[0])
        self.assertIn('softmax',manifest['omitted_operations'])
        for p in manifest['programs']:
            words = [int(line.split()[0],16) for line in (self.output/p['trace']).read_text().splitlines()]
            config = decode_packet(words[:8])
            self.assertEqual(config.payload[0]&0xffffffff,p['config']['m'])
            self.assertEqual(config.payload[0]>>32,28)
            self.assertEqual(config.payload[5],0x40401)  # df=1,mcf=4,ucf=4
            self.assertEqual(len(p['members']),4)
            self.assertEqual([m['kv_head'] for m in p['members']],[0,1,2,3])
            for key in ('stream_tensor','resident_tensor','output_tensor','query_heads'):
                self.assertIsNone(p[key])  # Never silently identify a pack as its first member.
            self.assertEqual(p['padded_shape'][0],8)
        qwen = ModelConfig.from_json(self.model)
        mistral = ModelConfig.from_json(ROOT/'tests/fixtures/models/mistral-7b-v0.3.json')
        self.assertEqual((qwen.hidden_size,qwen.intermediate_size,qwen.head_dim),(3584,18944,128))
        self.assertEqual((mistral.query_heads,mistral.kv_heads,mistral.head_dim),(32,8,128))
        ini = configparser.ConfigParser()
        ini.read(self.output/'simulator.ini')
        self.assertEqual(ini['hbnpu']['execution_mode'],'tile_instructions')

    def test_canonical_head_batch_packing_and_split_fallback(self):
        self.hardware = ROOT/'configs/HBNPU_8Gb.ini'
        for batch,expected in [(1,(512,28,128,4,4)),
                               (2,(1024,56,128,2,8)),
                               (4,(2048,112,128,1,16)),
                               (5,(2048,112,128,1,16))]:
            with self.subTest(batch=batch):
                self.output = self.base/('batch'+str(batch))
                result = self.run_cli('--operations','qk','--batch',str(batch),
                                      '--sequence-length','2048')
                self.assertEqual(result.returncode,0,result.stderr)
                manifest = json.loads((self.output/'manifest.json').read_text())
                programs = manifest['programs']
                c = programs[0]['config']
                self.assertEqual(tuple(c[k] for k in ('m','n','k','mcf','ucf')),expected)
                members = [m for p in programs for m in p['members']]
                self.assertEqual(sorted((m['batch'],m['kv_head']) for m in members),
                                 [(b,k) for b in range(batch) for k in range(4)])
                for p in programs:
                    self.assertEqual(sorted(b for m in p['members'] for b in m['banks']),list(range(16)))
                    self.assertEqual(p['config']['columns_per_burst'],64)
                if batch == 5:
                    self.assertEqual([len(p['members']) for p in programs],[16,4])

    def test_invalid_model_and_capacity_leave_no_output(self):
        invalid = self.base/'bad.json'
        invalid.write_text('{"model_type":"unknown"}')
        self.model = invalid
        result = self.run_cli()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('model',result.stderr)
        self.assertFalse(self.output.exists())

    def test_non_power_of_two_groups_and_query_chunks(self):
        self.model = self.base/'six-groups.json'
        self.model.write_text(json.dumps(dict(model_type='qwen2',hidden_size=6912,
            intermediate_size=256,num_attention_heads=54,num_key_value_heads=6,
            num_hidden_layers=1,max_position_embeddings=4096)))
        result = self.run_cli('--operations','qk')
        self.assertEqual(result.returncode,0,result.stderr)
        programs = json.loads((self.output/'manifest.json').read_text())['programs']
        self.assertEqual([p['config']['ucf'] for p in programs],[4,4,2,2])
        self.assertEqual([p['config']['n'] for p in programs],[32,4,16,2])
        self.assertEqual(programs[0]['config']['input_row'],programs[1]['config']['input_row'])
        identities = [(m['batch'],h) for p in programs for m in p['members'] for h in m['query_heads']]
        self.assertEqual(sorted(identities),[(0,h) for h in range(54)])

    def test_prefill_does_not_pack_different_query_positions(self):
        result = self.run_cli('--operations','qk','--phase','prefill','--sequence-length','3','--batch','2')
        self.assertEqual(result.returncode,0,result.stderr)
        programs = json.loads((self.output/'manifest.json').read_text())['programs']
        self.assertEqual(len(programs),3)
        self.assertEqual([p['config']['n'] for p in programs],[56,56,56])
        self.assertEqual([{m['token'] for m in p['members']} for p in programs],[{0},{1},{2}])

    def test_capacity_failure_leaves_no_output(self):
        self.model = ROOT/'tests/fixtures/models/qwen2.5-7b.json'
        small = self.base/'small.ini'
        small.write_text(self.hardware.read_text().replace('rows = 32768','rows = 1'))
        self.hardware = small
        result = self.run_cli('--operations','qk')
        self.assertNotEqual(result.returncode,0)
        self.assertIn('row capacity',result.stderr)
        self.assertFalse(self.output.exists())

    def test_existing_output_is_untouched(self):
        self.output.mkdir()
        sentinel = self.output/'keep'
        sentinel.write_text('untouched')
        result = self.run_cli()
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(list(self.output.iterdir()),[sentinel])
        self.assertEqual(sentinel.read_text(),'untouched')

    def test_unrepresentable_hardware_rejected_before_output(self):
        original = self.hardware.read_text()
        for index,(before,after) in enumerate([('columns = 64','columns = 66'),
                             ('rows = 32768','rows = 32769'),
                             ('rows = 32768','rows = 64'),
                             ('channel_size = 1024','channel_size = 3000'),
                             ('device_width = 128','device_width = 0'),
                             ('bankgroups = 4','bankgroups = 4\nbankgroup_enable = False')]):
            with self.subTest(after=after):
                self.output = self.base/('bad-'+str(index))
                self.hardware = self.base/'unsupported.ini'
                self.hardware.write_text(original.replace(before,after))
                result = self.run_cli('--operations','qk','sv')
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(self.output.exists())

    def test_default_emits_separate_gated_projections(self):
        result = self.run_cli('--tp','4')
        self.assertEqual(result.returncode,0,result.stderr)
        manifest = json.loads((self.output/'manifest.json').read_text())
        ops = {r['operation'] for r in manifest['workloads']}
        self.assertEqual(ops,{'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','qk','sv'})
        self.assertEqual(manifest['shard'],dict(tp_rank=0,pp_stage=0))


if __name__ == '__main__': unittest.main()
