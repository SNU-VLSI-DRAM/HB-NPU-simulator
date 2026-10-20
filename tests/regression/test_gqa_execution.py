"""Actual CPU/ISA/controller execution of grouped LH matrix workloads."""
from collections import Counter
import configparser
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gen_gqa_workload import generate
from compare_instruction_execution import run_one, validate_rows, validate_timing


class GqaExecutionTest(unittest.TestCase):
    binary = None

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.serial = 0

    def generated(self,group,length,**kwargs):
        self.serial += 1
        path = self.base/('model{}.json'.format(self.serial))
        path.write_text(json.dumps(dict(model_type='qwen2',hidden_size=group*128,
                                       intermediate_size=256,num_attention_heads=group,
                                       num_key_value_heads=1,num_hidden_layers=1,
                                       max_position_embeddings=8192)))
        return self.from_config(path,length,**kwargs)

    def from_config(self,path,length,phase='decode',operations=('qk','sv'),batch=1,
                    hardware='HBM2_8Gb_x128_pim.ini'):
        output = self.base/('generated{}'.format(self.serial))
        manifest = generate(path,ROOT/'configs'/hardware,output,
                            phase=phase,sequence_length=length,operations=operations,batch=batch)
        ini = configparser.ConfigParser()
        ini.read(output/'simulator.ini')
        ini['hbnpu']['instruction_capacity']='1'
        with (output/'simulator.ini').open('w') as stream: ini.write(stream)
        return output,manifest

    def check_program(self,output,program,reads,writes,resident,cycles=20000):
        result = run_one(self.binary,output/'simulator.ini',output/program['trace'],
                         output/(program['trace']+'.run'),cycles,60)
        commands,issues,complete,_,metrics,rows = result
        self.assertTrue(complete,program['id'])
        self.assertTrue(metrics['end_completed'])
        self.assertTrue(rows)
        self.assertEqual(validate_timing(commands,issues,output/'simulator.ini'),[])
        self.assertEqual(validate_rows(commands,issues,rows,output/'simulator.ini'),[])
        regions = {r['row']:r for r in json.loads((output/'manifest.json').read_text())['regions'].values()}
        for channel,trace in commands.items():
            count = Counter(c['kind'] for c in trace)
            self.assertEqual(count['lh_read']+count['lh_read_p'],reads,channel)
            self.assertEqual(count['gh_read']+count['gh_read_p'],resident,channel)
            self.assertEqual(count['pim_write']+count['pim_write_p'],writes,channel)
            stores = [c for c in trace if c['kind'] in ('pim_write','pim_write_p')]
            self.assertEqual(len({(c['row'],c['column']) for c in stores}),writes)
            # Every issued data address belongs to its declared region, not a live input.
            for command in trace:
                kind = command['kind']
                key = ('input_row' if kind.startswith('lh_read') else
                       'weight_row' if kind.startswith('gh_read') else
                       'output_row' if kind.startswith('pim_write') else None)
                if key:
                    base = program['config'][key]
                    region = regions[base]
                    self.assertTrue(base <= command['row'] < base+region['rows'])
        return result

    def test_group_and_sequence_tails(self):
        # Literal expectations: (group,L,LH physical reads,QK stores,QK loads,SV loads).
        for group,length,reads,writes,qloads,sloads in [
                (1,1,128,1,1,1),(3,127,128,1,3,3),(4,128,128,1,4,4),
                (7,129,256,2,7,14),(8,143,256,2,8,16),
                (7,2049,2176,17,14,119)]:
            with self.subTest(group=group,length=length):
                output,manifest = self.generated(group,length)
                self.check_program(output,manifest['programs'][0],reads,writes,qloads)
                self.check_program(output,manifest['programs'][1],reads,1,sloads)

    def test_split_group_restreams_shared_cache(self):
        output,manifest = self.generated(9,143)
        programs = manifest['programs']
        self.assertEqual([p['query_heads'] for p in programs[:2]],[list(range(8)),[8]])
        self.assertEqual(programs[0]['config']['input_row'],programs[1]['config']['input_row'])
        for p,loads,writes in zip(programs,[8,1,16,2],[2,2,1,1]):
            self.check_program(output,p,256,writes,loads)

    def test_prefill_position_and_refresh_crossing(self):
        output,manifest = self.generated(7,3,phase='prefill')
        selected = [p for p in manifest['programs'] if '/token2/' in p['id']]
        self.assertEqual(len(selected),2)
        for p in selected: self.check_program(output,p,128,1,7)
        output,manifest = self.generated(7,4097,operations=('sv',))
        result = self.check_program(output,manifest['programs'][0],4224,1,231,100000)
        self.assertTrue(any('refresh' in c['kind'] for c in result[0]['0']))

    def test_publisher_models_attention_and_projection(self):
        for name,groups,group,qreads,qstores,qloads in [
                ('qwen2.5-7b',4,7,100352,28,56),
                ('mistral-7b-v0.3',8,4,131072,32,64)]:
            with self.subTest(model=name):
                self.serial += 1
                output,manifest = self.from_config(ROOT/'tests/fixtures/models'/ (name+'.json'),143,
                                                   operations=('qk','sv','q_proj'))
                for op,reads,stores,loads in [('qk',256*groups,2,group*groups),('sv',256*groups,1,2*group*groups),
                                             ('q_proj',qreads,qstores,qloads)]:
                    p = next(p for p in manifest['programs'] if p['operation']==op)
                    self.check_program(output,p,reads,stores,loads,200000)

    def test_canonical_all_bank_row_coverage(self):
        # d=128,L=2048: bank stream vectors = groups * 128, for both QK/SV.
        # With four groups SV consumes 32 columns per K tile; with sixteen,
        # both operations consume full 64-column rows.
        for name,batch,reads,qloads,sloads,sv_columns in [
                ('qwen2.5-7b',1,8192,112,448,32),
                ('qwen2.5-7b',4,32768,1792,1792,64),
                ('mistral-7b-v0.3',1,16384,256,512,64)]:
            with self.subTest(model=name,batch=batch):
                self.serial += 1
                output,manifest = self.from_config(ROOT/'tests/fixtures/models'/(name+'.json'),2048,
                                                   batch=batch,hardware='HBNPU_8Gb.ini')
                for p in manifest['programs']:
                    qk = p['operation']=='qk'
                    result = self.check_program(output,p,reads,16 if qk else 1,
                                                qloads if qk else sloads,100000)
                    for trace in result[0].values():
                        for bg in range(4):
                            for bank in range(4):
                                stream = [c for c in trace if c['bankgroup']==bg and c['bank']==bank
                                          and c['kind'] in ('lh_read','lh_read_p')]
                                self.assertEqual(len(stream),reads//16)
                                self.assertEqual([c['column'] for c in stream],list(range(64))*(reads//1024))
                                self.assertEqual([c['row'] for c in stream],
                                                 [p['config']['input_row']+i//64 for i in range(reads//16)])
                                closes = [c['column'] for c in stream if c['kind']=='lh_read_p']
                                # Row-lockstep may close at a tile boundary as well as row end.
                                self.assertTrue(all(c==63 or (not qk and sv_columns==32 and c==31)
                                                    for c in closes))

    def test_canonical_packed_query_chunks_and_group_remainder(self):
        path = self.base/'six-groups.json'
        path.write_text(json.dumps(dict(model_type='qwen2',hidden_size=6912,
            intermediate_size=256,num_attention_heads=54,num_key_value_heads=6,
            num_hidden_layers=1,max_position_embeddings=4096)))
        output,manifest = self.from_config(path,143,operations=('qk',),hardware='HBNPU_8Gb.ini')
        self.assertEqual([len(p['members']) for p in manifest['programs']],[4,4,2,2])
        for p,reads,loads in zip(manifest['programs'],[1024,1024,512,512],[32,4,16,2]):
            self.check_program(output,p,reads,2,loads)


if __name__ == '__main__':
    GqaExecutionTest.binary = Path(sys.argv.pop(1)).resolve()
    unittest.main()
