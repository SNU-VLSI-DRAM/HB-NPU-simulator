"""Controlled fallback experiment: column splitting versus padded, unsplit banks.

LH has no inactive-bank mask. The no-split baseline fills 16 group slots,
zero-padding unused groups, including their resident-query slots. It is NOT a
simulation of hypothetical masked-bank hardware. Both use the existing ISA.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gen_gqa_workload import generate
from hbnpu_gqa import RowAllocator
from hbnpu_instruction import KernelConfig, _geometry, compile_kernel, write_trace
from compare_instruction_execution import run_one, validate_rows, validate_timing


def analyze(binary,output):
    output.mkdir(parents=True,exist_ok=False)
    hardware = ROOT/'configs/HBNPU_8Gb.ini'
    reports = []
    for name,tp,batch in [('qwen2.5-7b',4,1),('qwen2.5-7b',1,1),
                          ('mistral-7b-v0.3',1,1),('qwen2.5-7b',1,4)]:
        case = '{}-tp{}-b{}'.format(name,tp,batch)
        folder = output/case
        manifest = generate(ROOT/'tests/fixtures/models'/(name+'.json'),hardware,folder,
                            phase='decode',sequence_length=2048,tp=tp,batch=batch,operations=('qk','sv'))
        for p in manifest['programs']:
            split = KernelConfig(**p['config'])
            shards = p['banks_per_group']
            # One bank per group, sixteen physical slots. Slots >= real group
            # count contain zeros; their outputs are ignored. Keep q, K, O and
            # real tensor identities unchanged. No bank masking exists today.
            plain = replace(split,m=split.m*shards,n=split.n*shards,mcf=1,ucf=16)
            spans = _geometry(plain)[-1]
            allocator = RowAllocator(hardware)
            plain = replace(plain,input_row=allocator.allocate('stream',spans[0]),
                            weight_row=allocator.allocate('resident',spans[1]),
                            output_row=allocator.allocate('output',spans[2]))
            baseline = folder/('unsplit-'+p['operation']+'.trace')
            write_trace(compile_kernel(plain),baseline)
            members = [dict(member,banks=[i],column_shards=[dict(bank=i,
                       padded_column_range=(0,p['padded_shape'][2]))])
                       for i,member in enumerate(p['members'])]
            (folder/('unsplit-'+p['operation']+'.json')).write_text(json.dumps(dict(
                config=asdict(plain),real_groups=p['packed_groups'],padded_group_slots=16,
                unused_slots_are_zero=True,logical_members=members),indent=2)+'\n')
            row = dict(case=case,operation=p['operation'],real_groups=p['packed_groups'],
                       banks_per_group=shards,sequence_length=2048,head_dim=128)
            for mode,trace,c in [('split',folder/p['trace'],split),('unsplit',baseline,plain)]:
                commands,issues,done,cycles,metrics,rows = run_one(binary,folder/'simulator.ini',trace,
                    folder/(mode+'-'+p['operation']+'.run'),100000,60)
                errors = validate_timing(commands,issues,folder/'simulator.ini')
                errors += validate_rows(commands,issues,rows,folder/'simulator.ini',done)
                if not done or errors: raise AssertionError((case,mode,done,errors[:3]))
                counts = Counter(x['kind'] for v in commands.values() for x in v)
                reads = counts['lh_read']+counts['lh_read_p']
                expected_bytes = (p['packed_groups'] if mode=='split' else 16)*128*2048*2
                if reads*32!=expected_bytes: raise AssertionError('LH byte count')
                for v in commands.values():
                    for bg in range(4):
                        for bank in range(4):
                            stream = [x for x in v if x['bankgroup']==bg and x['bank']==bank
                                      and x['kind'] in ('lh_read','lh_read_p')]
                            if [(x['row'],x['column']) for x in stream] != [
                                    (c.input_row+i//64,i%64) for i in range(reads//128)]:
                                raise AssertionError('bank row/column coverage')
                row[mode] = dict(cycles=cycles,stream_bytes=reads*32,
                    resident_commands=counts['gh_read']+counts['gh_read_p'],
                    store_commands=counts['pim_write']+counts['pim_write_p'],
                    stream_bound_cycles=reads*32/2048,
                    row_wait_cycles=metrics['row_boundary_wait_cycles'],
                    row_segments=dict(Counter(r['vectors'] for r in rows)))
            row['speedup'] = row['unsplit']['cycles']/row['split']['cycles']
            row['latency_reduction_percent'] = 100*(1-1/row['speedup'])
            if shards==1 and row['split']!=row['unsplit']:
                raise AssertionError('no-fallback control changed')
            reports.append(row)
            print(json.dumps(row),flush=True)
            (output/'report.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.binary.resolve(),args.output.resolve())
