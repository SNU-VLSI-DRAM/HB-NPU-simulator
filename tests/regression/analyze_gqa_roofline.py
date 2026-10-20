"""Run canonical GQA attention traces and report ideal versus issued-traffic bounds.

Independent kernels, not end-to-end inference. No fitted correction factor.
Retains raw traces; incomplete runs fail and never receive a latency ratio.
"""
import argparse
from collections import Counter
import configparser
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from gen_gqa_workload import generate
from hbnpu_gqa import RowAllocator, lower_lh
from hbnpu_instruction import compile_kernel, write_trace
from compare_instruction_execution import run_one, validate_rows, validate_timing


def analyze(binary, destination):
    destination.mkdir(parents=True,exist_ok=False)
    hardware = ROOT/'configs/HBNPU_8Gb.ini'
    ini = configparser.ConfigParser()
    ini.read(hardware)
    d,s,t = ini['dram_structure'],ini['system'],ini['timing']
    request_bytes = int(s['bus_width'])//8*int(d['BL'])
    banks = int(d['bankgroups'])*int(d['banks_per_group'])
    channels = int(s['channels'])
    period = int(t['tCCD_L'])
    bandwidth = request_bytes*banks*channels/period
    compute_rate = 2*128*128/period
    reports = []
    # Aligned contexts, a padding tail, and a small prefill causal-prefix case.
    for name,batch,length,phase in [
            ('qwen2.5-7b',1,2048,'decode'),('qwen2.5-7b',4,2048,'decode'),
            ('mistral-7b-v0.3',1,2048,'decode'),('mistral-7b-v0.3',2,2048,'decode'),
            ('qwen2.5-7b',1,143,'decode'),('qwen2.5-7b',2,3,'prefill')]:
        case = '{}-b{}-L{}-{}'.format(name,batch,length,phase)
        folder = destination/case
        manifest = generate(ROOT/'tests/fixtures/models'/(name+'.json'),hardware,folder,
                            phase=phase,sequence_length=length,batch=batch,operations=('qk','sv'))
        records = {r['id']:r for r in manifest['workloads']}
        for p in manifest['programs']:
            result = run_one(binary,folder/'simulator.ini',folder/p['trace'],
                             folder/(p['trace']+'.run'),100000,60)
            commands,issues,complete,cycles,metrics,rows = result
            errors = validate_timing(commands,issues,folder/'simulator.ini')
            errors += validate_rows(commands,issues,rows,folder/'simulator.ini',complete)
            if errors or not complete:
                raise RuntimeError((case,p['id'],complete,errors[:3]))
            counts = Counter(c['kind'] for trace in commands.values() for c in trace)
            reads = counts['lh_read']+counts['lh_read_p']
            loads = counts['gh_read']+counts['gh_read_p']
            stores = counts['pim_write']+counts['pim_write_p']
            g,k,n = p['valid_shape']
            groups = p['packed_groups']
            _,kp,np = p['padded_shape']
            useful_stream = groups*k*n*2
            padded_stream = groups*kp*np*2
            if reads*request_bytes != padded_stream:
                raise AssertionError(('stream bytes',case,reads*request_bytes,padded_stream))
            # Check all banks, not just a sampled channel. Columns must progress
            # consecutively in the declared stream region, including padding.
            for trace in commands.values():
                for bg in range(4):
                    for bank in range(4):
                        stream = [c for c in trace if c['bankgroup']==bg and c['bank']==bank
                                  and c['kind'] in ('lh_read','lh_read_p')]
                        if len(stream)!=reads//128:
                            raise AssertionError('unequal bank coverage')
                        if [c['column'] for c in stream] != [i%64 for i in range(len(stream))]:
                            raise AssertionError('column coverage')
                        if [c['row'] for c in stream] != [p['config']['input_row']+i//64 for i in range(len(stream))]:
                            raise AssertionError('row coverage')
            minimum_bytes = useful_stream+groups*g*(k+n)*2
            roof = max(minimum_bytes/bandwidth,2*groups*g*k*n/compute_rate)
            stream_bound = reads*request_bytes/bandwidth
            row_wait = metrics['row_boundary_wait_cycles']
            # Same-binary old single-group lowering: isolate the effect of
            # packing from previous simulator/geometry changes. No old goldens.
            split_cycles = 0
            split_reads = 0
            for member in p['members']:
                record = records[member['workload_id']]
                alloc = RowAllocator(hardware)
                old = lower_lh(record,alloc)[0]  # fixtures have <=8 queries/group
                out = folder/('split-'+str(len(reports))+'-'+str(member['batch'])+'-'+str(member['kv_head']))
                out.mkdir()
                trace_path = out/'instructions.trace'
                write_trace(compile_kernel(old['config']),trace_path)
                a,ai,done,elapsed,_,ar = run_one(binary,folder/'simulator.ini',trace_path,out/'run',100000,60)
                if (not done or validate_timing(a,ai,folder/'simulator.ini')
                        or validate_rows(a,ai,ar,folder/'simulator.ini',done)):
                    raise AssertionError('split baseline invalid')
                split_cycles += elapsed
                split_reads += sum(c['kind'] in ('lh_read','lh_read_p') for v in a.values() for c in v)
            if split_reads!=reads:
                raise AssertionError('packing changed total padded stream traffic')
            lengths = Counter(r['vectors'] for r in rows)
            report = dict(case=case,operation=p['operation'],groups=groups,
                          physical_M=p['config']['m'],physical_N=p['config']['n'],
                          physical_K=p['config']['k'],cycles=cycles,split_cycles=split_cycles,
                          split_over_packed=split_cycles/cycles,
                          useful_stream_bytes=useful_stream,padded_stream_bytes=padded_stream,
                          lh_commands=reads,gh_load_commands=loads,store_commands=stores,
                          roofline_cycles=roof,cycles_over_roofline=cycles/roof,
                          stream_bound_cycles=stream_bound,row_wait_cycles=row_wait,
                          residual_cycles=cycles-stream_bound-row_wait,
                          row_segment_vectors=dict(lengths),complete=complete)
            reports.append(report)
            print(json.dumps(report),flush=True)
            (destination/'report.json').write_text(json.dumps(reports,indent=2)+'\n')
    with (destination/'report.csv').open('w') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    analyze(args.binary.resolve(),args.output.resolve())
