"""Generate independent LH instruction programs from a local dense model config."""
import argparse
import configparser
from dataclasses import asdict
import json
from pathlib import Path

from hbnpu_gqa import RowAllocator, build_workloads, lower_workloads
from hbnpu_instruction import compile_kernel, write_trace
from hbnpu_model import ModelConfig

OPERATIONS = ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','qk','sv')


def generate(model_config, hardware, output, *, phase, sequence_length,
             batch=1, tp=1, pp=1, layer=0, operations=OPERATIONS):
    """Validate the complete selection before creating any output artifacts."""
    output = Path(output)
    if output.exists():
        raise ValueError('output destination already exists')
    if not operations or any(op not in OPERATIONS for op in operations):
        raise ValueError('unsupported or empty operation selection')
    model = ModelConfig.from_json(model_config,tp=tp,pp=pp)
    allocator = RowAllocator(hardware)
    records,programs = [],[]
    for record in build_workloads(model,phase,sequence_length,batch,layer=layer):
        if record['operation'] not in operations:
            continue
        records.append(record)
    for program in lower_workloads(records,allocator):
        # Exhaust compilation before writes to catch invalid geometry/packet bounds.
        packets = list(compile_kernel(program['config']))
        filename = '{:06d}_{}.trace'.format(len(programs),program['operation'])
        programs.append((dict(program,trace=filename),packets))
    ini = configparser.ConfigParser()
    ini.read(hardware)
    if not ini.has_section('hbnpu'): ini.add_section('hbnpu')
    ini['hbnpu'].update(row_lockstep='true',gh_read_delay='true')
    ini['system']['enable_self_refresh'] = 'false'
    manifest = dict(format_version=2,model=asdict(model),phase=phase,
                    sequence_length=sequence_length,batch=batch,modeled_layers=[layer],
                    shard=dict(tp_rank=0,pp_stage=0),
                    scope='Independent matrix-kernel timing programs; not end-to-end inference.',
                    output_layout='Existing padded LH full-array footprint; inactive query lanes retained.',
                    attention_mapping='Head/batch first, power-of-two packs up to 16 groups; remaining banks shard output columns. members is authoritative for tensor identities.',
                    padding='Zero-fill padded K/N elements; inactive rows are ignored. Values are not simulated.',
                    cache_layout='Shared within each padded view; distinct prefix views may require separate packing.',
                    omitted_operations=['softmax','RoPE','normalization','SiLU','gating multiply',
                                        'bias','residual addition','collectives','embedding','lm_head',
                                        'KV update','layout conversion'],
                    kv_cache_bytes_full_model=model.kv_cache_bytes(sequence_length,batch),
                    kv_cache_bytes_local_shard=model.kv_cache_bytes(sequence_length,batch)//tp//pp,
                    workloads=records,regions=allocator.regions,
                    programs=[dict(p,config=asdict(p['config'])) for p,_ in programs])
    # No destination exists until model, bounds, geometry and all packets validate.
    output.mkdir(parents=True,exist_ok=False)
    with (output/'simulator.ini').open('x') as stream: ini.write(stream)
    for program,packets in programs:
        write_trace(packets,output/program['trace'])
    with (output/'manifest.json').open('x') as stream:
        json.dump(manifest,stream,indent=2)
        stream.write('\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-config',required=True,type=Path)
    parser.add_argument('--hardware',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('decode','prefill'))
    parser.add_argument('--sequence-length',required=True,type=int,
                        help='Decode: KV length including current token; prefill: number of prompt tokens')
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--tp',type=int,default=1)
    parser.add_argument('--pp',type=int,default=1)
    parser.add_argument('--layer',type=int,default=0,help='One layer in PP stage 0 (default: 0)')
    parser.add_argument('--operations',nargs='+',choices=OPERATIONS,default=OPERATIONS)
    args = parser.parse_args()
    try:
        manifest = generate(**vars(args))
    except (ValueError,OSError,KeyError,configparser.Error) as exc:
        parser.error(str(exc))
    print('Generated {} independent programs in {}'.format(len(manifest['programs']),args.output))


if __name__ == '__main__': main()
