"""Logical grouped attention and lowering into the existing padded LH profile.

Logical shape is (query/token rows, reduction dimension, output columns).
The legacy workload's (M,K,N) ordering is (output columns, reduction, rows).
No tensor values, softmax, or attention-specific hardware instructions are used.
"""
import configparser
from dataclasses import replace

from hbnpu_instruction import KernelConfig, _geometry
from hbnpu_model import positive


def build_workloads(model, phase, sequence_length, batch=1, *, layer=None):
    """Yield matrix records for TP rank 0 / PP stage 0; layer=None yields all local layers.

    Decode length includes the current token. Prefill starts at token zero.
    Each query position uses its own causal prefix; no incompatible masks are fused.
    """
    if phase not in ('decode','prefill'):
        raise ValueError('phase must be decode or prefill')
    positive(sequence_length,'sequence_length')
    positive(batch,'batch')
    if sequence_length > model.context_limit:
        raise ValueError('sequence length exceeds model context limit')
    if layer is not None and (type(layer) is not int or not 0 <= layer < model.local_layers):
        raise ValueError('layer outside local PP stage')
    group = model.query_heads // model.kv_heads
    positions = range(sequence_length) if phase == 'prefill' else [sequence_length-1]
    for li in range(model.local_layers) if layer is None else [layer]:
        prefix = 'layer{}'.format(li)
        h = model.hidden_size
        intermediate = model.intermediate_size // model.tp
        qwidth = model.local_query_heads * model.head_dim
        kvwidth = model.local_kv_heads * model.head_dim
        shapes = [('q_proj',h,qwidth),('k_proj',h,kvwidth),('v_proj',h,kvwidth),
                  ('o_proj',qwidth,h),('gate_proj',h,intermediate),
                  ('up_proj',h,intermediate),('down_proj',intermediate,h)]
        for bi in range(batch):
            identity = '{}/batch{}'.format(prefix,bi)
            for op,k,n in shapes:
                rid = identity+'/'+op
                yield dict(id=rid,operation=op,layer=li,batch=bi,token=None,
                           shape=(sequence_length if phase=='prefill' else 1,k,n),
                           stream_tensor=prefix+'/weights/'+op,
                           stream_orientation='weight_KN',
                           resident_tensor=rid+'/activation',output_tensor=rid+'/output',
                           query_heads=[],kv_head=None,valid_kv_range=None)
            for token in positions:
                length = token+1
                for kv in range(model.local_kv_heads):
                    heads = list(range(kv*group,(kv+1)*group))
                    group_id = '{}/token{}/kv{}'.format(identity,token,kv)
                    for op,k,n,cache in [('qk',model.head_dim,length,'K'),
                                         ('sv',length,model.head_dim,'V')]:
                        rid = group_id+'/'+op
                        yield dict(id=rid,operation=op,layer=li,batch=bi,token=token,
                                   shape=(group,k,n),query_heads=heads,kv_head=kv,
                                   valid_kv_range=(0,length),
                                   stream_tensor='{}/cache/{}/kv{}'.format(identity,cache,kv),
                                   stream_orientation='K_transposed' if op=='qk' else 'V',
                                   resident_from=(dict(tensor=group_id+'/qk/output',operation='softmax',executed=False)
                                                  if op=='sv' else None),
                                   resident_tensor=group_id+('/query' if op=='qk' else '/normalized_scores'),
                                   output_tensor=rid+'/output')


def _ceil(value,unit): return (value+unit-1)//unit


class RowAllocator:
    """Conservative disjoint row regions across all banks/channels, shared by view ID.

    Distinct padded cache views may occupy distinct regions. Query chunks of the
    same view reuse one region; no per-query KV copies are allocated.
    """
    def __init__(self, hardware):
        ini = configparser.ConfigParser()
        if not ini.read(hardware):
            raise ValueError('cannot read hardware configuration')
        d,s = ini['dram_structure'],ini['system']
        if (d['protocol'] not in ('HBM','HBM2') or int(d['bankgroups'])!=4
                or int(d['banks_per_group'])!=4 or int(s['channels'])!=8
                or int(d['BL'])!=4 or int(s['bus_width']) not in (64,128)
                or int(d['device_width'])!=int(s['bus_width']) or not d.getboolean('bankgroup_enable',True)):
            raise ValueError('LH grouped profile requires 8 channels, 4x4 banks, BL=4, matching 64/128-bit HBM devices/bus')
        self.columns = int(d['columns'])*2//4
        self.rows = positive(int(d['rows']),'hardware rows')
        power = lambda value: value > 0 and value & (value-1) == 0
        if not power(int(d['columns'])) or not power(self.columns) or self.columns>65535:
            raise ValueError('unsupported hardware column geometry')
        if not power(self.rows) or self.rows >= 1 << 31:
            raise ValueError('hardware row capacity must be a representable power of two')
        # Mirror Config::CalculateSize's integer arithmetic before any division.
        # Tiny rows/pages can otherwise yield a zero-sized rank and crash startup.
        page_bytes = int(d['columns'])*2*int(d['device_width'])//8
        size_product = page_bytes*(self.rows//1024)
        rank_mb = size_product//1024*16
        channel_mb = positive(int(s['channel_size']),'channel_size')
        if size_product >= 1 << 31 or not 0 < rank_mb < 1 << 31:
            raise ValueError('hardware row capacity produces zero or overflowing rank capacity')
        if (channel_mb >= 1 << 31 or channel_mb % rank_mb
                or not power(channel_mb//rank_mb)):
            raise ValueError('channel_size must represent an exact power-of-two number of ranks')
        self.next_row = 0
        self.regions = {}

    def allocate(self, identity, vectors):
        positive(vectors,'region vectors')
        rows = _ceil(vectors,self.columns)
        if identity in self.regions:
            existing = self.regions[identity]
            if existing['vectors'] != vectors:
                raise ValueError('inconsistent tensor view extent')
            return existing['row']
        if rows > self.rows-self.next_row:
            raise ValueError('tensor regions exceed hardware row capacity')
        row = self.next_row
        self.regions[identity] = dict(row=row,rows=rows,vectors=vectors)
        self.next_row += rows
        return row


def lower_lh(record, allocator):
    """Single-record fallback, also used for projections."""
    return _lower_pack([record], allocator)


def lower_workloads(records, allocator):
    """Pack compatible attention groups first, then shard columns within each group.

    Factor fields must be powers of two. A remainder of six groups therefore
    becomes packs of four and two, not eight with duplicated/padded groups.
    Projections keep their existing independent lowering.
    """
    buckets = {}
    for record in records:
        if record['operation'] not in ('qk','sv'):
            key = ('independent',record['id'])
        else:
            key = (record['operation'],record['layer'],record['token'],
                   tuple(record['shape']),tuple(record['valid_kv_range']),
                   record['stream_orientation'])
        buckets.setdefault(key,[]).append(record)
    for records in buckets.values():
        offset = 0
        while offset < len(records):
            count = 1 << (min(16,len(records)-offset).bit_length()-1)
            yield from _lower_pack(records[offset:offset+count],allocator)
            offset += count


def _lower_pack(records, allocator):
    """Queries are resident; each independent group owns a disjoint bank stripe.

    The tensor layout is declarative: the simulator models addresses/timing,
    not values or the cost of repacking cache views. Up to eight query rows per
    group use the existing padded output protocol; larger groups re-stream KV.
    """
    record = records[0]
    groups = len(records)
    shards = 16//groups
    logical_m,k,n = record['shape']
    for name,value in [('M',logical_m),('K',k),('N',n)]: positive(value,name)
    kp,np = _ceil(k,128)*128,_ceil(n,128)*128
    stream_view = '{}:lh:G{}:K{}:N{}'.format('|'.join(r['stream_tensor'] for r in records),groups,kp,np)
    programs = []
    for start in range(0,logical_m,8):
        count = min(8,logical_m-start)
        config = KernelConfig(np//shards,count*groups,kp,128,0,0,0,1,shards,groups,8,16,allocator.columns)
        _,_,_,_,spans = _geometry(config)
        chunk_id = '{}:rows{}-{}'.format('|'.join(r['id'] for r in records),start,start+count)
        # _geometry returns conservative input(stream), weight(resident), output spans.
        config = replace(config,
                         input_row=allocator.allocate(stream_view,spans[0]),
                         weight_row=allocator.allocate(chunk_id+':resident',spans[1]),
                         output_row=allocator.allocate(chunk_id+':output',spans[2]))
        members = [dict(workload_id=r['id'],batch=r['batch'],token=r['token'],
                        kv_head=r['kv_head'],query_heads=r['query_heads'][start:start+count],
                        banks=list(range(i*shards,(i+1)*shards)),
                        # Loaded vectors are compact; each group's physical PE
                        # lanes remain eight wide even for seven active queries.
                        resident_vector_range=(i*count,(i+1)*count),
                        stream_tensor=r['stream_tensor'],resident_tensor=r['resident_tensor'],
                        output_tensor=r['output_tensor'],
                        column_shards=[dict(bank=i*shards+s,
                            padded_column_range=(s*(np//shards),(s+1)*(np//shards)))
                            for s in range(shards)]) for i,r in enumerate(records)]
        programs.append(dict(id=chunk_id,config=config,operation=record['operation'],
                             members=members,packed_groups=groups,banks_per_group=shards,
                             valid_shape=(count,k,n),padded_shape=(8,kp,np),
                             logical_row_range=(start,start+count),
                             query_heads=record['query_heads'][start:start+count] if groups==1 else None,
                             stream_view=stream_view,stream_tensor=record['stream_tensor'] if groups==1 else None,
                             resident_tensor=record['resident_tensor'] if groups==1 else None,
                             output_tensor=record['output_tensor'] if groups==1 else None,
                             output_vectors_per_channel=np//128))
    return programs
