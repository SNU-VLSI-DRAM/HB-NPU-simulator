"""Validated, local-only dense Qwen2/Mistral matrix-model descriptions."""
from dataclasses import dataclass
import json
from pathlib import Path


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


@dataclass(frozen=True)
class ModelConfig:
    model_type: str
    hidden_size: int
    intermediate_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    layers: int
    context_limit: int
    tp: int = 1
    pp: int = 1

    @classmethod
    def from_json(cls, path, *, tp=1, pp=1):
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError('model configuration must be an object')
        kind = raw.get('model_type')
        if kind not in ('qwen2', 'mistral'):
            raise ValueError('only dense qwen2 and mistral model types are supported')
        if any(raw.get(key) not in (None, 0) for key in ('num_local_experts','num_experts')):
            raise ValueError('MoE configurations are not supported')
        if raw.get('hidden_act', 'silu') != 'silu':
            raise ValueError('only SiLU gated MLP descriptions are supported')
        if raw.get('use_sliding_window', False) or (kind == 'mistral' and raw.get('sliding_window') is not None):
            raise ValueError('sliding-window attention is not supported')
        names = ('hidden_size','intermediate_size','num_attention_heads',
                 'num_key_value_heads','num_hidden_layers','max_position_embeddings')
        values = [positive(raw.get(name),name) for name in names]
        hidden, intermediate, q, kv, layers, context = values
        if q % kv:
            raise ValueError('query heads must be divisible by KV heads')
        dim = raw.get('head_dim')
        if dim is None:
            if hidden % q:
                raise ValueError('hidden size must divide evenly into query heads')
            dim = hidden // q
        positive(dim,'head_dim')
        positive(tp,'tp')
        positive(pp,'pp')
        if q % tp or kv % tp or intermediate % tp or layers % pp:
            raise ValueError('TP must divide Q/KV heads and intermediate size; PP must divide layers')
        return cls(kind,hidden,intermediate,q,kv,dim,layers,context,tp,pp)

    @property
    def local_query_heads(self): return self.query_heads // self.tp

    @property
    def local_kv_heads(self): return self.kv_heads // self.tp

    @property
    def local_layers(self): return self.layers // self.pp

    def kv_cache_bytes(self, tokens, batch=1, bytes_per_element=2):
        for name,value in [('tokens',tokens),('batch',batch),('bytes_per_element',bytes_per_element)]:
            positive(value,name)
        return 2 * self.layers * self.kv_heads * self.head_dim * tokens * batch * bytes_per_element
