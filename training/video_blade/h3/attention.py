"""H3's native processor with the official Blackwell FA4 attention operation."""

import importlib.metadata
import sys
from functools import lru_cache
from pathlib import Path

from diffusers.models.transformers.transformer_minimax_h3 import (
    MiniMaxH3Attention,
    _apply_rotary_emb,
)


@lru_cache(maxsize=1)
def flash4_function():
    import flash_attn

    # FA4 is a namespace addition. The image's regular FA2 package otherwise
    # hides a newer cute/ installed in a system-site-packages virtualenv.
    package = Path(
        importlib.metadata.distribution("flash-attn-4").locate_file("flash_attn")
    )
    if str(package) not in flash_attn.__path__:
        if "flash_attn.cute" in sys.modules:
            raise RuntimeError(
                "Load H3's FA4 helper before importing another flash_attn.cute version"
            )
        flash_attn.__path__.insert(0, str(package))
    from flash_attn.cute import flash_attn_func

    return flash_attn_func


class H3Flash4Processor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        flash_attn_func = flash4_function()

        if attention_mask is not None:
            raise ValueError("H3 FA4 requires the native unpadded sequence")
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        output = flash_attn_func(query, key, value, causal=False)
        if isinstance(output, tuple):
            output = output[0]
        output = output.flatten(2, 3).type_as(query)
        return attn.to_out[1](attn.to_out[0](output))


def use_flash4(model):
    for module in model.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.set_processor(H3Flash4Processor())
