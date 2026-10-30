"""v2_h9_split_attn — H9 from docs/human_audit.md (human idea, written 30 Oct, before Glimmer saw the task).

Measured 30 Oct (bench/verify_cost.py --context): a 3-token forward pass costs 1.25x a 1-token pass at
3.5K context and 1.44x at 13.8K, while a 2-token pass stays at 1.07-1.09x. Cause (MLX 0.32.4 source,
scaled_dot_product_attention.cpp, has_fused_kernel): the fused "vector" attention kernel for short
queries requires query_length x GQA_factor <= 32. Glimmer has 32 query heads per 2 key/value heads
(GQA factor 16), so a 3-token pass (48) falls back to unfused attention, whose cost grows with context.

This variant splits the queries of such passes into chunks of at most 32 // GQA = 2 rows. Each chunk
attends to exactly the keys its queries may see (keys up to its last query; the causal mask is
aligned to the end of the keys, as in MLX), so every chunk uses the fused kernel. The result is the
same attention, computed in pieces.

Only passes of 3-8 tokens change (H8's check passes). One-token decoding and prompt processing
(chunks of > 8 tokens) run the original code, so the golden check must stay bit-exact.
"""
import mlx.core as mx
from mlx_lm.models import muse_glimmer as mg

NAME = "v2_h9_split_attn"
EXACT = True          # for the golden check (1-token decode + long prompt chunks): unchanged code path
DESCRIPTION = "Split 3-8 token attention into <=2-query chunks so MLX's fused vector kernel is used."

_orig = mg.scaled_dot_product_attention
CALLS = {"split": 0}


def split_sdpa(queries, keys, values, cache, scale, mask, sinks=None):
    L = queries.shape[2]
    gqa = queries.shape[1] // keys.shape[1]
    if (L <= 1 or L > 8 or L * gqa <= 32 or mask is None or sinks is not None
            or hasattr(cache, "bits")):
        return _orig(queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks)
    step = max(1, 32 // gqa)
    K = keys.shape[2]
    outs = []
    for r0 in range(0, L, step):
        r1 = min(r0 + step, L)
        kl = K - L + r1                                   # keys visible to this chunk's last query
        m = mask if isinstance(mask, str) else mask[..., r0:r1, :kl]
        outs.append(mx.fast.scaled_dot_product_attention(
            queries[:, :, r0:r1], keys[:, :, :kl], values[:, :, :kl], scale=scale, mask=m))
    CALLS["split"] += 1
    return mx.concatenate(outs, axis=2)


def apply():
    # Attention.__call__ looks the function up in the muse_glimmer module at call time.
    mg.scaled_dot_product_attention = split_sdpa


def post_load(model):
    pass
