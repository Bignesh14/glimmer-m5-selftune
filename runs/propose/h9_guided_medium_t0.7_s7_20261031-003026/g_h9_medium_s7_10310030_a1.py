"""v1_head_split — split query heads to stay inside the fused SDPA kernel."""

NAME = "g_h9_medium_s7_10310030_a1"
EXACT = True
DESCRIPTION = "Split query heads for 2-8 token queries so query_length * heads_ratio <=32, keeping fused SDPA. 1-token paths unchanged."

import mlx.core as mx
from mlx_lm.models import muse_glimmer as mg
from mlx_lm.models import base as base_mod

def apply():
    orig_call = mg.Attention.__call__

    def new_call(self, x, mask=None, cache=None):
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        q = mx.fast.rms_norm(q, None, self.eps) * self.qk_scale_factor
        k = mx.fast.rms_norm(k, None, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        # Only change 2-8 token passes; head splitting keeps the attention math identical
        if 2 <= L <= 8:
            heads_max = int(32 * self.n_kv_heads / L)
            if 0 < heads_max < self.n_heads:
                outs = []
                for start in range(0, self.n_heads, heads_max):
                    end = start + heads_max
                    if end > self.n_heads:
                        end = self.n_heads
                    q_chunk = q[:, start:end, :, :]
                    out_chunk = base_mod.scaled_dot_product_attention(
                        q_chunk, k, v, cache=cache, scale=self.scale, mask=mask
                    )
                    outs.append(out_chunk)
                out = mx.concatenate(outs, axis=1)
            else:
                out = base_mod.scaled_dot_product_attention(
                    q, k, v, cache=cache, scale=self.scale, mask=mask
                )
        else:
            out = base_mod.scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = new_call


def post_load(model):
    pass
