"""v1_head_split — split query heads to stay in fused SDPA kernel for 3-8 token passes."""

NAME = "g_h9_medium_s5_10310019_a1"
EXACT = True
DESCRIPTION = "Split query heads for 3-8 token decoding passes to keep query_length * heads_ratio <=32 and use fused SDPA kernel."

def apply():
    from mlx_lm.models import muse_glimmer
    from mlx_lm.models.base import scaled_dot_product_attention
    import mlx.core as mx

    orig_call = muse_glimmer.Attention.__call__

    def new_call(self, x: mx.array, mask=None, cache=None):
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

        # For 3-8 token decoding passes keep the fused vector kernel.
        # Fused condition: query_length * (n_heads / n_kv_heads) <= 32
        # -> max heads per call = floor(32 * n_kv_heads / L)
        if cache is not None and 2 < L <= 8:
            max_heads = 32 * self.n_kv_heads // L
            if max_heads > 0 and max_heads < self.n_heads:
                parts = []
                for start in range(0, self.n_heads, max_heads):
                    end = min(start + max_heads, self.n_heads)
                    q_chunk = q[:, start:end, :, :]
                    out_chunk = scaled_dot_product_attention(
                        q_chunk, k, v, cache=cache, scale=self.scale, mask=mask
                    )
                    parts.append(out_chunk)
                out = mx.concatenate(parts, axis=1)
            else:
                out = scaled_dot_product_attention(
                    q, k, v, cache=cache, scale=self.scale, mask=mask
                )
        else:
            out = scaled_dot_product_attention(
                q, k, v, cache=cache, scale=self.scale, mask=mask
            )

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    muse_glimmer.Attention.__call__ = new_call


def post_load(model):
    pass
