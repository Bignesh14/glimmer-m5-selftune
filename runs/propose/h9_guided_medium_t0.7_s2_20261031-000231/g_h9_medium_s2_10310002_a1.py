"""v_split_heads — split query heads to stay in fused SDPA for 3-8 token passes."""

NAME = "g_h9_medium_s2_10310002_a1"
EXACT = True
DESCRIPTION = "Split query heads to keep query_length * heads_ratio <=32, forcing fused SDPA for 3-8 token passes."

def apply():
    from mlx_lm.models.muse_glimmer import Attention
    from mlx_lm.models.base import scaled_dot_product_attention
    import mlx.core as mx

    original_call = Attention.__call__

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

        # Use head splitting only for 2-8 token passes where fused kernel would otherwise be missed
        if 3 <= L <= 8:
            n_heads = self.n_heads
            n_kv = self.n_kv_heads
            # max heads per call to stay in fused vector kernel
            max_heads = (32 // L) * n_kv
            if max_heads < n_heads and max_heads > 0:
                chunk = max_heads
                outs = []
                for start in range(0, n_heads, chunk):
                    end = min(start + chunk, n_heads)
                    q_chunk = q[:, start:end, :, :]
                    out_chunk = scaled_dot_product_attention(
                        q_chunk, k, v, cache=cache, scale=self.scale, mask=mask
                    )
                    outs.append(out_chunk)
                out = mx.concatenate(outs, axis=1)
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

    Attention.__call__ = new_call


def post_load(model):
    pass
