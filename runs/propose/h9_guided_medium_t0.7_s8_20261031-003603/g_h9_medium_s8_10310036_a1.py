"""split_heads_fused_sdpa — split query heads for short queries to hit the fused SDPA kernel."""

NAME = "g_h9_medium_s8_10310036_a1"
EXACT = True
DESCRIPTION = "Split query heads into 8-head chunks for 3-8 token queries to satisfy the fused SDPA condition query_len * (heads/kv) <= 32."


def apply():
    from mlx_lm.models.muse_glimmer import Attention
    from mlx_lm.models.muse_glimmer import scaled_dot_product_attention
    import mlx.core as mx

    original_call = Attention.__call__

    def new_call(self, x: mx.array, mask=None, cache=None) -> mx.array:
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

        # Use head splitting for short queries to stay in the fused SDPA kernel.
        # Condition for fused kernel: L * (chunk_heads / n_kv_heads) <= 32
        # With n_kv_heads=2, chunk_heads=8 gives L*4 <=32 for L<=8.
        if 3 <= L <= 8:
            n_heads = self.n_heads
            chunk = 8
            outs = []
            for i in range(0, n_heads, chunk):
                q_chunk = q[:, i:i+chunk, :, :]
                out_chunk = scaled_dot_product_attention(
                    q_chunk, k, v, cache=cache, scale=self.scale, mask=mask
                )
                outs.append(out_chunk)
            out = mx.concatenate(outs, axis=1)
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
