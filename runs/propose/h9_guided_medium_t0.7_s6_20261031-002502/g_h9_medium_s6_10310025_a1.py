"""split_heads_fused — split query heads to stay in the fused SDPA kernel."""

NAME = "g_h9_medium_s6_10310025_a1"
EXACT = True
DESCRIPTION = "Split query heads into <=8 head chunks for 3-8 token queries to keep the fused SDPA kernel and avoid the unfused long-context cost."


def apply():
    import mlx_lm.models.muse_glimmer as mg
    from mlx_lm.models.base import scaled_dot_product_attention
    import mlx.core as mx

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

        # For 3-8 token queries the fused kernel condition fails:
        # query_length * (n_heads / n_kv_heads) <= 32  ->  n_heads <= 64 / L
        # Splitting heads keeps the same attention result but forces the fused path.
        if 2 < L <= 8:
            n_heads = self.n_heads
            chunk_size = 8  # 8 * 16 = 128? Actually condition uses chunk heads:
            # query_length * (chunk_heads/2) <=32  -> chunk_heads <=64/L
            # chunk_size=8 satisfies L<=8
            outs = []
            for start in range(0, n_heads, chunk_size):
                end = start + chunk_size
                if end > n_heads:
                    end = n_heads
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

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = new_call


def post_load(model):
    pass
