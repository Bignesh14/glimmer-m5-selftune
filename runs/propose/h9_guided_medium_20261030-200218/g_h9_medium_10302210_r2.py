"""v1_head_split_8 — split query heads into 8-head chunks for short decoding passes."""

NAME = "g_h9_medium_10302210_r2"
EXACT = True
DESCRIPTION = "Split query heads into 8-head chunks for 2-8 token decoding passes to satisfy fused SDPA condition."

def apply():
    import mlx_lm.models.muse_glimmer as mg
    from mlx_lm.models.base import scaled_dot_product_attention
    import mlx.core as mx

    _orig_call = mg.Attention.__call__

    def _patched_call(self, x: mx.array, mask=None, cache=None) -> mx.array:
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

        # Only modify short decoding passes; keep 1-token and prompt processing bit-identical
        if cache is not None and 2 <= L <= 8:
            n_heads = self.n_heads
            n_kv = self.n_kv_heads
            # 8-head chunks give ratio 8/2=4, so L*4 <=32 for L<=8 -> fused kernel
            chunk_heads = 8
            # ensure multiple of n_kv and divides n_heads
            if chunk_heads % n_kv != 0:
                chunk_heads = (chunk_heads // n_kv) * n_kv
            if chunk_heads > n_heads:
                chunk_heads = n_heads

            # use split only when it helps reach fused condition
            if L * (n_heads // n_kv) > 32 and L * (chunk_heads // n_kv) <= 32:
                outs = []
                for start in range(0, n_heads, chunk_heads):
                    end = start + chunk_heads
                    if end > n_heads:
                        end = n_heads
                    q_chunk = q[:, start:end, :, :]
                    out_chunk = scaled_dot_product_attention(q_chunk, k, v, cache=cache, scale=self.scale, mask=mask)
                    outs.append(out_chunk)
                out = mx.concatenate(outs, axis=1)
            else:
                out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
        else:
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = _patched_call


def post_load(model):
    pass
