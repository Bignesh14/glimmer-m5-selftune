"""v1_split_heads_fixed2 — split query heads into 16-head chunks for L=3 to hit fused SDPA."""

NAME = "g_h9_medium_10302010_a3"
EXACT = True
DESCRIPTION = "For 3-token decoding passes split query heads into 16-head chunks (multiple of n_kv) to satisfy fused SDPA condition."

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

        # Only modify the 3-token decoding pass; keep 1-token and prompt processing bit-identical
        if cache is not None and L == 3:
            n_heads = self.n_heads
            n_kv = self.n_kv_heads
            # Use 16-head chunks: 16/2=8, 3*8=24 <=32 -> fused kernel
            chunk_heads = 16
            # Ensure chunk is multiple of n_kv and divides n_heads cleanly
            if chunk_heads % n_kv != 0:
                chunk_heads = (chunk_heads // n_kv) * n_kv
            if chunk_heads > n_heads:
                chunk_heads = n_heads

            if chunk_heads >= n_heads:
                out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)
            else:
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

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = _patched_call


def post_load(model):
    pass
