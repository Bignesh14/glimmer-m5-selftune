"""v1_seq_split_correct — split query sequence into 2-token chunks for 3-token decoding passes."""

NAME = "g_h9_medium_10302229_r4"
EXACT = True
DESCRIPTION = "For 3-token decoding passes split the query sequence into 2+1 token chunks to force the fused SDPA kernel."

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

        # Only modify 3-token decoding passes; keep 1-token and prompt processing bit-identical
        if cache is not None and L == 3:
            # Split into 2 + 1 to force query_length * (n_heads/n_kv) <= 32
            # First chunk: tokens 0-1, second chunk: token 2
            q0 = q[:, :, 0:2, :]
            q1 = q[:, :, 2:3, :]
            out0 = scaled_dot_product_attention(q0, k, v, cache=cache, scale=self.scale, mask=mask)
            out1 = scaled_dot_product_attention(q1, k, v, cache=cache, scale=self.scale, mask=mask)
            out = mx.concatenate([out0, out1], axis=2)
        else:
            out = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.scale, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    mg.Attention.__call__ = _patched_call


def post_load(model):
    pass
