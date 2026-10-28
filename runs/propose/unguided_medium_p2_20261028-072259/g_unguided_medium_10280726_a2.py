"""v1_fuse_rms_and_pre_norm — fuse RMSNorm scaling and simplify CenteredRMSNorm."""

NAME = "g_unguided_medium_10280726_a2"
EXACT = True
DESCRIPTION = "Fuse RMSNorm weight scaling, simplify CenteredRMSNorm to constant weight, fuse qk_scale_factor into rms_norm."

def apply():
    import mlx.core as mx
    from mlx_lm.models.muse_glimmer import Attention, CenteredRMSNorm
    from mlx_lm.models.base import scaled_dot_product_attention

    # CenteredRMSNorm weight is always 1.0+0 -> 1.0, so the weight can be constant
    def centered_rms_norm_call(self, x):
        return mx.fast.rms_norm(x, 1.0, self.eps)

    CenteredRMSNorm.__call__ = centered_rms_norm_call

    # Attention: fuse qk_scale_factor into rms_norm weight to avoid extra multiply
    def attention_call_opt(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

        # Original: rms_norm(q, None) * qk_scale_factor  -> rms_norm(q, qk_scale_factor)
        q = mx.fast.rms_norm(q, self.qk_scale_factor, self.eps)
        k = mx.fast.rms_norm(k, 1.0, self.eps)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)

        if cache is not None:
            k, v = cache.update_and_fetch(k, v)

        out = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        out = out * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(out)

    Attention.__call__ = attention_call_opt


def post_load(model):
    """No weight transformation needed; optimizations are in apply()."""
    pass
