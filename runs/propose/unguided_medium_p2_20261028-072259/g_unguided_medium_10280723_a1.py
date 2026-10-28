"""v1_fuse_rms_and_pre_norm — fuse RMSNorm scaling and pre-normalize embeddings."""

NAME = "g_unguided_medium_10280723_a1"
EXACT = True
DESCRIPTION = "Fuse RMSNorm weight scaling, replace CenteredRMSNorm with constant weight, pre-normalize embeddings to remove per-token rms_norm."

def apply():
    import mlx.core as mx
    from mlx_lm.models.muse_glimmer import Attention, CenteredRMSNorm, MuseGlimmerModel
    from mlx_lm.models.base import scaled_dot_product_attention

    # CenteredRMSNorm weight is always 1.0+0 -> 1.0
    def centered_rms_norm_call(self, x):
        return mx.fast.rms_norm(x, 1.0, self.eps)

    CenteredRMSNorm.__call__ = centered_rms_norm_call

    # Attention: fuse qk_scale_factor into rms_norm weight
    def attention_call_opt(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)

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

    # MuseGlimmerModel: skip per-token embedding rms_norm (pre-normalized in post_load)
    def mg_call_opt(self, inputs, cache=None):
        h = self.embed_tokens(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        masks = self._masks(h, cache)
        for layer, c, mask in zip(self.layers, cache, masks):
            h = layer(h, mask, c)
        return self.norm(h)

    MuseGlimmerModel.__call__ = mg_call_opt


def post_load(model):
    import mlx.core as mx
    # Pre-normalize embedding matrix so the per-token rms_norm can be removed
    embed = model.model.embed_tokens
    weight = embed.weight
    eps = model.model.embed_eps
    normalized = mx.fast.rms_norm(weight, None, eps)
    mx.eval(normalized)
    embed.weight = normalized
