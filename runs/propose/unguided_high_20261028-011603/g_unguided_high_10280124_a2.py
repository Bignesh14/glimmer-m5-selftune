"""v1_fuse_embed_and_head — pre-normalize embeddings and fold output multiplier."""

NAME = "g_unguided_high_10280124_a2"
EXACT = True
DESCRIPTION = "Pre-normalize embed_tokens rows with RMSNorm and fold output_multiplier into lm_head weights; skip runtime RMSNorm and multiply."


def apply():
    import mlx_lm.models.muse_glimmer as mg
    import mlx.core as mx

    # Patch MuseGlimmerModel to skip runtime embedding RMSNorm
    orig_muse_call = mg.MuseGlimmerModel.__call__

    def patched_muse_call(self, inputs, cache=None):
        h = self.embed_tokens(inputs)          # weights already RMS-normalized
        # embedding RMSNorm removed
        if cache is None:
            cache = [None] * len(self.layers)
        masks = self._masks(h, cache)
        for layer, c, mask in zip(self.layers, cache, masks):
            h = layer(h, mask, c)
        return self.norm(h)

    mg.MuseGlimmerModel.__call__ = patched_muse_call

    # Patch Model to skip output_multiplier multiply
    orig_model_call = mg.Model.__call__

    def patched_model_call(self, inputs, cache=None):
        out = self.model(inputs, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        # output_multiplier folded into lm_head weights in post_load
        cap = self.args.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    mg.Model.__call__ = patched_model_call


def post_load(model):
    import mlx.core as mx

    # 1. Pre-normalize embedding matrix so runtime RMSNorm can be skipped
    embed = model.model.embed_tokens
    w = embed.weight
    eps = model.model.embed_eps
    norm_w = mx.fast.rms_norm(w, None, eps)
    embed.weight = norm_w

    # 2. Fold output_multiplier into lm_head weights
    if not model.tie_word_embeddings:
        lm = model.lm_head
        mult = model.args.output_multiplier
        lm.weight = lm.weight * mult
    # tie_word_embeddings case keeps the original multiply to avoid changing input embeddings
