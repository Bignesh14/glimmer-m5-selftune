"""v1_fuse_centered_and_head — fold CenteredRMSNorm bias and output multiplier."""

NAME = "g_unguided_high_10280131_a3"
EXACT = True
DESCRIPTION = "Fold +1 into CenteredRMSNorm weights and fold output_multiplier into lm_head weights; remove runtime adds/multiplies."


def apply():
    import mlx_lm.models.muse_glimmer as mg
    import mlx.core as mx

    # CenteredRMSNorm: pre-add 1 to weight, use weight directly at runtime
    orig_centered_call = mg.CenteredRMSNorm.__call__

    def new_centered_call(self, x):
        # weight already contains the +1 offset
        return mx.fast.rms_norm(x, self.weight, self.eps)

    mg.CenteredRMSNorm.__call__ = new_centered_call

    # Model: skip output_multiplier multiply, it is folded into lm_head
    orig_model_call = mg.Model.__call__

    def new_model_call(self, inputs, cache=None):
        out = self.model(inputs, cache=cache)
        if self.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        cap = self.args.final_logit_softcapping
        return mx.tanh(out / cap) * cap

    mg.Model.__call__ = new_model_call


def post_load(model):
    # Fold +1 into all CenteredRMSNorm weights
    for layer in model.model.layers:
        for norm in (
            layer.input_layernorm,
            layer.post_attention_layernorm,
            layer.pre_feedforward_layernorm,
            layer.post_feedforward_layernorm,
        ):
            # weight was used as 1 + weight, now store 1 + weight directly
            norm.weight = norm.weight + 1.0

    # Fold output_multiplier into lm_head weights
    if not model.tie_word_embeddings:
        lm = model.lm_head
        mult = model.args.output_multiplier
        lm.weight = lm.weight * mult
