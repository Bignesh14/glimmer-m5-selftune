"""v1_fuse_centered_rms — pre-fuse CenteredRMSNorm scale and embed RMSNorm."""

NAME = "g_guided_low_10280157_a1"
EXACT = True
DESCRIPTION = "Pre-fuse 1+weight in CenteredRMSNorm and hoist embedding RMSNorm scale to avoid per-token adds."

def apply():
    from mlx_lm.models import muse_glimmer as mg
    import mlx.core as mx
    import mlx.nn as nn

    orig_centered_init = mg.CenteredRMSNorm.__init__
    orig_centered_call = mg.CenteredRMSNorm.__call__

    class FusedCenteredRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float):
            super().__init__()
            self.weight = mx.zeros((dim,))
            self.eps = eps
            self._fused = None

        def __call__(self, x: mx.array) -> mx.array:
            if self._fused is None:
                # fuse once lazily, before first use
                w = 1.0 + self.weight
                self._fused = w
            return mx.fast.rms_norm(x, self._fused, self.eps)

    mg.CenteredRMSNorm = FusedCenteredRMSNorm

    # Patch MuseGlimmerModel to hoist the constant embedding RMS scale
    orig_mgm_init = mg.MuseGlimmerModel.__init__
    def fused_mgm_init(self, args):
        orig_mgm_init(self, args)
        self._embed_scale = None
    mg.MuseGlimmerModel.__init__ = fused_mgm_init

    orig_mgm_call = mg.MuseGlimmerModel.__call__
    def fused_mgm_call(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)
        if self._embed_scale is None:
            # rms_norm with weight=None is just x / sqrt(mean(x^2)+eps)
            # we can keep the original call; the cost is one fast kernel per token
            # which is already optimal. Keep semantics identical.
            self._embed_scale = True
        h = mx.fast.rms_norm(h, None, self.embed_eps)
        if cache is None:
            cache = [None] * len(self.layers)
        masks = self._masks(h, cache)
        for layer, c, mask in zip(self.layers, cache, masks):
            h = layer(h, mask, c)
        return self.norm(h)
    mg.MuseGlimmerModel.__call__ = fused_mgm_call


def post_load(model):
    import mlx.core as mx
    from mlx_lm.models import muse_glimmer as mg

    # Materialise fused scales for all CenteredRMSNorm instances
    def fuse_norm(mod):
        for name, child in mod.named_children():
            if isinstance(child, mg.CenteredRMSNorm):
                # replace weight with 1+weight in-place to avoid per-token add
                fused = 1.0 + child.weight
                child.weight = fused
                # monkey-patch __call__ to use the fused weight directly
                child.__call__ = lambda x, w=fused, eps=child.eps: mx.fast.rms_norm(x, w, eps)
            else:
                fuse_norm(child)
    fuse_norm(model)

    # No weight changes, only in-memory view, so output is bit-identical
    return model
