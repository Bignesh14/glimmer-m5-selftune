"""v1_centered_scale_cache — cache CenteredRMSNorm scale to avoid per-token addition."""

NAME = "g_unguided_low_10280045_a2"
EXACT = True
DESCRIPTION = "Cache 1+weight for CenteredRMSNorm to remove per-token addition overhead."

def apply():
    """Patch CenteredRMSNorm to use a cached scale."""
    import mlx.core as mx
    from mlx_lm.models.muse_glimmer import CenteredRMSNorm

    orig_call = CenteredRMSNorm.__call__

    def cached_call(self, x: mx.array) -> mx.array:
        # Lazily compute and cache the scale 1 + weight once per instance
        if not hasattr(self, "_scale"):
            scale = 1.0 + self.weight
            # store as a plain attribute, no dunder access
            self._scale = scale
        return mx.fast.rms_norm(x, self._scale, self.eps)

    CenteredRMSNorm.__call__ = cached_call


def post_load(model):
    """No weight transformation needed; scale is cached lazily on first use."""
    pass
