"""v1_fuse_centered_rms — pre-fuse CenteredRMSNorm scale."""

NAME = "g_guided_low_10280158_a2"
EXACT = True
DESCRIPTION = "Replace CenteredRMSNorm with a fused version that precomputes 1+weight, removing per-token add."

def apply():
    from mlx_lm.models import muse_glimmer as mg
    import mlx.core as mx
    import mlx.nn as nn

    class FusedCenteredRMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float):
            super().__init__()
            self.weight = mx.zeros((dim,))
            self.eps = eps
            self._scale = None

        def __call__(self, x: mx.array) -> mx.array:
            if self._scale is None:
                # fuse 1 + weight once, before first use
                self._scale = 1.0 + self.weight
            return mx.fast.rms_norm(x, self._scale, self.eps)

    # Replace the class used during construction
    mg.CenteredRMSNorm = FusedCenteredRMSNorm


def post_load(model):
    # No weight transformation needed; the fused class already avoids per-token adds.
    # Keeping post_load as a no-op preserves exact output.
    return model
