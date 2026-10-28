"""v1_centered_scale_cache — cache CenteredRMSNorm scale to avoid per-token addition."""

NAME = "g_unguided_low_10280044_a1"
EXACT = True
DESCRIPTION = "Cache 1+weight for CenteredRMSNorm to remove per-token addition overhead."

def apply():
    """Patch CenteredRMSNorm to use a cached scale."""
    import mlx.core as mx
    from mlx_lm.models.muse_glimmer import CenteredRMSNorm

    orig_call = CenteredRMSNorm.__call__

    def cached_call(self, x: mx.array) -> mx.array:
        # Use pre-computed scale if available, otherwise fall back to original
        scale = getattr(self, "_scale", 1.0 + self.weight)
        return mx.fast.rms_norm(x, scale, self.eps)

    CenteredRMSNorm.__call__ = cached_call


def post_load(model):
    """Pre-compute 1+weight for every CenteredRMSNorm in the model."""
    import mlx.core as mx
    from mlx_lm.models.muse_glimmer import CenteredRMSNorm

    def walk(module):
        # Recursively walk nn.Module tree
        for name, child in module.__dict__.items():
            if isinstance(child, CenteredRMSNorm):
                # weight is already loaded, compute scale once
                scale = 1.0 + child.weight
                child._scale = scale
            # recurse into sub-modules
            if hasattr(child, "__dict__"):
                walk(child)
        # also handle list of layers
        for attr_name in ["layers"]:
            if hasattr(module, attr_name):
                lst = getattr(module, attr_name)
                if isinstance(lst, list):
                    for sub in lst:
                        walk(sub)

    walk(model)
    # Ensure the model itself is walked
    walk(model.model)
