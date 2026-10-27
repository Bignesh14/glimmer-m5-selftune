"""v1_h1_norm_precompute — H1 from docs/human_audit.md.

Glimmer's CenteredRMSNorm computes `1.0 + self.weight` on every call. The weight never
changes after loading, so this variant computes it ONCE per norm (208 norms = 4 per
layer x 52 layers) and reuses the stored result. Same operation, same dtype, only the
moment it is computed changes -> the output must be bit-identical.
"""
import mlx.core as mx
from mlx_lm.models import muse_glimmer as mg

NAME = "v1_h1_norm_precompute"
EXACT = True          # claim: bit-identical to the original; the bench enforces it
DESCRIPTION = "Precompute 1.0 + weight once per CenteredRMSNorm instead of on every call."

EXPECTED_NORMS = 208  # 4 norms per DecoderLayer x 52 layers


def fast_call(self, x):
    # The original line was:  mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
    # Here self._scale already holds 1.0 + self.weight (computed once in post_load).
    return mx.fast.rms_norm(x, self._scale, self.eps)


def apply():
    """Before the weights load: point the class's __call__ at our version.
    Every CenteredRMSNorm object shares this one method, so all 208 are switched."""
    mg.CenteredRMSNorm.__call__ = fast_call


def post_load(model):
    """After the weights load: give every CenteredRMSNorm its precomputed _scale.
    The leading underscore keeps _scale out of the model's parameter list (MLX rule)."""
    scales = []
    for m in model.modules():
        if isinstance(m, mg.CenteredRMSNorm):
            m._scale = 1.0 + m.weight
            scales.append(m._scale)
    mx.eval(scales)   # MLX is lazy: force the 208 additions to happen now, once
    print(f"patched {len(scales)} CenteredRMSNorm modules")
    n_layers = len(model.layers)
    if len(scales) != 4 * n_layers:
        raise SystemExit(f"ABORT: expected {4 * n_layers} norms, found {len(scales)}")
