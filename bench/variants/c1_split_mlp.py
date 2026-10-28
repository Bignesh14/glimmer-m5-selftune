"""c1_split_mlp — CONTROL, not an optimization.

Purpose: measure how much the outputs move when the maths stays identical but the
quantized matrix multiplications run with a different shape. Each MLP gate_proj and
up_proj is computed as two half-size quantized matmuls whose results are concatenated.
Every output number is the same dot product over the same weights as in the original;
only the GPU kernel's work split changes. Whatever difference this produces is pure
floating-point reordering noise, and it calibrates the tolerance for variants (like
merged projections) that change matmul shapes without changing the maths.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import muse_glimmer as mg

NAME = "c1_split_mlp"
EXACT = False
DESCRIPTION = "CONTROL: gate_proj/up_proj each computed as two half-size quantized matmuls."


def _half_matmul(q, x):
    parts = [mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True,
                                 group_size=q.group_size, bits=q.bits, mode=q.mode)
             for w, s, b in q._halves]
    return mx.concatenate(parts, axis=-1)


def split_mlp_call(self, x):
    return self.down_proj(nn.silu(_half_matmul(self.gate_proj, x)) * _half_matmul(self.up_proj, x))


def apply():
    mg.MLP.__call__ = split_mlp_call


def post_load(model):
    made = []
    for m in model.modules():
        if isinstance(m, mg.MLP):
            for q in (m.gate_proj, m.up_proj):
                h = q.weight.shape[0] // 2
                q._halves = [(q.weight[:h], q.scales[:h], q.biases[:h]),
                             (q.weight[h:], q.scales[h:], q.biases[h:])]
                made += [a for trip in q._halves for a in trip]
    mx.eval(made)
    print(f"control: split {len(made) // 6} projections into halves")
