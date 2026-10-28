"""pad16 — human baseline, no new kernel: for 5-15 tokens, pad the batch with zero rows to 16 tokens.

qmm_probe.py (28 Oct) showed that for the MLP gate/up shape MLX is 3.65x slower at 9 tokens than at
1, but only 1.63x at 16 tokens (a different MLX kernel is chosen). Padding 9 -> 16 rows does more
arithmetic but may run on the faster path. The extra rows are zeros and are dropped afterwards.
For other shapes 16 tokens is NOT faster, so this candidate is expected to help only some shapes.
"""
import mlx.core as mx


def qmm(x, w, scales, biases, group_size, bits):
    m = x.shape[0]
    if 5 <= m <= 15:
        x = mx.concatenate([x, mx.zeros((16 - m, x.shape[1]), dtype=x.dtype)], axis=0)
    y = mx.quantized_matmul(x, w, scales=scales, biases=biases, transpose=True,
                            group_size=group_size, bits=bits)
    return y[:m]
