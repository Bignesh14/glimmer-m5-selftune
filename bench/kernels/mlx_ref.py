"""mlx_ref — MLX's own kernel, used to check the referee itself (must be exact, ratio ~1.00)."""
import mlx.core as mx


def qmm(x, w, scales, biases, group_size, bits):
    return mx.quantized_matmul(x, w, scales=scales, biases=biases, transpose=True,
                               group_size=group_size, bits=bits)
