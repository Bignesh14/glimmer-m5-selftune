"""choose: pad 5-15 tokens up to 16 ONLY for the MLP gate/up shape (out 19968),
where MLX's 16-row kernel is faster; every other shape and M goes to MLX unchanged."""
import mlx.core as mx

GATE_UP_OUT = 19968


def qmm(x, w, scales, biases, group_size, bits):
    def mm(a):
        return mx.quantized_matmul(a, w, scales=scales, biases=biases, transpose=True,
                                   group_size=group_size, bits=bits)
    lead, d = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, d)
    m = x2.shape[0]
    if w.shape[0] == GATE_UP_OUT and 5 <= m <= 15:
        xp = mx.concatenate([x2, mx.zeros((16 - m, d), dtype=x.dtype)], axis=0)
        return mm(xp)[:m].reshape(*lead, w.shape[0])
    return mm(x)
