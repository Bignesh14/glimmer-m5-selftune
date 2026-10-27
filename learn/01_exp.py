import mlx.core as mx

source = """
    uint elem = thread_position_in_grid.x;
    T tmp = inp[elem];
    out[elem] = metal::exp(tmp);
"""

kernel = mx.fast.metal_kernel(
    name="myexp",
    input_names=["inp"],
    output_names=["out"],
    source=source,
)

a = mx.random.normal(shape=(4, 16)).astype(mx.float16)
out = kernel(
    inputs=[a],
    template=[("T", mx.float32)],
    grid=(a.size, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[a.shape],
    output_dtypes=[a.dtype],
    verbose=True,
)[0]
print("elements:", a.size)
print("max error vs mx.exp:", mx.max(mx.abs(out - mx.exp(a))).item())
