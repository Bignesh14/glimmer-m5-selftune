"""qmm_probe.py — time ONLY MLX's 4-bit matrix multiply, for Glimmer's real layer shapes,
with 1 to 512 tokens at once. Does not load Glimmer (random 4-bit weights of the same
shapes; speed does not depend on the values).

  python bench/qmm_probe.py

Why: a 9-token forward pass of Glimmer costs 2.29x a 1-token pass, although both read the
same weights. If the 4-bit matmul is the cause, it shows up here as a jump in time per call
between M=1 and M=9 that is far larger than the extra arithmetic needs.

Method per (shape, M): 8 different weight matrices (so the GPU cannot reuse one from cache),
40 matmuls queued and evaluated together, 3 warm-up rounds + 7 timed rounds, median.
Reported: microseconds per matmul, ratio to M=1, and effective weight-read bandwidth
(bytes of 4-bit weights + scales + biases / time). 460 GB/s is the M5 Max spec (Apple).
"""
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx

import common as C

SHAPES = {                      # name: (out_features, in_features) as in Glimmer's QuantizedLinear
    "mlp gate/up": (19968, 6656),
    "mlp down":    (6656, 19968),
    "attn q":      (4096, 6656),
    "attn k/v":    (256, 6656),
}
MS = [1, 2, 4, 5, 8, 9, 12, 16, 17, 24, 32, 64, 512]
COPIES, CALLS, WARM, ROUNDS = 8, 40, 3, 7


def probe(out_f, in_f):
    mx.random.seed(0)
    ws = []
    for _ in range(COPIES):
        w = mx.random.normal((out_f, in_f)).astype(mx.bfloat16)
        ws.append(mx.quantize(w, group_size=64, bits=4))          # (packed weight, scales, biases)
    mx.eval(ws)
    nbytes = sum(a.nbytes for a in ws[0])
    rows = []
    base = None
    for m in MS:
        x = mx.random.normal((m, in_f)).astype(mx.bfloat16)
        mx.eval(x)
        times = []
        for r in range(WARM + ROUNDS):
            outs = [mx.quantized_matmul(x, *ws[i % COPIES], transpose=True, group_size=64, bits=4)
                    for i in range(CALLS)]
            t0 = time.perf_counter()
            mx.eval(outs)
            dt = (time.perf_counter() - t0) / CALLS
            if r >= WARM:
                times.append(dt)
        us = st.median(times) * 1e6
        base = base or us
        rows.append((m, us, us / base, nbytes / (us * 1e-6) / 1e9))
    return nbytes, rows


def main():
    C.guard()
    print(f"{C.versions()}  | dtype bfloat16, 4-bit, group 64")
    for name, (o, i) in SHAPES.items():
        nbytes, rows = probe(o, i)
        print(f"\n{name}: out {o} x in {i}  ({nbytes / 1e6:.1f} MB of 4-bit weights per call)")
        for m, us, ratio, gbs in rows:
            print(f"  M={m:4d}: {us:9.1f} us  = {ratio:5.2f}x M=1   weight-read {gbs:6.1f} GB/s")


if __name__ == "__main__":
    main()
