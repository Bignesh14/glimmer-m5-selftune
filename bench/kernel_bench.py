"""kernel_bench.py — the referee for 4-bit matmul kernels (small batches of 1-16 tokens).

  python bench/kernel_bench.py mlx_ref          # sanity: MLX against itself (must be exact, ratio ~1.00)
  python bench/kernel_bench.py pad16            # a candidate in bench/kernels/pad16.py

A candidate is a file bench/kernels/<name>.py that defines

    def qmm(x, w, scales, biases, group_size, bits):
        # must equal mx.quantized_matmul(x, w, scales=scales, biases=biases,
        #                                transpose=True, group_size=group_size, bits=bits)
        ...

It may use mx.fast.metal_kernel. For each of Glimmer's layer shapes and M = 1..16 tokens the
referee (1) checks the candidate's output against MLX's, (2) times both the same way
(8 weight copies so nothing is served from cache, 40 calls per round, 3 warm-up + 7 timed
rounds, median) and reports candidate/MLX time. Target from 28 Oct: 9 tokens in <= 1.3x the
time MLX needs for 1 token, on the MLP shapes.

Correctness: bf16 output, so tiny differences from a different summation order are expected.
A shape/M passes if max|cand - ref| <= TOL_REL * max|ref| (TOL_REL = 4e-3, about half a bf16
step at the largest output value), and the report says whether it was bit-exact.
"""
import importlib
import os
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx

import common as C

SHAPES = {"mlp gate/up": (19968, 6656), "mlp down": (6656, 19968),
          "attn q": (4096, 6656), "attn k/v": (256, 6656)}
if os.environ.get("KERNEL_BENCH_TINY") == "1":          # testing the referee on a non-Mac only
    SHAPES = {"tiny": (256, 512)}
MS = [1, 2, 4, 5, 8, 9, 12, 16]
COPIES, CALLS, WARM, ROUNDS = 8, 40, 3, 7
GS, BITS = 64, 4
TOL_REL = 4e-3


def ref_qmm(x, w, s, b):
    return mx.quantized_matmul(x, w, scales=s, biases=b, transpose=True, group_size=GS, bits=BITS)


def timed(fn, x, ws):
    times = []
    for r in range(WARM + ROUNDS):
        outs = [fn(x, *ws[i % COPIES]) for i in range(CALLS)]
        t0 = time.perf_counter()
        mx.eval(outs)
        dt = (time.perf_counter() - t0) / CALLS
        if r >= WARM:
            times.append(dt)
    return st.median(times) * 1e6


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python bench/kernel_bench.py <candidate name in bench/kernels/>")
    name = sys.argv[1]
    cand = importlib.import_module(f"kernels.{name}")
    C.guard()
    print(f"candidate {name} | {C.versions()} | bf16, {BITS}-bit, group {GS}")

    def cand_fn(x, w, s, b):
        return cand.qmm(x, w, s, b, GS, BITS)

    all_ok, worst9 = True, None
    for sname, (out_f, in_f) in SHAPES.items():
        mx.random.seed(0)
        ws = [mx.quantize(mx.random.normal((out_f, in_f)).astype(mx.bfloat16), group_size=GS, bits=BITS)
              for _ in range(COPIES)]
        mx.eval(ws)
        print(f"\n{sname}: out {out_f} x in {in_f}")
        mlx_m1 = None
        for m in MS:
            x = mx.random.normal((m, in_f)).astype(mx.bfloat16)
            mx.eval(x)
            r, c = ref_qmm(x, *ws[0]), cand_fn(x, *ws[0])
            mx.eval(r, c)
            if c.shape != r.shape:
                print(f"  M={m:2d}: WRONG SHAPE {c.shape} (expected {r.shape})")
                all_ok = False
                continue
            exact = bool(mx.array_equal(c, r).item()) and c.dtype == r.dtype
            err = mx.max(mx.abs(c.astype(mx.float32) - r.astype(mx.float32))).item()
            scale = mx.max(mx.abs(r.astype(mx.float32))).item()
            ok = err <= TOL_REL * scale
            all_ok &= ok
            t_ref, t_c = timed(ref_qmm, x, ws), timed(cand_fn, x, ws)
            mlx_m1 = mlx_m1 or t_ref
            vs_m1 = t_c / mlx_m1
            if m == 9 and sname.startswith("mlp"):
                worst9 = max(worst9 or 0, vs_m1)
            print(f"  M={m:2d}: {'exact' if exact else 'ok   ' if ok else 'WRONG'}  err {err:.3g} (max {scale:.3g})"
                  f"   MLX {t_ref:8.1f} us   cand {t_c:8.1f} us   cand/MLX {t_c / t_ref:5.2f}"
                  f"   cand vs MLX@M=1 {vs_m1:5.2f}x")
    print("\nVERDICT:", "all outputs correct" if all_ok else "SOME OUTPUTS WRONG",
          "|", f"MLP shapes at 9 tokens: worst {worst9:.2f}x of MLX's 1-token time (target <= 1.30)"
          if worst9 else "")


if __name__ == "__main__":
    main()
