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

Correctness (rule of 29 Oct, fixed before any Glimmer kernel was judged):
MLX itself uses two kernels with different precision — one for 1-few tokens (used while
generating) and a tiled one for many tokens (used for every prompt). Their outputs differ from
each other by up to ~15 bf16 steps on ~39 % of elements, so "equal to MLX" is not a usable bar.
Instead every output is compared with an exact float32 reference (the same 4-bit weights
dequantized in float32, matmul in float32). A candidate PASSES if its largest error against that
reference is at most 1.25x the larger error of MLX's own two kernels on the same input (MLX at
this M, and MLX's prompt kernel forced by padding to 128 rows). I.e. "no less accurate than what
MLX already ships". The report also shows bit-exactness against MLX and the bf16-step comparison.
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
        w0, s0, b0 = ws[0]
        w32 = mx.dequantize(w0, s0.astype(mx.float32), b0.astype(mx.float32), group_size=GS, bits=BITS)
        mx.eval(w32)
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
            rf, cf = r.astype(mx.float32), c.astype(mx.float32)
            floor = max(1.0, mx.max(mx.abs(rf)).item() / 64)   # near-zero outputs: judged at 1/64 of the largest
            ulp = mx.power(2.0, mx.floor(mx.log2(mx.maximum(mx.abs(rf), floor))) - 7)
            worst = mx.max(mx.abs(cf - rf) / ulp).item()
            differ = 100 * mx.mean((cf != rf).astype(mx.float32)).item()
            ref32 = x.astype(mx.float32) @ w32.T
            xp = mx.concatenate([x, mx.zeros((128 - m, in_f), dtype=x.dtype)], axis=0)
            prompt_k = ref_qmm(xp, *ws[0])[:m]
            mx.eval(ref32, prompt_k)

            def err(a):
                return mx.max(mx.abs(a.astype(mx.float32) - ref32)).item()

            e_c, e_mlx, e_pk = err(c), err(r), err(prompt_k)
            tol = 1.25 * max(e_mlx, e_pk)
            ok = e_c <= tol
            all_ok &= ok
            t_ref, t_c = timed(ref_qmm, x, ws), timed(cand_fn, x, ws)
            mlx_m1 = mlx_m1 or t_ref
            vs_m1 = t_c / mlx_m1
            if m == 9 and sname.startswith("mlp"):
                worst9 = max(worst9 or 0, vs_m1)
            print(f"  M={m:2d}: {'exact' if exact else 'ok   ' if ok else 'WRONG'}  vs fp32: cand {e_c:5.2f}  MLX {e_mlx:5.2f}"
                  f"  MLX-prompt {e_pk:5.2f} | vs MLX: {worst:4.1f} ulp, {differ:5.2f}% differ"
                  f"   MLX {t_ref:8.1f} us   cand {t_c:8.1f} us   cand/MLX {t_c / t_ref:5.2f}"
                  f"   cand vs MLX@M=1 {vs_m1:5.2f}x")
    print("\nVERDICT:", "all outputs correct" if all_ok else "SOME OUTPUTS WRONG",
          "|", f"MLP shapes at 9 tokens: worst {worst9:.2f}x of MLX's 1-token time (target <= 1.30)"
          if worst9 else "")


if __name__ == "__main__":
    main()
