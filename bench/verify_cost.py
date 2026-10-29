"""verify_cost.py — how much more does one forward pass cost when it checks several tokens at once?

  python bench/verify_cost.py                 # stock MLX
  python bench/verify_cost.py choose          # every 4-bit matmul routed through bench/kernels/choose.py
  python bench/verify_cost.py --lengths 1,2,3,4,5,9,17     # other pass lengths (default 1,5,9,17)

Prompt-lookup decoding replaces "one pass per token" with "one pass that checks 1 + K guessed
tokens". Its real speedup = (tokens per pass, from lookup_sim.py) / (cost of a (1+K)-token pass
relative to a 1-token pass). This script measures the second number on this GPU.

Method: process the 1K test prompt once (context ~1080 tokens, below Glimmer's 2048-token
window so the cache can be rolled back), then repeatedly run one forward pass of L new tokens,
force it to finish, time it, and roll the cache back by L so every pass sees the same context.
L = 1, 5, 9, 17 (= K of 0, 4, 8, 16). 5 warm-up + 30 timed passes each, median reported.

With a kernel name, nn.QuantizedLinear is patched (before loading) so every affine 4-bit
matmul calls that candidate's qmm(); the script counts the routed calls and aborts if the
patch never ran (proof it took effect, not just a flag).
"""
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import importlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

import common as C


ROUTED = [0]


def patch(name):
    cand = importlib.import_module(f"kernels.{name}")
    stock = nn.QuantizedLinear.__call__

    def call(self, x):
        if self.mode != "affine" or self.get("biases") is None:
            return stock(self, x)
        ROUTED[0] += 1
        y = cand.qmm(x, self["weight"], self["scales"], self["biases"], self.group_size, self.bits)
        if "bias" in self:
            y = y + self["bias"]
        return y

    nn.QuantizedLinear.__call__ = call


def main():
    C.guard()
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", nargs="?", default=None)
    ap.add_argument("--lengths", default="1,5,9,17")
    args = ap.parse_args()
    kname = args.kernel
    lengths = [int(v) for v in args.lengths.split(",")]
    if lengths[0] != 1:
        raise SystemExit("ABORT: the first length must be 1 (it is the reference)")
    if kname:
        patch(kname)
    model, tok = load(str(C.MODEL_DIR))
    ids = C.render_ids(tok._tokenizer, C.user_messages()["p1024"])
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(ids)[None], cache=cache))
    mx.eval([c.state for c in cache])
    print(f"kernel: {kname or 'stock MLX'} | context: {len(ids)} tokens (cache offset {cache[0].offset})")
    if kname:
        before = ROUTED[0]
        mx.eval(model(mx.array(ids[:9])[None], cache=cache))
        if trim_prompt_cache(cache, 9) != 9:
            raise SystemExit("ABORT: cache could not be rolled back")
        if ROUTED[0] == before:
            raise SystemExit("ABORT: patch never ran - no matmul went through the candidate")
        print(f"patch active: {ROUTED[0] - before} quantized matmuls routed per forward pass")
    base = None
    for L in lengths:
        x = mx.array(ids[:L])[None]          # any L real tokens; content does not matter for timing
        times = []
        for i in range(35):
            t0 = time.perf_counter()
            mx.eval(model(x, cache=cache))
            dt = time.perf_counter() - t0
            if trim_prompt_cache(cache, L) != L:
                raise SystemExit("ABORT: cache could not be rolled back")
            if i >= 5:
                times.append(dt)
        med = st.median(times) * 1000
        base = base or med
        print(f"L={L:2d} (K={L - 1:2d}): median {med:7.2f} ms per pass   = {med / base:4.2f}x a 1-token pass   "
              f"(spread {min(times) * 1000:.2f}-{max(times) * 1000:.2f} ms)")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
