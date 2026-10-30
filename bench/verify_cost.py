"""verify_cost.py — how much more does one forward pass cost when it checks several tokens at once?

  python bench/verify_cost.py                 # stock MLX
  python bench/verify_cost.py choose          # every 4-bit matmul routed through bench/kernels/choose.py
  python bench/verify_cost.py --lengths 1,2,3,4,5,9,17     # other pass lengths (default 1,5,9,17)
  python bench/verify_cost.py --lengths 1,2,3 --context "<recorded prompt id>"   # long context
  python bench/verify_cost.py --lengths 1,2,3 --context "<id>" --variant v2_h9_split_attn

--context (30 Oct): use a recorded self-tuning prompt (bench/prompts_index.py) as the context
instead of the 1K test prompt. Past 2048 tokens mlx_lm's trim is invalid for the sliding-window
layers, so the cache is rolled back with lookup_decode.rewind (exact, see that file).

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
    ap.add_argument("--context", default=None, help="recorded prompt id (bench/prompts_index.py)")
    ap.add_argument("--variant", default="v0_original", help="model-code variant (bench/variants/)")
    args = ap.parse_args()
    kname = args.kernel
    lengths = [int(v) for v in args.lengths.split(",")]
    if lengths[0] != 1:
        raise SystemExit("ABORT: the first length must be 1 (it is the reference)")
    if kname:
        patch(kname)
    vmod, vsha = C.load_variant(args.variant)
    vmod.apply()
    model, tok = load(str(C.MODEL_DIR))
    vmod.post_load(model)
    if args.context:
        from prompts_index import get_prompt
        _, ids = get_prompt(tok._tokenizer, C, args.context)
    else:
        ids = C.render_ids(tok._tokenizer, C.user_messages()["p1024"])
    from lookup_decode import PREFILL_STEP, rewind
    cache = make_prompt_cache(model)
    for i in range(0, len(ids), PREFILL_STEP):
        model(mx.array(ids[i:i + PREFILL_STEP])[None], cache=cache)
        mx.eval([c.state for c in cache])

    def roll_back(n):
        """Undo the pass just timed. Multi-token passes: lookup_decode.rewind (exact past the
        window). One-token passes: mlx_lm trim while below the window; past it, a one-token pass
        cannot be undone (in-place ring buffer), so the context simply grows by 1 token per pass
        (35 tokens over the measurement, negligible next to the context)."""
        if n > 1:
            rewind(cache, n)
        elif cache[0].offset <= 2048:
            if trim_prompt_cache(cache, n) != n:
                raise SystemExit("ABORT: cache could not be rolled back")
    print(f"variant {args.variant} ({vsha}) | kernel: {kname or 'stock MLX'} | context: {len(ids)} tokens (cache offset {cache[0].offset})")
    if kname:
        before = ROUTED[0]
        mx.eval(model(mx.array(ids[:9])[None], cache=cache))
        roll_back(9)
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
            roll_back(L)
            if i >= 5:
                times.append(dt)
        med = st.median(times) * 1000
        base = base or med
        print(f"L={L:2d} (K={L - 1:2d}): median {med:7.2f} ms per pass   = {med / base:4.2f}x a 1-token pass   "
              f"(spread {min(times) * 1000:.2f}-{max(times) * 1000:.2f} ms)")
    print(f"final cache offset {cache[0].offset} | peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
