"""h8_logit_diff.py — WHY can a multi-token pass pick a different token than one-token decoding?

  python bench/h8_logit_diff.py

Tiny random Muse Glimmer (as in test_lookup_decode.py, float32, window 16). Same context,
same next 3 tokens, computed three ways; reports the largest log-prob difference from A:
  A  three one-token passes                  (normal decoding)
  B  one 3-token pass                        (H8 verify pass, everything accepted)
  C  one 3-token pass with 2 wrong tokens, rewind(2), then two one-token passes
     (H8 verify pass with rejection + rollback)
for a context BELOW the window (12 tokens: no rotation) and far BEYOND it (60 tokens).

Reading it: if B-A is about the same below and beyond the window, the difference comes from
the GPU kernels (precision), not from the cache. If it is much larger beyond the window, the
multi-token pass sees a different context there -> a real problem. C-A must equal B-A's
order of magnitude or be 0 (rewind must not add error).
Also: raw float32 matmul, GPU vs CPU, 1 row vs 3 rows (kernel precision on its own).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from lookup_decode import rewind
from test_lookup_decode import tiny_model


def lp(logits):
    x = logits.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def fresh(model, ctx):
    cache = make_prompt_cache(model)
    model(mx.array(ctx)[None], cache=cache)
    return cache


def main():
    model, args = tiny_model()
    mx.random.seed(2)
    toks = mx.random.randint(0, 64, (80,)).tolist()
    print(f"device {mx.default_device()} | window {args.sliding_window}")
    for c in (12, 60):
        ctx, nxt = toks[:c], toks[c:c + 3]
        cache = fresh(model, ctx[:-1])
        a = [lp(model(mx.array([t])[None], cache=cache)[0, -1]) for t in [ctx[-1]] + nxt[:2]]
        cache = fresh(model, ctx[:-1])
        b = lp(model(mx.array([ctx[-1]] + nxt[:2])[None], cache=cache)[0])
        cache = fresh(model, ctx[:-1])
        bad = [(nxt[0] + 1) % 64, (nxt[1] + 1) % 64]
        c0 = lp(model(mx.array([ctx[-1]] + bad)[None], cache=cache)[0])[0]
        rewind(cache, 2)
        cs = [c0] + [lp(model(mx.array([t])[None], cache=cache)[0, -1]) for t in nxt[:2]]
        dba = max(mx.max(mx.abs(b[i] - a[i])).item() for i in range(3))
        dca = max(mx.max(mx.abs(cs[i] - a[i])).item() for i in range(3))
        where = "below window" if c < args.sliding_window else "beyond window"
        print(f"context {c:3d} ({where:13s}): max |logprob diff|  B-A {dba:.2e}   C-A {dca:.2e}")
    mx.random.seed(3)
    W = mx.random.normal((128, 64))
    X = mx.random.normal((3, 64))
    g3 = X @ W.T
    g1 = mx.concatenate([X[i:i + 1] @ W.T for i in range(3)])
    cpu = mx.matmul(X, W.T, stream=mx.cpu)
    mx.eval(g3, g1, cpu)
    print(f"raw float32 matmul: |3-row - 1-row| {mx.max(mx.abs(g3 - g1)).item():.2e}   "
          f"|3-row - CPU| {mx.max(mx.abs(g3 - cpu)).item():.2e}   |1-row - CPU| {mx.max(mx.abs(g1 - cpu)).item():.2e}")


if __name__ == "__main__":
    main()
