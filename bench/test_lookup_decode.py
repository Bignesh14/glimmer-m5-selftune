"""test_lookup_decode.py — is H8 output what normal greedy decoding would produce?

  python bench/test_lookup_decode.py

Uses a TINY random Muse Glimmer (same code as the real model, float32, 4 layers, sliding
window 16, vocabulary 64) so that the context passes the sliding window within a few dozen
tokens. No weights are downloaded; runs in seconds.

The check (rule of 29 Oct, fixed after the first GPU run diverged at token 192 and BEFORE
looking at why): every token H8 produced is re-scored by normal one-token-at-a-time decoding,
fed H8's own output (teacher forcing). A token is CORRECT if it is the argmax there, or a
near-tie: its log-prob is within TIE_TOL of the argmax's. A multi-token pass uses different
GPU kernels than a one-token pass, so float rounding can flip a near-tie; that is not a bug.
A wrong cache rollback gives a wrong context and large margins.
Exact identity with mlx_lm's generate_step is also reported (first differing token).

  test 1  H8 with its n-gram drafter, k=2 and k=4, 400 tokens
  test 2  forced drafts taken from the reference with 25 % corrupted, k=1..4, so passes with
          full, partial and zero acceptance occur many times past the window
  test 3  negative control: test 2 rolled back with mlx_lm's trim_prompt_cache (only valid
          below the window). The check must FAIL it, otherwise tests 1-2 prove nothing.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models import muse_glimmer
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from lookup_decode import lookup_generate

N = 400
TIE_TOL = 1e-3      # nats, float32


def tiny_model():
    args = muse_glimmer.ModelArgs(hidden_size=64, num_hidden_layers=4, intermediate_size=128,
                                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                  vocab_size=64, sliding_window=16)
    mx.random.seed(0)
    model = muse_glimmer.Model(args)
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    return model, args


def reference(model, prompt):
    sampler = make_sampler(temp=0.0)
    return [int(t) for t, _ in generate_step(mx.array(prompt), model, max_tokens=N, sampler=sampler)]


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1 if len(a) == len(b) else min(len(a), len(b)))


def rescore(model, prompt, toks):
    """Normal one-token decoding fed `toks`: returns (wrong, worst_margin, ties)."""
    cache = make_prompt_cache(model)
    model(mx.array(prompt[:-1])[None], cache=cache)
    y, wrong, ties, worst = prompt[-1], 0, 0, 0.0
    for t in toks:
        logits = model(mx.array([y])[None], cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        a = int(mx.argmax(lp).item())
        if a != t:
            margin = (lp[a] - lp[t]).item()
            worst = max(worst, margin)
            if margin <= TIE_TOL:
                ties += 1
            else:
                wrong += 1
        y = t
    return wrong, worst, ties


def report(name, model, prompt, ref, out, st):
    wrong, worst, ties = rescore(model, prompt, out)
    d = first_diff(ref, out)
    ident = "identical to generate_step" if d == -1 else f"differs from generate_step at token {d}"
    print(f"{name}: {wrong} wrong, {ties} near-ties (worst margin {worst:.2e} nats) | {ident} | "
          f"{st['tok_per_pass']:.2f} tok/pass, accepted {100 * st['accept_rate']:.0f}% of draft tokens")
    return wrong == 0


def main():
    model, args = tiny_model()
    print(f"tiny Muse Glimmer: layers {args.layer_types}, window {args.sliding_window}, "
          f"device {mx.default_device()}, tie tolerance {TIE_TOL} nats")
    mx.random.seed(1)
    seg = mx.random.randint(0, 64, (12,)).tolist()
    prompt = seg + mx.random.randint(0, 64, (10,)).tolist() + seg + seg[:5]
    ref = reference(model, prompt)
    ok = report("ref     generate_step itself", model, prompt, ref, ref,
                {"tok_per_pass": 1.0, "accept_rate": 0.0})
    for k in (2, 4):
        out, st = lookup_generate(model, prompt, N, k=k, n_min=2)
        ok &= report(f"test 1  H8 k={k}            ", model, prompt, ref, out, st)

    def forced_drafts():
        rng = random.Random(0)

        def forced(out, limit):
            d = ref[len(out):len(out) + min(rng.randint(1, 4), limit)]
            return [t if rng.random() > 0.25 else (t + 1) % 64 for t in d]
        return forced

    out, st = lookup_generate(model, prompt, N, draft_fn=forced_drafts())
    ok &= report("test 2  forced drafts       ", model, prompt, ref, out, st)
    out, st = lookup_generate(model, prompt, N, draft_fn=forced_drafts(),
                              rewind_fn=lambda c, n: trim_prompt_cache(c, n))
    caught = not report("test 3  negative control    ", model, prompt, ref, out, st)
    print(f"negative control {'CAUGHT' if caught else 'NOT CAUGHT'}")
    print("VERDICT:", "PASS" if ok and caught else "FAIL")


if __name__ == "__main__":
    main()
