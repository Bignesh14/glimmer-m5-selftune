"""lookup_decode.py — prompt-lookup decoding (H8) for Glimmer. Greedy only.

  from lookup_decode import lookup_generate
  tokens, stats = lookup_generate(model, prompt_ids, max_tokens=256, k=2, n_min=2)

Each step: look up the last n generated tokens (n = 4 down to n_min) at their most recent
earlier occurrence in prompt + output; if found, the k tokens that followed are the draft.
One forward pass checks [last token] + draft. The longest draft prefix the model agrees with
is kept, plus the model's own next token (so every pass yields >= 1 token). Policy K=2,
n_min=2 was chosen on 29 Oct from bench/lookup_policy.py with measured pass costs.

Cache rollback (why mlx_lm's trim_prompt_cache is not used): Glimmer's 39 sliding-window
layers use RotatingKVCache(max_size=2048). mlx_lm can only trim it while fewer than 2048
tokens have been seen (is_trimmable), and its speculative decoder refuses otherwise. But a
multi-token pass goes through RotatingKVCache._update_concat, which leaves the buffer in
temporal order with the new tokens at the END (_idx == buffer length). Right after such a
pass, the rejected tokens can therefore be removed exactly by slicing them off the end —
that is what rewind() does, and it checks that the cache is in that state. Full-attention
layers (KVCache) use their normal trim().
"""
import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache

PREFILL_STEP = 2048          # same chunk size as mlx_lm.generate.generate_step's default


def rewind(cache, n):
    """Drop the last n tokens from every layer's cache, right after a multi-token pass."""
    if n == 0:
        return
    for c in cache:
        if isinstance(c, RotatingKVCache):
            if c.keys is None or c._idx != c.keys.shape[2]:
                raise RuntimeError("rewind: RotatingKVCache is not in the post-multi-token state")
            c.keys = c.keys[..., : c._idx - n, :]
            c.values = c.values[..., : c._idx - n, :]
            c.offset -= n
            c._idx -= n
        elif isinstance(c, KVCache):
            if c.trim(n) != n:
                raise RuntimeError("rewind: KVCache could not be trimmed")
        else:
            raise TypeError(f"rewind: unsupported cache type {type(c).__name__}")


class Drafter:
    """n-gram index over prompt + output: n-gram -> position right after its latest occurrence."""

    def __init__(self, ids, k, n_min, n_max=4):
        self.ctx, self.k, self.ns = [], k, list(range(max(n_max, n_min), n_min - 1, -1))
        self.index = {n: {} for n in self.ns}
        for t in ids:
            self.push(t)

    def push(self, t):
        # index the n-grams ending at the current last token, then append t
        L = len(self.ctx)
        for n in self.ns:
            if L >= n:
                self.index[n][tuple(self.ctx[L - n:L])] = L
        self.ctx.append(t)

    def draft(self, limit):
        L = len(self.ctx)
        for n in self.ns:
            if L >= n:
                hit = self.index[n].get(tuple(self.ctx[L - n:L]))
                if hit is not None and hit < L:
                    return self.ctx[hit:hit + min(self.k, limit)]
        return []


def lookup_generate(model, prompt_ids, max_tokens, k=2, n_min=2, eos_ids=(), cache=None,
                    rewind_fn=rewind, draft_fn=None):
    """Greedy generation of up to max_tokens tokens. Returns (tokens, stats).
    draft_fn(out, limit) -> list replaces the n-gram drafter (used by the tests only)."""
    cache = cache if cache is not None else make_prompt_cache(model)
    ids = list(prompt_ids)
    pre = ids[:-1]
    for i in range(0, len(pre), PREFILL_STEP):
        model(mx.array(pre[i:i + PREFILL_STEP])[None], cache=cache)
        mx.eval([c.state for c in cache])
    drafter = Drafter(ids[:-1], k, n_min)
    y, out = ids[-1], []
    passes = drafted = proposed = accepted = 0
    while len(out) < max_tokens:
        drafter.push(y)                      # drafter context = prompt + output, ending in y
        limit = max_tokens - len(out) - 1
        d = draft_fn(out, limit) if draft_fn else drafter.draft(limit)
        logits = model(mx.array([y] + d)[None], cache=cache)
        pred = mx.argmax(logits[0], axis=-1).tolist()
        a = 0
        while a < len(d) and d[a] == pred[a]:
            a += 1
        rewind_fn(cache, len(d) - a)
        new = d[:a] + [pred[a]]
        passes += 1
        drafted += bool(d)
        proposed += len(d)
        accepted += a
        for t in new[:-1]:
            drafter.push(t)
        y = new[-1]
        stop = False
        for t in new:
            out.append(t)
            if t in eos_ids or len(out) >= max_tokens:
                stop = True
                break
        if stop:
            break
    stats = {"passes": passes, "tokens": len(out), "tok_per_pass": len(out) / max(passes, 1),
             "drafted_share": drafted / max(passes, 1), "accept_rate": accepted / max(proposed, 1)}
    return out, stats
