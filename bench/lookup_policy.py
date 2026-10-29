"""lookup_policy.py — estimated REAL speedup of prompt-lookup decoding, per drafting policy.

Offline, no GPU, takes about a minute:
  python bench/lookup_policy.py
  python bench/lookup_policy.py --cost "1:1,5:1.38,9:2.29,17:3.13"     # measured pass costs

Why this exists (29 Oct): the first H8 estimate divided tokens-per-pass (lookup_sim.py) by the
cost of a 9-token pass. That is wrong in two ways: a pass with no draft costs a 1-token pass,
and a short draft costs a short pass. This script replays every recorded Glimmer generation
(same data and n-gram method as lookup_sim.py), records each pass's actual draft length, and
charges each pass the measured cost of a pass of that length:

    estimated speedup = generated tokens / sum over passes of cost(1 + draft length)

cost(L) is relative to a 1-token pass, from verify_cost.py; lengths between the measured
points are interpolated linearly (approximate). Policies compared:
  K       maximum draft length
  n_min   shortest n-gram match allowed to produce a draft (longer = fewer, safer drafts)
  where   draft everywhere, or only inside the visible answer (after "to=user<|message|>")
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NGRAM_MAX = 4
MARKER = "to=user<|message|>"


def make_cost(spec):
    pts = sorted((int(a), float(b)) for a, b in (p.split(":") for p in spec.split(",")))

    def cost(L):
        if L <= pts[0][0]:
            return pts[0][1]
        for (l0, c0), (l1, c1) in zip(pts, pts[1:]):
            if L <= l1:
                return c0 + (c1 - c0) * (L - l0) / (l1 - l0)
        return pts[-1][1]
    return cost


def simulate(prompt, gen, k, n_min, start):
    """Return list of (draft_len, accepted) per forward pass. Drafting only for pos >= start."""
    ctx = list(prompt)
    nmax = max(NGRAM_MAX, n_min)
    index = {n: {} for n in range(n_min, nmax + 1)}

    def add(upto):
        for n in index:
            if upto >= n:
                index[n][tuple(ctx[upto - n:upto])] = upto

    for i in range(1, len(ctx)):
        add(i)
    passes, pos = [], 0
    while pos < len(gen):
        draft = []
        if pos >= start:
            for n in range(nmax, n_min - 1, -1):
                if len(ctx) >= n:
                    hit = index[n].get(tuple(ctx[-n:]))
                    if hit is not None and hit < len(ctx):
                        draft = ctx[hit:hit + k]
                        break
        acc = 0
        while acc < len(draft) and pos + acc < len(gen) and draft[acc] == gen[pos + acc]:
            acc += 1
        step = min(acc + 1, len(gen) - pos)
        for t in gen[pos:pos + step]:
            add(len(ctx))
            ctx.append(t)
        passes.append((len(draft), acc))
        pos += step
    return passes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cost", default="1:1,5:1.38,9:2.29,17:3.13",
                   help="pass cost by length, relative to 1 token (default: stock MLX, 29 Oct)")
    a = p.parse_args()
    cost = make_cost(a.cost)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_DIR))
    gens = []
    for conv_path in sorted((C.ROOT / "runs" / "propose").glob("*/conv.json")):
        conv = json.loads(conv_path.read_text())
        folder = conv_path.parent
        raws = sorted(folder.glob("*.raw.txt"), key=lambda q: q.stat().st_mtime)
        turns = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        for raw, a_idx in zip(raws, turns):
            strength = "high" if "_high_" in folder.name else "medium" if "_medium_" in folder.name else "low"
            rendered = tok.apply_chat_template(conv[:a_idx], tokenize=False, add_generation_prompt=True,
                                               reasoning_strength=strength, current_date=C.FIXED_DATE)
            prompt = tok.encode(rendered, add_special_tokens=False)
            text = raw.read_text()
            gen = tok.encode(text, add_special_tokens=False)
            if len(gen) < 50:
                continue
            head, sep, _ = text.partition(MARKER)
            b = len(tok.encode(head + sep, add_special_tokens=False)) if sep else len(gen)
            gens.append((prompt, gen, b))
    tot = sum(len(g) for _, g, _ in gens)
    print(f"{len(gens)} generations, {tot} tokens | cost points {a.cost}")
    print(f"{'where':7s} {'K':>3s} {'n_min':>5s} | {'tok/pass':>8s} {'drafted':>8s} {'accept':>7s} | est. speedup")
    best = None
    for where in ("all", "answer"):
        for k in (2, 4, 8, 16):
            for n_min in (2, 3, 4):
                n_tok = n_pass = n_draft = n_acc = n_prop = 0
                t = 0.0
                for prompt, gen, b in gens:
                    ps = simulate(prompt, gen, k, n_min, 0 if where == "all" else b)
                    n_tok += len(gen)
                    n_pass += len(ps)
                    n_draft += sum(1 for d, _ in ps if d)
                    n_prop += sum(d for d, _ in ps)
                    n_acc += sum(x for _, x in ps)
                    t += sum(cost(1 + d) for d, _ in ps)
                sp = n_tok / t
                best = max(best or (0,), (sp, where, k, n_min))
                print(f"{where:7s} {k:3d} {n_min:5d} | {n_tok / n_pass:8.2f} {100 * n_draft / n_pass:7.1f}% "
                      f"{100 * n_acc / max(n_prop, 1):6.1f}% | {sp:5.3f}x")
    print(f"\nBEST: {best[0]:.3f}x  (where={best[1]}, K={best[2]}, n_min={best[3]})"
          "  — 1.000x = no gain; below 1 = slower than normal decoding")
    print("drafted = share of passes that verified a draft; accept = share of proposed draft tokens accepted")


if __name__ == "__main__":
    main()
