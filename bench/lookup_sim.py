"""lookup_sim.py — how much could prompt-lookup decoding speed up Glimmer's OWN self-tuning runs?

Offline estimate, no GPU, takes seconds:
  python bench/lookup_sim.py

Prompt-lookup decoding: before each step, take the last n tokens generated so far, find the
most recent earlier place in the context (prompt + output so far) where the same n tokens
occur, and propose the K tokens that followed them there as a "draft". The model then checks
the whole draft in ONE forward pass and keeps the longest correct prefix (+1 token it gets
for free). Under greedy decoding the output is the same as normal decoding (mathematically).

This script replays every recorded Glimmer generation in runs/propose (its prompt and its
actual output) and counts, for several K, how many tokens each forward pass would produce.
It cannot measure time; it measures the opportunity. Tokens are re-encoded from the saved
text, so counts are approximate (within a few tokens of the original).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NGRAM_MAX, NGRAM_MIN = 4, 2
MARKER = "to=user<|message|>"     # where Glimmer's visible answer starts
KS = [4, 8, 16]


def simulate(prompt, gen, k, boundary=None):
    """Return number of forward passes needed to produce `gen` with lookup drafts of size k.
    If `boundary` is given, also return the passes that start before / after that position
    (used to split hidden reasoning from the visible answer)."""
    ctx = list(prompt)
    index = {n: {} for n in range(NGRAM_MIN, NGRAM_MAX + 1)}   # n-gram -> position after it

    def add(upto):
        for n in index:
            if upto >= n:
                index[n][tuple(ctx[upto - n:upto])] = upto

    for i in range(1, len(ctx)):          # index the prompt (positions BEFORE the last token)
        add(i)
    passes, pos, before = 0, 0, 0
    while pos < len(gen):
        draft = []
        for n in range(NGRAM_MAX, NGRAM_MIN - 1, -1):
            if len(ctx) >= n:
                hit = index[n].get(tuple(ctx[-n:]))
                if hit is not None and hit < len(ctx):
                    draft = ctx[hit:hit + k]
                    break
        acc = 0
        while acc < len(draft) and pos + acc < len(gen) and draft[acc] == gen[pos + acc]:
            acc += 1
        step = min(acc + 1, len(gen) - pos)   # accepted draft tokens + the model's own next token
        for t in gen[pos:pos + step]:
            add(len(ctx))                     # index n-grams ending at the current last token
            ctx.append(t)
        if boundary is not None and pos < boundary:
            before += 1
        pos += step
        passes += 1
    if boundary is not None:
        return passes, before, passes - before
    return passes


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_DIR))
    rows, split = [], []
    for conv_path in sorted((C.ROOT / "runs" / "propose").glob("*/conv.json")):
        conv = json.loads(conv_path.read_text())
        folder = conv_path.parent
        raws = sorted(folder.glob("*.raw.txt"), key=lambda p: p.stat().st_mtime)
        # the i-th raw output was produced from the conversation up to the i-th assistant turn
        user_turns = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        for raw, a_idx in zip(raws, user_turns):
            strength = "high" if "_high_" in folder.name else "medium" if "_medium_" in folder.name else "low"
            rendered = tok.apply_chat_template(conv[:a_idx], tokenize=False, add_generation_prompt=True,
                                               reasoning_strength=strength, current_date=C.FIXED_DATE)
            prompt = tok.encode(rendered, add_special_tokens=False)
            text = raw.read_text()
            gen = tok.encode(text, add_special_tokens=False)
            if len(gen) < 50:
                continue
            res = {k: len(gen) / simulate(prompt, gen, k) for k in KS}
            head, sep, _ = text.partition(MARKER)
            b = len(tok.encode(head + sep, add_special_tokens=False)) if sep else len(gen)
            _, p_r, p_a = simulate(prompt, gen, 8, boundary=b)
            split.append((b, p_r, len(gen) - b, p_a))
            rows.append((raw.name, len(prompt), len(gen), res))
            print(f"{raw.name[:44]:44s} prompt {len(prompt):6d}  gen {len(gen):6d}  "
                  + "  ".join(f"K={k}: {v:4.2f} tok/pass" for k, v in res.items()))
    if rows:
        tot = sum(r[2] for r in rows)
        print(f"\n{len(rows)} generations, {tot} generated tokens. Token-weighted average tokens per forward pass:")
        for k in KS:
            passes = sum(r[2] / r[3][k] for r in rows)
            print(f"  K={k:2d}: {tot / passes:.2f}")
        rt, rp = sum(x[0] for x in split), sum(x[1] for x in split)
        at, ap = sum(x[2] for x in split), sum(x[3] for x in split)
        print(f"\nSplit at K=8:  hidden reasoning {rt} tokens -> {rt / max(rp, 1):.2f} tok/pass   |   "
              f"visible answer {at} tokens -> {at / max(ap, 1):.2f} tok/pass")
        print("(1.00 = no gain. The real speedup is lower than this, because checking K tokens in one "
              "pass costs somewhat more than generating one; that must be measured on the GPU.)")


if __name__ == "__main__":
    main()
