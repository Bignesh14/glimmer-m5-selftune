"""spec_bench.py — normal decoding vs H8+H9 on a Spec-Bench subset (the standard benchmark for speculative decoding).

  git clone --depth 1 https://github.com/hemingkx/Spec-Bench.git ~/Projects/Spec-Bench     # data, Apache-2.0
  python bench/spec_bench.py --data ~/Projects/Spec-Bench/data/spec_bench/question.jsonl --round 1
  python bench/spec_bench.py --data ... --round 2          # replication (opposite run order)

Spec-Bench (Xia et al. 2024, arXiv 2401.07851) has 480 prompts in 6 tasks: multi-turn conversation (MT-bench, 80),
translation (WMT14 de-en), summarization (CNN/DM), question answering (Natural Questions), mathematical reasoning
(GSM8K), retrieval-augmented generation (80 each). Its leaderboard reports, per task, the speedup of a method over
normal decoding (and the mean accepted tokens per step).

Settings (fixed 1 Nov, before any run):
- subset: 10 evenly spaced prompts per task (every 8th of its 80, by question_id; for MT-bench this covers all 8
  of its categories) = 60 prompts; first turn only (the second
  MT-bench turn needs the model's own first answer, which differs between the two decoders)
- Glimmer 30B 4-bit with the H9 variant installed (normal decoding stays bit-exact); greedy; reasoning strength low
  (so the output is mostly the answer); up to 1024 new tokens like Spec-Bench, both decoders stop at end-of-turn
- per prompt: one normal run and one H8+H9 run, order alternating per prompt and reversed in round 2
- per task: tokens/s = all tokens produced / all wall time (prompt processing included, as Spec-Bench measures the
  whole generation call); speedup = H8+H9 tokens/s / normal tokens/s. Also reported: decode-only speedup (time after
  the first forward pass) and mean tokens per forward pass (Spec-Bench's "mean accepted tokens").
Prompt texts are NOT saved (CNN/DM and retrieved passages are third-party text): results store question ids only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.sample_utils import make_sampler

TASKS = {"mt_bench": {"writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"},
         "translation": {"translation"}, "summarization": {"summarization"}, "qa": {"qa"},
         "math_reasoning": {"math_reasoning"}, "rag": {"rag"}}
PER_TASK, MAX_NEW, STRENGTH = 10, 1024, "low"
# Spec-Bench leaderboard, PLD on an RTX 3090 with Vicuna-7B (Leaderboard.md), for reference only
PLD_3090 = {"mt_bench": 1.64, "translation": 1.15, "summarization": 2.46, "qa": 1.28, "math_reasoning": 1.72,
            "rag": 1.71, "overall": 1.64}


def subset(path):
    rows = [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]
    out = []
    for task, cats in TASKS.items():
        sel = sorted((r for r in rows if r["category"] in cats), key=lambda r: r["question_id"])[::8][:PER_TASK]
        out += [(task, r["question_id"], r["turns"][0]) for r in sel]
    return out


def normal(model, ids, eos):
    t0 = time.perf_counter()
    n = 0
    with wired_limit(model, [generation_stream]):
        for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=MAX_NEW,
                                                 sampler=make_sampler(temp=0.0))):
            n += 1
            if i == 0:
                t1 = time.perf_counter()
            if int(t) in eos:
                break
    t2 = time.perf_counter()
    return {"tokens": n, "wall": t2 - t0, "decode_tokens": n - 1, "decode_time": t2 - t1, "tok_per_pass": 1.0}


def h8(model, ids, eos):
    from lookup_decode import lookup_generate
    tm = {}
    t0 = time.perf_counter()
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        out, st = lookup_generate(model, ids, MAX_NEW, k=2, n_min=2, eos_ids=eos, timing=tm)
    return {"tokens": len(out), "wall": tm["t_end"] - t0, "decode_tokens": len(out) - tm["first_pass_tokens"],
            "decode_time": tm["t_end"] - tm["t_first"], "tok_per_pass": st["tok_per_pass"]}


def main():
    import common as C
    from mlx_lm import load
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Spec-Bench data/spec_bench/question.jsonl")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--variant", default="v2_h9_split_attn")
    a = p.parse_args()
    C.guard()
    items = subset(a.data)
    vmod, vsha = C.load_variant(a.variant)
    vmod.apply()
    model, tok = load(str(C.MODEL_DIR))
    vmod.post_load(model)
    hf, eos = tok._tokenizer, set(tok.eos_token_ids)
    print(f"{C.versions()} | {C.MODEL_DIR.name} (4-bit) | variant {a.variant} ({vsha}) | git {C.git_head()} | "
          f"Spec-Bench subset {len(items)} prompts | round {a.round} | max {MAX_NEW} new tokens | reasoning {STRENGTH}",
          flush=True)
    normal(model, hf.encode("warm up", add_special_tokens=False), eos)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_csv = C.ROOT / "runs" / "h8" / f"specbench_r{a.round}_{stamp}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["task", "question_id", "order", "prompt_tokens", "decoder", "tokens", "wall", "decode_tokens",
              "decode_time", "tok_per_pass"]
    rows = []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for j, (task, qid, text) in enumerate(items):
            s = hf.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                       add_generation_prompt=True, reasoning_strength=STRENGTH,
                                       current_date=C.FIXED_DATE)
            ids = hf.encode(s, add_special_tokens=False)
            first_normal = (j % 2 == 0) == (a.round % 2 == 1)
            order = ["normal", "h8"] if first_normal else ["h8", "normal"]
            res = {}
            for d in order:
                res[d] = normal(model, ids, eos) if d == "normal" else h8(model, ids, eos)
                r = {"task": task, "question_id": qid, "order": "-".join(order), "prompt_tokens": len(ids),
                     "decoder": d, **{k: round(v, 4) if isinstance(v, float) else v for k, v in res[d].items()}}
                w.writerow(r)
                rows.append(r)
            f.flush()
            mx.clear_cache()
            print(f"[{j + 1:2d}/{len(items)}] {task:14s} q{qid:<4d} prompt {len(ids):5d} | normal "
                  f"{res['normal']['tokens'] / res['normal']['wall']:6.2f} tok/s | H8+H9 "
                  f"{res['h8']['tokens'] / res['h8']['wall']:6.2f} tok/s | {res['h8']['tok_per_pass']:.2f} tok/pass",
                  flush=True)
    print(f"\nround {a.round}: speedup of H8+H9 over normal decoding (Glimmer 30B, M5 Max)")
    print(f"{'task':16s} {'wall-time speedup':>18s} {'decode-only':>12s} {'tok/pass':>9s} | PLD in Spec-Bench (3090)")
    summ = {}
    for task in list(TASKS) + ["overall"]:
        sel = [r for r in rows if task == "overall" or r["task"] == task]

        def tot(d, k):
            return sum(r[k] for r in sel if r["decoder"] == d)
        wall = (tot("h8", "tokens") / tot("h8", "wall")) / (tot("normal", "tokens") / tot("normal", "wall"))
        dec = (tot("h8", "decode_tokens") / tot("h8", "decode_time")) / (
            tot("normal", "decode_tokens") / tot("normal", "decode_time"))
        h8rows = [r for r in sel if r["decoder"] == "h8"]
        tpp = sum(r["tok_per_pass"] * r["tokens"] for r in h8rows) / max(sum(r["tokens"] for r in h8rows), 1)
        summ[task] = {"wall_speedup": wall, "decode_speedup": dec, "tok_per_pass": tpp}
        print(f"{task:16s} {wall:17.3f}x {dec:11.3f}x {tpp:9.2f} | {PLD_3090[task]:.2f}x")
    C.save_json(out_csv.with_suffix(".json"), {"time": C.now(), "versions": C.versions(), "round": a.round,
                                                "variant": a.variant, "settings": {"per_task": PER_TASK,
                                                "max_new": MAX_NEW, "strength": STRENGTH}, "summary": summ})
    print(f"saved {out_csv.relative_to(C.ROOT)} (+ .json)")


if __name__ == "__main__":
    main()
