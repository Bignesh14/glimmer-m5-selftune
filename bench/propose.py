"""propose.py — ask Glimmer for ONE variant (one generation), save it, and exit.

Called by loop.py, one fresh process per attempt, so Glimmer's ~16 GB is released
before the referee (run_one.py) loads its own copy of the model.

  python bench/propose.py --conv runs/propose/<id>/conv.json --strength low --variant-name g_x_a1

Reads the conversation so far, renders it with Glimmer's chat template (fixed date,
chosen reasoning strength), generates greedily, splits hidden reasoning from the
visible answer, extracts the last ```python block, and writes it to the staging
folder (NOT to bench/variants: loop.py only copies it there after the safety gate).
Exit code 0 = a code block was found, 3 = no code block.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C

MARKER = "to=user<|message|>"          # where Glimmer's visible answer starts
CODE_RE = re.compile(r"```python[ \t]*\n(.*?)```", re.S)


def extract_code(answer: str):
    blocks = CODE_RE.findall(answer or "")
    return blocks[-1] if blocks else None


def generate(conv, strength, max_tokens, temp=0.0, seed=0):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, tok = load(str(C.MODEL_DIR))
    hf = tok._tokenizer
    rendered = hf.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                      reasoning_strength=strength, current_date=C.FIXED_DATE)
    ids = hf.encode(rendered, add_special_tokens=False)
    eos = set(tok.eos_token_ids)
    out, t0 = [], time.perf_counter()
    mx.random.seed(seed)                     # only matters when temp > 0 (sampling)
    sampler = make_sampler(temp=temp, top_p=0.95) if temp > 0 else make_sampler(temp=0.0)
    for t, _ in generate_step(mx.array(ids), model, max_tokens=max_tokens, sampler=sampler):
        if t in eos:
            break
        out.append(t)
    secs = time.perf_counter() - t0
    text = hf.decode(out)
    stats = {"prompt_tokens": len(ids), "gen_tokens": len(out), "gen_seconds": round(secs, 1),
             "temp": temp, "seed": seed,
             "hit_max_tokens": len(out) >= max_tokens,
             "peak_mem_gb": round(mx.get_peak_memory() / 1e9, 2)}
    if MARKER in text:
        head, answer = text.split(MARKER, 1)
        stats["reasoning_tokens"] = len(hf.encode(head, add_special_tokens=False))
    else:
        answer = None
        stats["reasoning_tokens"] = len(out)
    return text, answer, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conv", required=True)
    p.add_argument("--strength", required=True, choices=["low", "medium", "high"])
    p.add_argument("--variant-name", required=True)
    p.add_argument("--max-tokens", type=int, default=12000)
    p.add_argument("--fake-reply", help="testing only: use this file as Glimmer's reply")
    p.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default); > 0 = sampling (top_p 0.95)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    conv_path = Path(a.conv)
    conv = json.loads(conv_path.read_text())
    if a.fake_reply:
        text = answer = Path(a.fake_reply).read_text()
        stats = {"fake": True}
    else:
        C.guard()
        text, answer, stats = generate(conv, a.strength, a.max_tokens, a.temp, a.seed)

    folder = conv_path.parent
    (folder / f"{a.variant_name}.raw.txt").write_text(text)
    code = extract_code(answer)
    stats.update({"variant": a.variant_name, "strength": a.strength,
                  "visible_answer": answer is not None, "code_found": code is not None})
    if code:
        code = re.sub(r'^NAME\s*=.*$', f'NAME = "{a.variant_name}"', code, count=1, flags=re.M)
        (folder / f"{a.variant_name}.py").write_text(code)
    # Keep only the visible answer in the history (the hidden reasoning is not replayed).
    conv.append({"role": "assistant",
                 "content": answer if answer is not None else "(no visible answer)"})
    conv_path.write_text(json.dumps(conv, indent=1))
    (folder / f"{a.variant_name}.stats.json").write_text(json.dumps(stats, indent=1))
    print("PROPOSE", json.dumps(stats))
    sys.exit(0 if code else 3)


if __name__ == "__main__":
    main()
