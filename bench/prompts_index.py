"""prompts_index.py — every recorded Glimmer self-tuning prompt, with an id, for H8 races.

  python bench/prompts_index.py            # list: id, prompt tokens, tokens Glimmer generated

A recorded prompt is the conversation in runs/propose/<folder>/conv.json up to one of
Glimmer's turns (the i-th raw output was produced from the conversation up to the i-th
assistant turn, as in lookup_sim.py). Its id is "<folder>#<turn>" (turn counts from 1).
Rendered with the same chat template settings Glimmer used (reasoning strength from the
folder name, the fixed date). CPU only, a few seconds.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def recorded_prompts(hf_tok, C):
    out = []
    for conv_path in sorted((C.ROOT / "runs" / "propose").glob("*/conv.json")):
        conv = json.loads(conv_path.read_text())
        folder = conv_path.parent.name
        strength = "high" if "_high_" in folder else "medium" if "_medium_" in folder else "low"
        raws = sorted(conv_path.parent.glob("*.raw.txt"), key=lambda p: p.stat().st_mtime)
        turns = [i for i, m in enumerate(conv) if m["role"] == "assistant"]
        for n, (raw, a_idx) in enumerate(zip(raws, turns), start=1):
            rendered = hf_tok.apply_chat_template(conv[:a_idx], tokenize=False, add_generation_prompt=True,
                                                  reasoning_strength=strength, current_date=C.FIXED_DATE)
            ids = hf_tok.encode(rendered, add_special_tokens=False)
            gen = len(hf_tok.encode(raw.read_text(), add_special_tokens=False))
            out.append({"id": f"{folder}#{n}", "folder": folder, "turn": n, "ids": ids, "gen_tokens": gen})
    return out


def get_prompt(hf_tok, C, pid):
    for p in recorded_prompts(hf_tok, C):
        if p["id"] == pid:
            return pid, p["ids"]
    raise SystemExit(f"ABORT: no recorded prompt with id {pid} (list them: python bench/prompts_index.py)")


def main():
    import common as C
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(C.MODEL_DIR))
    ps = recorded_prompts(tok, C)
    print(f"{len(ps)} recorded prompts")
    print(f"{'id':48s} {'prompt':>7s} {'generated':>9s}")
    for p in ps:
        print(f"{p['id']:48s} {len(p['ids']):7d} {p['gen_tokens']:9d}")


if __name__ == "__main__":
    main()
