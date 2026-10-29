"""h8_check.py — does H8 on the REAL Glimmer produce what normal greedy decoding would?

  python bench/h8_check.py | tee runs/h8/h8_check_<date>.txt

Cases: the 4 golden prompts (256 tokens each, contexts below the 2048 window) and one long
recorded self-tuning prompt (>= 3000 tokens, so every pass is past the window; 512 tokens).

Rule (fixed 29 Oct, before the first real-model run): every token H8 produces is re-scored by
normal one-token-at-a-time decoding fed H8's own output (teacher forcing, same 2048-token
prefill chunks as mlx_lm's generate_step). A token is correct if it is the argmax there, or a
near-tie within TIE_TOL = 0.125 nats (the tolerance run_one.py already uses: one bf16 step).
Anything else is WRONG. PASS = 0 wrong tokens in all cases. For information it also reports
the first token where H8 differs from the golden reference (generate_step), and H8's real
tokens per pass and acceptance rate (the simulation predicted 1.43 tok/pass, 40 %).

Diagnostics added 29 Oct AFTER the first run failed (1 wrong token, margin 0.25 nats, in the
"chat" case at 247 tokens of context, i.e. below the window):
  --fp32           run the model with float32 activations (scales/biases cast to float32,
                   4-bit weights unchanged) and MLX_ENABLE_TF32=0. Rounding then nearly
                   vanishes, so H8 must give 0 wrong and 0 near-ties; a wrong token here = bug.
  --mlx-baseline   bf16, no H8: re-score the golden reference (MLX's OWN normal decoding output)
                   with MLX's prompt path (one forward over prompt + output, as when the text is
                   read back as a prompt), same rule. Shows how often MLX disagrees with itself.
  --cases a,b      run only these cases (chat, code, p1024, p4096, selftune)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from lookup_decode import PREFILL_STEP, lookup_generate

TIE_TOL = 0.125
LONG_TOKENS = 512
LONG_MIN_PROMPT = 3000
K, N_MIN = 2, 2


def prefill(model, ids, cache):
    pre = ids[:-1]
    for i in range(0, len(pre), PREFILL_STEP):
        model(mx.array(pre[i:i + PREFILL_STEP])[None], cache=cache)
        mx.eval([c.state for c in cache])


def rescore(model, ids, toks):
    """Normal one-token decoding fed `toks`. Returns (wrong, ties, worst_margin, first_wrongs)."""
    cache = make_prompt_cache(model)
    prefill(model, ids, cache)
    y, wrong, ties, worst, firsts = ids[-1], 0, 0, 0.0, []
    for i, t in enumerate(toks):
        logits = model(mx.array([y])[None], cache=cache)[0, -1].astype(mx.float32)
        lp = logits - mx.logsumexp(logits)
        a = int(mx.argmax(lp).item())
        if a != t:
            m = (lp[a] - lp[t]).item()
            worst = max(worst, m)
            if m <= TIE_TOL:
                ties += 1
            else:
                wrong += 1
                if len(firsts) < 5:
                    firsts.append({"pos": i, "normal": a, "h8": t, "margin": round(m, 4)})
        y = t
        if i % 256 == 0:
            mx.clear_cache()
    return wrong, ties, worst, firsts


def rescore_prefill(model, ids, toks):
    """MLX's prompt path: forward over prompt + toks (2048-token chunks), score each tok."""
    seq = list(ids) + list(toks)
    inp, start = seq[:-1], len(ids) - 1          # row p predicts seq[p + 1]
    cache = make_prompt_cache(model)
    wrong, ties, worst, firsts = 0, 0, 0.0, []
    for i in range(0, len(inp), PREFILL_STEP):
        logits = model(mx.array(inp[i:i + PREFILL_STEP])[None], cache=cache)[0]
        lo = max(start, i)
        if lo >= i + logits.shape[0]:
            mx.eval([c.state for c in cache])
            continue
        rows = logits[lo - i:].astype(mx.float32)
        lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
        tgt = mx.array(seq[lo + 1:i + logits.shape[0] + 1])
        am = mx.argmax(lp, axis=-1)
        marg = mx.take_along_axis(lp, am[:, None], -1)[:, 0] - mx.take_along_axis(lp, tgt[:, None], -1)[:, 0]
        for j, (a, t, m) in enumerate(zip(am.tolist(), tgt.tolist(), marg.tolist())):
            if a != t:
                worst = max(worst, m)
                if m <= TIE_TOL:
                    ties += 1
                else:
                    wrong += 1
                    if len(firsts) < 5:
                        firsts.append({"pos": lo - start + j, "prompt_path": a, "decoded": t, "margin": round(m, 4)})
        mx.eval([c.state for c in cache])
    return wrong, ties, worst, firsts


def baseline_case(model, name, ids, golden_tokens):
    wrong, ties, worst, firsts = rescore_prefill(model, ids, golden_tokens)
    print(f"{name:28s} prompt {len(ids):5d} | MLX prompt path vs MLX decoding: {wrong} wrong, {ties} near-ties "
          f"(worst {worst:.3f} nats) over {len(golden_tokens)} tokens", flush=True)
    for f in firsts:
        print(f"    WRONG at {f['pos']}: prompt path picks {f['prompt_path']}, decoding gave {f['decoded']}, margin {f['margin']}")
    return {"case": name, "mode": "mlx-baseline", "wrong": wrong, "ties": ties, "worst_margin": worst,
            "first_wrongs": firsts}


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), -1)


def check_case(model, name, ids, n, golden_tokens=None):
    t0 = time.perf_counter()
    out, st = lookup_generate(model, ids, n, k=K, n_min=N_MIN)
    t_h8 = time.perf_counter() - t0
    wrong, ties, worst, firsts = rescore(model, ids, out)
    gd = first_diff(golden_tokens, out) if golden_tokens else None
    gtxt = "n/a" if gd is None else ("identical" if gd == -1 else f"first differs at {gd}")
    print(f"{name:28s} prompt {len(ids):5d} | {wrong} wrong, {ties} near-ties (worst {worst:.3f} nats) | "
          f"vs golden: {gtxt} | {st['tok_per_pass']:.2f} tok/pass, accepted {100 * st['accept_rate']:.0f}% "
          f"| H8 {len(out) / t_h8:.1f} tok/s incl. prefill", flush=True)
    for f in firsts:
        print(f"    WRONG at {f['pos']}: normal decoding picks {f['normal']}, H8 gave {f['h8']}, margin {f['margin']}")
    return {"case": name, "prompt_tokens": len(ids), "tokens": len(out), "wrong": wrong, "ties": ties,
            "worst_margin": worst, "first_wrongs": firsts, "golden_first_diff": gd, **st}


def long_prompt(hf_tok, C):
    """The first recorded self-tuning prompt (sorted by folder name) of >= LONG_MIN_PROMPT tokens."""
    for conv_path in sorted((C.ROOT / "runs" / "propose").glob("*/conv.json")):
        conv = json.loads(conv_path.read_text())
        idx = next((i for i, m in enumerate(conv) if m["role"] == "assistant"), None)
        if idx is None:
            continue
        folder = conv_path.parent.name
        strength = "high" if "_high_" in folder else "medium" if "_medium_" in folder else "low"
        rendered = hf_tok.apply_chat_template(conv[:idx], tokenize=False, add_generation_prompt=True,
                                              reasoning_strength=strength, current_date=C.FIXED_DATE)
        ids = hf_tok.encode(rendered, add_special_tokens=False)
        if len(ids) >= LONG_MIN_PROMPT:
            return folder, ids
    raise SystemExit(f"ABORT: no recorded self-tuning prompt of >= {LONG_MIN_PROMPT} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--mlx-baseline", action="store_true")
    ap.add_argument("--cases", default="chat,code,p1024,p4096,selftune")
    a = ap.parse_args()
    want = a.cases.split(",")
    if a.fp32 and os.environ.get("MLX_ENABLE_TF32") != "0":
        raise SystemExit("ABORT: --fp32 needs MLX_ENABLE_TF32=0 (otherwise multi-row float32 runs in TF32)")
    import common as C
    from mlx_lm import load
    C.guard()
    model, tok = load(str(C.MODEL_DIR))
    if a.fp32:
        model.set_dtype(mx.float32)          # floating params only; packed 4-bit weights stay uint32
        mx.eval(model.parameters())
    hf = tok._tokenizer
    golden = json.loads((C.GOLDEN_DIR / "tokens.json").read_text())
    mode = "fp32" if a.fp32 else "mlx-baseline" if a.mlx_baseline else "bf16"
    print(f"{C.versions()} | model {C.MODEL_DIR.name} (4-bit) | weights {C.model_fingerprint()} bytes | "
          f"MLX_ENABLE_TF32={os.environ.get('MLX_ENABLE_TF32', 'unset (default 1)')} | mode {mode} | "
          f"K={K} n_min={N_MIN} | tie tolerance {TIE_TOL} nats | git {C.git_head()}")
    results = []
    for name, text in C.user_messages().items():
        if name not in want:
            continue
        ids = C.render_ids(hf, text)
        if a.mlx_baseline:
            results.append(baseline_case(model, name, ids, golden[name]["tokens"]))
        else:
            results.append(check_case(model, name, ids, C.CHECK_TOKENS,
                                      None if a.fp32 else golden[name]["tokens"]))
    if "selftune" in want and not a.mlx_baseline:
        folder, ids = long_prompt(hf, C)
        results.append(check_case(model, f"selftune {folder[:19]}", ids, LONG_TOKENS))
    total = sum(r["wrong"] for r in results)
    out = C.ROOT / "runs" / "h8" / f"h8_check_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    C.save_json(out, {"time": C.now(), "versions": C.versions(), "mode": mode, "tie_tol": TIE_TOL, "k": K,
                      "n_min": N_MIN, "results": results})
    print(f"saved {out.relative_to(C.ROOT)}")
    print("VERDICT:", "PASS (0 wrong tokens)" if total == 0 else f"FAIL ({total} wrong tokens)")

if __name__ == "__main__":
    main()
