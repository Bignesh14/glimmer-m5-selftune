"""h8_sweep.py — how much does H8 gain on EVERY recorded self-tuning prompt? (breadth, not a win test)

  PYTHONUNBUFFERED=1 python bench/h8_sweep.py | tee runs/h8/sweep_<date>.txt

The formal races (ab.py, fresh process per run, 6 pairs) already proved H8 is faster on three
prompts. This sweep measures how the gain VARIES: one process, model loaded once, and for each
prompt four runs in the order normal, H8, H8, normal (so slow drift cancels), 800 tokens each,
same steady-state timing as run_one.py --plan h8. Speedup = median(H8) / median(normal).

Prompts (fixed 30 Oct, before any sweep): every recorded prompt in bench/prompts_index.py whose
recorded Glimmer answer is >= 800 tokens (shorter answers would make us time text generated
after Glimmer's natural end), plus the fixed code task. Also reported per prompt:
  eos   where normal decoding first emits an end-of-turn token within the 800 (none = never), i.e.
        whether part of the timed text is past Glimmer's natural end
  same  first token where H8's output differs from normal decoding's (bf16; -1 = identical)
"""
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.sample_utils import make_sampler

N = 800
MIN_ANSWER = 800


def run_normal(model, ids):
    sampler = make_sampler(temp=0.0)
    toks = []
    with wired_limit(model, [generation_stream]):
        for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=N, sampler=sampler)):
            toks.append(int(t))
            if i == 0:
                t_first = time.perf_counter()
        t_end = time.perf_counter()
    return toks, (len(toks) - 1) / (t_end - t_first), None


def run_h8(model, ids):
    from lookup_decode import lookup_generate
    tm = {}
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        toks, stt = lookup_generate(model, ids, N, k=2, n_min=2, timing=tm)
    return toks, (len(toks) - tm["first_pass_tokens"]) / (tm["t_end"] - tm["t_first"]), stt


def main():
    import common as C
    from mlx_lm import load
    from prompts_index import recorded_prompts
    C.guard()
    model, tok = load(str(C.MODEL_DIR))
    hf = tok._tokenizer
    eos = set(tok.eos_token_ids)
    cases = [("code task (decode case)", C.render_ids(hf, C.CODE_TASK))]
    cases += [(p["id"], p["ids"]) for p in recorded_prompts(hf, C) if p["gen_tokens"] >= MIN_ANSWER]
    print(f"{C.versions()} | {C.MODEL_DIR.name} (4-bit) | git {C.git_head()} | {len(cases)} prompts | "
          f"{N} tokens | order normal, H8, H8, normal", flush=True)
    run_normal(model, cases[0][1][:64])                      # warm-up, not recorded
    rows = []
    print(f"{'prompt':40s} {'size':>6s} {'normal':>8s} {'H8':>8s} {'gain':>7s} {'tok/pass':>8s} "
          f"{'accept':>6s} {'eos':>5s} {'same':>5s}", flush=True)
    for name, ids in cases:
        n1, s_n1, _ = run_normal(model, ids)
        h1, s_h1, stt = run_h8(model, ids)
        _, s_h2, _ = run_h8(model, ids)
        _, s_n2, _ = run_normal(model, ids)
        normal, h8 = st.median([s_n1, s_n2]), st.median([s_h1, s_h2])
        gain = 100 * (h8 / normal - 1)
        e = next((i for i, t in enumerate(n1) if t in eos), -1)
        d = next((i for i, (a, b) in enumerate(zip(n1, h1)) if a != b), -1)
        r = {"prompt": name, "prompt_tokens": len(ids), "normal_tps": [s_n1, s_n2], "h8_tps": [s_h1, s_h2],
             "gain_pct": gain, "tok_per_pass": stt["tok_per_pass"], "accept": stt["accept_rate"],
             "first_eos": e, "first_diff": d}
        rows.append(r)
        print(f"{name[:40]:40s} {len(ids):6d} {normal:8.3f} {h8:8.3f} {gain:+6.2f}% {stt['tok_per_pass']:8.3f} "
              f"{stt['accept_rate']:6.3f} {'none' if e < 0 else e:>5} {d:5d}", flush=True)
        mx.clear_cache()
    self_rows = [r for r in rows[1:]]
    g = [r["gain_pct"] for r in self_rows]
    print(f"\nself-tuning prompts ({len(g)}): gain median {st.median(g):+.2f}%, range {min(g):+.2f} .. {max(g):+.2f}%")
    clean = [r["gain_pct"] for r in self_rows if r["first_eos"] < 0]
    if clean:
        print(f"  of which never reach end-of-turn within {N} tokens ({len(clean)}): median {st.median(clean):+.2f}%")
    out = C.ROOT / "runs" / "h8" / f"sweep_{time.strftime('%Y%m%d-%H%M%S')}.json"
    C.save_json(out, {"time": C.now(), "versions": C.versions(), "git": C.git_head(), "tokens": N, "rows": rows})
    print(f"saved {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
