"""run_one.py — one process, one variant, one job.

  python bench/run_one.py --variant v0_original --mode golden       # make the reference (once)
  python bench/run_one.py --variant v1_xxx     --mode check        # correctness vs the reference
  python bench/run_one.py --variant v1_xxx     --mode speed ...    # one timed run (ab.py calls this)
  python bench/run_one.py --variant v0_original --mode speed --plan h8 --decoder lookup ...

--plan h8 (29 Oct): only the two decode cases that matter for prompt-lookup decoding, the
fixed code task and a recorded self-tuning prompt (bench/h8_check.py long_prompt), each
DECODE_TOKENS tokens, with ONE timing definition for both decoders: steady-state decode rate =
tokens after the first forward pass / time after the first forward pass (prefill and the
first pass excluded). --decoder lookup uses bench/lookup_decode.py (K=2, n_min=2); its sha
is recorded in notes, and the variant is recorded as "<variant>@lookup".

Order inside every run: safety check -> variant.apply() -> load weights ->
variant.post_load(model) -> the job. A fresh process per run means no state
leaks from one variant into another.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "import common" / "variants.x" work

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.sample_utils import make_sampler

import common as C

# Tolerance for variants that are NOT expected to be bit-exact (provisional; see notes in check()).
TOP_K = 20          # compare log-probabilities on the reference's 20 most likely tokens
TOL = 0.125         # = one bf16 rounding step at magnitudes 16-32


def build_cases(hf_tok):
    return {name: C.render_ids(hf_tok, text) for name, text in C.user_messages().items()}


def run_tokens(model, ids, n_tokens, n_logit_steps):
    """Greedy generation of exactly n_tokens; returns token list and the first log-prob vectors."""
    sampler = make_sampler(temp=0.0)
    toks, lps = [], []
    for i, (tok, lp) in enumerate(generate_step(mx.array(ids), model,
                                                max_tokens=n_tokens, sampler=sampler)):
        toks.append(int(tok))
        if i < n_logit_steps:
            lps.append(lp)
    logprobs = mx.stack(lps)
    mx.eval(logprobs)
    return toks, logprobs


def golden(model, cases):
    C.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    tokens = {}
    for name, ids in cases.items():
        toks, lps = run_tokens(model, ids, C.CHECK_TOKENS, C.LOGIT_STEPS)
        mx.save_safetensors(str(C.GOLDEN_DIR / f"{name}.safetensors"), {"logprobs": lps})
        tokens[name] = {"prompt_tokens": len(ids), "tokens": toks}
        print(f"golden {name}: prompt {len(ids)} tok, generated {len(toks)}, dtype {lps.dtype}")
    C.save_json(C.GOLDEN_DIR / "tokens.json", tokens)


def check(model, cases, variant_mod, vname, vsha):
    ref_tokens = json.loads((C.GOLDEN_DIR / "tokens.json").read_text())
    report = {"variant": vname, "variant_sha": vsha, "expected_exact": variant_mod.EXACT,
              "time": C.now(), "versions": C.versions(), "cases": {}}
    all_exact, all_tol = True, True
    for name, ids in cases.items():
        ref = ref_tokens[name]
        if ref["prompt_tokens"] != len(ids):
            raise SystemExit(f"ABORT: prompt for {name} differs from the golden run")
        toks, lps = run_tokens(model, ids, C.CHECK_TOKENS, C.LOGIT_STEPS)
        ref_lps = mx.load(str(C.GOLDEN_DIR / f"{name}.safetensors"))["logprobs"]

        g = ref["tokens"]
        first = next((i for i, (a, b) in enumerate(zip(g, toks)) if a != b), -1)
        match = sum(a == b for a, b in zip(g, toks)) / len(g)
        # Log-probs are only comparable while both runs have seen the same tokens,
        # i.e. up to and including the first differing step.
        k = C.LOGIT_STEPS if first == -1 else min(C.LOGIT_STEPS, first + 1)
        a, b = ref_lps[:k], lps[:k]
        bitexact = bool(mx.array_equal(a, b).item()) and a.dtype == b.dtype
        diff_all = mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
        top = mx.argsort(-a.astype(mx.float32), axis=-1)[:, :TOP_K]
        diff_top = mx.max(mx.abs(mx.take_along_axis(a.astype(mx.float32), top, -1)
                                 - mx.take_along_axis(b.astype(mx.float32), top, -1))).item()
        case_exact = (first == -1) and bitexact
        case_tol = (first == -1 or first >= C.LOGIT_STEPS) and diff_top <= TOL
        all_exact &= case_exact
        all_tol &= case_tol
        report["cases"][name] = {"first_mismatch": first, "token_match": round(match, 4),
                                 "steps_compared": k, "bitexact_logprobs": bitexact,
                                 "max_abs_diff_all": diff_all, "max_abs_diff_top20": diff_top}
        print(f"{name:6s} first_mismatch={first:4d} match={match:.4f} bitexact={bitexact} "
              f"maxdiff_all={diff_all:.4g} maxdiff_top20={diff_top:.4g}")
    if all_exact:
        verdict = "PASS (bit-exact)"
    elif variant_mod.EXACT:
        verdict = "FAIL (variant claims bit-exact, output differs)"
    elif all_tol:
        verdict = f"PASS (within tolerance {TOL} on top-{TOP_K}, tokens agree for {C.LOGIT_STEPS}+ steps)"
    else:
        verdict = "FAIL (outside tolerance)"
    report["verdict"] = verdict
    print("VERDICT:", verdict)
    C.save_json(C.OUT_DIR / "checks" / f"{vname}_{vsha}.json", report)


def timed_run(model, ids, n_tokens):
    """Same timing definition as mlx_lm's stream_generate (prompt time = until the first
    token; generation tok/s = tokens / time after the first token), but it never stops
    at an end-of-text token, so every run generates exactly n_tokens."""
    sampler = make_sampler(temp=0.0)
    prompt = mx.array(ids)
    with wired_limit(model, [generation_stream]):
        tic = time.perf_counter()
        for i, _ in enumerate(generate_step(prompt, model, max_tokens=n_tokens, sampler=sampler)):
            if i == 0:
                prompt_tps = prompt.size / (time.perf_counter() - tic)
                tic = time.perf_counter()
        gen_tps = (i + 1) / (time.perf_counter() - tic)
    return prompt.size, prompt_tps, i + 1, gen_tps


def timed_h8(model, ids, n_tokens, decoder):
    """Steady-state decode rate, same definition for both decoders (see module docstring)."""
    if decoder == "lookup":
        from lookup_decode import lookup_generate
        tm = {}
        with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
            out, st = lookup_generate(model, ids, n_tokens, k=2, n_min=2, timing=tm)
        n = len(out)
        return n, (n - tm["first_pass_tokens"]) / (tm["t_end"] - tm["t_first"]), st
    sampler = make_sampler(temp=0.0)
    with wired_limit(model, [generation_stream]):
        for i, _ in enumerate(generate_step(mx.array(ids), model, max_tokens=n_tokens, sampler=sampler)):
            if i == 0:
                t_first = time.perf_counter()
        t_end = time.perf_counter()
    return i + 1, i / (t_end - t_first), None


def speed_h8(model, cases, a, vname, vsha, lpm):
    from h8_check import long_prompt
    import hashlib
    dsha = hashlib.sha256((Path(__file__).parent / "lookup_decode.py").read_bytes()).hexdigest()[:12]
    label = f"{vname}@lookup" if a.decoder == "lookup" else vname
    folder, long_ids = long_prompt(cases["_hf"], C)
    timed_h8(model, cases["chat"], 16, a.decoder)          # warm-up, not recorded
    load_peak = mx.get_peak_memory() / 1e9
    rows = []
    for case, ids in (("decode", cases["code"]), ("selftune", long_ids)):
        mx.reset_peak_memory()
        g_tok, g_tps, st = timed_h8(model, ids, C.DECODE_TOKENS, a.decoder)
        peak = mx.get_peak_memory() / 1e9
        extra = (f" tok_per_pass={st['tok_per_pass']:.3f} accept={st['accept_rate']:.3f} decoder_sha={dsha}"
                 if st else "")
        rows.append({"timestamp": C.now(), "session": a.session, "pair": a.pair, "slot": a.slot,
                     "variant": label, "variant_sha": vsha, "git_head": C.git_head(),
                     "case": case, "prompt_tokens": len(ids), "prompt_tps": "",
                     "gen_tokens": g_tok, "gen_tps": round(g_tps, 3),
                     "peak_mem_gb": round(peak, 3), "lowpowermode": lpm, "versions": C.versions(),
                     "notes": f"plan=h8 peak_incl_load={load_peak:.2f} selftune_prompt={folder}{extra}"})
        print(f"{label} {case}: prompt {len(ids)} tok, gen {g_tok} @ {g_tps:.3f} tok/s (steady state), "
              f"peak {peak:.2f} GB{extra}")
    C.append_rows(rows)


def speed(model, cases, a, vname, vsha, lpm):
    timed_run(model, cases["chat"], 16)                   # warm-up, not recorded
    # Peak since the process started: includes loading, post_load and the first forward pass,
    # which the per-case peaks below (reset before each case) do not see.
    load_peak = mx.get_peak_memory() / 1e9
    print(f"{vname} peak incl. loading + first pass: {load_peak:.2f} GB")
    plan = [("decode", "code", C.DECODE_TOKENS),
            ("prefill_1k", "p1024", C.PREFILL_TOKENS),
            ("prefill_4k", "p4096", C.PREFILL_TOKENS)]
    rows = []
    for case, src, n in plan:
        mx.reset_peak_memory()
        p_tok, p_tps, g_tok, g_tps = timed_run(model, cases[src], n)
        peak = mx.get_peak_memory() / 1e9
        rows.append({"timestamp": C.now(), "session": a.session, "pair": a.pair, "slot": a.slot,
                     "variant": vname, "variant_sha": vsha, "git_head": C.git_head(),
                     "case": case, "prompt_tokens": p_tok, "prompt_tps": round(p_tps, 3),
                     "gen_tokens": g_tok, "gen_tps": round(g_tps, 3),
                     "peak_mem_gb": round(peak, 3), "lowpowermode": lpm,
                     "versions": C.versions(), "notes": f"peak_incl_load={load_peak:.2f}"})
        print(f"{vname} {case}: prompt {p_tok} tok @ {p_tps:.1f} tok/s, "
              f"gen {g_tok} @ {g_tps:.3f} tok/s, peak {peak:.2f} GB")
    C.append_rows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--mode", required=True, choices=["golden", "check", "speed"])
    p.add_argument("--session", default="manual")
    p.add_argument("--pair", default="0")
    p.add_argument("--slot", default="-")
    p.add_argument("--plan", default="standard", choices=["standard", "h8"])
    p.add_argument("--decoder", default="greedy", choices=["greedy", "lookup"])
    a = p.parse_args()
    if a.decoder == "lookup" and (a.mode != "speed" or a.plan != "h8"):
        raise SystemExit("ABORT: --decoder lookup is only defined for --mode speed --plan h8")

    lpm = C.guard()
    vmod, vsha = C.load_variant(a.variant)
    if a.mode == "golden" and a.variant != "v0_original":
        raise SystemExit("ABORT: the golden reference must come from v0_original")

    vmod.apply()                                      # patch code BEFORE weights load
    model, tok = load(str(C.MODEL_DIR))
    vmod.post_load(model)                             # e.g. precompute / merge weights
    cases = build_cases(tok._tokenizer)
    if a.plan == "h8":
        cases["_hf"] = tok._tokenizer

    print(f"variant {a.variant} ({vsha}) | {C.versions()} | lowpowermode={lpm} | "
          f"weights {C.model_fingerprint()} bytes")
    if a.mode == "golden":
        golden(model, cases)
    elif a.mode == "check":
        check(model, cases, vmod, a.variant, vsha)
    elif a.plan == "h8":
        speed_h8(model, cases, a, a.variant, vsha, lpm)
    else:
        speed(model, cases, a, a.variant, vsha, lpm)


if __name__ == "__main__":
    main()
