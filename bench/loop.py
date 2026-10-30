"""loop.py — Glimmer tries to speed up its own model code; the referee judges every try.

  python bench/loop.py --arm unguided --strength low  --attempts 3
  python bench/loop.py --arm guided   --strength high --attempts 3 --race
  python bench/loop.py --task h9 --arm guided --strength medium --attempts 4 --race   # 30 Oct task
  python bench/loop.py --task h9 --arm guided --strength medium --attempts 1 --temp 0.7 --seed 3   # one sample

--temp/--seed (30 Oct, "best-of-N"): Glimmer samples (top_p 0.95) instead of answering greedily, so
independent runs with different seeds give different solutions; the referee judges each the same way.

--task h9 (30 Oct): Glimmer is asked to make H8's 3-token check passes cheaper. It gets the measured pass
costs and MLX's attention-kernel limit, NOT our solution (docs/human_audit.md H9, recorded before this
task existed). Judging: safety gate -> golden check (bit-exact) -> pass cost at 13,759 tokens of context
(bench/verify_cost.py; stock 3-token pass = 1.44x) -> if it improves (<= 1.40x), H8 correctness on that
prompt (bench/h8_check.py, 0 wrong tokens). Every number goes back to Glimmer; it stops at the first
variant that passes all steps (then --race compares H8 with and without it, 6 pairs).

One attempt =
  1. propose.py   (fresh process): Glimmer reads the conversation and writes a variant file
  2. safety gate  (safety.py): reject disallowed imports/calls BEFORE anything runs
  3. run_one.py --mode check (fresh process): correctness against the golden reference
  4. if it failed: the exact error / verdict goes back to Glimmer as the next message
Stops at the first PASS (then optionally races it with ab.py) or after --attempts.
Every attempt is logged to runs/propose/log.csv; all raw outputs are kept.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C
import safety

PROPOSE_DIR = C.ROOT / "runs" / "propose"
LOG = PROPOSE_DIR / "log.csv"
LOG_FIELDS = ["time", "run_id", "arm", "strength", "attempt", "variant", "gen_tokens",
              "reasoning_tokens", "gen_seconds", "code_found", "safety", "verdict", "notes"]

TASK = """You are Muse Glimmer. Below is the exact Python source code that runs YOU on this computer:
an Apple M5 Max (Mac Studio, 32-core GPU, 36 GB unified memory) using MLX 0.32.4 and mlx-lm 0.32.2,
with your weights quantized to 4 bits (group size 64).

YOUR TASK: make your own text generation faster on this machine WITHOUT changing what you output.

HOW A CHANGE IS APPLIED: you write one Python "variant" file. It is imported before your weights load.
`apply()` is called BEFORE the weights load; `post_load(model)` is called AFTER they load. A variant may
patch classes/functions in `mlx_lm.models.muse_glimmer` and transform the loaded weights in memory.
It must not edit files on disk.

RULES
1. Same output. If you set EXACT = True, outputs must be bit-identical to the original. If EXACT = False,
   log-probabilities may differ only by tiny rounding and the generated tokens must still match for at
   least the first 64 steps.
2. Allowed imports only: mlx.core, mlx.nn, mlx.utils, mlx_lm.models.muse_glimmer, mlx_lm.models.base,
   math, functools, typing, dataclasses. No file, network or system access. No eval/exec/open.
3. Peak memory must stay under 25 GB (the weights alone use about 16 GB).
4. Reply with a short explanation of your idea, then the COMPLETE variant file in ONE ```python block.

HOW IT IS JUDGED: first a correctness check against reference outputs on 4 prompts (63 to 4152 tokens);
then timed runs alternating with the original code: decode speed (800 generated tokens) and prompt
processing speed (1K and 4K tokens). Faster by more than 0.1% in every paired run counts as a win.

MLX FACTS (this is MLX, not PyTorch; verified against mlx 0.32.4)
- nn.Module is a dict: sub-modules and arrays are dict entries, NOT entries of __dict__.
  Walk the model with model.modules(), model.named_modules() or model.children().
  There is no named_children(), no state_dict(), no .to(device), no torch-style hooks.
- Every nn.Linear in the loaded model has been replaced by nn.QuantizedLinear (4-bit, affine, group 64):
  its entries are `weight` (uint32, packed: 8 values per uint32, shape [out, in/8]),
  `scales` and `biases` (shape [out, in/64]), plus attributes group_size=64, bits=4, mode="affine".
  Its __call__ is mx.quantized_matmul(x, weight, scales=scales, biases=biases, transpose=True,
  group_size=group_size, bits=bits, mode=mode). Packed uint32 weights cannot be used in float math;
  use mx.dequantize / mx.quantize to convert, and keep weights quantized for speed and memory.
- MLX is lazy: arrays are computed when mx.eval(...) is called or a value is needed.
"""
PROMPT_VERSION = "p2"   # p1 = 27 Oct overnight runs (no MLX FACTS, no duplicate gate)

GUIDANCE = """
MEASUREMENTS ON THIS MACHINE (from our benchmarks of the original code)
- Decode: 26.6 tokens/s = 37.6 ms per token. The GPU is ~99.8% busy during decode. Decode is limited by
  memory bandwidth: each token reads ~15 GB of weights, about 87% of the 460 GB/s peak.
- Prompt processing: ~810 tokens/s for 1K-4K token prompts; compute-bound.
- Run-to-run noise is ~0.05%.
- Removing one small elementwise GPU operation saves only ~0.15 microseconds per token. Measured example
  (already done; a variant that only changes CenteredRMSNorm is REJECTED as a duplicate): precomputing
  `1.0 + weight` once in CenteredRMSNorm (208 ops per token) gave only +0.08% decode.
- Directions more likely to matter: fewer separate large matrix multiplications or GPU kernel launches
  per layer, less memory traffic per token, fusing chains of elementwise operations into one kernel.
"""


P8 = "guided_medium_p2_20261028-072922#8"          # 13,759-token recorded prompt
H9_STOCK_L3, H9_TARGET_L3 = 1.44, 1.40

TASK_H9 = """You are Muse Glimmer. Below is the exact Python source code that runs YOU on this computer:
an Apple M5 Max (Mac Studio, 32-core GPU, 36 GB unified memory) using MLX 0.32.4 and mlx-lm 0.32.2,
with your weights quantized to 4 bits (group size 64).

BACKGROUND: we now generate your text with prompt-lookup decoding. When the last 2 tokens you wrote
appeared earlier in the context, we guess that the 2 tokens which followed them come next, and check
the guesses in ONE forward pass of 3 tokens (your last token + 2 guesses). This made you 9-12 % faster
on short contexts, but the gain shrinks on long contexts, because a 3-token pass gets expensive.

MEASURED ON THIS MACHINE (cost of one forward pass, relative to a 1-token pass; bench/verify_cost.py)
  context  3,531 tokens: 2-token pass 1.07x, 3-token pass 1.25x, 4-token pass 1.32x
  context 13,759 tokens: 2-token pass 1.09x, 3-token pass 1.44x, 4-token pass 1.52x

MLX FACT (from MLX 0.32.4's source, mlx/backend/metal/scaled_dot_product_attention.cpp):
mx.fast.scaled_dot_product_attention has a fused "vector" kernel for short queries (query length <= 8),
but only if  query_length x (number of query heads / number of key-value heads) <= 32.
Otherwise it falls back to an unfused implementation whose cost grows with the number of keys.
You have 32 query heads and 2 key-value heads.

YOUR TASK: make the 3-token forward pass cheaper at long context, by changing your own model code.

RULES
1. One-token decoding and prompt processing must stay BIT-IDENTICAL to the original (set EXACT = True):
   only forward passes of 2 to 8 tokens may change. In those passes the outputs must be the same
   attention computed differently (tiny rounding differences are allowed there).
2. Allowed imports only: mlx.core, mlx.nn, mlx.utils, mlx_lm.models.muse_glimmer, mlx_lm.models.base,
   math, functools, typing, dataclasses. No file, network or system access. No eval/exec/open.
3. Peak memory must stay under 25 GB.
4. Reply with a short explanation of your idea, then the COMPLETE variant file in ONE ```python block.

HOW A CHANGE IS APPLIED: you write one Python "variant" file, imported before your weights load.
`apply()` is called BEFORE the weights load; `post_load(model)` AFTER. A variant may patch classes or
functions in mlx_lm.models.muse_glimmer. Attention.__call__ calls scaled_dot_product_attention, which
muse_glimmer imports from mlx_lm.models.base (source below). The `mask` it receives is None (1-token
passes), the string "causal" (MLX aligns a causal mask to the END of the keys: query i of L sees keys
0 .. K-L+i), or an array of shape (..., L, K) (sliding-window layers past the window).

HOW IT IS JUDGED: (1) the correctness check on 4 prompts must be bit-exact; (2) the cost of a 3-token
pass at 13,759 tokens of context must drop below 1.40x (stock: 1.44x); (3) prompt-lookup decoding with
your variant must produce no wrong tokens on that long prompt. You get every measurement back.
"""


def first_message_h9():
    import inspect
    import mlx_lm.models.base as mb
    import mlx_lm.models.muse_glimmer as mg
    template = (HERE / "variants" / "v0_original.py").read_text()
    msg = TASK_H9
    msg += f"\nVARIANT FILE TEMPLATE (the unmodified reference):\n```python\n{template}```\n"
    msg += f"\nFROM mlx_lm/models/base.py:\n```python\n{inspect.getsource(mb.scaled_dot_product_attention)}```\n"
    msg += f"\nSOURCE OF mlx_lm/models/muse_glimmer.py:\n```python\n{Path(mg.__file__).read_text()}```\n"
    return msg


def judge_h9(vname, folder):
    """Steps 2-3 of the h9 task. Returns (passed, feedback text, notes)."""
    import re
    c = subprocess.run([sys.executable, str(HERE / "verify_cost.py"), "--lengths", "1,2,3",
                        "--context", P8, "--variant", vname], capture_output=True, text=True, timeout=1800)
    out = (c.stdout + c.stderr).replace(str(Path.home()), "~")
    (folder / f"{vname}.cost.txt").write_text(out)
    m = re.search(r"L= 3 .*?= *([0-9.]+)x", out)
    if c.returncode != 0 or not m:
        return False, ("Your variant passed the correctness check, but measuring the pass cost failed:\n```\n"
                       + tail(out) + "\n```"), "cost run failed"
    l3 = float(m.group(1))
    if l3 > H9_TARGET_L3:
        return False, (f"Correctness check: PASS (bit-exact). Pass cost at 13,759 tokens of context:\n```\n"
                       + tail(out, 6) + f"\n```\nThe 3-token pass costs {l3:.2f}x (stock {H9_STOCK_L3}x, target <= "
                       f"{H9_TARGET_L3}x): not cheaper yet."), f"L3={l3:.2f}"
    h = subprocess.run([sys.executable, str(HERE / "h8_check.py"), "--variant", vname, "--cases", "selftune",
                        "--selftune", P8], capture_output=True, text=True, timeout=3600)
    hout = (h.stdout + h.stderr).replace(str(Path.home()), "~")
    (folder / f"{vname}.h8check.txt").write_text(hout)
    if h.returncode == 0 and "VERDICT: PASS" in hout:
        return True, "", f"L3={l3:.2f}; h8_check PASS"
    return False, (f"Correctness: PASS. 3-token pass: {l3:.2f}x (good). But prompt-lookup decoding with your "
                   f"variant produced wrong tokens on the long prompt:\n```\n" + tail(hout, 10) + "\n```"), \
        f"L3={l3:.2f}; h8_check FAIL"


def first_message(arm):
    import mlx_lm.models.muse_glimmer as mg
    template = (HERE / "variants" / "v0_original.py").read_text()
    source = Path(mg.__file__).read_text()
    msg = TASK + (GUIDANCE if arm == "guided" else "")
    msg += f"\nVARIANT FILE TEMPLATE (the unmodified reference):\n```python\n{template}```\n"
    msg += f"\nSOURCE OF mlx_lm/models/muse_glimmer.py:\n```python\n{source}```\n"
    return msg


def log_row(row):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def tail(text, n=40):
    """Last n lines, with the home-folder path shortened to ~ (keeps usernames out of logs)."""
    text = text.replace(str(Path.home()), "~")
    return "\n".join(text.strip().splitlines()[-n:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["unguided", "guided"])
    p.add_argument("--strength", required=True, choices=["low", "medium", "high"])
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--race", action="store_true", help="race a passing variant against v0")
    p.add_argument("--fake-dir", help="testing only: replies a1.txt, a2.txt, ... instead of Glimmer")
    p.add_argument("--resume", help="continue an earlier run's conversation (closed loop: refine)")
    p.add_argument("--feedback-file", help="with --resume: measurements to send Glimmer first")
    p.add_argument("--task", default="speed", choices=["speed", "h9"])
    p.add_argument("--temp", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    C.guard()
    stamp = time.strftime('%Y%m%d-%H%M%S')
    if a.resume:
        # Closed loop: same conversation, plus our measurements of Glimmer's earlier variant.
        folder = Path(a.resume)
        conv_path = folder / "conv.json"
        conv = json.loads(conv_path.read_text())
        extra = Path(a.feedback_file).read_text().strip()
        if conv[-1]["role"] == "user":            # merge with the referee's last message
            conv[-1]["content"] += "\n\n" + extra
        else:
            conv.append({"role": "user", "content": extra})
        conv_path.write_text(json.dumps(conv, indent=1))
        run_id = f"{folder.name}_refine_{stamp}"
        tag = "r"
    else:
        run_id = (f"{a.arm}_{a.strength}_{PROMPT_VERSION}_{stamp}" if a.task == "speed"
                  else f"h9_{a.arm}_{a.strength}{f'_t{a.temp}_s{a.seed}' if a.temp > 0 else ''}_{stamp}")
        folder = PROPOSE_DIR / run_id
        folder.mkdir(parents=True, exist_ok=True)
        conv_path = folder / "conv.json"
        msg = first_message(a.arm) if a.task == "speed" else first_message_h9()
        conv_path.write_text(json.dumps([{"role": "user", "content": msg}], indent=1))
        tag = "a"
    print(f"run {run_id}: arm={a.arm} strength={a.strength} attempts={a.attempts} temp={a.temp} seed={a.seed}")

    winner = None
    for k in range(1, a.attempts + 1):
        vname = (f"g_{a.arm}_{a.strength}_{time.strftime('%m%d%H%M')}_{tag}{k}" if a.task == "speed"
                 else f"g_h9_{a.strength}{f'_s{a.seed}' if a.temp > 0 else ''}_{time.strftime('%m%d%H%M')}_{tag}{k}")
        row = {"time": C.now(), "run_id": run_id, "arm": a.arm, "strength": a.strength,
               "attempt": k, "variant": vname}
        cmd = [sys.executable, str(HERE / "propose.py"), "--conv", str(conv_path),
               "--strength", a.strength, "--variant-name", vname, "--max-tokens", str(a.max_tokens),
               "--temp", str(a.temp), "--seed", str(a.seed + 1000 * k)]
        if a.fake_dir:
            cmd += ["--fake-reply", str(Path(a.fake_dir) / f"a{k}.txt")]
        print(f"\n=== attempt {k}: asking Glimmer ({vname})", flush=True)
        r = subprocess.run(cmd)
        stats_file = folder / f"{vname}.stats.json"
        stats = json.loads(stats_file.read_text()) if stats_file.exists() else {}
        row.update({key: stats.get(key, "") for key in ("gen_tokens", "reasoning_tokens", "gen_seconds")})
        row["code_found"] = r.returncode == 0

        conv = json.loads(conv_path.read_text())
        if r.returncode != 0:
            row.update({"safety": "-", "verdict": "NO CODE",
                        "notes": "hit max tokens" if stats.get("hit_max_tokens") else ""})
            feedback = ("Your reply did not contain the complete variant file in one ```python block. "
                        "Reply again with a short explanation and the COMPLETE file in ONE ```python block.")
        else:
            src = (folder / f"{vname}.py").read_text()
            problems = safety.check_source(src)
            if a.task == "speed" and a.arm == "guided" and safety.only_norm_change(src):
                problems.append("duplicate idea: this variant only changes CenteredRMSNorm, which is "
                                "already measured (+0.08%). Propose a DIFFERENT optimization.")
            if problems:
                row.update({"safety": "REJECTED", "verdict": "-", "notes": "; ".join(problems)})
                feedback = ("The safety gate rejected your file before running it:\n- "
                            + "\n- ".join(problems)
                            + "\nFix this and reply with the COMPLETE corrected file in ONE ```python block.")
            else:
                row["safety"] = "ok"
                shutil.copy(folder / f"{vname}.py", HERE / "variants" / f"{vname}.py")
                print(f"--- referee: correctness check of {vname}", flush=True)
                try:
                    c = subprocess.run([sys.executable, str(HERE / "run_one.py"), "--variant", vname,
                                        "--mode", "check"], capture_output=True, text=True, timeout=1200)
                    out = c.stdout + c.stderr
                except subprocess.TimeoutExpired:
                    c, out = None, "TIMEOUT: the correctness check took longer than 20 minutes."
                out = out.replace(str(Path.home()), "~")
                (folder / f"{vname}.check.txt").write_text(out)
                print(tail(out, 8))
                verdict = next((ln.split("VERDICT:", 1)[1].strip() for ln in out.splitlines()
                                if ln.startswith("VERDICT:")), "CRASHED")
                row["verdict"] = verdict
                if a.task == "h9" and c is not None and c.returncode == 0 and verdict.startswith("PASS (bit-exact)"):
                    print(f"--- referee: pass cost + H8 correctness of {vname}", flush=True)
                    ok, fb, notes = judge_h9(vname, folder)
                    row["notes"] = notes
                    print(notes, flush=True)
                    if ok:
                        winner = vname
                        log_row(row)
                        break
                    (HERE / "variants" / f"{vname}.py").unlink()
                    log_row(row)
                    conv.append({"role": "user", "content": fb + "\nImprove it and reply with the COMPLETE file in ONE ```python block."})
                    conv_path.write_text(json.dumps(conv, indent=1))
                    continue
                if a.task == "h9" and verdict.startswith("PASS"):
                    verdict = row["verdict"] = verdict + " (h9 requires bit-exact)"
                if a.task == "speed" and c is not None and c.returncode == 0 and verdict.startswith("PASS"):
                    winner = vname
                    log_row(row)
                    break
                (HERE / "variants" / f"{vname}.py").unlink()     # failed files leave the variants folder
                pre = ("Rule 1 requires the correctness check to be BIT-EXACT (one-token decoding and prompt "
                       "processing unchanged).\n" if a.task == "h9" else "")
                feedback = (pre + "The referee ran your file. Result (last lines of its output):\n```\n"
                            + tail(out) + "\n```\nFix the problem and reply with the COMPLETE corrected "
                            "file in ONE ```python block.")
        log_row(row)
        conv.append({"role": "user", "content": feedback})
        conv_path.write_text(json.dumps(conv, indent=1))

    if winner:
        print(f"\nPASS: {winner} is correct. It stays in bench/variants/.")
        if a.race and a.task == "h9":
            subprocess.run([sys.executable, str(HERE / "ab.py"), "--a", "v0_original@lookup",
                            "--b", f"{winner}@lookup", "--plan", "h8", "--selftune", P8, "--pairs", "6"])
        elif a.race:
            subprocess.run([sys.executable, str(HERE / "ab.py"), "--a", "v0_original",
                            "--b", winner, "--pairs", "6"])
        else:
            print(f"Race it with:  python bench/ab.py --a v0_original --b {winner} --pairs 6")
    else:
        print(f"\nNo passing variant after {a.attempts} attempts. Everything is in {folder}")


if __name__ == "__main__":
    main()
