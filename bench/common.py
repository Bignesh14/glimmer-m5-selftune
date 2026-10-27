"""common.py — shared pieces of the Glimmer bench (the "referee").

Everything that must be identical for every variant lives here: paths, the fixed
prompts, the fixed chat-template settings, the safety checks, and the CSV format.
"""
import csv
import hashlib
import importlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # the repo folder
MODEL_DIR = ROOT / "models" / "Muse-Glimmer-30B-4bit"
PROMPT_DIR = ROOT / "prompts"                          # private, not in git
OUT_DIR = ROOT / "runs" / "bench"
GOLDEN_DIR = OUT_DIR / "golden"                        # large, not in git
CSV_PATH = OUT_DIR / "bench.csv"

# Fixed chat-template settings: the date is part of Glimmer's system prompt,
# so it must never come from the clock, or outputs change from day to day.
FIXED_DATE = "2026-10-27"
REASONING = "high"                                     # the template's default

# Correctness settings
CHECK_TOKENS = 256      # tokens generated per correctness case
LOGIT_STEPS = 64        # how many steps of full log-probabilities to keep/compare

# Speed settings (decode case matches the baseline: 800 generated tokens)
DECODE_TOKENS = 800
PREFILL_TOKENS = 8      # prefill cases only need the first few tokens

SHORT_CHAT = "Explain loop tiling in two sentences."
CODE_TASK = "Write a C function that transposes an N x N matrix of floats in place."


def user_messages():
    """The four fixed test inputs, as user messages (name -> text)."""
    return {
        "chat": SHORT_CHAT,
        "code": CODE_TASK,
        "p1024": (PROMPT_DIR / "p1024.txt").read_text(),
        "p4096": (PROMPT_DIR / "p4096.txt").read_text(),
    }


def render_ids(hf_tok, text):
    """Apply Glimmer's chat template with the fixed settings; return token ids."""
    rendered = hf_tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength=REASONING,
        current_date=FIXED_DATE,
    )
    return hf_tok.encode(rendered, add_special_tokens=False)


def low_power_mode():
    """Read Low Power Mode from pmset. Returns '0', '1' or 'unknown'."""
    if platform.system() != "Darwin":
        return "unknown"
    out = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "lowpowermode":
            return parts[1]
    return "unknown"


def guard():
    """Refuse to measure unless Low Power Mode is provably OFF.
    (BENCH_ALLOW_UNKNOWN_LPM=1 exists only for testing the bench on a non-Mac.)"""
    lpm = low_power_mode()
    allow_unknown = os.environ.get("BENCH_ALLOW_UNKNOWN_LPM") == "1"
    if lpm == "0" or (allow_unknown and lpm == "unknown"):
        return lpm
    raise SystemExit(f"ABORT: lowpowermode={lpm}. Turn Low Power Mode off first.")


def load_variant(name):
    """Import bench/variants/<name>.py and return the module and a short hash of it."""
    mod = importlib.import_module(f"variants.{name}")
    src = Path(mod.__file__).read_bytes()
    return mod, hashlib.sha256(src).hexdigest()[:12]


def model_fingerprint():
    """Total bytes of the weight files — a cheap check that the weights are the expected ones."""
    return sum(p.stat().st_size for p in MODEL_DIR.glob("*.safetensors"))


def versions():
    import mlx.core as mx
    import mlx_lm
    return f"mlx {mx.__version__} / mlx_lm {mlx_lm.__version__}"


def git_head():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


CSV_FIELDS = ["timestamp", "session", "pair", "slot", "variant", "variant_sha", "git_head",
              "case", "prompt_tokens", "prompt_tps", "gen_tokens", "gen_tps", "peak_mem_gb",
              "lowpowermode", "versions", "notes"]


def append_rows(rows):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
