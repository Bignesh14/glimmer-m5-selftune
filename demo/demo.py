"""demo.py — Muse Glimmer writes the same answer twice on this Mac: normal decoding, then H8+H9.

  python demo/demo.py                                   # Muse Glimmer 30B (needs a Mac with >= 32 GB)
  python demo/demo.py --small                           # Meta Llama 3.2 3B (4-bit), for 8-16 GB Macs
  python demo/demo.py --prompt "Explain loop tiling." --strength medium --max-tokens 500
  python demo/demo.py --model <local folder or Hugging Face id of an mlx-lm model>

Models are downloaded from Hugging Face on first use (Glimmer ~19 GB, Llama 3.2 3B ~2 GB). All speed
numbers in the README were measured with Glimmer on a Mac Studio M5 Max; --small only shows the same
decoder working on a small Meta model (its gain on your Mac is whatever the demo measures).

What it does (one process, model loaded once, with the H9 variant installed; H9 leaves normal
decoding bit-exact, verified by the golden check):
  1. warm-up (not shown)
  2. NORMAL: mlx-lm's own greedy decoder (generate_step), text streamed live
  3. H8+H9: prompt-lookup decoding (bench/lookup_decode.py, K=2) with split attention, streamed live
  4. summary: tokens, seconds, tokens/s (after the first token, as in the bench), speed-up, tokens
     per forward pass, where the two texts first differ (bf16 near-ties, see README)
  5. writes demo/out/replay_<time>.html: both runs side by side, replayed with their REAL measured
     per-token timings (the runs themselves happen one after the other; one GPU, no interference)
"""
import argparse
import html
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step, generation_stream, wired_limit
from mlx_lm.sample_utils import make_sampler

import common as C
from lookup_decode import Drafter, lookup_generate

GRAY, BOLD, RESET = "\033[90m", "\033[1m", "\033[0m"


def render(hf, text, strength):
    s = hf.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                               reasoning_strength=strength, current_date=C.FIXED_DATE)
    return hf.encode(s, add_special_tokens=False)


class Stream:
    """Turns tokens into text pieces with timestamps, and prints them live."""

    def __init__(self, tok, t0, show=True):
        self.d = tok.detokenizer
        self.d.reset()
        self.t0, self.show, self.events, self.n = t0, show, [], 0

    def add(self, toks, t):
        for x in toks:
            self.d.add_token(x)
            self.n += 1
        seg = self.d.last_segment
        if seg:
            self.events.append([round(t - self.t0, 4), self.n, seg])
            if self.show:
                print(seg, end="", flush=True)

    def close(self, t):
        self.d.finalize()
        seg = self.d.last_segment
        if seg:
            self.events.append([round(t - self.t0, 4), self.n, seg])
            if self.show:
                print(seg, end="", flush=True)


def run_normal(model, tok, ids, n, eos, show=True):
    toks, t0 = [], time.perf_counter()
    st = Stream(tok, t0, show)
    with wired_limit(model, [generation_stream]):
        for i, (t, _) in enumerate(generate_step(mx.array(ids), model, max_tokens=n,
                                                 sampler=make_sampler(temp=0.0))):
            now = time.perf_counter()
            if i == 0:
                t_first = now
            toks.append(int(t))
            st.add([int(t)], now)
            if int(t) in eos:
                break
    t_end = time.perf_counter()
    st.close(t_end)
    return {"tokens": toks, "events": st.events, "t_first": t_first - t0, "t_end": t_end - t0,
            "first_pass_tokens": 1, "tok_per_pass": 1.0}


def run_h8(model, tok, ids, n, eos, show=True):
    t0 = time.perf_counter()
    st = Stream(tok, t0, show)
    mine = Drafter(ids[:-1], 2, 2)
    seen = {"pushed": False, "n": 0}

    def draft_fn(out, limit):
        # called at the start of every pass: everything in `out` is already computed
        now = time.perf_counter()
        if not seen["pushed"]:
            mine.push(ids[-1])
            seen["pushed"] = True
        new = out[seen["n"]:]
        for x in new:
            mine.push(x)
        if new:
            st.add(new, now)
        seen["n"] = len(out)
        return mine.draft(limit)

    tm = {}
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        out, stats = lookup_generate(model, ids, n, k=2, n_min=2, eos_ids=eos, draft_fn=draft_fn, timing=tm)
    if out[seen["n"]:]:
        st.add(out[seen["n"]:], tm["t_end"])
    st.close(tm["t_end"])
    return {"tokens": out, "events": st.events, "t_first": tm["t_first"] - t0, "t_end": tm["t_end"] - t0,
            "first_pass_tokens": tm["first_pass_tokens"], "tok_per_pass": stats["tok_per_pass"],
            "accept": stats["accept_rate"]}


def speed(r):
    return (len(r["tokens"]) - r["first_pass_tokens"]) / (r["t_end"] - r["t_first"])


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Glimmer: normal vs H8+H9</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--ink:#e8eaf0;--mute:#8b93a7;--a:#7aa2f7;--b:#9ece6a;--line:#262b36}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}
header{padding:18px 24px 6px}h1{margin:0;font-size:20px}header p{margin:4px 0 0;color:var(--mute);font-size:13px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px 24px}
.p{background:var(--panel);border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;height:72vh}
.h{display:flex;justify-content:space-between;align-items:baseline;padding:10px 14px;border-bottom:1px solid var(--line)}
.h b{font-size:16px}.A b{color:var(--a)}.B b{color:var(--b)}
.st{font-variant-numeric:tabular-nums;color:var(--mute);font-size:13px}.st span{color:var(--ink);font-weight:600}
pre{margin:0;padding:12px 14px;white-space:pre-wrap;word-break:break-word;overflow:auto;flex:1;font:13px/1.45 ui-monospace,Menlo,monospace}
.done{color:var(--b);font-weight:700}
footer{padding:0 24px 18px;color:var(--mute);font-size:12px}
button{background:var(--line);color:var(--ink);border:0;border-radius:6px;padding:6px 12px;margin-left:8px;cursor:pointer}
@media(max-width:800px){.grid{grid-template-columns:1fr}.p{height:45vh}}
</style></head><body>
<header><h1>__TITLE__ &mdash; same prompt, two decoders
<button onclick="go()">replay</button></h1><p>__INFO__</p></header>
<div class="grid">
<div class="p A"><div class="h"><b>Normal decoding (mlx-lm)</b><div class="st" id="sa"></div></div><pre id="ta"></pre></div>
<div class="p B"><div class="h"><b>__BNAME__ (this project)</b><div class="st" id="sb"></div></div><pre id="tb"></pre></div>
</div>
<footer>Replay of two real runs with their measured per-token timings (run one after the other on the same
machine, model loaded once). __FOOT__</footer>
<script>
const D=__DATA__;
function pane(r,t,te,se,ti){let i=0,txt="";return function(now){while(i<r.events.length&&r.events[i][0]<=now){txt+=r.events[i][2];i++}
te.textContent=txt;te.scrollTop=te.scrollHeight;const n=i?r.events[i-1][1]:0,el=Math.min(now,r.t_end);
const fin=now>=r.t_end;se.innerHTML=`<span>${n}</span> tokens &middot; <span>${el.toFixed(1)}</span> s &middot; `+
(fin?`<span class="done">${r.speed.toFixed(1)} tok/s &mdash; done</span>`:`<span>${(n/Math.max(el,1e-3)).toFixed(1)}</span> tok/s`);return fin}}
let raf;function go(){cancelAnimationFrame(raf);const a=pane(D.a,0,ta,sa),b=pane(D.b,0,tb,sb),s=performance.now();
(function f(){const now=(performance.now()-s)/1000;const x=a(now),y=b(now);if(!(x&&y))raf=requestAnimationFrame(f)})()}
go();
</script></body></html>"""


GLIMMER_HF = "mlx-community/Muse-Glimmer-30B-4bit"
SMALL_HF = "mlx-community/Llama-3.2-3B-Instruct-4bit"


def ram_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None, help="local folder or Hugging Face id (default: Muse Glimmer 30B)")
    p.add_argument("--small", action="store_true", help=f"use {SMALL_HF} (8 GB+ Macs)")
    p.add_argument("--prompt", default=None, help="user message (default: the bench's code task)")
    p.add_argument("--strength", default="low", choices=["low", "medium", "high"])
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--variant", default="v2_h9_split_attn")
    a = p.parse_args()
    model_id = a.model or (SMALL_HF if a.small else (str(C.MODEL_DIR) if C.MODEL_DIR.exists() else GLIMMER_HF))
    is_glimmer = "glimmer" in model_id.lower()
    ram = ram_gb()
    if is_glimmer and 0 < ram < 32:
        raise SystemExit(f"This Mac has {ram:.0f} GB of memory; Muse Glimmer 30B (4-bit) needs about 20 GB free.\n"
                         f"Try the small model instead:  python demo/demo.py --small")
    C.guard()
    vmod, vsha = C.load_variant(a.variant)
    vmod.apply()                       # H9 only patches muse_glimmer; other models are unaffected
    print(f"loading {model_id} (first run downloads it) ...", flush=True)
    model, tok = load(model_id)
    vmod.post_load(model)
    model_name = model_id.rstrip("/").split("/")[-1]
    hf, eos = tok._tokenizer, set(tok.eos_token_ids)
    text = a.prompt or C.CODE_TASK
    ids = render(hf, text, a.strength)
    print(f"{C.versions()} | {model_name} | variant {a.variant} ({vsha}) | "
          f"git {C.git_head()}\nprompt ({len(ids)} tokens, reasoning {a.strength}): {text}\n")
    run_normal(model, tok, ids[:32] if len(ids) > 32 else ids, 24, eos, show=False)          # warm-up
    print(f"{BOLD}=== NORMAL decoding (mlx-lm generate_step) ==={RESET}")
    ra = run_normal(model, tok, ids, a.max_tokens, eos)
    print(f"\n\n{BOLD}=== {'H8 + H9 (prompt lookup K=2, split attention)' if is_glimmer else 'H8 (prompt lookup K=2)'} ==={RESET}")
    rb = run_h8(model, tok, ids, a.max_tokens, eos)
    sa_, sb_ = speed(ra), speed(rb)
    diff = next((i for i, (x, y) in enumerate(zip(ra["tokens"], rb["tokens"])) if x != y), -1)
    same = "identical" if diff == -1 and len(ra["tokens"]) == len(rb["tokens"]) else f"first differ at token {diff}"
    print(f"\n\n{BOLD}SUMMARY{RESET}")
    print(f"  normal : {len(ra['tokens'])} tokens in {ra['t_end']:.2f} s  -> {sa_:.2f} tok/s")
    lab = "H8+H9" if is_glimmer else "H8   "
    print(f"  {lab}  : {len(rb['tokens'])} tokens in {rb['t_end']:.2f} s  -> {sb_:.2f} tok/s  "
          f"({rb['tok_per_pass']:.2f} tokens per forward pass, {100 * rb['accept']:.0f}% of guesses accepted)")
    print(f"  speed-up {100 * (sb_ / sa_ - 1):+.1f}%   texts: {same}"
          f"{'' if diff == -1 else ' (a bf16 near-tie; both are valid greedy continuations)'}")
    for r, s in ((ra, sa_), (rb, sb_)):
        r["speed"] = s
    info = (f"Prompt: &ldquo;{html.escape(text)}&rdquo; &middot; reasoning {a.strength} &middot; greedy &middot; "
            f"normal {sa_:.1f} tok/s vs {lab.strip()} {sb_:.1f} tok/s ({100 * (sb_ / sa_ - 1):+.1f}%)")
    foot = (f"{html.escape(C.versions())} &middot; {html.escape(model_name)} &middot; git {C.git_head()} &middot; "
            f"texts {same} &middot; {lab.strip()}: {rb['tok_per_pass']:.2f} tokens per forward pass.")
    data = {"a": {k: ra[k] for k in ("events", "t_end", "speed")}, "b": {k: rb[k] for k in ("events", "t_end", "speed")}}
    out = HERE / "out" / f"replay_{time.strftime('%Y%m%d-%H%M%S')}.html"
    out.parent.mkdir(exist_ok=True)
    title = ("Muse Glimmer 30B (4-bit) on this Mac" if is_glimmer else f"{html.escape(model_name)} on this Mac")
    out.write_text(PAGE.replace("__TITLE__", title).replace("__BNAME__", "H8 + H9" if is_glimmer else "H8").replace("__INFO__", info).replace("__FOOT__", foot)
                   .replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    print(f"\nreplay page: {out.relative_to(HERE.parent)}   (open it: open {out.relative_to(HERE.parent)})")


if __name__ == "__main__":
    main()
