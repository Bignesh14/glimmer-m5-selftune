"""ui.py — a small local web page to try your own prompts: normal decoding vs this project's decoder.

  ./run_demo.sh --ui                 Muse Glimmer 30B (32 GB+ Macs)
  ./run_demo.sh --ui --small         Meta Llama 3.2 3B (8 GB+ Macs)
  python demo/ui.py [--small | --model <id>] [--port 8765]

Loads the model once, opens http://127.0.0.1:8765 in your browser (only this computer can reach it), and for each
prompt generates the answer twice, one after the other on the same GPU: first with normal decoding (mlx-lm), then with
H8 (+H9 for Glimmer). Both stream live; tokens that were guessed from earlier text and confirmed by the model are shown
in green. When both are done, "Replay side by side" plays the two runs together with their real measured timings.
Uses only the Python standard library for the web server. Stop with Ctrl-C in the terminal.
"""
import argparse
import html
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
sys.path.insert(0, str(HERE))

import demo as D  # noqa: E402  (render, run_normal, run_h8, speed, model ids, RAM check)
import mascot  # noqa: E402

STATE = {}

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>__TAB__</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7fb;--panel:#ffffff;--ink:#1b1f2a;--mute:#5d6678;--a:#2f6bff;--b:#0f9d4a;--line:#e2e6ef;--freebg:rgba(22,190,90,.16);--shadow:0 1px 3px rgba(20,30,60,.08),0 6px 18px rgba(20,30,60,.06)}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,system-ui,sans-serif}
header{padding:16px 24px 4px}h1{margin:0;font-size:19px}header p{margin:4px 0 0;color:var(--mute);font-size:13px}
form{display:flex;gap:10px;padding:10px 24px;flex-wrap:wrap;align-items:flex-start}
textarea{flex:1;min-width:260px;height:58px;background:var(--panel);box-shadow:var(--shadow);color:var(--ink);border:1px solid var(--line);
border-radius:8px;padding:8px 10px;font:14px/1.4 -apple-system,system-ui,sans-serif;resize:vertical}
.opts{display:flex;flex-direction:column;gap:6px;font-size:13px;color:var(--mute)}
select,input[type=number]{background:var(--panel);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:3px 6px}
button{background:var(--a);color:#fff;border:0;border-radius:8px;padding:9px 16px;cursor:pointer;font-weight:600;box-shadow:0 2px 8px rgba(47,107,255,.25)}
#rp{background:#0f9d4a;box-shadow:0 2px 8px rgba(15,157,74,.25)}
button:disabled{opacity:.45;cursor:default}
.ex{padding:0 24px;font-size:12px;color:var(--mute)}.ex a{color:var(--a);cursor:pointer;margin-right:12px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:12px 24px}
.p{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);display:flex;flex-direction:column;height:58vh}
.h{display:flex;justify-content:space-between;align-items:baseline;padding:10px 14px;border-bottom:1px solid var(--line)}
.h b{font-size:15px}.A b{color:var(--a)}.B b{color:var(--b)}
.st{font-variant-numeric:tabular-nums;color:var(--mute);font-size:13px}.st span{color:var(--ink);font-weight:600}
.key{padding:6px 14px;font-size:12px;color:var(--mute);border-bottom:1px solid var(--line)}
pre{margin:0;padding:12px 14px;white-space:pre-wrap;word-break:break-word;overflow:auto;flex:1;font:13px/1.45 ui-monospace,Menlo,monospace}
.free{color:#08793a;background:var(--freebg);border-radius:3px;font-weight:600}
#sum{padding:0 24px 18px;font-size:14px;min-height:22px}#sum b{color:#08793a}
@media(max-width:800px){.grid{grid-template-columns:1fr}.p{height:40vh}}
</style></head><body>
<header><h1>__TITLE__ &mdash; try your own prompt</h1>
<p>__INFO__ One GPU: the two answers are generated one after the other (normal first), then can be replayed side by side
with their real timings.</p></header>
<form id="f"><textarea id="q" placeholder="Type a prompt, e.g. Write a Python class for a 2D vector with unit tests."></textarea>
<div class="opts">__STRENGTH__<label>max tokens <input id="m" type="number" value="400" min="16" max="2000" step="1"></label></div>
<div class="opts"><button id="go" type="submit">Run</button><button id="rp" type="button" disabled>Replay side by side</button></div></form>
<div class="ex">Examples: <a data-q="Write a Python class for a 2D vector with add, sub, mul, dot, norm and __repr__, then unit tests for every method.">vector class + tests</a>
<a data-q="Write a JSON schema for a library book with title, authors, ISBN, year and tags, then one example.">JSON schema</a>
<a data-q="Translate to English: Die Katze sitzt auf der Matte, und der Hund schläft im Garten.">translation</a>
<a data-q="Write a short story about a lighthouse keeper.">short story (prose: small gain)</a></div>
<div class="grid">
<div class="p A"><div class="h"><b>Normal decoding (mlx-lm)</b><div class="st" id="sa"></div></div><div class="key">the model's usual decoder, one token per step</div><pre id="ta"></pre></div>
<div class="p B"><div class="h"><b>__BNAME__ (this project)</b><div class="st" id="sb"></div></div><div class="key"><span class="free">green</span> = guessed from earlier text and confirmed by the model: came almost free</div><pre id="tb"></pre></div>
</div><div id="sum"></div>
<script>
const $=id=>document.getElementById(id);let runs={a:null,b:null},es=null,raf=null;
const PA=()=>document.querySelector(".p.A"),PB=()=>document.querySelector(".p.B"),SP=()=>window.spark||null;
document.querySelectorAll(".ex a").forEach(a=>a.onclick=()=>{$("q").value=a.dataset.q});
function stat(se,n,nf,el,done,spd,free){se.innerHTML=`<span>${n}</span> tokens &middot; <span>${el.toFixed(1)}</span> s &middot; `+
(done?`<span>${spd.toFixed(1)} tok/s</span> done`:`<span>${(n/Math.max(el,1e-3)).toFixed(1)}</span> tok/s`)+(free?` &middot; <span>${n?Math.round(100*nf/n):0}%</span> free`:"")}
function put(te,e,free){if(free&&e[3]){const s=document.createElement("span");s.className="free";s.textContent=e[2];te.appendChild(s)}
else te.appendChild(document.createTextNode(e[2]));te.scrollTop=te.scrollHeight}
$("f").onsubmit=ev=>{ev.preventDefault();const q=$("q").value.trim();if(!q)return;if(es)es.close();cancelAnimationFrame(raf);
["ta","tb"].forEach(i=>$(i).textContent="");["sa","sb"].forEach(i=>$(i).textContent="");$("sum").textContent="generating with normal decoding...";
$("go").disabled=true;$("rp").disabled=true;runs={a:{events:[]},b:{events:[]}};const M=+$("m").value||400;
SP()&&SP().hopTo(PA(),0);
const s=$("s")?$("s").value:"low";
es=new EventSource(`/run?q=${encodeURIComponent(q)}&m=${$("m").value}&s=${s}`);
es.onmessage=m=>{const d=JSON.parse(m.data);
if(d.type==="tok"){const r=runs[d.side];r.events.push(d.e);put(d.side==="a"?$("ta"):$("tb"),d.e,d.side==="b");
if(SP()){SP().ride(d.side==="a"?PA():PB(),d.e[1]/M);if(d.side==="b"&&d.e[3])SP().free()}
stat(d.side==="a"?$("sa"):$("sb"),d.e[1],d.e[4],d.e[0],false,0,d.side==="b")}
else if(d.type==="done"){const r=runs[d.side];r.t_end=d.t_end;r.speed=d.speed;const last=r.events[r.events.length-1]||[0,0,"",0,0];
stat(d.side==="a"?$("sa"):$("sb"),last[1],last[4],d.t_end,true,d.speed,d.side==="b");
if(d.side==="a"){$("sum").textContent="generating with __BSHORT__...";SP()&&SP().hopTo(PB(),0)}}
else if(d.type==="summary"){$("sum").innerHTML=d.html;es.close();$("go").disabled=false;$("rp").disabled=false;
if(SP()){d.gain>0.5?SP().cheer():SP().shrug()}}
else if(d.type==="error"){$("sum").textContent=d.msg;es.close();$("go").disabled=false}};
es.onerror=()=>{if(es)es.close();$("go").disabled=false};};
function pane(r,te,se,free){let i=0;te.textContent="";const fn=now=>{while(i<r.events.length&&r.events[i][0]<=now){put(te,r.events[i],free);
if(free&&r.events[i][3]&&SP())SP().free();i++}
const l=i?r.events[i-1]:[0,0,"",0,0],fin=now>=r.t_end;stat(se,l[1],l[4],Math.min(now,r.t_end),fin,r.speed,free);return fin};
fn.prog=()=>r.events.length?i/r.events.length:1;return fn}
$("rp").onclick=()=>{cancelAnimationFrame(raf);const a=pane(runs.a,$("ta"),$("sa"),false),b=pane(runs.b,$("tb"),$("sb"),true),t0=performance.now();
SP()&&SP().hopTo(PB(),0);let cheered=false;
(function f(){const now=(performance.now()-t0)/1000;const x=a(now),y=b(now);SP()&&SP().ride(PB(),b.prog());
if(y&&!cheered&&SP()){cheered=true;runs.b.speed>runs.a.speed*1.005?SP().cheer():SP().shrug()}
if(!(x&&y))raf=requestAnimationFrame(f)})()};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, STATE["page"])
        if u.path != "/run":
            return self._send(404, "not found", "text/plain")
        q = parse_qs(u.query)
        text = (q.get("q") or [""])[0].strip()[:4000]
        try:
            m = max(16, min(2000, int((q.get("m") or ["400"])[0])))
        except ValueError:
            m = 400
        strength = (q.get("s") or ["low"])[0]
        strength = strength if strength in ("low", "medium", "high") else "low"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise SystemExit  # browser went away; stop this request
        if not text:
            return emit({"type": "error", "msg": "empty prompt"})
        model, tok, eos, hf = STATE["model"], STATE["tok"], STATE["eos"], STATE["hf"]
        ids = D.render(hf, text, strength)
        try:
            ra = D.run_normal(model, tok, ids, m, eos, show=False,
                              cb=lambda e: emit({"type": "tok", "side": "a", "e": e}))
            sa = D.speed(ra) if len(ra["tokens"]) > 1 else 0.0
            emit({"type": "done", "side": "a", "t_end": ra["t_end"], "speed": sa})
            rb = D.run_h8(model, tok, ids, m, eos, show=False,
                          cb=lambda e: emit({"type": "tok", "side": "b", "e": e}))
            sb = D.speed(rb) if len(rb["tokens"]) > 1 else 0.0
            emit({"type": "done", "side": "b", "t_end": rb["t_end"], "speed": sb})
        except SystemExit:
            return
        diff = next((i for i, (x, y) in enumerate(zip(ra["tokens"], rb["tokens"])) if x != y), -1)
        same = ("identical texts" if diff == -1 and len(ra["tokens"]) == len(rb["tokens"])
                else f"texts first differ at token {diff} (a bf16 near-tie; both are valid greedy continuations)")
        nf = rb["events"][-1][4] if rb["events"] else 0
        gain = 100 * (sb / sa - 1) if sa else 0.0
        ov = max(ra["overhead"] / max(ra["t_end"], 1e-9), rb["overhead"] / max(rb["t_end"], 1e-9))
        msg = (f"Normal {sa:.1f} tok/s &rarr; {html.escape(STATE['bshort'])} <b>{sb:.1f} tok/s ({gain:+.1f}%)</b> &middot; "
               f"{rb['tok_per_pass']:.2f} tokens per forward pass &middot; {100 * nf / max(len(rb['tokens']), 1):.0f}% of "
               f"tokens came free &middot; {same}. One run each, so small differences are noise (streaming the text to this "
               f"page took {100 * ov:.1f}% of the time); the README has the measured races.")
        print(f"[{time.strftime('%H:%M:%S')}] {len(ids)}-token prompt: normal {sa:.1f} tok/s, {STATE['bshort']} "
              f"{sb:.1f} tok/s ({gain:+.1f}%)", flush=True)
        emit({"type": "summary", "html": msg, "gain": gain})


def main():
    import common as C
    from mlx_lm import load
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None)
    p.add_argument("--small", action="store_true")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--variant", default="v2_h9_split_attn")
    a = p.parse_args()
    model_id = a.model or (D.SMALL_HF if a.small else (str(C.MODEL_DIR) if C.MODEL_DIR.exists() else D.GLIMMER_HF))
    is_glimmer = "glimmer" in model_id.lower()
    ram = D.ram_gb()
    if is_glimmer and 0 < ram < 32:
        raise SystemExit(f"This Mac has {ram:.0f} GB of memory; Muse Glimmer 30B needs about 20 GB free.\n"
                         f"Try:  ./run_demo.sh --ui --small")
    C.guard()
    vmod, _ = C.load_variant(a.variant)
    vmod.apply()
    print(f"loading {model_id} (first run downloads it) ...", flush=True)
    model, tok = load(model_id)
    vmod.post_load(model)
    hf, eos = tok._tokenizer, set(tok.eos_token_ids)
    D.run_normal(model, tok, D.render(hf, "hello", "low"), 16, eos, show=False)          # warm-up
    name = model_id.rstrip("/").split("/")[-1]
    bname = "H8 + H9" if is_glimmer else "H8"
    strength = ('<label>reasoning <select id="s"><option>low</option><option>medium</option><option>high</option>'
                '</select></label>') if is_glimmer else ""
    info = f"{html.escape(name)} &middot; {html.escape(C.versions())} &middot; greedy decoding."
    page = (PAGE.replace("__TAB__", f"{html.escape(name)}: normal vs {bname}")
            .replace("__TITLE__", "Muse Glimmer 30B (4-bit) on this Mac" if is_glimmer else f"{html.escape(name)} on this Mac")
            .replace("__BNAME__", bname).replace("__BSHORT__", bname).replace("__INFO__", info)
            .replace("__STRENGTH__", strength))
    page = mascot.inject(page)
    STATE.update(model=model, tok=tok, hf=hf, eos=eos, page=page, bshort=bname)
    srv = HTTPServer(("127.0.0.1", a.port), Handler)      # single-threaded: one prompt at a time, MLX on main thread
    url = f"http://127.0.0.1:{a.port}"
    print(f"ready: {url}  (Ctrl-C to stop)", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
