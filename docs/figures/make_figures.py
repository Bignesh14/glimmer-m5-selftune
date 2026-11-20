# Rebuilds the README charts from the measured results in runs/h8/. Run from the repo root:
#   python docs/figures/make_figures.py
# Needs matplotlib and numpy; uses the Inter font if installed, otherwise the default font.
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "Inter" in f: fm.fontManager.addfont(f)
plt.rcParams["font.family"] = "Inter"
D = "runs/h8/"   # run from the repo root: python docs/figures/make_figures.py

def sweep(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) >= 9 and p[-5].endswith("%") and p[-8].isdigit():
            rows.append((" ".join(p[:-8]), int(p[-8]), float(p[-7]), float(p[-6]), float(p[-5].rstrip("%"))))
    return rows
with_split = sweep(D + "sweep_h9_30oct.txt"); lookup_only = sweep(D + "sweep_30oct.txt"); gen = sweep(D + "sweep_general_01nov.txt")
cost = {}
for name in ["verify_cost_ctx13759_30oct.txt", "verify_cost_ctx13759_h9_30oct.txt"]:
    cost[name] = [float(x) for x in re.findall(r"= ([\d.]+)x a 1-token", open(D + name).read())]
spec = []
for line in open(D + "specbench_toks_summary.txt"):
    # the log labels our decoder "H8+H9" (its internal run name)
    m = re.match(r"\s+(\w+)\s+normal\s+([\d.]+) tok/s\s+H8\+H9\s+([\d.]+) tok/s\s+([\d.]+)x", line)
    if m: spec.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))))
spec = spec[:7]   # round 1

THEMES = {
 "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", crit="#d03b3b", inside="#ffffff"),
 "dark":  dict(bg="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926", crit="#d03b3b", inside="#ffffff"),
}
def base(t, w=8, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    fig.patch.set_facecolor(t["bg"]); ax.set_facecolor(t["bg"])
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    return fig, ax
def title(fig, t, main, sub):
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, main, fontsize=13, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.02, 1 - 0.40 / h, sub, fontsize=9.5, color=t["ink2"], va="top")
def save(fig, name, mode):
    fig.savefig(f"docs/figures/{name}-{mode}.png", facecolor=fig.get_facecolor()); plt.close(fig)

SHORT = {"code task (decode case)": "Code task (benchmark prompt)",
 "Write a Python class for a 2D vector wit": "Python 2D-vector class + tests",
 "Refactor this into a function and add ty": "Refactor into a typed function",
 "Write a JSON schema for a library book w": "JSON schema for a library book",
 "Write a SQL query that returns the top 5": "SQL query: top 5 customers",
 "Explain how a hash map works to a beginn": "Explain a hash map",
 "Write a short story about a lighthouse k": "Short story",
 "Write a polite email to a professor aski": "Polite email",
 "A train travels 120 km in 1.5 hours and": "Maths word problem"}
def short(n):
    for k, v in SHORT.items():
        if n.startswith(k[:30]): return v
    return n

for mode, t in THEMES.items():
    # 1. everyday prompts
    rows = sorted(gen, key=lambda r: r[4])
    fig, ax = base(t, 8, 4.4)
    y = range(len(rows)); vals = [r[4] for r in rows]
    ax.barh(list(y), vals, height=0.62, color=t["s2"])
    for i, v in enumerate(vals): ax.text(v + 0.4, i, f"+{v:.1f}%", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(list(y)); ax.set_yticklabels([short(r[0]) for r in rows], color=t["ink2"], fontsize=9.5)
    ax.set_xlim(0, 25); ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"+{v:.0f}%"))
    title(fig, t, "Muse Glimmer 30B writes up to 22% faster on everyday prompts",
          "Prompt-lookup decoding with split attention vs normal decoding, Mac Studio M5 Max, 800 tokens each.")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.80, bottom=0.09); save(fig, "everyday-prompts", mode)

    # 2. gain vs context: prompt lookup alone vs with split attention (19 self-tuning prompts + code task)
    fig, ax = base(t, 8, 4.4)
    for rows, col, lab in [(lookup_only, t["s1"], "Prompt lookup alone"), (with_split, t["s2"], "Prompt lookup +\nsplit attention")]:
        rs = sorted(rows, key=lambda r: r[1])
        xs = [r[1] / 1000 for r in rs]; ys = [r[4] for r in rs]
        ax.scatter(xs, ys, s=34, color=col, edgecolor=t["bg"], linewidth=1.5, zorder=3, label=lab)
        import numpy as np
        k = np.polyfit(xs, ys, 1); xx = np.linspace(0, 14, 50)
        ax.plot(xx, np.polyval(k, xx), color=col, lw=2, alpha=.9, zorder=2)
        ax.text(14.2, np.polyval(k, 14), lab, color=t["ink"], fontsize=9.5, va="center", fontweight="bold")
    ax.set_xlim(0, 17.6); ax.set_ylim(0, 25); ax.set_xticks(range(0, 16, 2))
    ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"+{v:.0f}%"))
    ax.set_xlabel("Context length (thousand tokens)", color=t["muted"], fontsize=9)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=t["ink2"])
    title(fig, t, "Split attention keeps the speed-up alive on long contexts",
          "Gain on 20 recorded prompts (lines: straight-line fit). Prompt lookup alone fades to +1% at 14K tokens.")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.13); save(fig, "context-gain", mode)

    # 3. pass cost at 13,759 tokens
    import numpy as np
    stock = cost["verify_cost_ctx13759_30oct.txt"]; split = cost["verify_cost_ctx13759_h9_30oct.txt"]
    fig, ax = base(t, 8, 4.2); x = np.arange(4); w = 0.36
    ax.bar(x - w/2 - .01, stock, w, color=t["s1"], label="Stock MLX")
    ax.bar(x + w/2 + .01, split, w, color=t["s2"], label="With split attention")
    for i in range(4):
        ax.text(i - w/2 - .01, stock[i] + .03, f"{stock[i]:.2f}×", ha="center", fontsize=9, color=t["ink"])
        ax.text(i + w/2 + .01, split[i] + .03, f"{split[i]:.2f}×", ha="center", fontsize=9, color=t["ink"])
    ax.set_xticks(x); ax.set_xticklabels(["1 token", "2 tokens", "3 tokens\n(what prompt lookup checks)", "4 tokens"], color=t["ink2"], fontsize=9.5)
    ax.set_ylim(0, 1.75); ax.yaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}×"))
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=t["ink2"])
    title(fig, t, "Split attention: checking 3 tokens costs 1.21× a normal step, not 1.44×",
          "Time of one forward pass vs a 1-token pass, at 13,759 tokens of context (median of repeated runs).")
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.15); save(fig, "pass-cost", mode)

    # 4. Spec-Bench
    names = {"mt_bench": "Multi-turn chat", "translation": "Translation", "summarization": "Summarization", "qa": "Question answering",
             "math_reasoning": "Maths (GSM8K)", "rag": "Retrieval (RAG)", "overall": "Overall"}
    rows = [r for r in spec if r[0] != "overall"]; rows.sort(key=lambda r: r[3]); rows.append([r for r in spec if r[0] == "overall"][0])
    fig, ax = base(t, 8, 4.2)
    y = list(range(len(rows)))
    cols = [t["s2"] if r[0] != "overall" else t["ink2"] for r in rows]
    ax.barh(y, [r[3] - 1 for r in rows], left=1, height=0.6, color=cols)
    for i, r in enumerate(rows): ax.text(r[3] + .006, i, f"{r[3]:.2f}×", va="center", fontsize=9, color=t["ink"], fontweight="bold" if r[0]=="overall" else "normal")
    ax.set_yticks(y); ax.set_yticklabels([names[r[0]] for r in rows], color=t["ink2"], fontsize=9.5)
    ax.set_xlim(1, 1.5); ax.xaxis.grid(True, color=t["grid"], lw=0.8); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, p: f"{v:.1f}×"))
    title(fig, t, "Spec-Bench: every task faster, 1.15× overall",
          "60 prompts (10 per task), our decoder vs normal decoding, prompt processing included. Two rounds agree within 0.001.")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.80, bottom=0.09); save(fig, "spec-bench", mode)

    # 5. Glimmer's 16 attempts at the long-context fix
    fig, ax = base(t, 8, 2.6); ax.spines["bottom"].set_visible(False); ax.set_xticks([]); ax.set_yticks([])
    segs = [(10, "10 fast but wrong", t["crit"]), (5, "5 crashed", t["muted"]), (1, "1 ran out of tokens", t["axis"])]
    left = 0
    for n, lab, c in segs:
        ax.barh(0, n - .06, left=left + .03, height=0.55, color=c)
        if n > 2: ax.text(left + n / 2, 0, lab, ha="center", va="center", fontsize=10 if n > 2 else 9,
                color="#ffffff" if c != t["axis"] else t["ink"], fontweight="bold")
        left += n
    ax.text(16.15, 0, "0 correct", va="center", fontsize=10.5, color=t["ink"], fontweight="bold")
    ax.text(15.5, -0.48, "1 ran out of tokens", ha="right", fontsize=8.5, color=t["ink2"])
    ax.set_xlim(0, 18.2); ax.set_ylim(-0.7, 0.45)
    title(fig, t, "Glimmer tried to fix the long-context slowdown itself: 16 attempts, 0 correct",
          "10 versions were as fast as our split attention but let the model peek at the guesses it was checking. All rejected.")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.66, bottom=0.05); save(fig, "glimmer-attempts", mode)
print("charts written to docs/figures/")
