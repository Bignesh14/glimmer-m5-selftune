# glimmer-m5-selftune — Muse Glimmer tunes its own inference code on an M5 Max

Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Indie track, solo. Problem statement: **Push the Frontier**.

**Result.** On a Mac Studio M5 Max, Meta's Muse Glimmer 30B (4-bit, MLX) now writes **+17 % faster** on a coding task
and **+5 % to +21 % faster (median +12.8 %)** on its own 19 recorded self-tuning prompts, with **no extra memory**.
In exact (fp32) arithmetic the text is identical to normal decoding; in bf16 a near-tied token can occasionally flip
(details below). 21 replicated races, 252 paired timing runs: every one faster. Two pieces: **H8**, prompt-lookup decoding made exact
for Glimmer's sliding-window cache, and **H9**, a fix that keeps MLX's fast attention kernel in use for H8's 3-token
check passes, found by this project's referee and traced to MLX's source. Glimmer's own best change: +0.17 %, bit-exact;
asked to fix the long-context problem itself, it found the fast version in 10 of 16 attempts but never a correct one, and
the referee rejected every one (details below).

## Try it on your Mac (one command)

Needs an Apple Silicon Mac (M1 or newer), Python 3.10+ (`python3 --version`), internet for the first run, and Low Power
Mode off. In Terminal:

```
git clone https://github.com/Bignesh14/glimmer-m5-selftune.git && cd glimmer-m5-selftune
./run_demo.sh --small                 # any Apple Silicon Mac with 8 GB+: Meta Llama 3.2 3B (4-bit, ~2 GB download)
./run_demo.sh                         # Macs with 32 GB+: Muse Glimmer 30B (4-bit, ~19 GB download)
```

The first run creates a private Python environment (`.venv`), installs the pinned packages and downloads the model
(about 15–20 minutes on a slow connection; later runs start in seconds). The demo gives the model the same prompt twice,
with normal decoding (mlx-lm) and with this project's decoder, prints both answers and their speeds, and opens a
side-by-side page that replays both runs with their real measured timings.

**Your own prompt:** `./run_demo.sh --small --prompt "Write a Python class for a 2D vector with unit tests."`
(also `--max-tokens 400`, and `--strength low|medium|high` for Glimmer's reasoning effort; `--model <Hugging Face id or
folder>` for any other mlx-lm model). The decoder helps most when the answer repeats itself: code, edits, refactors,
structured text; on prose it can be **slower**. Measured with the demo on the C-transpose prompt: Mac Studio M5 Max,
Glimmer 30B **+15.9 %** (identical text) and Llama 3.2 3B **+11.6 %** (identical text). On a MacBook Air M3 with
Llama 3.2 3B (single demo runs, not races; the normal speed itself varied 40–45 tok/s between runs, so treat these as
rough):

| Prompt (Air M3, Llama 3.2 3B) | Normal | H8 | Change |
|---|---|---|---|
| Python 2D-vector class + unit tests | 40.5 tok/s | 55.6 tok/s | **+37 %** |
| C function: transpose a matrix in place | 45.0 | 46.4 | +3 % |
| JSON schema for a library book + example | 45.3 | 46.4 | +2 % |
| Explain a hash map to a beginner | 44.2 | 44.0 | −1 % |
| Refactor a loop into a typed function | 42.3 | 39.9 | −6 % |
| Short story about a lighthouse keeper | 43.0 | 39.7 | −8 % |

So on a small Mac the decoder pays off when the answer repeats itself a lot (unit tests, boilerplate) and costs a few
percent on prose, because a wrong guess still costs a slightly bigger forward pass. Switching guessing off when it is
not paying (adaptive drafting) is the obvious next step and is not done here. Every number in the Results section is
Glimmer on the M5 Max. H9 applies only to Glimmer (its attention layout is what triggers MLX's slow path).

**What this is.** Muse Glimmer 30B (Meta's open-weight model, 4-bit, MLX) runs on a Mac Studio and is asked to make
*its own* text generation faster without changing what it outputs. Every change it proposes passes through a
**referee** built for this project: a safety gate, a bit-exact correctness check against a golden reference, and
interleaved, replicated A/B timing races. Glimmer then gets its own measurements back and tries again (closed loop).

Everything in this repository was built during the hackathon (first commit: Oct 27, 2026).
**Pre-existing (before kickoff):** downloading the model, installing MLX, the four test-prompt text files used by
the bench (repeated technical paragraphs, kept out of git), and our own earlier speed measurements of the stock model
(not in this repo). Their headline figures (≈26.6 tok/s decode, ≈87 % of memory bandwidth, ≈810 tok/s prompt
processing) are quoted as context in `docs/human_audit.md` and in the guided prompt given to Glimmer. The stock model
was re-measured inside this repo during the event and is labelled `v0_original` everywhere.

## Setup and provenance

- Hardware: Mac Studio, base M5 Max — 18-core CPU, 32-core GPU, 36 GB unified memory, 460 GB/s *(Apple-official specs)*.
- Model: `mlx-community/Muse-Glimmer-30B-4bit` (4-bit affine quantization, group size 64). Weights not in this repo.
- Software: mlx 0.32.4, mlx-lm 0.32.2 (exact environment in `requirements.txt`). Every bench run prints the variant,
  its file hash, the MLX versions, Low Power Mode state and the weights' byte count; results are in `runs/`.

## Results (all measured during the event)

**Stock model (`v0_original`), Oct 27:** decode 26.60 tok/s (800 tokens), prompt processing 809 tok/s (1K) and
814 tok/s (4K), peak memory 15.8–16.6 GB. Two A/A races (original vs itself): differences ≤ 0.05 %, run-to-run CV
0.02–0.05 %. Win rule (fixed before any real test): faster in 6/6 paired runs (sign test p < 0.05) by more than 0.1 %,
replicated in a second race.

| Version | Who | Correct | Decode | Prompt 1K | Prompt 4K | Extra memory |
|---|---|---|---|---|---|---|
| H1: precompute `1 + weight` in 208 norms | human audit | bit-exact | +0.08 % (17/18, below bar) | ±0 | ±0 | 0 |
| Fused q/k/v + gate/up (round 2) | Glimmer, guided | not bit-exact | +0.23 % | −0.42 % | +0.40 % | +9 GB |
| Fused MLP, originals freed (refine 1) | Glimmer | bit-exact | +0.17 % | −0.74 % | +0.32 % | +0.3–0.9 GB |
| Fused MLP for decode only (refine 2) | Glimmer | bit-exact | +0.17–0.20 % | ±0 | ±0 | +7.6 GB |
| **Fused MLP, one weight copy, prompts via slices (refine 3)** | **Glimmer** | **bit-exact** | **+0.17 %** | **+0.13 %** | **+0.10 %** | **+0.2–0.8 GB** |
| **H8: prompt-lookup decoding (K=2), exact rollback past the 2048-token window** | human idea and analysis; decoder written with Claude | exact in fp32 (see below) | **+10.5 %**; **+11.6 %** on a self-tuning prompt | unchanged (same prefill code) | unchanged | 0 |
| **H8 + H9: split 3-token attention so MLX's fused kernel is used** | human idea (H9); Glimmer's 16 attempts: none correct | normal decoding bit-exact; H8+H9 exact in fp32 | **+17.1 %** code task; **+5.4 % to +21.0 %** on 6 raced self-tuning prompts | unchanged | unchanged | 0 |

Each Glimmer row was raced 2–3 times (12–18 paired runs per case, all in the same direction). H8 was raced 3 times
(18 paired runs per case, 18/18 faster; medians +10.47 to +10.53 % and +11.57 to +11.59 %), and on two larger
prompts 3 times each (+10.1 % at 4.5K tokens, +9.5 % at 7.1K). H8+H9 vs normal decoding was raced twice on
each of 6 self-tuning prompts (6 pairs per race): the sweep's worst prompt +5.39 / +5.42 %, the 14K prompt +7.73 / +7.74 %,
the sweep's median prompt +12.74 / +12.70 %, the first prompt +13.60 / +13.55 %, the 7K prompt +16.12 / +16.09 %, the
sweep's best prompt +21.02 / +21.04 %; the code task inside the same races +17.04 to +17.12 %. In total: 21 races, 252
paired runs, all faster, peak memory unchanged, run-to-run noise 0.01–0.05 %. (One earlier attempt at the last race
crashed at start-up in its final pair because of how the batch was launched; it produced no measurement and was re-run.)

**What we learned about the model as an engineer:** unguided at low reasoning it independently rediscovered our first
human idea (H1); at high reasoning it over-thought and hit the 12k-token cap; its crashes came from treating MLX like
PyTorch and from forgetting that 4-bit weights are packed `uint32`; in the closed loop it used a control experiment to
locate where rounding differences came from, and it under-estimated memory by forgetting the 52 layers — which the
referee caught.

**Prompt-lookup decoding (H8).** Glimmer often repeats what it already wrote (code, names, phrases). H8 guesses the
next 2 tokens from the earlier text and checks them in one forward pass, keeping the ones the model agrees with.
The technique itself is not ours: prompt-lookup decoding was introduced by Apoorv Saxena (2023,
github.com/apoorvumang/prompt-lookup-decoding) and is available in Hugging Face transformers
(`prompt_lookup_num_tokens`). What we built is the measured cost model that chose its settings for this GPU, the
exact cache rollback that makes it work with Glimmer's sliding-window layers in MLX, and the correctness evidence.
Our first estimate said this cannot pay off on stock MLX, because a 9-token pass costs 2.3× a 1-token pass (MLX's 4-bit
matmul reads weights at 436 GB/s for 1 token but is 2.3–4.9× slower for 5–15 tokens, `bench/qmm_probe.py`). That
estimate was wrong: it charged every pass as a 9-token pass. Re-scoring all 21 recorded Glimmer generations with the
*measured* cost of each pass length (`bench/verify_cost.py`, `bench/lookup_policy.py`) picked short drafts (K = 2,
estimated 1.31×; a 3-token pass costs only 1.17×). mlx-lm cannot roll back Glimmer's sliding-window cache once the
context passes 2048 tokens (its own speculative decoder refuses); ours does it exactly, by cutting the rejected tokens off
the end right after a multi-token pass (`bench/lookup_decode.py`).

*Correctness.* With float32 activations (and TF32 off), every token H8 produced on the real model equals what normal
greedy decoding picks: 0 differences in 1,024 tokens, 512 of them past the window (`bench/h8_check.py`). A tiny-model
test also proves a broken rollback is caught (`bench/test_lookup_decode.py`). In normal bf16, any batched check can flip
near-tied tokens: in 1,536 checked tokens, 5 differed from what normal decoding picks, 4 within one bf16 rounding step and
1 at two steps. After such a flip the text continues differently (in the sweep, after 14 to 657 tokens; in the 600-token
code-task demo the two texts were identical). MLX's own prompt and decode paths disagree with each other the same way
(up to 1 step). A float32 difference on the tiny model was traced to MLX's default TF32 for multi-row matmuls
(`MLX_ENABLE_TF32=1`, `bench/h8_logit_diff.py`).

*Scope.* Sweep over all 19 recorded self-tuning prompts whose recorded answer is at least 800 tokens
(`bench/h8_sweep.py`): H8 alone median +9.4 % (+1.3 to +22.8 %); H8+H9 median +12.8 % (+5.4 to +21.0 %). The gain depends
on how much the output repeats itself, and in bf16 also on which text is written after a near-tie.

**Why the gain shrank on long contexts, and the fix (H9).** The sweep showed H8's gain falling from ~11 % to ~1 % as the
context grew to 14K tokens, although its guesses were as good. Measured pass costs (`bench/verify_cost.py --context`): at
13,759 tokens a 3-token pass cost 1.44× a 1-token pass, a 2-token pass only 1.09×. MLX 0.32.4's fused attention kernel
for short queries requires query length × GQA factor ≤ 32; Glimmer's GQA factor is 16 (32 query heads, 2 key-value
heads), so H8's 3-token passes fell back to unfused attention. H9 (`bench/variants/v2_h9_split_attn.py`) computes those
passes' attention in chunks of at most 2 queries: 3-token pass 1.44× → 1.21× at 14K. Normal decoding stays bit-exact
(golden check); H8+H9 gives 0 wrong tokens on the 14K prompt in bf16 and is exact in fp32.

**Can Glimmer fix this itself? (`bench/loop.py --task h9`)** We gave Glimmer the measured pass costs and MLX's kernel
limit, but not our solution (H9 was committed to `docs/human_audit.md` first), and let the referee judge every attempt:
golden check, 3-token pass cost at 14K, then prompt-lookup correctness. 16 attempts (8 greedy with feedback, 8 sampled
independently at temperature 0.7): 10 found the speed-up (1.20–1.23×, as fast as H9) but were **incorrect**, 5 crashed,
1 ran out of tokens; **0 were correct**. In the three fast versions we read line by line (the first greedy round), queries
could attend to keys of later tokens in the same pass (the code ignored that MLX aligns a causal mask to the end of the
keys), i.e. the model could peek at the guesses it was checking: two of them "accepted" 93–95 % of guesses instead of the
normal 33–36 %. The other seven fast versions failed the same prompt-lookup correctness check. A speed-only reward
would have crowned all 10; the correctness check rejected them. That is the main lesson of this project: **an AI optimizing its own
code needs a referee that checks correctness, not just speed.**

**Next.** The referee is a reward function (task, gate, correctness, speed). Next step: use it to fine-tune Glimmer
with reinforcement learning (e.g. Fireworks' serverless training for Muse Glimmer 30B), so it learns to write correct
fast code instead of being caught; and propose the sliding-window rollback and the split attention upstream to mlx-lm.

*What did not work:* a hand-written Metal kernel for 9-token matmuls. MLX's own small-batch kernel re-reads the weights
every 5 tokens and is limited by arithmetic beyond that, so a simple kernel could not reach the target; our first kernel
referee also timed independent calls that the GPU overlaps, which overstated a padding trick ~7× (`bench/kernel_bench.py`).

## Repository map

- `bench/` — the referee: `run_one.py` (golden / check / speed), `ab.py` (interleaved races), `compare.py` (verdicts),
  `safety.py` (static gate), `loop.py` + `propose.py` (Glimmer proposes, referee judges, results fed back),
  `variants/` (every candidate, human and Glimmer), probes for the kernel work, `lookup_decode.py` (H8 decoder),
  `lookup_policy.py`, `verify_cost.py`, `h8_check.py`, `test_lookup_decode.py`, `h8_sweep.py`, `prompts_index.py`
  (H8/H9 analysis, correctness, sweep).
- `run_demo.sh`, `demo/demo.py` — the one-command side-by-side demo (normal vs H8+H9; `--small` for 8 GB Macs).
- `docs/human_audit.md` — our own optimization ideas, committed before Glimmer saw the code.
- `runs/` — every race (`bench/bench.csv`), correctness report, Glimmer conversation and raw output.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# put the model in models/Muse-Glimmer-30B-4bit and the test prompts in prompts/
python bench/run_one.py --variant v0_original --mode golden
python bench/run_one.py --variant g_guided_medium_10281717_r1 --mode check
python bench/ab.py --a v0_original --b g_guided_medium_10281717_r1 --pairs 6
python bench/test_lookup_decode.py                     # H8 logic, tiny model, seconds
python bench/h8_check.py                               # H8 on the real model vs normal decoding
python bench/ab.py --a v0_original --b v0_original@lookup --plan h8 --pairs 6
python bench/ab.py --a v0_original --b v2_h9_split_attn@lookup --plan h8 --pairs 6   # normal vs H8+H9
./run_demo.sh                                          # live side-by-side demo (or ./run_demo.sh --small)
python bench/loop.py --task h9 --arm guided --strength medium --attempts 4   # Glimmer's H9 task
```

## Credits and licences

Code: Apache-2.0 (this repo). Built by Bignesh Das with help from an AI coding assistant (Claude, Anthropic) for the
bench tooling; every `g_*` variant was written by Muse Glimmer and is kept verbatim. `runs/propose/*/conv.json` contain
the source of `mlx_lm/models/muse_glimmer.py` (Copyright © 2026 Apple Inc., MIT License) as part of Glimmer's prompt.
Muse Glimmer is distributed by Meta under Apache-2.0.
