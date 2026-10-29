# glimmer-m5-selftune — Muse Glimmer tunes its own inference code on an M5 Max

Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Indie track, solo. Problem statement: **Push the Frontier**.

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

Each Glimmer row was raced 2–3 times (12–18 paired runs per case, all in the same direction). H8 was raced 3 times
(18 paired runs per case, 18/18 faster; medians +10.47 to +10.53 % and +11.57 to +11.59 %).

**What we learned about the model as an engineer:** unguided at low reasoning it independently rediscovered our first
human idea (H1); at high reasoning it over-thought and hit the 12k-token cap; its crashes came from treating MLX like
PyTorch and from forgetting that 4-bit weights are packed `uint32`; in the closed loop it used a control experiment to
locate where rounding differences came from, and it under-estimated memory by forgetting the 52 layers — which the
referee caught.

**Prompt-lookup decoding (H8).** Glimmer often repeats what it already wrote (code, names, phrases). H8 guesses the
next 2 tokens from the earlier text and checks them in one forward pass, keeping the ones the model agrees with.
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
near-tied tokens: we measured 1 flip (2 bf16 steps) in 1,536 tokens; MLX's own prompt and decode paths disagree by up to
1 step. A float32 difference on the tiny model was traced to MLX's default TF32 for multi-row matmuls
(`MLX_ENABLE_TF32=1`, `bench/h8_logit_diff.py`).

*Scope.* "Self-tuning prompt" is one recorded Glimmer prompt (3,531 tokens); the gain depends on how much the output
repeats itself (simulation over all recorded runs: ≈1.3×).

*What did not work:* a hand-written Metal kernel for 9-token matmuls. MLX's own small-batch kernel re-reads the weights
every 5 tokens and is limited by arithmetic beyond that, so a simple kernel could not reach the target; our first kernel
referee also timed independent calls that the GPU overlaps, which overstated a padding trick ~7× (`bench/kernel_bench.py`).

## Repository map

- `bench/` — the referee: `run_one.py` (golden / check / speed), `ab.py` (interleaved races), `compare.py` (verdicts),
  `safety.py` (static gate), `loop.py` + `propose.py` (Glimmer proposes, referee judges, results fed back),
  `variants/` (every candidate, human and Glimmer), probes for the kernel work, `lookup_decode.py` (H8 decoder),
  `lookup_policy.py`, `verify_cost.py`, `h8_check.py`, `test_lookup_decode.py` (H8 analysis and correctness).
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
```

## Credits and licences

Code: Apache-2.0 (this repo). Built by Bignesh Das with help from an AI coding assistant (Claude, Anthropic) for the
bench tooling; every `g_*` variant was written by Muse Glimmer and is kept verbatim. `runs/propose/*/conv.json` contain
the source of `mlx_lm/models/muse_glimmer.py` (Copyright © 2026 Apple Inc., MIT License) as part of Glimmer's prompt.
Muse Glimmer is distributed by Meta under Apache-2.0.
