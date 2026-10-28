# glimmer-m5-selftune — Muse Glimmer tunes its own inference code on an M5 Max

Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026), Indie track, solo. Problem statement: **Push the Frontier**.

**What this is.** Muse Glimmer 30B (Meta's open-weight model, 4-bit, MLX) runs on a Mac Studio and is asked to make
*its own* text generation faster without changing what it outputs. Every change it proposes passes through a
**referee** built for this project: a safety gate, a bit-exact correctness check against a golden reference, and
interleaved, replicated A/B timing races. Glimmer then gets its own measurements back and tries again (closed loop).

Everything in this repository was built during the hackathon (first commit: Oct 27, 2026).
**Pre-existing (before kickoff):** downloading the model, installing MLX, and the four test-prompt text files used by
the bench (repeated technical paragraphs, kept out of git). The stock model was re-measured inside this repo during
the event and is labelled `v0_original` everywhere.

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

Each Glimmer row was raced 2–3 times (12–18 paired runs per case, all in the same direction).

**What we learned about the model as an engineer:** unguided at low reasoning it independently rediscovered our first
human idea (H1); at high reasoning it over-thought and hit the 12k-token cap; its crashes came from treating MLX like
PyTorch and from forgetting that 4-bit weights are packed `uint32`; in the closed loop it used a control experiment to
locate where rounding differences came from, and it under-estimated memory by forgetting the 52 layers — which the
referee caught.

**Next target found by measurement:** MLX's 4-bit matmul reads weights at 436 GB/s (95 % of spec) for 1 token but is
2.3–4.9× slower for 5–15 tokens at once (`bench/qmm_probe.py`), which is why prompt-lookup decoding does not pay off on
stock MLX (`bench/lookup_sim.py`, `bench/verify_cost.py`).

## Repository map

- `bench/` — the referee: `run_one.py` (golden / check / speed), `ab.py` (interleaved races), `compare.py` (verdicts),
  `safety.py` (static gate), `loop.py` + `propose.py` (Glimmer proposes, referee judges, results fed back),
  `variants/` (every candidate, human and Glimmer), probes for the kernel work.
- `docs/human_audit.md` — our own optimization ideas, committed before Glimmer saw the code.
- `runs/` — every race (`bench/bench.csv`), correctness report, Glimmer conversation and raw output.

## Reproduce

```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# put the model in models/Muse-Glimmer-30B-4bit and the test prompts in prompts/
python bench/run_one.py --variant v0_original --mode golden
python bench/run_one.py --variant g_guided_medium_10281717_r1 --mode check
python bench/ab.py --a v0_original --b g_guided_medium_10281717_r1 --pairs 6
```

## Credits and licences

Code: Apache-2.0 (this repo). Built by Bignesh Das with help from an AI coding assistant (Claude, Anthropic) for the
bench tooling; every `g_*` variant was written by Muse Glimmer and is kept verbatim. `runs/propose/*/conv.json` contain
the source of `mlx_lm/models/muse_glimmer.py` (Copyright © 2026 Apple Inc., MIT License) as part of Glimmer's prompt.
Muse Glimmer is distributed by Meta under Apache-2.0.
