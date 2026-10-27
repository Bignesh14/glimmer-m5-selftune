# Reference audit of `muse_glimmer.py` — written BEFORE Glimmer saw the code

**Date:** 27 Oct 2026. **Authors:** Bignesh Das, with an AI assistant (Claude, Anthropic).
**Code audited:** `mlx_lm/models/muse_glimmer.py` and the decode loop in `mlx_lm/generate.py`,
mlx-lm 0.32.2 / mlx 0.32.4 (exact versions in `requirements-baseline.txt`).

This list is committed before any Glimmer run so that its timestamp shows which ideas
came from us and which came from Glimmer. It is the "human" arm of the experiment:
human audit vs Glimmer unguided vs Glimmer guided, all judged by the same bench.

## Context (measured, see README)

- Decode ≈ 26.6 tok/s (mlx_lm, 70-token prompt) ≈ 37.6 ms per token; GPU ~99.8 % busy.
- Decode is memory-bandwidth-bound: ~87 % of the 460 GB/s spec (our estimate) → ~10–15 % headroom.
- Prefill ≈ 800 tok/s, compute-bound, ~20–27 % headroom (our estimate).
- Every token runs `DecoderLayer` 52 times (39 sliding-window layers, 13 global layers).

## Candidate changes, safest first

| # | Change | Why it might help | Expected effect on output |
|---|---|---|---|
| H1 | Compute `1.0 + weight` once at load instead of inside every `CenteredRMSNorm` call | removes 4 × 52 = 208 small GPU ops per token | bit-exact |
| H2 | Fold Q's `* qk_scale_factor` into the RMSNorm weight | one fewer pass over Q per layer | tiny rounding differences possible |
| H3 | Merge `q_proj, k_proj, v_proj, gate_proj` into one quantized matmul | input read once; 3 fewer launches per layer | tiny differences possible |
| H4 | Merge MLP `gate_proj + up_proj` into one matmul | one fewer launch; larger matmul | tiny differences possible; ~+8 GB if both layouts kept |
| H5 | `mx.compile` the elementwise chains (`silu(g) * u`; `out * sigmoid(gate)`) | fuses several passes into one | tiny differences possible |
| H6 | Tune `prefill_step_size` (default 2048) | prefill speed | expected identical |
| H7 | Quantized KV cache (`kv_bits`) | long contexts (decode 26.6 → 24.9 tok/s at 32K) | changes outputs — quality trade-off |
| H8 | Prompt-lookup / speculative decoding | potentially large on copy-heavy tasks | identical under greedy; larger build |
| H9 | Fused add + RMSNorm, fused gating as custom Metal kernels | kernel level (stage 2) | tiny differences possible |

**Honest ceiling:** H1–H6 remove overheads; none reduces the ~15 GB of weights read per
decoded token, so we expect low single-digit percent from them combined. H8 is the only
large lever.

## Acceptance rule (same for every arm)

A change counts only if (1) the correctness check passes — bit-exact tokens for changes
marked bit-exact; otherwise logits within tolerance over the first 50 steps and top-1
agreement — and (2) it is faster in interleaved A/B runs against the unmodified code,
beyond the measured run-to-run noise.
