# ADR 0014 — Training Qwen3-8B on 8GB

- **Status:** Accepted
- **Date:** 2026-09-30
- **Amends:** [ADR 0013](0013-target-hardware-rtx-3070-ti.md) — its finding that training
  does not run on this card held only for the original configuration.

## Context

ADR 0013 recorded that QLoRA training of Qwen3-8B fails at model load on the RTX 3070 Ti
8GB. The checkpoint that failed was Unsloth's *dynamic* 4-bit quantisation (6.97GB),
which `unsloth/Qwen3-8B` with `load_in_4bit` resolves to. Follow-up smoke runs (160 rows,
10 optimizer steps) measured what it takes to fit:

| Change | Effect |
|---|---|
| Standard bnb 4-bit checkpoint, `unsloth/Qwen3-8B-bnb-4bit` (5.66GB) | 5.73GB reserved after load, instead of failing |
| Batch 1 × grad-accum 16, `max_seq_length` 1536 | training peak **7.06-7.10GB** of 7.65GB visible |
| In-training evaluation on | **OOM** at the first eval pass, twice: it materialises full logits (vocab 151,936) in fp32 and asked for 636MB with ~600MB free |
| In-training evaluation off | run completes, adapter saved |

Step time was ~25 s at effective batch 16 (ADR 0012 measured 39.2 s on the 3060), which
prices the 10,000-row run at roughly 4.4 hours.

A second, unrelated blocker surfaced: triton JIT-compiles a C module at the first step
and needs `Python.h`, which the distro Python lacks without its `-dev` package.

## Decision

1. Train from `model.train_id: unsloth/Qwen3-8B-bnb-4bit`, loaded with
   `use_exact_model_name=True` so Unsloth does not remap it back to the dynamic
   checkpoint. `model.base_id` stays `unsloth/Qwen3-8B` and remains the source of the
   tokenizer and of the fp16 base export.
2. `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 16` (effective batch
   unchanged at 16), `max_seq_length: 1536`.
3. `train.eval_strategy: "no"`. The knob stays in the config so a larger card can turn
   evaluation back on.
4. `[tool.uv] python-preference = "only-managed"`, so the venv is built on a uv-managed
   CPython that ships its headers. No system package is required.

## Consequences

- The whole pipeline, including `make train`, runs on the 8GB card.
- The margin is ~0.5GB. A GPU-accelerated browser opened mid-run can OOM it; train with
  the desktop light or from a TTY.
- No validation loss during training. Overfitting or a bad learning rate only shows up in
  `make eval`. With one epoch over 10k unique rows this is a tolerable blind spot.
- The peak was measured on the smoke slice, whose longest example is 1458 tokens. Rows up
  to 1536 tokens in the full split will push it slightly higher.
- `max_seq_length` 1536 also tightens the filter in `01_prepare_data.py`; ADR 0012
  measured p99 = 1424 tokens, so roughly 1% of candidate rows or fewer are dropped.
- The adapter is trained against standard 4-bit weights rather than the dynamic ones, so
  results are not strictly comparable to `docs/RESULTS.md`, which also used a 2048 limit
  and batch 2 × 8.
- Invariant 5 is unaffected: tuned and base GGUFs still derive from the same fp16 base
  and the same convert and quantise steps.

## Alternatives considered

- **A smaller base model (~4B).** Comfortable headroom and room for evaluation, but it
  supersedes ADR 0003 and discards the 8B comparison. Not needed once the 8B fit.
- **Keep evaluation with a shorter eval sequence length.** Would need a second tokenised
  view of the validation set; not worth the complexity for a PoC.
- **Stop the GNOME session.** Reclaims only ~0.3-0.4GB here, not enough for the eval pass.
