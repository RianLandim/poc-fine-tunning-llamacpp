# Results — Qwen3-8B QLoRA persona fine-tune

The same pipeline has been run end to end on two GPUs. Run 2 is the current target of this
fork; run 1 is the original record and its numbers are unchanged.

| | Run 2 — RTX 3070 Ti 8GB | Run 1 — RTX 3060 12GB |
|---|---|---|
| Date | 2026-09-30 | 2026-09-13 |
| Training config | standard bnb 4-bit, batch 1 × 16, seq 1536, no in-training eval ([ADR 0014](adr/0014-training-qwen3-8b-on-8gb.md)) | dynamic 4-bit, batch 2 × 8, seq 2048, eval every 100 steps |
| Training time | **4h 26m** (25.6 s/step) | 7h 25m (42.7 s/step) |
| Peak VRAM | 7.10 GB of 7.65 GB | 9.07 GB of 11.63 GB |
| Format validity (tuned) | 0.990 | 1.000 |
| Perplexity base → tuned | 4.74 → **2.80** | 4.73 → **2.77** |
| Grounding recall (tuned / reference) | 0.872 / 0.860 | 0.903 / 0.867 |
| End-to-end wall clock | **≈ 5h 30m** | ≈ 9h |

**The 8GB card reproduces the result.** Format validity, perplexity and grounding land
within a few points of run 1, in 60% of the training time.

The two runs are close but not strictly comparable: the base checkpoint used for training
differs, and `max_seq_length` 1536 filters the data slightly differently, so the 200
test rows are not the same set (reference grounding 0.860 vs 0.867, reference length
3519 vs 3491 chars).

---

## Run 2 — RTX 3070 Ti 8GB

**Date:** 2026-09-30 · **Hardware:** single RTX 3070 Ti 8GB, 31GB RAM ·
**Config:** `configs/qwen3-8b-personas.yaml` as of ADR 0014

### Training

| | |
|---|---|
| Rows / steps | 10,000 / 625 (one epoch, effective batch 16 = 1 × 16 grad-accum) |
| Weights | `unsloth/Qwen3-8B-bnb-4bit` (standard 4-bit, 5.66GB) |
| Runtime | **4.43h** at 25.6 s/step |
| Peak VRAM | **7.10 GB** of 7.65 GB |
| Trainable params | 87,293,952 of 8,278,029,312 (1.05%) |
| Train loss | 1.716 (step 10) → 0.840 (step 620); run mean 0.9095 |

| Step | 10 | 90 | 170 | 250 | 330 | 410 | 490 | 570 | 620 |
|---|---|---|---|---|---|---|---|---|---|
| Train loss | 1.716 | 0.963 | 0.908 | 0.888 | 0.871 | 0.850 | 0.842 | 0.835 | 0.840 |

There is **no validation loss** for this run: an in-training eval pass materialises full
fp32 logits and OOMs the card (ADR 0014). Overfitting would only have shown up in the
final evaluation below; it did not.

### Process metrics

| Stage | Wall clock | Notes |
|---|---|---|
| Environment + llama.cpp CUDA build | ~25 min* | includes a local CUDA 12.8 toolkit via micromamba (ADR 0013) |
| Data preparation | ~2 min* | streamed 10,727 rows; 7 dropped for name, 19 over 1536 tokens |
| **Fine-tuning** | **4h 26m** | 625 steps at 25.6 s/step |
| GGUF export (tuned) | **6m 13s** | merge → f16 GGUF → Q4_K_M + Q8_0 |
| Evaluation | **29m 25s** | 200 rows × 2 models, sequential, plus perplexity |
| Report | < 1s | |
| **Total** | **≈ 5h 30m** | |

<sub>* approximate, download-bound.</sub>

| | Base | Fine-tuned |
|---|---|---|
| 200 generations | 666 s (3.3 s/row) | 1001 s (5.0 s/row) |
| Mean output | 2072 chars | 2998 chars |

### Base vs fine-tuned

200 held-out rows, Q4_K_M through the same quantisation lineage (same fp16 base, same
llama.cpp commit `bdeb855`), identical decoding (temp 0.7, top_p 0.8, top_k 20,
repeat_penalty 1.05, seed 3407), served sequentially.

| Metric | Base | Fine-tuned | |
|---|---|---|---|
| **Format validity** | 0.000 | **0.990** | 198/200 rows |
| **Perplexity** (held-out) | 4.7380 | **2.8042** | −41% |
| Think leak rate | 0.000 | 0.000 | ADR 0006 held |
| Portuguese rate | 1.000 | 1.000 | |
| Mean chars | 2072 | 2998 | reference: 3519 |
| Grounding recall | 1.000 | 0.872 | reference: 0.860 |

The two invalid rows had all six sections in the right order and then appended a
**seventh** section. No row was missing a section or had them out of order.

Grounding by field, scored with the same function against the ground-truth text:

| Field | Base | Fine-tuned | Ground truth |
|---|---|---|---|
| name | 1.000 | 1.000 | 1.000 |
| municipality | 1.000 | 0.995 | 1.000 |
| state | 1.000 | 0.855 | 0.830 |
| occupation | 1.000 | 0.890 | 0.890 |
| age | 1.000 | 0.620 | 0.580 |
| **recall** | **1.000** | **0.872** | **0.860** |

As in run 1, the fine-tune sits at or above the human-written reference on every field
except a single municipality miss; the base model's 1.000 comes from echoing the input
back as a form (see run 1's analysis).

---

## Run 1 — RTX 3060 12GB

**Date:** 2026-09-13 · **Hardware:** single RTX 3060 12GB · **Config:** `configs/qwen3-8b-personas.yaml`
at the time (dynamic 4-bit, batch 2 × 8, `max_seq_length` 2048)

### Training

| | |
|---|---|
| Rows / steps | 10,000 / 625 (exactly one epoch, ADR 0012) |
| Runtime | 7.42h at 41.6 s/step |
| Peak VRAM | 9.07 GB of 11.63 GB |
| Trainable params | 87,293,952 of 8,278,029,312 (1.05%) |
| Train loss | 1.719 → 0.9073 |

Eval loss fell monotonically at every checkpoint, with train and eval tracking
together — convergence without overfitting:

| Step | 200 | 300 | 400 | 500 | 600 | 625 |
|---|---|---|---|---|---|---|
| Eval loss | 0.9454 | 0.8900 | 0.8603 | 0.8417 | 0.8308 | **0.8267** |

### Process metrics

Every stage, measured end to end on one RTX 3060 12GB.

| Stage | Wall clock | Notes |
|---|---|---|
| Environment + llama.cpp CUDA build | ~25 min | download-bound; `sm_86` only |
| Data preparation | ~3 min | streamed 10,708 rows, 0.1% dropped |
| **Fine-tuning** | **7h 25m** | 625 steps at 42.7 s/step |
| GGUF export (per model) | **6m 26s** | merge → f16 GGUF → Q4_K_M + Q8_0 |
| Evaluation | **46m 19s** | 200 rows × 2 models, sequential |
| Report | < 1s | |
| **Total** | **≈ 9h** | |

Evaluation breakdown:

| | Base | Fine-tuned |
|---|---|---|
| 200 generations | 1050 s (5.3 s/row) | 1610 s (8.1 s/row) |
| Mean output | 2093 chars | 3072 chars |

The tuned model is ~50% slower per row because it produces ~50% more text — it fills all six
sections rather than stopping early. Single-stream generation is ~58 tok/s at `-ngl 99`.

#### Estimate vs measurement

| | Estimated | Measured | |
|---|---|---|---|
| Throughput | 1000-1400 tok/s | ~570 tok/s | 2.4× optimistic |
| Training (20k rows) | 3-5h | ~13.6h | budget cut to 10k ([ADR 0012](adr/0012-revised-training-budget.md)) |
| Peak VRAM | 7.7-8.3 GB | 9.07 GB | evaluation runs alongside the training allocation |

The throughput estimate in ADR 0009 was never measured; a 160-row smoke run disproved it in
7 minutes, and the budget was re-derived before committing a night of GPU time.

### Base vs fine-tuned

200 held-out rows, Q4_K_M through the same quantisation lineage, identical decoding
(temp 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05, seed 3407), served sequentially by
llama.cpp.

| Metric | Base | Fine-tuned | |
|---|---|---|---|
| **Format validity** | 0.000 | **1.000** | 200/200 rows, six sections in order |
| **Perplexity** (held-out) | 4.7283 | **2.7733** | −41% |
| Think leak rate | 0.000 | 0.000 | ADR 0006 held |
| Portuguese rate | 1.000 | 1.000 | |
| Mean chars | 2093 | 3072 | reference: 3491 |
| Grounding recall | 1.000 | 0.903 | see below |

**Format validity is the headline.** The base model never once produced the required
structure; the fine-tune produced it on every single row.

#### The grounding number is not a regression

Grounding recall reads 1.000 → 0.903, which looks like the fine-tune got worse at using
its input. Scoring the **ground truth** with the same function settles it:

| Field | Base | Fine-tuned | Ground truth |
|---|---|---|---|
| name | 1.000 | 0.995 | 1.000 |
| municipality | 1.000 | 0.995 | 1.000 |
| state | 1.000 | 0.890 | 0.815 |
| occupation | 1.000 | 0.890 | 0.870 |
| age | 1.000 | 0.745 | 0.650 |
| **recall** | **1.000** | **0.903** | **0.867** |

The human-written reference scores 0.867. The fine-tuned model scores **higher** than the
reference on state, occupation and age, and matches it on name and municipality. The base
model's perfect 1.000 comes from echoing the attribute block back verbatim in an
"Identificação Pessoal" section — a degenerate strategy that scores perfectly while
producing worse text.

This is exactly the saturated metric ADR 0010 anticipated, caught by comparing against the
reference rather than trusting the number. **Interpretation: grounding is a pass/fail
sanity check, not a ranking metric.** A future run should score it against the reference
as a baseline rather than against 1.0.

### Qualitative

Same input, both models:

> `José Pereira | Masculino | 49 | Diretor ou gerente | São Paulo, São Paulo`

**Fine-tuned** — correct structure, natural pt-BR, invents coherent specifics:
> `## Síntese`
> José Pereira é um diretor de 49 anos que equilibra liderança estratégica, fé evangélica
> ativa e paixão por futebol e música, mas costuma adiar a conclusão da graduação em
> Administração enquanto luta contra a procrastinação.

**Base** — wrong structure, restates the input as a form:
> `**1. Identificação Pessoal:**`
> José Pereira é um homem de 49 anos, solteiro, residente no município de São Paulo,
> estado de São Paulo, Brasil.

### Artifacts

Artifacts (gitignored): `outputs/adapter/` (333MB), `outputs/gguf/personas-tuned-q4_k_m.gguf`
(4.68GB), `outputs/eval/report.html`.

---

## Reproducing

```bash
make setup && make data && make train && make export && make eval && make report
```

On an 8GB card use the config as committed (ADR 0014). On a 12GB card, run 1's settings
can be restored: `model.train_id` removed, batch 2 × 8, `max_seq_length` 2048,
`train.eval_strategy: steps`.
