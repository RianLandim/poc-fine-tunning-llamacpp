# GPU comparison — RTX 3060 12GB vs RTX 3070 Ti 8GB

The same pipeline — Qwen3-8B QLoRA on 10,000 persona rows, GGUF export, llama.cpp
evaluation on 200 held-out rows — run end to end on two consumer Ampere cards. Detailed
per-run numbers live in [`RESULTS.md`](RESULTS.md); this page puts them side by side.

**Summary:** the 3070 Ti has a third less VRAM but ~70% more memory bandwidth. It needs a
tighter training configuration to fit, and in exchange it trains 40% faster, evaluates 36%
faster, and lands within a few points of the 3060 on every quality metric.

---

## Hardware

| | RTX 3060 12GB | RTX 3070 Ti 8GB |
|---|---|---|
| Chip | GA106 (Ampere, `sm_86`) | GA104 (Ampere, `sm_86`) |
| CUDA cores | 3,584 | 6,144 |
| VRAM | **12 GB** GDDR6 | **8 GB** GDDR6X |
| Memory bandwidth | 360 GB/s | **608 GB/s** |
| Board power | 170 W | 290 W |
| VRAM visible to torch | 11.63 GB | 7.65 GB |
| Held by the desktop (GNOME) | ~2.1–2.7 GB | ~0.1–0.4 GB |
| Host RAM | 30 GB | 31 GB |
| Driver | 595.84 | 580.178.04 |
| CUDA toolkit for the llama.cpp build | system, 13.3 | none on host; local 12.8.1 in `vendor/cuda` ([ADR 0013](adr/0013-target-hardware-rtx-3070-ti.md)) |

---

## Training configuration

Everything not listed is identical: LoRA r=32 / alpha 32 on all projection layers, lr 2e-4
cosine, adamw 8-bit, one epoch, effective batch 16, seed 3407.

| | RTX 3060 12GB | RTX 3070 Ti 8GB | Why it differs |
|---|---|---|---|
| 4-bit weights | Unsloth dynamic, 6.97 GB | standard bnb, **5.66 GB** | the dynamic checkpoint fails to load in 8GB |
| Batch × grad-accum | 2 × 8 | **1 × 16** | activation memory |
| `max_seq_length` | 2048 | **1536** | longest example is ~1450–1750 tokens; 2048 only reserved VRAM |
| In-training validation | every 100 steps, 150 rows | **off** | an eval pass materialises fp32 logits and OOMs |
| Rows dropped as too long | — | 19 of 10,727 | side effect of the 1536 limit |

Rationale and measurements: [ADR 0014](adr/0014-training-qwen3-8b-on-8gb.md).

---

## Training

| | RTX 3060 12GB | RTX 3070 Ti 8GB | Difference |
|---|---|---|---|
| Optimizer steps | 625 | 625 | |
| Step time | 42.7 s | **25.6 s** | −40% |
| Wall clock | 7h 25m | **4h 26m** | −3h |
| Peak VRAM | 9.07 GB | **7.10 GB** | |
| Headroom at peak | 2.56 GB of 11.63 | **0.55 GB** of 7.65 | 3070 Ti runs near the limit |
| Trainable params | 87.3M (1.05%) | 87.3M (1.05%) | same adapter shape |
| Train loss, start → end | 1.72 → 0.91 (run mean) | 1.72 → 0.84 (last log) · 0.91 run mean | |
| Validation loss | 0.945 → **0.827**, monotonic | not measured | |

The step-time gain tracks memory bandwidth: 4-bit QLoRA with gradient checkpointing is
bandwidth-bound, and 608 / 360 GB/s ≈ 1.7× against a measured 1.67× speedup.

The price is margin. On the 3060, opening a browser mid-run was a risk; on the 3070 Ti,
with ~0.5 GB spare, it is a likely OOM.

---

## Export and evaluation

| | RTX 3060 12GB | RTX 3070 Ti 8GB | Difference |
|---|---|---|---|
| GGUF export (tuned) | 6m 26s | 6m 13s | CPU / disk bound, effectively equal |
| Serving peak VRAM (Q4_K_M, `-c 4096`) | — | ~5.5 GB | fits either card with room to spare |
| Base: 200 generations | 1050 s (5.3 s/row) | **666 s (3.3 s/row)** | −37% |
| Tuned: 200 generations | 1610 s (8.1 s/row) | **1001 s (5.0 s/row)** | −38% |
| Whole `make eval` | 46m 19s | **29m 25s** | −36% |

Both cards evaluate base and tuned **sequentially**: two Q4_K_M 8B models need ~10 GB plus
KV cache, which fits neither.

---

## Quality — base vs fine-tuned

Q4_K_M, identical decoding (temp 0.7, top_p 0.8, top_k 20, repeat_penalty 1.05, seed 3407).

| Metric | Base (3060) | Base (3070 Ti) | Tuned (3060) | Tuned (3070 Ti) |
|---|---|---|---|---|
| **Format validity** | 0.000 | 0.000 | **1.000** | **0.990** |
| **Perplexity** | 4.728 | 4.738 | **2.773** | **2.804** |
| Grounding recall | 1.000 | 1.000 | 0.903 | 0.872 |
| Think leak | 0.000 | 0.000 | 0.000 | 0.000 |
| Portuguese | 1.000 | 1.000 | 1.000 | 1.000 |
| Mean chars | 2093 | 2072 | 3072 | 2998 |

The two base columns agree to within noise, which is what they should do: same model, same
quantisation, different test rows. The two tuned columns are within 1–3 points of each
other on every metric.

The 3070 Ti's two format failures had all six sections in order and appended a seventh;
none was missing or out of order.

### Grounding against the reference

Grounding recall is only meaningful relative to the human-written reference, scored with
the same function on the same rows (see run 1's analysis in `RESULTS.md`):

| Field | Tuned (3060) | Reference (3060 rows) | Tuned (3070 Ti) | Reference (3070 Ti rows) |
|---|---|---|---|---|
| name | 0.995 | 1.000 | 1.000 | 1.000 |
| municipality | 0.995 | 1.000 | 0.995 | 1.000 |
| state | 0.890 | 0.815 | 0.855 | 0.830 |
| occupation | 0.890 | 0.870 | 0.890 | 0.890 |
| age | 0.745 | 0.650 | 0.620 | 0.580 |
| **recall** | **0.903** | **0.867** | **0.872** | **0.860** |
| **Tuned − reference** | | **+0.036** | | **+0.012** |

Both fine-tunes sit above their reference. The 3060 run's margin is larger, most visibly
on age. With no validation loss on the 3070 Ti run and different test rows, this is not
enough to say which configuration grounds better; it would take the same test split on
both to settle it.

---

## End to end

| Stage | RTX 3060 12GB | RTX 3070 Ti 8GB |
|---|---|---|
| Setup (download-bound) | ~25 min | ~25 min |
| Data preparation | ~3 min | ~2 min |
| Fine-tuning | 7h 25m | **4h 26m** |
| GGUF export | 6m 26s | 6m 13s |
| Evaluation | 46m 19s | **29m 25s** |
| **Total** | **≈ 9h** | **≈ 5h 30m** |

---

## Caveats

- **Not a controlled experiment.** The training checkpoint, batch shape, sequence limit
  and validation schedule all changed together, and the 1536 limit changed which rows
  landed in each split. Differences of a few points between the tuned columns cannot be
  attributed to the GPU or to any single setting.
- **The base model and export chain are the same.** Both runs export from the fp16
  `unsloth/Qwen3-8B`, so inference-side numbers (base metrics, speed) compare cleanly.
- **Single run each**, one seed. No variance estimate.

## Which card to use

- **RTX 3070 Ti 8GB** — faster at every GPU-bound stage and good enough on quality. Train
  from a light desktop or a TTY, and accept that there is no validation loss during
  training.
- **RTX 3060 12GB** — slower, but the headroom allows in-training validation, the larger
  dynamic 4-bit checkpoint and a longer sequence limit, and tolerates a normal desktop
  session during a multi-hour run.
