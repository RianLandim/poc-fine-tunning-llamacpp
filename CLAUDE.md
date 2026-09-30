# poc-fine-tunning-llamacpp

Fine-tune **Qwen3-8B** with **Unsloth QLoRA** on NVIDIA's synthetic Brazilian persona
corpus, export to **GGUF**, serve and evaluate with **llama.cpp** — originally on a single
RTX 3060 (12GB). **This fork targets an RTX 3070 Ti (8GB)** — see ADR 0013 for what that
changes.

**Task:** demographic attributes in, six-section Brazilian-Portuguese persona narrative out.

## Read first

- [`docs/superpowers/specs/2026-09-12-persona-finetune-design.md`](docs/superpowers/specs/2026-09-12-persona-finetune-design.md) — the design
- [`docs/adr/README.md`](docs/adr/README.md) — why each choice was made

## Hardware and environment

| | |
|---|---|
| GPU | RTX 3070 Ti **8GB**, ~7.2GB free (GNOME holds ~0.3-0.4GB) |
| RAM / disk | 31GB / 259GB free |
| Driver / CUDA | 580.178.04, Ampere `sm_86`; **no system CUDA toolkit** — `vendor/cuda` (12.8.1) |
| Python | always use the project `uv` venv on 3.12, never the host Python |

Earlier ADRs, the design spec and `docs/RESULTS.md` were written for and measured on the
RTX 3060 12GB. They are the historical record; do not rewrite their numbers.

Never `pip install` into the host Python. Use `uv run` / `uv sync`; dependencies are
pinned in the committed `uv.lock` (ADR 0011).

## Pipeline

```
make setup → make data → make train → make export → make serve → make eval → make report
```

`make smoke` runs the whole chain on a 200-row slice in minutes. Run it before any
multi-hour training run.

**On this card `make train` and the tuned half of `make export` do not run** (see VRAM).
Everything else does: setup, data, base export, serve, eval, report.

## Invariants — breaking these silently invalidates results

1. **One prompt renderer.** Every prompt comes from `src/personas/prompt.py`. Never build a
   prompt string anywhere else. `tests/test_prompt_parity.py` enforces this (ADR 0006).
2. **Thinking mode off everywhere.** Always `apply_chat_template(..., enable_thinking=False)`.
   Any `<think>` in a generation is a bug and is scored as a format failure.
3. **Raw `/completion` only.** Never evaluate through `/v1/chat/completions` — llama-server
   re-applies its own Jinja template and breaks train/inference parity (ADR 0006, 0007).
4. **Base and tuned never run concurrently.** Two Q4_K_M 8B models (~10GB) do not fit in 8GB.
   Eval is sequential; the Makefile owns server lifecycle (ADR 0010).
5. **Base and tuned share a quantisation lineage.** Both go through the same merge →
   convert → quantise steps so the A/B measures the LoRA and nothing else (ADR 0008).
6. **Changing the system prompt invalidates the adapter.** Bump `PROMPT_VERSION` in
   `prompt.py`; eval checks it against the dataset manifest.
7. **LangChain lives only in stages 5-6.** It must never become a training dependency —
   its release cadence cannot be allowed to break a 4-hour GPU run (ADR 0007).

## VRAM

Qwen3-8B QLoRA does **not** train on this card (ADR 0013). The 4-bit checkpoint is 7.0GB
against ~7.2GB free, so the load fails before the first step, and the measured training
peak on the 3060 was 9.13GB. The old OOM ladder (batch 1 → `max_seq_length` 1536 → stop
GNOME) cannot close that gap — do not spend time on it. The tuned merge in
`03_export_gguf.py` loads the same model and fails the same way.

Serving fits: Q4_K_M 8B with `-ngl 99 -c 4096` peaks at ~5.5GB.

## Operational gotchas found the hard way

- **System RAM, not VRAM, killed the first training run.** torch inductor spawns one
  compile worker per CPU core (16 here) when kernels are JIT-compiled at the first step.
  `scripts/02_train.py` caps `TORCHINDUCTOR_COMPILE_THREADS=4` before importing torch.
  Keep that cap, and keep it above the torch import.
- **Never pin `unsloth` without pinning `unsloth-zoo`.** They move together. Pinning one
  resolved an 8-month-old Unsloth against transformers 5.17, whose zoo needed a torch
  symbol that did not exist yet. Let the resolver pick the coupled set as a unit.
- **`torchvision` must come from the same index as `torch`.** transformers imports it, and
  a build against a different CUDA major aborts the import. It is declared as a direct
  dependency purely so `[tool.uv.sources]` applies — source mappings do not reach
  transitive dependencies.
- **`datasets` streaming aborts the process at interpreter shutdown**, turning a
  successful run into exit 134. `scripts/01_prepare_data.py` exits via `os._exit` after
  flushing.
- **The llama.cpp build needs `nvcc`, which the torch wheels do not ship.** This host has
  no system toolkit and no passwordless sudo, so `scripts/00_setup_cuda.sh` installs one
  into `vendor/cuda` with micromamba, and `00_setup_llamacpp.sh` bakes its lib dir into the
  rpath. Without that rpath the binaries build fine and then fail at launch with
  `libcudart.so.12: cannot open shared object file`.
- **Never pipe a long-running script through `tail`** to inspect it — the pipeline's exit
  code is `tail`'s, so failures report as success. Redirect to a file instead.

## Data note

The dataset is CC-BY-4.0 and **fully synthetic** — no real PII, despite being "person
data". Generated samples can be shared freely.
