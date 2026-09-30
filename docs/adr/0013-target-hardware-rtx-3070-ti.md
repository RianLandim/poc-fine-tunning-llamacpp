# ADR 0013 — Target hardware: RTX 3070 Ti 8GB

- **Status:** Accepted
- **Date:** 2026-09-30

## Context

Every earlier ADR, the design spec and `docs/RESULTS.md` were written for and measured on
an RTX 3060 with 12GB (~9.5GB free). This fork targets a different machine:

| | |
|---|---|
| GPU | RTX 3070 Ti **8GB**, ~7.2GB free with GNOME resident (desktop holds ~0.3-0.4GB) |
| RAM / disk | 31GB / 259GB free |
| Driver | 580.178.04, Ampere `sm_86` |
| CUDA toolkit | none on the host, and no passwordless sudo |

Measured on this card on 2026-09-30:

- **Training does not start.** The 4-bit Qwen3-8B checkpoint is 7.0GB on disk. Loading it
  fails in `FastLanguageModel.from_pretrained` with "Some modules are dispatched on the
  CPU or the disk" — before the first step, so none of the OOM fallbacks in ADR 0002
  (batch 1, `max_seq_length` 1536, stopping GNOME) can help. ADR 0012 measured a 9.13GB
  training peak; the card has 8GB in total.
- **The tuned merge does not run** for the same reason: `03_export_gguf.py --which tuned`
  loads the same 4-bit model on the GPU.
- **Base export works.** It materialises fp16 weights on the CPU; conversion and
  quantisation are CPU-only.
- **Serving works.** `personas-base-q4_k_m.gguf` (4.68GB) with `-ngl 99 -c 4096` peaked at
  5.5GB of VRAM while generating through raw `/completion`.

## Decision

1. This fork targets the RTX 3070 Ti 8GB. Hardware figures in `CLAUDE.md` and the README
   describe this card; earlier ADRs and `docs/RESULTS.md` stay as the 3060 record.
2. The CUDA compiler for the llama.cpp build is a project-local toolkit in `vendor/cuda`,
   installed by `scripts/00_setup_cuda.sh` via micromamba (CUDA 12.8.1, matching the torch
   cu128 wheels of ADR 0011). `scripts/00_setup_llamacpp.sh` bakes its lib dir into the
   binaries' rpath so they run without `LD_LIBRARY_PATH`. No root is required.
3. llama.cpp is pinned at `bdeb855b30dfe7f6e695cba98445a7ba09e6416e`, the commit built and
   verified here.

## Consequences

- `make setup`, `make data`, base export, `make serve`, `make eval` and `make report` run
  on this card. Eval and report still need a tuned GGUF.
- `make train` and the tuned half of `make export` do not run with Qwen3-8B. Until the
  base model is revisited, a tuned GGUF has to be produced on a larger card.
- ADR 0003 chose Qwen3-8B partly because it fits 4-bit QLoRA on 12GB. That premise does
  not hold here; training on this card needs a smaller base model, which would be a new
  ADR superseding 0003 and would invalidate the numbers in `docs/RESULTS.md`.
- Invariant 4 still holds with less room: two Q4_K_M 8B models (~10GB) do not fit in 8GB.
- `vendor/cuda` costs ~2.3GB of disk and is not tracked by git.

## Alternatives considered

- **System CUDA toolkit via apt.** Needs sudo, and ties the build to whatever the distro
  ships. Rejected in favour of a pinned, removable local prefix.
- **CPU offload of part of the 4-bit model for training.** bitsandbytes keeps offloaded
  modules in fp32 on the CPU; step time would be far beyond the 39 s/step that already
  made ADR 0012 cut the budget. Not attempted.
