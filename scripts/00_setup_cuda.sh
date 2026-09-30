#!/usr/bin/env bash
# Stage 0 -- provide a CUDA compiler for the llama.cpp build without root (ADR 0013).
#
# The torch wheels bundle the CUDA *runtime*, but building llama.cpp needs nvcc, and this
# host has no system toolkit. Rather than require sudo, install one into vendor/cuda with
# micromamba. 12.8 matches the torch cu128 wheels (ADR 0011). A no-op when nvcc is
# already on PATH or the prefix already exists.
set -euo pipefail

CUDA_PREFIX="${CUDA_PREFIX:-vendor/cuda}"
CUDA_VERSION="${CUDA_VERSION:-12.8.1}"
MAMBA_DIR="${MAMBA_DIR:-vendor/mamba}"

if [ -x "$CUDA_PREFIX/bin/nvcc" ]; then
  echo ">> local CUDA toolkit already at $CUDA_PREFIX"
  exit 0
fi
if command -v nvcc >/dev/null 2>&1; then
  echo ">> system nvcc found at $(command -v nvcc); no local toolkit needed"
  exit 0
fi

if [ ! -x "$MAMBA_DIR/bin/micromamba" ]; then
  echo ">> downloading micromamba into $MAMBA_DIR"
  mkdir -p "$MAMBA_DIR"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "$MAMBA_DIR" bin/micromamba
fi

echo ">> installing CUDA $CUDA_VERSION toolkit into $CUDA_PREFIX (~2.3GB)"
"$MAMBA_DIR/bin/micromamba" create -y -q \
  -r "$MAMBA_DIR/root" -p "$CUDA_PREFIX" \
  -c "nvidia/label/cuda-$CUDA_VERSION" -c conda-forge \
  cuda-nvcc cuda-cudart-dev libcublas-dev

[ -x "$CUDA_PREFIX/bin/nvcc" ] || { echo "FATAL: nvcc missing after install" >&2; exit 1; }
echo ">> ok: $("$CUDA_PREFIX/bin/nvcc" --version | tail -1)"
