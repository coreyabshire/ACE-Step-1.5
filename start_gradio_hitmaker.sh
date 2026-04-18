#!/usr/bin/env bash
# ACE-Step Gradio UI launcher for HitMaker
#
# Same CUDA library isolation as start_api_server_hitmaker.sh.
# Ensures PyTorch uses its own bundled NVIDIA libraries instead of the
# system CUDA toolkit (12.9), which breaks half-precision cuBLAS GEMM
# operations needed for LoRA training auto-labeling.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect all bundled nvidia lib dirs from the venv
NVIDIA_LIBS=""
for d in "$SCRIPT_DIR"/.venv/lib/python3.12/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$d"
done

# Prepend so bundled libs win over system /usr/local/cuda
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$SCRIPT_DIR"
exec uv run acestep "$@"
