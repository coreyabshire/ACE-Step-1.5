#!/usr/bin/env bash
# ACE-Step API launcher for HitMaker
#
# Ensures PyTorch uses its own bundled NVIDIA libraries instead of the
# system CUDA toolkit.  This avoids version mismatches when the system
# toolkit (e.g. CUDA 12.9) is newer than what the PyTorch wheel was
# built against (cu128).
#
# The key symptom: libnvJitLink.so.12 resolving to /usr/local/cuda/lib64
# instead of the venv-bundled copy, causing excess VRAM usage and
# potential generation failures.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Collect all bundled nvidia lib dirs from the venv
NVIDIA_LIBS=""
for d in "$SCRIPT_DIR"/.venv/lib/python3.12/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && NVIDIA_LIBS="${NVIDIA_LIBS:+$NVIDIA_LIBS:}$d"
done

# Prepend so bundled libs win over system /usr/local/cuda
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Widen the safe-path root so ACE-Step can access HitMaker data dirs
export ACESTEP_SAFE_ROOT="/mnt/c/Users/corey/Music"

cd "$SCRIPT_DIR"
exec uv run acestep-api "$@"
