#!/usr/bin/env bash
# Start Registration Studio: creates the conda env "reg_env" on first use, then runs launch.py.
# Usage: ./start_linux.sh [launch.py options, e.g. --port 8780 --no-browser]
set -euo pipefail
cd "$(dirname "$0")"
ENV=reg_env
CONDA=$(command -v conda || echo "$HOME/miniforge3/bin/conda")
if [ ! -x "$CONDA" ]; then echo "conda not found: install Miniforge (https://conda-forge.org/download/)"; exit 1; fi
if ! "$CONDA" env list | grep -qE "^$ENV\s"; then
  echo "Creating conda environment $ENV (first start, a few minutes)…"
  "$CONDA" env create -f environment.yml
fi
PY=$("$CONDA" run -n $ENV python -c "import sys; print(sys.executable)")
if ! "$PY" -c "import torch" 2>/dev/null; then
  echo "Installing PyTorch…"
  if command -v nvidia-smi >/dev/null; then "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu124
  else "$PY" -m pip install torch; fi
fi
exec "$PY" launch.py "$@"
