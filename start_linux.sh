#!/usr/bin/env bash
# Registration Studio v1 — Linux launcher. Run ./start_linux.sh (or double-click → Run).
# First start: creates the conda env "reg_env" (or a venv in ~/.regstudio_venv without conda) and installs
# PyTorch for this machine (CUDA if an NVIDIA driver is present, else CPU). Then opens http://127.0.0.1:8780/.
# Extra arguments go to launch.py, e.g. ./start_linux.sh --port 8790 --no-browser
set -euo pipefail
cd "$(dirname "$0")"
ENV=reg_env
export PYTORCH_ENABLE_MPS_FALLBACK=1

find_conda() {
  command -v conda 2>/dev/null && return
  for c in "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           /opt/conda/bin/conda /opt/miniforge3/bin/conda; do [ -x "$c" ] && { echo "$c"; return; }; done
}
CONDA=$(find_conda || true)

if [ -n "$CONDA" ]; then
  if ! "$CONDA" env list | grep -qE "^$ENV[[:space:]]"; then
    echo "Creating conda environment $ENV (first start, a few minutes)…"
    "$CONDA" env create -f environment.yml
  fi
  PY=$("$CONDA" run -n $ENV python -c "import sys; print(sys.executable)")
else
  echo "conda not found → using a Python venv (~/.regstudio_venv). Miniforge is recommended: https://conda-forge.org/download/"
  PY3=$(command -v python3 || true)
  [ -z "$PY3" ] && { echo "python3 not found; install Python >= 3.10 or Miniforge"; exit 1; }
  if [ ! -x "$HOME/.regstudio_venv/bin/python" ]; then
    "$PY3" -m venv "$HOME/.regstudio_venv"
    "$HOME/.regstudio_venv/bin/python" -m pip install --upgrade pip
    "$HOME/.regstudio_venv/bin/python" -m pip install -r requirements.txt
  fi
  PY="$HOME/.regstudio_venv/bin/python"
fi

"$PY" tools/setup_torch.py
echo "Starting Registration Studio $(cat VERSION) … (keep this window open; Ctrl+C stops the server)"
exec "$PY" launch.py "$@"
