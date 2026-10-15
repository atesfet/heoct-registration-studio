#!/usr/bin/env bash
# Registration Studio v1 — macOS launcher. Double-click this file in Finder.
# The first time, macOS may block it: right-click → Open → Open (or run: chmod +x start_mac.command).
# First start: creates the conda env "reg_env" (or a venv in ~/.regstudio_venv) and installs PyTorch, which uses the
# Apple-silicon GPU (Metal/MPS) when available. Then opens http://127.0.0.1:8780/ in your browser.
set -euo pipefail
cd "$(dirname "$0")"
ENV=reg_env
export PYTORCH_ENABLE_MPS_FALLBACK=1   # ops MPS does not implement run on the CPU

find_conda() {
  command -v conda 2>/dev/null && return
  for c in "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           /opt/homebrew/Caskroom/miniforge/base/bin/conda /opt/homebrew/anaconda3/bin/conda /usr/local/anaconda3/bin/conda \
           /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda; do [ -x "$c" ] && { echo "$c"; return; }; done
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
  if [ -z "$PY3" ] || ! "$PY3" -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
    echo "Python >= 3.10 not found. Install Miniforge (https://conda-forge.org/download/) and double-click again."
    read -r -p "Press Enter to close…" _; exit 1
  fi
  if [ ! -x "$HOME/.regstudio_venv/bin/python" ]; then
    "$PY3" -m venv "$HOME/.regstudio_venv"
    "$HOME/.regstudio_venv/bin/python" -m pip install --upgrade pip
    "$HOME/.regstudio_venv/bin/python" -m pip install -r requirements.txt
  fi
  PY="$HOME/.regstudio_venv/bin/python"
fi

"$PY" tools/setup_torch.py
echo "Starting Registration Studio $(cat VERSION) … (keep this window open; close it or press Ctrl+C to stop)"
"$PY" launch.py "$@" || { echo "Registration Studio stopped with an error (see above)."; read -r -p "Press Enter to close…" _; }
