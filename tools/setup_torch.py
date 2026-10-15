#!/usr/bin/env python3
"""Install the right PyTorch build into the current environment (called by the start_* launchers).

  NVIDIA driver found (nvidia-smi)  -> CUDA 12.4 build   (Linux / Windows)
  macOS                             -> standard build     (Apple-silicon GPU via Metal/MPS)
  otherwise                         -> CPU-only build     (works everywhere, slower)

Does nothing if torch is already importable. Usage: python tools/setup_torch.py [--force]
"""
import platform
import shutil
import subprocess
import sys


def have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def main():
    force = "--force" in sys.argv
    if have_torch() and not force:
        import torch
        print(f"PyTorch {torch.__version__} already installed.")
        return
    pip = [sys.executable, "-m", "pip", "install", "--upgrade"]
    if sys.platform == "darwin":
        target, cmd = "macOS (Apple GPU / MPS if available)", pip + ["torch"]
    elif shutil.which("nvidia-smi"):
        target, cmd = "NVIDIA GPU (CUDA 12.4)", pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cu124"]
    else:
        target, cmd = "CPU only", pip + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
    print(f"Installing PyTorch for {target} on {platform.system()} {platform.machine()} …", flush=True)
    subprocess.check_call(cmd)
    import importlib
    importlib.invalidate_caches()
    import torch
    dev = "cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    print(f"PyTorch {torch.__version__} installed; compute device: {dev}")


if __name__ == "__main__":
    main()
