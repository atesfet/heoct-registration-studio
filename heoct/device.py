"""Compute-device selection: NVIDIA CUDA -> Apple-silicon MPS -> CPU.

Override with the environment variable HEOCT_DEVICE=cuda|mps|cpu.
PYTORCH_ENABLE_MPS_FALLBACK=1 lets ops MPS does not implement run on the CPU. Bicubic grid_sample is *implemented
but unsupported* on MPS (it raises), so register.grid_sample_safe handles that case explicitly.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def get_device() -> str:
    want = os.environ.get("HEOCT_DEVICE", "").strip().lower()
    if want:
        return want
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(dev: str | None = None) -> str:
    dev = dev or DEVICE
    if dev == "cuda":
        return torch.cuda.get_device_name(0)
    if dev == "mps":
        return "Apple GPU (Metal/MPS)"
    return f"CPU ({os.cpu_count()} threads)"


def empty_cache() -> None:
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


DEVICE = get_device()
