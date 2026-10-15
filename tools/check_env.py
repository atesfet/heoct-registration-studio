#!/usr/bin/env python3
"""Check that every package the pipeline needs is importable; pip-install any that are missing.

Called by the start_* launchers on every start, so an existing environment (e.g. a "reg_env" created
elsewhere) is completed instead of failing mid-run. Usage: python tools/check_env.py [--check-only]
"""
import importlib
import subprocess
import sys

# module to import -> pip package name
REQUIRED = {"numpy": "numpy", "scipy": "scipy", "skimage": "scikit-image", "tifffile": "tifffile",
            "imagecodecs": "imagecodecs", "cv2": "opencv-python-headless", "matplotlib": "matplotlib",
            "pandas": "pandas", "yaml": "pyyaml", "SimpleITK": "SimpleITK", "pymupdf": "pymupdf"}


def missing():
    out = []
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            out.append(pkg)
    return out


def main():
    m = missing()
    if not m:
        print(f"Environment OK (Python {sys.version.split()[0]}, all required packages present).")
        return
    print("Missing packages: " + ", ".join(m))
    if "--check-only" in sys.argv:
        sys.exit(1)
    print("Installing them into " + sys.executable + " …", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", *m])
    importlib.invalidate_caches()
    still = missing()
    if still:
        print("Still missing after install: " + ", ".join(still)); sys.exit(1)
    print("Environment OK.")


if __name__ == "__main__":
    main()
