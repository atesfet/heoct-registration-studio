#!/usr/bin/env python3
"""Start the H&E <-> OCT Registration Studio web app.

    python launch.py                                   # local: http://127.0.0.1:8780/ (opens the browser)
    python launch.py --no-browser --port 8780
    python launch.py --host 0.0.0.0 --token SECRET --allow-root /data   # deploy on a server (see README)

The platform launcher (start_linux.sh) creates/activates the conda environment and then runs this file.
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: unsupported ops fall back to CPU
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="interface to bind (127.0.0.1 = this machine only)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--token", help="require this access token (use 'auto' to generate one); recommended with --host 0.0.0.0")
    ap.add_argument("--allow-root", action="append", default=[], help="restrict browsing/reading/writing to this folder (repeatable)")
    a = ap.parse_args()
    token = secrets.token_urlsafe(18) if a.token == "auto" else a.token
    from webapp.server import serve
    serve(a.host, a.port, open_browser=not a.no_browser, token=token, allow_roots=a.allow_root)


if __name__ == "__main__":
    main()
