"""Top-level helper to run the minimal nirs4all UI.

Usage:
    python run_ui.py --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
import argparse
from nirs4all.ui.cli import main


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    main([f"--host={args.host}", f"--port={args.port}"] + (["--reload"] if args.reload else []))


if __name__ == "__main__":
    _main()
