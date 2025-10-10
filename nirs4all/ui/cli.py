"""Simple CLI to run the nirs4all UI backend (development mode).

This small helper starts uvicorn programmatically so the user can run:

    python -m nirs4all.ui.cli

or use the top-level helper script `run_ui.py` added to the project root.
"""
from __future__ import annotations
import argparse
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="nirs4all-ui")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind the server")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev only)")
    parser.add_argument("--pred-path", default="results", help="Path where predictions are stored; passed to GET /api/predictions/counts when the UI queries it")

    args = parser.parse_args(argv)

    try:
        import uvicorn
    except Exception:  # pragma: no cover - runtime dependency
        print("❌ uvicorn is required to run the web UI. Install it with: pip install uvicorn[standard]")
        sys.exit(2)

    # Run uvicorn programmatically, pointing to the server app
    uvicorn.run("nirs4all.ui.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
