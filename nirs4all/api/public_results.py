"""Public persistence and native result queries through the portable Core."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def open_experiment(directory: str | Path) -> Any:
    """Reopen checked native scores, prediction identities and optional model."""
    from nirs4all_core import open_experiment as core_open

    return core_open(directory)


def save_experiment(native_results_dir: str | Path, destination: str | Path, *, run_id: str,
                    winner_variant_id: str, model_archive: str | Path | None = None) -> Any:
    """Save native DAG results and an optional closed portable winner archive."""
    from nirs4all_core import save_experiment as core_save

    return core_save(native_results_dir, destination, run_id=run_id,
                     winner_variant_id=winner_variant_id, model_archive=model_archive)
