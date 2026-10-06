"""Common native generation and resumable Methods HPO frontends."""
from __future__ import annotations

from typing import Any


def generate_variants(choices: dict[str, list[Any]], **kwargs: Any) -> list[dict[str, Any]]:
    """Enumerate constrained native variants or a seeded random subset."""
    from nirs4all_core._tuning import generate
    return generate(choices, **kwargs)


def tune_native(data: Any, **kwargs: Any) -> Any:
    """Tune PLS components 1..3 and scaling with native OOF selection and refit."""
    from nirs4all_core._tuning import tune
    return tune(data, **kwargs)


def resume_native_tuning(checkpoint: Any, data: Any, **kwargs: Any) -> Any:
    """Extend the total native history budget from a closed archive checkpoint."""
    from nirs4all_core._tuning import resume_tuning
    return resume_tuning(checkpoint, data, **kwargs)


def load_native_tuning(directory: Any, **kwargs: Any) -> Any:
    """Reload the native selected model and signed optimizer options."""
    from nirs4all_core._tuning import load_tuning
    return load_tuning(directory, **kwargs)
