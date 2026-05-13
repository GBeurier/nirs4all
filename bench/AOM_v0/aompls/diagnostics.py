"""Diagnostic helpers for AOM-PLS / POP-PLS estimators.

Builds tabular summaries of operator selection, component scores, and run
metadata, suitable for embedding into benchmark CSV rows or for inspection in
a notebook.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunDiagnostics:
    """Diagnostic record produced by every fit call."""

    engine: str = ""
    selection: str = ""
    criterion: str = ""
    orthogonalization: str = ""
    operator_bank: str = ""
    selected_operator_indices: List[int] = field(default_factory=list)
    selected_operator_names: List[str] = field(default_factory=list)
    operator_scores: Dict[str, Any] = field(default_factory=dict)
    n_components_selected: int = 0
    max_components: int = 0
    fit_time_s: float = 0.0
    predict_time_s: float = 0.0
    backend: str = "numpy"
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def operator_sequence_string(diag: RunDiagnostics, max_len: int = 6) -> str:
    """Compact, human-readable operator sequence."""
    names = diag.selected_operator_names
    if not names:
        return "(none)"
    if len(set(names)) == 1:
        return f"{names[0]} x {len(names)}"
    if len(names) <= max_len:
        return " | ".join(names)
    head = names[: max_len - 1]
    return " | ".join(head) + f" | ... ({len(names) - max_len + 1} more)"
