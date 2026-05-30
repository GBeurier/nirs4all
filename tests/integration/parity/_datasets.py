"""Dataset-path resolution for the parity suite.

Cases reference datasets by stable string key — never by hardcoded path — so
the registry stays portable across cwd / IDE / CI.

Path resolution walks up from this file: `parents[3]` lands on the nirs4all
project root (`/home/delete/nirs4all/nirs4all/`), which contains the
`examples/sample_data/` and `examples/sample_datasets/` corpora the cases use.

Validation is **lazy**: `dataset_path(key)` only checks the directory on
lookup. Import-time validation was over-eager and made one missing fixture
abort the whole test collection.
"""

from __future__ import annotations

from pathlib import Path

# parents[0]=parity, [1]=integration, [2]=tests, [3]=project root (nirs4all/)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SAMPLE_DATA = _PROJECT_ROOT / "examples" / "sample_data"
_SAMPLE_DATASETS = _PROJECT_ROOT / "examples" / "sample_datasets"


DATASETS: dict[str, Path] = {
    # Single-source regression corpus (cal/val splits).
    "regression": _SAMPLE_DATA / "regression",
    # Secondary regression corpus, distinct sample count for cross-checks.
    "regression_2": _SAMPLE_DATA / "regression_2",
    # Multi-source corpus: 3 NIR sources (Xcal_1/2/3) over shared targets.
    "multi": _SAMPLE_DATA / "multi",
    # Binary classification (1d target).
    "binary": _SAMPLE_DATA / "binary",
    # Multi-class classification (string class labels).
    "classification": _SAMPLE_DATA / "classification",
}


# Parser-fixture datasets used by the loaders test suite. Surfaced here so the
# parity oracle can exercise multi-source-with-markers and repetition cases
# that need the richer folder layouts.
PARSER_FIXTURES: dict[str, Path] = {
    "dual_source": _SAMPLE_DATASETS / "E01_dual_source",
    "nir_markers": _SAMPLE_DATASETS / "E02_nir_markers",
    "aggregate_mean": _SAMPLE_DATASETS / "E04_aggregate_mean",
    "aggregate_outliers": _SAMPLE_DATASETS / "E05_aggregate_outliers",
    "custom_folds": _SAMPLE_DATASETS / "D05_custom_folds",
}


def dataset_path(key: str) -> str:
    """Resolve a parity dataset key to an absolute path.

    Raises:
        KeyError: if `key` is not registered in DATASETS or PARSER_FIXTURES.
        FileNotFoundError: if the resolved directory is missing — surfaces a
            misplaced fixture immediately rather than during a long pipeline run.
    """
    lookup = DATASETS.get(key) or PARSER_FIXTURES.get(key)
    if lookup is None:
        all_keys = sorted(DATASETS) + sorted(PARSER_FIXTURES)
        raise KeyError(f"unknown parity dataset key {key!r}; declared keys: {all_keys}")
    if not lookup.exists():
        raise FileNotFoundError(
            f"parity dataset {key!r} missing on disk at {lookup}; "
            f"examples/sample_data may need to be repopulated"
        )
    return str(lookup)


def known_keys() -> list[str]:
    """Stable sorted list of all dataset keys (used for registry validation)."""
    return sorted({*DATASETS.keys(), *PARSER_FIXTURES.keys()})
