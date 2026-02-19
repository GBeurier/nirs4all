# Beta-Readiness Audit: Data Module (`nirs4all/data/`) - Release Blockers Only (2026-02-19)

## Active Findings

(None remaining.)

## Beta Release Tasks (Open)

(All completed.)

## Resolved Findings
- `F-1.3 [RESOLVED]` Signal type detection no longer prefers `partition="train"`; `_detect_signal_type()` now uses all available data (`self.x({}, layout="2d")`) for unbiased detection.
- `F-1.5 [RESOLVED]` Broad `except Exception: pass` blocks in `dataset.py` metadata/stat extraction paths replaced with `logger.debug(exc)` — errors are now visible in debug logs instead of silently swallowed.
- `F-2.2 [RESOLVED]` Metadata reload path now routes through `_load_file_with_registry()` with `na_policy='ignore'` instead of raw `pd.read_csv()`.
- `F-3.2 [RESOLVED]` `FolderParser.SUPPORTED_EXTENSIONS` expanded to cover all loader-supported formats (CSV, TSV, TXT, Excel, Parquet, NumPy, MATLAB, compressed).
- `F-8.2 [RESOLVED]` `DataCache.get_or_load()` uses double-checked locking (`RLock`) to eliminate the TOCTOU race.
- `F-9.1 [RESOLVED]` `ConfigValidator` now emits `ErrorRegistry` codes (E1xx, E2xx, E6xx) instead of ad-hoc string codes; test assertions updated to match.
- `F-11.2 [RESOLVED]` `tests/unit/data/predictions/test_predictions.py` added with 44 tests covering empty-instance behavior, add/filter/top/merge, metadata utilities, clear/slice, conversion, context manager, and repetition column.
