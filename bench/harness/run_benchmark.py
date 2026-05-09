"""bench/harness/run_benchmark.py.

Resumable benchmark runner consumed by Agents A/B/C and the dashboard.

CLI:
    python bench/harness/run_benchmark.py \
        --cohort fast12_transfer_core \
        --pipeline bench/scenarios/fast_reliable.json \
        --workspace bench/scenarios/runs/fast_reliable \
        --seeds 0,1,2,3,4 \
        --n-jobs -1

Skeleton scope:

  * Resolves the cohort name to a list of datasets.
  * Loads a scenario manifest emitted by
    `bench/export_benchmark_scenarios.py` and walks its `candidates` list.
  * Resumes per `(dataset, seed, canonical_name, selection)` — already
    completed rows are read from the workspace CSV and skipped.
  * Writes a unified-schema CSV `results.csv` in `--workspace`.
  * Logs explicit `skipped` and `failed` rows with reasons.
  * Computes summary statistics (Wilcoxon paired, bootstrap CI,
    Friedman-Nemenyi, Nadeau-Bengio) over completed rows. Stats helpers
    are real but conservative — see `bench.harness.run_benchmark.stats_*`
    docstrings for caveats.

NOT in this skeleton (stubbed clearly):

  * Real dispatch of `module:model_class`. The skeleton dispatches via
    the `ModelDispatcher` interface; the production implementation will
    import the modules listed in the registry. For now the dispatcher
    raises `NotImplementedError` unless `--dry-run` is passed.

DECISION_PENDING_CODEX_REVIEW (D-C-006).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import importlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - probe path only
    yaml = None  # type: ignore[assignment]

# Dataset adapter import has to work both when this file is launched as a
# script (`python bench/harness/run_benchmark.py`) and when it is imported
# as a module (`python -m bench.harness.run_benchmark`). The fallback walks
# two parents up to put the project root on sys.path before retrying.
try:
    from bench.harness.dataset_adapter import (
        DatasetNotFoundError,
        load_dataset,
    )
except ImportError:  # pragma: no cover - script fallback only
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from bench.harness.dataset_adapter import (
        DatasetNotFoundError,
        load_dataset,
    )

BENCH = Path(__file__).resolve().parents[1]
SUBSET_REPORT = BENCH / "Subset_analysis" / "subset_transfer_summary.csv"
RETHOUGHT_PATH = BENCH / "Subset_analysis" / "rethought_subsets.json"
DEFAULT_SCHEMA_VERSION = "0.1.0"
COMPLETION_KEYS = ("dataset", "seed", "canonical_name", "selection")


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


RESULT_FIELDS: tuple[str, ...] = (
    "schema_version",
    "preset",
    "cohort",
    "dataset",
    "task",
    "canonical_name",
    "model_class",
    "module",
    "selection",
    "seed",
    "status",
    "error_message",
    "n_train",
    "n_test",
    "n_features",
    "rmsep",
    "mae",
    "r2",
    "balanced_accuracy",
    "macro_f1",
    "score_metric",
    "score_value",
    "lower_is_better",
    "ref_pls_score",
    "score_ratio_vs_pls",
    "fit_time_s",
    "predict_time_s",
    "started_at",
    "ended_at",
    "host",
    "config_template",
    "notes",
)


@dataclass
class ResultRow:
    schema_version: str = DEFAULT_SCHEMA_VERSION
    preset: str = ""
    cohort: str = ""
    dataset: str = ""
    task: str = "regression"
    canonical_name: str = ""
    model_class: str = ""
    module: str = ""
    selection: str = "default"
    seed: int = 0
    status: str = "ok"
    error_message: str = ""
    n_train: int | None = None
    n_test: int | None = None
    n_features: int | None = None
    rmsep: float | None = None
    mae: float | None = None
    r2: float | None = None
    balanced_accuracy: float | None = None
    macro_f1: float | None = None
    score_metric: str = ""
    score_value: float | None = None
    lower_is_better: bool = True
    ref_pls_score: float | None = None
    score_ratio_vs_pls: float | None = None
    fit_time_s: float | None = None
    predict_time_s: float | None = None
    started_at: str = ""
    ended_at: str = ""
    host: str = ""
    config_template: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Cohort resolver
# ---------------------------------------------------------------------------


def resolve_cohort(name: str) -> list[str]:
    """Return the list of dataset names for a cohort.

    Recognises:
      * `fast12_transfer_core`, `audit20_transfer_core`,
        `current_class_balanced_10`, `current_conservative_19`,
        `legacy_variant_heavy_10` from `bench/Subset_analysis/rethought_subsets.json`
        (when available).
      * `full57` and `full61` from the multiview source CSV.
      * Comma-separated dataset list passed verbatim.
    """
    if "," in name:
        return [item.strip() for item in name.split(",") if item.strip()]
    name_low = name.strip().lower()

    if RETHOUGHT_PATH.exists():
        data = json.loads(RETHOUGHT_PATH.read_text(encoding="utf-8"))
        subsets = data.get("subsets") if isinstance(data, dict) else None
        if isinstance(subsets, dict):
            for key, payload in subsets.items():
                if name_low in {key.lower(), key.replace("-", "_").lower()}:
                    if isinstance(payload, list):
                        return list(payload)
                    if isinstance(payload, dict) and "datasets" in payload:
                        return list(payload["datasets"])

    if name_low in {"full57", "full-57", "full_57"}:
        return _datasets_from_multiview_full57()

    if name_low in {"full61", "full-61", "full_61"}:
        return _datasets_from_multiview_full57(full=True)

    raise SystemExit(
        f"Unknown cohort '{name}'. Provide a comma-separated list, "
        f"or one of: fast12_transfer_core, audit20_transfer_core, full57."
    )


def _datasets_from_multiview_full57(*, full: bool = False) -> list[str]:
    path = BENCH / "AOM_v0" / "multiview" / "results" / "full57.csv"
    if not path.exists():
        return []
    seen: set[str] = set()
    out: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ds = (row.get("dataset") or "").strip()
            if not ds or ds in seen:
                continue
            seen.add(ds)
            out.append(ds)
    if not full and len(out) > 57:
        out = out[:57]
    return out


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if "candidates" not in payload:
        raise SystemExit(f"Manifest {path} has no 'candidates' key.")
    return payload


# ---------------------------------------------------------------------------
# Resume bookkeeping
# ---------------------------------------------------------------------------


def load_completed(workspace: Path) -> set[tuple[str, int, str, str]]:
    """Return the set of (dataset, seed, canonical_name, selection) tuples
    already present in the workspace's results.csv that should NOT be re-run
    on resume.

    Terminal statuses (no retry):
      * ``ok`` — successful completion
      * ``skipped`` — explicit skip (not_runnable_in_production, etc.)
      * ``dry_run`` — dry-run probe row
      * ``failed_terminal`` — D-C-018 dispatcher hardening: timeout / OOM-kill
        / worker-crash. Per Codex D-C-018 verdict Q4: these are infrastructure
        failures, retry would just burn CPU. User must manually delete the
        row to re-run after fixing the underlying issue (memory, timeout
        budget, etc.).

    Plain ``failed`` rows remain retriable so a build_error / dataset_load /
    score_error caused by a fixable bug gets re-tried on the next resume run.
    """
    terminal_statuses = {"ok", "skipped", "dry_run", "failed_terminal"}
    completed: set[tuple[str, int, str, str]] = set()
    results = workspace / "results.csv"
    if not results.exists():
        return completed
    with results.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") not in terminal_statuses:
                continue
            try:
                seed = int(row.get("seed", "0") or 0)
            except ValueError:
                continue
            completed.add(
                (
                    (row.get("dataset") or "").strip(),
                    seed,
                    (row.get("canonical_name") or "").strip(),
                    (row.get("selection") or "default").strip(),
                )
            )
    return completed


def open_results_writer(workspace: Path, append: bool) -> tuple[Any, csv.DictWriter]:
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / "results.csv"
    write_header = not (append and path.exists())
    handle = path.open("a" if append else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(RESULT_FIELDS), extrasaction="ignore")
    if write_header:
        writer.writeheader()
    return handle, writer


# ---------------------------------------------------------------------------
# Model dispatcher
# ---------------------------------------------------------------------------


@dataclass
class DispatchSpec:
    canonical_name: str
    model_class: str
    module: str
    config_template: str
    inputs: dict[str, Any] = field(default_factory=dict)
    not_runnable_in_production: bool = False


class ModelDispatcher:
    """Registry-driven model factory.

    The skeleton resolves the registry entry into a callable that returns
    a ResultRow. The production implementation will import
    `spec.module.spec.model_class`, load `spec.config_template`, and run
    the harness-specific train/predict loop. For now the dispatcher
    surfaces a clear NotImplementedError unless the harness is invoked
    with `--dry-run`.
    """

    def __init__(self, *, dry_run: bool, host: str) -> None:
        self.dry_run = dry_run
        self.host = host

    def dispatch(
        self,
        spec: DispatchSpec,
        dataset: str,
        seed: int,
        cohort: str,
        preset: str,
    ) -> ResultRow:
        started = datetime.now(UTC).isoformat()
        if spec.not_runnable_in_production:
            ended = datetime.now(UTC).isoformat()
            return ResultRow(
                preset=preset,
                cohort=cohort,
                dataset=dataset,
                canonical_name=spec.canonical_name,
                model_class=spec.model_class,
                module=spec.module,
                config_template=spec.config_template,
                seed=seed,
                status="skipped",
                error_message="not_runnable_in_production: registry flag set",
                started_at=started,
                ended_at=ended,
                host=self.host,
                notes="paper-only / archival evidence row; harness refuses dispatch.",
            )
        if self.dry_run:
            ended = datetime.now(UTC).isoformat()
            return ResultRow(
                preset=preset,
                cohort=cohort,
                dataset=dataset,
                canonical_name=spec.canonical_name,
                model_class=spec.model_class,
                module=spec.module,
                config_template=spec.config_template,
                seed=seed,
                status="dry_run",
                started_at=started,
                ended_at=ended,
                host=self.host,
                notes="dry-run: dispatcher not invoked; use the production harness for real runs.",
            )

        # Production fit/predict path (D-C-006 second half, PROVISIONAL).
        # Each guarded step returns a `_failed_row` with an explicit
        # `error_message` if it cannot proceed; production is never silently
        # corrupt.
        if yaml is None:
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error="dispatch_missing_pyyaml: install PyYAML",
            )
        config_path = (BENCH.parent / spec.config_template).resolve() if spec.config_template else None
        if config_path is None or not config_path.exists():
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"dispatch_missing_config_template: {spec.config_template}",
            )
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - production boundary
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"dispatch_yaml_error: {type(exc).__name__}: {exc}",
            )
        if not isinstance(config, dict) or "model" not in config:
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error="dispatch_invalid_config: missing top-level `model` key",
            )

        for entry in (config.get("dispatch") or {}).get("pythonpath_prepend") or []:
            full = (BENCH.parent / str(entry)).resolve()
            if str(full) not in sys.path:
                sys.path.insert(0, str(full))

        try:
            bundle = load_dataset(dataset)
        except DatasetNotFoundError as exc:
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"dataset_files_missing: {exc}",
            )
        except Exception as exc:  # noqa: BLE001 - production boundary
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"dataset_load_error: {type(exc).__name__}: {exc}",
            )

        try:
            estimator = _build_estimator(config, seed=seed)
        except Exception as exc:  # noqa: BLE001 - production boundary
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"build_error: {type(exc).__name__}: {exc}",
            )

        timeout_s = (config.get("dispatch") or {}).get("timeout_s")
        try:
            timeout_s = float(timeout_s) if timeout_s is not None else None
        except (TypeError, ValueError):
            timeout_s = None

        fit_start = time.perf_counter()
        try:
            _run_with_optional_timeout(
                estimator.fit, bundle.X_train, bundle.y_train,
                timeout_s=timeout_s,
            )
            fit_time = time.perf_counter() - fit_start
        except concurrent.futures.TimeoutError:
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"timeout_{int(timeout_s or 0)}s: fit exceeded dispatch.timeout_s budget",
                status="failed_terminal",
            )
        except Exception as exc:  # noqa: BLE001 - production boundary
            error_msg, status = _classify_fit_exception(exc)
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=error_msg, status=status,
            )

        predict_start = time.perf_counter()
        try:
            y_pred = _run_with_optional_timeout(
                estimator.predict, bundle.X_test,
                timeout_s=timeout_s,
            )
            predict_time = time.perf_counter() - predict_start
        except concurrent.futures.TimeoutError:
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"timeout_{int(timeout_s or 0)}s: predict exceeded dispatch.timeout_s budget",
                status="failed_terminal",
            )
        except Exception as exc:  # noqa: BLE001 - production boundary
            error_msg, status = _classify_fit_exception(exc)
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=error_msg.replace("fit_error", "predict_error"),
                status=status,
            )

        try:
            import numpy as np
            y_test = np.asarray(bundle.y_test).reshape(-1)
            y_pred = np.asarray(y_pred).reshape(-1)
            rmsep = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
            mae = float(np.mean(np.abs(y_test - y_pred)))
            ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
            ss_res = float(np.sum((y_test - y_pred) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        except Exception as exc:  # noqa: BLE001 - production boundary
            return _failed_row(
                spec, dataset, seed, cohort, preset, self.host, started,
                error=f"score_error: {type(exc).__name__}: {exc}",
            )

        ended = datetime.now(UTC).isoformat()
        return ResultRow(
            preset=preset,
            cohort=cohort,
            dataset=dataset,
            canonical_name=spec.canonical_name,
            model_class=spec.model_class,
            module=spec.module,
            config_template=spec.config_template,
            seed=seed,
            status="ok",
            n_train=bundle.n_train,
            n_test=bundle.n_test,
            n_features=bundle.n_features,
            rmsep=rmsep,
            mae=mae,
            r2=r2,
            score_metric="rmsep",
            score_value=rmsep,
            lower_is_better=True,
            fit_time_s=fit_time,
            predict_time_s=predict_time,
            started_at=started,
            ended_at=ended,
            host=self.host,
            notes="dispatch: provisional production fit/predict (D-C-006).",
        )

    def probe(
        self,
        spec: DispatchSpec,
        cohort: str,
        preset: str,
    ) -> ResultRow:
        """Probe mode — load config_template, apply pythonpath_prepend, import
        the module, resolve the class. Never call fit / predict. Used to
        validate the dispatch contract end-to-end before production
        hardening lands.
        """
        started = datetime.now(UTC).isoformat()
        if spec.not_runnable_in_production:
            ended = datetime.now(UTC).isoformat()
            return ResultRow(
                preset=preset,
                cohort=cohort,
                dataset="",
                canonical_name=spec.canonical_name,
                model_class=spec.model_class,
                module=spec.module,
                config_template=spec.config_template,
                seed=0,
                status="skipped",
                error_message="not_runnable_in_production: registry flag set",
                started_at=started,
                ended_at=ended,
                host=self.host,
                notes="probe skipped: paper-only / archival evidence row.",
            )

        if yaml is None:
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error="probe_missing_pyyaml: install PyYAML to use --probe",
            )

        config_path = (BENCH.parent / spec.config_template).resolve() if spec.config_template else None
        if config_path is None or not config_path.exists():
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error=f"probe_missing_config_template: {spec.config_template}",
            )

        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - probe boundary
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error=f"probe_yaml_error: {type(exc).__name__}: {exc}",
            )

        if not isinstance(config, dict) or "model" not in config:
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error="probe_invalid_config: missing top-level `model` key",
            )

        cn_in_config = str(config.get("canonical_name", ""))
        if cn_in_config and cn_in_config != spec.canonical_name:
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error=f"probe_canonical_name_mismatch: config={cn_in_config!r} registry={spec.canonical_name!r}",
            )

        dispatch_block = config.get("dispatch") or {}
        prepends = dispatch_block.get("pythonpath_prepend") or []
        prepended: list[str] = []
        for entry in prepends:
            full = (BENCH.parent / str(entry)).resolve()
            if str(full) not in sys.path:
                sys.path.insert(0, str(full))
                prepended.append(str(full))

        try:
            mod = importlib.import_module(spec.module)
        except Exception as exc:  # noqa: BLE001 - probe boundary
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error=f"probe_import_error: {type(exc).__name__}: {exc}",
            )

        cls = getattr(mod, spec.model_class, None)
        if cls is None:
            return _failed_row(
                spec, "", 0, cohort, preset, self.host, started,
                error=f"probe_missing_class: {spec.module}.{spec.model_class}",
            )

        protocol = (config.get("hyperparameter_search") or {}).get("protocol", "kfold")
        ended = datetime.now(UTC).isoformat()
        notes = (
            f"probe ok: config={spec.config_template} "
            f"protocol={protocol} class={cls.__module__}.{cls.__name__} "
            f"prepended={len(prepended)}"
        )
        return ResultRow(
            preset=preset,
            cohort=cohort,
            dataset="",
            canonical_name=spec.canonical_name,
            model_class=spec.model_class,
            module=spec.module,
            config_template=spec.config_template,
            seed=0,
            status="probe",
            started_at=started,
            ended_at=ended,
            host=self.host,
            notes=notes,
        )


def _failed_row(
    spec: DispatchSpec,
    dataset: str,
    seed: int,
    cohort: str,
    preset: str,
    host: str,
    started_at: str,
    *,
    error: str,
    ended_at: str | None = None,
    status: str = "failed",
) -> ResultRow:
    """Construct a failure row.

    `status` defaults to ``"failed"`` (retriable on resume — the user can fix
    the cause and re-run). Pass ``status="failed_terminal"`` for D-C-018
    Prong A/B/C terminal failures (timeout, OOM-kill, worker-crash) that
    must NOT be retried automatically (per Codex D-C-018 verdict Q4).
    """
    return ResultRow(
        preset=preset,
        cohort=cohort,
        dataset=dataset,
        canonical_name=spec.canonical_name,
        model_class=spec.model_class,
        module=spec.module,
        config_template=spec.config_template,
        seed=seed,
        status=status,
        error_message=error,
        started_at=started_at,
        ended_at=ended_at or datetime.now(UTC).isoformat(),
        host=host,
    )


def _resolve_dotted(path: str) -> Any:
    """Resolve `pkg.sub.ClassName` to the actual class object."""
    if not path:
        raise ValueError("empty class path")
    module_name, _, attr_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"unqualified class path {path!r}; need module prefix")
    mod = importlib.import_module(module_name)
    obj = getattr(mod, attr_name, None)
    if obj is None:
        raise AttributeError(f"{module_name!r} has no attribute {attr_name!r}")
    return obj


def _materialize_value(value: Any) -> Any:
    """Recursively materialise class-spec dicts inside a YAML config value.

    A class-spec dict is `{"class": "<dotted>", "params": {...}}`; it
    becomes `cls(**materialised_params)`. When the dict also carries a
    `"name"` key (e.g. items inside a `FeatureUnion.transformer_list`),
    the result is the `(name, instance)` tuple sklearn meta-estimators
    expect. Lists / tuples are walked element-wise so nested specs work
    without special-casing each meta-estimator.
    """
    if isinstance(value, dict) and isinstance(value.get("class"), str):
        cls = _resolve_dotted(value["class"])
        sub_params = value.get("params") or {}
        if not isinstance(sub_params, dict):
            raise ValueError(f"nested class-spec `params` must be a dict, got {type(sub_params).__name__}")
        materialised = {k: _materialize_value(v) for k, v in sub_params.items()}
        instance = cls(**materialised)
        if "name" in value:
            return (str(value["name"]), instance)
        return instance
    if isinstance(value, list):
        return [_materialize_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _materialize_value(v) for k, v in value.items()}
    return value


def _build_step(step: Any) -> tuple[str, Any]:
    """Build one (name, instance) tuple from a config_template `pipeline` entry.

    Accepted shapes:
      * `{class: <dotted>, params: {...}}`           — generic transformer.
      * `{model: {class: <dotted>, params: {...}}}`  — terminal model step.

    `params` is recursively materialised so meta-estimators that take
    estimator instances (FeatureUnion, ColumnTransformer, etc.) can be
    expressed as YAML.
    """
    if not isinstance(step, dict):
        raise ValueError(f"pipeline step must be a dict, got {type(step).__name__}: {step!r}")
    if "model" in step:
        spec_dict = step["model"]
        is_model = True
    else:
        spec_dict = step
        is_model = False
    if not isinstance(spec_dict, dict):
        raise ValueError(f"pipeline step body must be a dict, got {type(spec_dict).__name__}")
    cls_path = spec_dict.get("class")
    if not isinstance(cls_path, str):
        raise ValueError(f"pipeline step needs `class: <dotted>` string, got {cls_path!r}")
    params = spec_dict.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"pipeline step `params` must be a dict, got {type(params).__name__}")
    cls = _resolve_dotted(cls_path)
    materialised_params = {k: _materialize_value(v) for k, v in params.items()}
    instance = cls(**materialised_params)
    name = ("model" if is_model else cls.__name__.lower())
    return (name, instance)


def _inject_seed_recursive(obj: Any, seed: int) -> None:
    """Override `random_state` on `obj` and any nested `(name, estimator)`
    tuples (D-C-018 Prong D).

    For `protocol == "model_native"` the runtime `seed` must propagate into
    the estimator's `random_state` for the multi-seed protocol to be
    meaningful. The harness materialises YAML params literally; without this
    helper the YAML-baked `random_state: 0` would be reused across all seeds.

    Walks `atoms` and `light_atoms` attributes (per AdaptiveSuperLearner
    convention from `bench/AOM_v0/multiview/multiview/super_learner.py`).
    Tuples in those attrs follow the `(name, estimator)` shape produced by
    `_materialize_value`'s name-keyed convention.
    """
    import contextlib
    if hasattr(obj, "set_params"):
        # estimator may not accept random_state, or set_params may not be
        # sklearn-compatible — fall through to direct attribute set.
        with contextlib.suppress(ValueError, TypeError, AttributeError):
            obj.set_params(random_state=seed)
    if hasattr(obj, "random_state"):
        with contextlib.suppress(AttributeError, TypeError):
            obj.random_state = seed
    for attr in ("atoms", "light_atoms"):
        items = getattr(obj, attr, None)
        if items is None:
            continue
        try:
            iterable = list(items)
        except TypeError:
            continue
        for item in iterable:
            if isinstance(item, tuple) and len(item) == 2:
                _inject_seed_recursive(item[1], seed)
            elif hasattr(item, "set_params") or hasattr(item, "random_state"):
                _inject_seed_recursive(item, seed)


_TERMINAL_FAILURE_PREFIXES = ("timeout_", "oom_kill", "worker_terminated")


def _classify_fit_exception(exc: BaseException) -> tuple[str, str]:
    """Classify a fit/predict exception as terminal vs retriable.

    Returns ``(error_message, status)`` where status is one of:
      * ``"failed"`` — retriable on resume (build_error / dataset_load /
        score_error / generic fit_error from a bug in the model).
      * ``"failed_terminal"`` — final, no retry on resume (timeout, OOM,
        worker process kill). D-C-018 Prong B/C: these are infrastructure
        failures, retry would just burn CPU.

    Pattern-matches the exception type name (so we don't have to import
    `joblib.externals.loky.process_executor.TerminatedWorkerError`
    conditionally) and returncode markers from subprocess crashes.
    """
    cls = type(exc).__name__
    msg = str(exc) or cls
    lowered = (cls + " " + msg).lower()
    if "terminatedworker" in lowered or "brokenprocesspool" in lowered:
        return (f"oom_kill_or_worker_terminated: {cls}: {msg}", "failed_terminal")
    if "memoryerror" in lowered:
        return (f"oom_kill_local: {cls}: {msg}", "failed_terminal")
    return (f"fit_error: {cls}: {msg}", "failed")


def _run_with_optional_timeout(
    func: Any,
    *args: Any,
    timeout_s: float | None,
    **kwargs: Any,
) -> Any:
    """Run ``func(*args, **kwargs)`` either inline (no timeout) or under a
    ThreadPoolExecutor with a wall-clock budget (D-C-018 Prong A).

    On ``TimeoutError`` the caller is responsible for writing a
    ``failed_terminal`` row. The thread itself is not killed (Python has no
    safe way to kill a CPU-bound thread); the dispatch loop simply moves on
    and the leaked thread completes its fit silently in the background.
    Acceptable trade-off per Codex D-C-018 verdict ; preferable to silent
    blocking of the cohort. Per Codex Q3: fail-open when ``timeout_s`` is
    ``None`` or non-positive (current 23 YAML configs vary in declared
    values, some null).

    NOTE: do NOT use ``with concurrent.futures.ThreadPoolExecutor(...) as
    executor`` — the context manager calls ``executor.shutdown(wait=True)``
    on exit, which BLOCKS until the runaway worker thread completes,
    making the timeout decorative. Use ``shutdown(wait=False)`` so the
    dispatch loop moves on immediately while the leaked thread runs in
    background. Discovered 2026-05-09 19:55 CEST in Phase 2 fast_reliable
    LMA fit: 1200s budget, fit ran 35+ min before being killed externally
    (RSS 42 GB).
    """
    if not timeout_s or timeout_s <= 0:
        return func(*args, **kwargs)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=float(timeout_s))
    finally:
        executor.shutdown(wait=False)


def _build_estimator(config: dict[str, Any], *, seed: int) -> Any:
    """Build a fitted-able estimator from a parsed config_template.

    Produces, in order of preference:
      * a single estimator if the pipeline has exactly one step (typically
        the `model_native` pattern for AOM-PLS / AOMRidge);
      * a `sklearn.pipeline.Pipeline` for multi-step sklearn pipelines;
      * a `sklearn.model_selection.GridSearchCV` wrapping the above when
        `hyperparameter_search.protocol == "kfold"` with a non-empty
        `param_grid` — preserving nested CV (D-C-010a).

    For `protocol == "model_native"`, the runtime `seed` is recursively
    injected into the estimator's and any nested-atom's `random_state`
    (D-C-018 Prong D).
    """
    pipeline_block = (config.get("model") or {}).get("pipeline")
    if not isinstance(pipeline_block, list) or not pipeline_block:
        raise ValueError("config `model.pipeline` must be a non-empty list")

    steps = [_build_step(step) for step in pipeline_block]
    if len(steps) == 1:
        base = steps[0][1]
    else:
        from sklearn.pipeline import Pipeline
        base = Pipeline(steps)

    hps = config.get("hyperparameter_search") or {}
    protocol = (hps.get("protocol") or "kfold").lower()
    param_grid = hps.get("param_grid")
    if protocol == "model_native" or not param_grid:
        _inject_seed_recursive(base, seed)
        return base

    if protocol != "kfold":
        raise ValueError(f"unsupported hyperparameter_search.protocol={protocol!r}")

    from sklearn.model_selection import GridSearchCV, KFold
    n_splits = int(hps.get("n_splits") or 5)
    shuffle = bool(hps.get("shuffle", True))
    random_state = seed + int(hps.get("random_state_offset") or 0)
    cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state if shuffle else None)
    if isinstance(base, type(steps[0][1])) and len(steps) == 1:
        # Single-step: rename grid keys to match the bare estimator
        prefixed_grid = param_grid
    else:
        # Multi-step: prefix each grid key with the model step name (last step is "model")
        last_step_name = steps[-1][0]
        prefixed_grid = {f"{last_step_name}__{k}": v for k, v in param_grid.items()}
    scoring = hps.get("scoring") or "neg_root_mean_squared_error"
    refit = bool(hps.get("refit", True))
    return GridSearchCV(
        estimator=base,
        param_grid=prefixed_grid,
        cv=cv,
        scoring=scoring,
        refit=refit,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def stats_wilcoxon(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Two-sided paired Wilcoxon signed-rank.

    Lazy import of scipy.stats so that callers without scipy still get a
    helpful error message. Returns a payload suitable for JSON.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"error": "scipy_not_installed", "n_pairs": len(x)}
    if len(x) != len(y) or len(x) < 6:
        return {"error": "insufficient_pairs", "n_pairs": len(x)}
    statistic, pvalue = wilcoxon(x, y, alternative="two-sided")
    return {"statistic": float(statistic), "p_value": float(pvalue), "n_pairs": len(x)}


def stats_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI of the median delta."""
    if not deltas:
        return {"error": "empty"}
    rng = random.Random(seed)
    n = len(deltas)
    medians: list[float] = []
    sample = list(deltas)
    for _ in range(n_bootstrap):
        draw = [sample[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(draw))
    medians.sort()
    lo_idx = max(0, int((1 - confidence) / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, int((1 - (1 - confidence) / 2) * n_bootstrap))
    return {
        "median_delta": statistics.median(deltas),
        "ci_low": medians[lo_idx],
        "ci_high": medians[hi_idx],
        "n_pairs": n,
    }


def stats_friedman_nemenyi(per_dataset_scores: dict[str, list[float]], model_names: Sequence[str]) -> dict[str, Any]:
    """Friedman + Nemenyi post-hoc on dataset-by-model rank matrix.

    `per_dataset_scores` maps dataset -> list of scores aligned to
    `model_names`. Lower-is-better. Returns chi-square statistic, p-value,
    and the critical-difference (CD) used by the Nemenyi post-hoc plot.
    """
    try:
        from scipy.stats import friedmanchisquare
    except ImportError:
        return {"error": "scipy_not_installed"}
    matrices = [scores for scores in per_dataset_scores.values() if len(scores) == len(model_names)]
    if len(matrices) < 5:
        return {"error": "insufficient_datasets", "n_datasets": len(matrices)}
    columns = list(zip(*matrices, strict=False))
    statistic, p_value = friedmanchisquare(*columns)
    n_datasets = len(matrices)
    k = len(model_names)
    q_alpha = 2.728  # Nemenyi q for k <= 5 alpha=0.05; conservative skeleton constant
    cd = q_alpha * math.sqrt(k * (k + 1) / (6 * n_datasets))
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_datasets": n_datasets,
        "n_models": k,
        "critical_difference": cd,
    }


def stats_nadeau_bengio(
    paired: Sequence[tuple[float, float]],
    *,
    n_train: int,
    n_test: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Nadeau-Bengio corrected paired t-test.

    For repeated K-fold CV, the standard paired t over per-fold deltas
    underestimates variance because of overlapping training sets.
    Nadeau-Bengio multiplies the variance by (1/k + n_test/n_train).
    Returns the corrected t statistic, p-value, and CI.
    """
    try:
        from scipy.stats import t as student_t
    except ImportError:
        return {"error": "scipy_not_installed"}
    if not paired:
        return {"error": "empty"}
    deltas = [a - b for a, b in paired]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    var = statistics.pvariance(deltas) if n > 1 else 0.0
    correction = 1 / n + (n_test / n_train if n_train > 0 else 0.0)
    corrected_var = var * correction
    if corrected_var <= 0:
        return {"error": "non_positive_variance", "n_pairs": n}
    se = math.sqrt(corrected_var)
    t_stat = mean / se if se > 0 else 0.0
    df = max(1, n - 1)
    p = 2 * (1 - float(student_t.cdf(abs(t_stat), df=df)))
    half = float(student_t.ppf((1 + confidence) / 2, df=df)) * se
    return {
        "n_pairs": n,
        "mean_delta": mean,
        "t_statistic": t_stat,
        "p_value": p,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "correction_factor": correction,
    }


# ---------------------------------------------------------------------------
# Run driver
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench/harness/run_benchmark.py",
        description="Resumable benchmark harness for nirs4all bench presets.",
    )
    parser.add_argument("--cohort", required=True, help="Cohort name or comma-separated dataset list.")
    parser.add_argument("--pipeline", required=True, type=Path, help="Path to a scenario manifest JSON produced by export_benchmark_scenarios.py.")
    parser.add_argument("--workspace", required=True, type=Path, help="Output directory; harness writes results.csv and stats.json under it.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seed list.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Reserved for parallel dispatch; unused in the skeleton.")
    parser.add_argument("--dry-run", action="store_true", help="Skip module import and emit `dry_run` rows for every (dataset, seed, model).")
    parser.add_argument("--max-models", type=int, default=None, help="Optional cap on number of candidate models.")
    parser.add_argument("--max-datasets", type=int, default=None, help="Optional cap on number of datasets.")
    parser.add_argument("--no-resume", action="store_true", help="Truncate existing results.csv before running.")
    parser.add_argument("--stats", action="store_true", help="Compute summary stats over completed rows after the run.")
    parser.add_argument(
        "--probe",
        default=None,
        help=(
            "Probe a single model by canonical_name: load its config_template, "
            "honour pythonpath_prepend, import the module, resolve the class. "
            "Never fits or predicts. Useful to validate the dispatch contract "
            "before production hardening lands."
        ),
    )
    return parser.parse_args(argv)


def _run_probe(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    preset: str,
    candidates: list[dict[str, Any]],
) -> int:
    """Probe a single registry entry by canonical_name. No CSV is written;
    the result row is printed as JSON.
    """
    target = args.probe
    candidate = next((c for c in candidates if c.get("canonical_name") == target), None)
    if candidate is None:
        names = sorted({c.get("canonical_name", "") for c in candidates})
        print(
            f"[probe] canonical_name={target!r} not found in manifest "
            f"({len(names)} candidates); did you mean one of: {names[:5]}{'…' if len(names) > 5 else ''}",
            file=sys.stderr,
        )
        return 2
    spec = DispatchSpec(
        canonical_name=candidate.get("canonical_name", ""),
        model_class=candidate.get("model_class", ""),
        module=candidate.get("module", ""),
        config_template=candidate.get("config_template", ""),
        inputs=candidate.get("input_constraints", {}),
        not_runnable_in_production=bool(candidate.get("not_runnable_in_production", False)),
    )
    dispatcher = ModelDispatcher(dry_run=False, host=os.uname().nodename)
    row = dispatcher.probe(spec=spec, cohort=args.cohort, preset=preset)
    print(json.dumps(row.to_dict(), indent=2, ensure_ascii=False))
    return 0 if row.status == "probe" else 1


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.pipeline)
    preset = manifest.get("preset", "")
    candidates = manifest.get("candidates", [])
    if args.max_models:
        candidates = candidates[: args.max_models]

    if args.probe:
        return _run_probe(args, manifest, preset, candidates)

    datasets = resolve_cohort(args.cohort)
    if args.max_datasets:
        datasets = datasets[: args.max_datasets]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    completed = set() if args.no_resume else load_completed(args.workspace)
    handle, writer = open_results_writer(args.workspace, append=not args.no_resume)
    dispatcher = ModelDispatcher(dry_run=args.dry_run, host=os.uname().nodename)

    n_planned = len(datasets) * len(seeds) * len(candidates)
    n_skipped_resume = 0
    n_skipped_not_runnable = 0
    n_run = 0
    n_failed = 0
    n_failed_terminal = 0

    try:
        for candidate in candidates:
            spec = DispatchSpec(
                canonical_name=candidate.get("canonical_name", ""),
                model_class=candidate.get("model_class", ""),
                module=candidate.get("module", ""),
                config_template=candidate.get("config_template", ""),
                inputs=candidate.get("input_constraints", {}),
                not_runnable_in_production=bool(candidate.get("not_runnable_in_production", False)),
            )
            for seed in seeds:
                for dataset in datasets:
                    key = (dataset, seed, spec.canonical_name, "default")
                    if key in completed:
                        n_skipped_resume += 1
                        continue
                    started = time.time()
                    row = dispatcher.dispatch(
                        spec=spec,
                        dataset=dataset,
                        seed=seed,
                        cohort=args.cohort,
                        preset=preset,
                    )
                    elapsed = time.time() - started
                    if row.fit_time_s is None:
                        row.fit_time_s = elapsed
                    writer.writerow(row.to_dict())
                    handle.flush()
                    if row.status == "failed":
                        n_failed += 1
                    elif row.status == "failed_terminal":
                        n_failed_terminal += 1
                    elif row.status in {"ok", "dry_run"}:
                        n_run += 1
                    elif row.status == "skipped":
                        n_skipped_not_runnable += 1
    finally:
        handle.close()

    print(
        f"[harness] preset={preset} cohort={args.cohort} "
        f"planned={n_planned} run={n_run} "
        f"skipped(resume)={n_skipped_resume} skipped(not_runnable)={n_skipped_not_runnable} "
        f"failed={n_failed} failed_terminal={n_failed_terminal}"
    )
    if args.stats:
        write_stats(args.workspace)
    return 1 if (n_failed or n_failed_terminal) and not args.dry_run else 0


def write_stats(workspace: Path) -> None:
    """Compute high-level paired stats across completed rows.

    Builds a (dataset, model) -> score table from results.csv, then runs
    Wilcoxon paired vs the best PLS row per dataset, plus a Friedman test
    across all models with full coverage.
    """
    results = workspace / "results.csv"
    if not results.exists():
        print("[harness] No results.csv yet; skipping stats.")
        return
    by_dataset_model: dict[str, dict[str, float]] = defaultdict(dict)
    with results.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            score = row.get("score_value")
            if not score:
                continue
            try:
                value = float(score)
            except ValueError:
                continue
            ds = (row.get("dataset") or "").strip()
            model = (row.get("canonical_name") or "").strip()
            if ds and model:
                by_dataset_model[ds][model] = value

    if not by_dataset_model:
        print("[harness] No completed rows with score_value; nothing to summarise.")
        return

    pls_per_dataset: dict[str, float] = {}
    for ds, scores in by_dataset_model.items():
        pls_score = next((v for k, v in scores.items() if "PLS" in k.upper()), None)
        if pls_score is not None:
            pls_per_dataset[ds] = pls_score

    payload: dict[str, Any] = {"per_model_vs_pls_wilcoxon": {}, "friedman": None}
    models: set[str] = {m for scores in by_dataset_model.values() for m in scores}
    for model in sorted(models):
        deltas: list[float] = []
        for ds, scores in by_dataset_model.items():
            if model not in scores or ds not in pls_per_dataset:
                continue
            deltas.append(scores[model] - pls_per_dataset[ds])
        payload["per_model_vs_pls_wilcoxon"][model] = {
            "wilcoxon": stats_wilcoxon([0.0] * len(deltas), deltas),
            "bootstrap_ci": stats_bootstrap_ci(deltas, n_bootstrap=1000),
        }

    common_models = sorted(
        m
        for m in models
        if all(m in scores for scores in by_dataset_model.values())
    )
    if common_models and by_dataset_model:
        per_dataset = {ds: [scores[m] for m in common_models] for ds, scores in by_dataset_model.items()}
        payload["friedman"] = stats_friedman_nemenyi(per_dataset, common_models)

    out = workspace / "stats.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[harness] Wrote {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
