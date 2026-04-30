"""Phase 0 baseline benchmark runner — Ridge, PLS, nicon, decon.

Usage::

    PYTHONPATH=bench/nicon_v2:bench/AOM_v0:bench/AOM_v0/Ridge \\
      python bench/nicon_v2/benchmarks/run_baseline_benchmark.py \\
        --workspace bench/nicon_v2/benchmark_runs/smoke \\
        --cohort smoke --variants smoke --seed 0

Resumability: rows already present with ``status="OK"`` are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make the package importable when run as a script from the repo root.
_HERE = Path(__file__).resolve()
_BENCH_ROOT = _HERE.parent.parent          # bench/nicon_v2
for path in (str(_BENCH_ROOT), str(_BENCH_ROOT.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402  (after sys.path setup)

from nicon_v2 import CODE_VERSION  # noqa: E402
from nicon_v2.datasets import (  # noqa: E402
    DatasetSpec,
    EXTENDED_SMOKE_DATASETS,
    SMOKE_DATASETS,
    load_cohort_manifest,
    load_dataset,
)
from nicon_v2.metrics import bias as bias_metric  # noqa: E402
from nicon_v2.metrics import mae, r2, relative_rmsep, rmse  # noqa: E402
from nicon_v2.models.baseline import (  # noqa: E402
    PLSBaseline,
    RidgeBaseline,
    build_decon_torch,
    build_nicon_torch,
    count_parameters,
    cuda_peak_mb,
)
from nicon_v2.models.searched_baseline import SearchedPLS, SearchedRidge  # noqa: E402
from nicon_v2.augmentation import (  # noqa: E402
    AugmentationPlan,
    BjerrumConfig,
    CMixupConfig,
)
from nicon_v2.models.v1a_minimal_repair import (  # noqa: E402
    build_nicon_v1a,
    build_nicon_v1a_activation_only,
    build_nicon_v1a_head_only,
)
from nicon_v2.models.v1b_concat_aug import build_nicon_v1b  # noqa: E402
from nicon_v2.models.v1c_gap_backbone import build_nicon_v1c  # noqa: E402
from nicon_v2.models.v2_aom_cnn import build_nicon_v2a  # noqa: E402
from nicon_v2.models.stacking import StackedRegressor, StackingConfig  # noqa: E402
from nicon_v2.training import (  # noqa: E402
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    pick_device,
    predict_torch_regressor,
    set_global_seed,
    train_torch_regressor,
)


RESULT_COLUMNS = [
    "dataset_group",
    "dataset",
    "task",
    "n_train",
    "n_test",
    "n_features",
    "variant",
    "model_version",
    "seed",
    "cv_fold",
    "cv_protocol",
    "status",
    "error_message",
    "rmsep",
    "mae",
    "r2",
    "bias",
    "ref_rmse_pls",
    "ref_rmse_paper_ridge",
    "ref_rmse_tabpfn_raw",
    "ref_rmse_tabpfn_opt",
    "ref_rmse_cnn",
    "ref_rmse_catboost",
    "ref_rmse_aom_ridge_curated_best",
    "relative_rmsep_vs_pls",
    "relative_rmsep_vs_paper_ridge",
    "relative_rmsep_vs_tabpfn_raw",
    "relative_rmsep_vs_tabpfn_opt",
    "relative_rmsep_vs_cnn",
    "relative_rmsep_vs_aom_ridge_curated_best",
    "fit_time_s",
    "predict_time_s",
    "total_params",
    "peak_vram_mb",
    "hyperparams_json",
    "python_version",
    "torch_version",
    "cuda_version",
    "git_sha",
    "host",
]


@dataclass
class Variant:
    label: str
    family: str        # "ridge" | "pls" | "nicon" | "decon"
    extra: dict = field(default_factory=dict)


SMOKE_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NICON-baseline", family="nicon"),
    Variant("DECON-baseline", family="decon"),
)

PHASE1A_VARIANTS: tuple[Variant, ...] = SMOKE_VARIANTS + (
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("NiconV1a-activation-only", family="nicon_v1a_act"),
    Variant("NiconV1a-baseline", family="nicon_v1a"),
)

# Phase 1b ablation: V1a-head-only baseline + concat-deriv only + concat+Bjerrum + concat+Bjerrum+C-Mixup.
PHASE1B_VARIANTS: tuple[Variant, ...] = (
    # Phase 1a accepted control:
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Phase 1b ablation cells:
    Variant("NiconV1b-concat-only",     family="nicon_v1b", extra={"bjerrum": False, "cmixup": False, "vanilla_mixup": False}),
    Variant("NiconV1b-concat-bjerrum",  family="nicon_v1b", extra={"bjerrum": True,  "cmixup": False, "vanilla_mixup": False}),
    Variant("NiconV1b-concat-cmixup",   family="nicon_v1b", extra={"bjerrum": True,  "cmixup": True,  "vanilla_mixup": False}),
    Variant("NiconV1b-concat-mixup",    family="nicon_v1b", extra={"bjerrum": True,  "cmixup": False, "vanilla_mixup": True}),
)

# Phase 5 / H12 stacking: V1c + PLS via Ridge meta.
PHASE_STACK_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": False, "vanilla_mixup": False}),
    Variant("Stack-Ridge-PLS",        family="stack",
            extra={"_base": ("ridge", "pls")}),
    Variant("Stack-Ridge-PLS-V1c",    family="stack",
            extra={"_base": ("ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
    Variant("Stack-Ridge-PLS-V1aHead",family="stack",
            extra={"_base": ("ridge", "pls", "v1a_head")}),
)

# Phase 5b: stacks using AOM-Ridge (84 % wins vs paper Ridge) as a base / instead of Ridge.
PHASE_STACK_AOM_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("AOMRidge-base", family="stack", extra={"_base": ("aom_ridge",)}),
    Variant("Stack-AOMRidge-PLS",     family="stack", extra={"_base": ("aom_ridge", "pls")}),
    Variant("Stack-AOMRidge-PLS-V1c", family="stack",
            extra={"_base": ("aom_ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
    Variant("Stack-AOMRidge-Ridge-PLS-V1c", family="stack",
            extra={"_base": ("aom_ridge", "ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
)

# Phase 2 / round 5: AOM-superblock CNN (NiconV2A) ablation set.
PHASE_V2A_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NICON-baseline", family="nicon"),
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True}),
    # V2A — bare strict-linear AOM bank, frozen ops:
    Variant("V2A-compact-frozen",  family="nicon_v2a", extra={"bank": "compact",  "trainable_ops": False}),
    Variant("V2A-extended-frozen", family="nicon_v2a", extra={"bank": "extended", "trainable_ops": False}),
    # V2A + Bjerrum aug (pre-branch, per Codex F5):
    Variant("V2A-extended-frozen-bjerrum", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": False, "bjerrum": True}),
    # V2A + CNN-only extras (SNV/MSC/Gaussian) + Bjerrum aug:
    Variant("V2A-full-frozen-bjerrum", family="nicon_v2a",
            extra={"bank": "full", "trainable_ops": False, "bjerrum": True}),
    # V2B — learnable operators with L2-from-init regulariser:
    Variant("V2B-extended-trainable", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True}),
)


# Phase 1d: cartesian-preprocessing-search Ridge / PLS (mirror tabpfn paper baseline).
PHASE_SEARCHED_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("SearchedRidge", family="searched_ridge"),
    Variant("SearchedPLS",   family="searched_pls"),
    Variant("Stack-SearchedRidge-SearchedPLS", family="stack",
            extra={"_base": ("searched_ridge", "searched_pls")}),
    Variant("Stack-SearchedRidge-SearchedPLS-V1c", family="stack",
            extra={"_base": ("searched_ridge", "searched_pls", "v1c"), "_cnn_use_bjerrum": True}),
)


# Phase 1c: GAP backbone alone, GAP+concat-deriv, GAP+concat+Bjerrum+C-Mixup, plus norm A/B (LayerNorm/BatchNorm/GroupNorm).
PHASE1C_VARIANTS: tuple[Variant, ...] = (
    # Anchor controls:
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("NICON-baseline", family="nicon"),
    # Phase 1c GAP backbone — bare:
    Variant("NiconV1c-bare-LN", family="nicon_v1c", extra={"norm": "layer", "use_concat_derivatives": False}),
    Variant("NiconV1c-bare-BN", family="nicon_v1c", extra={"norm": "batch", "use_concat_derivatives": False}),
    Variant("NiconV1c-bare-GN", family="nicon_v1c", extra={"norm": "group", "use_concat_derivatives": False}),
    # GAP + concat-deriv (Phase 1b H5 lifted onto the new backbone):
    Variant("NiconV1c-concat", family="nicon_v1c", extra={"norm": "layer", "use_concat_derivatives": True}),
    # GAP + concat + Bjerrum:
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": False, "vanilla_mixup": False}),
    # GAP + concat + Bjerrum + C-Mixup (the kitchen sink):
    Variant("NiconV1c-concat-bjerrum-cmixup", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": True, "vanilla_mixup": False}),
)


# ---------------------------------------------------------------------------
# Result storage helpers
# ---------------------------------------------------------------------------


def write_predictions_parquet(
    workspace: Path,
    variant: str,
    dataset: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path | None:
    """Write per-sample predictions to a parquet (Codex F14).

    Falls back silently to CSV if pyarrow is missing.
    """
    pred_dir = workspace / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "_")
    safe_variant = variant.replace("/", "_")
    parquet_path = pred_dir / f"{safe_variant}__{safe_dataset}__seed{seed}.parquet"
    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "fold": np.full(len(y_true), -1, dtype=np.int64),
            "seed": np.full(len(y_true), seed, dtype=np.int64),
            "variant": np.full(len(y_true), variant, dtype=object),
            "dataset": np.full(len(y_true), dataset, dtype=object),
        }
    )
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        # pyarrow may be missing; fall back to CSV with the same path stem.
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def existing_done_rows(csv_path: Path) -> set[tuple[str, str, int]]:
    """Return the set of ``(dataset, variant, seed)`` rows already finished with status=OK."""
    if not csv_path.is_file():
        return set()
    done: set[tuple[str, str, int]] = set()
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).upper() == "OK":
                try:
                    done.add(
                        (
                            str(row["dataset"]),
                            str(row["variant"]),
                            int(row["seed"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    return done


def open_result_writer(csv_path: Path) -> tuple[csv.DictWriter, "object"]:
    """Open the CSV in append mode, writing the header if file does not exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file() or csv_path.stat().st_size == 0
    fh = csv_path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        fh.flush()
    return writer, fh


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------


def _run_ridge(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = RidgeBaseline(seed=seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams, total_params=spec.n_features, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_stack(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int, extra: dict | None) -> dict:
    set_global_seed(seed)
    extra = extra or {}
    base = tuple(extra.get("_base", ("ridge", "pls", "v1c")))
    cfg = StackingConfig(
        base_learners=base,
        seed=seed,
        cnn_use_bjerrum=bool(extra.get("_cnn_use_bjerrum", True)),
    )
    model = StackedRegressor(cfg=cfg)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(
        spec, pred, y_test, fit_time, pred_time, model.hyperparams,
        total_params=len(base), peak_vram_mb=cuda_peak_mb() if torch.cuda.is_available() else float("nan"),
    )
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_pls(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = PLSBaseline(seed=seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams, total_params=model.selected_n_components_, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_searched_ridge(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = SearchedRidge(seed=seed)
    t0 = time.time(); model.fit(X_train, y_train); fit_time = time.time() - t0
    t0 = time.time(); pred = model.predict(X_test); pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams,
                      total_params=spec.n_features, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_searched_pls(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = SearchedPLS(seed=seed)
    t0 = time.time(); model.fit(X_train, y_train); fit_time = time.time() - t0
    t0 = time.time(); pred = model.predict(X_test); pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams,
                      total_params=model.selected_n_components_, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _build_aug_plan(extra: dict) -> AugmentationPlan | None:
    """Translate variant.extra options into an AugmentationPlan, or None if no aug."""
    bjerrum_enabled = bool(extra.get("bjerrum", False))
    cmixup_enabled = bool(extra.get("cmixup", False))
    vanilla_mixup = bool(extra.get("vanilla_mixup", False))
    if not (bjerrum_enabled or cmixup_enabled or vanilla_mixup):
        return None
    bjer_cfg = BjerrumConfig(enabled=bjerrum_enabled)
    cmix_cfg = CMixupConfig(
        enabled=(cmixup_enabled or vanilla_mixup),
        sigma_y=(float("inf") if vanilla_mixup else None),
    )
    return AugmentationPlan(bjerrum=bjer_cfg, cmixup=cmix_cfg)


def _run_torch_cnn(
    spec: DatasetSpec,
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
    family: str,
    extra_options: dict | None = None,
) -> dict:
    set_global_seed(seed)
    device = pick_device("auto")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    x_proc = StandardXProcessor().fit(X_train)
    y_proc = StandardYProcessor().fit(y_train)
    X_train_s = x_proc.transform(X_train)
    X_test_s = x_proc.transform(X_test)
    y_train_s = y_proc.transform(y_train)

    n_features = X_train_s.shape[1]
    builder_map = {
        "nicon": build_nicon_torch,
        "decon": build_decon_torch,
        "nicon_v1a": build_nicon_v1a,
        "nicon_v1a_head": build_nicon_v1a_head_only,
        "nicon_v1a_act": build_nicon_v1a_activation_only,
        "nicon_v1b": build_nicon_v1b,
        "nicon_v1c": build_nicon_v1c,
        "nicon_v2a": build_nicon_v2a,
    }
    builder = builder_map[family]
    builder_params = {k: v for k, v in (extra_options or {}).items() if not k.startswith("_")
                      and k not in ("bjerrum", "cmixup", "vanilla_mixup")}
    model = builder((1, n_features), params=builder_params).to(device)
    # NiconV2A needs branch-statistics fitting (MSC reference, RMSBranchNorm).
    if family == "nicon_v2a" and hasattr(model, "fit_branches"):
        model.fit_branches(torch.from_numpy(X_train_s.reshape(-1, 1, n_features)).float().to(device))
    config = TrainConfig(seed=seed, device=device.type, batch_size=min(32, max(8, X_train_s.shape[0] // 8)))

    # Phase 1b: optional augmentation hooks (Bjerrum / C-Mixup).
    if extra_options:
        plan = _build_aug_plan(extra_options)
        if plan is not None:
            bjer, cmix, sigma_y = plan.build(X_train_s, y_train_s, seq_len=n_features, device=device)
            if plan.bjerrum.enabled:
                config.augmenter = bjer
            if plan.cmixup.enabled:
                config.cmixup = cmix
                config.cmixup_sigma_y = sigma_y

    t0 = time.time()
    model, info = train_torch_regressor(model, X_train_s, y_train_s, config)
    fit_time = time.time() - t0
    t0 = time.time()
    pred_scaled = predict_torch_regressor(model, X_test_s, device=device)
    pred_time = time.time() - t0
    pred = y_proc.inverse_transform(pred_scaled)

    hp = {
        "model": family,
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "device": str(device),
        "best_epoch": info["best_epoch"],
        "best_val_loss_scaled": info["best_val_loss"],
    }
    row = _result_row(
        spec, pred, y_test, fit_time, pred_time, hp,
        total_params=count_parameters(model),
        peak_vram_mb=cuda_peak_mb(),
    )
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _result_row(
    spec: DatasetSpec,
    pred: np.ndarray,
    y_test: np.ndarray,
    fit_time: float,
    pred_time: float,
    hyperparams: dict,
    *,
    total_params: int | float,
    peak_vram_mb: float,
) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()
    rmsep_value = rmse(y_test, pred)
    return {
        "rmsep": rmsep_value,
        "mae": mae(y_test, pred),
        "r2": r2(y_test, pred),
        "bias": bias_metric(y_test, pred),
        "fit_time_s": fit_time,
        "predict_time_s": pred_time,
        "total_params": int(total_params) if not isinstance(total_params, float) or not np.isnan(total_params) else 0,
        "peak_vram_mb": peak_vram_mb,
        "hyperparams_json": json.dumps(hyperparams, default=str),
        "ref_rmse_pls": spec.ref_rmse_pls,
        "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
        "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
        "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
        "ref_rmse_cnn": spec.ref_rmse_cnn,
        "ref_rmse_catboost": spec.ref_rmse_catboost,
        "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best,
        "relative_rmsep_vs_pls": relative_rmsep(rmsep_value, spec.ref_rmse_pls),
        "relative_rmsep_vs_paper_ridge": relative_rmsep(rmsep_value, spec.ref_rmse_paper_ridge),
        "relative_rmsep_vs_tabpfn_raw": relative_rmsep(rmsep_value, spec.ref_rmse_tabpfn_raw),
        "relative_rmsep_vs_tabpfn_opt": relative_rmsep(rmsep_value, spec.ref_rmse_tabpfn_opt),
        "relative_rmsep_vs_cnn": relative_rmsep(rmsep_value, spec.ref_rmse_cnn),
        "relative_rmsep_vs_aom_ridge_curated_best": relative_rmsep(rmsep_value, spec.ref_rmse_aom_ridge_curated_best),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_BENCH_ROOT.parent.parent),
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def run_one(
    variant: Variant,
    spec: DatasetSpec,
    seed: int,
) -> dict:
    """Run a single (variant, dataset, seed) and return a result row dict (no metadata)."""
    X_train, y_train, X_test, y_test = load_dataset(spec)
    if variant.family == "ridge":
        result = _run_ridge(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "pls":
        result = _run_pls(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "searched_ridge":
        result = _run_searched_ridge(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "searched_pls":
        result = _run_searched_pls(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "stack":
        result = _run_stack(spec, X_train, y_train, X_test, y_test, seed, extra=variant.extra)
    elif variant.family in ("nicon", "decon", "nicon_v1a", "nicon_v1a_head", "nicon_v1a_act", "nicon_v1b", "nicon_v1c", "nicon_v2a"):
        result = _run_torch_cnn(
            spec, X_train, y_train, X_test, y_test, seed,
            family=variant.family, extra_options=variant.extra,
        )
    else:
        raise ValueError(f"unknown variant family: {variant.family!r}")
    return result


def filter_specs(all_specs: list[DatasetSpec], cohort: str, only: list[str] | None) -> list[DatasetSpec]:
    if only:
        keep = set(only)
        return [s for s in all_specs if s.dataset in keep]
    return all_specs


def main() -> int:
    parser = argparse.ArgumentParser(description="nicon_v2 baseline benchmark runner")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cohort", default="smoke",
                        choices=["smoke", "extended_smoke", "representative", "curated", "full"])
    parser.add_argument("--variants", default="smoke",
                        choices=["smoke", "phase1a", "phase1b", "phase1c", "stack", "stack_aom", "searched", "v2a"])
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="run multiple seeds (overrides --seed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to dataset names")
    parser.add_argument("--skip-cnn", action="store_true", help="skip nicon/decon variants (fast smoke)")
    args = parser.parse_args()

    workspace: Path = args.workspace
    csv_path = workspace / "results.csv"
    workspace.mkdir(parents=True, exist_ok=True)

    specs = load_cohort_manifest(args.cohort)
    specs = filter_specs(specs, args.cohort, args.only)

    if args.variants == "smoke":
        variants = list(SMOKE_VARIANTS)
    elif args.variants == "phase1a":
        variants = list(PHASE1A_VARIANTS)
    elif args.variants == "phase1b":
        variants = list(PHASE1B_VARIANTS)
    elif args.variants == "phase1c":
        variants = list(PHASE1C_VARIANTS)
    elif args.variants == "stack":
        variants = list(PHASE_STACK_VARIANTS)
    elif args.variants == "stack_aom":
        variants = list(PHASE_STACK_AOM_VARIANTS)
    elif args.variants == "searched":
        variants = list(PHASE_SEARCHED_VARIANTS)
    elif args.variants == "v2a":
        variants = list(PHASE_V2A_VARIANTS)
    else:
        raise ValueError(f"unknown variants set: {args.variants!r}")
    if args.skip_cnn:
        variants = [v for v in variants if v.family in ("ridge", "pls")]
    seeds = args.seeds if args.seeds else [args.seed]

    done = existing_done_rows(csv_path)
    writer, fh = open_result_writer(csv_path)
    git_sha = get_git_sha()
    cuda_version = torch.version.cuda or ""

    try:
      for current_seed in seeds:
        for spec in specs:
            for variant in variants:
                key = (spec.dataset, variant.label, current_seed)
                if key in done:
                    print(f"skip [done] {variant.label} on {spec.dataset} seed={current_seed}", flush=True)
                    continue
                row: dict = {
                    "dataset_group": spec.database_name,
                    "dataset": spec.dataset,
                    "task": "regression",
                    "n_train": spec.n_train,
                    "n_test": spec.n_test,
                    "n_features": spec.n_features,
                    "variant": variant.label,
                    "model_version": CODE_VERSION,
                    "seed": current_seed,
                    "cv_fold": -1,
                    "cv_protocol": "predefined",
                    "status": "OK",
                    "error_message": "",
                    "rmsep": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "bias": float("nan"),
                    "fit_time_s": float("nan"),
                    "predict_time_s": float("nan"),
                    "total_params": 0,
                    "peak_vram_mb": float("nan"),
                    "hyperparams_json": "",
                    "ref_rmse_pls": spec.ref_rmse_pls,
                    "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
                    "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
                    "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
                    "ref_rmse_cnn": spec.ref_rmse_cnn,
                    "ref_rmse_catboost": spec.ref_rmse_catboost,
                    "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best,
                    "relative_rmsep_vs_pls": None,
                    "relative_rmsep_vs_paper_ridge": None,
                    "relative_rmsep_vs_tabpfn_raw": None,
                    "relative_rmsep_vs_tabpfn_opt": None,
                    "relative_rmsep_vs_cnn": None,
                    "relative_rmsep_vs_aom_ridge_curated_best": None,
                    "python_version": sys.version.split(" ")[0],
                    "torch_version": torch.__version__,
                    "cuda_version": cuda_version,
                    "git_sha": git_sha,
                    "host": platform.node(),
                }
                t_start = time.time()
                try:
                    out = run_one(variant, spec, current_seed)
                    pred_arr = out.pop("_pred", None)
                    y_test_arr = out.pop("_y_test", None)
                    row.update(out)
                    if pred_arr is not None and y_test_arr is not None:
                        write_predictions_parquet(
                            workspace, variant.label, spec.dataset, current_seed,
                            y_test_arr, pred_arr,
                        )
                    print(
                        f"OK    {variant.label:<22}  {spec.dataset:<35}  rmse={row['rmsep']:.4f}  "
                        f"fit={row['fit_time_s']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # benchmark must keep running on per-row errors
                    row["status"] = "ERROR"
                    row["error_message"] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"ERROR {variant.label:<18}  {spec.dataset:<35}  {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                writer.writerow(row)
                fh.flush()
        print(f"done in {time.time() - t_start:.1f}s; rows written to {csv_path}")
    finally:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
