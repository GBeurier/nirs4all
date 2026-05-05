from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median
from typing import Any


BENCH = Path(__file__).resolve().parent
OUT_CSV = BENCH / "benchmark_master_results.csv"
OUT_MD = BENCH / "benchmark_synthesis.md"


FIELDNAMES = [
    "record_id",
    "record_type",
    "source_family",
    "source_kind",
    "source_path",
    "source_run",
    "dataset_group",
    "dataset",
    "task",
    "evaluation_split",
    "status",
    "error_message",
    "model_name",
    "variant",
    "model_class",
    "strategy_family",
    "preprocessing_pipeline",
    "operator_bank",
    "selection",
    "method",
    "backend",
    "engine",
    "seed",
    "run_seed",
    "cv_fold",
    "cv_protocol",
    "n_splits",
    "n_train",
    "n_test",
    "n_features",
    "n_components",
    "rmsep",
    "rmsecv",
    "rmse_mf",
    "mae",
    "r2",
    "bias",
    "balanced_accuracy",
    "macro_f1",
    "log_loss",
    "ece",
    "fit_time_s",
    "predict_time_s",
    "ref_rmse_pls",
    "ref_rmse_ridge",
    "ref_rmse_paper_ridge",
    "ref_rmse_tabpfn_raw",
    "ref_rmse_tabpfn_opt",
    "ref_rmse_cnn",
    "ref_rmse_catboost",
    "ref_rmse_aom_ridge_curated_best",
    "relative_rmsep_vs_pls",
    "relative_rmsep_vs_ridge",
    "relative_rmsep_vs_paper_ridge",
    "relative_rmsep_vs_tabpfn_raw",
    "relative_rmsep_vs_tabpfn_opt",
    "relative_rmsep_vs_cnn",
    "relative_rmsep_vs_catboost",
    "delta_rmsep_vs_master_pls",
    "delta_rmsep_vs_tabpfn_raw",
    "delta_rmsep_vs_tabpfn_opt",
    "score_metric",
    "score_value",
    "lower_is_better",
    "dataset_pls_score",
    "score_ratio_vs_dataset_pls",
    "source_run_pls_score",
    "score_ratio_vs_source_run_pls",
    "dataset_best_score",
    "dataset_best_model_class",
    "dataset_best_model_name",
    "dataset_best_record_id",
    "score_ratio_to_dataset_oracle",
    "model_class_oracle_score",
    "model_class_oracle_model_name",
    "model_class_oracle_record_id",
    "score_ratio_to_model_class_oracle",
    "is_model_class_oracle",
    "oracle_scope",
    "oracle_winner_record_id",
    "config_json",
    "notes",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\xa0", " ").replace("\r", " ").replace("\n", " ").strip()


def first(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value != "":
            return value
    return ""


def as_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    text = text.replace("%", "").replace(",", ".")
    text = re.sub(r"[^0-9eE+\-\.]", "", text)
    if text in {"", ".", "+", "-"}:
        return None
    try:
        value_f = float(text)
    except ValueError:
        return None
    if math.isnan(value_f) or math.isinf(value_f):
        return None
    return value_f


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def status_ok(status: str) -> bool:
    return clean_text(status).lower() in {"", "ok", "success", "done", "complete", "completed"}


def infer_dataset_group(dataset: str, explicit: str = "") -> str:
    explicit = clean_text(explicit)
    if explicit:
        return explicit
    dataset = clean_text(dataset)
    if not dataset:
        return ""
    special = {
        "All_manure": "MANURE",
        "Beer": "BEER",
        "DIESEL": "DIESEL",
        "WOOD": "WOOD",
        "Rice": "AMYLOSE",
        "Beef": "BEEFMARBLING",
        "Milk": "MILK",
        "Corn": "CORN",
        "Biscuit": "BISCUIT",
        "ALPINE": "ALPINE",
        "Chla+b": "LEAF",
    }
    for prefix, group in special.items():
        if dataset.startswith(prefix):
            return group
    return dataset.split("_", 1)[0].upper()


def source_family(path: Path) -> str:
    rel = path.as_posix()
    if "tabpfn_paper" in rel:
        return "tabpfn_paper"
    if "fck_pls" in rel:
        return "fck_pls"
    if "nicon_v2" in rel:
        return "nicon_v2"
    if "AOM_v0/Ridge" in rel:
        return "AOM_v0_Ridge"
    if "AOM_v0/Multi-kernel" in rel:
        return "AOM_v0_MultiKernel"
    if "AOM_v0/multiview" in rel:
        return "AOM_v0_multiview"
    if "AOM_v0" in rel:
        return "AOM_v0"
    return "bench"


def run_name(path: Path) -> str:
    parts = path.relative_to(BENCH).parts
    if "benchmark_runs" in parts:
        idx = parts.index("benchmark_runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "results" in parts:
        idx = parts.index("results")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def classify_model(label: str, family: str) -> tuple[str, str]:
    text = clean_text(label)
    low = text.lower()

    if "catboost" in low:
        return "CatBoost", "tree ensemble"
    if "tabpfn" in low:
        return "TabPFN", "foundation tabular prior"
    if "fck" in low or "fractional" in low:
        return "FCK-PLS", "learnable spectral filters"
    if "stack" in low or "residual" in low or "boost" in low:
        if "aom" in low:
            return "Hybrid CNN+AOM", "hybrid residual/stacking"
        return "Hybrid CNN+linear", "hybrid residual/stacking"
    if "ensemble" in low or "super-learner" in low or "superlearner" in low or low.startswith("mean-") or low.startswith("trimmed-mean"):
        return "Meta-selector/MoE", "adaptive model selection"
    if "moe" in low or "meta" in low or "lazy" in low:
        return "Meta-selector/MoE", "adaptive model selection"
    if "nicon" in low or "decon" in low or re.search(r"\bv[0-9][a-z]?", low):
        return "NICON/CNN", "deep spectral model"
    if "cnn" in low:
        return "NICON/CNN", "deep spectral model"
    if "aomridge" in low or "aom-ridge" in low or family == "AOM_v0_Ridge":
        if "ridge-raw" in low or "ridge-baseline" in low:
            return "Ridge", "linear baseline"
        return "AOM-Ridge", "operator-adaptive ridge"
    if "pop" in low:
        return "POP-PLS", "operator-adaptive PLS"
    if "aom" in low or "active" in low or "superblock" in low or "pipeline-" in low:
        return "AOM-PLS", "operator-adaptive PLS"
    if "pls" in low:
        return "PLS", "linear latent-variable baseline"
    if "ridge" in low:
        return "Ridge", "linear baseline"
    if "mkr" in low or "mkm" in low or "blup" in low or family == "AOM_v0_MultiKernel":
        return "Multi-kernel ridge", "kernel ensemble"
    return "Other", "other"


def make_record(**kwargs: Any) -> dict[str, Any]:
    record = {key: "" for key in FIELDNAMES}
    for key, value in kwargs.items():
        if key in record:
            record[key] = fmt(value)
    return record


def score_record(record: dict[str, Any]) -> None:
    task = clean_text(record.get("task")).lower()
    rmsep = as_float(record.get("rmsep"))
    bal_acc = as_float(record.get("balanced_accuracy"))
    if task == "classification" or (rmsep is None and bal_acc is not None):
        if bal_acc is not None:
            record["score_metric"] = "balanced_accuracy"
            record["score_value"] = fmt(bal_acc)
            record["lower_is_better"] = "False"
        return
    if rmsep is not None:
        record["score_metric"] = "rmsep"
        record["score_value"] = fmt(rmsep)
        record["lower_is_better"] = "True"


def add_generic_record(records: list[dict[str, Any]], path: Path, row: dict[str, Any]) -> None:
    family = source_family(path)
    dataset = first(row, "dataset", "dataset_name")
    task = first(row, "task") or ("classification" if first(row, "balanced_accuracy") else "regression")
    status = first(row, "status") or "ok"
    model = first(row, "variant", "model", "result_label", "model_class")
    if not model:
        model = path.stem
    model_class, strategy = classify_model(model, family)

    rmsep = as_float(first(row, "rmsep", "RMSEP", "RMSE"))
    rmsecv = as_float(first(row, "RMSECV"))
    rmse_mf = as_float(first(row, "RMSE_MF"))
    n_components = first(row, "n_components", "n_components_selected", "best_n_components", "effective_components")
    record = make_record(
        record_type="observed",
        source_family=family,
        source_kind="benchmark_results",
        source_path=path.as_posix(),
        source_run=run_name(path),
        dataset_group=infer_dataset_group(dataset, first(row, "dataset_group", "database_name")),
        dataset=dataset,
        task=task,
        evaluation_split="test",
        status=status,
        error_message=first(row, "error", "error_message", "status_details"),
        model_name=model,
        variant=first(row, "variant", "aom_variant", "result_label", "model"),
        model_class=model_class,
        strategy_family=strategy,
        preprocessing_pipeline=first(row, "preprocessing_pipeline", "branch_preproc", "x_scale"),
        operator_bank=first(row, "operator_bank"),
        selection=first(row, "selection", "criterion"),
        method=first(row, "method"),
        backend=first(row, "backend"),
        engine=first(row, "engine"),
        seed=first(row, "seed", "random_state"),
        run_seed=first(row, "run_seed"),
        cv_fold=first(row, "cv_fold"),
        cv_protocol=first(row, "cv_protocol"),
        n_splits=first(row, "n_splits"),
        n_train=as_float(first(row, "n_train")),
        n_test=as_float(first(row, "n_test")),
        n_features=as_float(first(row, "n_features", "p")),
        n_components=n_components,
        rmsep=rmsep,
        rmsecv=rmsecv,
        rmse_mf=rmse_mf,
        mae=as_float(first(row, "mae", "MAE_test")),
        r2=as_float(first(row, "r2", "r2_test")),
        bias=as_float(first(row, "bias")),
        balanced_accuracy=as_float(first(row, "balanced_accuracy")),
        macro_f1=as_float(first(row, "macro_f1")),
        log_loss=as_float(first(row, "log_loss")),
        ece=as_float(first(row, "ece")),
        fit_time_s=as_float(first(row, "fit_time_s")),
        predict_time_s=as_float(first(row, "predict_time_s")),
        ref_rmse_pls=as_float(first(row, "ref_rmse_pls")),
        ref_rmse_ridge=as_float(first(row, "ref_rmse_ridge", "ref_rmse_ridge_raw")),
        ref_rmse_paper_ridge=as_float(first(row, "ref_rmse_paper_ridge")),
        ref_rmse_tabpfn_raw=as_float(first(row, "ref_rmse_tabpfn_raw")),
        ref_rmse_tabpfn_opt=as_float(first(row, "ref_rmse_tabpfn_opt")),
        ref_rmse_cnn=as_float(first(row, "ref_rmse_cnn")),
        ref_rmse_catboost=as_float(first(row, "ref_rmse_catboost")),
        ref_rmse_aom_ridge_curated_best=as_float(first(row, "ref_rmse_aom_ridge_curated_best")),
        relative_rmsep_vs_pls=as_float(first(row, "relative_rmsep_vs_pls", "relative_rmsep_vs_pls_standard", "rel_rmsep_vs_pls")),
        relative_rmsep_vs_ridge=as_float(first(row, "relative_rmsep_vs_ridge", "relative_rmsep_vs_ridge_raw", "rel_rmsep_vs_ridge")),
        relative_rmsep_vs_paper_ridge=as_float(first(row, "relative_rmsep_vs_paper_ridge")),
        relative_rmsep_vs_tabpfn_raw=as_float(first(row, "relative_rmsep_vs_tabpfn_raw")),
        relative_rmsep_vs_tabpfn_opt=as_float(first(row, "relative_rmsep_vs_tabpfn_opt", "rel_rmsep_vs_tabpfn_opt")),
        relative_rmsep_vs_cnn=as_float(first(row, "relative_rmsep_vs_cnn")),
        delta_rmsep_vs_master_pls=as_float(first(row, "delta_rmsep_vs_master_pls")),
        delta_rmsep_vs_tabpfn_raw=as_float(first(row, "delta_rmsep_vs_tabpfn_raw")),
        delta_rmsep_vs_tabpfn_opt=as_float(first(row, "delta_rmsep_vs_tabpfn_opt")),
        config_json=first(
            row,
            "hyperparams_json",
            "best_config_json",
            "best_model_params_json",
            "selected_operator_sequence_json",
            "ridgepls_diagnostics",
        ),
        notes=first(row, "notes"),
    )
    score_record(record)
    records.append(record)


def read_csv_dicts(path: Path, *, delimiter: str = ",", encoding: str = "utf-8-sig") -> list[dict[str, str]]:
    with path.open(newline="", encoding=encoding) as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def collect_result_paths() -> list[Path]:
    paths: set[Path] = set()
    paths.update((BENCH / "nicon_v2" / "benchmark_runs").rglob("results*.csv"))
    paths.update((BENCH / "AOM_v0" / "benchmark_runs").rglob("results*.csv"))
    paths.update((BENCH / "AOM_v0" / "Ridge" / "benchmark_runs").rglob("results*.csv"))
    paths.update((BENCH / "AOM_v0" / "Multi-kernel" / "benchmark_runs").rglob("results*.csv"))
    for name in ["full57.csv", "smoke10.csv", "smoke4_baseline.csv", "smoke_classification.csv"]:
        path = BENCH / "AOM_v0" / "multiview" / "results" / name
        if path.exists():
            paths.add(path)
    return sorted(paths)


def parse_master_pivot(records: list[dict[str, Any]]) -> None:
    path = BENCH / "AOM_v0" / "publication" / "tables" / "master_pivot.csv"
    if not path.exists():
        return
    for row in read_csv_dicts(path):
        dataset = first(row, "dataset")
        group = infer_dataset_group(dataset, first(row, "database_name"))
        for model in ["CNN", "Catboost", "PLS", "Ridge", "TabPFN-Raw", "TabPFN-opt"]:
            rmsep = as_float(row.get(model))
            if rmsep is None:
                continue
            model_class, strategy = classify_model(model, "paper_reference")
            record = make_record(
                record_type="reference_paper",
                source_family="paper_master_pivot",
                source_kind="paper_reference",
                source_path=path.as_posix(),
                source_run="master_pivot",
                dataset_group=group,
                dataset=dataset,
                task="regression",
                evaluation_split="test",
                status="ok",
                model_name=model,
                variant=model,
                model_class=model_class,
                strategy_family=strategy,
                rmsep=rmsep,
            )
            score_record(record)
            records.append(record)


def parse_tabpfn_light(records: list[dict[str, Any]]) -> None:
    path = BENCH / "tabpfn_paper" / "table_results_tabpfn_final_light.csv"
    if not path.exists():
        return
    lines = path.read_text(encoding="latin1").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(";", 4)
        if len(parts) < 4:
            continue
        dataset, rmse_fold, rmse_test, preprocessing = [clean_text(x) for x in parts[:4]]
        rest = parts[4] if len(parts) > 4 else ""
        cells = [clean_text(cell) for cell in rest.split("|")]
        cells = [cell for cell in cells if cell]
        rmsep_from_table = None
        rmsecv_from_table = None
        rmse_mf_from_table = None
        selected_by = ""
        final_preproc = preprocessing
        if "TabPFN" in cells:
            idx = cells.index("TabPFN")
            rmsep_from_table = as_float(cells[idx + 1] if idx + 1 < len(cells) else "")
            rmsecv_from_table = as_float(cells[idx + 4] if idx + 4 < len(cells) else "")
            rmse_mf_from_table = as_float(cells[idx + 5] if idx + 5 < len(cells) else "")
            selected_by = clean_text(cells[idx + 6] if idx + 6 < len(cells) else "")
            if idx + 7 < len(cells):
                final_preproc = f"{preprocessing} | paper_best={cells[idx + 7]}"
        rmsep = rmsep_from_table if rmsep_from_table is not None else as_float(rmse_test)
        record = make_record(
            record_type="observed",
            source_family="tabpfn_paper",
            source_kind="hpo_final_light",
            source_path=path.as_posix(),
            source_run="table_results_tabpfn_final_light",
            dataset_group=infer_dataset_group(dataset),
            dataset=dataset,
            task="regression",
            evaluation_split="test",
            status="ok" if rmsep is not None else "missing_test_metric",
            model_name="TabPFN-HPO-preprocessing",
            variant="TabPFN-HPO-preprocessing",
            model_class="TabPFN",
            strategy_family="foundation tabular prior",
            preprocessing_pipeline=final_preproc,
            rmsep=rmsep,
            rmsecv=rmsecv_from_table if rmsecv_from_table is not None else as_float(rmse_fold),
            rmse_mf=rmse_mf_from_table,
            config_json=json.dumps({"selected_by": selected_by}, ensure_ascii=True) if selected_by else "",
        )
        score_record(record)
        records.append(record)


def parse_fck_reports(records: list[dict[str, Any]]) -> None:
    for path in sorted((BENCH / "fck_pls").rglob("Report_best*.csv")):
        dataset = ""
        pipeline_steps = ""
        manifest = path.parent / "manifest.yaml"
        if manifest.exists():
            text = manifest.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"^dataset:\s*(.+)$", text, flags=re.MULTILINE)
            if match:
                dataset = clean_text(match.group(1).strip("'\""))
        if not dataset:
            parts = path.parts
            dataset = parts[parts.index("runs") + 1] if "runs" in parts else path.parent.parent.name
        pipeline = path.parent / "pipeline.json"
        if pipeline.exists():
            try:
                data = json.loads(pipeline.read_text(encoding="utf-8"))
                pipeline_steps = " > ".join(
                    step if isinstance(step, str) else clean_text(step.get("class") or step.get("model", {}).get("class"))
                    for step in data
                )
            except Exception:
                pipeline_steps = ""
        parent_label = re.sub(r"^\d+_", "", path.parent.name)
        parent_label = re.sub(r"_[0-9a-f]{6}$", "", parent_label)
        stem = re.sub(r"^Report_best_", "", path.stem)
        stem = re.sub(r"_[0-9a-f]{12,}$", "", stem)
        model_label = parent_label
        for token in ["FCK-PLS-v2-tuned", "FCK-PLS-v1-tuned", "FCK-PLS-Static-PP", "FCK-PLS-Static-Raw", "FCKPLS", "PLS_pp_tuned", "PLS-raw_tuned", "PLS-Tuned", "PLS-Baseline", "PLSRegression"]:
            if token.lower() in stem.lower():
                model_label = token
                break
        model_class, strategy = classify_model(f"{model_label} {stem}", "fck_pls")
        for row in read_csv_dicts(path):
            split = clean_text(row.get(""))
            eval_split = {"Cros Val": "cv", "Train": "train", "Test": "test"}.get(split, split.lower())
            record = make_record(
                record_type="observed",
                source_family="fck_pls",
                source_kind="report_best",
                source_path=path.as_posix(),
                source_run=run_name(path),
                dataset_group=infer_dataset_group(dataset),
                dataset=dataset,
                task="regression",
                evaluation_split=eval_split,
                status="ok",
                model_name=model_label,
                variant=parent_label,
                model_class=model_class,
                strategy_family=strategy,
                preprocessing_pipeline=pipeline_steps,
                n_train=as_float(row.get("Nsample")) if eval_split == "train" else "",
                n_test=as_float(row.get("Nsample")) if eval_split == "test" else "",
                n_features=as_float(row.get("Nfeature")),
                rmsep=as_float(row.get("RMSE")),
                mae=as_float(row.get("MAE")),
                r2=as_float(row.get("R2")),
                bias=as_float(row.get("Bias")),
                notes=f"FCK report split: {split}",
            )
            if eval_split == "cv":
                record["rmsecv"] = record["rmsep"]
            score_record(record)
            records.append(record)


def parse_meta_selector(records: list[dict[str, Any]]) -> None:
    path = BENCH / "AOM_v0" / "multiview" / "results" / "meta_selector_full57.csv"
    if not path.exists():
        return
    metric_columns = {
        "meta_logreg_rmsep": "meta_logreg_variant",
        "meta_rf_rmsep": "meta_rf_variant",
        "oracle_rmsep": "oracle_variant",
    }
    fixed_cols = {
        "dataset",
        "meta_logreg_variant",
        "meta_rf_variant",
        "meta_logreg_rmsep",
        "meta_rf_rmsep",
        "oracle_rmsep",
        "oracle_variant",
    }
    for row in read_csv_dicts(path):
        dataset = first(row, "dataset")
        for metric_col, label_col in metric_columns.items():
            label = first(row, label_col) or metric_col.replace("_rmsep", "")
            rmsep = as_float(row.get(metric_col))
            if rmsep is None:
                continue
            model_class, strategy = classify_model(label, "AOM_v0_multiview")
            record_type = "source_oracle" if metric_col == "oracle_rmsep" else "observed"
            record = make_record(
                record_type=record_type,
                source_family="AOM_v0_multiview",
                source_kind="meta_selector",
                source_path=path.as_posix(),
                source_run="meta_selector_full57",
                dataset_group=infer_dataset_group(dataset),
                dataset=dataset,
                task="regression",
                evaluation_split="test",
                status="ok",
                model_name=label,
                variant=label,
                model_class=model_class if record_type != "source_oracle" else "Oracle_All_Classes",
                strategy_family=strategy if record_type != "source_oracle" else "oracle",
                rmsep=rmsep,
            )
            score_record(record)
            records.append(record)
        for label, value in row.items():
            if label in fixed_cols:
                continue
            rmsep = as_float(value)
            if rmsep is None:
                continue
            model_class, strategy = classify_model(label, "AOM_v0_multiview")
            record = make_record(
                record_type="observed",
                source_family="AOM_v0_multiview",
                source_kind="meta_selector_wide_variant",
                source_path=path.as_posix(),
                source_run="meta_selector_full57",
                dataset_group=infer_dataset_group(dataset),
                dataset=dataset,
                task="regression",
                evaluation_split="test",
                status="ok",
                model_name=label,
                variant=label,
                model_class=model_class,
                strategy_family=strategy,
                rmsep=rmsep,
            )
            score_record(record)
            records.append(record)


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    parse_master_pivot(records)
    parse_tabpfn_light(records)
    parse_fck_reports(records)
    parse_meta_selector(records)
    for path in collect_result_paths():
        for row in read_csv_dicts(path):
            add_generic_record(records, path, row)
    for i, record in enumerate(records, start=1):
        record["record_id"] = f"obs_{i:06d}"
    return records


def eligible(record: dict[str, Any]) -> bool:
    if record.get("record_type") not in {"observed", "reference_paper"}:
        return False
    if not status_ok(record.get("status", "")):
        return False
    if not record.get("score_metric"):
        return False
    split = clean_text(record.get("evaluation_split")).lower()
    if split in {"train", "cv", "cross_val", "cross-validation", "cros val"}:
        return False
    return True


def score_value(record: dict[str, Any]) -> float:
    value = as_float(record.get("score_value"))
    if value is None:
        raise ValueError("record has no numeric score")
    return value


def is_lower(record: dict[str, Any]) -> bool:
    return clean_text(record.get("lower_is_better")).lower() != "false"


def better(left: dict[str, Any], right: dict[str, Any] | None) -> bool:
    if right is None:
        return True
    if is_lower(left):
        return score_value(left) < score_value(right)
    return score_value(left) > score_value(right)


def ratio_to_oracle(record: dict[str, Any], oracle: dict[str, Any] | None) -> float | None:
    if oracle is None:
        return None
    score = score_value(record)
    best = score_value(oracle)
    if score <= 0 or best <= 0:
        return None
    if is_lower(record):
        return score / best
    return best / score


def enrich_with_oracles(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    elig = [record for record in records if eligible(record)]
    best_dataset: dict[tuple[str, str, str], dict[str, Any]] = {}
    best_class: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    best_pls: dict[tuple[str, str], dict[str, Any]] = {}
    best_source_pls: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for record in elig:
        metric = clean_text(record.get("score_metric"))
        task = clean_text(record.get("task")) or "regression"
        dataset = clean_text(record.get("dataset"))
        model_class = clean_text(record.get("model_class"))
        key_dataset = (dataset, task, metric)
        if better(record, best_dataset.get(key_dataset)):
            best_dataset[key_dataset] = record
        key_class = (dataset, task, metric, model_class)
        if better(record, best_class.get(key_class)):
            best_class[key_class] = record
        if metric == "rmsep" and model_class == "PLS":
            key_pls = (dataset, task)
            if better(record, best_pls.get(key_pls)):
                best_pls[key_pls] = record
            key_source_pls = (
                clean_text(record.get("source_family")),
                clean_text(record.get("source_run")),
                dataset,
                task,
            )
            if better(record, best_source_pls.get(key_source_pls)):
                best_source_pls[key_source_pls] = record

    for record in records:
        if not record.get("score_metric") or not clean_text(record.get("dataset")):
            continue
        metric = clean_text(record.get("score_metric"))
        task = clean_text(record.get("task")) or "regression"
        dataset = clean_text(record.get("dataset"))
        model_class = clean_text(record.get("model_class"))
        d_oracle = best_dataset.get((dataset, task, metric))
        c_oracle = best_class.get((dataset, task, metric, model_class))
        pls = best_pls.get((dataset, task))
        if pls is not None and metric == "rmsep":
            record["dataset_pls_score"] = fmt(score_value(pls))
            score = as_float(record.get("score_value"))
            if score is not None and score > 0 and score_value(pls) > 0:
                record["score_ratio_vs_dataset_pls"] = fmt(score / score_value(pls))
        source_pls = best_source_pls.get(
            (
                clean_text(record.get("source_family")),
                clean_text(record.get("source_run")),
                dataset,
                task,
            )
        )
        if source_pls is not None and metric == "rmsep":
            record["source_run_pls_score"] = fmt(score_value(source_pls))
            score = as_float(record.get("score_value"))
            if score is not None and score > 0 and score_value(source_pls) > 0:
                record["score_ratio_vs_source_run_pls"] = fmt(score / score_value(source_pls))
        if d_oracle is not None:
            record["dataset_best_score"] = fmt(score_value(d_oracle))
            record["dataset_best_model_class"] = d_oracle.get("model_class", "")
            record["dataset_best_model_name"] = d_oracle.get("model_name", "")
            record["dataset_best_record_id"] = d_oracle.get("record_id", "")
            record["score_ratio_to_dataset_oracle"] = fmt(ratio_to_oracle(record, d_oracle))
        if c_oracle is not None:
            record["model_class_oracle_score"] = fmt(score_value(c_oracle))
            record["model_class_oracle_model_name"] = c_oracle.get("model_name", "")
            record["model_class_oracle_record_id"] = c_oracle.get("record_id", "")
            record["score_ratio_to_model_class_oracle"] = fmt(ratio_to_oracle(record, c_oracle))
            record["is_model_class_oracle"] = str(record.get("record_id") == c_oracle.get("record_id"))

    oracle_rows: list[dict[str, Any]] = []
    for i, ((dataset, task, metric, model_class), winner) in enumerate(sorted(best_class.items()), start=1):
        row = dict(winner)
        row["record_id"] = f"oracle_class_{i:06d}"
        row["record_type"] = "oracle_by_model_class"
        row["source_kind"] = "derived_oracle"
        row["model_name"] = f"{model_class} oracle"
        row["variant"] = winner.get("model_name", "")
        row["oracle_scope"] = f"dataset={dataset};task={task};metric={metric};model_class={model_class}"
        row["oracle_winner_record_id"] = winner.get("record_id", "")
        row["is_model_class_oracle"] = "True"
        row["score_ratio_to_model_class_oracle"] = "1"
        row["score_ratio_to_dataset_oracle"] = fmt(ratio_to_oracle(winner, best_dataset.get((dataset, task, metric))))
        oracle_rows.append(row)

    for i, ((dataset, task, metric), winner) in enumerate(sorted(best_dataset.items()), start=1):
        row = dict(winner)
        row["record_id"] = f"oracle_dataset_{i:06d}"
        row["record_type"] = "oracle_global_dataset"
        row["source_kind"] = "derived_oracle"
        row["model_class"] = "Oracle_All_Classes"
        row["strategy_family"] = "oracle"
        row["model_name"] = "Dataset oracle"
        row["variant"] = winner.get("model_name", "")
        row["oracle_scope"] = f"dataset={dataset};task={task};metric={metric};all_classes"
        row["oracle_winner_record_id"] = winner.get("record_id", "")
        row["score_ratio_to_dataset_oracle"] = "1"
        oracle_rows.append(row)

    return records + oracle_rows


def write_csv(records: list[dict[str, Any]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in FIELDNAMES})


def class_oracle_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        row
        for row in records
        if row.get("record_type") == "oracle_by_model_class"
        and row.get("score_metric") == "rmsep"
        and as_float(row.get("score_ratio_vs_dataset_pls")) is not None
    ]
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("model_class") == "Oracle_All_Classes":
            continue
        by_class[row["model_class"]].append(row)
    summary = []
    for model_class, items in by_class.items():
        ratios = [as_float(row.get("score_ratio_vs_dataset_pls")) for row in items]
        ratios = [value for value in ratios if value is not None]
        wins = sum(1 for value in ratios if value < 1.0)
        summary.append(
            {
                "model_class": model_class,
                "n_datasets": len(ratios),
                "median_rel_pls": median(ratios) if ratios else None,
                "wins_vs_pls": wins,
                "win_rate": wins / len(ratios) if ratios else None,
            }
        )
    return sorted(summary, key=lambda row: (row["median_rel_pls"] is None, row["median_rel_pls"] or 999, -row["n_datasets"]))


def normalize_model_key(model_name: str) -> str:
    text = clean_text(model_name)
    low = text.lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "tabpfn-opt": "TabPFN-opt",
        "tabpfn-raw": "TabPFN-Raw",
        "tabpfn-hpo-preprocessing": "TabPFN-HPO-preprocessing",
        "pls-standard": "PLS-standard",
        "pls-standard-numpy": "PLS-standard",
        "pls-standard-numpy-paper": "PLS-standard",
        "pls": "PLS-standard",
        "ridge-ref": "Ridge",
        "ridge": "Ridge",
        "ridge-raw": "Ridge-raw",
    }
    return aliases.get(low, text)


def ranking_ratio(row: dict[str, Any]) -> float | None:
    return as_float(row.get("score_ratio_vs_source_run_pls")) or as_float(row.get("score_ratio_vs_dataset_pls"))


def variant_leaderboard(records: list[dict[str, Any]], *, min_datasets: int = 10) -> list[dict[str, Any]]:
    rows = [
        row
        for row in records
        if row.get("record_type") in {"observed", "reference_paper"}
        and row.get("score_metric") == "rmsep"
        and eligible(row)
        and ranking_ratio(row) is not None
        and row.get("model_class") != "Oracle_All_Classes"
    ]
    best_by_variant_dataset: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        model_key = normalize_model_key(row.get("model_name") or row.get("variant"))
        key = (row.get("model_class", ""), model_key, row.get("dataset", ""))
        previous = best_by_variant_dataset.get(key)
        if previous is None or (ranking_ratio(row) or 999) < (ranking_ratio(previous) or 999):
            best_by_variant_dataset[key] = row
    by_variant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model_class, model_key, _dataset), row in best_by_variant_dataset.items():
        by_variant[(model_class, model_key)].append(row)
    leaderboard = []
    for (model_class, model_key), items in by_variant.items():
        ratios = [ranking_ratio(row) for row in items]
        ratios = [value for value in ratios if value is not None]
        if len(ratios) < min_datasets:
            continue
        leaderboard.append(
            {
                "model_class": model_class,
                "model_name": model_key,
                "n_datasets": len(ratios),
                "median_rel_pls": median(ratios),
                "wins_vs_pls": sum(1 for value in ratios if value < 1.0),
                "example_variant": Counter(clean_text(row.get("variant")) for row in items).most_common(1)[0][0],
            }
        )
    return sorted(leaderboard, key=lambda row: (row["median_rel_pls"], -row["n_datasets"], row["model_name"]))


def named_variant_summaries(records: list[dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    out = []
    for name in names:
        best_source: dict[str, float] = {}
        best_global: dict[str, float] = {}
        for row in records:
            if row.get("record_type") not in {"observed", "reference_paper"}:
                continue
            if normalize_model_key(row.get("model_name") or row.get("variant")) != normalize_model_key(name):
                continue
            dataset = clean_text(row.get("dataset"))
            source_ratio = as_float(row.get("score_ratio_vs_source_run_pls"))
            global_ratio = as_float(row.get("score_ratio_vs_dataset_pls"))
            if dataset and source_ratio is not None:
                best_source[dataset] = min(best_source.get(dataset, float("inf")), source_ratio)
            if dataset and global_ratio is not None:
                best_global[dataset] = min(best_global.get(dataset, float("inf")), global_ratio)
        if not best_source and not best_global:
            continue
        source_vals = list(best_source.values())
        global_vals = list(best_global.values())
        out.append(
            {
                "model_name": normalize_model_key(name),
                "n_source": len(source_vals),
                "median_source": median(source_vals) if source_vals else None,
                "wins_source": sum(value < 1.0 for value in source_vals),
                "n_global": len(global_vals),
                "median_global": median(global_vals) if global_vals else None,
                "wins_global": sum(value < 1.0 for value in global_vals),
            }
        )
    return out


def subset_transfer_rows() -> list[dict[str, str]]:
    path = BENCH / "Subset_analysis" / "subset_transfer_summary.csv"
    if not path.exists():
        return []
    wanted = [
        ("current_class_balanced_10", "all_candidates"),
        ("current_class_balanced_10", "no_tabpfn"),
        ("current_class_balanced_10", "aom_pls_only"),
        ("current_class_balanced_10", "aom_ridge_only"),
        ("current_conservative_19", "all_candidates"),
        ("current_conservative_19", "no_tabpfn"),
        ("legacy_variant_heavy_10", "no_tabpfn"),
    ]
    wanted_set = set(wanted)
    rows = []
    for row in read_csv_dicts(path):
        if (row.get("subset_name"), row.get("scope")) in wanted_set:
            rows.append(row)
    order = {item: idx for idx, item in enumerate(wanted)}
    return sorted(rows, key=lambda row: order.get((row.get("subset_name"), row.get("scope")), 999))


def model_explanation(model_name: str, model_class: str) -> tuple[str, str, str]:
    low = f"{model_name} {model_class}".lower()
    if "tabpfn" in low:
        return (
            "TabPFN regression with a searched spectral preprocessing chain before the foundation tabular prior.",
            "Very strong small-tabular prior; benefits from the preprocessing HPO already done in `tabpfn_paper`.",
            "Expensive and hard to interpret; performance depends heavily on preprocessing search and may not extrapolate to larger or shifted domains.",
        )
    if "aomridge" in low or model_class == "AOM-Ridge":
        return (
            "Ridge regression after selecting or blending spectral operator branches; variants differ by global, local, auto-select, and blender selection.",
            "Best broad empirical challenger to TabPFN-opt; cheap inference and strong median gains over Ridge/PLS.",
            "Selection layer is complex and can overfit small validation splits; branch/local/MKL variants add variance if not locked down.",
        )
    if "aom" in low and "ridge" not in low:
        return (
            "PLS with an adaptive bank of spectral operators and fold-based or holdout model selection.",
            "Fast and spectroscopically grounded; ASLS plus compact/CV variants are robust first-line baselines.",
            "Large operator banks trigger winner's-curse selection; OSC/EMSC/POP variants can fail badly on small n.",
        )
    if model_class == "Multi-kernel ridge":
        return (
            "Combines kernels or branch-specific ridge models built from multiple spectral preprocessors.",
            "Good way to average complementary transformations without committing to a single operator.",
            "The all-dataset gains are modest; weighting/REML choices add moving parts and can underperform simple AOM-Ridge.",
        )
    if model_class == "Meta-selector/MoE":
        return (
            "Chooses or averages candidate predictors/views with a meta-model, soft gating rule, or simple ensemble aggregation.",
            "Captures complementarity between strong base learners and is useful for estimating oracle headroom.",
            "High leakage risk unless selection is nested; small-cohort ensemble gains may disappear when the candidate set is frozen.",
        )
    if "stack" in low or "residual" in low or "boost" in low:
        return (
            "Hybrid model that stacks or residualizes CNN/NICON features with Ridge, PLS, or AOM predictions.",
            "Can extract non-linear residual signal on some small plant/chemistry datasets.",
            "Pure CNN signal is weak on many NIRS sets; stacked gains are not enough to beat AOM-Ridge/TabPFN globally.",
        )
    if model_class == "NICON/CNN":
        return (
            "One-dimensional convolutional spectral neural network, including NICON/DECON and V2/V6 variants.",
            "Useful diagnostic for non-linear spectral features and residual learning.",
            "Data-hungry; unstable across seeds; often worse than tuned linear spectral models.",
        )
    if model_class == "FCK-PLS":
        return (
            "Learnable fractional convolutional filters feeding a PLS-style solved head.",
            "Interpretable spectral-filter idea; can adapt derivative-like kernels to the dataset.",
            "Mostly tested on small/synthetic cohorts; full-batch training and validation split design make scaling awkward.",
        )
    if model_class == "PLS":
        return (
            "Classical PLS latent-variable regression, sometimes with tuned preprocessing.",
            "Simple, fast, stable, and still the right anchor for relative scoring.",
            "Limited non-linearity; preprocessing and component selection become the main source of performance.",
        )
    if model_class == "Ridge":
        return (
            "Linear ridge regression on raw or preprocessed spectra.",
            "Strong, stable baseline for high-dimensional spectra; cheap and reproducible.",
            "Cannot choose spectral transformations by itself; underfits when baseline or scatter effects dominate.",
        )
    if model_class == "CatBoost":
        return (
            "Gradient-boosted trees from the paper benchmark reference table.",
            "Competitive tabular baseline on some small datasets without spectral assumptions.",
            "Less spectroscopically structured and can be brittle on high-dimensional smooth spectra.",
        )
    return (
        "Variant-specific benchmark entry from the merged result table.",
        "Useful as an explored point in the search space.",
        "Needs a locked protocol before treating the number as a production claim.",
    )


def write_md(records: list[dict[str, Any]]) -> None:
    observed = [row for row in records if row.get("record_type") in {"observed", "reference_paper", "source_oracle"}]
    eligible_rows = [row for row in observed if eligible(row)]
    source_counts = Counter(row.get("source_family", "") for row in observed)
    dataset_count = len({row.get("dataset") for row in eligible_rows if row.get("dataset")})
    class_summary = class_oracle_summary(records)
    top25 = variant_leaderboard(records)[:25]
    subset_rows = subset_transfer_rows()
    aom_check = named_variant_summaries(
        records,
        [
            "AOM-PLS-compact-numpy",
            "ASLS-AOM-compact-cv5-numpy",
            "ASLS-AOM-compact-repcv3-numpy",
            "ASLS-AOM-compact-cv3-numpy",
            "AOM-compact-cv5-numpy",
            "nirs4all-AOM-PLS-default",
        ],
    )

    lines: list[str] = []
    lines.append("# Benchmark Strategy Synthesis")
    lines.append("")
    lines.append(f"Generated on {date.today().isoformat()} from `{OUT_CSV.name}`.")
    lines.append("")
    lines.append("## Reformulation of the whole project")
    lines.append("")
    lines.append(
        "The project is an empirical search for robust NIRS prediction models across many small-to-medium spectral datasets. "
        "The current practical baseline is the `tabpfn_paper` HPO TabPFN run: TabPFN plus a broad preprocessing search. "
        "The rest of `bench/` explores whether spectroscopy-aware linear models, operator selection, kernel mixtures, CNNs, "
        "hybrids, and learnable convolutional filters can beat or complement that baseline without losing robustness."
    )
    lines.append("")
    lines.append(
        f"The merged CSV stores {len(observed)} observed/reference rows from {len(source_counts)} source families, "
        f"covering {dataset_count} eligible dataset/task pairs. It also adds derived oracle rows per dataset/model class "
        "and per dataset globally, so strategy-level visualizations can ask: if this class were allowed to pick its best "
        "executed variant per dataset, how far would it get?"
    )
    lines.append("")
    lines.append("### Oracle by model class")
    lines.append("")
    lines.append("| Model class | datasets | median rel. RMSEP vs PLS | wins vs PLS |")
    lines.append("|---|---:|---:|---:|")
    for row in class_summary[:20]:
        lines.append(
            f"| {row['model_class']} | {row['n_datasets']} | {row['median_rel_pls']:.3f} | "
            f"{row['wins_vs_pls']}/{row['n_datasets']} |"
        )
    lines.append("")
    lines.append("Interpretation: this is an optimistic oracle within each class, not a deployable protocol. It answers which strategy family contains useful models somewhere in the search space.")
    lines.append("")

    lines.append("## Synthesis of explored strategies")
    lines.append("")
    strategies = [
        ("TabPFN + preprocessing HPO", "The strongest current baseline: try many spectral corrections, reductions, and normalizations before TabPFN, then select by validation/test protocol. It is powerful because TabPFN supplies a strong small-tabular prior while preprocessing makes spectra look less pathological."),
        ("Classical PLS/Ridge references", "PLS and Ridge remain the anchors for judging progress. They are fast, stable, and define the scale of the problem; most claimed gains should be expressed as RMSEP ratios against these references."),
        ("AOM-PLS", "Adaptive operator selection before PLS. The main lesson is that compact banks plus ASLS and CV selection work better than huge banks; more operators often create selection variance instead of signal."),
        ("AOM-Ridge", "Replace the PLS head with Ridge and explore global, split-aware, local, auto-select, and blender variants. This is the most convincing non-TabPFN direction because it keeps spectral inductive bias while improving over linear baselines on many datasets."),
        ("Multi-kernel ridge / MKM", "Combine multiple preprocessing branches through kernel weighting, REML/BLUP-style mixtures, or softmax CV. Useful for testing whether averaging transformations beats selecting one; so far it is competitive but not the dominant global answer."),
        ("NICON/CNN and deep spectral models", "Several CNN architectures, distillation, low-rank, LUCAS pretraining, and residual variants were tried. Pure CNNs underperform tuned linear models globally; hybrids can help on selected small plant/chemistry datasets."),
        ("Hybrid stacking/residual models", "Stack Ridge/PLS/AOM predictions with CNN outputs or learn residual corrections. This improved over internal PLS/CNN baselines but still struggles against AOM-Ridge and TabPFN-opt at the global level."),
        ("FCK-PLS", "Learn fractional convolutional filters before a PLS solved head. It is a promising interpretable filter-learning idea, but the evidence is still narrow and not yet comparable to the 57-dataset TabPFN/AOM benchmark."),
        ("Meta-selector / MoE", "Select or combine views/operators per dataset using meta-features. This is valuable as an oracle/diagnostic tool, but it needs strict nested validation before it can be trusted as a production selector."),
    ]
    for name, body in strategies:
        lines.append(f"### {name}")
        lines.append(body)
        lines.append("")

    lines.append("## Why AOM-PLS was hidden in the first ranking")
    lines.append("")
    lines.append(
        "The first report sorted variants with `score_ratio_vs_dataset_pls`, where the denominator was the best PLS row found anywhere for a dataset, across mixed paper, AOM, multiview, and legacy runs. "
        "That answers a harsh absolute-leaderboard question, but it is not a fair protocol-local question. AOM-PLS is designed to be fast and reliable inside its own PLS/AOM benchmark protocol; comparing it to a separately tuned or paper-level PLS reference can hide that value."
    )
    lines.append("")
    lines.append(
        "The CSV now has both views. Use `score_ratio_vs_source_run_pls` for within-protocol reliability and `score_ratio_vs_dataset_pls` for the strict cross-protocol leaderboard. "
        "Under the within-protocol view, AOM-PLS does appear in the top list and the main compact/ASLS variants are clearly useful."
    )
    lines.append("")
    lines.append("| AOM-PLS checkpoint | datasets | median rel. vs source-run PLS | wins | median rel. vs global-best PLS |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in aom_check:
        source = "" if row["median_source"] is None else f"{row['median_source']:.3f}"
        global_ = "" if row["median_global"] is None else f"{row['median_global']:.3f}"
        lines.append(
            f"| {row['model_name']} | {row['n_source']} | {source} | "
            f"{row['wins_source']}/{row['n_source']} | {global_} |"
        )
    lines.append("")

    lines.append("## Top 25 best models")
    lines.append("")
    lines.append(
        "Ranking rule: for each variant, keep its best observed row per dataset, then rank by `score_ratio_vs_source_run_pls` when a source-run PLS exists, otherwise by `score_ratio_vs_dataset_pls`. "
        "This keeps AOM-PLS visible in its own fair protocol while still allowing paper reference rows and TabPFN-HPO rows to participate. It is still optimistic when a variant was rerun many times."
    )
    lines.append("")
    for idx, row in enumerate(top25, start=1):
        how, strengths, flaws = model_explanation(row["model_name"], row["model_class"])
        lines.append(f"### {idx}. {row['model_name']}")
        lines.append(f"- Class: {row['model_class']}; datasets: {row['n_datasets']}; median rel. RMSEP vs PLS: {row['median_rel_pls']:.3f}; wins: {row['wins_vs_pls']}/{row['n_datasets']}.")
        lines.append(f"- How it works: {how}")
        lines.append(f"- Strong points: {strengths}")
        lines.append(f"- Flaws: {flaws}")
        lines.append("")

    lines.append("## Actionable Waypoints")
    lines.append("")
    waypoints = [
        ("W0 - Freeze the evaluation table", "Canonicalize one row per `(dataset, model_name, protocol)` and mark exploratory rows as non-ranking. Checkpoint: every leaderboard chart can be rebuilt from `benchmark_master_results.csv` with a documented filter."),
        ("W1 - Define two leaderboards", "Keep a strict global leaderboard using `score_ratio_vs_dataset_pls`, and a protocol-local reliability leaderboard using `score_ratio_vs_source_run_pls`. Checkpoint: AOM-PLS, AOM-Ridge, TabPFN, and ensembles each have paired deltas on the same dataset set."),
        ("W2 - Freeze candidate families", "Candidate production set should be small: `TabPFN-HPO-preprocessing`, `TabPFN-opt`, `AOM-PLS-compact/ASLS-CV`, `AOMRidge-Blender`, `AOMRidge-AutoSelect`, `MKM/MKR`, and one residual/stacking candidate. Checkpoint: no new family enters until it beats one frozen candidate on a held-out global analysis."),
        ("W3 - Redo subset selection", "A subset is valid only if the model selected on it transfers to the full set. Checkpoint: across bootstrap/random subset simulations, subset-chosen top-1/top-3 models should land within 1-2 percent median RMSEP of the full-set oracle ranking and preserve class win-rate ordering."),
        ("W4 - Build nested selector", "Only after W2/W3, train a selector from dataset meta-features to choose among frozen candidates. Checkpoint: leave-one-dataset-out or repeated outer folds must beat the best single default, not the oracle."),
        ("W5 - Residual workstream", "For datasets where TabPFN and AOM-Ridge disagree strongly, inspect residuals and prediction range compression. Checkpoint: classify failures into baseline/scatter, small-n variance, y-extreme sigmoid, domain/sensor shift, or nonlinear residual."),
        ("W6 - New-model gate", "Any new synthetic/PFN/CNN idea must state which failure bucket it targets and must pass W1/W3 before more architecture iteration. Checkpoint: one page with expected lift, target datasets, runtime, and paired test outcome."),
    ]
    for title, body in waypoints:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")

    if subset_rows:
        lines.append("## Subset Selection Checkpoints")
        lines.append("")
        lines.append(
            "The subset question was redone as subset-to-global transfer, not just representativeness. "
            "A subset is accepted only if the model selected on it remains near the full-core winner."
        )
        lines.append("")
        lines.append("| Subset | Scope | Status | Spearman | Winner full rank | Regret | Winner class |")
        lines.append("|---|---|---:|---:|---:|---:|---|")
        for row in subset_rows:
            spearman = as_float(row.get("spearman_subset_vs_global"))
            rank = as_float(row.get("winner_global_rank"))
            regret = as_float(row.get("global_regret_abs"))
            lines.append(
                f"| {row.get('subset_name', '')} | {row.get('scope', '')} | {row.get('status', '')} | "
                f"{'' if spearman is None else f'{spearman:.3f}'} | "
                f"{'' if rank is None else f'{rank:.0f}'} | "
                f"{'' if regret is None else f'{regret:.4f}'} | {row.get('subset_winner_class', '')} |"
            )
        lines.append("")
        lines.append(
            "Operational conclusion: the current 10-dataset class-balanced subset is acceptable as a fast screening gate when TabPFN is included, because it selects the same full-core winner. "
            "It is not safe by itself for choosing the best non-TabPFN challenger: in `no_tabpfn` it selects a rank-9 full-core model with 0.0280 absolute median-ratio regret. "
            "Use the subset for triage, then require full-core confirmation before claiming an AOM-Ridge, AOM-PLS, MKR, or hybrid challenger wins."
        )
        lines.append("")
        lines.append("Detailed files: `bench/Subset_analysis/SUBSET_TRANSFER_REPORT.md`, `subset_transfer_summary.csv`, `subset_representativeness.csv`, and `subset_transfer_random_baselines.csv`.")
        lines.append("")

    lines.append("## Dataviz Guide")
    lines.append("")
    lines.append("A starter plotting script is available at `bench/plot_benchmark_master.py`; it writes initial figures to `bench/figures/benchmark_master/`.")
    lines.append("")
    lines.append("Start with three filters: regression only, source rows only, and successful rows only.")
    lines.append("")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append("df = pd.read_csv('bench/benchmark_master_results.csv')")
    lines.append("base = df[(df.record_type.isin(['observed', 'reference_paper'])) &")
    lines.append("          (df.task == 'regression') &")
    lines.append("          (df.score_metric == 'rmsep') &")
    lines.append("          (df.status.str.lower().isin(['ok', '']))].copy()")
    lines.append("base['rel_source_pls'] = pd.to_numeric(base.score_ratio_vs_source_run_pls, errors='coerce')")
    lines.append("base['rel_global_pls'] = pd.to_numeric(base.score_ratio_vs_dataset_pls, errors='coerce')")
    lines.append("```")
    lines.append("")
    lines.append("Recommended first plots:")
    lines.append("")
    lines.append("1. **Model-class oracle bar chart**: filter `record_type == 'oracle_by_model_class'`, plot median `score_ratio_vs_dataset_pls` by `model_class`. This answers which global strategy family has headroom.")
    lines.append("2. **Protocol-local leaderboard**: group source rows by `model_name`, take the best `rel_source_pls` per dataset, then plot median and interquartile range for models with at least 10 datasets. This is where fast AOM-PLS should be judged.")
    lines.append("3. **Strict global leaderboard**: same plot using `rel_global_pls`. This shows who beats the best PLS ever observed for the dataset, but it mixes protocols.")
    lines.append("4. **Heatmap model x dataset**: rows are the top 15 models, columns are datasets, color is `rel_source_pls` clipped to a readable range. Values below 1 beat PLS; values above 1 lose to PLS.")
    lines.append("5. **Subset-transfer chart**: x-axis subset size, y-axis full-set median regret of the model chosen on the subset. This is the key plot for the new `Subset_analysis` pass.")
    lines.append("6. **Runtime vs accuracy Pareto**: x-axis median `fit_time_s`, y-axis median `rel_source_pls`, point size `n_datasets`, color `model_class`. AOM-PLS should be evaluated here, not only on accuracy rank.")
    lines.append("")
    lines.append("## CSV notes")
    lines.append("")
    lines.append(f"- Master CSV: `{OUT_CSV.as_posix()}`")
    lines.append("- `record_type=observed` and `record_type=reference_paper` are source rows.")
    lines.append("- `record_type=source_oracle` is an oracle value already present in a source table; it is kept for audit but excluded from derived oracle calculations.")
    lines.append("- `record_type=oracle_by_model_class` is the best eligible row for a dataset/task/metric/model_class.")
    lines.append("- `record_type=oracle_global_dataset` is the best eligible row across all classes for that dataset/task/metric.")
    lines.append("- `score_ratio_vs_source_run_pls` is the protocol-local reliability normalization; lower than 1 means better than PLS in the same source/run.")
    lines.append("- `score_ratio_vs_dataset_pls` is the strict cross-protocol normalization; lower than 1 means better than the best observed PLS row for that dataset.")
    lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    records = collect_records()
    records = enrich_with_oracles(records)
    write_csv(records)
    write_md(records)
    source_rows = sum(1 for row in records if row.get("record_type") in {"observed", "reference_paper", "source_oracle"})
    oracle_rows = sum(1 for row in records if row.get("record_type", "").startswith("oracle_"))
    print(f"Wrote {OUT_CSV} with {len(records)} rows ({source_rows} source rows, {oracle_rows} derived oracle rows)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
