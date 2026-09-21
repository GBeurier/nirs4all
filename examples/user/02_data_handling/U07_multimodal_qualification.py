"""Measure synthetic unimodal and fusion runs with identical grouped splits.

Run: python examples/user/02_data_handling/U07_multimodal_qualification.py --output /tmp/multimodal-qualification

This is a software qualification fixture, not evidence of scientific superiority.
The output contains JSON evidence, CSV predictions and a Markdown summary; no
external corpus, plot, pretrained weights or network access is required.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
from nirs4all_io import MultimodalDataset
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from U07_multimodal import make_cohort, make_pipeline

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor
from nirs4all.operators.models.sklearn.mbpls import MBPLS
from nirs4all.pipeline.config.component_serialization import serialize_component

SEEDS = (17, 23)
CASES = ("nir", "image", "series", "metadata", "early", "intermediate", "late")


def _sources(cohort: MultimodalDataset, names: list[str]) -> MultimodalDataset:
    return MultimodalDataset(
        {name: cohort.sources[name] for name in names}, sample_ids=cohort.sample_ids,
        y=cohort.y, groups=cohort.groups, partitions=cohort.partitions, name=cohort.name,
    )


def _pipeline(case: str, seed: int) -> tuple[list[Any], dict[str, Any]]:
    base = make_pipeline()[-1]["model"]
    transformers = {name: clone(transformer) for name, transformer in base.transformers.items()}
    for transformer in transformers.values():
        if "random_state" in transformer.get_params():
            transformer.set_params(random_state=seed)
    if case == "late":
        branches = {name: [transformer, Ridge(alpha=1.0)] for name, transformer in transformers.items()}
        return [GroupKFold(3), {"branch": {"by_source": True, "steps": branches}}, {"merge": "predictions"}, Ridge(alpha=1.0)], {
            "candidates": 1, "search": "none", "base_alpha": 1.0, "meta_alpha": 1.0, "inner_group_folds": 2,
        }
    if case in transformers:
        transformers = {case: transformers[case]}
    intermediate = case == "intermediate"
    model = MultimodalRegressor(
        transformers, MBPLS(n_components=2, standardize=False) if intermediate else Ridge(alpha=1.0),
        fusion="intermediate" if intermediate else "early",
    )
    grid = {"model__n_components": [1, 2]} if intermediate else {"model__alpha": [0.1, 1.0]}
    return [GroupKFold(3), {"model": model, "_grid_": grid}], {"candidates": 2, "search": "native_grid", "space": grid}


def _split_evidence(cohort: MultimodalDataset) -> dict[str, Any]:
    training = [index for index, partition in enumerate(cohort.partitions) if partition == "train"]
    testing = [index for index, partition in enumerate(cohort.partitions) if partition == "test"]
    groups = np.asarray(cohort.groups)[training]
    folds = []
    for fit, validation in GroupKFold(3).split(np.zeros((len(training), 1)), groups=groups):
        folds.append({
            "train_ids": [cohort.sample_ids[training[index]] for index in fit],
            "validation_ids": [cohort.sample_ids[training[index]] for index in validation],
        })
    return {"train_rows": len(training), "test_rows": len(testing), "train_groups": len(set(groups)),
            "test_groups": len({cohort.groups[index] for index in testing}), "folds": folds,
            "test_ids": [cohort.sample_ids[index] for index in testing]}


def _predictions(result: Any, cohort: MultimodalDataset, case: str, seed: int) -> list[dict[str, Any]]:
    winner = result.cv_best["result_metadata"]["dagml_projection"]["variant_id"]
    rows = result.predictions.filter_predictions(partition="val")
    rows = [row for row in rows if str(row["fold_id"]) in {"0", "1", "2"}
            and row["result_metadata"]["dagml_projection"]["variant_id"] == winner]
    rows += result.predictions.filter_predictions(partition="test", fold_id="final")
    records = []
    for row in rows:
        indices = row.get("sample_indices", [])
        truth = np.asarray(row["y_true"]).reshape(-1)
        predicted = np.asarray(row["y_pred"]).reshape(-1)
        for index, observed, estimate in zip(indices, truth, predicted, strict=True):
            records.append({
                "seed": seed, "case": case, "partition": row["partition"], "fold": str(row["fold_id"]),
                "sample_id": cohort.sample_ids[index], "group": cohort.groups[index],
                "y_true": float(observed), "y_pred": float(estimate), "error": float(estimate - observed),
            })
    for partition, expected in (("val", "train"), ("test", "test")):
        expected_ids = {sample for sample, role in zip(cohort.sample_ids, cohort.partitions, strict=True) if role == expected}
        actual_ids = [row["sample_id"] for row in records if row["partition"] == partition]
        if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
            raise RuntimeError(f"{case}: native {partition} prediction rows do not cover their declared samples exactly once")
    for index, fold in enumerate(_split_evidence(cohort)["folds"]):
        actual_fold_ids = {row["sample_id"] for row in records if row["partition"] == "val" and row["fold"] == str(index)}
        if actual_fold_ids != set(fold["validation_ids"]):
            raise RuntimeError(f"{case}: native fold {index} differs from the shared grouped split")
    return records


def _qualify(case: str, seed: int, cohort: MultimodalDataset, output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    names = [case] if case in cohort.sources else list(cohort.sources)
    data = _sources(cohort, names)
    pipeline, budget = _pipeline(case, seed)
    directory = output / f"seed-{seed}" / case
    directory.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    started = time.perf_counter()
    result: Any = None
    try:
        result = nirs4all.run(pipeline, data, engine="dag-ml", workspace_path=directory / "workspace",
                              random_state=seed, verbose=0, save_charts=False, refit=True)
        selected = result.runs[-1] if case == "late" else result
        rows = _predictions(selected, data, case, seed)
        archive = Path(selected.export(directory / "model.n4a"))
        test_ids = [sample for sample, role in zip(data.sample_ids, data.partitions, strict=True) if role == "test"]
        replay = nirs4all.predict(archive, data.take(test_ids), verbose=0)
        test_by_id = {row["sample_id"]: row["y_pred"] for row in rows if row["partition"] == "test"}
        np.testing.assert_allclose(np.asarray(replay.y_pred).reshape(-1), [test_by_id[sample] for sample in test_ids], rtol=1e-9, atol=1e-9)
        if replay.metadata["training_performed"] is not False:
            raise RuntimeError("archive replay unexpectedly performed training")
        metrics = {}
        for partition, label in (("val", "cv"), ("test", "test")):
            errors = np.asarray([row["error"] for row in rows if row["partition"] == partition])
            metrics[f"{label}_rmse"] = float(np.sqrt(np.mean(errors ** 2)))
            metrics[f"{label}_mae"] = float(np.mean(np.abs(errors)))
        np.testing.assert_allclose(metrics["cv_rmse"], selected.cv_best_score, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(metrics["test_rmse"], selected.best_rmse, rtol=1e-9, atol=1e-9)
        summary = {
            "seed": seed, "case": case, "sources": names, "engine": result.execution_engine,
            "budget": budget, **metrics, "elapsed_seconds": time.perf_counter() - started,
            "python_peak_bytes": tracemalloc.get_traced_memory()[1], "archive_bytes": archive.stat().st_size,
            "archive": str(archive.relative_to(output)), "replay_matches_test_predictions": True,
            "training_performed_on_reload": False, "pipeline": serialize_component(pipeline),
            "selected_variant": selected.cv_best["result_metadata"]["dagml_projection"]["variant_id"],
        }
        return summary, rows
    finally:
        if result is not None:
            result.close()
        tracemalloc.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("multimodal_qualification"))
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, cohorts = [], [], {}
    for seed in SEEDS:
        cohort = make_cohort(seed)
        cohorts[str(seed)] = {**_split_evidence(cohort), "schema": cohort.schema_descriptors(),
                              "raw_shapes": {name: list(source.values.shape) for name, source in cohort.sources.items()}}
        for case in CASES:
            summary, rows = _qualify(case, seed, cohort, output)
            summaries.append(summary)
            predictions.extend(rows)
            print(f"seed={seed} {case}: CV RMSE={summary['cv_rmse']:.5f}, test RMSE={summary['test_rmse']:.5f}", flush=True)
    report: dict[str, Any] = {
        "fixture": "deterministic synthetic software qualification; no claim of scientific superiority",
        "seeds": list(SEEDS), "selection": "native grid, minimum pooled OOF RMSE; test excluded from selection",
        "measurement": "elapsed run+export+test replay; tracemalloc Python peak bytes, not process RSS or all native allocations",
        "budget_note": "two candidates per unimodal/early/intermediate case; late has one fixed recipe and nested GroupKFold(2)",
        "target_unit": "synthetic arbitrary units", "cohorts": cohorts, "runs": summaries,
        "versions": {name: importlib.metadata.version(name) for name in ("nirs4all", "nirs4all-io", "dag-ml", "numpy", "scikit-learn")},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["seed", "case", "candidates", "cv_rmse", "test_rmse", "cv_mae", "test_mae", "elapsed_seconds", "python_peak_bytes", "archive_bytes"]
    table = [{key: row["budget"]["candidates"] if key == "candidates" else row[key] for key in columns} for row in summaries]
    for filename, records in (("summary.csv", table), ("predictions.csv", predictions)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    markdown = ["Synthetic multimodal software qualification", "", report["fixture"], "", report["measurement"], "", report["budget_note"], "",
                "| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    markdown.extend("| " + " | ".join(f"{row[key]:.6f}" if isinstance(row[key], float) else str(row[key]) for key in columns) + " |" for row in table)
    (output / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Evidence written to {output}")


if __name__ == "__main__":
    main()
