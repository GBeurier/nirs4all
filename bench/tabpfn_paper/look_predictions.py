"""
Explore predictions from workspace DuckDB stores.

Prints the best final model RMSEP per dataset, exactly like the
global summary printed at the end of a nirs4all.run().

Usage:
    python look_predictions.py                          # default: cwd
    python look_predictions.py --dataset "CORN*"        # filter
"""

import argparse
from fnmatch import fnmatch
from pathlib import Path

from nirs4all.data.predictions import Predictions
from nirs4all.visualization.reports import TabReportManager


def main():
    parser = argparse.ArgumentParser(description="Explore workspace DuckDB predictions")
    parser.add_argument("--dataset", "-d", default=None, help="Filter datasets by glob pattern (e.g. 'CORN*', '*LeafTraits*')")
    args = parser.parse_args()

    workspace_path = Path('.')
    if not (workspace_path / "store.duckdb").exists():
        print(f"Error: No store.duckdb found at {workspace_path}")
        return

    # Load predictions from DuckDB
    predictions = Predictions.from_workspace(workspace_path, load_arrays=False)
    print(f"Loaded {predictions.num_predictions} predictions: {len(predictions.get_datasets())} datasets, {len(predictions.get_models())} models\n")

    # Filter datasets if pattern provided
    if args.dataset:
        matching = [d for d in predictions.get_datasets() if fnmatch(d, args.dataset)]
        if not matching:
            print(f"No datasets match pattern '{args.dataset}'")
            return
        # Keep only matching dataset predictions in buffer
        predictions._buffer = [r for r in predictions._buffer if r.get("dataset_name") in matching]
        print(f"Filtered to {len(matching)} datasets matching '{args.dataset}'\n")

    # Get refit (final) entries
    refit_entries = predictions.filter_predictions(fold_id="final", load_arrays=False)
    rankable = [e for e in refit_entries if e.get("test_score") is not None]

    if not rankable:
        print("No final (refit) entries found. Showing best CV results instead.\n")
        # Fallback: show top per dataset from CV
        for ds in predictions.get_datasets():
            top = predictions.top(1, rank_partition="val", score_scope="cv", dataset_name=ds)
            if top:
                r = top[0]
                print(f"  {ds:<50} {r.get('model_name', '?'):<20} val={r.get('val_score', 0):.4f}  test={r.get('test_score', 0):.4f}  [{r.get('metric', '?')}]")
        return

    # Deduplicate: keep one entry per (dataset, model, step, config), prefer test partition
    seen: dict[tuple, dict] = {}
    for e in rankable:
        key = (e.get("dataset_name"), e.get("model_name"), e.get("step_idx"), e.get("config_name"))
        existing = seen.get(key)
        if existing is None or e.get("partition") == "test":
            seen[key] = e
    rankable = list(seen.values())

    # Enrich with CV metrics (RMSECV, ensemble test, etc.)
    metric = rankable[0].get("metric", "rmse")
    pred_index = TabReportManager._build_prediction_index(predictions)

    # Enrich w_avg test scores
    w_avg_index = pred_index.get("w_avg", {})
    for entry in rankable:
        chain_id = entry.get("chain_id")
        branch_id = entry.get("branch_id") if chain_id is None else None
        w_avg_key = (entry["dataset_name"], entry.get("config_name"), entry["model_name"], entry.get("step_idx", 0), chain_id or branch_id)
        w_avg_parts = w_avg_index.get(w_avg_key, {})
        test_entry = w_avg_parts.get("test")
        if test_entry is not None:
            entry["cv_test_score"] = test_entry.get("test_score")

    TabReportManager.enrich_refit_entries(rankable, pred_index, metric)

    datasets = set(e.get("dataset_name") for e in rankable)
    summary = TabReportManager.generate_per_model_summary(
        rankable,
        ascending=True,
        metric=metric,
        predictions=predictions,
        pred_index=pred_index,
    )

    print(f"{'=' * 120}")
    print(f"GLOBAL SUMMARY: {len(rankable)} final models across {len(datasets)} dataset(s)")
    print(f"{'=' * 120}")
    print(summary)
    print(f"{'=' * 120}")


if __name__ == "__main__":
    main()
