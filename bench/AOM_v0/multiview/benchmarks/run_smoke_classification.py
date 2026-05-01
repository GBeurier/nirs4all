"""Classification smoke benchmark.

Quick smoke-4 for classification: wraps the regression smoke pattern but
uses balanced accuracy and applies our classifiers (AOMMoEClassifier,
BlockSparseAOMMBPLSClassifier) to a small classification cohort.

Variants:
- AOMPLSDAClassifier (compact bank, holdout) — reference
- moe-view-soft-clf
- moe-preproc-soft-clf-compact
- block-sparse-V1-clf

Outputs `bench/AOM_v0/multiview/results/smoke_classification.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(os.path.abspath(__file__)).resolve()
_MULTIVIEW_ROOT = _HERE.parent.parent
_AOM_ROOT = _MULTIVIEW_ROOT.parent
if str(_AOM_ROOT) not in sys.path:
    sys.path.insert(0, str(_AOM_ROOT))
if str(_MULTIVIEW_ROOT) not in sys.path:
    sys.path.insert(0, str(_MULTIVIEW_ROOT))

from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

from aompls.classification import AOMPLSDAClassifier  # noqa: E402
from multiview.classifiers import (  # noqa: E402
    AOMMoEClassifier,
    BlockSparseAOMMBPLSClassifier,
)

from benchmarks.run_smoke4 import (  # noqa: E402
    _load_csv_array,
    _load_csv_target,
    _existing_keys,
    _append,
    COLUMNS,
)


SMOKE_CLF_DATASETS = [
    "Beef_Impurity_60_AlJowder",
    "CoffeeType_kenstone70_strat",
    "Sporozoite2C_229_Maia_Acc94.5",
    "Genotype10_250",
]


def _run_classifier(*, Xtr, ytr, Xte, yte, clf, seed: int) -> dict:
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    fit_time = time.perf_counter() - t0
    pred = clf.predict(Xte)
    bal_acc = float(balanced_accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    return {
        "rmsep": 1.0 - bal_acc,  # use rmsep column for "error" so synth_results works
        "r2": bal_acc,
        "n_components": -1,
        "fit_time_s": float(fit_time),
        "selected_operators": f"bal_acc={bal_acc:.4f},f1_macro={f1:.4f}",
    }


def _runner_aom_pls_da(Xtr, ytr, Xte, yte, seed, max_components):
    clf = AOMPLSDAClassifier(
        n_components="auto", max_components=max_components,
        engine="simpls_covariance", selection="global",
        criterion="holdout", operator_bank="compact",
        random_state=seed,
    )
    return _run_classifier(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, clf=clf, seed=seed)


def _runner_moe_view_soft(Xtr, ytr, Xte, yte, seed, max_components):
    clf = AOMMoEClassifier(
        expert_layout="per_view", routing="soft", K=3,
        per_expert_components=min(10, max_components),
        random_state=seed,
    )
    return _run_classifier(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, clf=clf, seed=seed)


def _runner_moe_preproc_soft(Xtr, ytr, Xte, yte, seed, max_components):
    clf = AOMMoEClassifier(
        expert_layout="per_preproc", routing="soft",
        bank_name="compact",
        per_expert_components=min(10, max_components),
        random_state=seed,
    )
    return _run_classifier(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, clf=clf, seed=seed)


def _runner_block_sparse_v1(Xtr, ytr, Xte, yte, seed, max_components):
    clf = BlockSparseAOMMBPLSClassifier(
        K=3, preproc_bank_name=None, max_components=max_components,
        criterion="holdout", random_state=seed,
    )
    return _run_classifier(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, clf=clf, seed=seed)


VARIANTS = [
    ("clf-AOMPLSDA-compact-numpy", _runner_aom_pls_da),
    ("clf-moe-view-soft-pls", _runner_moe_view_soft),
    ("clf-moe-preproc-soft-pls-compact", _runner_moe_preproc_soft),
    ("clf-block-sparse-V1-blocks3", _runner_block_sparse_v1),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="bench/AOM_v0/multiview/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument(
        "--cohort",
        default="bench/AOM_v0/benchmarks/cohort_classification.csv",
    )
    args = parser.parse_args(argv)

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    results_path = workspace / "smoke_classification.csv"
    existing = _existing_keys(results_path)

    cohort = pd.read_csv(args.cohort)
    smoke = cohort[
        cohort["dataset"].isin(SMOKE_CLF_DATASETS) & (cohort["status"] == "ok")
    ].copy()
    print(
        f"[smoke-clf] {len(smoke)} datasets x {len(VARIANTS)} variants "
        f"on seed {args.seed} -> {results_path}"
    )

    n_appended = 0
    for _, cohort_row in smoke.iterrows():
        try:
            Xtr = _load_csv_array(cohort_row["train_path"])
            Xte = _load_csv_array(cohort_row["test_path"])
            ytr = _load_csv_target(cohort_row["ytrain_path"])
            yte = _load_csv_target(cohort_row["ytest_path"])
            le = LabelEncoder()
            all_y = np.concatenate([ytr.ravel(), yte.ravel()])
            le.fit(all_y)
            ytr = le.transform(ytr.ravel())
            yte = le.transform(yte.ravel())
        except Exception as exc:
            print(f"[smoke-clf] ERROR loading {cohort_row['dataset']}: {exc}")
            continue
        for label, runner in VARIANTS:
            key = (str(cohort_row["dataset"]), label, int(args.seed))
            if key in existing:
                print(f"[smoke-clf] skip {key}")
                continue
            base = {
                "database_name": cohort_row.get("database_name", ""),
                "dataset": cohort_row["dataset"],
                "variant": label,
                "seed": int(args.seed),
                "n_train": int(Xtr.shape[0]),
                "n_test": int(Xte.shape[0]),
                "n_features": int(Xtr.shape[1]),
                "status": "ok",
            }
            try:
                metrics = runner(Xtr, ytr, Xte, yte, args.seed, args.max_components)
                row = {**base, **metrics}
            except Exception as exc:
                row = {
                    **base,
                    "status": "error",
                    "status_details": str(exc)[:200],
                }
            _append(results_path, row, COLUMNS)
            existing.add(key)
            n_appended += 1
            print(
                f"[smoke-clf] {cohort_row['dataset']:<48s} {label:<40s} "
                f"err={row.get('rmsep', 'NA')} bal_acc={row.get('r2', 'NA')} "
                f"t={row.get('fit_time_s', 'NA')}"
            )

    print(f"[smoke-clf] appended {n_appended} rows -> {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
