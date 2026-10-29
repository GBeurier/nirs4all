"""Native macro reports remain authoritative in the Store-v5 projection."""
import json
import sqlite3

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

import nirs4all
from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer
from nirs4all.pipeline.dagml.target_domains import fold_local_domains_available
from nirs4all.pipeline.storage.target_domain_summary import KEY, TargetDomainComparisonError
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore


def run_native(workspace, strategy="quantile", *, seed=29):
    x = np.random.default_rng(seed).normal(size=(48, 6))
    y = 3 * x[:, 0] + .1 * x[:, 1]
    return nirs4all.run(
        [{"y_processing": IntegerKBinsDiscretizer(n_bins=3, strategy=strategy)},
         KFold(3, shuffle=True, random_state=7), LogisticRegression()],
        {"x_train": x[:36], "y_train": y[:36], "x_test": x[36:], "y_test": y[36:]},
        workspace_path=workspace, engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0,
    )


pytestmark = pytest.mark.skipif(not fold_local_domains_available(), reason="native fold-local target domains required")


def test_real_native_macro_copied_exactly_without_extra_fold_or_pooled_scores(tmp_path, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = run_native(tmp_path)
    macro = result.cv_best
    expected_context = macro["result_metadata"]["dagml_projection"]["target_domain_context"]
    with WorkspaceStore(tmp_path) as store:
        summary = store.query_chain_summaries().row(0, named=True)
        assert summary["cv_fold_count"] == 3
        assert summary["cv_val_score"] == macro["val_score"]
        assert summary["cv_test_score"] is None and summary["cv_train_score"] is None
        assert json.loads(summary["cv_scores"]) == macro["scores"]
        assert summary["target_domain_context"] == expected_context
        assert summary["cv_summary_fold_id"] == "macro"
        assert store._fetch_one("SELECT COUNT(*) AS count FROM predictions")["count"] == 12
        run_id = summary["run_id"]
        metadata = store.get_run(run_id)["summary"]
        assert metadata[KEY] == {"schema_version": 1, "chains": {summary["chain_id"]: {
            "cv_summary_fold_id": "macro", "target_domain_context": expected_context}}}
        store.update_chain_summary(summary["chain_id"])
        store.bulk_update_chain_summaries([summary["chain_id"]])
        assert store.query_chain_summaries().row(0, named=True)["cv_val_score"] == macro["val_score"]
        # Parent-run completion must not discard child-pipeline provenance.
        store.complete_run(run_id, {"parent_completed": True})
        assert store.get_run(run_id)["summary"][KEY] == metadata[KEY]
        ranked = store.query_top_chains(metric="balanced_accuracy").row(0, named=True)
        assert ranked["target_domain_context"] == expected_context
        with pytest.raises(TargetDomainComparisonError, match="individual fold predictions"):
            store.top_predictions(n=10)
        with pytest.raises(TargetDomainComparisonError, match="aggregate prediction arrays"):
            store.query_aggregated_predictions()
    result.close()


def test_real_same_context_ranked_but_different_recipes_refused_before_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    first = run_native(tmp_path)
    second = run_native(tmp_path)
    with WorkspaceStore(tmp_path) as store:
        assert store.query_top_chains(metric="balanced_accuracy", n=10).height == 2
        assert store.query_top_aggregated_predictions(metric="balanced_accuracy", n=10).height == 2
    x = np.random.default_rng(29).normal(size=(48, 6))
    ordinary = nirs4all.run(
        [KFold(3, shuffle=True, random_state=7), LogisticRegression()],
        {"x_train": x[:36], "y_train": (x[:36, 0] > 0).astype(int),
         "x_test": x[36:], "y_test": (x[36:, 0] > 0).astype(int)},
        workspace_path=tmp_path, engine="dag-ml", save_artifacts=False, save_charts=False, verbose=0,
    )
    with WorkspaceStore(tmp_path) as store:
        with pytest.raises(TargetDomainComparisonError, match="Cannot rank results"):
            store.query_top_chains(metric="balanced_accuracy", n=1)
    third = run_native(tmp_path, "uniform")
    with WorkspaceStore(tmp_path) as store:
        for query in (store.query_top_chains, store.query_top_aggregated_predictions):
            with pytest.raises(TargetDomainComparisonError, match="Cannot rank results"):
                query(metric="balanced_accuracy", n=1)
        first_run = first.cv_best["chain_id"]
        assert store.query_chain_summaries(chain_id=first_run).height == 1
    for result in (first, second, ordinary, third):
        result.close()


def test_existing_macro_without_context_remains_visible_but_cannot_be_ranked(tmp_path, monkeypatch):
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = run_native(tmp_path)
    with WorkspaceStore(tmp_path) as store:
        store._execute_with_retry("UPDATE runs SET summary='{}'")
        assert store._fetch_one("SELECT COUNT(*) AS count FROM predictions")["count"] == 12
        assert store.query_predictions().height == 12
        with pytest.raises(TargetDomainComparisonError, match="lack captured target-domain context"):
            store.query_top_chains(metric="balanced_accuracy", n=1)
    result.close()


def test_context_validation_and_rank_query_share_one_snapshot(tmp_path, monkeypatch):
    from nirs4all.pipeline.storage import target_domain_summary

    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    result = run_native(tmp_path)
    validate = target_domain_summary.validate_comparison

    def concurrent_writer(contexts):
        validate(contexts)
        with sqlite3.connect(tmp_path / "store.sqlite") as other:
            other.execute("UPDATE runs SET summary='{}'")

    with WorkspaceStore(tmp_path) as store:
        monkeypatch.setattr(target_domain_summary, "validate_comparison", concurrent_writer)
        row = store.query_top_chains(metric="balanced_accuracy").row(0, named=True)
        assert row["target_domain_context"] == result.cv_best["result_metadata"]["dagml_projection"]["target_domain_context"]
        monkeypatch.setattr(target_domain_summary, "validate_comparison", validate)
        with pytest.raises(TargetDomainComparisonError, match="lack captured target-domain context"):
            store.query_top_chains(metric="balanced_accuracy")
    result.close()
