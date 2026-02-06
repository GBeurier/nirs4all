"""Tests for refit infrastructure (Tasks 2.1, 2.2, 2.3).

Covers:
- Task 2.1: refit_context column, fold_id="final", schema migration
- Task 2.2: refit_params keyword parsing and resolve_refit_params
- Task 2.3: Winning configuration extraction (extract_winning_config)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nirs4all.pipeline.storage.store_schema import (
    REFIT_CONTEXT_STACKING,
    REFIT_CONTEXT_STANDALONE,
    create_schema,
)
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Helpers
# =========================================================================

def _make_store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at *tmp_path*."""
    return WorkspaceStore(tmp_path / "workspace")


def _create_run_with_predictions(
    store: WorkspaceStore,
    *,
    n_variants: int = 1,
    best_variant_idx: int = 0,
    dataset_name: str = "wheat",
) -> dict:
    """Create a run with multiple pipeline variants and predictions.

    Returns dict with run_id, pipeline_ids, chain_ids, and pred_ids.
    """
    run_id = store.begin_run(
        "test_run",
        config={"metric": "rmse"},
        datasets=[{"name": dataset_name}],
    )

    pipeline_ids = []
    chain_ids = []
    pred_ids = []

    for i in range(n_variants):
        steps = [{"step": "MinMaxScaler"}, {"model": "PLSRegression", "params": {"n_components": 5 + i}}]
        gen_choices = [{"_or_": f"variant_{i}"}]

        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name=f"variant_{i}",
            expanded_config=steps,
            generator_choices=gen_choices,
            dataset_name=dataset_name,
            dataset_hash=f"hash_{i}",
        )
        pipeline_ids.append(pipeline_id)

        # Create a chain (required by FK constraint on predictions)
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[
                {"step_idx": 0, "operator_class": "MinMaxScaler", "params": {}, "artifact_id": None, "stateless": True},
                {"step_idx": 1, "operator_class": "PLSRegression", "params": {"n_components": 5 + i}, "artifact_id": None, "stateless": False},
            ],
            model_step_idx=1,
            model_class="sklearn.cross_decomposition.PLSRegression",
            preprocessings="MinMaxScaler",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )
        chain_ids.append(chain_id)

        # Create predictions -- the best variant gets the lowest RMSE
        val_score = 0.10 + i * 0.05 if i != best_variant_idx else 0.05
        best_params = {"n_components": 5 + i} if i == best_variant_idx else {}

        pred_id = store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name=dataset_name,
            model_name="PLSRegression",
            model_class="sklearn.cross_decomposition.PLSRegression",
            fold_id="fold_0",
            partition="val",
            val_score=val_score,
            test_score=val_score + 0.02,
            train_score=val_score - 0.03,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={"val": {"rmse": val_score}},
            best_params=best_params,
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
        )
        pred_ids.append(pred_id)

        store.complete_pipeline(
            pipeline_id=pipeline_id,
            best_val=val_score,
            best_test=val_score + 0.02,
            metric="rmse",
            duration_ms=100,
        )

    store.complete_run(run_id, summary={"total_pipelines": n_variants})

    return {
        "run_id": run_id,
        "pipeline_ids": pipeline_ids,
        "chain_ids": chain_ids,
        "pred_ids": pred_ids,
    }


# =========================================================================
# Task 2.1: Prediction Store -- refit_context column
# =========================================================================

class TestRefitContextColumn:
    """Task 2.1: refit_context column and fold_id='final' support."""

    def test_refit_context_constants(self):
        """Constants are defined correctly."""
        assert REFIT_CONTEXT_STANDALONE == "standalone"
        assert REFIT_CONTEXT_STACKING == "stacking"

    def test_predictions_table_has_refit_context_column(self):
        """The predictions table includes refit_context in its DDL."""
        conn = duckdb.connect(":memory:")
        try:
            create_schema(conn)
            result = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions' ORDER BY ordinal_position"
            ).fetchall()
            columns = [row[0] for row in result]
            assert "refit_context" in columns
        finally:
            conn.close()

    def test_refit_context_default_null(self):
        """refit_context defaults to NULL for CV entries."""
        conn = duckdb.connect(":memory:")
        try:
            create_schema(conn)
            # Insert run and pipeline first (FK constraint)
            conn.execute("INSERT INTO runs (run_id, name) VALUES ('r1', 'run')")
            conn.execute(
                "INSERT INTO pipelines (pipeline_id, run_id, name, dataset_name) "
                "VALUES ('p1', 'r1', 'pipe', 'ds1')"
            )
            # Insert prediction without specifying refit_context
            conn.execute(
                "INSERT INTO predictions "
                "(prediction_id, pipeline_id, dataset_name, model_name, "
                "model_class, fold_id, partition, metric, task_type) "
                "VALUES ('pr1', 'p1', 'ds1', 'M', 'M', 'fold_0', 'val', 'rmse', 'regression')"
            )
            row = conn.execute(
                "SELECT refit_context FROM predictions WHERE prediction_id = 'pr1'"
            ).fetchone()
            assert row[0] is None
        finally:
            conn.close()

    def test_store_save_prediction_with_refit_context(self, tmp_path):
        """WorkspaceStore.save_prediction accepts refit_context."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pipe",
            expanded_config=[],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h",
        )

        # Create a chain (required by FK constraint)
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        # Save a refit prediction with fold_id="final"
        pred_id = store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="final",
            partition="train",
            val_score=0.10,
            test_score=0.12,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            refit_context=REFIT_CONTEXT_STANDALONE,
        )

        # Query it back
        pred = store.get_prediction(pred_id)
        assert pred is not None
        assert pred["fold_id"] == "final"
        assert pred["refit_context"] == "standalone"
        store.close()

    def test_refit_and_cv_predictions_coexist(self, tmp_path):
        """CV and refit predictions coexist in the same pipeline."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pipe",
            expanded_config=[],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h",
        )

        # Create a chain (required by FK constraint)
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        # CV prediction
        cv_id = store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="fold_0",
            partition="val",
            val_score=0.12,
            test_score=0.15,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=80,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            refit_context=None,  # CV entry
        )

        # Refit prediction
        refit_id = store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="final",
            partition="train",
            val_score=0.10,
            test_score=0.12,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            refit_context=REFIT_CONTEXT_STANDALONE,
        )

        # Both should be queryable
        all_preds = store.query_predictions(pipeline_id=pipeline_id)
        assert len(all_preds) == 2

        cv_pred = store.get_prediction(cv_id)
        assert cv_pred["refit_context"] is None

        refit_pred = store.get_prediction(refit_id)
        assert refit_pred["refit_context"] == "standalone"
        store.close()

    def test_aggregated_view_excludes_refit_entries(self, tmp_path):
        """The v_aggregated_predictions view filters out refit entries."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pipe",
            expanded_config=[],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h",
        )

        # Need a chain for the aggregated view JOIN
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        # CV prediction (should appear in aggregated view)
        store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="fold_0",
            partition="val",
            val_score=0.12,
            test_score=0.15,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=80,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            refit_context=None,
        )

        # Refit prediction (should NOT appear in aggregated view)
        store.save_prediction(
            pipeline_id=pipeline_id,
            chain_id=chain_id,
            dataset_name="ds",
            model_name="PLS",
            model_class="PLS",
            fold_id="final",
            partition="train",
            val_score=0.10,
            test_score=0.12,
            train_score=0.08,
            metric="rmse",
            task_type="regression",
            n_samples=100,
            n_features=200,
            scores={},
            best_params={},
            branch_id=None,
            branch_name=None,
            exclusion_count=0,
            exclusion_rate=0.0,
            refit_context=REFIT_CONTEXT_STANDALONE,
        )

        agg = store.query_aggregated_predictions(pipeline_id=pipeline_id)
        # Should only see the CV prediction (fold_count=1, not 2)
        assert len(agg) == 1
        assert agg["fold_count"][0] == 1
        store.close()

    def test_schema_migration_adds_refit_context(self):
        """Migration adds refit_context to existing databases."""
        conn = duckdb.connect(":memory:")
        try:
            # Simulate old schema without refit_context
            old_ddl = """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id VARCHAR PRIMARY KEY,
                pipeline_id VARCHAR NOT NULL,
                fold_id VARCHAR NOT NULL,
                partition VARCHAR NOT NULL,
                val_score DOUBLE,
                metric VARCHAR NOT NULL,
                task_type VARCHAR NOT NULL
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id VARCHAR PRIMARY KEY,
                artifact_path VARCHAR NOT NULL,
                content_hash VARCHAR NOT NULL,
                operator_class VARCHAR,
                artifact_type VARCHAR,
                format VARCHAR DEFAULT 'joblib',
                size_bytes BIGINT,
                ref_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT current_timestamp
            );
            """
            for stmt in old_ddl.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)

            # Insert old-format data
            conn.execute("INSERT INTO runs (run_id, name) VALUES ('r1', 'old_run')")

            # Now run the migration
            from nirs4all.pipeline.storage.store_schema import _migrate_schema
            _migrate_schema(conn)

            # Check column was added
            result = conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'predictions'"
            ).fetchall()
            columns = [row[0] for row in result]
            assert "refit_context" in columns
        finally:
            conn.close()


# =========================================================================
# Task 2.1: Predictions class -- refit_context support
# =========================================================================

class TestPredictionsRefitContext:
    """Test that the Predictions class supports refit_context."""

    def test_add_prediction_with_refit_context(self):
        """add_prediction accepts refit_context parameter."""
        from nirs4all.data.predictions import Predictions

        preds = Predictions()
        pred_id = preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS",
            fold_id="final",
            partition="train",
            refit_context=REFIT_CONTEXT_STANDALONE,
        )

        row = preds.get_prediction_by_id(pred_id)
        assert row is not None
        assert row["refit_context"] == "standalone"
        assert row["fold_id"] == "final"

    def test_add_prediction_without_refit_context(self):
        """add_prediction defaults refit_context to None."""
        from nirs4all.data.predictions import Predictions

        preds = Predictions()
        pred_id = preds.add_prediction(
            dataset_name="wheat",
            model_name="PLS",
            fold_id="0",
            partition="val",
        )

        row = preds.get_prediction_by_id(pred_id)
        assert row is not None
        assert row["refit_context"] is None

    def test_flush_preserves_refit_context(self, tmp_path):
        """Flushing predictions to store preserves refit_context."""
        from nirs4all.data.predictions import Predictions

        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pipe",
            expanded_config=[],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h",
        )

        # Create a chain (required by FK constraint)
        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={},
            shared_artifacts={},
        )

        preds = Predictions(store=store)
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            model_classname="PLS",
            fold_id="final",
            partition="train",
            val_score=0.10,
            metric="rmse",
            task_type="regression",
            refit_context=REFIT_CONTEXT_STANDALONE,
        )

        preds.flush(pipeline_id=pipeline_id, chain_id=chain_id)

        # Query from store
        store_preds = store.query_predictions(pipeline_id=pipeline_id)
        assert len(store_preds) == 1
        assert store_preds["refit_context"][0] == "standalone"
        assert store_preds["fold_id"][0] == "final"
        store.close()


# =========================================================================
# Task 2.2: refit_params keyword support
# =========================================================================

class TestRefitParamsKeyword:
    """Task 2.2: refit_params as a recognized keyword."""

    def test_refit_params_in_reserved_keywords(self):
        """refit_params is in the parser's RESERVED_KEYWORDS."""
        from nirs4all.pipeline.steps.parser import StepParser
        assert "refit_params" in StepParser.RESERVED_KEYWORDS

    def test_step_with_refit_params_parses_correctly(self):
        """A model step with refit_params parses without error."""
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        step = {
            "model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}},
            "train_params": {"verbose": 0},
            "refit_params": {"verbose": 1, "warm_start": True},
        }

        parsed = parser.parse(step)
        assert parsed.keyword == "model"
        # refit_params should NOT be treated as an operator keyword
        assert parsed.keyword != "refit_params"

    def test_refit_params_not_treated_as_operator(self):
        """refit_params is not mistaken for an operator keyword."""
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        # Step with only refit_params and model -- model should be the keyword
        step = {
            "model": {"class": "sklearn.linear_model.Ridge"},
            "refit_params": {"warm_start": True, "warm_start_fold": "best"},
        }

        parsed = parser.parse(step)
        assert parsed.keyword == "model"

    def test_pipeline_config_preserves_refit_params(self):
        """PipelineConfigs._preprocess_steps preserves refit_params."""
        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        steps = [
            {
                "model": {"class": "sklearn.cross_decomposition.PLSRegression", "params": {"n_components": 10}},
                "train_params": {"verbose": 0},
                "refit_params": {"verbose": 1},
            }
        ]

        processed = PipelineConfigs._preprocess_steps(steps)
        # refit_params should be preserved in the step dict
        assert "refit_params" in processed[0]
        assert processed[0]["refit_params"] == {"verbose": 1}

    def test_resolve_refit_params_merges_correctly(self):
        """resolve_refit_params merges refit over train."""
        from nirs4all.pipeline.config.refit_params import resolve_refit_params

        config = {
            "train_params": {"verbose": 0, "n_jobs": 4, "learning_rate": 0.01},
            "refit_params": {"verbose": 1, "warm_start": True},
        }

        result = resolve_refit_params(config)
        assert result["verbose"] == 1  # Overridden by refit_params
        assert result["n_jobs"] == 4  # Inherited from train_params
        assert result["learning_rate"] == 0.01  # Inherited from train_params
        assert result["warm_start"] is True  # New from refit_params

    def test_resolve_refit_params_no_train_params(self):
        """resolve_refit_params works without train_params."""
        from nirs4all.pipeline.config.refit_params import resolve_refit_params

        config = {
            "refit_params": {"warm_start": True, "warm_start_fold": "best"},
        }

        result = resolve_refit_params(config)
        assert result["warm_start"] is True
        assert result["warm_start_fold"] == "best"

    def test_resolve_refit_params_no_refit_params(self):
        """resolve_refit_params returns train_params when no refit_params."""
        from nirs4all.pipeline.config.refit_params import resolve_refit_params

        config = {
            "train_params": {"verbose": 0, "n_jobs": 4},
        }

        result = resolve_refit_params(config)
        assert result == {"verbose": 0, "n_jobs": 4}

    def test_resolve_refit_params_empty_config(self):
        """resolve_refit_params returns empty dict for empty config."""
        from nirs4all.pipeline.config.refit_params import resolve_refit_params

        assert resolve_refit_params({}) == {}

    def test_resolve_refit_params_warm_start_fold_values(self):
        """resolve_refit_params supports special warm_start_fold values."""
        from nirs4all.pipeline.config.refit_params import resolve_refit_params

        for fold_spec in ("best", "last", "fold_0", "fold_3"):
            config = {"refit_params": {"warm_start_fold": fold_spec}}
            result = resolve_refit_params(config)
            assert result["warm_start_fold"] == fold_spec


# =========================================================================
# Task 2.3: Winning configuration extraction
# =========================================================================

class TestExtractWinningConfig:
    """Task 2.3: extract_winning_config function."""

    def test_single_variant_extraction(self, tmp_path):
        """Extract config from a run with a single pipeline variant."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=1, best_variant_idx=0)

        config = extract_winning_config(store, ids["run_id"])

        assert config.variant_index == 0
        assert config.pipeline_id == ids["pipeline_ids"][0]
        assert config.metric == "rmse"
        assert config.best_score == 0.05
        assert len(config.expanded_steps) > 0
        store.close()

    def test_multi_variant_selects_best(self, tmp_path):
        """Extract config from a run with multiple variants -- picks the best."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=3, best_variant_idx=1)

        config = extract_winning_config(store, ids["run_id"])

        assert config.variant_index == 1
        assert config.pipeline_id == ids["pipeline_ids"][1]
        assert config.best_score == 0.05  # Best variant always gets 0.05
        store.close()

    def test_extraction_includes_best_params(self, tmp_path):
        """Extracted config includes best_params from predictions."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=2, best_variant_idx=0)

        config = extract_winning_config(store, ids["run_id"])

        # best_variant_idx=0 gets n_components=5
        assert config.best_params.get("n_components") == 5
        store.close()

    def test_extraction_includes_generator_choices(self, tmp_path):
        """Extracted config includes generator_choices."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=2, best_variant_idx=0)

        config = extract_winning_config(store, ids["run_id"])

        assert config.generator_choices == [{"_or_": "variant_0"}]
        store.close()

    def test_extraction_includes_expanded_steps(self, tmp_path):
        """Extracted config includes the full expanded step list."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=2, best_variant_idx=0)

        config = extract_winning_config(store, ids["run_id"])

        assert isinstance(config.expanded_steps, list)
        assert len(config.expanded_steps) == 2
        # The best variant (idx=0) has n_components=5
        model_step = config.expanded_steps[1]
        assert model_step.get("params", {}).get("n_components") == 5
        store.close()

    def test_extraction_raises_on_empty_run(self, tmp_path):
        """extract_winning_config raises ValueError for empty run."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        run_id = store.begin_run("empty", config={}, datasets=[])

        with pytest.raises(ValueError, match="has no pipelines"):
            extract_winning_config(store, run_id)
        store.close()

    def test_extraction_raises_on_no_completed_pipelines(self, tmp_path):
        """extract_winning_config raises ValueError when all pipelines failed."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        run_id = store.begin_run("run", config={}, datasets=[])
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name="pipe",
            expanded_config=[],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash="h",
        )
        store.fail_pipeline(pipeline_id, "test error")

        with pytest.raises(ValueError, match="has no completed pipelines"):
            extract_winning_config(store, run_id)
        store.close()

    def test_extraction_with_explicit_metric(self, tmp_path):
        """extract_winning_config respects explicit metric argument."""
        from nirs4all.pipeline.execution.refit import extract_winning_config

        store = _make_store(tmp_path)
        ids = _create_run_with_predictions(store, n_variants=2, best_variant_idx=0)

        config = extract_winning_config(store, ids["run_id"], metric="rmse", ascending=True)

        assert config.metric == "rmse"
        assert config.pipeline_id == ids["pipeline_ids"][0]
        assert config.best_score == 0.05
        store.close()

    def test_refit_config_dataclass(self):
        """RefitConfig dataclass works correctly."""
        from nirs4all.pipeline.execution.refit import RefitConfig

        config = RefitConfig(
            expanded_steps=[{"model": "PLS"}],
            best_params={"n_components": 10},
            variant_index=2,
            generator_choices=[{"_or_": "SNV"}],
            pipeline_id="abc-123",
            metric="rmse",
            best_score=0.05,
        )

        assert config.expanded_steps == [{"model": "PLS"}]
        assert config.best_params == {"n_components": 10}
        assert config.variant_index == 2
        assert config.generator_choices == [{"_or_": "SNV"}]
        assert config.pipeline_id == "abc-123"
        assert config.metric == "rmse"
        assert config.best_score == 0.05

    def test_refit_config_defaults(self):
        """RefitConfig uses sensible defaults."""
        from nirs4all.pipeline.execution.refit import RefitConfig

        config = RefitConfig(expanded_steps=[])

        assert config.best_params == {}
        assert config.variant_index == 0
        assert config.generator_choices == []
        assert config.pipeline_id == ""
        assert config.metric == ""
        assert config.best_score == 0.0
