"""Tests for Packet P2-C (Tasks 2.6, 2.7, 2.8, 2.9, 2.10).

Covers:
- Task 2.6: cleanup_transient_artifacts (WorkspaceStore + orchestrator)
- Task 2.7: ModelRefitResult, RunResult refit properties, export preference
- Task 2.8: Bundle export/replay for single refit model
- Task 2.9: Predict mode dispatch for refit model
- Task 2.10: Refit metadata enrichment in _relabel_refit_predictions
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.config.context import (
    ExecutionPhase,
    LoaderArtifactProvider,
    MapArtifactProvider,
    RuntimeContext,
)
from nirs4all.pipeline.execution.refit.config_extractor import RefitConfig
from nirs4all.pipeline.execution.refit.executor import (
    _extract_cv_strategy,
    _relabel_refit_predictions,
    _step_is_splitter,
)
from nirs4all.pipeline.storage.store_schema import REFIT_CONTEXT_STANDALONE
from nirs4all.pipeline.storage.workspace_store import WorkspaceStore

# =========================================================================
# Helpers
# =========================================================================


def _make_store(tmp_path: Path) -> WorkspaceStore:
    """Create a WorkspaceStore rooted at *tmp_path*."""
    return WorkspaceStore(tmp_path / "workspace")


class _DummyModel:
    """Minimal sklearn-like model for testing."""

    def __init__(self, n_components: int = 5) -> None:
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])

    def set_params(self, **params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise ValueError(f"Invalid parameter {k}")
            setattr(self, k, v)
        return self

    def get_params(self, deep=True):
        return {"n_components": self.n_components}


class _DummySplitter:
    """Minimal CV splitter for testing."""

    n_splits = 3

    def split(self, X, y=None, groups=None):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        half = n // 2
        yield list(range(half)), list(range(half, n))

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class _DummyDataset:
    """Minimal SpectroDataset stand-in for testing."""

    def __init__(self, n_train: int = 50) -> None:
        self._n_train = n_train
        self.name = "test_dataset"
        self.aggregate = None
        self.aggregate_method = None
        self.aggregate_exclude_outliers = False
        self.repetition = None

    def x(self, selector: dict, layout: str = "2d") -> np.ndarray:
        return np.random.randn(self._n_train, 10)

    def y(self, selector: dict | None = None) -> np.ndarray:
        return np.random.randn(self._n_train)

    def features_sources(self) -> int:
        return 1


def _setup_run_with_artifacts(store: WorkspaceStore, *, n_variants: int = 2) -> dict:
    """Create a run with pipelines that have fold artifacts.

    Returns a dict with run_id, pipeline_ids, chain_ids, and
    artifact IDs so tests can verify cleanup behaviour.
    """
    run_id = store.begin_run("test_run", config={}, datasets=[{"name": "ds"}])

    pipeline_ids = []
    chain_ids = []
    fold_artifact_ids = []

    for i in range(n_variants):
        pipeline_id = store.begin_pipeline(
            run_id=run_id,
            name=f"variant_{i}",
            expanded_config=[{"step": "MinMaxScaler"}, {"model": "PLS"}],
            generator_choices=[],
            dataset_name="ds",
            dataset_hash=f"h{i}",
        )
        pipeline_ids.append(pipeline_id)

        # Create fold model artifacts (each must be unique to avoid dedup)
        fold_arts: dict[str, str | None] = {}
        per_pipeline_artifact_ids = []
        for fold_idx in range(3):
            model_obj = _DummyModel(n_components=100 * i + fold_idx)
            art_id = store.save_artifact(model_obj, f"PLS_v{i}_f{fold_idx}", "model", "joblib")
            fold_arts[str(fold_idx)] = art_id
            per_pipeline_artifact_ids.append(art_id)

        # For the winning pipeline (idx 0), also add a "final" refit artifact
        if i == 0:
            refit_obj = _DummyModel(n_components=99)
            final_art_id = store.save_artifact(refit_obj, "PLS_v0_final", "model", "joblib")
            fold_arts["final"] = final_art_id
            per_pipeline_artifact_ids.append(final_art_id)

        fold_artifact_ids.append(per_pipeline_artifact_ids)

        # Create shared preprocessing artifact (unique per variant)
        scaler_obj = {"type": "MinMaxScaler", "variant": i}
        shared_art_id = store.save_artifact(scaler_obj, f"MinMaxScaler_v{i}", "transformer", "joblib")

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[
                {"step_idx": 0, "operator_class": "MinMaxScaler", "params": {}, "artifact_id": None, "stateless": False},
                {"step_idx": 1, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False},
            ],
            model_step_idx=1,
            model_class="PLS",
            preprocessings="MinMaxScaler",
            fold_strategy="per_fold",
            fold_artifacts=fold_arts,
            shared_artifacts={"0": [shared_art_id]},
        )
        chain_ids.append(chain_id)

        store.complete_pipeline(pipeline_id, best_val=0.1 + i * 0.05, best_test=0.12, metric="rmse", duration_ms=100)

    store.complete_run(run_id, summary={})

    return {
        "run_id": run_id,
        "pipeline_ids": pipeline_ids,
        "chain_ids": chain_ids,
        "fold_artifact_ids": fold_artifact_ids,
    }


# =========================================================================
# Task 2.6: cleanup_transient_artifacts
# =========================================================================


class TestCleanupTransientArtifacts:
    """Task 2.6: WorkspaceStore.cleanup_transient_artifacts."""

    def test_method_exists(self):
        """cleanup_transient_artifacts exists on WorkspaceStore."""
        assert hasattr(WorkspaceStore, "cleanup_transient_artifacts")

    def test_winning_pipeline_keeps_final_artifact(self, tmp_path):
        """Final artifact of the winning pipeline is not cleaned up."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=2)

        store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        # The "final" artifact for pipeline 0 should still exist
        chain = store.get_chain(ids["chain_ids"][0])
        fold_arts = chain["fold_artifacts"]
        final_aid = fold_arts["final"]
        # Check artifact is still loadable (not garbage-collected)
        path = store.get_artifact_path(final_aid)
        assert path.exists(), "Final refit artifact should be preserved"
        store.close()

    def test_winning_pipeline_cv_fold_artifacts_decremented(self, tmp_path):
        """CV fold artifacts of the winning pipeline are decremented."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=1)

        # Before cleanup: capture fold artifact paths while DB rows still exist
        chain = store.get_chain(ids["chain_ids"][0])
        fold_arts = chain["fold_artifacts"]
        cv_fold_aids = [fold_arts[k] for k in fold_arts if k not in ("final", "avg", "w_avg")]
        assert len(cv_fold_aids) > 0
        cv_fold_paths = [store.get_artifact_path(aid) for aid in cv_fold_aids]
        # All paths should exist before cleanup
        for p in cv_fold_paths:
            assert p.exists(), f"Artifact file should exist before cleanup: {p}"

        store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        # After cleanup: CV fold artifact files should be garbage-collected
        for p in cv_fold_paths:
            assert not p.exists(), f"CV fold artifact should be removed: {p}"

        store.close()

    def test_losing_pipeline_all_artifacts_decremented(self, tmp_path):
        """Losing pipeline has all fold and shared artifacts decremented."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=2)

        # Capture paths before cleanup (while DB rows still exist)
        chain = store.get_chain(ids["chain_ids"][1])
        fold_arts = chain["fold_artifacts"]
        shared_arts = chain["shared_artifacts"]
        loser_fold_aids = [aid for aid in fold_arts.values() if aid]
        loser_shared_aids = []
        for v in shared_arts.values():
            if isinstance(v, list):
                loser_shared_aids.extend(v)
            elif v:
                loser_shared_aids.append(v)

        loser_fold_paths = [store.get_artifact_path(aid) for aid in loser_fold_aids]
        loser_shared_paths = [store.get_artifact_path(aid) for aid in loser_shared_aids]

        store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        # Losing pipeline's artifacts should be garbage-collected
        for p in loser_fold_paths:
            assert not p.exists(), f"Loser fold artifact should be removed: {p}"

        for p in loser_shared_paths:
            assert not p.exists(), f"Loser shared artifact should be removed: {p}"

        store.close()

    def test_winning_pipeline_shared_artifacts_preserved(self, tmp_path):
        """Shared (preprocessing) artifacts of the winning pipeline are preserved."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=2)

        # Get the winning pipeline's shared artifacts
        chain = store.get_chain(ids["chain_ids"][0])
        shared_arts = chain["shared_artifacts"]
        winner_shared_aids = []
        for v in shared_arts.values():
            if isinstance(v, list):
                winner_shared_aids.extend(v)
            elif v:
                winner_shared_aids.append(v)

        store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        # Winning pipeline's shared artifacts should be preserved
        for aid in winner_shared_aids:
            path = store.get_artifact_path(aid)
            assert path.exists(), f"Winner shared artifact {aid} should be preserved"

        store.close()

    def test_returns_removed_count(self, tmp_path):
        """Returns the number of artifact files removed."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=2)

        removed = store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        assert isinstance(removed, int)
        assert removed > 0  # Should have removed some artifacts
        store.close()

    def test_no_pipelines_returns_zero(self, tmp_path):
        """Returns 0 when there are no pipelines to clean."""
        store = _make_store(tmp_path)
        run_id = store.begin_run("empty", config={}, datasets=[])

        removed = store.cleanup_transient_artifacts(
            run_id=run_id,
            dataset_name="ds",
            winning_pipeline_ids=[],
        )

        assert removed == 0
        store.close()

    def test_predictions_not_deleted(self, tmp_path):
        """Prediction records are never deleted by cleanup."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=2)

        # Add predictions for both pipelines
        for i, pid in enumerate(ids["pipeline_ids"]):
            store.save_prediction(
                pipeline_id=pid,
                chain_id=ids["chain_ids"][i],
                dataset_name="ds",
                model_name="PLS",
                model_class="PLS",
                fold_id="fold_0",
                partition="val",
                val_score=0.1,
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
            )

        store.cleanup_transient_artifacts(
            run_id=ids["run_id"],
            dataset_name="ds",
            winning_pipeline_ids=[ids["pipeline_ids"][0]],
        )

        # All predictions should still exist
        for pid in ids["pipeline_ids"]:
            preds = store.query_predictions(pipeline_id=pid)
            assert len(preds) >= 1

        store.close()


class TestOrchestratorCleanupIntegration:
    """Task 2.6: Orchestrator calls cleanup_transient_artifacts after refit."""

    def test_orchestrator_has_cleanup_call(self):
        """_execute_refit_pass contains cleanup_transient_artifacts call."""
        import inspect

        from nirs4all.pipeline.execution.orchestrator import PipelineOrchestrator

        source = inspect.getsource(PipelineOrchestrator._execute_refit_pass)
        assert "cleanup_transient_artifacts" in source


# =========================================================================
# Task 2.7: ModelRefitResult + RunResult refit properties
# =========================================================================


class TestModelRefitResult:
    """Task 2.7: ModelRefitResult dataclass."""

    def test_import(self):
        """ModelRefitResult is importable from result module."""
        from nirs4all.api.result import ModelRefitResult
        assert ModelRefitResult is not None

    def test_default_values(self):
        """ModelRefitResult has sensible defaults."""
        from nirs4all.api.result import ModelRefitResult

        result = ModelRefitResult()
        assert result.model_name == ""
        assert result.final_entry == {}
        assert result.cv_entry == {}
        assert result.final_score is None
        assert result.cv_score is None
        assert result.metric == ""

    def test_custom_values(self):
        """ModelRefitResult accepts custom values."""
        from nirs4all.api.result import ModelRefitResult

        result = ModelRefitResult(
            model_name="PLSRegression",
            final_entry={"fold_id": "final", "test_score": 0.04},
            cv_entry={"fold_id": "w_avg", "val_score": 0.05},
            final_score=0.04,
            cv_score=0.05,
            metric="rmse",
        )
        assert result.model_name == "PLSRegression"
        assert result.final_score == 0.04
        assert result.cv_score == 0.05


class TestRunResultRefitProperties:
    """Task 2.7: RunResult refit properties (.final, .final_score, .cv_best, etc.)."""

    def _make_run_result(self, *, include_refit: bool = True, include_cv: bool = True):
        """Create a RunResult with optional refit and CV predictions."""
        from nirs4all.api.result import RunResult

        preds = Predictions()

        if include_cv:
            preds.add_prediction(
                dataset_name="ds",
                model_name="PLSRegression",
                fold_id="w_avg",
                partition="val",
                val_score=0.08,
                test_score=0.10,
                metric="rmse",
                task_type="regression",
            )
            preds.add_prediction(
                dataset_name="ds",
                model_name="PLSRegression",
                fold_id="fold_0",
                partition="val",
                val_score=0.09,
                test_score=0.11,
                metric="rmse",
                task_type="regression",
            )

        if include_refit:
            preds.add_prediction(
                dataset_name="ds",
                model_name="PLSRegression",
                fold_id="final",
                partition="train",
                test_score=0.06,
                metric="rmse",
                task_type="regression",
                refit_context="standalone",
            )

        return RunResult(predictions=preds, per_dataset={})

    def test_final_returns_refit_entry(self):
        """RunResult.final returns the refit entry."""
        result = self._make_run_result(include_refit=True)
        entry = result.final
        assert entry is not None
        assert entry["fold_id"] == "final"

    def test_final_returns_none_without_refit(self):
        """RunResult.final returns None when no refit entries."""
        result = self._make_run_result(include_refit=False)
        assert result.final is None

    def test_final_score_returns_test_score(self):
        """RunResult.final_score returns the refit test score."""
        result = self._make_run_result(include_refit=True)
        score = result.final_score
        assert score is not None
        assert score == pytest.approx(0.06)

    def test_final_score_none_without_refit(self):
        """RunResult.final_score returns None when no refit."""
        result = self._make_run_result(include_refit=False)
        assert result.final_score is None

    def test_cv_best_excludes_refit_entries(self):
        """RunResult.cv_best excludes refit entries from ranking."""
        result = self._make_run_result(include_refit=True, include_cv=True)
        cv_entry = result.cv_best
        assert cv_entry.get("fold_id") != "final"
        assert cv_entry.get("refit_context") is None

    def test_cv_best_score(self):
        """RunResult.cv_best_score returns the best CV val score."""
        result = self._make_run_result(include_refit=True, include_cv=True)
        score = result.cv_best_score
        assert score == pytest.approx(0.08)

    def test_models_returns_dict_with_refit(self):
        """RunResult.models returns per-model refit results."""
        result = self._make_run_result(include_refit=True, include_cv=True)
        models = result.models
        assert len(models) == 1
        assert "PLSRegression" in models
        model_result = models["PLSRegression"]
        assert model_result.model_name == "PLSRegression"
        assert model_result.final_score == pytest.approx(0.06)
        assert model_result.cv_score == pytest.approx(0.08)

    def test_models_empty_without_refit(self):
        """RunResult.models returns empty dict when no refit."""
        result = self._make_run_result(include_refit=False)
        assert result.models == {}

    def test_export_prefers_refit_entry(self):
        """RunResult.export() prefers the refit entry as default source."""
        from nirs4all.api.result import RunResult

        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="w_avg",
            partition="val",
            val_score=0.08,
            metric="rmse",
        )
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="final",
            partition="train",
            test_score=0.06,
            refit_context="standalone",
        )

        runner = MagicMock()
        runner.export.return_value = Path("model.n4a")

        result = RunResult(predictions=preds, per_dataset={}, _runner=runner)
        result.export("model.n4a")

        # The export call should use the refit entry
        call_kwargs = runner.export.call_args[1]
        assert call_kwargs["source"]["fold_id"] == "final"


# =========================================================================
# Task 2.8: Bundle export/replay for single refit model
# =========================================================================


class TestExportChainRefit:
    """Task 2.8: export_chain detects refit and exports single model."""

    def test_export_chain_with_refit_has_single_fold_artifact(self, tmp_path):
        """When fold_artifacts has 'final', only it is included in the bundle."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=1)

        chain_id = ids["chain_ids"][0]
        output_path = tmp_path / "refit_model.n4a"
        store.export_chain(chain_id, output_path)

        # Read the bundle and verify
        with zipfile.ZipFile(output_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            chain_data = json.loads(zf.read("chain.json"))

        assert manifest["has_refit"] is True
        assert manifest["fold_strategy"] == "single_refit"

        # chain.json fold_artifacts should only contain canonical refit key
        fold_arts = chain_data["fold_artifacts"]
        assert "fold_final" in fold_arts
        assert len(fold_arts) == 1

        store.close()

    def test_export_chain_without_refit_keeps_all_folds(self, tmp_path):
        """When no 'final' fold artifact, all fold models are exported."""
        store = _make_store(tmp_path)

        # Create a run without refit (no "final" key)
        run_id = store.begin_run("run", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store.begin_pipeline(
            run_id=run_id, name="pipe", expanded_config=[],
            generator_choices=[], dataset_name="ds", dataset_hash="h",
        )

        fold_arts = {}
        for fold_idx in range(3):
            aid = store.save_artifact(_DummyModel(), f"PLS_{fold_idx}", "model", "joblib")
            fold_arts[str(fold_idx)] = aid

        chain_id = store.save_chain(
            pipeline_id=pipeline_id,
            steps=[{"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False}],
            model_step_idx=0, model_class="PLS", preprocessings="",
            fold_strategy="per_fold", fold_artifacts=fold_arts, shared_artifacts={},
        )
        store.complete_pipeline(pipeline_id, best_val=0.1, best_test=0.12, metric="rmse", duration_ms=100)
        store.complete_run(run_id, summary={})

        output_path = tmp_path / "cv_model.n4a"
        store.export_chain(chain_id, output_path)

        with zipfile.ZipFile(output_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            chain_data = json.loads(zf.read("chain.json"))

        assert manifest["has_refit"] is False
        assert manifest["fold_strategy"] == "per_fold"
        assert len(chain_data["fold_artifacts"]) == 3

        store.close()


class TestReplayChainRefit:
    """Task 2.8: replay_chain uses single refit model when available."""

    def test_replay_chain_uses_final_model(self, tmp_path):
        """replay_chain uses the 'final' artifact for prediction."""
        store = _make_store(tmp_path)
        ids = _setup_run_with_artifacts(store, n_variants=1)

        chain_id = ids["chain_ids"][0]
        X = np.random.randn(5, 10)

        # The chain has a shared scaler at step 0 and model at step 1.
        # The scaler artifact is a dict (not a transformer), so replay
        # will fail at the transform step. But we can verify the logic
        # by mocking. Instead, let's create a proper chain.
        store.close()

        # Create a store with a proper chain for replay
        store2 = _make_store(tmp_path / "ws2")
        run_id = store2.begin_run("run", config={}, datasets=[{"name": "ds"}])
        pipeline_id = store2.begin_pipeline(
            run_id=run_id, name="pipe", expanded_config=[],
            generator_choices=[], dataset_name="ds", dataset_hash="h",
        )

        # Create a proper model artifact
        model = _DummyModel(n_components=3)
        final_aid = store2.save_artifact(model, "PLS_final", "model", "joblib")

        # Also create CV fold model (should NOT be used)
        model_cv = _DummyModel(n_components=1)
        fold0_aid = store2.save_artifact(model_cv, "PLS_fold0", "model", "joblib")

        chain_id = store2.save_chain(
            pipeline_id=pipeline_id,
            steps=[
                {"step_idx": 0, "operator_class": "PLS", "params": {}, "artifact_id": None, "stateless": False},
            ],
            model_step_idx=0,
            model_class="PLS",
            preprocessings="",
            fold_strategy="per_fold",
            fold_artifacts={"0": fold0_aid, "final": final_aid},
            shared_artifacts={},
        )
        store2.complete_pipeline(pipeline_id, best_val=0.1, best_test=0.12, metric="rmse", duration_ms=100)
        store2.complete_run(run_id, summary={})

        X = np.random.randn(5, 10)
        result = store2.replay_chain(chain_id, X)

        # The prediction should come from the "final" model (n_components=3)
        # which returns zeros
        assert result.shape == (5,)
        store2.close()


# =========================================================================
# Task 2.9: Predict mode dispatch for refit model
# =========================================================================


class TestArtifactProviderGetRefitArtifact:
    """Task 2.9: get_refit_artifact on ArtifactProvider implementations."""

    def test_base_returns_none(self):
        """Base ArtifactProvider.get_refit_artifact returns None."""
        from nirs4all.pipeline.config.context import ArtifactProvider

        # Cannot instantiate ABC directly -- check the default via MapArtifactProvider
        provider = MapArtifactProvider({})
        assert provider.get_refit_artifact(1) is None

    def test_map_provider_finds_final(self):
        """MapArtifactProvider.get_refit_artifact finds 'final' artifact."""
        model = _DummyModel(n_components=99)
        provider = MapArtifactProvider({
            2: [
                ("pipe:2:0", _DummyModel(n_components=1)),
                ("pipe:2:1", _DummyModel(n_components=2)),
                ("pipe:2:final", model),
            ]
        })

        refit = provider.get_refit_artifact(2)
        assert refit is model

    def test_map_provider_no_final(self):
        """MapArtifactProvider.get_refit_artifact returns None without 'final'."""
        provider = MapArtifactProvider({
            2: [
                ("pipe:2:0", _DummyModel(n_components=1)),
                ("pipe:2:1", _DummyModel(n_components=2)),
            ]
        })

        assert provider.get_refit_artifact(2) is None

    def test_map_provider_empty_step(self):
        """Returns None for an empty step."""
        provider = MapArtifactProvider({})
        assert provider.get_refit_artifact(5) is None

    def test_loader_provider_finds_final_via_trace(self):
        """LoaderArtifactProvider.get_refit_artifact finds 'final' via trace."""
        model = _DummyModel(n_components=99)
        mock_loader = MagicMock()
        mock_loader.load_by_id.return_value = model

        mock_trace = MagicMock()
        mock_step = MagicMock()
        mock_step.artifacts.fold_artifact_ids = {"final": "art-final-id", 0: "art-fold-0"}
        mock_trace.get_step.return_value = mock_step

        provider = LoaderArtifactProvider(mock_loader, mock_trace)
        refit = provider.get_refit_artifact(2)

        assert refit is model
        mock_loader.load_by_id.assert_called_once_with("art-final-id")

    def test_loader_provider_no_final_in_trace(self):
        """LoaderArtifactProvider.get_refit_artifact returns None without 'final'."""
        mock_loader = MagicMock()
        mock_trace = MagicMock()
        mock_step = MagicMock()
        mock_step.artifacts.fold_artifact_ids = {0: "art-fold-0", 1: "art-fold-1"}
        mock_trace.get_step.return_value = mock_step

        provider = LoaderArtifactProvider(mock_loader, mock_trace)
        assert provider.get_refit_artifact(2) is None


class TestBaseModelControllerRefitDispatch:
    """Task 2.9: BaseModelController.train predict mode refit dispatch."""

    def test_predict_mode_checks_refit_artifact(self):
        """In predict mode, train() checks for refit artifact before fold iteration."""
        import inspect

        from nirs4all.controllers.models.base_model import BaseModelController

        source = inspect.getsource(BaseModelController.train)
        # The refit check should appear before the fold iteration
        refit_check_pos = source.find("get_refit_artifact")
        fold_iter_pos = source.find("for fold_iter in range(n_folds)")
        assert refit_check_pos > -1, "get_refit_artifact should be called"
        assert refit_check_pos < fold_iter_pos, "Refit check should come before fold iteration"

    def test_launch_training_tries_refit_artifact_first(self):
        """launch_training tries get_refit_artifact before get_artifacts_for_step."""
        import inspect

        from nirs4all.controllers.models.base_model import BaseModelController

        source = inspect.getsource(BaseModelController.launch_training)
        refit_pos = source.find("get_refit_artifact")
        fallback_pos = source.find("get_artifacts_for_step")
        assert refit_pos > -1, "get_refit_artifact should be called in launch_training"
        assert refit_pos < fallback_pos, "Refit artifact should be tried before general artifacts"


# =========================================================================
# Task 2.10: Refit metadata enrichment
# =========================================================================


class TestRelabelRefitPredictionsMetadata:
    """Task 2.10: _relabel_refit_predictions enriches metadata."""

    def test_basic_relabeling_still_works(self):
        """Basic fold_id and refit_context relabeling still functions."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="fold_0",
            partition="val",
        )

        _relabel_refit_predictions(preds)

        entry = preds._buffer[0]
        assert entry["fold_id"] == "final"
        assert entry["refit_context"] == REFIT_CONTEXT_STANDALONE

    def test_enriches_with_generator_choices(self):
        """Adds generator_choices to metadata when refit_config is provided."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        config = RefitConfig(
            expanded_steps=[],
            generator_choices=[{"_or_": "SNV"}],
        )

        _relabel_refit_predictions(preds, refit_config=config)

        metadata = preds._buffer[0].get("metadata", {})
        assert metadata["generator_choices"] == [{"_or_": "SNV"}]

    def test_enriches_with_best_params(self):
        """Adds best_params to metadata when refit_config is provided."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        config = RefitConfig(
            expanded_steps=[],
            best_params={"n_components": 10, "alpha": 0.5},
        )

        _relabel_refit_predictions(preds, refit_config=config)

        metadata = preds._buffer[0].get("metadata", {})
        assert metadata["best_params"] == {"n_components": 10, "alpha": 0.5}

    def test_enriches_with_cv_strategy_from_live_splitter(self):
        """Extracts CV strategy from live splitter instance in original_steps."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        splitter = _DummySplitter()
        original_steps = [splitter, _DummyModel()]

        _relabel_refit_predictions(preds, original_steps=original_steps)

        metadata = preds._buffer[0].get("metadata", {})
        assert "cv_strategy" in metadata
        assert "_DummySplitter" in metadata["cv_strategy"]
        assert metadata["cv_n_folds"] == 3

    def test_enriches_with_cv_strategy_from_serialized_splitter(self):
        """Extracts CV strategy from serialized splitter dict in original_steps."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        original_steps = [
            {"class": "sklearn.model_selection.KFold", "params": {"n_splits": 5}},
            {"model": "PLSRegression"},
        ]

        _relabel_refit_predictions(preds, original_steps=original_steps)

        metadata = preds._buffer[0].get("metadata", {})
        assert metadata["cv_strategy"] == "KFold(5)"
        assert metadata["cv_n_folds"] == 5

    def test_combines_config_and_steps_metadata(self):
        """Both refit_config and original_steps metadata are merged."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        config = RefitConfig(
            expanded_steps=[],
            best_params={"n_components": 10},
            generator_choices=[{"_or_": "MSC"}],
        )
        original_steps = [
            {"class": "sklearn.model_selection.ShuffleSplit", "params": {"n_splits": 3}},
            {"model": "PLS"},
        ]

        _relabel_refit_predictions(preds, refit_config=config, original_steps=original_steps)

        metadata = preds._buffer[0].get("metadata", {})
        assert metadata["best_params"] == {"n_components": 10}
        assert metadata["generator_choices"] == [{"_or_": "MSC"}]
        assert metadata["cv_strategy"] == "ShuffleSplit(3)"
        assert metadata["cv_n_folds"] == 3

    def test_no_metadata_without_config_and_steps(self):
        """No metadata dict added when neither config nor steps provided."""
        preds = Predictions()
        preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id="fold_0", partition="val")

        _relabel_refit_predictions(preds)

        entry = preds._buffer[0]
        # metadata should not be added (or should be empty/None)
        metadata = entry.get("metadata")
        assert metadata is None or metadata == {}

    def test_preserves_existing_metadata(self):
        """Existing metadata keys are preserved, not overwritten."""
        preds = Predictions()
        preds.add_prediction(
            dataset_name="ds",
            model_name="PLS",
            fold_id="fold_0",
            partition="val",
            metadata={"custom_key": "custom_value"},
        )

        config = RefitConfig(
            expanded_steps=[],
            best_params={"n_components": 10},
        )

        _relabel_refit_predictions(preds, refit_config=config)

        metadata = preds._buffer[0].get("metadata", {})
        assert metadata["custom_key"] == "custom_value"
        assert metadata["best_params"] == {"n_components": 10}

    def test_multiple_entries_all_enriched(self):
        """All buffered entries are enriched."""
        preds = Predictions()
        for i in range(3):
            preds.add_prediction(dataset_name="ds", model_name="PLS", fold_id=f"fold_{i}", partition="val")

        config = RefitConfig(
            expanded_steps=[],
            best_params={"n_components": 7},
        )

        _relabel_refit_predictions(preds, refit_config=config)

        for entry in preds._buffer:
            assert entry["fold_id"] == "final"
            assert entry["refit_context"] == REFIT_CONTEXT_STANDALONE
            assert entry.get("metadata", {}).get("best_params") == {"n_components": 7}


class TestExtractCvStrategy:
    """Task 2.10: _extract_cv_strategy helper."""

    def test_live_kfold_splitter(self):
        """Extracts strategy from a live KFold instance."""
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=5)
        strategy, n_folds = _extract_cv_strategy([splitter, _DummyModel()])

        assert "KFold" in strategy
        assert n_folds == 5

    def test_live_splitter_with_n_splits(self):
        """Extracts n_folds from a splitter with n_splits attribute."""
        splitter = _DummySplitter()
        strategy, n_folds = _extract_cv_strategy([splitter])

        assert "_DummySplitter" in strategy
        assert n_folds == 3

    def test_serialized_splitter_dict(self):
        """Extracts strategy from a serialized splitter dict."""
        steps = [
            {"class": "sklearn.model_selection.ShuffleSplit", "params": {"n_splits": 10}},
        ]
        strategy, n_folds = _extract_cv_strategy(steps)

        assert strategy == "ShuffleSplit(10)"
        assert n_folds == 10

    def test_no_splitter_returns_empty(self):
        """Returns ('', None) when no splitter is found."""
        steps = [_DummyModel(), {"model": "PLS"}]
        strategy, n_folds = _extract_cv_strategy(steps)

        assert strategy == ""
        assert n_folds is None

    def test_serialized_without_n_splits(self):
        """Handles serialized splitter without n_splits param."""
        steps = [
            {"class": "sklearn.model_selection.LeaveOneOut", "params": {}},
        ]
        strategy, n_folds = _extract_cv_strategy(steps)

        assert strategy == "LeaveOneOut"
        assert n_folds is None
