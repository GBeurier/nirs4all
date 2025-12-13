"""
Unit tests for meta-model serialization and prediction mode (Phase 3 & 6).

Tests cover:
- SourceModelReference serialization/deserialization
- MetaModelArtifact serialization/deserialization
- MetaModelSerializer build and validate methods
- Stacking config serialization
- Roundtrip JSON serialization
- Error handling for invalid artifacts
"""

import pytest
import json
from datetime import datetime

from nirs4all.controllers.models.stacking.serialization import (
    SourceModelReference,
    MetaModelArtifact,
    MetaModelSerializer,
    stacking_config_to_dict,
    stacking_config_from_dict,
)
from nirs4all.operators.models.meta import (
    MetaModel,
    StackingConfig,
    CoverageStrategy,
    TestAggregation,
    BranchScope,
)
from nirs4all.operators.models.selection import ModelCandidate
from nirs4all.controllers.models.stacking.exceptions import (
    InvalidMetaModelArtifactError,
)


# =============================================================================
# SourceModelReference Tests
# =============================================================================

class TestSourceModelReference:
    """Tests for SourceModelReference dataclass."""

    def test_create_reference(self):
        """Test creating a SourceModelReference."""
        ref = SourceModelReference(
            model_name="PLSRegression",
            model_classname="sklearn.cross_decomposition.PLSRegression",
            step_idx=3,
            artifact_id="pipeline:3:0",
            feature_index=0,
            fold_id="0",
            branch_id=None,
            val_score=0.92,
            metric="r2"
        )

        assert ref.model_name == "PLSRegression"
        assert ref.step_idx == 3
        assert ref.artifact_id == "pipeline:3:0"
        assert ref.feature_index == 0
        assert ref.val_score == 0.92

    def test_to_dict(self):
        """Test serializing to dictionary."""
        ref = SourceModelReference(
            model_name="RF",
            model_classname="RandomForestRegressor",
            step_idx=4,
            artifact_id="p1:4:all",
            feature_index=1,
            branch_id=0,
            branch_name="snv",
        )

        d = ref.to_dict()

        assert d['model_name'] == "RF"
        assert d['step_idx'] == 4
        assert d['artifact_id'] == "p1:4:all"
        assert d['feature_index'] == 1
        assert d['branch_id'] == 0
        assert d['branch_name'] == "snv"

    def test_from_dict(self):
        """Test deserializing from dictionary."""
        d = {
            'model_name': 'XGBoost',
            'model_classname': 'XGBRegressor',
            'step_idx': 5,
            'artifact_id': 'p1:5:0',
            'feature_index': 2,
            'val_score': 0.95,
            'metric': 'r2',
        }

        ref = SourceModelReference.from_dict(d)

        assert ref.model_name == 'XGBoost'
        assert ref.step_idx == 5
        assert ref.artifact_id == 'p1:5:0'
        assert ref.feature_index == 2
        assert ref.val_score == 0.95

    def test_roundtrip_dict(self):
        """Test dict roundtrip preserves all fields."""
        original = SourceModelReference(
            model_name="PLS",
            model_classname="PLSRegression",
            step_idx=2,
            artifact_id="test:2:1",
            feature_index=0,
            fold_id="1",
            branch_id=1,
            branch_name="snv_branch",
            branch_path=[0, 1],
            val_score=0.88,
            metric="rmse",
        )

        d = original.to_dict()
        restored = SourceModelReference.from_dict(d)

        assert restored.model_name == original.model_name
        assert restored.step_idx == original.step_idx
        assert restored.artifact_id == original.artifact_id
        assert restored.feature_index == original.feature_index
        assert restored.fold_id == original.fold_id
        assert restored.branch_id == original.branch_id
        assert restored.branch_name == original.branch_name
        assert restored.branch_path == original.branch_path
        assert restored.val_score == original.val_score
        assert restored.metric == original.metric


# =============================================================================
# MetaModelArtifact Tests
# =============================================================================

class TestMetaModelArtifact:
    """Tests for MetaModelArtifact dataclass."""

    def test_create_artifact(self):
        """Test creating a MetaModelArtifact."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                artifact_id="p:2:all",
                feature_index=0,
            ),
            SourceModelReference(
                model_name="RF",
                model_classname="RandomForestRegressor",
                step_idx=3,
                artifact_id="p:3:all",
                feature_index=1,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="MetaModel_Ridge",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],
            stacking_config={
                "coverage_strategy": "strict",
                "test_aggregation": "mean",
            },
            n_folds=5,
            coverage_ratio=1.0,
            artifact_id="p:5:all",
        )

        assert artifact.meta_model_name == "MetaModel_Ridge"
        assert len(artifact.source_models) == 2
        assert artifact.feature_columns == ["PLS_pred", "RF_pred"]
        assert artifact.n_folds == 5

    def test_to_dict(self):
        """Test serializing artifact to dictionary."""
        source_ref = SourceModelReference(
            model_name="PLS",
            model_classname="PLSRegression",
            step_idx=2,
            artifact_id="p:2:all",
            feature_index=0,
        )

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="MetaModel_Ridge",
            meta_learner_class="Ridge",
            source_models=[source_ref],
            feature_columns=["PLS_pred"],
            stacking_config={"coverage_strategy": "strict"},
            use_proba=False,
            n_folds=5,
            coverage_ratio=1.0,
            artifact_id="p:5:all",
        )

        d = artifact.to_dict()

        assert d['meta_model_type'] == "MetaModel"
        assert d['meta_model_name'] == "MetaModel_Ridge"
        assert d['meta_learner_class'] == "Ridge"
        assert len(d['source_models']) == 1
        assert d['source_models'][0]['model_name'] == "PLS"
        assert d['feature_columns'] == ["PLS_pred"]
        assert d['n_folds'] == 5

    def test_from_dict(self):
        """Test deserializing artifact from dictionary."""
        d = {
            "meta_model_type": "MetaModel",
            "meta_model_name": "MetaModel_Ridge",
            "meta_learner_class": "Ridge",
            "source_models": [
                {
                    "model_name": "PLS",
                    "model_classname": "PLSRegression",
                    "step_idx": 2,
                    "artifact_id": "p:2:all",
                    "feature_index": 0,
                }
            ],
            "feature_columns": ["PLS_pred"],
            "stacking_config": {"coverage_strategy": "strict"},
            "use_proba": False,
            "n_folds": 5,
            "coverage_ratio": 1.0,
            "artifact_id": "p:5:all",
            "training_timestamp": "2024-12-12T10:00:00Z",
        }

        artifact = MetaModelArtifact.from_dict(d)

        assert artifact.meta_model_name == "MetaModel_Ridge"
        assert len(artifact.source_models) == 1
        assert artifact.source_models[0].model_name == "PLS"
        assert artifact.n_folds == 5

    def test_to_json(self):
        """Test JSON serialization."""
        source_ref = SourceModelReference(
            model_name="PLS",
            model_classname="PLSRegression",
            step_idx=2,
            artifact_id="p:2:all",
            feature_index=0,
        )

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="MetaModel_Ridge",
            meta_learner_class="Ridge",
            source_models=[source_ref],
            feature_columns=["PLS_pred"],
            stacking_config={},
            artifact_id="test",
        )

        json_str = artifact.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['meta_model_name'] == "MetaModel_Ridge"

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '''
        {
            "meta_model_type": "MetaModel",
            "meta_model_name": "MetaModel_Ridge",
            "meta_learner_class": "Ridge",
            "source_models": [
                {
                    "model_name": "PLS",
                    "model_classname": "PLSRegression",
                    "step_idx": 2,
                    "artifact_id": "p:2:all",
                    "feature_index": 0
                }
            ],
            "feature_columns": ["PLS_pred"],
            "stacking_config": {},
            "artifact_id": "test"
        }
        '''

        artifact = MetaModelArtifact.from_json(json_str)

        assert artifact.meta_model_name == "MetaModel_Ridge"
        assert len(artifact.source_models) == 1

    def test_get_source_artifact_ids(self):
        """Test getting ordered source artifact IDs."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="",
                step_idx=2,
                artifact_id="id_pls",
                feature_index=0,
            ),
            SourceModelReference(
                model_name="RF",
                model_classname="",
                step_idx=3,
                artifact_id="id_rf",
                feature_index=1,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],
            stacking_config={},
            artifact_id="meta",
        )

        ids = artifact.get_source_artifact_ids()

        assert ids == ["id_pls", "id_rf"]

    def test_get_source_by_index(self):
        """Test getting source model by feature index."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="",
                step_idx=2,
                artifact_id="id_pls",
                feature_index=0,
            ),
            SourceModelReference(
                model_name="RF",
                model_classname="",
                step_idx=3,
                artifact_id="id_rf",
                feature_index=1,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],
            stacking_config={},
            artifact_id="meta",
        )

        ref0 = artifact.get_source_by_index(0)
        ref1 = artifact.get_source_by_index(1)
        ref_none = artifact.get_source_by_index(99)

        assert ref0.model_name == "PLS"
        assert ref1.model_name == "RF"
        assert ref_none is None

    def test_validate_feature_alignment_valid(self):
        """Test feature alignment validation with valid artifact."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="",
                step_idx=2,
                artifact_id="id_pls",
                feature_index=0,
            ),
            SourceModelReference(
                model_name="RF",
                model_classname="",
                step_idx=3,
                artifact_id="id_rf",
                feature_index=1,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],
            stacking_config={},
            artifact_id="meta",
        )

        assert artifact.validate_feature_alignment() is True

    def test_validate_feature_alignment_count_mismatch(self):
        """Test feature alignment with mismatched counts."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="",
                step_idx=2,
                artifact_id="id_pls",
                feature_index=0,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],  # 2 columns but 1 source
            stacking_config={},
            artifact_id="meta",
        )

        assert artifact.validate_feature_alignment() is False

    def test_classification_fields(self):
        """Test Phase 5 classification support fields."""
        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="LogisticRegression",
            source_models=[],
            feature_columns=[],
            stacking_config={},
            artifact_id="meta",
            task_type="multiclass_classification",
            n_classes=5,
            feature_to_model_mapping={"RF_proba_0": "RF", "RF_proba_1": "RF"},
        )

        d = artifact.to_dict()

        assert d['task_type'] == "multiclass_classification"
        assert d['n_classes'] == 5
        assert d['feature_to_model_mapping'] == {"RF_proba_0": "RF", "RF_proba_1": "RF"}

        restored = MetaModelArtifact.from_dict(d)
        assert restored.task_type == "multiclass_classification"
        assert restored.n_classes == 5


# =============================================================================
# StackingConfig Serialization Tests
# =============================================================================

class TestStackingConfigSerialization:
    """Tests for stacking config serialization functions."""

    def test_to_dict_default_config(self):
        """Test serializing default StackingConfig."""
        config = StackingConfig()
        d = stacking_config_to_dict(config)

        assert d['coverage_strategy'] == 'strict'
        assert d['test_aggregation'] == 'mean'
        assert d['branch_scope'] == 'current_only'
        assert d['allow_no_cv'] is False
        assert d['min_coverage_ratio'] == 1.0

    def test_to_dict_custom_config(self):
        """Test serializing custom StackingConfig."""
        config = StackingConfig(
            coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
            test_aggregation=TestAggregation.WEIGHTED_MEAN,
            branch_scope=BranchScope.ALL_BRANCHES,
            allow_no_cv=True,
            min_coverage_ratio=0.8,
        )
        d = stacking_config_to_dict(config)

        assert d['coverage_strategy'] == 'drop_incomplete'
        assert d['test_aggregation'] == 'weighted'
        assert d['branch_scope'] == 'all_branches'
        assert d['allow_no_cv'] is True
        assert d['min_coverage_ratio'] == 0.8

    def test_from_dict(self):
        """Test deserializing StackingConfig from dict."""
        d = {
            'coverage_strategy': 'impute_mean',
            'test_aggregation': 'best',
            'branch_scope': 'specified',
            'allow_no_cv': True,
            'min_coverage_ratio': 0.5,
        }

        config = stacking_config_from_dict(d)

        assert config.coverage_strategy == CoverageStrategy.IMPUTE_MEAN
        assert config.test_aggregation == TestAggregation.BEST_FOLD
        assert config.branch_scope == BranchScope.SPECIFIED
        assert config.allow_no_cv is True
        assert config.min_coverage_ratio == 0.5

    def test_roundtrip_config(self):
        """Test roundtrip serialization of StackingConfig."""
        original = StackingConfig(
            coverage_strategy=CoverageStrategy.IMPUTE_ZERO,
            test_aggregation=TestAggregation.MEAN,
            min_coverage_ratio=0.7,
        )

        d = stacking_config_to_dict(original)
        restored = stacking_config_from_dict(d)

        assert restored.coverage_strategy == original.coverage_strategy
        assert restored.test_aggregation == original.test_aggregation
        assert restored.min_coverage_ratio == original.min_coverage_ratio


# =============================================================================
# MetaModelSerializer Tests
# =============================================================================

class TestMetaModelSerializer:
    """Tests for MetaModelSerializer class."""

    def _create_mock_meta_operator(self):
        """Create a mock MetaModel operator."""
        from sklearn.linear_model import Ridge
        return MetaModel(
            model=Ridge(alpha=1.0),
            source_models="all",
            use_proba=False,
            stacking_config=StackingConfig(
                coverage_strategy=CoverageStrategy.STRICT,
                test_aggregation=TestAggregation.MEAN,
            ),
        )

    def _create_mock_candidates(self):
        """Create mock ModelCandidate list."""
        return [
            ModelCandidate(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                fold_id="0",
                branch_id=None,
                val_score=0.85,
                metric="r2",
            ),
            ModelCandidate(
                model_name="RF",
                model_classname="RandomForestRegressor",
                step_idx=3,
                fold_id="0",
                branch_id=None,
                val_score=0.90,
                metric="r2",
            ),
        ]

    def test_build_artifact(self):
        """Test building MetaModelArtifact from training context."""
        serializer = MetaModelSerializer()
        meta_operator = self._create_mock_meta_operator()
        source_models = self._create_mock_candidates()

        artifact = serializer.build_artifact(
            meta_operator=meta_operator,
            source_models=source_models,
            artifact_id="test:5:all",
        )

        assert artifact.meta_model_type == "MetaModel"
        assert artifact.meta_model_name == "MetaModel_Ridge"
        assert artifact.meta_learner_class == "Ridge"
        assert len(artifact.source_models) == 2
        assert artifact.source_models[0].model_name == "PLS"
        assert artifact.source_models[1].model_name == "RF"
        assert artifact.feature_columns == ["PLS_pred", "RF_pred"]
        assert artifact.use_proba is False
        assert artifact.artifact_id == "test:5:all"

    def test_build_artifact_deduplicates_sources(self):
        """Test that build_artifact deduplicates source models by name."""
        serializer = MetaModelSerializer()
        meta_operator = self._create_mock_meta_operator()

        # Same model name, different folds
        source_models = [
            ModelCandidate(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                fold_id="0",
            ),
            ModelCandidate(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                fold_id="1",
            ),
            ModelCandidate(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                fold_id="2",
            ),
        ]

        artifact = serializer.build_artifact(
            meta_operator=meta_operator,
            source_models=source_models,
            artifact_id="test",
        )

        # Should deduplicate to single PLS entry
        assert len(artifact.source_models) == 1
        assert artifact.source_models[0].model_name == "PLS"

    def test_build_artifact_with_explicit_selector(self):
        """Test building artifact with explicit source model selector config."""
        from sklearn.linear_model import Ridge

        meta_operator = MetaModel(
            model=Ridge(),
            source_models=["PLS", "RF"],  # Explicit list
        )

        serializer = MetaModelSerializer()
        source_models = self._create_mock_candidates()

        artifact = serializer.build_artifact(
            meta_operator=meta_operator,
            source_models=source_models,
            artifact_id="test",
        )

        assert artifact.selector_config is not None
        assert artifact.selector_config['type'] == 'ExplicitModelSelector'
        assert artifact.selector_config['params']['model_names'] == ['PLS', 'RF']

    def test_validate_artifact_valid(self):
        """Test validation of valid artifact."""
        serializer = MetaModelSerializer()

        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                artifact_id="p:2:all",
                feature_index=0,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="MetaModel_Ridge",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred"],
            stacking_config={"coverage_strategy": "strict"},
            artifact_id="test",
        )

        errors = serializer.validate_artifact(artifact)
        assert errors == []

    def test_validate_artifact_missing_name(self):
        """Test validation catches missing meta_model_name."""
        serializer = MetaModelSerializer()

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="",  # Missing
            meta_learner_class="Ridge",
            source_models=[],
            feature_columns=[],
            stacking_config={},
            artifact_id="test",
        )

        errors = serializer.validate_artifact(artifact)
        assert any("meta_model_name" in e for e in errors)

    def test_validate_artifact_no_sources(self):
        """Test validation catches no source models."""
        serializer = MetaModelSerializer()

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=[],  # Empty
            feature_columns=["PLS_pred"],
            stacking_config={},
            artifact_id="test",
        )

        errors = serializer.validate_artifact(artifact)
        assert any("No source models" in e for e in errors)

    def test_validate_artifact_missing_artifact_id(self):
        """Test validation catches missing artifact_id on source."""
        serializer = MetaModelSerializer()

        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                artifact_id="",  # Missing
                feature_index=0,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred"],
            stacking_config={},
            artifact_id="test",
        )

        errors = serializer.validate_artifact(artifact)
        assert any("artifact_id" in e.lower() for e in errors)

    def test_validate_artifact_feature_count_mismatch(self):
        """Test validation catches feature column count mismatch."""
        serializer = MetaModelSerializer()

        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="PLSRegression",
                step_idx=2,
                artifact_id="p:2:all",
                feature_index=0,
            ),
        ]

        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],  # 2 columns, 1 source
            stacking_config={},
            artifact_id="test",
        )

        errors = serializer.validate_artifact(artifact)
        assert any("count" in e.lower() for e in errors)


# =============================================================================
# Full Roundtrip Tests
# =============================================================================

class TestSerializationRoundtrip:
    """Test complete serialization roundtrips."""

    def test_full_artifact_json_roundtrip(self):
        """Test complete artifact JSON roundtrip."""
        source_refs = [
            SourceModelReference(
                model_name="PLS",
                model_classname="sklearn.cross_decomposition.PLSRegression",
                step_idx=2,
                artifact_id="pipeline:2:all",
                feature_index=0,
                branch_id=None,
                val_score=0.85,
                metric="r2",
            ),
            SourceModelReference(
                model_name="RF",
                model_classname="sklearn.ensemble.RandomForestRegressor",
                step_idx=3,
                artifact_id="pipeline:3:all",
                feature_index=1,
                branch_id=None,
                val_score=0.90,
                metric="r2",
            ),
        ]

        original = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="MetaModel_Ridge",
            meta_learner_class="Ridge",
            source_models=source_refs,
            feature_columns=["PLS_pred", "RF_pred"],
            stacking_config={
                "coverage_strategy": "strict",
                "test_aggregation": "mean",
                "branch_scope": "current_only",
            },
            selector_config={
                "type": "AllPreviousModelsSelector",
                "params": {},
            },
            branch_context={
                "branch_id": None,
                "branch_name": None,
            },
            use_proba=False,
            n_folds=5,
            coverage_ratio=1.0,
            artifact_id="pipeline:5:all",
            task_type="regression",
        )

        # Serialize to JSON
        json_str = original.to_json()

        # Parse JSON (to verify it's valid)
        parsed = json.loads(json_str)

        # Deserialize
        restored = MetaModelArtifact.from_json(json_str)

        # Verify all fields
        assert restored.meta_model_type == original.meta_model_type
        assert restored.meta_model_name == original.meta_model_name
        assert restored.meta_learner_class == original.meta_learner_class
        assert len(restored.source_models) == len(original.source_models)
        assert restored.feature_columns == original.feature_columns
        assert restored.stacking_config == original.stacking_config
        assert restored.use_proba == original.use_proba
        assert restored.n_folds == original.n_folds
        assert restored.coverage_ratio == original.coverage_ratio
        assert restored.artifact_id == original.artifact_id
        assert restored.task_type == original.task_type

        # Verify source model details
        for i, (orig_ref, rest_ref) in enumerate(zip(original.source_models, restored.source_models)):
            assert rest_ref.model_name == orig_ref.model_name
            assert rest_ref.model_classname == orig_ref.model_classname
            assert rest_ref.step_idx == orig_ref.step_idx
            assert rest_ref.artifact_id == orig_ref.artifact_id
            assert rest_ref.feature_index == orig_ref.feature_index

    def test_config_through_artifact_roundtrip(self):
        """Test StackingConfig through artifact roundtrip."""
        original_config = StackingConfig(
            coverage_strategy=CoverageStrategy.DROP_INCOMPLETE,
            test_aggregation=TestAggregation.WEIGHTED_MEAN,
            branch_scope=BranchScope.CURRENT_ONLY,
            min_coverage_ratio=0.8,
        )

        # Convert to dict for artifact storage
        config_dict = stacking_config_to_dict(original_config)

        # Create artifact with config dict
        artifact = MetaModelArtifact(
            meta_model_type="MetaModel",
            meta_model_name="Test",
            meta_learner_class="Ridge",
            source_models=[],
            feature_columns=[],
            stacking_config=config_dict,
            artifact_id="test",
        )

        # Serialize and deserialize artifact
        json_str = artifact.to_json()
        restored_artifact = MetaModelArtifact.from_json(json_str)

        # Restore config
        restored_config = stacking_config_from_dict(restored_artifact.stacking_config)

        assert restored_config.coverage_strategy == original_config.coverage_strategy
        assert restored_config.test_aggregation == original_config.test_aggregation
        assert restored_config.branch_scope == original_config.branch_scope
        assert restored_config.min_coverage_ratio == original_config.min_coverage_ratio


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
