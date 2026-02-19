"""
Unit tests for source model selectors in meta-model stacking (Phase 6).

Tests cover:
- AllPreviousModelsSelector
- ExplicitModelSelector
- TopKByMetricSelector
- DiversitySelector
- SelectorFactory

Test scenarios include:
- Basic selection functionality
- Filtering by step, branch, fold
- Score-based ranking
- Diversity selection
- Error conditions
- Factory creation
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pytest

from nirs4all.operators.models.selection import (
    AllPreviousModelsSelector,
    DiversitySelector,
    ExplicitModelSelector,
    ModelCandidate,
    SelectorFactory,
    SourceModelSelector,
    TopKByMetricSelector,
)

# =============================================================================
# Test Fixtures and Mocks
# =============================================================================

@dataclass
class MockSelector:
    """Mock selector with branch_id."""
    branch_id: int | None = None
    branch_name: str | None = None
    branch_path: list[int] = field(default_factory=list)

@dataclass
class MockState:
    """Mock execution state."""
    step_number: int = 10
    mode: str = "train"

class MockExecutionContext:
    """Mock ExecutionContext for testing selectors."""

    def __init__(
        self,
        step_number: int = 10,
        branch_id: int | None = None,
        branch_name: str | None = None
    ):
        self.selector = MockSelector(branch_id=branch_id, branch_name=branch_name)
        self.state = MockState(step_number=step_number)
        self.custom = {}

class MockPredictionStore:
    """Mock prediction store (unused by most selectors)."""

    def filter_predictions(self, **kwargs):
        return []

def create_candidates(configs):
    """Create ModelCandidate objects from simplified config dicts.

    Args:
        configs: List of dicts with keys: name, step, fold, branch, classname, score

    Returns:
        List of ModelCandidate objects.
    """
    candidates = []
    for cfg in configs:
        candidate = ModelCandidate(
            model_name=cfg.get('name', 'Model'),
            model_classname=cfg.get('classname', 'SomeModel'),
            step_idx=cfg.get('step', 0),
            fold_id=str(cfg.get('fold', 0)) if cfg.get('fold') is not None else None,
            branch_id=cfg.get('branch'),
            branch_name=cfg.get('branch_name'),
            val_score=cfg.get('score'),
            metric=cfg.get('metric', 'rmse'),
        )
        candidates.append(candidate)
    return candidates

# =============================================================================
# ModelCandidate Tests
# =============================================================================

class TestModelCandidate:
    """Tests for ModelCandidate dataclass."""

    def test_create_candidate(self):
        """Test basic ModelCandidate creation."""
        candidate = ModelCandidate(
            model_name="PLSRegression",
            model_classname="sklearn.cross_decomposition.PLSRegression",
            step_idx=3,
            fold_id="0",
            val_score=0.92,
            metric="r2"
        )

        assert candidate.model_name == "PLSRegression"
        assert candidate.step_idx == 3
        assert candidate.fold_id == "0"
        assert candidate.val_score == 0.92

    def test_candidate_optional_fields(self):
        """Test ModelCandidate with optional fields as None."""
        candidate = ModelCandidate(
            model_name="Model",
            model_classname="SomeClass",
            step_idx=1
        )

        assert candidate.branch_id is None
        assert candidate.branch_name is None
        assert candidate.val_score is None
        assert candidate.fold_id is None

# =============================================================================
# AllPreviousModelsSelector Tests
# =============================================================================

class TestAllPreviousModelsSelector:
    """Tests for AllPreviousModelsSelector."""

    def test_select_previous_steps_only(self):
        """Test that only models from previous steps are selected."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'RF', 'step': 5, 'fold': 0},
            {'name': 'XGB', 'step': 8, 'fold': 0},
            {'name': 'Meta', 'step': 10, 'fold': 0},  # Current step
            {'name': 'Future', 'step': 12, 'fold': 0},  # After current
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'RF' in names
        assert 'XGB' in names
        assert 'Meta' not in names  # Current step excluded
        assert 'Future' not in names  # Future step excluded

    def test_exclude_averaged_folds(self):
        """Test that averaged folds are excluded by default."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'PLS', 'step': 2, 'fold': 1},
            {'name': 'PLS', 'step': 2, 'fold': 'avg'},  # Averaged
            {'name': 'PLS', 'step': 2, 'fold': 'w_avg'},  # Weighted average
        ])

        selector = AllPreviousModelsSelector(include_averaged=False)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        fold_ids = [c.fold_id for c in selected]
        assert 'avg' not in fold_ids
        assert 'w_avg' not in fold_ids
        assert len(selected) == 2

    def test_include_averaged_folds(self):
        """Test that averaged folds can be included."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'PLS', 'step': 2, 'fold': 'avg'},
        ])

        selector = AllPreviousModelsSelector(include_averaged=True)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        assert len(selected) == 2
        fold_ids = [c.fold_id for c in selected]
        assert 'avg' in fold_ids

    def test_filter_by_branch_id(self):
        """Test filtering by branch_id in context."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'branch': 0},
            {'name': 'RF', 'step': 3, 'fold': 0, 'branch': 0},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'branch': 1},  # Different branch
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10, branch_id=0)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'RF' in names
        assert 'XGB' not in names  # Different branch

    def test_no_branch_filter_when_none(self):
        """Test that all branches are included when context has no branch_id."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'branch': 0},
            {'name': 'RF', 'step': 3, 'fold': 0, 'branch': 1},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'branch': None},
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10, branch_id=None)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # All models should be selected (no branch filter)
        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'RF' in names
        assert 'XGB' in names

    def test_exclude_classnames(self):
        """Test excluding specific model class names."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'classname': 'PLSRegression'},
            {'name': 'RF', 'step': 3, 'classname': 'RandomForestRegressor'},
            {'name': 'XGB', 'step': 4, 'classname': 'XGBRegressor'},
        ])

        selector = AllPreviousModelsSelector(
            exclude_classnames={'RandomForestRegressor'}
        )
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        classnames = [c.model_classname for c in selected]
        assert 'PLSRegression' in classnames
        assert 'XGBRegressor' in classnames
        assert 'RandomForestRegressor' not in classnames

    def test_ordering_by_step(self):
        """Test that results are ordered by step index."""
        candidates = create_candidates([
            {'name': 'XGB', 'step': 5, 'fold': 0},
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'RF', 'step': 4, 'fold': 0},
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        steps = [c.step_idx for c in selected]
        assert steps == sorted(steps)

    def test_validation_empty_selection(self):
        """Test that validation fails on empty selection."""
        selector = AllPreviousModelsSelector()
        context = MockExecutionContext()

        with pytest.raises(ValueError, match="No source models selected"):
            selector.validate([], context)

# =============================================================================
# ExplicitModelSelector Tests
# =============================================================================

class TestExplicitModelSelector:
    """Tests for ExplicitModelSelector."""

    def test_select_by_names(self):
        """Test selecting models by explicit names."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'PLS', 'step': 2, 'fold': 1},
            {'name': 'RF', 'step': 3, 'fold': 0},
            {'name': 'XGB', 'step': 4, 'fold': 0},
        ])

        selector = ExplicitModelSelector(model_names=['PLS', 'XGB'])
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'XGB' in names
        assert 'RF' not in names

    def test_preserve_order(self):
        """Test that selection order matches model_names order."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'RF', 'step': 3, 'fold': 0},
            {'name': 'XGB', 'step': 4, 'fold': 0},
        ])

        # Order: XGB first, then PLS
        selector = ExplicitModelSelector(model_names=['XGB', 'PLS'])
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        # All XGB entries should come before PLS entries
        xgb_idx = [i for i, n in enumerate(names) if n == 'XGB']
        pls_idx = [i for i, n in enumerate(names) if n == 'PLS']
        assert all(x < p for x in xgb_idx for p in pls_idx)

    def test_strict_mode_missing_model_raises(self):
        """Test that strict mode raises error for missing models."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
        ])

        selector = ExplicitModelSelector(
            model_names=['PLS', 'NonExistent'],
            strict=True
        )
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        with pytest.raises(ValueError, match="not found in prediction store"):
            selector.select(candidates, context, store)

    def test_non_strict_mode_missing_model(self):
        """Test that non-strict mode allows missing models."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
        ])

        selector = ExplicitModelSelector(
            model_names=['PLS', 'NonExistent'],
            strict=False
        )
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        assert names == ['PLS']

    def test_empty_model_names_raises(self):
        """Test that empty model_names raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ExplicitModelSelector(model_names=[])

    def test_respects_step_filter(self):
        """Test that current/future steps are still filtered."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'Future', 'step': 15, 'fold': 0},  # After current step
        ])

        selector = ExplicitModelSelector(
            model_names=['PLS', 'Future'],
            strict=False
        )
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'Future' not in names

# =============================================================================
# TopKByMetricSelector Tests
# =============================================================================

class TestTopKByMetricSelector:
    """Tests for TopKByMetricSelector."""

    def test_select_top_k(self):
        """Test selecting top K models by score."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'score': 0.85},
            {'name': 'RF', 'step': 3, 'fold': 0, 'score': 0.92},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'score': 0.88},
            {'name': 'LR', 'step': 5, 'fold': 0, 'score': 0.80},
        ])

        selector = TopKByMetricSelector(k=2, metric='r2')  # Higher is better for r2
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should select RF (0.92) and XGB (0.88)
        assert len(selected) == 2
        names = [c.model_name for c in selected]
        assert 'RF' in names
        assert 'XGB' in names

    def test_infer_ascending_rmse(self):
        """Test that RMSE-like metrics infer ascending=True."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'score': 0.1},  # Best RMSE
            {'name': 'RF', 'step': 3, 'fold': 0, 'score': 0.5},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'score': 0.3},
        ])

        selector = TopKByMetricSelector(k=2, metric='rmse')  # Lower is better
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should select PLS (0.1) and XGB (0.3)
        names = [c.model_name for c in selected]
        assert 'PLS' in names
        assert 'XGB' in names
        assert 'RF' not in names

    def test_explicit_ascending(self):
        """Test explicit ascending parameter."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'score': 0.85},
            {'name': 'RF', 'step': 3, 'fold': 0, 'score': 0.92},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'score': 0.88},
        ])

        # Force ascending=True (lower is better)
        selector = TopKByMetricSelector(k=1, metric='custom', ascending=True)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should select PLS (0.85 - lowest)
        assert len(selected) == 1
        assert selected[0].model_name == 'PLS'

    def test_per_class_selection(self):
        """Test top K per class selection."""
        candidates = create_candidates([
            {'name': 'PLS_1', 'step': 2, 'fold': 0, 'classname': 'PLSRegression', 'score': 0.85},
            {'name': 'PLS_2', 'step': 2, 'fold': 1, 'classname': 'PLSRegression', 'score': 0.88},
            {'name': 'RF_1', 'step': 3, 'fold': 0, 'classname': 'RandomForest', 'score': 0.90},
            {'name': 'RF_2', 'step': 3, 'fold': 1, 'classname': 'RandomForest', 'score': 0.92},
        ])

        # Top 1 per class
        selector = TopKByMetricSelector(k=1, per_class=True)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should select best from each class: PLS_2 and RF_2
        names = [c.model_name for c in selected]
        assert len(selected) == 2
        assert 'PLS_2' in names  # Best PLS
        assert 'RF_2' in names  # Best RF

    def test_excludes_averaged_folds(self):
        """Test that averaged folds are excluded."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'score': 0.85},
            {'name': 'PLS', 'step': 2, 'fold': 'avg', 'score': 0.90},  # Averaged
        ])

        selector = TopKByMetricSelector(k=2)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should only select fold 0, not avg
        assert len(selected) == 1
        assert selected[0].fold_id == '0'

    def test_invalid_k_raises(self):
        """Test that k < 1 raises ValueError."""
        with pytest.raises(ValueError, match="k must be at least 1"):
            TopKByMetricSelector(k=0)

    def test_handles_none_scores(self):
        """Test handling of None scores."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'score': 0.85},
            {'name': 'RF', 'step': 3, 'fold': 0, 'score': None},  # No score
        ])

        selector = TopKByMetricSelector(k=5)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Only PLS should be selected (RF has no score)
        assert len(selected) == 1
        assert selected[0].model_name == 'PLS'

# =============================================================================
# DiversitySelector Tests
# =============================================================================

class TestDiversitySelector:
    """Tests for DiversitySelector."""

    def test_max_per_class(self):
        """Test limiting models per class."""
        candidates = create_candidates([
            {'name': 'PLS_1', 'step': 2, 'fold': 0, 'classname': 'PLSRegression', 'score': 0.85},
            {'name': 'PLS_2', 'step': 2, 'fold': 1, 'classname': 'PLSRegression', 'score': 0.88},
            {'name': 'PLS_3', 'step': 2, 'fold': 2, 'classname': 'PLSRegression', 'score': 0.82},
            {'name': 'RF_1', 'step': 3, 'fold': 0, 'classname': 'RandomForest', 'score': 0.90},
        ])

        selector = DiversitySelector(max_per_class=1)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should have 1 PLS (best) and 1 RF
        classnames = [c.model_classname for c in selected]
        assert classnames.count('PLSRegression') == 1
        assert classnames.count('RandomForest') == 1

    def test_preferred_classes_first(self):
        """Test that preferred classes are selected first."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0, 'classname': 'PLSRegression', 'score': 0.85},
            {'name': 'RF', 'step': 3, 'fold': 0, 'classname': 'RandomForest', 'score': 0.90},
            {'name': 'XGB', 'step': 4, 'fold': 0, 'classname': 'XGBRegressor', 'score': 0.92},
        ])

        # Prefer RandomForest
        selector = DiversitySelector(
            max_per_class=1,
            preferred_classes=['RandomForest']
        )
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # All should be selected, but we want to confirm RF is included
        names = [c.model_name for c in selected]
        assert 'RF' in names

    def test_multiple_per_class(self):
        """Test allowing multiple models per class."""
        candidates = create_candidates([
            {'name': 'PLS_1', 'step': 2, 'fold': 0, 'classname': 'PLSRegression', 'score': 0.85},
            {'name': 'PLS_2', 'step': 2, 'fold': 1, 'classname': 'PLSRegression', 'score': 0.88},
            {'name': 'PLS_3', 'step': 2, 'fold': 2, 'classname': 'PLSRegression', 'score': 0.82},
        ])

        selector = DiversitySelector(max_per_class=2)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # Should select top 2 by score (ascending, lower is better):
        # PLS_3 (0.82) and PLS_1 (0.85)
        assert len(selected) == 2
        names = [c.model_name for c in selected]
        assert 'PLS_3' in names
        assert 'PLS_1' in names
        assert 'PLS_2' not in names

    def test_invalid_max_per_class_raises(self):
        """Test that max_per_class < 1 raises ValueError."""
        with pytest.raises(ValueError, match="max_per_class must be at least 1"):
            DiversitySelector(max_per_class=0)

    def test_ordering_by_step(self):
        """Test that final selection is ordered by step."""
        candidates = create_candidates([
            {'name': 'XGB', 'step': 5, 'fold': 0, 'classname': 'XGBRegressor'},
            {'name': 'PLS', 'step': 2, 'fold': 0, 'classname': 'PLSRegression'},
            {'name': 'RF', 'step': 4, 'fold': 0, 'classname': 'RandomForest'},
        ])

        selector = DiversitySelector(max_per_class=1)
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        steps = [c.step_idx for c in selected]
        assert steps == sorted(steps)

# =============================================================================
# SelectorFactory Tests
# =============================================================================

class TestSelectorFactory:
    """Tests for SelectorFactory."""

    def test_create_all_selector(self):
        """Test creating AllPreviousModelsSelector."""
        selector = SelectorFactory.create('all')
        assert isinstance(selector, AllPreviousModelsSelector)

    def test_create_all_previous_alias(self):
        """Test 'all_previous' alias."""
        selector = SelectorFactory.create('all_previous')
        assert isinstance(selector, AllPreviousModelsSelector)

    def test_create_explicit_selector(self):
        """Test creating ExplicitModelSelector."""
        selector = SelectorFactory.create('explicit', model_names=['PLS', 'RF'])
        assert isinstance(selector, ExplicitModelSelector)
        assert selector.model_names == ['PLS', 'RF']

    def test_create_top_k_selector(self):
        """Test creating TopKByMetricSelector."""
        selector = SelectorFactory.create('top_k', k=3, metric='rmse')
        assert isinstance(selector, TopKByMetricSelector)
        assert selector.k == 3
        assert selector.metric == 'rmse'

    def test_create_topk_alias(self):
        """Test 'topk' alias."""
        selector = SelectorFactory.create('topk', k=5)
        assert isinstance(selector, TopKByMetricSelector)
        assert selector.k == 5

    def test_create_diversity_selector(self):
        """Test creating DiversitySelector."""
        selector = SelectorFactory.create('diversity', max_per_class=2)
        assert isinstance(selector, DiversitySelector)
        assert selector.max_per_class == 2

    def test_create_diverse_alias(self):
        """Test 'diverse' alias."""
        selector = SelectorFactory.create('diverse', max_per_class=1)
        assert isinstance(selector, DiversitySelector)

    def test_unknown_type_raises(self):
        """Test that unknown selector type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown selector type"):
            SelectorFactory.create('unknown_type')

    def test_case_insensitive(self):
        """Test that factory is case-insensitive."""
        selector1 = SelectorFactory.create('ALL')
        selector2 = SelectorFactory.create('Top_K', k=2)

        assert isinstance(selector1, AllPreviousModelsSelector)
        assert isinstance(selector2, TopKByMetricSelector)

    def test_register_custom_selector(self):
        """Test registering a custom selector class."""

        class CustomSelector(SourceModelSelector):
            def select(self, candidates, context, prediction_store):
                return candidates[:1]  # Just return first

        SelectorFactory.register('custom', CustomSelector)
        selector = SelectorFactory.create('custom')

        assert isinstance(selector, CustomSelector)

    def test_register_non_selector_raises(self):
        """Test that registering non-selector class raises TypeError."""

        class NotASelector:
            pass

        with pytest.raises(TypeError, match="must inherit from SourceModelSelector"):
            SelectorFactory.register('invalid', NotASelector)

# =============================================================================
# Edge Cases and Integration Tests
# =============================================================================

class TestSelectorEdgeCases:
    """Test edge cases for selectors."""

    def test_empty_candidates(self):
        """Test selection with no candidates."""
        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select([], context, store)

        assert len(selected) == 0

    def test_all_candidates_filtered(self):
        """Test when all candidates are filtered out."""
        candidates = create_candidates([
            {'name': 'Meta', 'step': 10, 'fold': 0},  # Current step
            {'name': 'Future', 'step': 15, 'fold': 0},  # Future step
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        assert len(selected) == 0

    def test_multiple_folds_same_model(self):
        """Test handling multiple folds for the same model."""
        candidates = create_candidates([
            {'name': 'PLS', 'step': 2, 'fold': 0},
            {'name': 'PLS', 'step': 2, 'fold': 1},
            {'name': 'PLS', 'step': 2, 'fold': 2},
            {'name': 'PLS', 'step': 2, 'fold': 3},
            {'name': 'PLS', 'step': 2, 'fold': 4},
        ])

        selector = AllPreviousModelsSelector()
        context = MockExecutionContext(step_number=10)
        store = MockPredictionStore()

        selected = selector.select(candidates, context, store)

        # All 5 fold entries should be selected
        assert len(selected) == 5
        assert all(c.model_name == 'PLS' for c in selected)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
