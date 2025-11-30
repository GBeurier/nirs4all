"""
Unit tests for stacking and ensemble model support in nirs4all.

Tests cover:
- Meta-estimator detection (is_meta_estimator)
- Serialization of stacking/voting models with nested estimators
- Deserialization and reconstruction of meta-estimators
- Clone functionality for meta-estimators
- Factory rebuild with force_params
"""

import pytest
import json
import copy

from sklearn.ensemble import (
    StackingRegressor, StackingClassifier,
    VotingRegressor, VotingClassifier,
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nirs4all.controllers.models.factory import ModelFactory
from nirs4all.pipeline.config.component_serialization import (
    serialize_component, deserialize_component,
    _is_meta_estimator, _is_meta_estimator_class,
    _serialize_meta_estimator, _deserialize_meta_estimator
)


class TestMetaEstimatorDetection:
    """Test detection of stacking/voting meta-estimators."""

    def test_is_meta_estimator_stacking_regressor(self):
        """Test detection of StackingRegressor instance."""
        stacking = StackingRegressor(
            estimators=[('pls', PLSRegression(n_components=5))],
            final_estimator=Ridge()
        )
        assert ModelFactory.is_meta_estimator(stacking) is True
        assert _is_meta_estimator(stacking) is True

    def test_is_meta_estimator_stacking_classifier(self):
        """Test detection of StackingClassifier instance."""
        stacking = StackingClassifier(
            estimators=[('rf', RandomForestClassifier(n_estimators=10))],
            final_estimator=LogisticRegression()
        )
        assert ModelFactory.is_meta_estimator(stacking) is True
        assert _is_meta_estimator(stacking) is True

    def test_is_meta_estimator_voting_regressor(self):
        """Test detection of VotingRegressor instance."""
        voting = VotingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=10))
            ]
        )
        assert ModelFactory.is_meta_estimator(voting) is True
        assert _is_meta_estimator(voting) is True

    def test_is_meta_estimator_voting_classifier(self):
        """Test detection of VotingClassifier instance."""
        voting = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=10)),
                ('lda', LinearDiscriminantAnalysis())
            ],
            voting='soft'
        )
        assert ModelFactory.is_meta_estimator(voting) is True
        assert _is_meta_estimator(voting) is True

    def test_is_not_meta_estimator_pls(self):
        """Test that PLSRegression is not detected as meta-estimator."""
        pls = PLSRegression(n_components=10)
        assert ModelFactory.is_meta_estimator(pls) is False
        assert _is_meta_estimator(pls) is False

    def test_is_not_meta_estimator_rf(self):
        """Test that RandomForestRegressor is not detected as meta-estimator."""
        rf = RandomForestRegressor(n_estimators=50)
        assert ModelFactory.is_meta_estimator(rf) is False
        assert _is_meta_estimator(rf) is False

    def test_is_meta_estimator_none(self):
        """Test that None returns False."""
        assert ModelFactory.is_meta_estimator(None) is False

    def test_is_meta_estimator_dict_with_estimators(self):
        """Test detection via dict with estimators key."""
        config = {'params': {'estimators': []}}
        assert ModelFactory.is_meta_estimator(config) is True

    def test_is_meta_estimator_dict_with_model_instance(self):
        """Test detection via dict with model_instance key."""
        stacking = StackingRegressor(
            estimators=[('pls', PLSRegression())],
            final_estimator=Ridge()
        )
        config = {'model_instance': stacking}
        assert ModelFactory.is_meta_estimator(config) is True

    def test_is_meta_estimator_class_check(self):
        """Test _is_meta_estimator_class function."""
        assert _is_meta_estimator_class(StackingRegressor) is True
        assert _is_meta_estimator_class(StackingClassifier) is True
        assert _is_meta_estimator_class(VotingRegressor) is True
        assert _is_meta_estimator_class(VotingClassifier) is True
        assert _is_meta_estimator_class(PLSRegression) is False
        assert _is_meta_estimator_class(RandomForestRegressor) is False


class TestMetaEstimatorSerialization:
    """Test serialization of stacking/voting models."""

    def test_serialize_stacking_regressor(self):
        """Test serialization of StackingRegressor with nested estimators."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10))
            ],
            final_estimator=Ridge(alpha=0.5),
            cv=3
        )

        serialized = serialize_component(stacking)

        # Check structure
        assert isinstance(serialized, dict)
        assert 'class' in serialized
        assert 'sklearn.ensemble._stacking.StackingRegressor' in serialized['class']
        assert 'params' in serialized

        # Check estimators serialized
        assert 'estimators' in serialized['params']
        estimators = serialized['params']['estimators']
        assert len(estimators) == 2
        assert estimators[0][0] == 'pls'
        assert estimators[1][0] == 'rf'

        # Check nested estimator params
        pls_config = estimators[0][1]
        assert isinstance(pls_config, dict)
        assert 'params' in pls_config
        assert pls_config['params']['n_components'] == 5

        # Check final_estimator serialized
        assert 'final_estimator' in serialized['params']
        final_est = serialized['params']['final_estimator']
        assert isinstance(final_est, dict)
        assert 'params' in final_est
        assert final_est['params']['alpha'] == 0.5

        # Check cv param
        assert serialized['params']['cv'] == 3

    def test_serialize_voting_classifier(self):
        """Test serialization of VotingClassifier."""
        voting = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50)),
                ('lda', LinearDiscriminantAnalysis())
            ],
            voting='soft'
        )

        serialized = serialize_component(voting)

        assert isinstance(serialized, dict)
        assert 'sklearn.ensemble._voting.VotingClassifier' in serialized['class']
        assert 'estimators' in serialized['params']
        assert len(serialized['params']['estimators']) == 2
        assert serialized['params']['voting'] == 'soft'

    def test_serialized_is_json_compatible(self):
        """Test that serialized output is JSON-compatible."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=50))
            ],
            final_estimator=Ridge()
        )

        serialized = serialize_component(stacking)

        # Should not raise
        json_str = json.dumps(serialized, indent=2)
        parsed = json.loads(json_str)

        assert 'class' in parsed
        assert 'params' in parsed
        assert 'estimators' in parsed['params']


class TestMetaEstimatorDeserialization:
    """Test deserialization of stacking/voting models."""

    def test_deserialize_stacking_regressor(self):
        """Test deserialization reconstructs StackingRegressor correctly."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10))
            ],
            final_estimator=Ridge(alpha=0.5),
            cv=3
        )

        serialized = serialize_component(stacking)
        deserialized = deserialize_component(serialized)

        # Check type
        assert isinstance(deserialized, StackingRegressor)

        # Check estimators
        assert len(deserialized.estimators) == 2
        assert deserialized.estimators[0][0] == 'pls'
        assert deserialized.estimators[1][0] == 'rf'

        # Check nested estimator types
        assert isinstance(deserialized.estimators[0][1], PLSRegression)
        assert isinstance(deserialized.estimators[1][1], RandomForestRegressor)

        # Check nested estimator params
        assert deserialized.estimators[0][1].n_components == 5
        assert deserialized.estimators[1][1].n_estimators == 100
        assert deserialized.estimators[1][1].max_depth == 10

        # Check final_estimator
        assert isinstance(deserialized.final_estimator, Ridge)
        assert deserialized.final_estimator.alpha == 0.5

        # Check cv
        assert deserialized.cv == 3

    def test_deserialize_voting_classifier(self):
        """Test deserialization reconstructs VotingClassifier correctly."""
        voting = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50)),
                ('lda', LinearDiscriminantAnalysis())
            ],
            voting='soft'
        )

        serialized = serialize_component(voting)
        deserialized = deserialize_component(serialized)

        assert isinstance(deserialized, VotingClassifier)
        assert len(deserialized.estimators) == 2
        assert deserialized.voting == 'soft'

    def test_roundtrip_json_serialization(self):
        """Test full JSON roundtrip: serialize -> JSON -> parse -> deserialize."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=10)),
                ('gbr', GradientBoostingRegressor(n_estimators=50, max_depth=5))
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=5
        )

        # Serialize to JSON string
        serialized = serialize_component(stacking)
        json_str = json.dumps(serialized)

        # Parse JSON and deserialize
        parsed = json.loads(json_str)
        deserialized = deserialize_component(parsed)

        # Verify
        assert isinstance(deserialized, StackingRegressor)
        assert len(deserialized.estimators) == 2
        assert deserialized.estimators[0][0] == 'pls'
        assert deserialized.estimators[0][1].n_components == 10
        assert deserialized.cv == 5

    def test_deserialize_meta_estimator_function(self):
        """Test _deserialize_meta_estimator helper function."""
        params = {
            'estimators': [
                ['pls', {'class': 'sklearn.cross_decomposition._pls.PLSRegression', 'params': {'n_components': 5}}]
            ],
            'final_estimator': {'class': 'sklearn.linear_model._ridge.Ridge', 'params': {'alpha': 0.5}},
            'cv': 3
        }

        result = _deserialize_meta_estimator(StackingRegressor, params)

        assert isinstance(result, StackingRegressor)
        assert len(result.estimators) == 1
        assert result.cv == 3


class TestMetaEstimatorClone:
    """Test cloning of meta-estimators."""

    def test_deepcopy_stacking_regressor(self):
        """Test that deepcopy works for StackingRegressor."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=50))
            ],
            final_estimator=Ridge()
        )

        cloned = copy.deepcopy(stacking)

        # Should be different objects
        assert cloned is not stacking
        assert cloned.estimators is not stacking.estimators

        # But same structure
        assert len(cloned.estimators) == len(stacking.estimators)
        assert cloned.estimators[0][0] == stacking.estimators[0][0]

        # Nested estimators should also be copies
        assert cloned.estimators[0][1] is not stacking.estimators[0][1]
        assert cloned.estimators[0][1].n_components == stacking.estimators[0][1].n_components

    def test_deepcopy_voting_classifier(self):
        """Test that deepcopy works for VotingClassifier."""
        voting = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=50)),
                ('lda', LinearDiscriminantAnalysis())
            ],
            voting='hard'
        )

        cloned = copy.deepcopy(voting)

        assert cloned is not voting
        assert len(cloned.estimators) == 2
        assert cloned.voting == 'hard'


class TestMetaEstimatorFactoryRebuild:
    """Test ModelFactory rebuild functionality for meta-estimators."""

    def test_rebuild_meta_estimator_with_cv(self):
        """Test _rebuild_meta_estimator with cv parameter."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
            ],
            final_estimator=Ridge(),
            cv=3
        )

        rebuilt = ModelFactory._rebuild_meta_estimator(stacking, {'cv': 5})

        assert isinstance(rebuilt, StackingRegressor)
        assert rebuilt.cv == 5
        # Estimators should be preserved
        assert len(rebuilt.estimators) == 1
        assert rebuilt.estimators[0][0] == 'pls'

    def test_rebuild_meta_estimator_with_n_jobs(self):
        """Test _rebuild_meta_estimator with n_jobs parameter."""
        voting = VotingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=50))
            ],
            n_jobs=1
        )

        rebuilt = ModelFactory._rebuild_meta_estimator(voting, {'n_jobs': -1})

        assert isinstance(rebuilt, VotingRegressor)
        assert rebuilt.n_jobs == -1
        assert len(rebuilt.estimators) == 2

    def test_from_instance_with_meta_estimator(self):
        """Test _from_instance handles meta-estimators correctly."""
        stacking = StackingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
            ],
            final_estimator=Ridge(),
            cv=3
        )

        result = ModelFactory._from_instance(stacking, {'cv': 10})

        assert isinstance(result, StackingRegressor)
        assert result.cv == 10
        assert len(result.estimators) == 1

    def test_from_instance_no_force_params(self):
        """Test _from_instance without force_params returns same instance."""
        stacking = StackingRegressor(
            estimators=[('pls', PLSRegression())],
            final_estimator=Ridge()
        )

        result = ModelFactory._from_instance(stacking, None)
        assert result is stacking

        result2 = ModelFactory._from_instance(stacking, {})
        assert result2 is stacking


class TestMetaEstimatorEdgeCases:
    """Test edge cases for meta-estimator support."""

    def test_stacking_with_default_final_estimator(self):
        """Test stacking with None final_estimator (uses default)."""
        stacking = StackingRegressor(
            estimators=[('pls', PLSRegression(n_components=5))],
            final_estimator=None  # Will use default RidgeCV
        )

        serialized = serialize_component(stacking)
        deserialized = deserialize_component(serialized)

        assert isinstance(deserialized, StackingRegressor)
        assert deserialized.final_estimator is None

    def test_voting_with_weights(self):
        """Test VotingRegressor with custom weights."""
        voting = VotingRegressor(
            estimators=[
                ('pls', PLSRegression(n_components=5)),
                ('rf', RandomForestRegressor(n_estimators=50))
            ],
            weights=[1, 2]  # Give more weight to RF
        )

        serialized = serialize_component(voting)

        # Check weights are serialized
        assert 'weights' in serialized['params']
        assert serialized['params']['weights'] == [1, 2]

        # Deserialize and verify
        deserialized = deserialize_component(serialized)
        assert deserialized.weights == [1, 2]

    def test_stacking_with_passthrough(self):
        """Test StackingRegressor with passthrough=True."""
        stacking = StackingRegressor(
            estimators=[('pls', PLSRegression(n_components=5))],
            final_estimator=Ridge(),
            passthrough=True
        )

        serialized = serialize_component(stacking)
        assert serialized['params']['passthrough'] is True

        deserialized = deserialize_component(serialized)
        assert deserialized.passthrough is True

    def test_nested_estimator_with_complex_params(self):
        """Test nested estimator with many non-default params."""
        stacking = StackingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                ))
            ],
            final_estimator=Ridge(alpha=0.1)
        )

        serialized = serialize_component(stacking)
        deserialized = deserialize_component(serialized)

        rf = deserialized.estimators[0][1]
        assert rf.n_estimators == 200
        assert rf.max_depth == 15
        assert rf.min_samples_split == 5
        assert rf.min_samples_leaf == 2
        assert rf.random_state == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
