"""Unit tests for pipeline topology analyzer.

Tests cover all documented pipeline topology patterns:
- Simple pipeline (no branching)
- Stacking (merge: "predictions")
- Nested stacking
- Separation branches (by_metadata, by_tag, by_source)
- Mixed merge
- Branches without merge (competing branches)
- Sequential models
- Preprocessing-only pipelines (no model)
- Pipelines with no splitter
- Feature merge
- Concat merge
"""

from nirs4all.pipeline.analysis.topology import (
    ModelNodeInfo,
    PipelineTopology,
    analyze_topology,
)

# ---------------------------------------------------------------------------
# Helpers: mock serialized step dicts that mimic the expanded pipeline format
# ---------------------------------------------------------------------------

def _splitter(cls_name: str = "sklearn.model_selection._split.KFold", **params) -> dict:
    """Create a serialized splitter step."""
    step = {"class": cls_name}
    if params:
        step["params"] = params
    return step

def _transform(cls_name: str = "sklearn.preprocessing._data.MinMaxScaler", **params) -> dict:
    """Create a serialized transformer step."""
    step = {"class": cls_name}
    if params:
        step["params"] = params
    return step

def _model(cls_name: str = "sklearn.cross_decomposition._pls.PLSRegression", **params) -> dict:
    """Create a model step with explicit ``model`` keyword."""
    inner = {"class": cls_name}
    if params:
        inner["params"] = params
    return {"model": inner}

def _snv() -> dict:
    return {"class": "nirs4all.operators.transforms.snv.SNV"}

def _msc() -> dict:
    return {"class": "nirs4all.operators.transforms.msc.MSC"}

def _pls(n: int = 10) -> dict:
    return _model("sklearn.cross_decomposition._pls.PLSRegression", n_components=n)

def _rf() -> dict:
    return _model("sklearn.ensemble._forest.RandomForestRegressor")

def _ridge() -> dict:
    return _model("sklearn.linear_model._ridge.Ridge")

# ---------------------------------------------------------------------------
# Test: Simple pipeline
# ---------------------------------------------------------------------------

class TestSimplePipeline:
    """Test topology analysis of simple linear pipelines."""

    def test_simple_pipeline(self):
        """SNV -> KFold -> PLS produces a simple topology."""
        steps = [_snv(), _splitter(n_splits=5), _pls()]
        topo = analyze_topology(steps)

        assert not topo.has_stacking
        assert not topo.has_feature_merge
        assert not topo.has_mixed_merge
        assert not topo.has_concat_merge
        assert not topo.has_separation_branch
        assert not topo.has_branches_without_merge
        assert topo.max_stacking_depth == 0
        assert len(topo.model_nodes) == 1
        assert topo.model_nodes[0].branch_path == []
        assert topo.model_nodes[0].step_index == 2
        assert topo.splitter_step_index == 1
        assert not topo.has_sequential_models
        assert not topo.has_multi_source

    def test_model_class_extracted(self):
        """Model class name is correctly extracted from serialized step."""
        steps = [_pls(5)]
        topo = analyze_topology(steps)

        assert len(topo.model_nodes) == 1
        assert "PLSRegression" in topo.model_nodes[0].model_class

    def test_no_model(self):
        """Preprocessing-only pipeline has no model nodes."""
        steps = [_snv(), _transform()]
        topo = analyze_topology(steps)

        assert len(topo.model_nodes) == 0
        assert topo.splitter_step_index is None
        assert not topo.has_stacking
        assert not topo.has_sequential_models

    def test_no_splitter(self):
        """Pipeline without a splitter has splitter_step_index=None."""
        steps = [_snv(), _pls()]
        topo = analyze_topology(steps)

        assert topo.splitter_step_index is None
        assert len(topo.model_nodes) == 1

# ---------------------------------------------------------------------------
# Test: Stacking
# ---------------------------------------------------------------------------

class TestStacking:
    """Test topology analysis of stacking pipelines."""

    def test_basic_stacking(self):
        """branch([PLS, RF]) -> merge("predictions") -> Ridge."""
        steps = [
            _splitter(n_splits=5),
            {"branch": [
                [_snv(), _pls()],
                [_msc(), _rf()],
            ]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_stacking
        assert not topo.has_feature_merge
        assert not topo.has_concat_merge
        assert not topo.has_separation_branch
        assert not topo.has_branches_without_merge
        assert topo.max_stacking_depth == 1
        # 2 branch models + 1 meta model
        assert len(topo.model_nodes) == 3
        assert topo.splitter_step_index == 0

    def test_stacking_model_branch_paths(self):
        """Branch models have correct branch_path and meta model is at top level."""
        steps = [
            {"branch": [
                [_pls()],
                [_rf()],
            ]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        # Branch 0 model
        branch_0_models = [m for m in topo.model_nodes if m.branch_path == [0]]
        assert len(branch_0_models) == 1
        assert "PLSRegression" in branch_0_models[0].model_class

        # Branch 1 model
        branch_1_models = [m for m in topo.model_nodes if m.branch_path == [1]]
        assert len(branch_1_models) == 1
        assert "RandomForest" in branch_1_models[0].model_class

        # Meta model at top level
        top_models = [m for m in topo.model_nodes if m.branch_path == []]
        assert len(top_models) == 1
        assert "Ridge" in top_models[0].model_class

    def test_stacking_depth(self):
        """max_stacking_depth is 1 for flat stacking."""
        steps = [
            {"branch": [[_pls()], [_rf()]]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)
        assert topo.max_stacking_depth == 1

# ---------------------------------------------------------------------------
# Test: Nested stacking
# ---------------------------------------------------------------------------

class TestNestedStacking:
    """Test topology analysis of nested stacking pipelines."""

    def test_nested_stacking(self):
        """Stacking within a stacking branch produces depth >= 2."""
        # Outer branch 0: inner stacking
        inner_stacking_branch = [
            {"branch": [[_pls()], [_rf()]]},
            {"merge": "predictions"},
            _ridge(),
        ]
        # Outer branch 1: simple model
        outer_branch_1 = [_snv(), _rf()]

        steps = [
            _splitter(n_splits=5),
            {"branch": [
                inner_stacking_branch,
                outer_branch_1,
            ]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_stacking
        assert topo.max_stacking_depth == 2
        # Inner: PLS + RF + Ridge; Outer branch 1: RF; Top-level meta: Ridge = 5 models
        assert len(topo.model_nodes) == 5

# ---------------------------------------------------------------------------
# Test: Separation branches
# ---------------------------------------------------------------------------

class TestSeparationBranch:
    """Test topology analysis with separation branches."""

    def test_by_metadata(self):
        """branch(by_metadata="site") is detected as separation."""
        steps = [
            {"branch": {"by_metadata": "site", "steps": [_snv(), _pls()]}},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)

        assert topo.has_separation_branch
        assert topo.has_concat_merge
        assert not topo.has_multi_source
        assert len(topo.model_nodes) == 1

    def test_by_tag(self):
        """branch(by_tag=...) is detected as separation."""
        steps = [
            {"branch": {
                "by_tag": "y_outlier_iqr",
                "values": {"clean": False, "outliers": True},
                "steps": [_pls()],
            }},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)

        assert topo.has_separation_branch
        assert not topo.has_multi_source

    def test_by_source(self):
        """branch(by_source=True, ...) is detected as separation + multi_source."""
        steps = [
            {"branch": {
                "by_source": True,
                "steps": {
                    "NIR": [_snv(), _pls()],
                    "markers": [_transform(), _rf()],
                },
            }},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)

        assert topo.has_separation_branch
        assert topo.has_multi_source
        assert topo.has_concat_merge
        # 2 models: one per source branch
        assert len(topo.model_nodes) == 2

# ---------------------------------------------------------------------------
# Test: Feature merge
# ---------------------------------------------------------------------------

class TestFeatureMerge:
    """Test feature merge detection."""

    def test_feature_merge(self):
        """merge: "features" sets has_feature_merge."""
        steps = [
            {"branch": [[_snv()], [_msc()]]},
            {"merge": "features"},
            _pls(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_feature_merge
        assert not topo.has_stacking
        assert not topo.has_concat_merge

# ---------------------------------------------------------------------------
# Test: Mixed merge
# ---------------------------------------------------------------------------

class TestMixedMerge:
    """Test mixed merge detection (features + predictions in one merge step)."""

    def test_mixed_merge(self):
        """merge: {"features": [...], "predictions": [...]} sets has_mixed_merge."""
        steps = [
            {"branch": [
                [_snv(), _pls()],
                [_msc()],
                [_snv(), _rf()],
            ]},
            {"merge": {"features": [1], "predictions": [0, 2]}},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_mixed_merge
        # Mixed merge also contributes to stacking via the predictions portion
        # (but has_stacking is specifically for merge:"predictions", so it stays False)
        assert not topo.has_stacking

# ---------------------------------------------------------------------------
# Test: Branches without merge (competing branches)
# ---------------------------------------------------------------------------

class TestBranchesWithoutMerge:
    """Test detection of duplication branches with no subsequent merge."""

    def test_no_merge_after_branch(self):
        """Duplication branch without merge -> has_branches_without_merge."""
        steps = [
            _splitter(n_splits=5),
            {"branch": [
                [_snv(), _pls()],
                [_msc(), _rf()],
            ]},
            # No merge step here
        ]
        topo = analyze_topology(steps)

        assert topo.has_branches_without_merge
        assert not topo.has_stacking

    def test_branch_followed_by_model_not_merge(self):
        """Branch followed by model (not merge) -> has_branches_without_merge."""
        steps = [
            {"branch": [
                [_snv()],
                [_msc()],
            ]},
            _pls(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_branches_without_merge

# ---------------------------------------------------------------------------
# Test: Sequential models
# ---------------------------------------------------------------------------

class TestSequentialModels:
    """Test detection of sequential model steps (no branch/merge between them)."""

    def test_sequential_models(self):
        """Two model steps without branch/merge -> has_sequential_models."""
        steps = [_snv(), _pls(), _ridge()]
        topo = analyze_topology(steps)

        assert topo.has_sequential_models
        assert len(topo.model_nodes) == 2

    def test_models_with_branch_between(self):
        """Models in different branches are not sequential."""
        steps = [
            {"branch": [[_pls()], [_rf()]]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        # Models in branches are at different branch_paths -> not sequential at top level
        # The meta Ridge is the only top-level model
        assert not topo.has_sequential_models

    def test_single_model_not_sequential(self):
        """A single model is not sequential."""
        steps = [_snv(), _pls()]
        topo = analyze_topology(steps)

        assert not topo.has_sequential_models

# ---------------------------------------------------------------------------
# Test: Multi-source
# ---------------------------------------------------------------------------

class TestMultiSource:
    """Test multi-source detection."""

    def test_by_source_sets_multi_source(self):
        """by_source in branch sets has_multi_source."""
        steps = [
            {"branch": {
                "by_source": True,
                "steps": [_snv(), _pls()],
            }},
        ]
        topo = analyze_topology(steps)

        assert topo.has_multi_source
        assert topo.has_separation_branch

# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and unusual configurations."""

    def test_empty_pipeline(self):
        """Empty step list produces default topology."""
        topo = analyze_topology([])

        assert not topo.has_stacking
        assert not topo.has_feature_merge
        assert not topo.has_mixed_merge
        assert not topo.has_concat_merge
        assert not topo.has_separation_branch
        assert not topo.has_branches_without_merge
        assert topo.max_stacking_depth == 0
        assert len(topo.model_nodes) == 0
        assert topo.splitter_step_index is None
        assert not topo.has_sequential_models
        assert not topo.has_multi_source

    def test_splitter_step_index_is_zero_based(self):
        """Splitter index is 0-based within the step list."""
        steps = [_splitter(n_splits=5), _pls()]
        topo = analyze_topology(steps)
        assert topo.splitter_step_index == 0

    def test_first_splitter_wins(self):
        """If multiple splitters, the first one's index is recorded."""
        steps = [
            _snv(),
            _splitter("sklearn.model_selection._split.KFold", n_splits=5),
            _transform(),
            _splitter("sklearn.model_selection._split.ShuffleSplit", n_splits=3),
            _pls(),
        ]
        topo = analyze_topology(steps)
        assert topo.splitter_step_index == 1

    def test_branch_depth_tracked(self):
        """Model nodes report correct branch_depth."""
        steps = [
            {"branch": [
                [_pls()],
                [_rf()],
            ]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        branch_models = [m for m in topo.model_nodes if m.branch_depth > 0]
        assert all(m.branch_depth == 1 for m in branch_models)

        top_models = [m for m in topo.model_nodes if m.branch_depth == 0]
        assert len(top_models) == 1

    def test_merge_type_propagated_to_branch_models(self):
        """Models inside branches know the merge type of the enclosing branch."""
        steps = [
            {"branch": [[_pls()], [_rf()]]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        branch_models = [m for m in topo.model_nodes if m.branch_depth > 0]
        assert all(m.merge_type == "predictions" for m in branch_models)

    def test_concat_merge_type(self):
        """Separation branch models get concat merge type."""
        steps = [
            {"branch": {
                "by_metadata": "site",
                "steps": [_pls()],
            }},
            {"merge": "concat"},
        ]
        topo = analyze_topology(steps)

        assert topo.has_concat_merge
        assert topo.has_separation_branch
        # The model inside gets merge_type="concat"
        assert len(topo.model_nodes) == 1
        assert topo.model_nodes[0].merge_type == "concat"

    def test_merge_sources(self):
        """merge_sources keyword is recognized."""
        steps = [
            _snv(),
            {"merge_sources": "concat"},
            _pls(),
        ]
        topo = analyze_topology(steps)

        # merge_sources: "concat" is recognized as concat merge
        assert topo.has_concat_merge

    def test_model_node_info_fields(self):
        """ModelNodeInfo has all expected fields populated."""
        steps = [_pls(5)]
        topo = analyze_topology(steps)

        node = topo.model_nodes[0]
        assert isinstance(node.model_class, str)
        assert isinstance(node.branch_path, list)
        assert isinstance(node.step_index, int)
        assert node.merge_type is None  # No branch enclosing this model
        assert node.branch_depth == 0

    def test_separation_branch_without_steps(self):
        """Separation branch without 'steps' key still detects separation."""
        steps = [
            {"branch": {"by_metadata": "site"}},
        ]
        topo = analyze_topology(steps)

        assert topo.has_separation_branch
        assert not topo.has_multi_source

    def test_multiple_merges(self):
        """Pipeline with both feature and prediction merges."""
        steps = [
            {"branch": [[_snv()], [_msc()]]},
            {"merge": "features"},
            _transform(),
            {"branch": [[_pls()], [_rf()]]},
            {"merge": "predictions"},
            _ridge(),
        ]
        topo = analyze_topology(steps)

        assert topo.has_feature_merge
        assert topo.has_stacking
        assert topo.max_stacking_depth == 1
