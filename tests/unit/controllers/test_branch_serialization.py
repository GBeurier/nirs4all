"""
Unit tests for Branch Serialization.

Tests pipeline JSON serialization with branches:
- Branch structure preserved in serialization
- Named branches serialization
- Generator syntax serialization
- Nested branches serialization
"""

import pytest
import json
from copy import deepcopy

from nirs4all.pipeline.config.component_serialization import (
    serialize_component,
    deserialize_component
)


class TestBranchListSerialization:
    """Test serialization of branch list syntax."""

    def test_serialize_simple_branch_list(self):
        """Test serialization of [[steps], [steps]] format."""
        branch_step = {
            "branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]
        }

        serialized = serialize_component(branch_step)

        assert "branch" in serialized
        assert isinstance(serialized["branch"], list)
        assert len(serialized["branch"]) == 2

    def test_serialize_branch_with_multi_step(self):
        """Test serialization of branches with multiple steps."""
        branch_step = {
            "branch": [
                [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}},
                ],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]
        }

        serialized = serialize_component(branch_step)

        assert len(serialized["branch"]) == 2
        assert len(serialized["branch"][0]) == 2  # First branch has 2 steps
        assert len(serialized["branch"][1]) == 1  # Second branch has 1 step

    def test_roundtrip_branch_list(self):
        """Test roundtrip: serialize → deserialize → same structure."""
        original = {
            "branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]
        }

        serialized = serialize_component(original)
        # Convert to JSON string and back to simulate file save/load
        json_str = json.dumps(serialized)
        reloaded = json.loads(json_str)

        assert reloaded["branch"] == serialized["branch"]


class TestBranchDictSerialization:
    """Test serialization of named branch syntax."""

    def test_serialize_named_branches(self):
        """Test serialization of {"name": [steps]} format."""
        branch_step = {
            "branch": {
                "snv_pca": [{"class": "nirs4all.preprocessing.SNV"}],
                "msc": [{"class": "nirs4all.preprocessing.MSC"}],
            }
        }

        serialized = serialize_component(branch_step)

        assert "branch" in serialized
        assert isinstance(serialized["branch"], dict)
        assert "snv_pca" in serialized["branch"]
        assert "msc" in serialized["branch"]

    def test_roundtrip_named_branches(self):
        """Test roundtrip with named branches."""
        original = {
            "branch": {
                "alpha": [{"class": "sklearn.preprocessing.StandardScaler"}],
                "beta": [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            }
        }

        serialized = serialize_component(original)
        json_str = json.dumps(serialized)
        reloaded = json.loads(json_str)

        assert "alpha" in reloaded["branch"]
        assert "beta" in reloaded["branch"]


class TestBranchGeneratorSerialization:
    """Test serialization of generator syntax inside branches."""

    def test_serialize_or_generator_in_branch(self):
        """Test serialization of {"_or_": [...]} inside branch."""
        branch_step = {
            "branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                ]
            }
        }

        serialized = serialize_component(branch_step)

        assert "branch" in serialized
        assert "_or_" in serialized["branch"]
        assert len(serialized["branch"]["_or_"]) == 2

    def test_serialize_range_generator_in_branch(self):
        """Test serialization of {"_range_": [start, end, step]} inside branch."""
        branch_step = {
            "branch": {
                "_range_": [5, 15, 5]
            }
        }

        serialized = serialize_component(branch_step)

        assert "branch" in serialized
        assert "_range_" in serialized["branch"]
        assert serialized["branch"]["_range_"] == [5, 15, 5]


class TestBranchOutlierExcluderSerialization:
    """Test serialization of outlier excluder branch syntax."""

    def test_serialize_outlier_excluder(self):
        """Test serialization of {"by": "outlier_excluder", ...} syntax."""
        branch_step = {
            "branch": {
                "by": "outlier_excluder",
                "strategies": [
                    None,
                    {"method": "isolation_forest", "contamination": 0.1},
                    {"method": "mahalanobis", "threshold": 3.0},
                ],
            }
        }

        serialized = serialize_component(branch_step)

        assert serialized["branch"]["by"] == "outlier_excluder"
        assert "strategies" in serialized["branch"]
        assert len(serialized["branch"]["strategies"]) == 3

    def test_serialize_sample_partitioner(self):
        """Test serialization of sample_partitioner syntax."""
        branch_step = {
            "branch": {
                "by": "sample_partitioner",
                "filter": {"method": "y_outlier", "threshold": 1.5},
            }
        }

        serialized = serialize_component(branch_step)

        assert serialized["branch"]["by"] == "sample_partitioner"
        assert "filter" in serialized["branch"]
        assert serialized["branch"]["filter"]["method"] == "y_outlier"


class TestNestedBranchSerialization:
    """Test serialization with multiple branch steps (nested)."""

    def test_serialize_pipeline_with_nested_branches(self):
        """Test serialization of pipeline with multiple branch steps."""
        pipeline = [
            {"class": "sklearn.model_selection.ShuffleSplit", "params": {"n_splits": 2}},
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
                [{"class": "sklearn.preprocessing.MinMaxScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 10}}],
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
            {"model": {"class": "sklearn.linear_model.Ridge"}},
        ]

        # Serialize each step
        serialized = [serialize_component(step) for step in pipeline]

        # Should have 4 steps
        assert len(serialized) == 4

        # Steps 1 and 2 should be branch steps
        assert "branch" in serialized[1]
        assert "branch" in serialized[2]

    def test_roundtrip_nested_branches(self):
        """Test roundtrip with nested branches."""
        pipeline = [
            {"branch": [
                [{"class": "sklearn.preprocessing.StandardScaler"}],
            ]},
            {"branch": [
                [{"class": "sklearn.decomposition.PCA", "params": {"n_components": 5}}],
            ]},
        ]

        serialized = [serialize_component(step) for step in pipeline]
        json_str = json.dumps(serialized)
        reloaded = json.loads(json_str)

        assert len(reloaded) == 2
        assert "branch" in reloaded[0]
        assert "branch" in reloaded[1]


class TestBranchMetadataSerialization:
    """Test serialization of branch metadata."""

    def test_serialize_branch_with_count(self):
        """Test serialization of branch with count limit."""
        branch_step = {
            "branch": {
                "_or_": [
                    {"class": "sklearn.preprocessing.StandardScaler"},
                    {"class": "sklearn.preprocessing.MinMaxScaler"},
                    {"class": "sklearn.preprocessing.RobustScaler"},
                ],
                "count": 2
            }
        }

        serialized = serialize_component(branch_step)

        assert "count" in serialized["branch"]
        assert serialized["branch"]["count"] == 2

    def test_serialize_branch_with_pick(self):
        """Test serialization of branch with pick modifier."""
        branch_step = {
            "branch": {
                "_or_": ["A", "B", "C"],
                "pick": 2
            }
        }

        serialized = serialize_component(branch_step)

        assert "pick" in serialized["branch"]
        assert serialized["branch"]["pick"] == 2
