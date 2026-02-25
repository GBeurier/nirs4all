"""Unit tests for MergeController._resolve_auto_merge.

Verifies that auto-detect merge syntaxes ("auto", True, {"branch": ...})
are resolved to the correct explicit merge configuration based on
branch context flags.
"""

import pytest

from nirs4all.controllers.data.merge import MergeController
from nirs4all.pipeline.config.context import ExecutionContext


def _make_context(**custom_flags) -> ExecutionContext:
    """Create an ExecutionContext with the given custom flags."""
    return ExecutionContext(custom=custom_flags)


def _controller() -> MergeController:
    return MergeController()


# ---------------------------------------------------------------------------
# Duplication branch
# ---------------------------------------------------------------------------

class TestAutoMergeDuplicationBranch:
    """Auto-detect resolves to 'features' for duplication branches."""

    @pytest.fixture
    def ctx(self):
        return _make_context(
            in_branch_mode=True,
            branch_type="duplication",
        )

    def test_auto_string(self, ctx):
        assert _controller()._resolve_auto_merge("auto", ctx) == "features"

    def test_true(self, ctx):
        assert _controller()._resolve_auto_merge(True, ctx) == "features"

    def test_dict_branch_key(self, ctx):
        assert _controller()._resolve_auto_merge({"branch": True}, ctx) == "features"

    def test_dict_branch_with_value(self, ctx):
        assert _controller()._resolve_auto_merge({"branch": "auto"}, ctx) == "features"


# ---------------------------------------------------------------------------
# Separation branch: by_tag
# ---------------------------------------------------------------------------

class TestAutoMergeSeparationByTag:
    """Auto-detect resolves to 'concat' for by_tag separation branches."""

    @pytest.fixture
    def ctx(self):
        return _make_context(
            in_branch_mode=True,
            branch_type="separation",
            separation_type="by_tag",
        )

    def test_auto_string(self, ctx):
        assert _controller()._resolve_auto_merge("auto", ctx) == "concat"

    def test_true(self, ctx):
        assert _controller()._resolve_auto_merge(True, ctx) == "concat"

    def test_dict_branch_key(self, ctx):
        assert _controller()._resolve_auto_merge({"branch": True}, ctx) == "concat"


# ---------------------------------------------------------------------------
# Separation branch: by_metadata
# ---------------------------------------------------------------------------

class TestAutoMergeSeparationByMetadata:
    """Auto-detect resolves to 'concat' for by_metadata separation branches."""

    @pytest.fixture
    def ctx(self):
        return _make_context(
            in_branch_mode=True,
            branch_type="separation",
            separation_type="by_metadata",
        )

    def test_auto_string(self, ctx):
        assert _controller()._resolve_auto_merge("auto", ctx) == "concat"

    def test_true(self, ctx):
        assert _controller()._resolve_auto_merge(True, ctx) == "concat"


# ---------------------------------------------------------------------------
# Separation branch: by_filter
# ---------------------------------------------------------------------------

class TestAutoMergeSeparationByFilter:
    """Auto-detect resolves to 'concat' for by_filter separation branches."""

    @pytest.fixture
    def ctx(self):
        return _make_context(
            in_branch_mode=True,
            branch_type="separation",
            separation_type="by_filter",
        )

    def test_auto_string(self, ctx):
        assert _controller()._resolve_auto_merge("auto", ctx) == "concat"

    def test_true(self, ctx):
        assert _controller()._resolve_auto_merge(True, ctx) == "concat"


# ---------------------------------------------------------------------------
# Separation branch: by_source
# ---------------------------------------------------------------------------

class TestAutoMergeSeparationBySource:
    """Auto-detect resolves to {"sources": "concat"} for by_source branches."""

    @pytest.fixture
    def ctx(self):
        return _make_context(
            in_branch_mode=True,
            in_source_branch_mode=True,
            branch_type="separation",
            separation_type="by_source",
        )

    def test_auto_string(self, ctx):
        assert _controller()._resolve_auto_merge("auto", ctx) == {"sources": "concat"}

    def test_true(self, ctx):
        assert _controller()._resolve_auto_merge(True, ctx) == {"sources": "concat"}

    def test_dict_branch_key(self, ctx):
        assert _controller()._resolve_auto_merge({"branch": True}, ctx) == {"sources": "concat"}

    def test_in_source_branch_mode_without_separation_type(self):
        """in_source_branch_mode alone is enough for source merge detection."""
        ctx = _make_context(in_branch_mode=True, in_source_branch_mode=True)
        assert _controller()._resolve_auto_merge("auto", ctx) == {"sources": "concat"}


# ---------------------------------------------------------------------------
# Passthrough: explicit configs are not modified
# ---------------------------------------------------------------------------

class TestAutoMergePassthrough:
    """Explicit merge configs pass through unchanged."""

    @pytest.fixture
    def ctx(self):
        return _make_context(in_branch_mode=True, branch_type="duplication")

    def test_features_string(self, ctx):
        assert _controller()._resolve_auto_merge("features", ctx) == "features"

    def test_predictions_string(self, ctx):
        assert _controller()._resolve_auto_merge("predictions", ctx) == "predictions"

    def test_concat_string(self, ctx):
        assert _controller()._resolve_auto_merge("concat", ctx) == "concat"

    def test_all_string(self, ctx):
        assert _controller()._resolve_auto_merge("all", ctx) == "all"

    def test_dict_with_sources_key(self, ctx):
        cfg = {"sources": "concat"}
        assert _controller()._resolve_auto_merge(cfg, ctx) is cfg

    def test_dict_with_features_key(self, ctx):
        cfg = {"features": "all"}
        assert _controller()._resolve_auto_merge(cfg, ctx) is cfg

    def test_dict_with_predictions_key(self, ctx):
        cfg = {"predictions": True}
        assert _controller()._resolve_auto_merge(cfg, ctx) is cfg

    def test_dict_branch_plus_features_not_auto(self, ctx):
        """Dict with 'branch' AND 'features' is NOT auto-detect."""
        cfg = {"branch": 0, "features": "all"}
        assert _controller()._resolve_auto_merge(cfg, ctx) is cfg

    def test_dict_branch_plus_sources_not_auto(self, ctx):
        """Dict with 'branch' AND 'sources' is NOT auto-detect."""
        cfg = {"branch": True, "sources": "concat"}
        assert _controller()._resolve_auto_merge(cfg, ctx) is cfg


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestAutoMergeErrors:
    """Auto-detect without branch context raises ValueError."""

    def test_auto_string_no_branch_context(self):
        ctx = _make_context()
        with pytest.raises(ValueError, match="requires an active branch context"):
            _controller()._resolve_auto_merge("auto", ctx)

    def test_true_no_branch_context(self):
        ctx = _make_context()
        with pytest.raises(ValueError, match="requires an active branch context"):
            _controller()._resolve_auto_merge(True, ctx)

    def test_dict_branch_no_branch_context(self):
        ctx = _make_context()
        with pytest.raises(ValueError, match="requires an active branch context"):
            _controller()._resolve_auto_merge({"branch": True}, ctx)
