"""Tests for multi-source loader guardrails."""

from __future__ import annotations

import numpy as np
import pytest

from nirs4all.data.loaders.loader import _audit_multisource_lengths
from nirs4all.data.relations import RelationValidationError


def test_audit_multisource_lengths_allows_equal_length_sources():
    _audit_multisource_lengths({}, [np.zeros((4, 3)), np.ones((4, 2))])


def test_audit_multisource_lengths_rejects_heterogeneous_sources_with_link_by_hint():
    config = {"_sources": [{"name": "MIR", "link_by": "sample_id"}, {"name": "RAMAN", "link_by": "sample_id"}]}

    with pytest.raises(RelationValidationError) as exc:
        _audit_multisource_lengths(config, [np.zeros((4, 3)), np.ones((6, 2))])

    assert exc.value.code == "REL-E009"
    assert "sample_id" in str(exc.value)
