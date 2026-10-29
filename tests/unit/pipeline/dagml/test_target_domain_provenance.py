"""Domain attestations capture fitted owner state without computing scores."""

import hashlib
import json

import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer

from nirs4all.operators.transforms.targets import IntegerKBinsDiscretizer, RangeDiscretizer
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.target_domains import bin_domain_policy, fitted_bin_domain


@pytest.mark.parametrize("factory", [
    lambda: RangeDiscretizer(3),
    lambda: IntegerKBinsDiscretizer(n_bins=3, strategy="uniform"),
    lambda: IntegerKBinsDiscretizer(n_bins=3, strategy="quantile"),
    lambda: KBinsDiscretizer(n_bins=3, strategy="quantile", encode="ordinal"),
])
def test_domain_attests_each_actual_fitted_partition_and_stable_recipe(factory):
    targets = np.random.default_rng(19).normal(size=(24, 1))
    policy = bin_domain_policy(factory(), "model:terminal")
    records = []
    for train, _ in KFold(3, shuffle=True, random_state=7).split(targets):
        fitted = factory().fit(targets[train])
        ids = [f"sample:{index}" for index in train]
        record = fitted_bin_domain(fitted, ids, policy)
        assert record["fit_sample_ids"] == sorted(ids)
        assert record["recipe_fingerprint"] == policy["recipe_fingerprint"]
        assert record["class_labels"] == [0., 1., 2.]
        assert record["state_fingerprint"] == hashlib.sha256(record["state_json"].encode("utf-8")).hexdigest()
        state = json.loads(record["state_json"])["fitted_state"]
        actual_edges = fitted._bins_array if isinstance(fitted, RangeDiscretizer) else (
            fitted.discretizer.bin_edges_[0] if isinstance(fitted, IntegerKBinsDiscretizer) else fitted.bin_edges_[0])
        np.testing.assert_array_equal(state.get("cut_points", state.get("bin_edges")), actual_edges)
        records.append(record)
    assert len({record["state_fingerprint"] for record in records}) == 3


def test_collapsed_quantile_bins_fail_closed():
    transformer = IntegerKBinsDiscretizer(n_bins=3, strategy="quantile")
    policy = bin_domain_policy(transformer, "model:terminal")
    with pytest.warns(UserWarning):
        transformer.fit(np.array([0., 0., 0., 1.]).reshape(-1, 1))
    with pytest.raises(DagMlUnsupported, match="collapsed"):
        fitted_bin_domain(transformer, ["a", "b", "c", "d"], policy)


def test_domain_refuses_recipe_drift_and_duplicate_fit_identity():
    transformer = RangeDiscretizer(2).fit([-1., 1.])
    policy = bin_domain_policy(transformer, "model:terminal")
    with pytest.raises(ValueError, match="unique"):
        fitted_bin_domain(transformer, ["same", "same"], policy)
    with pytest.raises(ValueError, match="disagrees"):
        fitted_bin_domain(transformer, ["a", "b"], {**policy, "recipe_fingerprint": "0" * 64})
