"""Parity tests for the nirs4all → dag-ml(-data) data layer (migration phase 2b-i).

Identity + resolver are pure nirs4all (no dag-ml / dag-ml-data import), so these run
unconditionally — they verify the stable-id round-trip and that the resolver serves real
``SpectroDataset`` rows **in the caller's requested order** (the load-bearing
identity-keyed invariant), entirely in-process with no CLI / execution / FFI.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from nirs4all.data.config import DatasetConfigs
from nirs4all.pipeline.dagml.identity import mint_identity
from nirs4all.pipeline.dagml.resolver import MaterializationResolver

from ._datasets import dataset_path

pytestmark = [pytest.mark.parity]


@pytest.fixture(scope="module")
def regression_dataset():
    return DatasetConfigs(dataset_path("regression")).get_dataset_at(0)


def test_identity_roundtrip_is_stable_and_legal(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    sample_ints = regression_dataset.index_column("sample", {})
    # Bidirectional and total over every sample.
    assert len(identity.identities) == len(sample_ints)
    for sample_int in sample_ints:
        assert identity.to_int(identity.to_wire(sample_int)) == sample_int
    # Every wire id is a legal dag-ml-data id (alnum + '_.-', no ':').
    legal = re.compile(r"^[A-Za-z0-9_.\-]{1,128}\Z")
    for observation_id in identity.observation_ids():
        assert legal.match(observation_id), observation_id
        assert ":" not in observation_id
    # The fingerprint is content-derived and stable across mints (not position-derived).
    assert identity.fingerprint == mint_identity(regression_dataset).fingerprint


def test_resolver_features_restore_request_order(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_ints = regression_dataset.index_column("sample", {})

    # A deliberately non-ascending request — SpectroDataset.x returns storage order, so
    # this only passes if the resolver re-orders back to the request.
    picked = [sample_ints[5], sample_ints[0], sample_ints[9], sample_ints[2]]
    wire = [identity.to_wire(s) for s in picked]
    out = resolver.resolve_features(wire)
    values = np.asarray(out["values"])

    ground_truth = np.stack([np.asarray(regression_dataset.x({"sample": [s]}, layout="2d"))[0] for s in picked])
    assert values.shape == ground_truth.shape
    assert np.array_equal(values, ground_truth)
    assert out["observation_ids"] == wire

    # Reversing the request reverses the rows — proves order tracks identity, not position.
    rev = resolver.resolve_features(list(reversed(wire)))
    assert np.array_equal(np.asarray(rev["values"]), values[::-1])


def test_resolver_targets_restore_request_order(regression_dataset) -> None:
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_ints = regression_dataset.index_column("sample", {})

    picked = [sample_ints[7], sample_ints[1], sample_ints[4]]
    wire = [identity.to_wire(s) for s in picked]
    out = resolver.resolve_targets(wire)
    values = np.asarray(out["values"], dtype=float)

    ground_truth = np.asarray([float(np.asarray(regression_dataset.y({"sample": [s]})).ravel()[0]) for s in picked])
    assert np.allclose(values, ground_truth)
    assert out["sample_ids"] == wire


def test_resolver_serves_real_spectra_not_a_hash(regression_dataset) -> None:
    """Guards the closed gap: the shipped conformance adapters synthesize X from hashed
    sample ids; this resolver must return the actual spectra."""
    identity = mint_identity(regression_dataset)
    resolver = MaterializationResolver(regression_dataset, identity)
    sample_int = regression_dataset.index_column("sample", {})[0]
    wire = identity.to_wire(sample_int)

    row = np.asarray(resolver.resolve_features([wire])["values"])[0]
    real = np.asarray(regression_dataset.x({"sample": [sample_int]}, layout="2d"))[0]
    assert np.array_equal(row, real)
    # A hashed-id synthesis would collapse 2151 wavelengths to a constant; real spectra vary.
    assert row.shape[0] > 1 and float(np.ptp(row)) > 0.0
