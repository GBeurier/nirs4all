"""Real native R payloads cover legacy and component-resolution wire families."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.portable_n4m_trained import PortableN4MTrainedPipeline

pytest.importorskip("pls4all")
pytest.importorskip("n4m")

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _document(fixture: dict, profile: str) -> dict:
    embedded = profile == "embedded"
    steps = [
        {"class": "n4m.SNV"},
        {"class": "n4m.SavitzkyGolay", "params": {
            "window_length": 5, "polyorder": 2, "deriv": 0,
            "delta": 1, "mode": "interp", "cval": 0,
        }},
    ] if embedded else []
    manifest = {
        "recipe": {"pipeline": [*steps, {"model": {
            "class": "n4m.PLS", "params": {"n_components": 2},
        }}]},
        "input_n_features": len(fixture["X"][0]), "feature_names": None,
        "preprocessing_owner": "embedded_methods" if embedded else "external",
        "step_states": [None, None] if embedded else [],
    }
    text = json.dumps(manifest, separators=(",", ":"))
    payload = fixture[profile]["payload"]
    return {
        "schema": "nirs4all.n4m.trained_pipeline.v1",
        "manifest_json": text, "manifest_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "model": {"kind": "n4m_model", "encoding": "base64-n4mm",
                  "payload": payload, "sha256": hashlib.sha256(base64.b64decode(payload)).hexdigest()},
    }


def _manifest(document: dict, manifest: dict) -> dict:
    changed = copy.deepcopy(document)
    text = json.dumps(manifest, separators=(",", ":"))
    changed.update(manifest_json=text, manifest_sha256=hashlib.sha256(text.encode()).hexdigest())
    return changed


@pytest.mark.methods
@pytest.mark.parametrize("name", ["n4mm_legacy_v1_v2.json", "n4mm_current_v3.json"])
@pytest.mark.parametrize("profile", ["plain", "embedded"])
def test_real_r_wire_versions_predict_and_roundtrip(name: str, profile: str) -> None:
    from pls4all import inspect_n4mm

    fixture = json.loads((_FIXTURES / name).read_text())
    document = _document(fixture, profile)
    descriptor = inspect_n4mm(base64.b64decode(document["model"]["payload"]))
    assert descriptor.format_version == fixture[profile]["format_version"]
    assert (descriptor.pipeline is not None) == (profile == "embedded")
    with PortableN4MTrainedPipeline(document) as imported:
        np.testing.assert_allclose(imported.predict(fixture["X"]), fixture[profile]["predictions"], rtol=0, atol=1e-12)
        with PortableN4MTrainedPipeline.from_json(imported.to_json()) as reloaded:
            np.testing.assert_allclose(reloaded.predict(fixture["X"]), fixture[profile]["predictions"], rtol=0, atol=1e-12)


@pytest.mark.methods
@pytest.mark.parametrize("profile", ["plain", "embedded"])
def test_v3_still_refuses_rehashed_recipe_dimensions_and_owner_mismatches(profile: str) -> None:
    fixture = json.loads((_FIXTURES / "n4mm_current_v3.json").read_text())
    document = _document(fixture, profile)
    manifest = json.loads(document["manifest_json"])
    wrong = copy.deepcopy(manifest)
    wrong["recipe"]["pipeline"][-1]["model"]["params"]["n_components"] = 1
    with pytest.raises(ValueError, match="descriptor does not match"):
        PortableN4MTrainedPipeline(_manifest(document, wrong))
    wrong = copy.deepcopy(manifest)
    wrong["input_n_features"] += 1
    with pytest.raises(ValueError, match="descriptor does not match"):
        PortableN4MTrainedPipeline(_manifest(document, wrong))
    opposite = _document(fixture, "embedded" if profile == "plain" else "plain")
    opposite["model"] = document["model"]
    with pytest.raises(ValueError, match="descriptor does not match|preprocessing does not match|external preprocessing"):
        PortableN4MTrainedPipeline(opposite)
    if profile == "embedded":
        wrong = copy.deepcopy(manifest)
        wrong["recipe"]["pipeline"][1]["params"]["window_length"] = 7
        with pytest.raises(ValueError, match="preprocessing does not match"):
            PortableN4MTrainedPipeline(_manifest(document, wrong))


@pytest.mark.methods
@pytest.mark.parametrize("profile", ["plain", "embedded"])
@pytest.mark.parametrize("mutation", ["checksum", "future_format", "resolution"])
def test_v3_native_validator_refuses_corruption_even_with_rehashed_envelope(profile: str, mutation: str) -> None:
    fixture = json.loads((_FIXTURES / "n4mm_current_v3.json").read_text())
    document = _document(fixture, profile)
    payload = bytearray(base64.b64decode(document["model"]["payload"]))
    if mutation == "future_format":
        payload[4:8] = (4).to_bytes(4, "little")
    elif mutation == "resolution":
        # A nonfinite component resolution is invalid even with a fresh native checksum.
        offset = 96  # Fixed shared header, followed by eleven length-prefixed vectors.
        for _ in range(11):
            count = int.from_bytes(payload[offset:offset + 8], "little")
            offset += 8 + count * 8
        assert int.from_bytes(payload[offset:offset + 8], "little") == 1
        payload[offset + 8:offset + 16] = bytes.fromhex("000000000000f87f")
    if mutation != "checksum":
        checksum = 14695981039346656037
        for value in payload[:-8]:
            checksum = ((checksum ^ value) * 1099511628211) & ((1 << 64) - 1)
        payload[-8:] = checksum.to_bytes(8, "little")
    else:
        payload[-1] ^= 1
    document["model"].update(payload=base64.b64encode(payload).decode(), sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises((ValueError, RuntimeError)):
        PortableN4MTrainedPipeline(document)
