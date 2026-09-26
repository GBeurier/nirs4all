"""Bounded v6 N4MP/N4MM cross-language trained envelope tests."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.portable_n4m_trained import PortableN4MTrainedPipeline

pytest.importorskip("pls4all")
pytest.importorskip("n4m")

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_R_ENVELOPE = _FIXTURES / "portable_n4mp_v6_r_envelope.json"
_R_ORACLE = _FIXTURES / "portable_n4mp_v6_r_oracle.json"


def _r_data() -> tuple[dict, dict]:
    return json.loads(_R_ENVELOPE.read_text()), json.loads(_R_ORACLE.read_text())


def _with_manifest(document: dict, manifest: dict) -> dict:
    changed = json.loads(json.dumps(document))
    changed["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
    changed["manifest_sha256"] = hashlib.sha256(changed["manifest_json"].encode()).hexdigest()
    return changed


@pytest.mark.methods
def test_r_v6_fixture_predict_retrain_and_python_export() -> None:
    pd = pytest.importorskip("pandas")
    document, oracle = _r_data()
    train = pd.DataFrame(oracle["train"], columns=oracle["feature_names"])
    held = pd.DataFrame(oracle["heldout"], columns=oracle["feature_names"])
    expected = np.asarray(oracle["predictions"])
    with PortableN4MTrainedPipeline(document) as imported:
        np.testing.assert_allclose(imported.predict(held), expected, rtol=0, atol=1e-10)
        with imported.retrain(train, oracle["y"]) as retrained:
            assert json.loads(retrained.to_json())["schema"] == "nirs4all.n4m.trained_pipeline.v6"
            np.testing.assert_allclose(retrained.predict(held), expected, rtol=0, atol=1e-8)
            with PortableN4MTrainedPipeline.from_json(retrained.to_json()) as restored:
                np.testing.assert_allclose(restored.predict(held), expected, rtol=0, atol=1e-10)
        with pytest.raises(ValueError, match="feature names or order"):
            imported.predict(held.iloc[:, ::-1])
        with pytest.raises(ValueError, match="feature names or order"):
            imported.predict(held.to_numpy())
        with pytest.raises(ValueError, match="feature names or order"):
            imported.retrain(train.iloc[:, ::-1], oracle["y"])
        with pytest.raises(ValueError, match="feature names or order"):
            imported.predict(held.iloc[:, :-1].to_numpy())


@pytest.mark.methods
def test_v6_refuses_manifest_payload_and_native_plan_tampering() -> None:
    document, _ = _r_data()
    changed = json.loads(json.dumps(document))
    changed["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        PortableN4MTrainedPipeline(changed)
    for field in ("preprocessing", "model"):
        changed = json.loads(json.dumps(document))
        changed[field]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="payload hash"):
            PortableN4MTrainedPipeline(changed)
    changed = json.loads(json.dumps(document))
    changed["unexpected"] = True
    with pytest.raises(ValueError, match="envelope"):
        PortableN4MTrainedPipeline(changed)

    manifest = json.loads(document["manifest_json"])
    manifest["preprocessing_owner"] = "external"
    with pytest.raises(ValueError, match="schema differs"):
        PortableN4MTrainedPipeline(_with_manifest(document, manifest))
    manifest = json.loads(document["manifest_json"])
    manifest["input_n_features"] += 1
    manifest["feature_names"] = None
    with pytest.raises(ValueError, match="descriptor does not match"):
        PortableN4MTrainedPipeline(_with_manifest(document, manifest))
    manifest = json.loads(document["manifest_json"])
    manifest["step_states"] = [None]
    with pytest.raises(ValueError, match="no external step states"):
        PortableN4MTrainedPipeline(_with_manifest(document, manifest))
    manifest = json.loads(document["manifest_json"])
    manifest["recipe"]["pipeline"][0:2] = manifest["recipe"]["pipeline"][1::-1]
    with pytest.raises(ValueError, match="native N4MP plan"):
        PortableN4MTrainedPipeline(_with_manifest(document, manifest))
    changed = json.loads(json.dumps(document))
    raw = bytearray(base64.b64decode(changed["preprocessing"]["payload"]))
    raw[0] = 0
    changed["preprocessing"]["payload"] = base64.b64encode(raw).decode()
    changed["preprocessing"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises((ValueError, RuntimeError)):
        PortableN4MTrainedPipeline(changed)


@pytest.mark.methods
@pytest.mark.parametrize("step", [
    {"class": "n4m.EMSC", "params": {"degree": 2}},
    {"class": "n4m.LSNV"},
    {"class": "n4m.SNV", "params": {"ddof": 1, "with_mean": True, "with_std": True}},
    {"class": "n4m.SavitzkyGolay", "params": {
        "window_length": 7, "polyorder": 2, "deriv": 0,
        "delta": 2, "mode": "interp", "cval": 0,
    }},
    {"branch": {"a": [{"class": "n4m.SNV"}], "b": [{"class": "n4m.MSC"}]}},
])
def test_v6_refuses_unqualified_steps(step: dict) -> None:
    _, oracle = _r_data()
    recipe = {"pipeline": [step, {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}}]}
    with pytest.raises(ValueError):
        PortableN4MTrainedPipeline.fit_recipe(
            recipe, np.asarray(oracle["train"]), np.asarray(oracle["y"]),
            preprocessing="native_n4mp",
        )


@pytest.mark.methods
def test_v6_group_sparse_preserves_interleaved_group_ids() -> None:
    samples = np.arange(1, 25, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    groups = [0, 1, 2] * 4
    recipe = {"pipeline": [
        {"class": "n4m.SNV"},
        {"model": {"class": "n4m.GroupSparsePLS", "params": {
            "n_components": 2, "group_assignment": groups, "group_lambda": 0.05,
        }}},
    ]}
    with PortableN4MTrainedPipeline.fit_recipe(
        recipe, X[:21], y[:21], preprocessing="native_n4mp",
    ) as fitted:
        document = json.loads(fitted.to_json())
        manifest = json.loads(document["manifest_json"])
        assert manifest["recipe"]["pipeline"][-1]["model"]["params"]["group_assignment"] == groups
        with PortableN4MTrainedPipeline(document) as imported:
            np.testing.assert_allclose(imported.predict(X[21:]), fitted.predict(X[21:]), rtol=0, atol=1e-12)
    bad = json.loads(json.dumps(document))
    manifest = json.loads(bad["manifest_json"])
    manifest["recipe"]["pipeline"][-1]["model"]["params"]["group_assignment"] = groups[:-1]
    with pytest.raises(ValueError, match="group assignment"):
        PortableN4MTrainedPipeline(_with_manifest(bad, manifest))
