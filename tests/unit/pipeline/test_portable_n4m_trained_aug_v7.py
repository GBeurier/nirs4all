"""Train-only native X augmentation in the bounded v7 N4MP envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from nirs4all.pipeline.config.pipeline_config import PipelineConfigs
from nirs4all.pipeline.portable_n4m_trained import PortableN4MTrainedPipeline

pytest.importorskip("pls4all")
pytest.importorskip("n4m")

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = np.arange(1, 25, dtype=np.float64)[:, None]
    bands = np.arange(1, 13, dtype=np.float64)[None, :]
    X = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * X[:, 1] - 0.4 * X[:, 5]
    return X[:21], y[:21], X[21:] + 0.031


def _augmentation(kind: str = "gaussian_noise", values: list[float] | None = None,
                  seed: int = 42) -> dict:
    return {"train_augmentation": {"class": "n4m.NativeXAugmentation", "params": {
        "kind": kind, "values": [0.03] if values is None else values, "seed": seed,
    }}}


def _recipe(augmentation: dict | None = None) -> dict:
    nodes = [] if augmentation is None else [augmentation]
    return {"pipeline": [*nodes, {"class": "n4m.SNV"},
                         {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}}]}


@pytest.mark.methods
def test_v7_train_only_augmentation_predict_retrain_and_native_oracle() -> None:
    from n4m.augmentation import run_native

    train, y, held = _data()
    recipe = _recipe(_augmentation())
    with PortableN4MTrainedPipeline.fit_recipe(
        recipe, train, y, preprocessing="native_n4mp",
    ) as fitted:
        document = json.loads(fitted.to_json())
        assert document["schema"] == "nirs4all.n4m.trained_pipeline.v7"
        assert json.loads(document["manifest_json"])["recipe"] == recipe
        predicted = fitted.predict(held)
        with PortableN4MTrainedPipeline.from_json(fitted.to_json()) as loaded:
            np.testing.assert_allclose(loaded.predict(held), predicted, rtol=0, atol=1e-12)
            with loaded.retrain(train, y) as refitted:
                assert json.loads(refitted.to_json())["schema"] == document["schema"]
                np.testing.assert_allclose(refitted.predict(held), predicted, rtol=0, atol=1e-12)
        augmented = run_native("gaussian_noise", train, [0.03], 42)
        with PortableN4MTrainedPipeline.fit_recipe(
            _recipe(), augmented, y, preprocessing="native_n4mp",
        ) as native_oracle:
            np.testing.assert_allclose(predicted, native_oracle.predict(held), rtol=0, atol=1e-10)


@pytest.mark.methods
def test_r_v7_trained_envelope_and_augmentation_oracle() -> None:
    from n4m.augmentation import run_native

    pd = pytest.importorskip("pandas")
    document = json.loads((_FIXTURES / "portable_n4mp_v7_r_envelope.json").read_text())
    oracle = json.loads((_FIXTURES / "portable_n4mp_v7_r_oracle.json").read_text())
    recipe = json.loads(document["manifest_json"])["recipe"]
    parsed = PipelineConfigs(recipe).steps[0]
    assert [node["train_augmentation"] for node in parsed[:2]] == [
        node["train_augmentation"] for node in recipe["pipeline"][:2]
    ]
    samples = np.arange(1, 25, dtype=np.float64)[:, None]
    bands = np.arange(1, 18, dtype=np.float64)[None, :]
    values = np.sin(samples * bands / 9) + np.cos(samples + bands / 7) + samples * bands / 100
    y = 1.3 + 0.7 * values[:, 1] - 0.4 * values[:, 5]
    names = [f"wl{index}" for index in range(1, 18)]
    train = pd.DataFrame(values, columns=names)
    held = pd.DataFrame(values[[1, 7, 16]] + 0.031, columns=names)
    augmented = run_native("gaussian_noise", values, [0.03], 42)
    augmented = run_native("multiplicative_noise", augmented, [0.02], 7)
    np.testing.assert_allclose(augmented, oracle["augmented_train"], rtol=0, atol=1e-12)
    with PortableN4MTrainedPipeline(document) as imported:
        np.testing.assert_allclose(imported.predict(held), oracle["predictions"], rtol=0, atol=1e-10)
        with imported.retrain(pd.DataFrame(values + 0.02, columns=names), y + 0.01) as retrained:
            np.testing.assert_allclose(retrained.predict(held), oracle["retrain_predictions"], rtol=0, atol=1e-10)
            assert json.loads(retrained.to_json())["schema"] == "nirs4all.n4m.trained_pipeline.v7"


@pytest.mark.methods
@pytest.mark.parametrize("node", [
    _augmentation(seed=-1), _augmentation(seed=2**53),
    _augmentation(values=[]), _augmentation(values=[float("nan")]),
    _augmentation(kind="mixup"),
    {"train_augmentation": {"class": "n4m.NativeXAugmentation", "params": {
        "kind": "gaussian_noise", "values": [0.03],
    }}},
])
def test_v7_rejects_invalid_train_augmentation(node: dict) -> None:
    train, y, _ = _data()
    with pytest.raises(ValueError):
        PortableN4MTrainedPipeline.fit_recipe(
            _recipe(node), train, y, preprocessing="native_n4mp",
        )


@pytest.mark.methods
def test_v7_refuses_nonprefix_and_manifest_tampering() -> None:
    train, y, _ = _data()
    nonprefix = {"pipeline": [{"class": "n4m.SNV"}, _augmentation(),
                              {"model": {"class": "n4m.PLS", "params": {"n_components": 2}}}]}
    with pytest.raises(ValueError):
        PortableN4MTrainedPipeline.fit_recipe(
            nonprefix, train, y, preprocessing="native_n4mp",
        )
    with PortableN4MTrainedPipeline.fit_recipe(
        _recipe(_augmentation()), train, y, preprocessing="native_n4mp",
    ) as fitted:
        document = json.loads(fitted.to_json())
    document["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest hash"):
        PortableN4MTrainedPipeline(document)
    document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
    manifest = json.loads(document["manifest_json"])
    manifest["recipe"]["pipeline"][0]["train_augmentation"]["params"]["seed"] = -1
    document["manifest_json"] = json.dumps(manifest, separators=(",", ":"))
    document["manifest_sha256"] = hashlib.sha256(document["manifest_json"].encode()).hexdigest()
    with pytest.raises(ValueError, match="train augmentation"):
        PortableN4MTrainedPipeline(document)
