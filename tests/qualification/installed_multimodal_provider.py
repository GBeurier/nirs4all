"""Qualify an installed four-modality provider without importing source checkouts."""

import os
from pathlib import Path

import nirs4all_io
import numpy as np
from nirs4all_io import DataProvider, MultimodalDataset, TensorSource
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA
from nirs4all.pipeline.dagml.multimodal_tuning import MultimodalTuningStopped

workspace = Path(__file__).resolve().parents[2]
assert not Path(nirs4all.__file__).resolve().is_relative_to(workspace)
assert not Path(nirs4all_io.__file__).resolve().is_relative_to(workspace)
root = Path(os.environ.get("N4A_SMOKE_ROOT", "/tmp/n4a-installed-provider-qualification"))


def cohort(*, prediction: bool = False) -> MultimodalDataset:
    rng = np.random.default_rng(181 if prediction else 79)
    rows = 5 if prediction else 16
    ids = [f"{'new' if prediction else 'train'}-{index:02d}" for index in range(rows)]
    latent = rng.normal(size=rows)
    nir = latent[:, None] + rng.normal(scale=0.2, size=(rows, 6))
    image = latent[:, None, None, None] + rng.normal(scale=0.1, size=(rows, 2, 2, 3))
    series = latent[:, None, None] + rng.normal(scale=0.15, size=(rows, 5, 2))
    metadata = np.array([[float(index) / rows, "a" if index % 2 else "b"] for index in range(rows)], dtype=object)
    sources = {
        "nir": TensorSource(nir, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"}, axis_coordinates={"wavelength": np.linspace(900, 1700, 6)}),
        "image": TensorSource(image, ids, representation_id="rgb_image"),
        "series": TensorSource(series, ids, representation_id="series_mv", axis_units={"time": "s"}, axis_coordinates={"time": np.arange(5)}),
        "metadata": TensorSource(metadata, ids, representation_id="tabular_mixed", feature_names=["age", "batch"]),
    }
    return MultimodalDataset(
        sources,
        sample_ids=ids,
        y=None if prediction else 2.1 * latent + np.arange(rows) * 0.03,
        groups=None if prediction else [f"plant-{index // 2}" for index in range(rows)],
        partitions=["predict"] * rows if prediction else ["train"] * 12 + ["test"] * 4,
        name="installed-wheel-qualification",
    )


def model() -> MultimodalRegressor:
    return MultimodalRegressor(
        {
            "nir": StandardScaler(),
            "image": TensorPCA(n_components=2, random_state=19),
            "series": TensorPCA(n_components=2, random_state=19),
            "metadata": ColumnTransformer(
                [
                    ("numeric", StandardScaler(), [0]),
                    ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [1]),
                ]
            ),
        },
        model=Ridge(alpha=0.2),
    )


def provider() -> DataProvider:
    base = cohort()

    def generate(**_: object) -> dict[str, object]:
        return {"sample_ids": list(base.sample_ids), "sources": {"nir": base.sources["nir"]}}

    def generate_view(*, sample_ids: list[str], seed: int, **_: object) -> dict[str, object]:
        source = base.take(sample_ids).sources["nir"]
        return {
            "sample_ids": sample_ids,
            "sources": {
                "nir": TensorSource(
                    np.asarray(source.values) + float(seed % 7),
                    sample_ids,
                    representation_id=source.representation_id,
                    axis_units=source.axis_units,
                    axis_coordinates=source.axis_coordinates,
                )
            },
        }

    return DataProvider(generate, generate_view=generate_view, provider_id="qualification.installed-wheel", base=base, replace_sources=["nir"])


def qualify_run_scoped_xy() -> None:
    """Exercise complete and partial X/y providers through the installed public API."""
    base = cohort()
    fixed = MultimodalDataset(
        {"nir": base.sources["nir"]}, sample_ids=base.sample_ids,
        groups=base.groups, partitions=base.partitions, name=base.name,
    )
    generators = (
        DataProvider(lambda **_: base, provider_id="qualification.installed-xy-complete"),
        DataProvider(
            lambda **_: {
                "sample_ids": list(base.sample_ids),
                "sources": {name: source for name, source in base.sources.items() if name != "nir"},
                "y": base.y,
            },
            provider_id="qualification.installed-xy-partial", base=fixed,
        ),
    )
    scores = []
    prior_mode = os.environ.get("N4A_DAGML_INPROCESS")
    os.environ["N4A_DAGML_INPROCESS"] = "1"
    try:
        for label, candidate in zip(("complete", "partial"), generators, strict=True):
            result = nirs4all.run(
                [GroupKFold(3), {"model": model()}], candidate,
                engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
                verbose=0, random_state=19, results_path=root / f"xy-{label}-results",
            )
            try:
                assert set(candidate.cohort.sources) == set(base.sources)
                np.testing.assert_array_equal(candidate.cohort.y, base.y)
                assert candidate.cohort.sample_ids == base.sample_ids
                np.testing.assert_array_equal(candidate.cohort.groups, base.groups)
                np.testing.assert_array_equal(candidate.cohort.partitions, base.partitions)
                assert np.isfinite(result.best_rmse)
                archive = result.export(root / f"xy-{label}.n4a")
                prediction = nirs4all.predict(archive, cohort(prediction=True))
                assert prediction.metadata["training_performed"] is False
                assert len(prediction.y_pred) == 5 and np.isfinite(prediction.y_pred).all()
                scores.append(result.best_rmse)
            finally:
                result.close()
    finally:
        if prior_mode is None:
            os.environ.pop("N4A_DAGML_INPROCESS", None)
        else:
            os.environ["N4A_DAGML_INPROCESS"] = prior_mode
    np.testing.assert_allclose(scores[0], scores[1], rtol=0, atol=0)
    print("INSTALLED_XY_ASSEMBLY_OK", scores)


qualify_run_scoped_xy()


trained = nirs4all.run(
    [GroupKFold(3), {"model": model()}],
    provider(),
    engine="dag-ml",
    refit=True,
    save_artifacts=False,
    save_charts=False,
    verbose=0,
    random_state=19,
    results_path=root / "cv-results",
)
try:
    assert np.isfinite(trained.best_rmse)
    assert trained._dagml_generated_view_manifest["views"]
    archive = trained.export(root / "installed-multimodal.n4a")
    prediction = nirs4all.predict(archive, cohort(prediction=True))
    assert prediction.metadata["training_performed"] is False
    assert np.isfinite(prediction.y_pred).all()
    print("INSTALLED_CV_ARCHIVE_OK", trained.best_rmse, len(trained._dagml_generated_view_manifest["views"]))
finally:
    trained.close()

if os.environ.get("N4A_DAGML_INPROCESS") == "0" and os.environ.get("ENABLE_HPO_SUBPROCESS") != "1":
    raise SystemExit(0)  # Generated HPO subprocess is deliberately unopened.


search = nirs4all.run(
    [GroupKFold(3), {"model": model()}],
    provider(),
    tuning={
        "engine": "n4m",
        "sampler": "random",
        "seed": 19,
        "metric": "rmse",
        "n_trials": 3,
        "n_jobs": 2,
        "space": {"model__alpha": [0.1, 1.0]},
        "storage": (root / ("hpo-study-subprocess-source" if os.environ.get("ENABLE_HPO_SUBPROCESS") == "1" else "hpo-study-pickle")).as_uri(),
        "study_name": "installed-hpo",
    },
    engine="dag-ml",
    refit=True,
    save_artifacts=False,
    save_charts=False,
    verbose=0,
    random_state=19,
    results_path=root / "hpo-results",
)
try:
    assert len(search.tuning_result.trials) == 3
    import cloudpickle

    payload = cloudpickle.dumps(search)
    restored = cloudpickle.loads(payload)
    assert restored.tuning_best_value == search.tuning_best_value
    print("HPO_RESULT_PICKLE_OK", len(payload))
    assert search._dagml_generated_view_manifest["views"]
    archive = search.export(root / "installed-hpo.n4a")
    prediction = nirs4all.predict(archive, cohort(prediction=True))
    assert prediction.metadata["training_performed"] is False
    assert np.isfinite(prediction.y_pred).all()
    print("INSTALLED_HPO_ARCHIVE_OK", search.tuning_best_value, len(search._dagml_generated_view_manifest["views"]))
finally:
    search.close()

study = root / "progress-study"
events = []


def stop_after_first(event):
    count = len(event["checkpoint"]["trials"])
    events.append((os.getpid(), count))
    return count < 1


spec = {"engine": "n4m", "sampler": "random", "seed": 19, "metric": "rmse", "n_trials": 2, "n_jobs": 1, "space": {"model__alpha": [0.1, 1.0]}, "storage": study.as_uri(), "study_name": "installed-progress"}
try:
    nirs4all.run(
        [GroupKFold(3), {"model": model()}],
        provider(),
        tuning={**spec, "progress_callback": stop_after_first},
        engine="dag-ml",
        refit=True,
        save_artifacts=False,
        save_charts=False,
        verbose=0,
        random_state=19,
        results_path=root / "stopped",
    )
except MultimodalTuningStopped:
    pass
else:
    raise AssertionError("progress callback did not stop")
assert events and all(pid == os.getpid() for pid, _ in events) and max(count for _, count in events) == 1, events


def continue_search(event):
    events.append((os.getpid(), len(event["checkpoint"]["trials"])))


resumed = nirs4all.run(
    [GroupKFold(3), {"model": model()}],
    provider(),
    tuning={**spec, "resume": True, "progress_callback": continue_search},
    engine="dag-ml",
    refit=True,
    save_artifacts=False,
    save_charts=False,
    verbose=0,
    random_state=19,
    results_path=root / "resumed",
)
try:
    assert len(resumed.tuning_result.trials) == 2
    assert events[-1] == (os.getpid(), 2), events
    print("INSTALLED_PROGRESS_RESUME_OK", events)
finally:
    resumed.close()


if os.environ.get("N4A_FULL_HPO_MATRIX") == "1":
    samplers = ("random", "sobol", "lhs", "ternary", "ga", "pso", "cmaes", "tpe", "gp_ei")
    pruners = ("none", "median", "asha", "hyperband", "racing")
    for sampler in samplers:
        for pruner in pruners:
            candidate = nirs4all.run(
                [GroupKFold(3), {"model": model()}], provider(),
                tuning={"engine": "n4m", "sampler": sampler, "pruner": pruner, "seed": 19,
                        "metric": "rmse", "n_trials": 2, "space": {"model__alpha": (0.01, 1.0)}},
                engine="dag-ml", refit=True, save_artifacts=False, save_charts=False,
                results_path=root / "optimizer-matrix" / f"{sampler}-{pruner}",
                random_state=19, verbose=0,
            )
            try:
                assert [trial.state for trial in candidate.tuning_result.trials] == ["COMPLETE", "COMPLETE"]
                assert candidate.tuning_best_value == min(trial.value for trial in candidate.tuning_result.trials)
                assert candidate._dagml_generated_view_manifest["views"]
                assert len(candidate._dagml_refit_artifacts) == 1
            finally:
                candidate.close()
    print("INSTALLED_GENERATED_HPO_MATRIX_OK", len(samplers) * len(pruners))
