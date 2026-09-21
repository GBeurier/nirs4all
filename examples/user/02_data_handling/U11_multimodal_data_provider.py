"""Execute synthetic data providers before native CV, then resume finite batches.

Run: python examples/user/02_data_handling/U11_multimodal_data_provider.py --output /tmp/mm-provider
The existing NIRS generator is reused; no real corpus is required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
from nirs4all_io import DataProvider, MultimodalDataset, TensorSource
from nirs4all_io.provider_adapters import SklearnProviderAdapter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import MultimodalRegressor, TensorPCA


def generate_cohort(*, seed: int, params: dict[str, Any], context: dict[str, Any]) -> MultimodalDataset:
    """Produce one complete aligned cohort, independently of later batch sizes."""
    count = int(params.get("n_samples", 32))
    # The scientific generator accepts a sklearn-compatible uint32 seed.
    synthesis_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
    spectra, targets = nirs4all.generate(
        count, random_state=synthesis_seed, complexity="simple",
        wavelength_range=(1000, 1150), as_dataset=False,
    )
    # generate(as_dataset=False) returns all component concentrations; predict
    # the first concentration in this single-target regression example.
    targets = np.asarray(targets)[:, 0]
    rng = np.random.default_rng(seed)
    latent = (targets - targets.mean()) / max(float(targets.std()), 1e-12)
    ids = [f"{context.get('prefix', 'sample')}-{index:03d}" for index in range(count)]
    sources = {
        "nir": TensorSource(spectra, ids, representation_id="signal_1d", axis_units={"wavelength": "nm"}),
        "image": TensorSource(latent[:, None, None, None] + rng.normal(scale=0.2, size=(count, 3, 3, 3)), ids, representation_id="rgb_image"),
        "series": TensorSource(latent[:, None, None] + rng.normal(scale=0.2, size=(count, 6, 2)), ids, representation_id="series_mv", axis_units={"time": "s"}),
        "metadata": TensorSource(np.column_stack([rng.normal(size=count), np.arange(count) / count]), ids, representation_id="tabular_mixed", feature_names=["covariate", "position"]),
    }
    n_train = 2 * ((count - count // 4) // 2)  # Keep each repeated group in one partition.
    return MultimodalDataset(sources, sample_ids=ids, y=targets, groups=[f"group-{index // 2}" for index in range(count)],
                             partitions=["train"] * n_train + ["test"] * (count - n_train), name="provider-demo", task_type="regression")


def make_provider() -> DataProvider:
    return DataProvider(generate_cohort, provider_id="nirs4all.example.synthetic", provider_version="1", params={"n_samples": 32}, seed=17)


def make_pipeline() -> list[Any]:
    return [GroupKFold(3), MultimodalRegressor({
        "nir": StandardScaler(), "image": TensorPCA(2, random_state=17),
        "series": TensorPCA(2, random_state=17), "metadata": StandardScaler(),
    }, model=Ridge(alpha=1.0))]


def run_demo(output: Path, *, torch_workers: int | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    provider = make_provider()
    result = nirs4all.run(make_pipeline(), provider, random_state=17, save_charts=False, verbose=0, workspace_path=output / "workspace")
    try:
        archive = result.export(output / "provider.n4a")
        evidence = next(iter(result.per_dataset.values()))["data_provider_evidence"]
        # The native task seed is retained in the provider checkpoint.
        batches = provider.batches(7)
        first = next(batches)
        state = batches.state_dict()
        (output / "batches.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        expected_tail = [sample for batch in batches for sample in batch.sample_ids]
        restored = make_provider()
        restored.load_state_dict(state["provider"])
        resumed = restored.batches(7)
        resumed.load_state_dict(state)
        actual_tail = [sample for batch in resumed for sample in batch.sample_ids]
        assert expected_tail == actual_tail
        assert len(first.sample_ids) + len(actual_tail) == len(provider)

        training_ids = [sample for sample, partition in zip(provider.cohort.sample_ids, provider.cohort.partitions, strict=True) if partition == "train"]
        X, y = SklearnProviderAdapter(provider, source="nir").arrays(training_ids)
        standalone = Ridge(alpha=1.0).fit(X, y)
        assert standalone.predict(X).shape == y.shape
        torch_rows = None
        if torch_workers is not None:
            from nirs4all_io.provider_adapters import TorchIterableDataset, collate_provider_samples
            from torch.utils.data import DataLoader

            loader = DataLoader(TorchIterableDataset(provider, sample_ids=training_ids, return_metadata=True),
                                batch_size=7, num_workers=torch_workers, collate_fn=collate_provider_samples,
                                **({"multiprocessing_context": "spawn"} if torch_workers else {}))
            loaded_ids = [sample for batch in loader for sample in batch["sample_id"]]
            assert len(loaded_ids) == len(set(loaded_ids)) == len(training_ids)
            assert set(loaded_ids) == set(training_ids)
            torch_rows = len(loaded_ids)

        # Supply only the image source while preserving the fixed NIRS/y rows.
        complete = provider.cohort
        base = MultimodalDataset({key: value for key, value in complete.sources.items() if key != "image"},
                                sample_ids=complete.sample_ids, y=complete.y, groups=complete.groups,
                                partitions=complete.partitions, name=complete.name, task_type="regression")

        def image_source(**kwargs: Any) -> dict[str, Any]:
            return {"sample_ids": complete.sample_ids, "sources": {"image": complete.sources["image"]}}

        partial = DataProvider(image_source, provider_id="nirs4all.example.image", base=base)
        partial_result = nirs4all.run(make_pipeline(), partial, random_state=17, save_charts=False, verbose=0, workspace_path=output / "partial-workspace")
        try:
            partial_score = partial_result.cv_best_score
        finally:
            partial_result.close()
        new_data = generate_cohort(seed=29, params={"n_samples": 12}, context={"prefix": "new"})
        prediction_data = MultimodalDataset(new_data.sources, sample_ids=new_data.sample_ids, partitions=["predict"] * 12, name="prediction-inputs")
        with patch.object(DataProvider, "materialize", side_effect=AssertionError("Replay must not generate training data")):
            prediction = nirs4all.predict(archive, prediction_data)
        (output / "prediction_dataset.json").write_text(json.dumps(prediction_data.to_dict(), indent=2) + "\n", encoding="utf-8")
        (output / "predictions.json").write_text(json.dumps(np.asarray(prediction.values).tolist()) + "\n", encoding="utf-8")
        report = {"engine": result.execution_engine, "provider": evidence, "cv_score": result.cv_best_score,
                  "partial_cv_score": partial_score, "resumed_rows": len(actual_tail), "replay_rows": len(prediction.values),
                  "sklearn_rows": len(y), "torch_rows": torch_rows}
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        result.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("provider-demo"))
    parser.add_argument("--torch-workers", type=int, default=None, help="Also exercise optional PyTorch DataLoader with this worker count")
    args = parser.parse_args()
    print(json.dumps(run_demo(args.output, torch_workers=args.torch_workers), indent=2))
