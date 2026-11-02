"""Keep target-missing samples out of training and scoring only.

A sample whose selected target is missing stays in the dataset: loading,
preview and analysis keep it unless the user asked ``remove_sample``. A DAG-ML
training run excludes it ONLY from fitting and scoring, as DAG-ML 0.3.44
specifies for ``FoldSet.target_missing_sample_ids``. Exclusion is the only
training behavior: DAG-ML refuses non-finite scoring targets in host results
and dag-ml-data refuses them in provider target tables, so no model can be
trained or scored on a missing target. During a run:

* the host moves it out of the train/test partitions of its working dataset
  for the duration of the run, so no fold, CV relation, held-out test cohort
  or provider target table ever contains it (the splitter partitions only the
  target-observed samples);
* every fold set of the run declares it in ``target_missing_sample_ids``;
* the captured refit model then predicts it through a native ``inference``
  cohort, published as a ``prediction`` row without targets or scores.

The selected targets are the target columns of the dataset being trained: a
sample is target-missing when any of them is missing, because one estimator
fits them together. Training a single target of a sparse multi-target dataset
loads that target alone, so only the samples missing it are excluded.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

PREDICTION_PARTITION = "prediction"
"""Partition label of prediction-only rows, as ``save_workspace_predict_result`` writes them."""

_ISOLATION_PARTITION = "predict"


@dataclass(frozen=True)
class TargetMissingSamples:
    """Sample ints whose selected target is missing, by original partition."""

    train: tuple[int, ...]
    test: tuple[int, ...]

    @property
    def samples(self) -> tuple[int, ...]:
        """Every target-missing sample int in storage order."""
        return tuple(sorted((*self.train, *self.test)))

    def to_metadata(self) -> dict[str, Any]:
        """JSON record of the exclusion for result provenance."""
        return {
            "policy": "excluded_from_training_and_scoring",
            "train_sample_indices": list(self.train),
            "test_sample_indices": list(self.test),
            "count": len(self.train) + len(self.test),
        }


def missing_target_rows(values: Any) -> np.ndarray:
    """Boolean mask of the rows where at least one target value is missing."""
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1, 1)
    rows = array.reshape(len(array), -1)
    return np.asarray(pd.isna(rows).any(axis=1), dtype=bool)


def find_target_missing(spectro: Any) -> TargetMissingSamples | None:
    """Return the train/test samples with a missing selected target, or ``None``.

    Multimodal IO cohorts carry their own authoritative ``target_mask`` and are
    left to that contract.
    """
    from nirs4all.data.multimodal import MultimodalSpectroDataset

    from .envelope import num_targets

    if isinstance(spectro, MultimodalSpectroDataset) or num_targets(spectro) == 0:
        return None
    found: dict[str, tuple[int, ...]] = {}
    for partition in ("train", "test"):
        selector = {"partition": partition}
        samples = spectro._indexer.x_indices(selector, include_augmented=False, include_excluded=True)
        values = spectro.y(selector, include_augmented=False, include_excluded=True)
        mask = missing_target_rows(values) if len(samples) else np.zeros(0, dtype=bool)
        found[partition] = tuple(int(sample) for sample in np.asarray(samples)[mask])
    if not found["train"] and not found["test"]:
        return None
    if len(found["train"]) == len(spectro._indexer.x_indices({"partition": "train"}, include_augmented=False, include_excluded=True)):
        raise ValueError("every training sample has a missing target; at least one observed target is required to train")
    return TargetMissingSamples(train=found["train"], test=found["test"])


@contextmanager
def isolate_target_missing(spectro: Any, missing: TargetMissingSamples | None) -> Iterator[None]:
    """Hold target-missing samples outside train/test while a run executes.

    The working dataset keeps every sample, so wire identities do not change.
    The samples' original partitions and the identity declaration are
    restored on exit, whatever the outcome of the run.
    """
    if missing is None:
        yield
        return
    spectro._indexer.update_by_indices(list(missing.samples), {"partition": _ISOLATION_PARTITION})
    spectro._dagml_target_missing = missing.samples
    try:
        yield
    finally:
        del spectro._dagml_target_missing
        for partition, samples in (("train", missing.train), ("test", missing.test)):
            if samples:
                spectro._indexer.update_by_indices(list(samples), {"partition": partition})


def attach_target_missing_predictions(result: Any, spectro: Any, missing: TargetMissingSamples) -> None:
    """Record the exclusion and predict the excluded samples with the refit model.

    ``spectro`` is the dataset as it was before the run: execution may add
    rows (augmentation) that are not prediction inputs.

    The run's exported refit model (single estimator or native aggregate such
    as a stack) is replayed through a native PREDICT-only ``inference`` cohort,
    exactly as ``nirs4all.predict`` replays an archive. Exporting persists
    nothing in the caller's workspace and leaves the result's export state
    unchanged. A run without an exportable refit model (``refit=False``, or a
    topology whose export is refused) records the exclusion and the reason,
    without predictions.
    """
    from nirs4all.core.logging import get_logger

    from .general_archive import predict_general_archive
    from .identity import mint_identity
    from .rt import RtError

    record: dict[str, Any] = {**missing.to_metadata(), "predicted": False}
    for metadata in result.per_dataset.values():
        metadata["target_missing"] = record
    if not result._dagml_refit_artifacts:
        record["unpredicted_reason"] = "the run has no refit model"
    else:
        export_state = (result._dagml_results_dir, result._dagml_export_directory)
        try:
            with TemporaryDirectory(prefix="n4a-target-missing-") as directory:
                prediction = predict_general_archive(result.export(Path(directory) / "refit.n4a"), spectro)
        except (RtError, NotImplementedError) as exc:
            record["unpredicted_reason"] = f"the refit model cannot be exported: {exc}"
        finally:
            export_directory = result._dagml_export_directory
            if export_directory is not None and export_directory is not export_state[1]:
                export_directory.cleanup()
            result._dagml_results_dir, result._dagml_export_directory = export_state
    if "unpredicted_reason" in record:
        get_logger(__name__).warning(
            "%d target-missing sample(s) were excluded from training and scoring and are not predicted: %s",
            record["count"], record["unpredicted_reason"],
        )
        return
    record["predicted"] = True
    identity = mint_identity(spectro)
    row_of = {sample.sample_int: row for row, sample in enumerate(identity.identities)}
    samples = list(missing.samples)
    # Label the row with the refit entry, or the CV winner for compositions
    # (e.g. stacking) that project no refit rows.
    source = result.final or result.best
    values = np.asarray(prediction.y_pred)[[row_of[sample] for sample in samples]]
    if "classif" in str(source.get("task_type", "")):
        # Replay decodes the user's labels; the run's rows hold class ids.
        values = np.asarray(spectro._targets.transform_predictions(values, "raw", "numeric"), dtype=float)
        values = values.ravel() if values.ndim == 2 and values.shape[1] == 1 else values
    result.predictions.add_prediction(
        dataset_name=spectro.name,
        config_name=source.get("config_name", ""),
        model_name=source.get("model_name", ""),
        model_classname=source.get("model_classname", ""),
        fold_id="final",
        partition=PREDICTION_PARTITION,
        sample_indices=samples,
        metadata={"physical_sample_id": [identity.to_wire(sample) for sample in samples]},
        y_pred=values,
        metric=source.get("metric", ""),
        task_type=source.get("task_type", "regression"),
        n_samples=len(samples),
        refit_context="standalone",
        result_metadata={"target_missing_prediction": {
            "schema": "nirs4all.target-missing-prediction.v1",
            "phase": prediction.metadata["phase"],
            "predict_cohort_role": prediction.metadata["predict_cohort"]["role"],
            "archive_fingerprint": prediction.metadata["archive_fingerprint"],
            "train_sample_indices": list(missing.train),
            "test_sample_indices": list(missing.test),
        }},
    )
    result.predictions.flush()
