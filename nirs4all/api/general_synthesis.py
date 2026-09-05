"""Bounded synthetic-dataset callable for the Rust-owned Studio host.

Rust owns transport, workspace authorization and linking.  This module accepts
one closed JSON request, delegates generation to :class:`SyntheticDatasetBuilder`
and publishes only a standard dataset folder below an explicitly supplied
output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from nirs4all.data.dataset import SpectroDataset
from nirs4all.synthesis.builder import SyntheticDatasetBuilder

from .studio_scientific import StudioScientificJobError
from .studio_scientific_general import _validate_json

STUDIO_SYNTHETIC_DATASET_JOB_SCHEMA: Final = "nirs4all.studio-synthetic-dataset-job.v1"
STUDIO_SYNTHETIC_DATASET_RESULT_SCHEMA: Final = "nirs4all.studio-synthetic-dataset-result.v1"
MAX_SYNTHETIC_REQUEST_BYTES: Final = 65_536
MAX_SYNTHETIC_RESPONSE_BYTES: Final = 65_536
MAX_SYNTHETIC_CELLS: Final = 100_000_000
MAX_SYNTHETIC_FEATURES: Final = 10_000
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TASK_TYPES = frozenset({"regression", "binary_classification", "multiclass_classification"})
_COMPLEXITIES = frozenset({"simple", "realistic", "complex"})


def _refuse(code: str, message: str) -> StudioScientificJobError:
    return StudioScientificJobError(code, message)


def _object(value: Any, *, required: set[str], allowed: set[str], where: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise _refuse("invalid_shape", f"{where} must be a JSON object")
    result = cast(dict[str, Any], value)
    missing = sorted(required - result.keys())
    unknown = sorted(result.keys() - allowed)
    if missing:
        raise _refuse("missing_field", f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise _refuse("unknown_field", f"{where} contains unknown fields: {', '.join(unknown)}")
    return result


def _integer(value: Any, where: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _refuse("invalid_integer", f"{where} must be an integer between {minimum} and {maximum}")
    return value


def _number(value: Any, where: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise _refuse("invalid_range", f"{where} must contain finite numbers")
    return float(value)


def _range(value: Any, where: str) -> tuple[float, float]:
    if type(value) is not list or len(value) != 2:
        raise _refuse("invalid_range", f"{where} must be a pair [minimum, maximum]")
    minimum, maximum = _number(value[0], where), _number(value[1], where)
    if minimum >= maximum:
        raise _refuse("invalid_range", f"{where} minimum must be smaller than maximum")
    return minimum, maximum


def _authorized_output(value: Any) -> Path:
    if type(value) is not str or not value or len(value.encode("utf-8")) > 4096:
        raise _refuse("output_forbidden", "output_dir must be one bounded absolute path authorized by Rust")
    raw = Path(value)
    if not raw.is_absolute() or raw.is_symlink():
        raise _refuse("output_forbidden", "output_dir must be an existing non-symlink absolute directory")
    try:
        resolved = raw.resolve(strict=True)
    except OSError as error:
        raise _refuse("output_forbidden", f"output_dir cannot be resolved: {error}") from error
    if not resolved.is_dir():
        raise _refuse("output_forbidden", "output_dir must be an existing directory")
    return resolved


def _csv_shape(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter=";")
        try:
            header = next(rows)
        except StopIteration as error:
            raise _refuse("generation_failed", f"generated file {path.name} is empty") from error
        return sum(1 for _ in rows), len(header)


def _manifest(path: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for child in sorted(path.iterdir()):
        if not child.is_file() or child.name not in {"Xcal.csv", "Ycal.csv", "Xval.csv", "Yval.csv"}:
            raise _refuse("generation_failed", "standard export produced an unexpected artifact set")
        digest = hashlib.sha256(child.read_bytes()).hexdigest()
        files.append({"path": child.name, "bytes": child.stat().st_size, "sha256": digest})
    if {item["path"] for item in files} != {"Xcal.csv", "Ycal.csv", "Xval.csv", "Yval.csv"}:
        raise _refuse("generation_failed", "standard export did not produce all four train/test artifacts")
    return files


def studio_synthetic_dataset_job_v1(request: object) -> dict[str, Any]:
    """Generate one bounded standard dataset without owning Studio state."""
    _validate_json(request)
    try:
        request_size = len(json.dumps(request, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise _refuse("non_json_value", "request must contain finite plain JSON values") from error
    if request_size > MAX_SYNTHETIC_REQUEST_BYTES:
        raise _refuse("request_too_large", "synthetic dataset request exceeds 64 KiB")

    envelope = _object(
        request,
        required={"schema", "operation", "request_id", "output_dir", "payload"},
        allowed={"schema", "operation", "request_id", "output_dir", "payload"},
        where="request",
    )
    if envelope["schema"] != STUDIO_SYNTHETIC_DATASET_JOB_SCHEMA or envelope["operation"] != "generate":
        raise _refuse("unsupported_contract", "unsupported synthetic dataset schema or operation")
    request_id = envelope["request_id"]
    if type(request_id) is not str or not request_id or len(request_id.encode("utf-8")) > 256:
        raise _refuse("invalid_request_id", "request_id must be a non-empty bounded string")
    output_dir = _authorized_output(envelope["output_dir"])
    payload = _object(
        envelope["payload"],
        required={"task_type", "n_samples", "complexity", "train_ratio", "random_state"},
        allowed={"task_type", "n_samples", "complexity", "n_classes", "target_range", "train_ratio", "wavelength_range", "name", "random_state"},
        where="payload",
    )

    task_type = payload["task_type"]
    if type(task_type) is not str or task_type not in _TASK_TYPES:
        raise _refuse("invalid_task_type", "task_type must be regression, binary_classification or multiclass_classification")
    complexity = payload["complexity"]
    if type(complexity) is not str or complexity not in _COMPLEXITIES:
        raise _refuse("invalid_complexity", "complexity must be simple, realistic or complex")
    complexity = cast(Literal["simple", "realistic", "complex"], complexity)
    n_samples = _integer(payload["n_samples"], "payload.n_samples", minimum=50, maximum=10_000)
    random_state = _integer(payload["random_state"], "payload.random_state", minimum=0, maximum=2**32 - 1)
    train_ratio = _number(payload["train_ratio"], "payload.train_ratio")
    if not 0.5 <= train_ratio <= 0.95:
        raise _refuse("invalid_range", "payload.train_ratio must be between 0.5 and 0.95")

    wavelength_range = _range(payload.get("wavelength_range", [1000.0, 2500.0]), "payload.wavelength_range")
    estimated_features = math.ceil((wavelength_range[1] - wavelength_range[0]) / 2.0 + 1)
    if estimated_features > MAX_SYNTHETIC_FEATURES or n_samples * estimated_features > MAX_SYNTHETIC_CELLS:
        raise _refuse("resource_limit", "requested synthetic matrix exceeds the 10,000-feature or 100,000,000-cell limit")

    supplied_name = payload.get("name", f"synthetic-{random_state}")
    if type(supplied_name) is not str or not _NAME.fullmatch(supplied_name) or supplied_name in {".", ".."}:
        raise _refuse("invalid_name", "payload.name must be a safe identifier of at most 128 characters")
    name = cast(str, supplied_name)
    target = output_dir / name
    if target.exists() or target.is_symlink():
        raise _refuse("output_exists", "the requested dataset output already exists")

    n_classes: int | None = None
    target_range: tuple[float, float] | None = None
    if task_type == "regression":
        if "n_classes" in payload:
            raise _refuse("unknown_field", "payload.n_classes is only valid for classification")
        if "target_range" in payload:
            target_range = _range(payload["target_range"], "payload.target_range")
    else:
        if "target_range" in payload:
            raise _refuse("unknown_field", "payload.target_range is only valid for regression")
        default_classes = 2 if task_type == "binary_classification" else 3
        n_classes = _integer(payload.get("n_classes", default_classes), "payload.n_classes", minimum=2, maximum=20)
        if task_type == "binary_classification" and n_classes != 2:
            raise _refuse("invalid_integer", "binary_classification requires exactly two classes")
        if task_type == "multiclass_classification" and n_classes < 3:
            raise _refuse("invalid_integer", "multiclass_classification requires at least three classes")

    try:
        temporary = Path(tempfile.mkdtemp(prefix=".n4a-synthetic-", dir=output_dir))
    except OSError as error:
        raise _refuse("output_forbidden", f"cannot create output below the authorized directory: {error}") from error
    target_created = False
    published = False
    try:
        builder = SyntheticDatasetBuilder(n_samples=n_samples, random_state=random_state, name=name)
        builder.with_features(wavelength_range=wavelength_range, complexity=complexity)
        if target_range is not None:
            builder.with_targets(range=target_range)
        if n_classes is not None:
            builder.with_classification(n_classes=n_classes)
        builder.with_partitions(train_ratio=train_ratio, shuffle=True)
        dataset = builder.build()
        if not isinstance(dataset, SpectroDataset):
            raise _refuse("generation_failed", "builder did not produce a SpectroDataset")
        builder.export(temporary, format="standard")

        train, features = _csv_shape(temporary / "Xcal.csv")
        test, test_features = _csv_shape(temporary / "Xval.csv")
        train_targets, _ = _csv_shape(temporary / "Ycal.csv")
        test_targets, _ = _csv_shape(temporary / "Yval.csv")
        if train + test != n_samples or train != train_targets or test != test_targets or features != test_features or features > MAX_SYNTHETIC_FEATURES:
            raise _refuse("generation_failed", "standard export dimensions do not match the generated dataset")
        actual_task = dataset.task_type.value if dataset.task_type is not None else None
        if actual_task != task_type:
            raise _refuse("generation_failed", f"generator reported unexpected task type {actual_task!r}")
        y = np.asarray(dataset.y({})).reshape(-1)
        actual_classes = int(np.unique(y).size) if task_type != "regression" else None
        if n_classes is not None and actual_classes != n_classes:
            raise _refuse("generation_failed", "generator did not produce every requested class")
        manifest = _manifest(temporary)

        response = {
            "schema": STUDIO_SYNTHETIC_DATASET_RESULT_SCHEMA,
            "request_id": request_id,
            "result": {
                "name": name,
                "relative_path": name,
                "files": manifest,
                "summary": {"samples": train + test, "features": features, "train": train, "test": test, "task": actual_task, "classes": actual_classes},
                "generation": {"random_state": random_state, "complexity": complexity, "train_ratio": train_ratio, "wavelength_range": list(wavelength_range)},
            },
        }
        response_size = len(json.dumps(response, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if response_size > MAX_SYNTHETIC_RESPONSE_BYTES:
            raise _refuse("response_too_large", "synthetic dataset response exceeds 64 KiB")
        try:
            target.mkdir(mode=0o755)
        except FileExistsError as error:
            raise _refuse("output_exists", "the requested dataset output was created concurrently") from error
        target_created = True
        for child in temporary.iterdir():
            os.replace(child, target / child.name)
        temporary.rmdir()
        published = True
        return response
    except StudioScientificJobError:
        raise
    except Exception as error:
        raise _refuse("generation_failed", f"synthetic dataset generation failed: {error}") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
        if target_created and not published and target.exists():
            shutil.rmtree(target)
