"""Phase A2 adapter from canonical prior records to finite dataset runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, NoReturn, cast

import numpy as np

from nirs4all.synthesis import ComponentLibrary, SyntheticNIRSGenerator
from nirs4all.synthesis.domains import get_domain_config
from nirs4all.synthesis.environmental import EnvironmentalEffectsConfig, TemperatureConfig
from nirs4all.synthesis.instruments import EdgeArtifactsConfig, get_instrument_archetype
from nirs4all.synthesis.scattering import (
    ParticleSizeConfig,
    ParticleSizeDistribution,
    ScatteringEffectsConfig,
)
from nirs4all.synthesis.targets import (
    NonLinearTargetConfig,
    NonLinearTargetProcessor,
    TargetGenerator,
)

from .prior_adapter import PriorConfigRecord

SUPPORTED_TARGET_MAPPINGS = {
    "component_concentration_regression",
    "mixture_classification",
}
CLASS_SEPARATION_MAP = {"hard": 0.75, "moderate": 1.5, "easy": 2.5}
NONLINEARITY_MAP = {
    "none": None,
    "mild": NonLinearTargetConfig(
        nonlinear_interactions="polynomial",
        interaction_strength=0.25,
        polynomial_degree=2,
    ),
    "moderate": NonLinearTargetConfig(
        nonlinear_interactions="polynomial",
        interaction_strength=0.55,
        polynomial_degree=2,
        noise_heteroscedasticity=0.05,
    ),
}


@dataclass(frozen=True)
class SyntheticDatasetRun:
    """Executable bench dataset output for classic X/y workflows."""

    X: np.ndarray
    y: np.ndarray
    wavelengths: np.ndarray
    metadata: dict[str, Any]
    latent_metadata: dict[str, Any]
    prior_config: dict[str, Any]
    builder_config: dict[str, Any]
    validation_summary: dict[str, Any]


class PriorDatasetAdapterError(ValueError):
    """Raised when a canonical prior cannot be mapped without fallback."""

    def __init__(self, validation_summary: dict[str, Any]) -> None:
        self.validation_summary = validation_summary
        failures = validation_summary.get("failures", [])
        super().__init__("; ".join(str(failure) for failure in failures) or "invalid dataset run")


def prior_to_builder_config(
    record: PriorConfigRecord,
    *,
    n_samples: int | None = None,
    random_seed: int | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    """Convert a canonical A1 record into an explicit generation config."""
    target_prior = record.target_prior
    mapping = target_prior.get("executable_mapping")
    if mapping not in SUPPORTED_TARGET_MAPPINGS:
        _raise_mapping_error(
            record,
            "unsupported_target_mapping",
            f"Unsupported target mapping {mapping!r}",
        )

    task_n_samples = _as_int(record.task_prior.get("n_samples"), default=100)
    resolved_n_samples = int(n_samples if n_samples is not None else task_n_samples)
    if resolved_n_samples < 2:
        _raise_mapping_error(
            record,
            "invalid_n_samples",
            f"n_samples must be >= 2, got {resolved_n_samples}",
        )

    seed = random_seed if random_seed is not None else record.random_seed
    if seed is None:
        _raise_mapping_error(record, "missing_seed", "A2 requires an explicit seed")
    assert seed is not None

    target_config = _target_builder_config(record)
    if target_config["type"] == "classification" and resolved_n_samples < target_config["n_classes"]:
        _raise_mapping_error(
            record,
            "invalid_n_samples",
            (
                "classification datasets need at least one sample per declared "
                f"class, got n_samples={resolved_n_samples}, "
                f"n_classes={target_config['n_classes']}"
            ),
        )
    nuisance_config = _nuisance_builder_config(record)
    domain_config = get_domain_config(record.domain_key)
    wavelength_start, wavelength_end = record.wavelength_policy["effective_range_nm"]
    wavelength_step = _effective_wavelength_step(record)
    wavelength_grid = _bounded_wavelength_grid(
        float(wavelength_start),
        float(wavelength_end),
        wavelength_step,
    )

    return {
        "adapter_version": "A2",
        "n_samples": resolved_n_samples,
        "random_state": int(seed),
        "name": f"a2_{record.domain_key}_{record.instrument_key}_{record.measurement_mode}",
        "domain": {
            "key": record.domain_key,
            "category": domain_config.category.value,
            "product_key": record.product_key,
            "aggregate_key": record.aggregate_key,
            "complexity": domain_config.complexity,
        },
        "features": {
            "wavelength_range": [float(wavelength_start), float(wavelength_end)],
            "wavelength_step": wavelength_step,
            "wavelength_grid": wavelength_grid.tolist(),
            "complexity": _complexity(domain_config.complexity),
            "components": list(record.component_keys),
            "instrument": record.instrument_key,
            "measurement_mode": record.measurement_mode,
        },
        "target": target_config,
        "nuisance": nuisance_config,
        "partition": {
            "train_ratio": float(train_ratio),
            "shuffle": True,
            "stratify": target_config["type"] == "classification",
        },
        "concentration_transform": {
            "source": "domain.sample_concentrations",
            "row_normalized": True,
            "row_sum": 1.0,
            "reason": (
                "SyntheticNIRSGenerator.generate_from_concentrations expects "
                "mixture fractions whose rows sum to approximately 1.0."
            ),
        },
    }


def build_synthetic_dataset_run(
    record: PriorConfigRecord,
    *,
    n_samples: int | None = None,
    random_seed: int | None = None,
) -> SyntheticDatasetRun:
    """Generate and validate a finite synthetic dataset from one canonical record."""
    builder_config = prior_to_builder_config(
        record,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(builder_config["random_state"])
    concentrations = _sample_concentrations(record, rng, builder_config["n_samples"])
    generator = _create_generator(builder_config)

    nuisance = builder_config["nuisance"]
    temperatures = np.full(builder_config["n_samples"], nuisance["temperature_c"])
    X, generation_metadata = generator.generate_from_concentrations(
        concentrations,
        include_batch_effects=nuisance["batch_effects"]["enabled"],
        n_batches=nuisance["batch_effects"]["n_batches"],
        include_instrument_effects=True,
        include_environmental_effects=True,
        include_scattering_effects=True,
        include_edge_artifacts=True,
        temperatures=temperatures,
    )
    wavelengths = np.asarray(generator.wavelengths, dtype=float)
    y = _generate_target(record, builder_config, concentrations, X)

    latent_metadata = {
        "concentrations": concentrations,
        "component_keys": list(record.component_keys),
        "concentration_transform": builder_config["concentration_transform"],
        "batch_ids": generation_metadata.get("batch_ids"),
        "temperature_c": temperatures,
    }
    validation_summary = _validate_run_arrays(
        X=np.asarray(X),
        y=np.asarray(y),
        wavelengths=wavelengths,
        record=record,
        builder_config=builder_config,
        concentrations=concentrations,
    )

    if validation_summary["status"] != "passed":
        raise PriorDatasetAdapterError(validation_summary)

    metadata = _metadata(
        record=record,
        builder_config=builder_config,
        validation_summary=validation_summary,
        generation_metadata=generation_metadata,
    )
    return SyntheticDatasetRun(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y),
        wavelengths=wavelengths,
        metadata=metadata,
        latent_metadata=_to_builtin(latent_metadata),
        prior_config=record.to_dict(),
        builder_config=_to_builtin(builder_config),
        validation_summary=_to_builtin(validation_summary),
    )


def _create_generator(builder_config: dict[str, Any]) -> SyntheticNIRSGenerator:
    features = builder_config["features"]
    nuisance = builder_config["nuisance"]
    library = ComponentLibrary.from_predefined(
        features["components"],
        random_state=builder_config["random_state"],
    )
    return SyntheticNIRSGenerator(
        wavelength_start=features["wavelength_range"][0],
        wavelength_end=features["wavelength_range"][1],
        wavelength_step=features["wavelength_step"],
        wavelengths=np.asarray(features["wavelength_grid"], dtype=float),
        component_library=library,
        complexity=features["complexity"],
        instrument=features["instrument"],
        measurement_mode=features["measurement_mode"],
        environmental_config=EnvironmentalEffectsConfig(
            temperature=TemperatureConfig(
                sample_temperature=nuisance["temperature_c"],
                temperature_variation=0.0,
            ),
            enable_temperature=True,
            enable_moisture=False,
        ),
        scattering_effects_config=ScatteringEffectsConfig(
            particle_size=ParticleSizeConfig(
                distribution=ParticleSizeDistribution(
                    mean_size_um=nuisance["particle_size_um"],
                    std_size_um=max(1e-6, nuisance["particle_size_um"] * 0.05),
                    min_size_um=nuisance["particle_size_um"],
                    max_size_um=nuisance["particle_size_um"],
                    distribution="normal",
                ),
                reference_size_um=nuisance["particle_size_um"],
            ),
            enable_particle_size=True,
            enable_emsc=True,
        ),
        edge_artifacts_config=EdgeArtifactsConfig(
            enable_detector_rolloff=True,
            rolloff_severity=nuisance["edge_artifacts"]["rolloff_severity"],
        ),
        custom_params=nuisance["custom_params"],
        random_state=builder_config["random_state"],
    )


def _target_builder_config(record: PriorConfigRecord) -> dict[str, Any]:
    target = record.target_prior
    target_type = target["type"]
    if target_type == "regression":
        n_targets = _as_int(target.get("n_targets"), default=1)
        if n_targets > len(record.component_keys):
            _raise_mapping_error(
                record,
                "unsupported_target_mapping",
                f"n_targets {n_targets} exceeds component count {len(record.component_keys)}",
            )
        nonlinearity = str(target.get("nonlinearity", "none"))
        if nonlinearity not in NONLINEARITY_MAP:
            _raise_mapping_error(
                record,
                "unsupported_target_mapping",
                f"Unsupported regression nonlinearity {nonlinearity!r}",
            )
        return {
            "type": "regression",
            "mapping": target["executable_mapping"],
            "n_targets": n_targets,
            "component_indices": list(range(n_targets)),
            "component_keys": list(record.component_keys[:n_targets]),
            "distribution": "uniform",
            "range": _target_range(record, n_targets),
            "nonlinearity": nonlinearity,
        }

    if target_type == "classification":
        separation_key = str(target.get("separation", "moderate"))
        if separation_key not in CLASS_SEPARATION_MAP:
            _raise_mapping_error(
                record,
                "unsupported_target_mapping",
                f"Unsupported class separation {separation_key!r}",
            )
        return {
            "type": "classification",
            "mapping": target["executable_mapping"],
            "n_classes": _as_int(target.get("n_classes"), default=2),
            "separation": CLASS_SEPARATION_MAP[separation_key],
            "separation_key": separation_key,
            "separation_method": "composition_quantile",
        }

    _raise_mapping_error(record, "unsupported_target_mapping", f"Unsupported target type {target_type!r}")
    raise AssertionError("unreachable")


def _nuisance_builder_config(record: PriorConfigRecord) -> dict[str, Any]:
    nuisance = record.nuisance_prior
    instrument = get_instrument_archetype(record.instrument_key)
    noise_level = float(nuisance["noise_level"])
    particle_size = float(nuisance["particle_size_um"])
    temperature = float(nuisance["temperature_c"])
    return {
        "matrix_type": nuisance["matrix_type"],
        "temperature_c": temperature,
        "particle_size_um": particle_size,
        "noise_level": noise_level,
        "custom_params": {
            "noise_base": 0.0004 * noise_level,
            "noise_signal_dep": 0.0015 * noise_level,
            "baseline_amplitude": 0.004 + 0.001 * noise_level,
            "scatter_alpha_std": min(0.3, 0.015 + particle_size / 5000.0),
            "scatter_beta_std": 0.002 * noise_level,
            "artifact_prob": min(0.05, 0.005 * noise_level),
            "instrumental_fwhm": instrument.spectral_resolution,
        },
        "environment": {
            "temperature_enabled": True,
            "moisture_enabled": False,
        },
        "scatter": {
            "particle_size_enabled": True,
            "emsc_enabled": True,
        },
        "edge_artifacts": {
            "detector_rolloff_enabled": True,
            "rolloff_severity": min(1.0, 0.08 * noise_level),
        },
        "batch_effects": {
            "enabled": True,
            "n_batches": 3,
        },
    }


def _sample_concentrations(
    record: PriorConfigRecord,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    domain = get_domain_config(record.domain_key)
    concentrations = domain.sample_concentrations(rng, list(record.component_keys), n_samples)
    row_sums = concentrations.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    return np.asarray(concentrations / row_sums)


def _generate_target(
    record: PriorConfigRecord,
    builder_config: dict[str, Any],
    concentrations: np.ndarray,
    spectra: np.ndarray,
) -> np.ndarray:
    target = builder_config["target"]
    generator = TargetGenerator(random_state=builder_config["random_state"])
    if target["type"] == "classification":
        return _quantile_classification_target(
            concentrations,
            n_classes=target["n_classes"],
            separation=target["separation"],
            random_state=builder_config["random_state"],
        )

    result = generator.regression(
        builder_config["n_samples"],
        concentrations,
        component=target["component_indices"],
        range=tuple(target["range"]),
        correlation=1.0,
        noise=0.0,
    )
    y = np.asarray(result)
    nonlinear_config = NONLINEARITY_MAP[target["nonlinearity"]]
    if nonlinear_config is not None:
        y = NonLinearTargetProcessor(
            nonlinear_config,
            random_state=builder_config["random_state"],
        ).process(concentrations=concentrations, y_base=y, spectra=spectra)
        y = _scale_to_range(np.asarray(y), tuple(target["range"]))
    return y


def _validate_run_arrays(
    *,
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    record: PriorConfigRecord,
    builder_config: dict[str, Any],
    concentrations: np.ndarray,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    target = builder_config["target"]
    n_samples = builder_config["n_samples"]
    if X.shape != (n_samples, wavelengths.size):
        failures.append({"reason": "shape_mismatch", "field": "X", "message": str(X.shape)})
    if y.shape[0] != n_samples:
        failures.append({"reason": "shape_mismatch", "field": "y", "message": str(y.shape)})
    if concentrations.shape != (n_samples, len(record.component_keys)):
        failures.append({
            "reason": "shape_mismatch",
            "field": "concentrations",
            "message": str(concentrations.shape),
        })
    concentration_row_sums = np.sum(concentrations, axis=1)
    if not np.isfinite(concentrations).all():
        failures.append({
            "reason": "non_finite",
            "field": "concentrations",
            "message": "concentrations contain non-finite values",
        })
    if not np.allclose(concentration_row_sums, 1.0, rtol=1e-9, atol=1e-9):
        failures.append({
            "reason": "concentration_row_sum_mismatch",
            "field": "concentrations",
            "message": (
                "row-normalized concentrations must sum to 1.0; "
                f"observed range={(float(np.min(concentration_row_sums)), float(np.max(concentration_row_sums)))}"
            ),
        })
    if not np.isfinite(X).all():
        failures.append({"reason": "non_finite", "field": "X", "message": "spectra contain non-finite values"})
    if not np.isfinite(y).all():
        failures.append({"reason": "non_finite", "field": "y", "message": "target contains non-finite values"})
    if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
        failures.append({
            "reason": "invalid_wavelengths",
            "field": "wavelengths",
            "message": "wavelengths must be finite and strictly increasing",
        })
    expected_low, expected_high = builder_config["features"]["wavelength_range"]
    if wavelengths[0] < expected_low - 1e-9 or wavelengths[-1] > expected_high + 1e-9:
        failures.append({
            "reason": "wavelength_range_mismatch",
            "field": "wavelengths",
            "message": f"{(wavelengths[0], wavelengths[-1])} outside {(expected_low, expected_high)}",
        })
    if target["type"] == "classification":
        observed = set(np.unique(y).astype(int).tolist())
        expected = set(range(target["n_classes"]))
        if observed != expected:
            failures.append({
                "reason": "invalid_class_labels",
                "field": "y",
                "message": f"observed={sorted(observed)}, expected={sorted(expected)}",
            })
    else:
        target_min, target_max = target["range"]
        if float(np.min(y)) < target_min - 1e-9 or float(np.max(y)) > target_max + 1e-9:
            failures.append({
                "reason": "target_range_mismatch",
                "field": "y",
                "message": f"target outside {(target_min, target_max)}",
            })

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "unsupported_fields": [],
        "adapter_notes": [
            (
                "measurement_mode is passed to SyntheticNIRSGenerator and preserved "
                "in metadata; A2 contract checks do not validate mode-specific "
                "optical physics."
            )
        ],
        "checks": {
            "shape": not any(f["reason"] == "shape_mismatch" for f in failures),
            "finite": bool(np.isfinite(X).all() and np.isfinite(y).all()),
            "wavelengths_monotonic": bool(np.all(np.diff(wavelengths) > 0)),
            "target_contract": not any(
                f["reason"] in {"invalid_class_labels", "target_range_mismatch"}
                for f in failures
            ),
            "concentrations_row_normalized": not any(
                f["reason"] == "concentration_row_sum_mismatch" for f in failures
            ),
            "seed": builder_config["random_state"],
        },
        "summary": {
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "wavelength_range_nm": [float(wavelengths[0]), float(wavelengths[-1])],
            "X_min": float(np.min(X)),
            "X_max": float(np.max(X)),
            "y_min": float(np.min(y)),
            "y_max": float(np.max(y)),
            "concentration_row_sum_min": float(np.min(concentration_row_sums)),
            "concentration_row_sum_max": float(np.max(concentration_row_sums)),
        },
    }


def _metadata(
    *,
    record: PriorConfigRecord,
    builder_config: dict[str, Any],
    validation_summary: dict[str, Any],
    generation_metadata: dict[str, Any],
) -> dict[str, Any]:
    provenance = {
        "source_prior_config": record.source_prior_config,
        "_raw_prior_config": record.source_prior_config.get("_raw_prior_config"),
        "_canonical_repairs": record.source_prior_config.get("_canonical_repairs"),
    }
    return cast("dict[str, Any]", _to_builtin({
        "domain": builder_config["domain"],
        "instrument": {
            "key": record.instrument_key,
            "category": record.source_prior_config.get("instrument_category"),
        },
        "mode": record.measurement_mode,
        "target": builder_config["target"],
        "nuisance": builder_config["nuisance"],
        "prior_config": record.to_dict(),
        "builder_config": builder_config,
        "validation_summary": validation_summary,
        "provenance_a1": provenance,
        "generation_metadata": generation_metadata,
    }))


def _effective_wavelength_step(record: PriorConfigRecord) -> float:
    resolution = record.wavelength_policy.get("spectral_resolution_nm")
    if resolution is None:
        return 2.0
    return float(max(2.0, round(float(resolution), 6)))


def _bounded_wavelength_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError(f"wavelength step must be positive, got {step}")
    grid = np.arange(start, end + step * 0.25, step, dtype=float)
    grid = grid[grid <= end + 1e-9]
    if grid.size == 0 or not np.isclose(grid[0], start):
        grid = np.insert(grid, 0, start)
    if grid[-1] > end + 1e-9:
        grid = grid[:-1]
    return grid


def _target_range(record: PriorConfigRecord, n_targets: int) -> list[float]:
    selected = record.component_keys[:n_targets]
    lower = min(float(record.concentration_prior[key]["min_value"]) for key in selected)
    upper = max(float(record.concentration_prior[key]["max_value"]) for key in selected)
    if not lower < upper:
        upper = lower + 1.0
    return [lower, upper]


def _scale_to_range(y: np.ndarray, target_range: tuple[float, float]) -> np.ndarray:
    target_min, target_max = target_range
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if y_max <= y_min:
        return np.full_like(y, (target_min + target_max) / 2)
    return (y - y_min) / (y_max - y_min) * (target_max - target_min) + target_min


def _quantile_classification_target(
    concentrations: np.ndarray,
    *,
    n_classes: int,
    separation: float,
    random_state: int,
) -> np.ndarray:
    """Create declared classes from mixture composition without class dropout."""
    if concentrations.shape[0] < n_classes:
        raise ValueError(
            f"n_samples ({concentrations.shape[0]}) must be >= n_classes ({n_classes})"
        )
    component_weights = np.linspace(separation, 1.0, concentrations.shape[1])
    scores = concentrations @ component_weights
    rng = np.random.default_rng(random_state)
    scores = scores + rng.normal(0.0, 1e-12, size=scores.shape)
    order = np.argsort(scores, kind="mergesort")
    labels = np.empty(concentrations.shape[0], dtype=np.int32)
    for class_id, indices in enumerate(np.array_split(order, n_classes)):
        labels[indices] = class_id
    return labels


def _complexity(value: str) -> Literal["simple", "realistic", "complex"]:
    if value in {"simple", "realistic", "complex"}:
        return cast("Literal['simple', 'realistic', 'complex']", value)
    return "realistic"


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _raise_mapping_error(record: PriorConfigRecord, reason: str, message: str) -> NoReturn:
    raise PriorDatasetAdapterError({
        "status": "failed",
        "record": {
            "domain_key": record.domain_key,
            "instrument_key": record.instrument_key,
            "measurement_mode": record.measurement_mode,
            "target_prior": record.target_prior,
        },
        "failures": [{"reason": reason, "field": "prior_config", "message": message}],
    })


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if hasattr(value, "__dataclass_fields__"):
        return _to_builtin(asdict(value))
    return value
