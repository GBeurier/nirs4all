"""Phase A3 adapter for ``RealDataFitter.to_full_config()`` outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NoReturn, cast

import numpy as np

from nirs4all.synthesis import (
    ComponentLibrary,
    NIRBand,
    SpectralComponent,
    SyntheticNIRSGenerator,
)
from nirs4all.synthesis._constants import COMPLEXITY_PARAMS
from nirs4all.synthesis.environmental import EnvironmentalEffectsConfig, MoistureConfig, TemperatureConfig
from nirs4all.synthesis.instruments import EdgeArtifactsConfig, get_instrument_archetype
from nirs4all.synthesis.scattering import (
    EMSCConfig,
    ParticleSizeConfig,
    ParticleSizeDistribution,
    ScatteringEffectsConfig,
)

from .builder_adapter import (
    PriorDatasetAdapterError,
    SyntheticDatasetRun,
    _generate_target,
    _metadata,
    _sample_concentrations,
    _to_builtin,
    _validate_run_arrays,
    prior_to_builder_config,
)
from .prior_adapter import PriorCanonicalizationError, PriorConfigRecord, canonicalize_prior_config

SUPPORTED_FITTED_FIELDS = {
    "wavelength_start",
    "wavelength_end",
    "wavelength_step",
    "complexity",
    "noise_base",
    "noise_signal_dep",
    "scatter_alpha_std",
    "scatter_beta_std",
    "path_length_std",
    "baseline_amplitude",
    "tilt_std",
    "global_slope_mean",
    "global_slope_std",
    "instrument",
    "measurement_mode",
    "domain",
    "components",
    "temperature_config",
    "moisture_config",
    "emsc_config",
}

PARTIALLY_SUPPORTED_FITTED_FIELDS = {
    "particle_size_config",
    "edge_artifacts_config",
    "boundary_components_config",
    "fitted_residual_effects_config",
}

SUPPORTED_TEMPERATURE_CONFIG_KEYS = {
    "sample_temperature",
    "temperature_variation",
    "reference_temperature",
    "enable_shift",
    "enable_intensity",
    "enable_broadening",
    "region_specific",
}

SUPPORTED_MOISTURE_CONFIG_KEYS = {
    "water_activity",
    "moisture_content",
    "free_water_fraction",
    "bound_water_shift",
    "reference_aw",
}

SUPPORTED_PARTICLE_SIZE_CONFIG_KEYS = {
    "mean_size_um",
    "std_size_um",
    "reference_size_um",
    "size_effect_strength",
    "wavelength_exponent",
    "include_path_length_effect",
    "path_length_sensitivity",
}

SUPPORTED_EMSC_CONFIG_KEYS = {
    "multiplicative_scatter_std",
    "additive_scatter_std",
    "polynomial_order",
    "include_wavelength_terms",
    "wavelength_coef_std",
}

SUPPORTED_BOUNDARY_COMPONENT_CONFIG_KEYS = {
    "components",
}

CALIBRATED_GENERATOR_CONTROL_FIELDS = {
    "path_length_std",
    "baseline_amplitude",
    "scatter_alpha_std",
    "scatter_beta_std",
    "tilt_std",
    "global_slope_mean",
    "global_slope_std",
}

DIRECT_GENERATOR_CONTROL_FIELDS = {
    "noise_base",
    "noise_signal_dep",
}


@dataclass(frozen=True)
class UnsupportedFittedField:
    """One fitted config field that the A3 bench adapter cannot execute yet."""

    field: str
    reason: str
    value_summary: Any

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


class FittedConfigAdapterError(ValueError):
    """Raised when a fitted config cannot be converted without ambiguity."""

    def __init__(self, validation_summary: dict[str, Any]) -> None:
        self.validation_summary = validation_summary
        failures = validation_summary.get("failures", [])
        super().__init__("; ".join(str(failure) for failure in failures) or "invalid fitted config")


def fitted_config_to_prior_record(
    fitted_config: dict[str, Any],
    *,
    n_samples: int,
    random_seed: int,
    target_config: dict[str, Any] | None = None,
    instrument_override: str | None = None,
    instrument_override_reason: str | None = None,
    component_override: list[str] | tuple[str, ...] | None = None,
    component_override_reason: str | None = None,
) -> tuple[PriorConfigRecord, list[dict[str, Any]], list[str]]:
    """Convert a full fitted config into the canonical A1 record required by A2.

    ``RealDataFitter.to_full_config()`` has no target or matrix contract, so this
    adapter uses an explicit smoke target and a solid-matrix bench assumption.
    All fitted fields that are not mapped into the A2 executable path are
    returned in ``unsupported_fields``.
    """
    unsupported_fields = _unsupported_fields(fitted_config)
    assumptions = [
        "A3 smoke uses a solid matrix because to_full_config() does not expose matrix_type.",
        "A3 smoke uses a component-concentration regression target because fitted spectra do not carry y.",
        (
            "A1 nuisance particle_size uses a range-safe placeholder during "
            "canonicalization; fitted particle_size_config.mean_size_um is "
            "calibrated against the assumed matrix range later in the A3 builder_config."
        ),
    ]
    fitted_components = _components(fitted_config)
    components = [str(component) for component in component_override] if component_override else fitted_components
    if not components:
        _raise_adapter_error(
            failures=[{
                "reason": "missing_fitted_components",
                "field": "components",
                "message": "Fitted config must expose at least one executable component.",
            }],
            unsupported_fields=unsupported_fields,
            assumptions=assumptions,
        )

    fitted_instrument_key = str(fitted_config.get("instrument", ""))
    instrument_key = str(instrument_override or fitted_instrument_key)
    instrument_resolution = {
        "fitted_instrument": fitted_instrument_key,
        "effective_instrument": instrument_key,
        "source": "instrument_override" if instrument_override else "fitted_config.instrument",
        "reason": instrument_override_reason,
        "fitted_only_executable_contract": instrument_override is None,
    }
    if instrument_override:
        assumptions.append(
            "A3 smoke uses an explicit instrument override because source provenance is available "
            f"and RealDataFitter inferred {fitted_instrument_key!r}."
        )
    component_resolution = {
        "fitted_components": fitted_components,
        "effective_components": components,
        "source": "source_provenance_component_override" if component_override else "fitted_config.components",
        "reason": component_override_reason,
        "fitted_only_executable_contract": component_override is None,
    }
    if component_override:
        assumptions.append(
            "A3 diagnostic uses source-provenance component override; this is an oracle "
            "ablation and not a fitted-only executable contract."
        )
    try:
        instrument_category = get_instrument_archetype(instrument_key).category.value
    except Exception:
        instrument_category = ""

    resolved_target = target_config or {
        "type": "regression",
        "n_targets": 1,
        "nonlinearity": "none",
    }
    source = {
        "domain": fitted_config.get("domain"),
        "domain_category": "real_fit",
        "instrument": instrument_key,
        "instrument_category": instrument_category,
        "wavelength_range": (
            _required_float(fitted_config, "wavelength_start"),
            _required_float(fitted_config, "wavelength_end"),
        ),
        "spectral_resolution": _required_float(fitted_config, "wavelength_step"),
        "measurement_mode": fitted_config.get("measurement_mode"),
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": components,
        "n_samples": int(n_samples),
        "target_config": resolved_target,
        "random_state": int(random_seed),
        "_raw_fitted_config": _to_builtin(fitted_config),
        "_a3_adapter_assumptions": assumptions,
        "_a3_adapter_instrument_resolution": instrument_resolution,
        "_a3_adapter_component_resolution": component_resolution,
    }
    try:
        return canonicalize_prior_config(source), unsupported_fields, assumptions
    except PriorCanonicalizationError as exc:
        _raise_adapter_error(
            failures=[
                {
                    "reason": issue.reason,
                    "field": issue.field,
                    "message": issue.message,
                }
                for issue in exc.issues
            ],
            unsupported_fields=unsupported_fields,
            assumptions=assumptions,
        )


def build_dataset_run_from_fitted_config(
    fitted_config: dict[str, Any],
    *,
    n_samples: int = 64,
    random_seed: int = 0,
    target_config: dict[str, Any] | None = None,
    fail_on_unsupported: bool = False,
    instrument_override: str | None = None,
    instrument_override_reason: str | None = None,
    component_override: list[str] | tuple[str, ...] | None = None,
    component_override_reason: str | None = None,
) -> SyntheticDatasetRun:
    """Generate an A2 ``SyntheticDatasetRun`` from a fitted full config.

    The generation path intentionally reuses A2 concentration sampling, target
    generation, generator construction, validation, and run dataclass. Unsupported
    fitted fields are always surfaced in the returned validation summary and can
    be promoted to a hard failure with ``fail_on_unsupported=True``.
    """
    record, unsupported_fields, assumptions = fitted_config_to_prior_record(
        fitted_config,
        n_samples=n_samples,
        random_seed=random_seed,
        target_config=target_config,
        instrument_override=instrument_override,
        instrument_override_reason=instrument_override_reason,
        component_override=component_override,
        component_override_reason=component_override_reason,
    )
    if fail_on_unsupported and unsupported_fields:
        _raise_adapter_error(
            failures=[{
                "reason": "unsupported_fitted_fields",
                "field": "fitted_config",
                "message": "Unsupported fitted fields were present and fail_on_unsupported=True.",
            }],
            unsupported_fields=unsupported_fields,
            assumptions=assumptions,
        )

    builder_config = prior_to_builder_config(record, n_samples=n_samples, random_seed=random_seed)
    builder_config["_a3_instrument_resolution"] = record.source_prior_config.get(
        "_a3_adapter_instrument_resolution",
        {},
    )
    builder_config["_a3_component_resolution"] = record.source_prior_config.get(
        "_a3_adapter_component_resolution",
        {},
    )
    _apply_fitted_overrides(builder_config, fitted_config)
    assumptions = [
        *assumptions,
        *builder_config["fitted_config_mapping"]["effect_reconstruction"]["assumptions"],
    ]

    rng = np.random.default_rng(builder_config["random_state"])
    concentrations = _sample_concentrations(record, rng, builder_config["n_samples"])
    generator = _create_fitted_generator(builder_config, fitted_config)
    nuisance = builder_config["nuisance"]
    temperatures = np.full(builder_config["n_samples"], nuisance["temperature_c"])
    generation_concentrations = _generation_concentrations_with_boundary_components(
        concentrations,
        builder_config["nuisance"],
    )
    X, generation_metadata = generator.generate_from_concentrations(
        generation_concentrations,
        include_batch_effects=nuisance["batch_effects"]["enabled"],
        n_batches=nuisance["batch_effects"]["n_batches"],
        include_instrument_effects=True,
        include_environmental_effects=True,
        include_scattering_effects=True,
        include_edge_artifacts=True,
        temperatures=temperatures,
    )
    wavelengths = np.asarray(generator.wavelengths, dtype=float)
    y = _generate_target(record, builder_config, concentrations, np.asarray(X))
    X, residual_effects_metadata = _apply_fitted_residual_effects(
        np.asarray(X, dtype=float),
        wavelengths=wavelengths,
        fitted_config=fitted_config,
        rng=rng,
    )
    generation_metadata["a3_fitted_residual_effects"] = residual_effects_metadata
    if generation_concentrations.shape[1] != concentrations.shape[1]:
        generation_metadata["a3_fitted_boundary_components"] = (
            builder_config["nuisance"]["fitted_effect_overrides"]
            .get("boundary_components_config", {})
            .get("components", [])
        )
        generation_metadata["a3_source_component_count"] = int(concentrations.shape[1])
    validation_summary = _validate_run_arrays(
        X=np.asarray(X),
        y=np.asarray(y),
        wavelengths=wavelengths,
        record=record,
        builder_config=builder_config,
        concentrations=concentrations,
    )
    validation_summary = _with_a3_validation(
        validation_summary,
        unsupported_fields=unsupported_fields,
        assumptions=assumptions,
    )
    if validation_summary["status"] != "passed":
        raise PriorDatasetAdapterError(validation_summary)

    latent_metadata = {
        "concentrations": concentrations,
        "component_keys": list(record.component_keys),
        "concentration_transform": builder_config["concentration_transform"],
        "batch_ids": generation_metadata.get("batch_ids"),
        "temperature_c": temperatures,
    }
    metadata = _metadata(
        record=record,
        builder_config=builder_config,
        validation_summary=validation_summary,
        generation_metadata=generation_metadata,
    )
    metadata["real_fit_adapter"] = {
        "adapter_version": "A3",
        "source": "RealDataFitter.to_full_config",
        "unsupported_fields": unsupported_fields,
        "assumptions": assumptions,
        "mapped_fields": sorted(SUPPORTED_FITTED_FIELDS),
        "partially_supported_fields": sorted(PARTIALLY_SUPPORTED_FITTED_FIELDS),
        "instrument_resolution": builder_config["fitted_config_mapping"]["instrument_resolution"],
        "component_resolution": builder_config["fitted_config_mapping"]["component_resolution"],
        "effect_reconstruction": builder_config["fitted_config_mapping"]["effect_reconstruction"],
    }
    metadata["validation_summary"] = validation_summary
    return SyntheticDatasetRun(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y),
        wavelengths=wavelengths,
        metadata=_to_builtin(metadata),
        latent_metadata=_to_builtin(latent_metadata),
        prior_config=record.to_dict(),
        builder_config=_to_builtin(builder_config),
        validation_summary=_to_builtin(validation_summary),
    )


def _apply_fitted_overrides(builder_config: dict[str, Any], fitted_config: dict[str, Any]) -> None:
    builder_config["adapter_version"] = "A3-real-fit-via-A2"
    features = builder_config["features"]
    effective_instrument = str(features["instrument"])
    fitted_instrument = str(fitted_config["instrument"])
    effective_components = [str(component) for component in features["components"]]
    fitted_components = _components(fitted_config)
    start = _required_float(fitted_config, "wavelength_start")
    end = _required_float(fitted_config, "wavelength_end")
    step = _required_float(fitted_config, "wavelength_step")
    features["wavelength_range"] = [start, end]
    features["wavelength_step"] = step
    features["wavelength_grid"] = _bounded_wavelength_grid(start, end, step).tolist()
    fitted_complexity = str(fitted_config["complexity"])
    domain_complexity = str(builder_config["domain"].get("complexity", "realistic"))
    effective_complexity = _effective_complexity(fitted_complexity, domain_complexity)
    features["complexity"] = effective_complexity
    features["instrument"] = effective_instrument
    features["fitted_instrument"] = fitted_instrument
    features["measurement_mode"] = str(fitted_config["measurement_mode"])
    features["components"] = effective_components
    features["fitted_components"] = fitted_components
    builder_config["name"] = (
        "a3_real_fit_"
        f"{builder_config['domain']['key']}_"
        f"{effective_instrument}_"
        f"{builder_config['features']['measurement_mode']}"
    )

    effect_reconstruction = _fitted_effect_reconstruction(
        fitted_config,
        effective_complexity=effective_complexity,
        fitted_complexity=fitted_complexity,
        domain_complexity=domain_complexity,
        matrix_type=str(builder_config["nuisance"]["matrix_type"]),
        prior_particle_size_um=float(builder_config["nuisance"]["particle_size_um"]),
        prior_rolloff_severity=float(builder_config["nuisance"]["edge_artifacts"]["rolloff_severity"]),
    )
    custom_params = builder_config["nuisance"]["custom_params"]
    for field, mapping in effect_reconstruction["fields"].items():
        if field in CALIBRATED_GENERATOR_CONTROL_FIELDS | DIRECT_GENERATOR_CONTROL_FIELDS:
            custom_params[field] = float(mapping["effective"])

    nested_overrides = effect_reconstruction["nested_overrides"]
    builder_config["nuisance"]["fitted_effect_overrides"] = nested_overrides

    particle_size = _particle_size_um(
        fitted_config,
        override=nested_overrides.get("particle_size_config"),
    )
    builder_config["nuisance"]["particle_size_um"] = particle_size
    rolloff = _detector_rolloff_severity(
        fitted_config,
        override=nested_overrides.get("edge_artifacts_config"),
    )
    if rolloff is not None:
        builder_config["nuisance"]["edge_artifacts"]["rolloff_severity"] = rolloff
    builder_config["fitted_config_mapping"] = {
        "source": "RealDataFitter.to_full_config",
        "mapped_fields": sorted(SUPPORTED_FITTED_FIELDS),
        "partially_supported_fields": sorted(PARTIALLY_SUPPORTED_FITTED_FIELDS),
        "effect_reconstruction": _to_builtin(effect_reconstruction),
        "instrument_resolution": {
            "fitted_instrument": fitted_instrument,
            "effective_instrument": effective_instrument,
            "source": (
                "instrument_override"
                if effective_instrument != fitted_instrument
                else "fitted_config.instrument"
            ),
            "reason": builder_config.get("_a3_instrument_resolution", {}).get("reason"),
            "fitted_only_executable_contract": builder_config.get(
                "_a3_instrument_resolution",
                {},
            ).get("fitted_only_executable_contract", True),
        },
        "component_resolution": {
            "fitted_components": fitted_components,
            "effective_components": effective_components,
            "source": builder_config.get("_a3_component_resolution", {}).get(
                "source",
                "fitted_config.components",
            ),
            "reason": builder_config.get("_a3_component_resolution", {}).get("reason"),
            "fitted_only_executable_contract": builder_config.get(
                "_a3_component_resolution",
                {},
            ).get("fitted_only_executable_contract", True),
        },
    }


def _effective_complexity(fitted_complexity: str, domain_complexity: str) -> str:
    if fitted_complexity == "complex" and domain_complexity in COMPLEXITY_PARAMS:
        return domain_complexity
    if fitted_complexity in COMPLEXITY_PARAMS:
        return fitted_complexity
    if domain_complexity in COMPLEXITY_PARAMS:
        return domain_complexity
    return "realistic"


def _fitted_effect_reconstruction(
    fitted_config: dict[str, Any],
    *,
    effective_complexity: str,
    fitted_complexity: str,
    domain_complexity: str,
    matrix_type: str,
    prior_particle_size_um: float,
    prior_rolloff_severity: float,
) -> dict[str, Any]:
    assumptions = [
        (
            "A3 fitted-only maps RealDataFitter nuisance/effect estimates through "
            "calibrated generator controls because to_full_config() estimates "
            "final-spectrum variability, not raw operator parameters."
        )
    ]
    fields: dict[str, dict[str, Any]] = {}
    for field in sorted(DIRECT_GENERATOR_CONTROL_FIELDS):
        if field in fitted_config:
            raw = float(fitted_config[field])
            fields[field] = {
                "raw": raw,
                "effective": raw,
                "mapping": "direct_fitted_generator_control",
                "reason": "Noise estimates are already generator-scale fitted controls.",
            }

    for field in sorted(CALIBRATED_GENERATOR_CONTROL_FIELDS):
        if field not in fitted_config:
            continue
        raw = float(fitted_config[field])
        default = _complexity_default(effective_complexity, field)
        effective = _cap_to_generator_default(raw, default, signed=field == "global_slope_mean")
        fields[field] = {
            "raw": raw,
            "effective": effective,
            "mapping": "fitted_value_capped_to_effective_complexity_default",
            "anchor_complexity": effective_complexity,
            "anchor_value": default,
            "reason": (
                "Fitter-derived smooth nuisance values can double-count already "
                "present path/scatter/baseline variation when injected directly."
            ),
        }

    nested_overrides = {
        "temperature_config": {},
        "moisture_config": {},
        "particle_size_config": _calibrated_particle_size_config(
            fitted_config,
            matrix_type=matrix_type,
            prior_particle_size_um=prior_particle_size_um,
        ),
        "emsc_config": _calibrated_emsc_config(fitted_config, effective_complexity),
        "edge_artifacts_config": _calibrated_edge_artifacts_config(
            fitted_config,
            prior_rolloff_severity=prior_rolloff_severity,
        ),
        "boundary_components_config": _calibrated_boundary_components_config(fitted_config),
    }
    if fitted_complexity != effective_complexity:
        fields["complexity"] = {
            "raw": fitted_complexity,
            "effective": effective_complexity,
            "mapping": "domain_anchored_complexity",
            "anchor_complexity": domain_complexity,
            "reason": (
                "The fitted complexity classifier is driven by cumulative final-spectrum "
                "variation; the executable generator controls are anchored to the "
                "inferred domain complexity for fitted-only reconstruction."
            ),
        }
    fields["particle_size_config"] = {
        "raw": _to_builtin(fitted_config.get("particle_size_config", {})),
        "effective": nested_overrides["particle_size_config"],
        "mapping": "range_safe_particle_size_reconstruction",
        "anchor_matrix_type": matrix_type,
        "valid_range_um": list(_particle_size_bounds(matrix_type)),
        "reason": (
            "Fitted particle-size estimates are inferred from aggregate curvature; "
            "the bench clips them to the assumed matrix range instead of "
            "reapplying out-of-contract fine-powder curvature directly."
        ),
    }
    fields["emsc_config"] = {
        "raw": _to_builtin(fitted_config.get("emsc_config", {})),
        "effective": nested_overrides["emsc_config"],
        "mapping": "fitted_emsc_capped_to_effective_complexity_defaults",
        "reason": "EMSC scatter controls are capped before execution to avoid fitted-effect double counting.",
    }
    fields["edge_artifacts_config"] = {
        "raw": _to_builtin(fitted_config.get("edge_artifacts_config", {})),
        "effective": nested_overrides["edge_artifacts_config"],
        "mapping": "range_safe_executable_edge_artifacts",
        "anchor_rolloff_cap": prior_rolloff_severity,
        "valid_ranges": {
            "rolloff_severity": [0.0, prior_rolloff_severity],
            "stray_fraction": [0.0, 0.02],
            "curvature_severity": [0.0, 1.0],
            "truncated_peak_amplitude": [0.0, 0.1],
        },
        "reason": (
            "Fitted edge artifact controls are mapped to the existing generator "
            "operators and clipped to documented physical/operator ranges."
        ),
    }
    fields["boundary_components_config"] = {
        "raw": _to_builtin(fitted_config.get("boundary_components_config", {})),
        "effective": nested_overrides["boundary_components_config"],
        "mapping": "explicit_fitted_boundary_components_added_to_component_library",
        "reason": (
            "Only explicitly fitted boundary_components_config.components are "
            "executed; n_components is not used to infer or impute components."
        ),
    }
    fields["temperature_config"] = {
        "raw": _to_builtin(fitted_config.get("temperature_config", {})),
        "effective": nested_overrides["temperature_config"],
        "mapping": "disabled_without_validated_fitted_environment_scale",
        "reason": (
            "Fitted environmental effects are inferred from final spectra; the "
            "A3 fitted-only gate reports them but does not reapply unvalidated "
            "temperature effects as generator controls."
        ),
    }
    fields["moisture_config"] = {
        "raw": _to_builtin(fitted_config.get("moisture_config", {})),
        "effective": nested_overrides["moisture_config"],
        "mapping": "disabled_without_validated_fitted_environment_scale",
        "reason": (
            "Fitted moisture effects are final-spectrum indicators and can "
            "double-count component water/moisture structure when executed directly."
        ),
    }
    residual_effects_config = fitted_config.get("fitted_residual_effects_config")
    fields["fitted_residual_effects_config"] = {
        "raw": _summarize_residual_effects_config(residual_effects_config),
        "effective": _summarize_residual_effects_config(residual_effects_config),
        "mapping": "observed_residual_distribution_additive_reconstruction",
        "reason": (
            "Residual effects are reconstructed only from observed fitted spectra and "
            "executed by sampling fitted baseline/peak coefficient distributions."
        ),
    }
    assumptions.append(
        "A3 reports raw and effective fitted nuisance/effect controls in builder_config and metadata."
    )
    if _residual_effects_enabled(residual_effects_config):
        assumptions.append(
            "A3 applies fitted_residual_effects_config additively from observed-spectrum coefficient distributions; no observed residual row is replayed."
        )
    return {
        "mode": "calibrated_fitted_nuisance_effect_controls",
        "assumptions": assumptions,
        "fields": fields,
        "nested_overrides": nested_overrides,
    }


def _complexity_default(complexity: str, field: str) -> float:
    params = COMPLEXITY_PARAMS.get(complexity, COMPLEXITY_PARAMS["realistic"])
    return float(params[field])


def _cap_to_generator_default(raw: float, default: float, *, signed: bool) -> float:
    if not np.isfinite(raw):
        return default
    if signed:
        if default == 0.0:
            return 0.0
        return float(np.clip(raw, -abs(default), abs(default)))
    return float(min(max(raw, 0.0), default))


def _calibrated_particle_size_config(
    fitted_config: dict[str, Any],
    *,
    matrix_type: str,
    prior_particle_size_um: float,
) -> dict[str, Any]:
    particle_config = fitted_config.get("particle_size_config")
    particle_values = particle_config if isinstance(particle_config, dict) else {}
    lower, upper = _particle_size_bounds(matrix_type)
    raw_mean = _finite_float(particle_values.get("mean_size_um"), default=prior_particle_size_um)
    mean_size = float(np.clip(raw_mean, lower, upper))
    raw_std = _finite_float(particle_values.get("std_size_um"), default=mean_size * 0.05)
    std_size = float(np.clip(raw_std, 1e-6, mean_size * 0.05))
    return {
        "mean_size_um": mean_size,
        "std_size_um": std_size,
        "reference_size_um": mean_size,
        "size_effect_strength": float(particle_values.get("size_effect_strength", 1.0)),
        "wavelength_exponent": float(particle_values.get("wavelength_exponent", 1.5)),
        "include_path_length_effect": bool(particle_values.get("include_path_length_effect", True)),
        "path_length_sensitivity": float(particle_values.get("path_length_sensitivity", 0.5)),
    }


def _calibrated_emsc_config(
    fitted_config: dict[str, Any],
    effective_complexity: str,
) -> dict[str, Any]:
    emsc_config = fitted_config.get("emsc_config")
    emsc_values = emsc_config if isinstance(emsc_config, dict) else {}
    return {
        "multiplicative_scatter_std": _cap_to_generator_default(
            float(emsc_values.get("multiplicative_scatter_std", 0.0)),
            _complexity_default(effective_complexity, "scatter_alpha_std"),
            signed=False,
        ),
        "additive_scatter_std": _cap_to_generator_default(
            float(emsc_values.get("additive_scatter_std", 0.0)),
            _complexity_default(effective_complexity, "scatter_beta_std"),
            signed=False,
        ),
        "polynomial_order": int(emsc_values.get("polynomial_order", 2)),
        "include_wavelength_terms": bool(emsc_values.get("include_wavelength_terms", True)),
        "wavelength_coef_std": _cap_to_generator_default(
            _finite_float(emsc_values.get("wavelength_coef_std"), default=0.02),
            0.02,
            signed=False,
        ),
    }


def _calibrated_edge_artifacts_config(
    fitted_config: dict[str, Any],
    *,
    prior_rolloff_severity: float,
) -> dict[str, Any]:
    edge_config = fitted_config.get("edge_artifacts_config")
    edge_values = edge_config if isinstance(edge_config, dict) else {}
    rolloff = edge_values.get("detector_rolloff")
    rolloff_values = rolloff if isinstance(rolloff, dict) else {}
    rolloff_enabled = bool(rolloff_values.get("enabled", False)) if rolloff_values else False
    severity = float(np.clip(
        _finite_float(rolloff_values.get("severity"), default=0.0),
        0.0,
        prior_rolloff_severity,
    ))
    stray = edge_values.get("stray_light")
    stray_values = stray if isinstance(stray, dict) else {}
    stray_fraction = float(np.clip(
        _finite_float(stray_values.get("stray_fraction"), default=0.0),
        0.0,
        0.02,
    ))
    curvature = edge_values.get("edge_curvature")
    curvature_values = curvature if isinstance(curvature, dict) else {}
    left_curvature = float(np.clip(
        _finite_float(curvature_values.get("left_severity"), default=0.0),
        0.0,
        1.0,
    ))
    right_curvature = float(np.clip(
        _finite_float(curvature_values.get("right_severity"), default=0.0),
        0.0,
        1.0,
    ))
    truncated = edge_values.get("truncated_peaks")
    truncated_values = truncated if isinstance(truncated, dict) else {}
    left_peak = float(np.clip(
        _finite_float(truncated_values.get("left_amplitude"), default=0.0),
        0.0,
        0.1,
    ))
    right_peak = float(np.clip(
        _finite_float(truncated_values.get("right_amplitude"), default=0.0),
        0.0,
        0.1,
    ))
    return {
        "detector_rolloff": {
            "enabled": rolloff_enabled and severity > 0.0,
            "detector_model": str(rolloff_values.get("detector_model", "generic_nir")),
            "severity": severity,
        },
        "stray_light": {
            "enabled": bool(stray_values.get("enabled", False)) and stray_fraction > 0.0,
            "stray_fraction": stray_fraction,
            "wavelength_dependent": bool(stray_values.get("wavelength_dependent", True)),
        },
        "edge_curvature": {
            "enabled": bool(curvature_values.get("enabled", False))
            and (left_curvature > 0.0 or right_curvature > 0.0),
            "curvature_type": _edge_curvature_type(curvature_values.get("curvature_type")),
            "left_severity": left_curvature,
            "right_severity": right_curvature,
        },
        "truncated_peaks": {
            "enabled": bool(truncated_values.get("enabled", False))
            and (left_peak > 0.0 or right_peak > 0.0),
            "left_amplitude": left_peak,
            "right_amplitude": right_peak,
        },
    }


def _calibrated_boundary_components_config(fitted_config: dict[str, Any]) -> dict[str, Any]:
    boundary_config = fitted_config.get("boundary_components_config")
    boundary_values = boundary_config if isinstance(boundary_config, dict) else {}
    raw_components = boundary_values.get("components", [])
    if not isinstance(raw_components, list):
        return {"components": []}
    start = _required_float(fitted_config, "wavelength_start")
    end = _required_float(fitted_config, "wavelength_end")
    width = max(end - start, 1.0)
    components: list[dict[str, Any]] = []
    for idx, raw_component in enumerate(raw_components):
        if not isinstance(raw_component, dict):
            continue
        amplitude = float(np.clip(
            _finite_float(raw_component.get("amplitude"), default=0.0),
            0.0,
            1.0,
        ))
        if amplitude <= 0.0:
            continue
        bandwidth = float(np.clip(
            _finite_float(raw_component.get("bandwidth"), default=width * 0.15),
            1.0,
            width,
        ))
        center = _finite_float(raw_component.get("band_center"), default=np.nan)
        edge = str(raw_component.get("edge", "")).lower()
        if edge not in {"left", "right"}:
            if np.isfinite(center) and center < start:
                edge = "left"
            elif np.isfinite(center) and center > end:
                edge = "right"
            else:
                continue
        if not np.isfinite(center):
            continue
        if edge == "left":
            center = float(np.clip(center, start - width, start - 1e-6))
        else:
            center = float(np.clip(center, end + 1e-6, end + width))
        name = str(raw_component.get("name") or f"boundary_{edge}_{idx}")
        components.append({
            "name": name,
            "band_center": center,
            "bandwidth": bandwidth,
            "amplitude": amplitude,
            "edge": edge,
        })
    return {"components": components}


def _edge_curvature_type(value: Any) -> str:
    curvature_type = str(value or "random")
    if curvature_type in {"random", "smile", "frown", "asymmetric"}:
        return curvature_type
    if curvature_type == "concave":
        return "frown"
    if curvature_type == "convex":
        return "smile"
    return "random"


def _particle_size_bounds(matrix_type: str) -> tuple[float, float]:
    if matrix_type == "powder":
        return 5.0, 100.0
    if matrix_type == "granular":
        return 50.0, 500.0
    if matrix_type in {"liquid", "emulsion"}:
        return 0.1, 10.0
    if matrix_type == "solid":
        return 100.0, 1000.0
    return 0.1, 1000.0


def _finite_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return parsed


def _create_fitted_generator(
    builder_config: dict[str, Any],
    fitted_config: dict[str, Any],
) -> SyntheticNIRSGenerator:
    features = builder_config["features"]
    nuisance = builder_config["nuisance"]
    library = ComponentLibrary.from_predefined(
        features["components"],
        random_state=builder_config["random_state"],
    )
    _add_boundary_components_to_library(library, nuisance)
    return SyntheticNIRSGenerator(
        wavelength_start=features["wavelength_range"][0],
        wavelength_end=features["wavelength_range"][1],
        wavelength_step=features["wavelength_step"],
        wavelengths=np.asarray(features["wavelength_grid"], dtype=float),
        component_library=library,
        complexity=features["complexity"],
        instrument=features["instrument"],
        measurement_mode=features["measurement_mode"],
        environmental_config=_environmental_config(fitted_config, nuisance),
        scattering_effects_config=_scattering_config(fitted_config, nuisance),
        edge_artifacts_config=_edge_artifacts_config(fitted_config, nuisance),
        custom_params=nuisance["custom_params"],
        random_state=builder_config["random_state"],
    )


def _add_boundary_components_to_library(
    library: ComponentLibrary,
    nuisance: dict[str, Any],
) -> None:
    boundary_config = (
        nuisance.get("fitted_effect_overrides", {})
        .get("boundary_components_config", {})
    )
    raw_components = boundary_config.get("components", []) if isinstance(boundary_config, dict) else []
    if not isinstance(raw_components, list):
        return
    for raw_component in raw_components:
        if not isinstance(raw_component, dict):
            continue
        name = str(raw_component["name"])
        band = NIRBand(
            center=float(raw_component["band_center"]),
            sigma=float(raw_component["bandwidth"]),
            gamma=0.0,
            amplitude=float(raw_component["amplitude"]),
            name=name,
        )
        library.add_component(SpectralComponent(
            name=name,
            bands=[band],
            category="boundary_effect",
            tags=["fitted", "boundary", str(raw_component["edge"])],
        ))


def _generation_concentrations_with_boundary_components(
    concentrations: np.ndarray,
    nuisance: dict[str, Any],
) -> np.ndarray:
    boundary_config = (
        nuisance.get("fitted_effect_overrides", {})
        .get("boundary_components_config", {})
    )
    raw_components = boundary_config.get("components", []) if isinstance(boundary_config, dict) else []
    if not isinstance(raw_components, list) or not raw_components:
        return concentrations
    boundary_concentrations = np.ones((concentrations.shape[0], len(raw_components)), dtype=float)
    return np.hstack([concentrations, boundary_concentrations])


def _environmental_config(
    fitted_config: dict[str, Any],
    nuisance: dict[str, Any],
) -> EnvironmentalEffectsConfig:
    effect_overrides = nuisance.get("fitted_effect_overrides", {})
    temperature_config = effect_overrides.get("temperature_config", fitted_config.get("temperature_config"))
    moisture_config = effect_overrides.get("moisture_config", fitted_config.get("moisture_config"))
    temperature_values = temperature_config if isinstance(temperature_config, dict) else {}
    moisture_values = moisture_config if isinstance(moisture_config, dict) else {}
    return EnvironmentalEffectsConfig(
        temperature=TemperatureConfig(
            sample_temperature=float(temperature_values.get("sample_temperature", nuisance["temperature_c"])),
            temperature_variation=float(temperature_values.get("temperature_variation", 0.0)),
            reference_temperature=float(temperature_values.get("reference_temperature", 25.0)),
            enable_shift=bool(temperature_values.get("enable_shift", True)),
            enable_intensity=bool(temperature_values.get("enable_intensity", True)),
            enable_broadening=bool(temperature_values.get("enable_broadening", True)),
            region_specific=bool(temperature_values.get("region_specific", True)),
        ),
        moisture=MoistureConfig(
            water_activity=float(moisture_values.get("water_activity", 0.5)),
            moisture_content=float(moisture_values.get("moisture_content", 0.10)),
            free_water_fraction=float(moisture_values.get("free_water_fraction", 0.3)),
            bound_water_shift=float(moisture_values.get("bound_water_shift", 25.0)),
            reference_aw=float(moisture_values.get("reference_aw", 0.5)),
        ),
        enable_temperature=True,
        enable_moisture=bool(moisture_values),
    )


def _scattering_config(
    fitted_config: dict[str, Any],
    nuisance: dict[str, Any],
) -> ScatteringEffectsConfig:
    effect_overrides = nuisance.get("fitted_effect_overrides", {})
    particle_config = effect_overrides.get("particle_size_config", fitted_config.get("particle_size_config"))
    particle_values = particle_config if isinstance(particle_config, dict) else {}
    mean_size = float(particle_values.get("mean_size_um", nuisance["particle_size_um"]))
    std_size = float(particle_values.get("std_size_um", max(1e-6, mean_size * 0.05)))
    emsc_config = effect_overrides.get("emsc_config", fitted_config.get("emsc_config"))
    emsc_values = emsc_config if isinstance(emsc_config, dict) else {}
    include_wavelength_terms = bool(emsc_values.get("include_wavelength_terms", True))
    wavelength_coef_std = float(emsc_values.get("wavelength_coef_std", 0.02))
    if not include_wavelength_terms:
        wavelength_coef_std = 0.0
    return ScatteringEffectsConfig(
        particle_size=ParticleSizeConfig(
            distribution=ParticleSizeDistribution(
                mean_size_um=mean_size,
                std_size_um=max(1e-6, std_size),
                min_size_um=max(1e-6, mean_size - 3.0 * std_size),
                max_size_um=max(mean_size + 3.0 * std_size, mean_size),
                distribution="normal",
            ),
            reference_size_um=float(particle_values.get("reference_size_um", mean_size)),
            size_effect_strength=float(particle_values.get("size_effect_strength", 1.0)),
            wavelength_exponent=float(particle_values.get("wavelength_exponent", 1.5)),
            include_path_length_effect=bool(particle_values.get("include_path_length_effect", True)),
            path_length_sensitivity=float(particle_values.get("path_length_sensitivity", 0.5)),
        ),
        emsc=EMSCConfig(
            multiplicative_scatter_std=float(
                emsc_values.get("multiplicative_scatter_std", nuisance["custom_params"]["scatter_alpha_std"])
            ),
            additive_scatter_std=float(
                emsc_values.get("additive_scatter_std", nuisance["custom_params"]["scatter_beta_std"])
            ),
            polynomial_order=int(emsc_values.get("polynomial_order", 2)),
            include_wavelength_terms=include_wavelength_terms,
            wavelength_coef_std=wavelength_coef_std,
        ),
        enable_particle_size=True,
        enable_emsc=True,
    )


def _edge_artifacts_config(
    fitted_config: dict[str, Any],
    nuisance: dict[str, Any],
) -> EdgeArtifactsConfig:
    effect_overrides = nuisance.get("fitted_effect_overrides", {})
    edge_config = effect_overrides.get("edge_artifacts_config", fitted_config.get("edge_artifacts_config"))
    values = edge_config if isinstance(edge_config, dict) else {}
    rolloff = values.get("detector_rolloff")
    rolloff_values = rolloff if isinstance(rolloff, dict) else {}
    stray = values.get("stray_light")
    stray_values = stray if isinstance(stray, dict) else {}
    curvature = values.get("edge_curvature")
    curvature_values = curvature if isinstance(curvature, dict) else {}
    truncated = values.get("truncated_peaks")
    truncated_values = truncated if isinstance(truncated, dict) else {}
    left_curvature = float(curvature_values.get("left_severity", 0.0))
    right_curvature = float(curvature_values.get("right_severity", 0.0))
    return EdgeArtifactsConfig(
        enable_detector_rolloff=bool(rolloff_values.get("enabled", True)),
        detector_model=str(rolloff_values.get("detector_model", "generic_nir")),
        rolloff_severity=float(
            rolloff_values.get("severity", nuisance["edge_artifacts"]["rolloff_severity"])
        ),
        enable_stray_light=bool(stray_values.get("enabled", False)),
        stray_fraction=float(stray_values.get("stray_fraction", 0.001)),
        stray_wavelength_dependent=bool(stray_values.get("wavelength_dependent", True)),
        enable_edge_curvature=bool(curvature_values.get("enabled", False)),
        curvature_type=str(curvature_values.get("curvature_type", "random")),
        left_curvature_severity=left_curvature,
        right_curvature_severity=right_curvature,
        enable_truncated_peaks=bool(truncated_values.get("enabled", False)),
        left_peak_amplitude=float(truncated_values.get("left_amplitude", 0.0)),
        right_peak_amplitude=float(truncated_values.get("right_amplitude", 0.0)),
    )


def _apply_fitted_residual_effects(
    X: np.ndarray,
    *,
    wavelengths: np.ndarray,
    fitted_config: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = fitted_config.get("fitted_residual_effects_config")
    if not _residual_effects_enabled(config):
        return X, {
            "applied": False,
            "reason": "missing_or_disabled_fitted_residual_effects_config",
        }
    assert isinstance(config, dict)
    effect = np.zeros_like(X, dtype=float)
    fit = config.get("fit", {})
    fit_values = fit if isinstance(fit, dict) else {}
    application = config.get("application", {})
    application_values = application if isinstance(application, dict) else {}
    baseline = fit_values.get("residual_baseline", {})
    baseline_values = baseline if isinstance(baseline, dict) else {}
    baseline_coefficients = baseline_values.get("coefficients", [])
    baseline_count = 0
    if isinstance(baseline_coefficients, list) and baseline_coefficients:
        basis = _chebyshev_basis(wavelengths, int(baseline_values.get("order", len(baseline_coefficients) - 1)))
        sampled = np.column_stack([
            _sample_distribution(coefficient, X.shape[0], rng)
            for coefficient in baseline_coefficients
            if isinstance(coefficient, dict)
        ])
        if sampled.size:
            usable = min(sampled.shape[1], basis.shape[0])
            effect += sampled[:, :usable] @ basis[:usable]
            baseline_count = int(usable)

    local_details = fit_values.get("local_details", {})
    local_detail_values = local_details if isinstance(local_details, dict) else {}
    local_templates = local_detail_values.get("templates", [])
    active_detail_counts: list[int] = []
    if isinstance(local_templates, list) and local_templates:
        target_count_distribution = local_detail_values.get("active_count", {})
        target_counts = _sample_integer_distribution(target_count_distribution, X.shape[0], rng)
        weights = _local_detail_weights(local_templates)
        current = X + effect
        for sample_idx in range(X.shape[0]):
            current_count = _local_peak_count(current[sample_idx], wavelengths)
            active_count = int(np.clip(target_counts[sample_idx] - current_count, 0, 96))
            active_detail_counts.append(active_count)
            if active_count <= 0:
                continue
            chosen = rng.choice(len(local_templates), size=active_count, replace=True, p=weights)
            for template_idx in np.atleast_1d(chosen):
                template = local_templates[int(template_idx)]
                if not isinstance(template, dict):
                    continue
                center = _sample_template_center(template, rng)
                sigma = max(_sample_distribution_value(template.get("sigma_nm", {}), rng) * 0.65, 1e-6)
                amplitude = max(_sample_distribution_value(template.get("amplitude_abs", {}), rng) * 1.35, 0.0)
                positive_probability = float(np.clip(
                    _finite_float(template.get("positive_probability"), default=1.0),
                    0.0,
                    1.0,
                ))
                positive_probability = max(positive_probability, 0.9)
                sign = 1.0 if rng.random() <= positive_probability else -1.0
                basis = np.exp(-0.5 * np.square((wavelengths - center) / sigma))
                effect[sample_idx] += sign * amplitude * basis

    peaks = fit_values.get("residual_peaks", {})
    peak_values = peaks if isinstance(peaks, dict) else {}
    templates = peak_values.get("templates", [])
    peak_count = 0
    if isinstance(templates, list):
        for template in templates:
            if not isinstance(template, dict):
                continue
            coefficient = template.get("coefficient", {})
            if not isinstance(coefficient, dict):
                continue
            center = _finite_float(template.get("center_nm"), default=float(np.mean(wavelengths)))
            sigma = max(_finite_float(template.get("sigma_nm"), default=1.0), 1e-6)
            basis = np.exp(-0.5 * np.square((wavelengths - center) / sigma))
            effect += _sample_distribution(coefficient, X.shape[0], rng)[:, None] * basis[None, :]
            peak_count += 1

    observed_distributions = config.get("observed_distributions", {})
    observed_values = observed_distributions if isinstance(observed_distributions, dict) else {}
    if bool(application_values.get("match_observed_metric_distributions", False)):
        effect = _match_observed_metric_distributions(
            X,
            effect,
            wavelengths=wavelengths,
            observed_distributions=observed_values,
            rng=rng,
        )
        if isinstance(local_templates, list) and local_templates:
            target_count_distribution = local_detail_values.get("active_count", {})
            target_counts = _sample_integer_distribution(target_count_distribution, X.shape[0], rng)
            weights = _local_detail_weights(local_templates)
            current = X + effect
            for sample_idx in range(X.shape[0]):
                current_count = _local_peak_count(current[sample_idx], wavelengths)
                active_count = int(np.clip(target_counts[sample_idx] - current_count, 0, 96))
                active_detail_counts.append(active_count)
                if active_count <= 0:
                    continue
                chosen = rng.choice(len(local_templates), size=active_count, replace=True, p=weights)
                for template_idx in np.atleast_1d(chosen):
                    template = local_templates[int(template_idx)]
                    if not isinstance(template, dict):
                        continue
                    center = _sample_template_center(template, rng)
                    sigma = max(_sample_distribution_value(template.get("sigma_nm", {}), rng) * 0.5, 1e-6)
                    amplitude = max(_sample_distribution_value(template.get("amplitude_abs", {}), rng) * 1.75, 0.0)
                    basis = np.exp(-0.5 * np.square((wavelengths - center) / sigma))
                    effect[sample_idx] += amplitude * basis
            effect = _match_observed_metric_distributions(
                X,
                effect,
                wavelengths=wavelengths,
                observed_distributions=observed_values,
                rng=rng,
            )

    max_abs_effect = _finite_float(application_values.get("max_abs_effect"), default=np.nan)
    if np.isfinite(max_abs_effect) and max_abs_effect > 0.0:
        effect = np.clip(effect, -max_abs_effect, max_abs_effect)
    if not np.isfinite(effect).all():
        return X, {
            "applied": False,
            "reason": "non_finite_sampled_residual_effect",
        }
    return X + effect, {
        "applied": bool(baseline_count or peak_count or active_detail_counts),
        "source": config.get("source"),
        "oracle_provenance": bool(config.get("oracle_provenance", True)),
        "no_oracle": bool(config.get("no_oracle", False)),
        "baseline_coefficients": baseline_count,
        "peak_templates": peak_count,
        "local_detail_templates": len(local_templates) if isinstance(local_templates, list) else 0,
        "active_detail_counts": {
            "mean": float(np.mean(active_detail_counts)) if active_detail_counts else 0.0,
            "max": int(np.max(active_detail_counts)) if active_detail_counts else 0,
            "total": int(np.sum(active_detail_counts)) if active_detail_counts else 0,
        },
        "target_observed_distributions": _to_builtin(observed_values),
        "max_abs_applied": float(np.max(np.abs(effect))) if effect.size else 0.0,
    }


def _sample_distribution(
    distribution: dict[str, Any],
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    raw_probabilities = distribution.get("quantile_probabilities", [])
    raw_values = distribution.get("quantile_values", [])
    if (
        isinstance(raw_probabilities, list)
        and isinstance(raw_values, list)
        and len(raw_probabilities) == len(raw_values)
        and len(raw_values) >= 2
    ):
        probabilities = np.asarray([_finite_float(item, default=np.nan) for item in raw_probabilities], dtype=float)
        values = np.asarray([_finite_float(item, default=np.nan) for item in raw_values], dtype=float)
        finite = np.isfinite(probabilities) & np.isfinite(values)
        probabilities = probabilities[finite]
        values = values[finite]
        if probabilities.size >= 2:
            order = np.argsort(probabilities)
            probabilities = np.clip(probabilities[order], 0.0, 1.0)
            values = values[order]
            unique_probabilities, unique_indices = np.unique(probabilities, return_index=True)
            if unique_probabilities.size >= 2:
                samples = np.interp(rng.random(n_samples), unique_probabilities, values[unique_indices])
                clip_abs = _finite_float(distribution.get("clip_abs"), default=np.nan)
                if np.isfinite(clip_abs) and clip_abs > 0.0:
                    samples = np.clip(samples, -clip_abs, clip_abs)
                return np.asarray(samples, dtype=float)
    mean = _finite_float(distribution.get("mean"), default=0.0)
    std = max(_finite_float(distribution.get("std"), default=0.0), 0.0)
    if std == 0.0:
        samples = np.full(n_samples, mean, dtype=float)
    else:
        samples = rng.normal(mean, std, size=n_samples)
    q_low = _finite_float(distribution.get("q01", distribution.get("q05")), default=mean)
    q_high = _finite_float(distribution.get("q99", distribution.get("q95")), default=mean)
    lower = min(q_low, q_high)
    upper = max(q_low, q_high)
    clip_abs = _finite_float(distribution.get("clip_abs"), default=np.nan)
    if np.isfinite(clip_abs) and clip_abs > 0.0:
        lower = max(lower, -clip_abs)
        upper = min(upper, clip_abs)
    return np.asarray(np.clip(samples, lower, upper), dtype=float)


def _sample_distribution_value(value: Any, rng: np.random.Generator) -> float:
    if not isinstance(value, dict):
        return 0.0
    return float(_sample_distribution(value, 1, rng)[0])


def _sample_integer_distribution(
    value: Any,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not isinstance(value, dict):
        return np.zeros(n_samples, dtype=int)
    raw_values = value.get("values", [])
    raw_probabilities = value.get("probabilities", [])
    if not isinstance(raw_values, list) or not raw_values:
        mean = max(_finite_float(value.get("mean"), default=0.0), 0.0)
        std = max(_finite_float(value.get("std"), default=0.0), 0.0)
        return np.asarray(np.maximum(np.rint(rng.normal(mean, std, size=n_samples)), 0), dtype=int)
    values = np.asarray([int(max(_finite_float(item, default=0.0), 0.0)) for item in raw_values], dtype=int)
    if isinstance(raw_probabilities, list) and len(raw_probabilities) == values.size:
        probabilities = np.asarray([max(_finite_float(item, default=0.0), 0.0) for item in raw_probabilities], dtype=float)
    else:
        probabilities = np.ones(values.size, dtype=float)
    probability_sum = float(np.sum(probabilities))
    if probability_sum <= 0.0:
        probabilities = np.ones(values.size, dtype=float) / float(values.size)
    else:
        probabilities = probabilities / probability_sum
    return np.asarray(rng.choice(values, size=n_samples, replace=True, p=probabilities), dtype=int)


def _local_detail_weights(templates: list[Any]) -> np.ndarray:
    weights = np.asarray([
        max(_finite_float(template.get("weight"), default=0.0), 0.0) if isinstance(template, dict) else 0.0
        for template in templates
    ], dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones(len(templates), dtype=float) / float(len(templates))
    return weights / total


def _sample_template_center(template: dict[str, Any], rng: np.random.Generator) -> float:
    center = _finite_float(template.get("center_nm"), default=0.0)
    center_std = max(_finite_float(template.get("center_std_nm"), default=0.0), 0.0)
    if center_std <= 0.0:
        return center
    return float(rng.normal(center, center_std))


def _match_observed_metric_distributions(
    X: np.ndarray,
    effect: np.ndarray,
    *,
    wavelengths: np.ndarray,
    observed_distributions: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    curvature_distribution = observed_distributions.get("baseline_curvature", {})
    derivative_distribution = observed_distributions.get("derivative_std", {})
    if not isinstance(curvature_distribution, dict) and not isinstance(derivative_distribution, dict):
        return effect
    adjusted = X + effect
    for sample_idx in range(adjusted.shape[0]):
        row = adjusted[sample_idx]
        target_curvature = _sample_distribution_value(curvature_distribution, rng)
        if target_curvature > 0.0:
            polynomial = _polynomial_baseline(row, order=3)
            residual = row - polynomial
            current_curvature = float(np.std(residual))
            if current_curvature > 1e-12:
                scale = float(np.clip(target_curvature / current_curvature, 0.45, 1.65))
                row = polynomial + residual * scale
        target_derivative = _sample_distribution_value(derivative_distribution, rng)
        if target_derivative > 0.0:
            current_derivative = _derivative_std(row, wavelengths)
            detail = row - _moving_average(row, window=9)
            detail_std = _derivative_std(detail, wavelengths)
            if current_derivative > target_derivative and detail_std > 1e-12:
                scale = float(np.clip(target_derivative / current_derivative, 0.45, 1.0))
                row = row + (scale - 1.0) * detail
            elif current_derivative < target_derivative and detail_std > 1e-12:
                needed = np.sqrt(max(target_derivative**2 - current_derivative**2, 0.0))
                gain = float(np.clip(needed / detail_std, 0.0, 1.25))
                polarity = 1.0 if rng.random() < 0.5 else -1.0
                row = row + polarity * gain * detail
        if target_curvature > 0.0:
            polynomial = _polynomial_baseline(row, order=3)
            residual = row - polynomial
            current_curvature = float(np.std(residual))
            if current_curvature > 1e-12:
                scale = float(np.clip(target_curvature / current_curvature, 0.45, 1.65))
                row = polynomial + residual * scale
        adjusted[sample_idx] = row
    return np.asarray(adjusted - X, dtype=float)


def _polynomial_baseline(row: np.ndarray, *, order: int) -> np.ndarray:
    positions = np.linspace(-1.0, 1.0, row.size)
    basis = np.vander(positions, order + 1, increasing=True)
    coefficients, *_ = np.linalg.lstsq(basis, row, rcond=None)
    return np.asarray(basis @ coefficients, dtype=float)


def _derivative_std(row: np.ndarray, wavelengths: np.ndarray) -> float:
    if row.size < 2 or wavelengths.size < 2:
        return 0.0
    derivative = np.diff(row) / np.diff(wavelengths)
    return float(np.std(derivative))


def _moving_average(row: np.ndarray, *, window: int) -> np.ndarray:
    if row.size < 3:
        return row.copy()
    size = int(min(max(window, 3), row.size if row.size % 2 == 1 else row.size - 1))
    if size < 3:
        return row.copy()
    kernel = np.ones(size, dtype=float) / float(size)
    pad = size // 2
    padded = np.pad(row, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _local_peak_count(row: np.ndarray, wavelengths: np.ndarray) -> int:
    if row.size < 3:
        return 0
    prominence = max(float(np.std(row)) * 0.04, 1e-10)
    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(row, distance=1, prominence=prominence)
        return int(peaks.size)
    except Exception:
        local = np.flatnonzero((row[1:-1] > row[:-2]) & (row[1:-1] >= row[2:])) + 1
        return int(np.sum(row[local] >= prominence))


def _chebyshev_basis(wavelengths: np.ndarray, order: int) -> np.ndarray:
    if wavelengths.size == 0:
        return np.empty((0, 0), dtype=float)
    lower = float(wavelengths[0])
    upper = float(wavelengths[-1])
    if np.isclose(lower, upper):
        scaled = np.zeros_like(wavelengths, dtype=float)
    else:
        scaled = 2.0 * (wavelengths - lower) / (upper - lower) - 1.0
    return np.polynomial.chebyshev.chebvander(scaled, max(0, order)).T.astype(float)


def _summarize_residual_effects_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"enabled": False, "status": "missing"}
    fit = value.get("fit", {})
    fit_values = fit if isinstance(fit, dict) else {}
    baseline = fit_values.get("residual_baseline", {})
    baseline_values = baseline if isinstance(baseline, dict) else {}
    peaks = fit_values.get("residual_peaks", {})
    peak_values = peaks if isinstance(peaks, dict) else {}
    local_details = fit_values.get("local_details", {})
    local_detail_values = local_details if isinstance(local_details, dict) else {}
    coefficients = baseline_values.get("coefficients", [])
    templates = peak_values.get("templates", [])
    local_templates = local_detail_values.get("templates", [])
    return {
        "enabled": bool(value.get("enabled", False)),
        "status": value.get("status"),
        "source": value.get("source"),
        "oracle_provenance": bool(value.get("oracle_provenance", True)),
        "no_oracle": bool(value.get("no_oracle", False)),
        "baseline_coefficients": len(coefficients) if isinstance(coefficients, list) else 0,
        "peak_templates": len(templates) if isinstance(templates, list) else 0,
        "local_detail_templates": len(local_templates) if isinstance(local_templates, list) else 0,
        "observed_distributions": _to_builtin(value.get("observed_distributions", {})),
        "residual_summary": _to_builtin(value.get("residual_summary", {})),
    }


def _residual_effects_enabled(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value.get("enabled", False))
        and value.get("source") == "observed_fitted_spectra"
        and value.get("oracle_provenance") is False
    )


def _with_a3_validation(
    validation_summary: dict[str, Any],
    *,
    unsupported_fields: list[dict[str, Any]],
    assumptions: list[str],
) -> dict[str, Any]:
    updated = dict(validation_summary)
    updated["unsupported_fields"] = unsupported_fields
    updated["adapter_notes"] = [
        *updated.get("adapter_notes", []),
        "A3 maps RealDataFitter.to_full_config() through the A2 dataset run path.",
        "Unsupported fitted fields are reported explicitly; they are not silently interpreted.",
    ]
    updated["adapter_assumptions"] = assumptions
    return cast("dict[str, Any]", _to_builtin(updated))


def _unsupported_fields(fitted_config: dict[str, Any]) -> list[dict[str, Any]]:
    unsupported: list[UnsupportedFittedField] = []
    for field, value in fitted_config.items():
        if field == "temperature_config":
            unsupported.extend(_unsupported_nested(
                field,
                value,
                supported_keys=SUPPORTED_TEMPERATURE_CONFIG_KEYS,
                reason="A3 maps only executable temperature generator parameters.",
            ))
            continue
        if field == "moisture_config":
            unsupported.extend(_unsupported_nested(
                field,
                value,
                supported_keys=SUPPORTED_MOISTURE_CONFIG_KEYS,
                reason="A3 maps only executable moisture generator parameters.",
            ))
            continue
        if field == "emsc_config":
            unsupported.extend(_unsupported_nested(
                field,
                value,
                supported_keys=SUPPORTED_EMSC_CONFIG_KEYS,
                reason="A3 maps only executable EMSC generator parameters.",
            ))
            continue
        if field in SUPPORTED_FITTED_FIELDS:
            continue
        if field == "particle_size_config":
            unsupported.extend(_unsupported_nested(
                field,
                value,
                supported_keys=SUPPORTED_PARTICLE_SIZE_CONFIG_KEYS,
                reason="A3 maps only executable particle-size generator parameters.",
            ))
            continue
        if field == "edge_artifacts_config":
            unsupported.extend(_unsupported_edge_artifacts(value))
            continue
        if field == "boundary_components_config":
            unsupported.extend(_unsupported_nested(
                field,
                value,
                supported_keys=SUPPORTED_BOUNDARY_COMPONENT_CONFIG_KEYS,
                reason="A3 maps only explicit executable boundary component definitions.",
            ))
            continue
        if field == "fitted_residual_effects_config":
            continue
        unsupported.append(UnsupportedFittedField(
            field=field,
            reason="No executable A2 mapping in the A3 bench adapter.",
            value_summary=_summarize_value(value),
        ))
    return [item.to_dict() for item in unsupported]


def _unsupported_nested(
    field: str,
    value: Any,
    *,
    supported_keys: set[str],
    reason: str,
) -> list[UnsupportedFittedField]:
    if not isinstance(value, dict):
        return [UnsupportedFittedField(field, reason, _summarize_value(value))]
    return [
        UnsupportedFittedField(
            field=f"{field}.{key}",
            reason=reason,
            value_summary=_summarize_value(nested),
        )
        for key, nested in value.items()
        if key not in supported_keys
    ]


def _unsupported_edge_artifacts(value: Any) -> list[UnsupportedFittedField]:
    if not isinstance(value, dict):
        return [UnsupportedFittedField(
            "edge_artifacts_config",
            "A3 maps only executable edge artifact generator parameters.",
            _summarize_value(value),
        )]
    unsupported: list[UnsupportedFittedField] = []
    for artifact, artifact_config in value.items():
        if artifact == "detector_rolloff" and isinstance(artifact_config, dict):
            unsupported.extend(_unsupported_nested(
                "edge_artifacts_config.detector_rolloff",
                artifact_config,
                supported_keys={"enabled", "detector_model", "severity"},
                reason="A3 maps only executable detector-rolloff generator parameters.",
            ))
            continue
        if artifact == "stray_light" and isinstance(artifact_config, dict):
            unsupported.extend(_unsupported_nested(
                "edge_artifacts_config.stray_light",
                artifact_config,
                supported_keys={"enabled", "stray_fraction", "wavelength_dependent"},
                reason="A3 maps only executable stray-light generator parameters.",
            ))
            continue
        if artifact == "edge_curvature" and isinstance(artifact_config, dict):
            unsupported.extend(_unsupported_nested(
                "edge_artifacts_config.edge_curvature",
                artifact_config,
                supported_keys={"enabled", "curvature_type", "left_severity", "right_severity"},
                reason="A3 maps only executable edge-curvature generator parameters.",
            ))
            continue
        if artifact == "truncated_peaks" and isinstance(artifact_config, dict):
            unsupported.extend(_unsupported_nested(
                "edge_artifacts_config.truncated_peaks",
                artifact_config,
                supported_keys={"enabled", "left_amplitude", "right_amplitude"},
                reason="A3 maps only executable truncated-peak generator parameters.",
            ))
            continue
        unsupported.append(UnsupportedFittedField(
            field=f"edge_artifacts_config.{artifact}",
            reason="A2 has no executable mapping for this fitted edge artifact.",
            value_summary=_summarize_value(artifact_config),
        ))
    return unsupported


def _components(fitted_config: dict[str, Any]) -> list[str]:
    raw_components = fitted_config.get("components", [])
    if not isinstance(raw_components, (list, tuple)):
        return []
    return [str(component) for component in raw_components if str(component)]


def _particle_size_um(
    fitted_config: dict[str, Any],
    *,
    override: dict[str, Any] | None = None,
) -> float:
    particle_config = override if override is not None else fitted_config.get("particle_size_config")
    if isinstance(particle_config, dict) and "mean_size_um" in particle_config:
        return float(particle_config["mean_size_um"])
    return 100.0


def _detector_rolloff_severity(
    fitted_config: dict[str, Any],
    *,
    override: dict[str, Any] | None = None,
) -> float | None:
    edge_config = override if override is not None else fitted_config.get("edge_artifacts_config")
    if not isinstance(edge_config, dict):
        return None
    detector_rolloff = edge_config.get("detector_rolloff")
    if not isinstance(detector_rolloff, dict) or "severity" not in detector_rolloff:
        return None
    return float(detector_rolloff["severity"])


def _required_float(config: dict[str, Any], field: str) -> float:
    try:
        return float(config[field])
    except Exception as exc:
        raise FittedConfigAdapterError({
            "status": "failed",
            "failures": [{
                "reason": "missing_or_invalid_fitted_field",
                "field": field,
                "message": f"Required fitted config field {field!r} is missing or not numeric.",
            }],
            "unsupported_fields": _unsupported_fields(config),
        }) from exc


def _bounded_wavelength_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise FittedConfigAdapterError({
            "status": "failed",
            "failures": [{
                "reason": "invalid_wavelength_step",
                "field": "wavelength_step",
                "message": f"wavelength_step must be positive, got {step}",
            }],
            "unsupported_fields": [],
        })
    grid = np.arange(start, end + step * 0.25, step, dtype=float)
    grid = grid[grid <= end + 1e-9]
    if grid.size == 0 or not np.isclose(grid[0], start):
        grid = np.insert(grid, 0, start)
    return grid


def _summarize_value(value: Any) -> Any:
    value = _to_builtin(value)
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value)}
    if isinstance(value, list):
        return {"type": "list", "length": len(value), "sample": value[:3]}
    return value


def _raise_adapter_error(
    *,
    failures: list[dict[str, Any]],
    unsupported_fields: list[dict[str, Any]],
    assumptions: list[str],
) -> NoReturn:
    raise FittedConfigAdapterError({
        "status": "failed",
        "failures": failures,
        "unsupported_fields": unsupported_fields,
        "adapter_assumptions": assumptions,
    })
