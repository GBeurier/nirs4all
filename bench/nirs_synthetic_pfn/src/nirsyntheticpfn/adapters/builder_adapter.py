"""Phase A2 adapter from canonical prior records to finite dataset runs."""

from __future__ import annotations

import hashlib
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

# R2a sentinel mechanistic profiles. Bench-only. Non-oracle, non-learned, deterministic
# by profile name + seed. They never read real spectra, labels, splits, targets, or
# AUC; they never modify gate thresholds or metric definitions. These are fixed
# mechanistic approximations applied post-generation in bench-only experiments.
R2A_MECHANISTIC_PROFILES: tuple[str, ...] = (
    "r2a_baseline",
    "r2a_pathlength_drift",
    "r2a_baseline_curvature",
    "r2a_emsc_like_scatter",
    "r2a_instrumental_broadening",
    "r2a_structured_noise",
)

# R2c sentinel matrix remediation profiles. Bench-only opt-in remediation that
# (a) re-biases composition with a tight Dirichlet centered on textbook matrix
# composition and (b) applies a mechanistic optical-path scale to the generated
# absorbance. Constants are mechanistic (no real spectra, labels, splits,
# targets, or AUC consulted) and audit trails are written to
# ``metadata["r2c_mechanistic_remediation"]``.
R2C_REMEDIATION_PROFILES: tuple[str, ...] = ("r2c_sentinel_matrix_v1",)

# R2d sentinel matrix remediation profiles. Bench-only opt-in extension of the
# R2c rule table that adds mechanistic remediations for BEER (``beverage_wine``)
# and CORN (``agriculture_grain``) sentinels while keeping the DIESEL
# (``petrochem_fuels``) and MILK (``food_dairy``) rules from R2c byte-for-byte
# identical. Same audit contract as R2c (no real spectra, labels, splits,
# targets, or AUC consulted); audit metadata records the exact requested
# profile id under the shared ``metadata["r2c_mechanistic_remediation"]`` key
# (kept for backward compatibility with the R2c CSV/MD column schema).
R2D_REMEDIATION_PROFILES: tuple[str, ...] = ("r2d_sentinel_matrix_v1",)

# R2f sentinel matrix remediation profiles. Bench-only opt-in extension of R2d
# that adds a mechanistic clear-juice remediation for ``beverage_juice`` while
# leaving R2c/R2d behavior unchanged unless this exact profile is requested.
R2F_REMEDIATION_PROFILES: tuple[str, ...] = ("r2f_sentinel_matrix_v1",)

# R2g sentinel matrix remediation profiles. Bench-only opt-in extension of R2f
# that adds mechanistic topsoil remediation for ``environmental_soil`` while
# preserving every R2f rule and constant for existing domains.
R2G_REMEDIATION_PROFILES: tuple[str, ...] = ("r2g_sentinel_matrix_v1",)

# R2h sentinel matrix remediation profiles. Bench-only opt-in extension of R2g
# that adds a BERRY/juice apparent percent-transmittance/intensity readout while
# preserving the R2g rule table for every non-BERRY routed row. The R2h
# constants are fixed cloudy-berry optical priors, not calibration parameters
# and not statistics captured from real sentinels.
R2H_REMEDIATION_PROFILES: tuple[str, ...] = ("r2h_sentinel_matrix_v1",)

# R2i sentinel matrix remediation profiles. Bench-only opt-in extension used
# for FruitPuree semi-solid fruit paste routing. BERRY remains routed through
# R2h by the audit layer; this builder profile keeps beverage_juice on the R2f
# clear-juice rule and adds a separate agriculture_fruit puree rule.
R2I_REMEDIATION_PROFILES: tuple[str, ...] = ("r2i_sentinel_matrix_v1",)

# R2j sentinel matrix remediation profiles. Bench-only opt-in extension of R2i
# that changes only ``petrochem_fuels``: DIESEL is treated as a micro-path raw
# transmission/absorbance readout with a small fixed detector baseline. This is
# a fixed optical prior, not a calibration to sentinels or real-stat capture.
R2J_REMEDIATION_PROFILES: tuple[str, ...] = ("r2j_sentinel_matrix_v1",)

# R2k sentinel matrix remediation profiles. Bench-only opt-in extension of R2i
# that changes only ``petrochem_fuels``. Unlike R2j's uniform micro-path
# compression, R2k keeps a micro-path continuum readout but preserves fuel CH
# overtone contrast through a fixed continuum/residual split and a
# wavelength-dependent path perturbation. Constants are fixed petrochemical
# optical priors, not calibration or real-stat capture.
R2K_REMEDIATION_PROFILES: tuple[str, ...] = ("r2k_sentinel_matrix_v1",)

# R2l sentinel matrix remediation profiles. Bench-only opt-in extension of R2k
# for LUCAS-style raw soil rows. The builder only enables the soil rule when a
# bench audit has attached the explicit ``_r2l_lucas_soil_route`` provenance
# marker; unmarked soil records fall back to R2g so PHOSPHORUS is not silently
# changed by a domain-only profile call.
R2L_REMEDIATION_PROFILES: tuple[str, ...] = ("r2l_sentinel_matrix_v1",)

# R2m sentinel matrix remediation profiles. Bench-only opt-in extension of R2l
# that changes only ``food_dairy``/MILK-style raw emulsion rows. The milk rule
# exposes generated absorbance through a fixed fat-globule scatter +
# inverse-transflectance intensity readout instead of treating raw spectra as
# absorbance. Constants are optical/readout priors, not calibration or
# real-stat capture.
R2M_REMEDIATION_PROFILES: tuple[str, ...] = ("r2m_sentinel_matrix_v1",)

# R2n sentinel matrix remediation profiles. Bench-only opt-in extension of R2m
# that changes only MANURE21-marked dried/ground manure rows. The route uses a
# separate organic-mineral manure matrix and a dark organic-albedo diffuse
# reflectance readout; it does not reuse the LUCAS R2l mineral-soil readout or
# the PHOSPHORUS R2g soil route by domain alone.
R2N_REMEDIATION_PROFILES: tuple[str, ...] = ("r2n_sentinel_matrix_v1",)

# R2o sentinel matrix remediation profiles. Bench-only opt-in extension of R2n
# that changes only explicit BEER-marked fermented liquid rows. Beer remains a
# liquid transmission/cuvette route, but the raw readout includes fixed
# long-path, haze/carbonation/turbidity attenuation instead of exposing the
# generated absorbance through the older R2d long-path scale alone.
R2O_REMEDIATION_PROFILES: tuple[str, ...] = ("r2o_sentinel_matrix_v1",)

# R2p sentinel matrix remediation profiles. Bench-only opt-in extension of R2o
# that changes only explicit PHOSPHORUS-marked mineral/phosphate soil rows. It
# keeps the R2g soil composition but exposes the centered absorbance residual
# against a fixed phosphate/mineral albedo continuum instead of reusing the
# LUCAS topsoil floor.
R2P_REMEDIATION_PROFILES: tuple[str, ...] = ("r2p_sentinel_matrix_v1",)

# R2q sentinel matrix remediation profiles. Bench-only opt-in extension of R2p
# that changes only explicitly marked LUCAS pH Organic rows. It does not change
# the general LUCAS mineral-organic topsoil route, PHOSPHORUS, or other R2p
# inherited routes.
R2Q_REMEDIATION_PROFILES: tuple[str, ...] = ("r2q_sentinel_matrix_v1",)

# R2r sentinel matrix remediation profiles. Bench-only opt-in extension of R2q
# that changes only explicitly marked FruitPuree rows. It keeps BERRY on R2h
# and beverage_juice on the inherited juice routes; the puree readout is a
# separate semi-solid tissue transflectance residual prior.
R2R_REMEDIATION_PROFILES: tuple[str, ...] = ("r2r_sentinel_matrix_v1",)

# R2s sentinel matrix remediation profiles. Bench-only opt-in extension of R2r
# that changes only explicitly marked DIESEL/petrochem fuel rows. The readout
# keeps R2k's CH-overtone residual mechanism but uses a shorter fixed
# micro-path continuum to reduce raw absorbance mean shift without falling back
# to R2j's derivative-suppressing uniform compression.
R2S_REMEDIATION_PROFILES: tuple[str, ...] = ("r2s_sentinel_matrix_v1",)

# R2t sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicitly marked MANURE21 dried/ground manure rows.
# The readout keeps R2n's dark organic/mineral albedo prior but adds fixed
# heterogeneous diffuse-reflectance terms for particle-size scatter, residual
# moisture patches, and organic/mineral/ash lumps to address under-amplified
# manure matrix variability without using calibration or real-stat capture.
R2T_REMEDIATION_PROFILES: tuple[str, ...] = ("r2t_sentinel_matrix_v1",)

# R2u sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that supersedes the R2t MANURE21 trial without removing it. R2u keeps the
# R2n dark organic/mineral continuum and adds only centered, bounded spectral
# scatter/residual terms so amplitude can increase without a global mean lift.
R2U_REMEDIATION_PROFILES: tuple[str, ...] = ("r2u_sentinel_matrix_v1",)

# R2v sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicit MANURE21 rows. R2v keeps R2u's fixed manure
# continuum expectation but uses balanced bounded prior draws and centered
# spectral-shape scatter terms to avoid repeated seed-level mean drift.
R2V_REMEDIATION_PROFILES: tuple[str, ...] = ("r2v_sentinel_matrix_v1",)

# R2w sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicit MANURE21 rows. R2w keeps R2v's centered
# residual/scatter readout but widens the fixed dark organic/mineral albedo
# prior to transfer dried-manure cup heterogeneity without calibration or
# statistical capture.
R2W_REMEDIATION_PROFILES: tuple[str, ...] = ("r2w_sentinel_matrix_v1",)

# Union of every supported remediation profile id. Used by the bench-only
# ``--remediation-profile`` flag and the validation guard.
ALL_REMEDIATION_PROFILES: tuple[str, ...] = (
    *R2C_REMEDIATION_PROFILES,
    *R2D_REMEDIATION_PROFILES,
    *R2F_REMEDIATION_PROFILES,
    *R2G_REMEDIATION_PROFILES,
    *R2H_REMEDIATION_PROFILES,
    *R2I_REMEDIATION_PROFILES,
    *R2J_REMEDIATION_PROFILES,
    *R2K_REMEDIATION_PROFILES,
    *R2L_REMEDIATION_PROFILES,
    *R2M_REMEDIATION_PROFILES,
    *R2N_REMEDIATION_PROFILES,
    *R2O_REMEDIATION_PROFILES,
    *R2P_REMEDIATION_PROFILES,
    *R2Q_REMEDIATION_PROFILES,
    *R2R_REMEDIATION_PROFILES,
    *R2S_REMEDIATION_PROFILES,
    *R2T_REMEDIATION_PROFILES,
    *R2U_REMEDIATION_PROFILES,
    *R2V_REMEDIATION_PROFILES,
    *R2W_REMEDIATION_PROFILES,
)

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
    mechanistic_profile: str | None = None,
    remediation_profile: str | None = None,
) -> SyntheticDatasetRun:
    """Generate and validate a finite synthetic dataset from one canonical record.

    When ``mechanistic_profile`` is ``None`` (default) the generation path is
    unchanged. When set to a known R2a sentinel profile name, a deterministic
    mechanistic transform (pathlength, baseline, scatter, broadening, structured
    noise) is applied post-generation as a bench-only approximation. The
    transform is fully determined by ``profile`` + ``random_seed`` and never
    reads real spectra, labels, splits, targets, or AUC.

    When ``remediation_profile`` is ``None`` (default) the generation path is
    unchanged. When set to ``"r2c_sentinel_matrix_v1"`` and ``record.domain_key``
    matches a domain with a textbook composition rule (currently
    ``petrochem_fuels`` and ``food_dairy``), composition is re-sampled from a
    tight Dirichlet centered on textbook fractions and a mechanistic optical
    path scale is applied to the generated absorbance. Constants are documented
    and never derived from real spectra, labels, splits, targets, or AUC.

    When set to ``"r2d_sentinel_matrix_v1"``, the same R2c rules apply
    unchanged for ``petrochem_fuels``/``food_dairy`` and additional bench-only
    mechanistic rules apply for ``beverage_wine`` (BEER: long-liquid optical
    path scale) and ``agriculture_grain`` (CORN: instrumental Gaussian
    smoothing plus light powder scatter). The audit object recorded under
    ``metadata["r2c_mechanistic_remediation"]`` records the exact requested
    profile id. Same non-oracle / no-real-stat-capture contract as R2c.

    When set to ``"r2f_sentinel_matrix_v1"``, the R2d rule table is extended
    with a ``beverage_juice`` clear-juice rule using only registered juice
    components and a bounded liquid cuvette path scale. R2c/R2d defaults and
    constants are unchanged.

    When set to ``"r2g_sentinel_matrix_v1"``, the R2f rule table is extended
    with an ``environmental_soil`` topsoil rule using registered soil
    mineral/organic components, Gaussian smoothing, and bounded diffuse powder
    path/scatter compression. This remains audit-only; it does not claim to
    solve mean shift and it never reads real statistics.

    When set to ``"r2h_sentinel_matrix_v1"``, the R2g rule table is extended
    with a ``beverage_juice`` cloudy-berry readout rule that converts generated
    absorbance to bounded apparent percent-transmittance/intensity using fixed
    optical prior constants and per-sample broadband turbidity offsets. This is
    a raw-instrument readout transform for datasets whose axis is already in
    percent/intensity units; it is not PCA/statistical capture, a real-fit, or
    threshold calibration from sentinel morphology metrics.

    When set to ``"r2i_sentinel_matrix_v1"``, the R2g rule table is extended
    with an ``agriculture_fruit`` semi-solid puree rule using registered fruit
    tissue components, bounded paste path/scatter compression, smoothing, and a
    small fixed raw-absorbance baseline. BERRY percent/intensity readout is not
    part of this builder profile and must be routed separately by the audit.

    When set to ``"r2j_sentinel_matrix_v1"``, the R2i rule table is extended
    only for ``petrochem_fuels``: generated diesel absorbance is exposed through
    a fixed micro-path raw absorbance readout with a small detector baseline.
    For all non-``petrochem_fuels`` domains the builder delegates to the R2i
    profile so inherited routes keep byte-identical composition/spectra draws.
    This remains an optical prior only, not real-stat capture or calibration.

    When set to ``"r2k_sentinel_matrix_v1"``, the R2i inheritance contract is
    the same as R2j for non-fuel domains, but DIESEL/petrochem fuels use a
    micro-path continuum readout with fixed CH overtone feature contrast and a
    wavelength-dependent path perturbation so derivatives are not uniformly
    compressed. This remains a fixed mechanistic prior only.

    When set to ``"r2l_sentinel_matrix_v1"``, R2k inheritance is kept for
    petrochem fuels and non-soil domains, while ``environmental_soil`` only
    uses the LUCAS-style raw apparent absorbance readout when the record carries
    the explicit bench-only LUCAS route provenance marker. Unmarked soil records
    fall back to R2g so PHOSPHORUS remains unchanged unless explicitly routed
    otherwise.

    When set to ``"r2m_sentinel_matrix_v1"``, R2l inheritance is kept for
    non-dairy rows. Only ``food_dairy`` records use the MILK raw emulsion
    inverse-transflectance readout.

    When set to ``"r2n_sentinel_matrix_v1"``, R2m inheritance is kept for all
    non-MANURE21 rows. Only records carrying the explicit bench-only MANURE21
    route marker use the dried/ground organic-mineral manure readout.

    When set to ``"r2o_sentinel_matrix_v1"``, R2n inheritance is kept for all
    non-BEER rows. Only records carrying the explicit bench-only BEER route
    marker use the fermented-beer turbid cuvette readout.

    When set to ``"r2p_sentinel_matrix_v1"``, R2o inheritance is kept for all
    non-PHOSPHORUS rows. Only records carrying the explicit bench-only
    PHOSPHORUS route marker use the phosphate/mineral albedo residual readout.

    When set to ``"r2q_sentinel_matrix_v1"``, R2p inheritance is kept for all
    non-LUCAS-pH-Organic rows. Only records carrying the explicit bench-only
    LUCAS pH Organic route marker use the humic organic-topsoil albedo/OH
    readout.

    When set to ``"r2r_sentinel_matrix_v1"``, R2q inheritance is kept for all
    non-FruitPuree rows. Only ``agriculture_fruit`` records carrying the
    explicit bench-only FruitPuree route marker use the strawberry puree
    transflectance residual readout.

    When set to ``"r2s_sentinel_matrix_v1"``, R2r inheritance is kept for all
    non-DIESEL rows. Only ``petrochem_fuels`` records carrying the explicit
    bench-only DIESEL route marker use the shorter blank-referenced micro-path
    continuum while retaining the R2k CH-overtone residual readout.

    When set to ``"r2t_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only records carrying the explicit bench-only MANURE21
    route marker use the heterogeneous dried-manure scatter/patch readout.

    When set to ``"r2u_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only explicit MANURE21 records use the bounded centered
    dried-manure scatter readout, preserving the R2n dark continuum while
    increasing spectral amplitude without global mean-lift constants.

    When set to ``"r2v_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only explicit MANURE21 records use the balanced centered
    dried-manure scatter readout, preserving the fixed continuum expectation
    while using bounded centered shape perturbations to reduce seed-dependent
    mean drift.
    """
    builder_config = prior_to_builder_config(
        record,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(builder_config["random_state"])
    concentrations = _sample_concentrations(record, rng, builder_config["n_samples"])

    effective_remediation_profile = (
        _effective_builder_remediation_profile(remediation_profile, record)
        if remediation_profile is not None
        else None
    )
    remediation_audit: dict[str, Any] | None = None
    if effective_remediation_profile is not None:
        concentrations, remediation_audit = _apply_r2c_concentration_remediation(
            concentrations,
            record=record,
            profile=effective_remediation_profile,
            seed=int(builder_config["random_state"]),
        )
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

    X_arr = np.asarray(X, dtype=float)
    if effective_remediation_profile is not None:
        assert remediation_audit is not None
        X_arr, remediation_audit = _apply_r2c_spectra_remediation(
            X_arr,
            wavelengths=wavelengths,
            audit=remediation_audit,
            record=record,
            profile=effective_remediation_profile,
            seed=int(builder_config["random_state"]),
        )
    profile_audit = _r2a_disabled_profile_audit()
    if mechanistic_profile is not None:
        X_arr, profile_audit = _apply_r2a_mechanistic_profile(
            X_arr,
            wavelengths,
            profile=mechanistic_profile,
            seed=int(builder_config["random_state"]),
        )

    latent_metadata = {
        "concentrations": concentrations,
        "component_keys": list(record.component_keys),
        "concentration_transform": builder_config["concentration_transform"],
        "batch_ids": generation_metadata.get("batch_ids"),
        "temperature_c": temperatures,
    }
    validation_summary = _validate_run_arrays(
        X=X_arr,
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
    metadata["r2a_mechanistic_profile"] = profile_audit
    if remediation_audit is not None:
        metadata["r2c_mechanistic_remediation"] = remediation_audit
    return SyntheticDatasetRun(
        X=X_arr,
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


_R2A_AUDIT_BASE: dict[str, Any] = {
    "scope": "bench_only_r2a_sentinel_mechanistic_ablation",
    "oracle": False,
    "label_inputs_used": False,
    "target_inputs_used": False,
    "split_inputs_used": False,
    "source_oracle_used": False,
    "learned": False,
    "real_stat_capture": False,
    "thresholds_modified": False,
    "metrics_modified": False,
    "imputed": False,
    "replays_real_rows": False,
}


def _r2a_disabled_profile_audit() -> dict[str, Any]:
    return {
        **_R2A_AUDIT_BASE,
        "enabled": False,
        "profile": None,
        "seed": None,
        "transform_params": {},
        "reason": "mechanistic_profile not requested",
    }


def _apply_r2a_mechanistic_profile(
    X: np.ndarray,
    wavelengths: np.ndarray,
    *,
    profile: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a fixed R2a mechanistic transform; return (X_transformed, audit)."""
    if profile not in R2A_MECHANISTIC_PROFILES:
        raise ValueError(
            f"unknown R2a mechanistic profile {profile!r}; "
            f"valid profiles are {list(R2A_MECHANISTIC_PROFILES)}"
        )
    profile_seed = _profile_seed(profile, seed)
    rng = np.random.default_rng(profile_seed)
    n_samples, n_wavelengths = X.shape
    wl = np.asarray(wavelengths, dtype=float)
    span = float(wl[-1] - wl[0]) if n_wavelengths > 1 else 1.0
    span = span if span > 0 else 1.0
    normalized_wl = (wl - wl[0]) / span

    transform_params: dict[str, Any]
    if profile == "r2a_baseline":
        X_out = X.copy()
        transform_params = {"effect": "identity_control"}
    elif profile == "r2a_pathlength_drift":
        # Per-sample multiplicative pathlength factor in [0.85, 1.15], deterministic.
        factors = np.asarray(rng.uniform(0.85, 1.15, size=n_samples), dtype=float)
        X_out = X * factors[:, None]
        transform_params = {
            "factor_min": float(factors.min()),
            "factor_max": float(factors.max()),
            "factor_distribution": "uniform_0p85_1p15",
        }
    elif profile == "r2a_baseline_curvature":
        # Per-sample additive quadratic baseline drift.
        amp_linear = np.asarray(rng.uniform(-0.02, 0.02, size=n_samples), dtype=float)
        amp_quad = np.asarray(rng.uniform(-0.01, 0.01, size=n_samples), dtype=float)
        baseline = (
            amp_linear[:, None] * normalized_wl[None, :]
            + amp_quad[:, None] * (normalized_wl[None, :] ** 2)
        )
        X_out = X + baseline
        transform_params = {
            "linear_amp_range": [-0.02, 0.02],
            "quadratic_amp_range": [-0.01, 0.01],
            "basis": "normalized_wavelength_quadratic",
        }
    elif profile == "r2a_emsc_like_scatter":
        # Multiplicative scatter + additive offset per sample, deterministic.
        scale = np.asarray(rng.uniform(0.9, 1.1, size=n_samples), dtype=float)
        offset = np.asarray(rng.uniform(-0.01, 0.01, size=n_samples), dtype=float)
        X_out = X * scale[:, None] + offset[:, None]
        transform_params = {
            "scale_range": [0.9, 1.1],
            "offset_range": [-0.01, 0.01],
        }
    elif profile == "r2a_instrumental_broadening":
        # Convolve each spectrum with a fixed Gaussian kernel (mechanistic ILS).
        fwhm_nm = 8.0
        median_step = float(np.median(np.diff(wl))) if n_wavelengths > 1 else 1.0
        median_step = median_step if median_step > 0 else 1.0
        sigma_bins = max(0.5, (fwhm_nm / 2.354820045) / median_step)
        kernel = _gaussian_kernel(sigma_bins)
        X_out = _convolve_rows(X, kernel)
        transform_params = {
            "fwhm_nm": fwhm_nm,
            "sigma_bins": float(sigma_bins),
            "kernel_size": int(kernel.size),
            "median_step_nm": median_step,
        }
    elif profile == "r2a_structured_noise":
        # Smooth correlated noise per sample, fixed amplitude.
        amp = 0.005
        raw_noise = rng.standard_normal(size=(n_samples, n_wavelengths)) * amp
        smooth_kernel = _gaussian_kernel(sigma_bins=2.0)
        smooth_noise = _convolve_rows(raw_noise, smooth_kernel)
        X_out = X + smooth_noise
        transform_params = {
            "amp": amp,
            "sigma_bins": 2.0,
            "structure": "gaussian_smoothed_iid_noise",
        }
    else:  # pragma: no cover - guarded by membership check above
        raise AssertionError(f"unhandled profile {profile!r}")

    audit = {
        **_R2A_AUDIT_BASE,
        "enabled": True,
        "profile": profile,
        "seed": int(profile_seed),
        "input_seed": int(seed),
        "n_samples": int(n_samples),
        "n_wavelengths": int(n_wavelengths),
        "wavelength_min": float(wl[0]) if n_wavelengths else None,
        "wavelength_max": float(wl[-1]) if n_wavelengths else None,
        "transform_params": transform_params,
        "reason": (
            "fixed mechanistic approximation determined by profile name and seed; "
            "no real data, labels, targets, splits, or AUC consulted"
        ),
    }
    return np.asarray(X_out, dtype=float), audit


def _profile_seed(profile: str, seed: int) -> int:
    digest = hashlib.sha256(f"{profile}|{int(seed)}".encode()).hexdigest()
    return int(digest[:8], 16)


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return np.array([1.0])
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    weights = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    weights /= float(weights.sum())
    return weights


def _convolve_rows(X: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if kernel.size <= 1 or X.shape[1] == 0:
        return X.copy()
    pad_left = kernel.size // 2
    pad_right = kernel.size - 1 - pad_left
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        padded = np.pad(X[i], (pad_left, pad_right), mode="edge")
        out[i] = np.convolve(padded, kernel, mode="valid")
    return out


_R2C_AUDIT_BASE: dict[str, Any] = {
    "scope": "bench_only_r2c_sentinel_matrix_remediation",
    "oracle": False,
    "label_inputs_used": False,
    "target_inputs_used": False,
    "split_inputs_used": False,
    "source_oracle_used": False,
    "learned": False,
    "real_stat_capture": False,
    "thresholds_modified": False,
    "metrics_modified": False,
    "imputed": False,
    "replays_real_rows": False,
}

# Tight Dirichlet alphas centered on textbook diesel composition (saturated
# alkanes dominant, aromatics secondary). Constants are mechanistic prior
# knowledge, not derived from real spectra.
_R2C_PETROCHEM_FUELS_DIESEL_ALPHAS: dict[str, float] = {
    "alkane": 8.0,
    "oil": 6.0,
    "aromatic": 2.0,
    "unsaturated_fat": 1.0,
    "methanol": 0.5,
    "ethanol": 0.5,
}
# Beer-Lambert short liquid optical path attenuation. The factor range models
# a transmission cuvette with a sub-millimetre path against a long path: the
# resulting absorbance is reduced multiplicatively in [0.45, 0.65].
_R2C_PETROCHEM_FUELS_PATH_FACTOR_RANGE: tuple[float, float] = (0.45, 0.65)

# Tight Dirichlet alphas centered on textbook bovine milk composition
# (water-dominant emulsion with secondary lipid/casein/lactose). Constants are
# mechanistic prior knowledge, not derived from real MILK spectra or labels.
_R2C_FOOD_DAIRY_EMULSION_ALPHAS: dict[str, float] = {
    "water": 18.0,
    "moisture": 18.0,
    "lipid": 1.2,
    "casein": 1.2,
    "lactose": 1.5,
    "protein": 0.9,
}
# Transflectance double-pass raw-intensity attenuation against an emulsion
# matrix. The factor range models a short reflective probe path: the resulting
# absorbance is scaled multiplicatively in [0.7, 0.9].
_R2C_FOOD_DAIRY_PATH_FACTOR_RANGE: tuple[float, float] = (0.7, 0.9)

_R2C_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    "petrochem_fuels": {
        "alphas": _R2C_PETROCHEM_FUELS_DIESEL_ALPHAS,
        "path_factor_range": _R2C_PETROCHEM_FUELS_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_diesel_centered",
        "spectra_rule": "short_liquid_optical_path_scale",
        "composition_source": "textbook_diesel_composition",
        "spectra_source": "beer_lambert_short_path",
    },
    "food_dairy": {
        "alphas": _R2C_FOOD_DAIRY_EMULSION_ALPHAS,
        "path_factor_range": _R2C_FOOD_DAIRY_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_milk_emulsion_centered",
        "spectra_rule": "transflectance_raw_intensity_scale",
        "composition_source": "textbook_dairy_emulsion_composition",
        "spectra_source": "double_pass_emulsion_attenuation",
    },
}

_R2D_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2d_sentinel_matrix_remediation",
}

_R2F_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2f_sentinel_matrix_remediation",
}

_R2G_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2g_sentinel_matrix_remediation",
}

_R2H_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2h_sentinel_matrix_remediation",
}

_R2I_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2i_sentinel_matrix_remediation",
}

_R2J_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2j_sentinel_matrix_remediation",
}

_R2K_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2k_sentinel_matrix_remediation",
}

_R2L_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2l_sentinel_matrix_remediation",
}

_R2M_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2m_sentinel_matrix_remediation",
}

_R2N_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2n_sentinel_matrix_remediation",
}

_R2O_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2o_sentinel_matrix_remediation",
}

_R2P_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2p_sentinel_matrix_remediation",
}

_R2Q_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2q_sentinel_matrix_remediation",
}

_R2R_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2r_sentinel_matrix_remediation",
}

_R2S_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2s_sentinel_matrix_remediation",
}

_R2T_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2t_sentinel_matrix_remediation",
}

_R2U_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2u_sentinel_matrix_remediation",
}

_R2V_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2v_sentinel_matrix_remediation",
}

_R2W_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2w_sentinel_matrix_remediation",
}

# Tight Dirichlet alphas centered on textbook beer-style beverage composition
# (water-dominant aqueous solution with secondary ethanol and reducing sugars,
# trace acids/tannins). Constants are mechanistic prior knowledge, not derived
# from real BEER spectra or labels.
_R2D_BEVERAGE_WINE_ALPHAS: dict[str, float] = {
    "water": 18.0,
    "ethanol": 1.5,
    "glucose": 1.0,
    "fructose": 1.0,
    "glycerol": 0.5,
    "tartaric_acid": 0.3,
    "malic_acid": 0.3,
    "tannins": 0.3,
}
# Long-pathlength transmittance through a beverage cuvette (typical 5-10 mm)
# vs. a short-path reference: by Beer-Lambert ``A = epsilon * c * l`` the
# resulting absorbance scales multiplicatively in [1.2, 1.6]. Strictly > 1.0
# so the BEER spectra audit can guard against accidental attenuation.
_R2D_BEVERAGE_WINE_PATH_FACTOR_RANGE: tuple[float, float] = (1.2, 1.6)

# Tight Dirichlet alphas centered on textbook corn-grain composition
# (starch-dominant kernel with secondary moisture/protein, minor lipid and
# fibre). Constants are mechanistic prior knowledge, not derived from real
# CORN spectra or labels.
_R2D_AGRICULTURE_GRAIN_ALPHAS: dict[str, float] = {
    "starch": 14.0,
    "moisture": 3.0,
    "protein": 2.0,
    "lipid": 1.0,
    "cellulose": 0.6,
    "gluten": 0.5,
    "hemicellulose": 1.2,
    "dietary_fiber": 0.5,
}
# Powder reflectance against a kernel of CORN: light multiplicative scatter in
# a tight band around 1.0 plus a fixed Gaussian instrumental smoothing kernel
# of FWHM 12 nm to attenuate over-sharp synthetic derivatives. Constants are
# mechanistic and bounded; never derived from real spectra.
_R2D_AGRICULTURE_GRAIN_PATH_FACTOR_RANGE: tuple[float, float] = (0.95, 1.10)
_R2D_AGRICULTURE_GRAIN_SMOOTHING_FWHM_NM: float = 12.0

_R2D_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2C_DOMAIN_RULES,
    "beverage_wine": {
        "alphas": _R2D_BEVERAGE_WINE_ALPHAS,
        "path_factor_range": _R2D_BEVERAGE_WINE_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_beer_centered",
        "spectra_rule": "long_liquid_optical_path_scale",
        "composition_source": "textbook_beer_composition",
        "spectra_source": "beer_lambert_long_path",
    },
    "agriculture_grain": {
        "alphas": _R2D_AGRICULTURE_GRAIN_ALPHAS,
        "path_factor_range": _R2D_AGRICULTURE_GRAIN_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_corn_grain_centered",
        "spectra_rule": "powder_reflectance_smoothing_and_scatter",
        "composition_source": "textbook_corn_grain_composition",
        "spectra_source": "instrumental_broadening_and_powder_scatter",
        "smoothing_fwhm_nm": _R2D_AGRICULTURE_GRAIN_SMOOTHING_FWHM_NM,
    },
}

# Tight Dirichlet alphas centered on clear fruit juice composition:
# water dominant, fruit sugars secondary, organic acids minor, carotenoid
# trace. Constants are mechanistic prior knowledge and intentionally exclude
# pectin/polyphenols/anthocyanin/tannins for the R2f juice rule.
_R2F_BEVERAGE_JUICE_ALPHAS: dict[str, float] = {
    "water": 24.0,
    "glucose": 1.5,
    "fructose": 1.8,
    "sucrose": 1.4,
    "citric_acid": 0.35,
    "malic_acid": 0.25,
    "carotenoid": 0.08,
}
# Moderate liquid cuvette pathlength scale for clear juice transmittance.
# Bounded and strictly > 1.0 by Beer-Lambert pathlength scaling, but narrower
# than the R2d beer long-path rule.
_R2F_BEVERAGE_JUICE_PATH_FACTOR_RANGE: tuple[float, float] = (1.05, 1.25)

_R2F_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2D_DOMAIN_RULES,
    "beverage_juice": {
        "alphas": _R2F_BEVERAGE_JUICE_ALPHAS,
        "path_factor_range": _R2F_BEVERAGE_JUICE_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_fruit_juice_centered",
        "spectra_rule": "moderate_liquid_cuvette_path_scale",
        "composition_source": "textbook_fruit_juice_composition",
        "spectra_source": "beer_lambert_moderate_cuvette_path",
    },
}

# Tight Dirichlet alphas centered on a mechanistic mineral-organic topsoil
# matrix. Components are exactly registered ``environmental_soil`` components;
# constants are prior chemistry/physics only, not captured from real soil
# spectra, labels, targets, splits, or morphology metrics.
_R2G_ENVIRONMENTAL_SOIL_ALPHAS: dict[str, float] = {
    "moisture": 2.5,
    "carbonates": 2.0,
    "kaolinite": 5.0,
    "gypsum": 1.2,
    "cellulose": 1.2,
    "lignin": 0.8,
    "protein": 0.6,
}
# Diffuse powder reflectance compression for topsoil: a fixed broad smoothing
# kernel attenuates over-sharp derivatives, then a bounded multiplicative
# path/scatter factor in [0.55, 0.75] compresses raw absorbance scale.
_R2G_ENVIRONMENTAL_SOIL_PATH_FACTOR_RANGE: tuple[float, float] = (0.55, 0.75)
_R2G_ENVIRONMENTAL_SOIL_SMOOTHING_FWHM_NM: float = 24.0

_R2G_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2F_DOMAIN_RULES,
    "environmental_soil": {
        "alphas": _R2G_ENVIRONMENTAL_SOIL_ALPHAS,
        "path_factor_range": _R2G_ENVIRONMENTAL_SOIL_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_mineral_organic_topsoil_centered",
        "spectra_rule": "diffuse_powder_smoothing_and_scatter_compression",
        "composition_source": "mechanistic_mineral_organic_topsoil_composition",
        "spectra_source": "diffuse_reflectance_powder_path_scatter_compression",
        "smoothing_fwhm_nm": _R2G_ENVIRONMENTAL_SOIL_SMOOTHING_FWHM_NM,
    },
}

# Cloudy berry juice near-IR bench readout. The composition remains limited to
# registered ``beverage_juice`` components; the spectra rule maps generated
# absorbance onto apparent percent-transmittance/intensity, matching raw
# instrument datasets whose Y axis is reported as percent or intensity instead
# of absorbance. These values are fixed mechanistic priors for a cloudy berry
# optical path and detector readout; they are not fitted to BERRY rows, PCA
# loadings, marginal/covariance statistics, adversarial AUC, morphology gaps,
# or gate thresholds.
_R2H_BERRY_JUICE_ALPHAS: dict[str, float] = {
    "water": 18.0,
    "glucose": 2.0,
    "fructose": 2.4,
    "sucrose": 1.2,
    "citric_acid": 0.7,
    "malic_acid": 0.7,
    "carotenoid": 0.2,
}
_R2H_BERRY_ABSORBANCE_PATH_FACTOR_RANGE: tuple[float, float] = (4.25, 4.75)
_R2H_BERRY_DETECTOR_BASELINE_PERCENT: float = 30.0
_R2H_BERRY_DETECTOR_DYNAMIC_PERCENT: float = 20.0
_R2H_BERRY_TURBIDITY_OFFSET_PERCENT_RANGE: tuple[float, float] = (-20.0, 20.0)

_R2H_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2G_DOMAIN_RULES,
    "beverage_juice": {
        "alphas": _R2H_BERRY_JUICE_ALPHAS,
        "composition_rule": "tight_dirichlet_cloudy_berry_juice_centered",
        "spectra_rule": "cloudy_berry_percent_transmittance_readout",
        "composition_source": "textbook_cloudy_berry_juice_composition",
        "spectra_source": "beer_lambert_percent_transmittance_with_turbidity",
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "apparent_percent_transmittance_intensity",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "absorbance_path_factor_range": _R2H_BERRY_ABSORBANCE_PATH_FACTOR_RANGE,
        "detector_baseline_percent": _R2H_BERRY_DETECTOR_BASELINE_PERCENT,
        "detector_dynamic_percent": _R2H_BERRY_DETECTOR_DYNAMIC_PERCENT,
        "turbidity_offset_percent_range": _R2H_BERRY_TURBIDITY_OFFSET_PERCENT_RANGE,
        "output_clip_percent": (0.0, 100.0),
    },
}

# Semi-solid strawberry/fruit puree short-wave NIR bench rule. This is separate
# from both R2f clear juice and R2h cloudy BERRY percent/intensity readouts:
# fruit puree is treated as a paste/tissue matrix with cell-wall solids, short
# effective path length, diffuse scatter compression, and a fixed raw
# pseudo-absorbance floor. Constants are prior optical assumptions only.
_R2I_FRUIT_PUREE_ALPHAS: dict[str, float] = {
    "water": 18.0,
    "glucose": 2.0,
    "fructose": 2.4,
    "sucrose": 1.0,
    "cellulose": 1.2,
    "starch": 0.4,
    "malic_acid": 0.55,
    "citric_acid": 0.45,
    "carotenoid": 0.12,
}
_R2I_FRUIT_PUREE_PATH_FACTOR_RANGE: tuple[float, float] = (0.12, 0.22)
_R2I_FRUIT_PUREE_SMOOTHING_FWHM_NM: float = 16.0
_R2I_FRUIT_PUREE_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (0.002, 0.008)

_R2I_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2G_DOMAIN_RULES,
    "agriculture_fruit": {
        "alphas": _R2I_FRUIT_PUREE_ALPHAS,
        "path_factor_range": _R2I_FRUIT_PUREE_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_semi_solid_fruit_puree_centered",
        "spectra_rule": "semi_solid_fruit_puree_short_path_scatter_smoothing",
        "composition_source": "textbook_fruit_puree_tissue_composition",
        "spectra_source": "semi_solid_paste_transflectance_scatter_compression",
        "smoothing_fwhm_nm": _R2I_FRUIT_PUREE_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2I_FRUIT_PUREE_BASELINE_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "semi_solid_puree_raw_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
    },
}

# Hydrocarbon fuel short-path transmission bench rule. The R2c path scale still
# leaves fuel spectra over-amplified in raw absorbance. R2j keeps the same
# diesel-centered composition but changes only the optical readout: a
# micro-path liquid transmission cell compresses absorbance by Beer-Lambert path
# length, then a small detector/dark-current floor is added. Constants are fixed
# instrument/optics priors and are not derived from real DIESEL spectra,
# labels, targets, splits, PCA/covariance/marginal summaries, morphology gaps,
# adversarial AUC, thresholds, or downstream feedback.
_R2J_PETROCHEM_FUELS_MICRO_PATH_FACTOR_RANGE: tuple[float, float] = (0.03, 0.05)
_R2J_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.0005,
    0.002,
)

_R2J_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2I_DOMAIN_RULES,
    "petrochem_fuels": {
        "alphas": _R2C_PETROCHEM_FUELS_DIESEL_ALPHAS,
        "path_factor_range": _R2J_PETROCHEM_FUELS_MICRO_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_diesel_centered",
        "spectra_rule": "micro_path_fuel_transmission_absorbance_floor",
        "composition_source": "textbook_diesel_composition",
        "spectra_source": "beer_lambert_micro_path_with_detector_floor",
        "additive_baseline_range": _R2J_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "micro_path_raw_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
    },
}

# Hydrocarbon fuel R2k rule. R2j's uniform micro-path factor preserves the mean
# readout hypothesis but multiplies every spectral derivative by the same small
# factor; the detector floor adds no derivative at all. R2k keeps a micro-path
# continuum, then reintroduces fixed CH overtone/residual contrast from the
# generated fuel absorbance itself. This is still a mechanistic transformation
# of synthetic spectra only: no real spectra/statistics or thresholds are read.
_R2K_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (0.055, 0.085)
_R2K_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE: tuple[float, float] = (0.24, 0.34)
_R2K_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.0005,
    0.002,
)
_R2K_PETROCHEM_FUELS_CONTINUUM_SMOOTHING_FWHM_NM: float = 96.0
_R2K_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
    1720.0,
)
_R2K_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM: float = 34.0
_R2K_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.10, 0.18)

_R2K_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2I_DOMAIN_RULES,
    "petrochem_fuels": {
        "alphas": _R2C_PETROCHEM_FUELS_DIESEL_ALPHAS,
        "path_factor_range": _R2K_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "feature_contrast_range": _R2K_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE,
        "composition_rule": "tight_dirichlet_diesel_centered",
        "spectra_rule": "micro_path_fuel_ch_overtone_contrast_readout",
        "composition_source": "textbook_diesel_composition",
        "spectra_source": "beer_lambert_micro_path_with_fixed_ch_overtone_contrast",
        "continuum_smoothing_fwhm_nm": _R2K_PETROCHEM_FUELS_CONTINUUM_SMOOTHING_FWHM_NM,
        "ch_overtone_centers_nm": _R2K_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
        "ch_overtone_width_nm": _R2K_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R2K_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "additive_baseline_range": _R2K_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "micro_path_ch_overtone_raw_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "contrast_source": "fixed_hydrocarbon_ch_overtone_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

# LUCAS-style raw soil spectra in the sentinel set remain mean-shift dominated
# under R2g because diffuse powder smoothing/compression exposes only a low
# pseudo-absorbance residual. R2l adds a fixed mineral-albedo apparent
# absorbance floor: A=-log10(0.5)=0.30103 for a 50% mineral diffuse-reflectance
# continuum, then keeps a narrow compressed residual path. This is a prior on
# the optical readout, not a fit to LUCAS marginal means or thresholds.
_R2L_LUCAS_SOIL_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.20, 0.25)
_R2L_LUCAS_SOIL_SMOOTHING_FWHM_NM: float = 24.0
_R2L_LUCAS_SOIL_MINERAL_ALBEDO_ABSORBANCE_BASELINE: float = 0.30103

_R2L_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2K_DOMAIN_RULES,
    "environmental_soil": {
        "alphas": _R2G_ENVIRONMENTAL_SOIL_ALPHAS,
        "path_factor_range": _R2L_LUCAS_SOIL_RESIDUAL_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_mineral_organic_topsoil_centered",
        "spectra_rule": "lucas_mineral_albedo_absorbance_floor_scatter_readout",
        "composition_source": "mechanistic_mineral_organic_topsoil_composition",
        "spectra_source": "fixed_mineral_albedo_floor_plus_diffuse_scatter_residual",
        "smoothing_fwhm_nm": _R2L_LUCAS_SOIL_SMOOTHING_FWHM_NM,
        "additive_baseline_range": (
            _R2L_LUCAS_SOIL_MINERAL_ALBEDO_ABSORBANCE_BASELINE,
            _R2L_LUCAS_SOIL_MINERAL_ALBEDO_ABSORBANCE_BASELINE,
        ),
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "lucas_raw_soil_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "baseline_source": "mineral_albedo_A_equals_minus_log10_0p5",
    },
}

# Raw MILK spectra in the current sentinel set behave like apparent
# transflectance/reflectance intensity: water bands around 1450/1940 nm are
# valleys in the raw curve, not absorbance peaks. R2m keeps the R2c textbook
# emulsion composition but exposes generated absorbance through a fixed
# fat-globule scatter continuum and Beer-Lambert inverse readout. The compact
# full-range variant is selected only when the audit route marker records a
# full-range milk readout row; it uses a lower detector gain because the 1940 nm
# water band is in range.
_R2M_MILK_SMOOTHING_FWHM_NM: float = 18.0
_R2M_MILK_PATH_FACTOR_RANGE: tuple[float, float] = (0.55, 0.85)
_R2M_MILK_SCATTER_EXPONENT_RANGE: tuple[float, float] = (1.0, 1.6)
_R2M_MILK_DETECTOR_OFFSET_RANGE: tuple[float, float] = (0.0, 0.04)
_R2M_MILK_SHORTWAVE_DETECTOR_DYNAMIC_RANGE: tuple[float, float] = (1.8, 2.6)
_R2M_MILK_FULLRANGE_DETECTOR_DYNAMIC_RANGE: tuple[float, float] = (1.0, 1.8)

_R2M_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2L_DOMAIN_RULES,
    "food_dairy": {
        "alphas": _R2C_FOOD_DAIRY_EMULSION_ALPHAS,
        "path_factor_range": _R2M_MILK_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_milk_emulsion_centered",
        "spectra_rule": "milk_emulsion_scatter_inverse_transflectance_readout",
        "composition_source": "textbook_dairy_emulsion_composition",
        "spectra_source": "fat_globule_scatter_inverse_beer_lambert_transflectance",
        "smoothing_fwhm_nm": _R2M_MILK_SMOOTHING_FWHM_NM,
        "scatter_exponent_range": _R2M_MILK_SCATTER_EXPONENT_RANGE,
        "detector_offset_range": _R2M_MILK_DETECTOR_OFFSET_RANGE,
        "detector_dynamic_range": _R2M_MILK_SHORTWAVE_DETECTOR_DYNAMIC_RANGE,
        "fullrange_detector_dynamic_range": _R2M_MILK_FULLRANGE_DETECTOR_DYNAMIC_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "milk_raw_transflectance_intensity",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "scatter_source": "fixed_fat_globule_mie_scatter_prior",
        "output_clip_intensity": (0.0, 6.0),
        "fullrange_output_clip_intensity": (0.0, 3.0),
    },
}

# Dried/ground manure is neither grain nor LUCAS-style mineral topsoil. The
# matrix is an organic-mineral mixture: residual moisture, cellulose/lignin and
# protein from bedding/fecal organic matter, plus ash/mineral fractions. Its raw
# diffuse reflectance readout should carry a darker organic continuum than
# mineral topsoil while preserving bounded organic/mineral residual structure.
_R2N_MANURE21_ALPHAS: dict[str, float] = {
    "moisture": 1.6,
    "cellulose": 3.2,
    "lignin": 2.4,
    "protein": 2.2,
    "carbonates": 1.8,
    "gypsum": 1.1,
    "kaolinite": 0.9,
}
_R2N_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.30, 0.42)
_R2N_MANURE21_SMOOTHING_FWHM_NM: float = 20.0
_R2N_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (0.60, 0.78)

_R2N_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2M_DOMAIN_RULES,
    "environmental_soil": {
        "alphas": _R2N_MANURE21_ALPHAS,
        "path_factor_range": _R2N_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_dried_manure_organic_mineral_centered",
        "spectra_rule": "dried_manure_organic_mineral_albedo_scatter_readout",
        "composition_source": "textbook_dried_manure_organic_mineral_composition",
        "spectra_source": "fixed_dark_organic_albedo_plus_diffuse_scatter_residual",
        "smoothing_fwhm_nm": _R2N_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2N_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "dried_ground_manure_raw_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_manure21_route",
        "matrix_source": "fixed_dried_manure_organic_mineral_prior",
        "scatter_source": "fixed_powder_diffuse_scatter_prior",
        "albedo_source": "fixed_dark_organic_reflectance_prior",
    },
}

# Beer OriginalExtract rows are liquid fermented beverages, not berry juice or
# fruit puree. R2d's long-path absorbance scale leaves them mean-shift
# dominated because the synthetic route has no broadband optical attenuation
# from a turbid/carbonated beer cuvette. R2o keeps the beer aqueous
# ethanol/extract composition but exposes generated absorbance through a fixed
# apparent raw absorbance readout: long Beer-Lambert path, haze/scattering
# attenuation, and a small carbonated-liquid broadband residual. Constants are
# fixed optical priors only; they are not fitted to BEER spectra or metrics.
_R2O_BEER_TURBID_PATH_FACTOR_RANGE: tuple[float, float] = (1.75, 2.35)
_R2O_BEER_HAZE_ABSORBANCE_BASELINE_RANGE: tuple[float, float] = (1.75, 2.15)
_R2O_BEER_HAZE_SLOPE_ABSORBANCE_RANGE: tuple[float, float] = (0.06, 0.18)
_R2O_BEER_CARBONATION_RESIDUAL_RANGE: tuple[float, float] = (0.00, 0.05)
_R2O_BEER_SMOOTHING_FWHM_NM: float = 10.0

_R2O_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2N_DOMAIN_RULES,
    "beverage_wine": {
        "alphas": _R2D_BEVERAGE_WINE_ALPHAS,
        "path_factor_range": _R2O_BEER_TURBID_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_beer_centered",
        "spectra_rule": "fermented_beer_turbid_cuvette_absorbance_readout",
        "composition_source": "textbook_beer_composition",
        "spectra_source": "beer_lambert_long_path_with_fixed_haze_carbonation",
        "smoothing_fwhm_nm": _R2O_BEER_SMOOTHING_FWHM_NM,
        "haze_absorbance_baseline_range": _R2O_BEER_HAZE_ABSORBANCE_BASELINE_RANGE,
        "haze_slope_absorbance_range": _R2O_BEER_HAZE_SLOPE_ABSORBANCE_RANGE,
        "carbonation_residual_absorbance_range": _R2O_BEER_CARBONATION_RESIDUAL_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "fermented_beer_raw_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_beer_route",
        "matrix_source": "fixed_fermented_beer_water_ethanol_extract_prior",
        "scatter_source": "fixed_turbid_carbonated_liquid_cuvette_prior",
        "output_clip_absorbance": (0.0, 5.0),
    },
}

# PHOSPHORUS rows are mineral soil/fertilizer/powder readouts rather than the
# broad LUCAS topsoil population. R2g leaves them mean-shift dominated and
# mean-curve inverted because the synthetic absorbance residual is exposed
# directly. R2p keeps the mineral-organic soil composition but uses a fixed
# phosphate/clay albedo continuum: A=-log10(0.63)=0.20066. Centered synthetic
# absorbance residuals are subtracted from that continuum, as stronger
# phosphate/mineral absorption lowers apparent diffuse albedo. The continuum
# and residual scale are fixed optical priors, not fitted to PHOSPHORUS rows.
_R2P_PHOSPHORUS_RESIDUAL_SCALE_RANGE: tuple[float, float] = (0.95, 1.05)
_R2P_PHOSPHORUS_SMOOTHING_FWHM_NM: float = 24.0
_R2P_PHOSPHORUS_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (0.195, 0.210)

_R2P_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2O_DOMAIN_RULES,
    "environmental_soil": {
        "alphas": _R2G_ENVIRONMENTAL_SOIL_ALPHAS,
        "path_factor_range": _R2P_PHOSPHORUS_RESIDUAL_SCALE_RANGE,
        "composition_rule": "tight_dirichlet_mineral_organic_topsoil_centered",
        "spectra_rule": "phosphorus_mineral_fertilizer_albedo_residual_readout",
        "composition_source": "mechanistic_mineral_organic_topsoil_composition",
        "spectra_source": "fixed_phosphate_mineral_albedo_plus_inverted_residual",
        "smoothing_fwhm_nm": _R2P_PHOSPHORUS_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2P_PHOSPHORUS_ALBEDO_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "phosphorus_raw_mineral_soil_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_phosphorus_route",
        "matrix_source": "fixed_phosphate_mineral_soil_prior",
        "scatter_source": "fixed_mineral_powder_diffuse_albedo_prior",
        "albedo_source": "phosphate_mineral_albedo_A_equals_minus_log10_0p63",
        "output_clip_absorbance": (0.0, None),
    },
}

# LUCAS pH Organic is a humic/organic-soil subset, not the broad mineral LUCAS
# topsoil population used by R2l. Organic topsoil has a darker diffuse albedo
# continuum from humic matter, higher bound-water/OH contribution, and reduced
# mineral carbonate dominance. R2q uses fixed priors for that route only:
# reflectance around 35-40% (A about 0.40-0.46), a small diffuse residual, and
# broad weak OH bands. These constants are not fitted to LUCAS spectra or mean
# morphology gaps.
_R2Q_LUCAS_PH_ORGANIC_ALPHAS: dict[str, float] = {
    "moisture": 2.6,
    "cellulose": 2.4,
    "lignin": 3.2,
    "protein": 1.4,
    "carbonates": 0.8,
    "kaolinite": 1.1,
    "gypsum": 0.5,
}
_R2Q_LUCAS_PH_ORGANIC_RESIDUAL_SCALE_RANGE: tuple[float, float] = (0.22, 0.32)
_R2Q_LUCAS_PH_ORGANIC_SMOOTHING_FWHM_NM: float = 24.0
_R2Q_LUCAS_PH_ORGANIC_HUMIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.405,
    0.455,
)
_R2Q_LUCAS_PH_ORGANIC_HUMIC_SLOPE_RANGE: tuple[float, float] = (0.015, 0.045)
_R2Q_LUCAS_PH_ORGANIC_OH_BAND_ABSORBANCE_RANGE: tuple[float, float] = (
    0.005,
    0.025,
)

_R2Q_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2P_DOMAIN_RULES,
    "environmental_soil": {
        "alphas": _R2Q_LUCAS_PH_ORGANIC_ALPHAS,
        "path_factor_range": _R2Q_LUCAS_PH_ORGANIC_RESIDUAL_SCALE_RANGE,
        "composition_rule": "tight_dirichlet_lucas_ph_organic_topsoil_centered",
        "spectra_rule": "lucas_ph_organic_humic_albedo_oh_readout",
        "composition_source": "fixed_lucas_ph_organic_humic_topsoil_composition",
        "spectra_source": "fixed_humic_dark_albedo_plus_oh_residual_readout",
        "smoothing_fwhm_nm": _R2Q_LUCAS_PH_ORGANIC_SMOOTHING_FWHM_NM,
        "additive_baseline_range": (
            _R2Q_LUCAS_PH_ORGANIC_HUMIC_ALBEDO_ABSORBANCE_RANGE
        ),
        "humic_slope_absorbance_range": _R2Q_LUCAS_PH_ORGANIC_HUMIC_SLOPE_RANGE,
        "oh_band_absorbance_range": _R2Q_LUCAS_PH_ORGANIC_OH_BAND_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "lucas_ph_organic_raw_soil_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_lucas_ph_organic_route",
        "matrix_source": "fixed_lucas_humic_organic_topsoil_prior",
        "scatter_source": "fixed_organic_topsoil_diffuse_scatter_prior",
        "albedo_source": "humic_organic_topsoil_albedo_A_approx_minus_log10_0p35_to_0p40",
        "oh_band_centers_nm": (1450.0, 1940.0),
        "oh_band_width_nm": 70.0,
        "output_clip_absorbance": (0.0, None),
    },
}

# Strawberry puree is a semi-solid cellular paste, not clear juice and not the
# BERRY percent/intensity readout. R2r keeps the registered fruit components
# but shifts the raw readout toward a short-range transflectance residual:
# water/sugar absorption is carried as fixed broad bands, while the high-albedo
# tissue/seed/skin scatter residual is compressed and sign-inverted relative
# to exposed absorbance. Constants are fixed optical priors, not real-stat
# capture or calibration.
_R2R_FRUIT_PUREE_ALPHAS: dict[str, float] = {
    "water": 14.0,
    "glucose": 2.2,
    "fructose": 2.6,
    "sucrose": 1.1,
    "cellulose": 2.4,
    "starch": 0.8,
    "malic_acid": 0.65,
    "citric_acid": 0.5,
    "carotenoid": 0.18,
}
_R2R_FRUIT_PUREE_RESIDUAL_SCALE_RANGE: tuple[float, float] = (0.045, 0.075)
_R2R_FRUIT_PUREE_SMOOTHING_FWHM_NM: float = 22.0
_R2R_FRUIT_PUREE_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (0.006, 0.009)
_R2R_FRUIT_PUREE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.0015, 0.0015)
_R2R_FRUIT_PUREE_WATER_BAND_RANGE: tuple[float, float] = (0.0010, 0.0025)
_R2R_FRUIT_PUREE_SUGAR_SOLIDS_BAND_RANGE: tuple[float, float] = (0.0003, 0.0012)

_R2R_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2Q_DOMAIN_RULES,
    "agriculture_fruit": {
        "alphas": _R2R_FRUIT_PUREE_ALPHAS,
        "path_factor_range": _R2R_FRUIT_PUREE_RESIDUAL_SCALE_RANGE,
        "composition_rule": "tight_dirichlet_strawberry_puree_cellular_matrix",
        "spectra_rule": "strawberry_puree_transflectance_residual_readout",
        "composition_source": "textbook_strawberry_puree_cellular_composition",
        "spectra_source": "fixed_puree_transflectance_albedo_residual_readout",
        "smoothing_fwhm_nm": _R2R_FRUIT_PUREE_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2R_FRUIT_PUREE_BASELINE_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2R_FRUIT_PUREE_SCATTER_SLOPE_RANGE,
        "water_band_absorbance_range": _R2R_FRUIT_PUREE_WATER_BAND_RANGE,
        "sugar_solids_band_absorbance_range": (
            _R2R_FRUIT_PUREE_SUGAR_SOLIDS_BAND_RANGE
        ),
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "strawberry_puree_raw_transflectance_residual_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_fruitpuree_route",
        "matrix_source": "fixed_semi_solid_strawberry_puree_prior",
        "scatter_source": "fixed_seed_skin_pectin_cellular_scatter_prior",
        "water_band_center_nm": 970.0,
        "water_band_width_nm": 42.0,
        "sugar_solids_band_center_nm": 1040.0,
        "sugar_solids_band_width_nm": 36.0,
        "output_clip_absorbance": (0.0, 0.035),
    },
}

# R2s keeps the same synthetic diesel chemistry and CH-overtone residual
# readout family as R2k, but treats the smooth continuum as a shorter,
# blank-referenced micro-cell path. This lowers broadband raw absorbance while
# a slightly larger fixed CH path perturbation preserves derivative contrast.
_R2S_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.030,
    0.045,
)
_R2S_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.12, 0.20)

_R2S_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2R_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R2K_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R2S_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "ch_overtone_gain_range": _R2S_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "spectra_source": (
            "beer_lambert_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"
        ),
        "readout_space": "blank_referenced_micro_path_ch_overtone_raw_absorbance",
        "contrast_source": "fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

# R2t manure matrix variability rule. R2n fixed the main readout level by using
# a dark organic/mineral albedo plus a compressed residual, but MANURE21 stayed
# amplitude-under because dried manure is patchy at the sample cup scale:
# bedding/fecal organics, ash/mineral grains, moisture pockets, and particle
# size all perturb diffuse reflectance over broad bands. These ranges are fixed
# physical priors; they are not estimated from MANURE21 spectra or metrics.
_R2T_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.36, 0.54)
_R2T_MANURE21_SMOOTHING_FWHM_NM: float = 14.0
_R2T_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (0.66, 0.90)
_R2T_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.26, 0.26)
_R2T_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.12)
_R2T_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.10)
_R2T_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.09, 0.09)

_R2T_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2T_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_heterogeneous_scatter_patch_readout",
        "spectra_source": (
            "fixed_dark_organic_albedo_plus_particle_scatter_moisture_mineral_lumps"
        ),
        "smoothing_fwhm_nm": _R2T_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2T_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2T_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2T_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2T_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2T_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "heterogeneity_source": (
            "fixed_dried_manure_particle_size_moisture_organic_mineral_lump_prior"
        ),
        "scatter_source": "fixed_dried_manure_particle_size_diffuse_scatter_prior",
        "albedo_source": "fixed_dark_organic_mineral_patch_reflectance_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2u keeps R2n's dark organic/mineral continuum range. The extra MANURE21
# variability is carried by centered residual and broad-band scatter terms, so
# the expected row mean stays anchored to the R2n albedo instead of the higher
# R2t continuum.
_R2U_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.52, 0.70)
_R2U_MANURE21_SMOOTHING_FWHM_NM: float = 16.0
_R2U_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.74,
    0.86,
)
_R2U_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.18, 0.18)
_R2U_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.085)
_R2U_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.075)
_R2U_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.06, 0.06)

_R2U_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2U_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_bounded_centered_scatter_readout",
        "spectra_source": (
            "fixed_dark_organic_albedo_plus_centered_particle_scatter_bands"
        ),
        "smoothing_fwhm_nm": _R2U_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2U_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2U_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2U_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2U_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2U_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "heterogeneity_source": (
            "fixed_dried_manure_centered_particle_moisture_mineral_scatter_prior"
        ),
        "scatter_source": "fixed_dried_manure_bounded_centered_scatter_prior",
        "albedo_source": "fixed_dark_organic_reflectance_prior_r2n_continuum",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2v keeps the R2u continuum expectation but removes seed-dependent continuum
# drift by using balanced bounded draws for explicit MANURE21 rows. Additional
# broad-band perturbations are centered spectral shapes only; they increase
# MANURE matrix variance/amplitude without row-uniform lift/downshift.
_R2V_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.60, 0.76)
_R2V_MANURE21_SMOOTHING_FWHM_NM: float = 14.0
_R2V_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.74,
    0.86,
)
_R2V_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.16, 0.16)
_R2V_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.105)
_R2V_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.095)
_R2V_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.075, 0.075)

_R2V_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2V_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_balanced_centered_scatter_readout",
        "spectra_source": (
            "fixed_dark_organic_albedo_plus_balanced_centered_particle_scatter_bands"
        ),
        "smoothing_fwhm_nm": _R2V_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2V_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2V_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2V_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2V_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2V_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": (
            "fixed_dried_manure_balanced_centered_particle_moisture_mineral_scatter_prior"
        ),
        "scatter_source": "fixed_dried_manure_balanced_centered_scatter_prior",
        "albedo_source": "fixed_dark_organic_reflectance_prior_r2n_continuum",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2w keeps R2v's balanced centering but transfers more MANURE21 cup-scale
# heterogeneity through a wider fixed albedo prior and slightly stronger broad
# bands. Constants are fixed dried-manure physics priors, not real-stat capture.
_R2W_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.80, 1.00)
_R2W_MANURE21_SMOOTHING_FWHM_NM: float = 14.0
_R2W_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.72,
    1.00,
)
_R2W_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.16, 0.16)
_R2W_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.14)
_R2W_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.13)
_R2W_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.10, 0.10)

_R2W_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2W_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_albedo_variance_centered_scatter_readout",
        "spectra_source": (
            "fixed_dark_organic_albedo_variance_plus_balanced_centered_particle_scatter_bands"
        ),
        "smoothing_fwhm_nm": _R2W_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2W_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2W_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2W_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2W_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2W_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": (
            "fixed_dried_manure_albedo_variance_centered_particle_moisture_mineral_scatter_prior"
        ),
        "scatter_source": "fixed_dried_manure_balanced_centered_scatter_prior",
        "albedo_source": "fixed_wide_dark_organic_mineral_albedo_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

_PROFILE_DOMAIN_RULES: dict[str, dict[str, dict[str, Any]]] = {
    "r2c_sentinel_matrix_v1": _R2C_DOMAIN_RULES,
    "r2d_sentinel_matrix_v1": _R2D_DOMAIN_RULES,
    "r2f_sentinel_matrix_v1": _R2F_DOMAIN_RULES,
    "r2g_sentinel_matrix_v1": _R2G_DOMAIN_RULES,
    "r2h_sentinel_matrix_v1": _R2H_DOMAIN_RULES,
    "r2i_sentinel_matrix_v1": _R2I_DOMAIN_RULES,
    "r2j_sentinel_matrix_v1": _R2J_DOMAIN_RULES,
    "r2k_sentinel_matrix_v1": _R2K_DOMAIN_RULES,
    "r2l_sentinel_matrix_v1": _R2L_DOMAIN_RULES,
    "r2m_sentinel_matrix_v1": _R2M_DOMAIN_RULES,
    "r2n_sentinel_matrix_v1": _R2N_DOMAIN_RULES,
    "r2o_sentinel_matrix_v1": _R2O_DOMAIN_RULES,
    "r2p_sentinel_matrix_v1": _R2P_DOMAIN_RULES,
    "r2q_sentinel_matrix_v1": _R2Q_DOMAIN_RULES,
    "r2r_sentinel_matrix_v1": _R2R_DOMAIN_RULES,
    "r2s_sentinel_matrix_v1": _R2S_DOMAIN_RULES,
    "r2t_sentinel_matrix_v1": _R2T_DOMAIN_RULES,
    "r2u_sentinel_matrix_v1": _R2U_DOMAIN_RULES,
    "r2v_sentinel_matrix_v1": _R2V_DOMAIN_RULES,
    "r2w_sentinel_matrix_v1": _R2W_DOMAIN_RULES,
}


def _validate_r2c_profile(profile: str) -> None:
    if profile not in ALL_REMEDIATION_PROFILES:
        raise ValueError(
            f"unknown remediation profile {profile!r}; "
            f"valid profiles are {list(ALL_REMEDIATION_PROFILES)}"
        )


def _is_r2l_lucas_soil_record(record: PriorConfigRecord) -> bool:
    route = record.source_prior_config.get("_r2l_lucas_soil_route")
    if not isinstance(route, dict):
        return False
    return (
        route.get("enabled") is True
        and route.get("route_marker") == "lucas"
        and route.get("non_oracle") is True
        and route.get("real_stat_capture") is False
    )


def _r2m_milk_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2m_milk_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "milk"
        and route.get("non_oracle") is True
        and route.get("real_stat_capture") is False
    ):
        return route
    return None


def _is_r2m_milk_record(record: PriorConfigRecord) -> bool:
    return _r2m_milk_route(record) is not None


def _r2m_milk_readout_variant(record: PriorConfigRecord) -> str:
    route = _r2m_milk_route(record)
    if route is not None and route.get("variant") == "fullrange":
        return "fullrange"
    return "shortwave"


def _r2n_manure21_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2n_manure21_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "manure21"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2n_manure21_record(record: PriorConfigRecord) -> bool:
    return _r2n_manure21_route(record) is not None


def _r2o_beer_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2o_beer_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "beer"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2o_beer_record(record: PriorConfigRecord) -> bool:
    return _r2o_beer_route(record) is not None


def _r2p_phosphorus_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2p_phosphorus_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "phosphorus"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2p_phosphorus_record(record: PriorConfigRecord) -> bool:
    return _r2p_phosphorus_route(record) is not None


def _r2q_lucas_ph_organic_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2q_lucas_ph_organic_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "lucas_ph_organic"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2q_lucas_ph_organic_record(record: PriorConfigRecord) -> bool:
    return _r2q_lucas_ph_organic_route(record) is not None


def _r2r_fruitpuree_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2r_fruitpuree_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "fruitpuree"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2r_fruitpuree_record(record: PriorConfigRecord) -> bool:
    return _r2r_fruitpuree_route(record) is not None


def _r2s_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2s_diesel_readout_route")
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "diesel"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r2s_diesel_record(record: PriorConfigRecord) -> bool:
    return _r2s_diesel_route(record) is not None


def _effective_builder_remediation_profile(
    profile: str,
    record: PriorConfigRecord,
) -> str:
    _validate_r2c_profile(profile)
    domain_key = record.domain_key
    if profile in R2W_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2V_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2U_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2T_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2S_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r2s_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2r_sentinel_matrix_v1",
            record,
        )
    if profile in R2R_REMEDIATION_PROFILES:
        if domain_key == "agriculture_fruit" and _is_r2r_fruitpuree_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2q_sentinel_matrix_v1",
            record,
        )
    if profile in R2Q_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2q_lucas_ph_organic_record(
            record
        ):
            return profile
        return _effective_builder_remediation_profile(
            "r2p_sentinel_matrix_v1",
            record,
        )
    if profile in R2P_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2p_phosphorus_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2o_sentinel_matrix_v1",
            record,
        )
    if profile in R2O_REMEDIATION_PROFILES:
        if domain_key == "beverage_wine" and _is_r2o_beer_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2n_sentinel_matrix_v1",
            record,
        )
    if profile in R2N_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2m_sentinel_matrix_v1",
            record,
        )
    if profile in R2M_REMEDIATION_PROFILES:
        if domain_key == "food_dairy" and _is_r2m_milk_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2l_sentinel_matrix_v1",
            record,
        )
    if profile in R2L_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2l_lucas_soil_record(record):
            return profile
        if domain_key == "environmental_soil":
            return "r2g_sentinel_matrix_v1"
        if domain_key == "petrochem_fuels":
            return "r2k_sentinel_matrix_v1"
        return "r2i_sentinel_matrix_v1"
    if (
        profile in R2J_REMEDIATION_PROFILES + R2K_REMEDIATION_PROFILES
        and domain_key != "petrochem_fuels"
    ):
        return "r2i_sentinel_matrix_v1"
    return profile


def _audit_base_for(profile: str) -> dict[str, Any]:
    if profile in R2W_REMEDIATION_PROFILES:
        return _R2W_AUDIT_BASE
    if profile in R2V_REMEDIATION_PROFILES:
        return _R2V_AUDIT_BASE
    if profile in R2U_REMEDIATION_PROFILES:
        return _R2U_AUDIT_BASE
    if profile in R2T_REMEDIATION_PROFILES:
        return _R2T_AUDIT_BASE
    if profile in R2S_REMEDIATION_PROFILES:
        return _R2S_AUDIT_BASE
    if profile in R2R_REMEDIATION_PROFILES:
        return _R2R_AUDIT_BASE
    if profile in R2Q_REMEDIATION_PROFILES:
        return _R2Q_AUDIT_BASE
    if profile in R2P_REMEDIATION_PROFILES:
        return _R2P_AUDIT_BASE
    if profile in R2O_REMEDIATION_PROFILES:
        return _R2O_AUDIT_BASE
    if profile in R2N_REMEDIATION_PROFILES:
        return _R2N_AUDIT_BASE
    if profile in R2M_REMEDIATION_PROFILES:
        return _R2M_AUDIT_BASE
    if profile in R2L_REMEDIATION_PROFILES:
        return _R2L_AUDIT_BASE
    if profile in R2K_REMEDIATION_PROFILES:
        return _R2K_AUDIT_BASE
    if profile in R2J_REMEDIATION_PROFILES:
        return _R2J_AUDIT_BASE
    if profile in R2I_REMEDIATION_PROFILES:
        return _R2I_AUDIT_BASE
    if profile in R2H_REMEDIATION_PROFILES:
        return _R2H_AUDIT_BASE
    if profile in R2G_REMEDIATION_PROFILES:
        return _R2G_AUDIT_BASE
    if profile in R2F_REMEDIATION_PROFILES:
        return _R2F_AUDIT_BASE
    if profile in R2D_REMEDIATION_PROFILES:
        return _R2D_AUDIT_BASE
    return _R2C_AUDIT_BASE


def _domain_rule_for(profile: str, domain_key: str) -> dict[str, Any] | None:
    rule_table = _PROFILE_DOMAIN_RULES.get(profile)
    if rule_table is None:
        return None
    return rule_table.get(domain_key)


def _apply_r2c_concentration_remediation(
    concentrations: np.ndarray,
    *,
    record: PriorConfigRecord,
    profile: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-bias composition with a domain-specific tight Dirichlet (opt-in).

    Supports the R2c rule table (DIESEL/MILK), the R2d superset
    (DIESEL/MILK + BEER/CORN), the R2f juice superset, the R2g soil
    superset, and the R2h cloudy-berry readout selected by ``profile``.
    """
    _validate_r2c_profile(profile)
    domain_key = record.domain_key
    profile_seed = _profile_seed(f"r2c_concentrations:{profile}", seed)
    audit_base: dict[str, Any] = {
        **_audit_base_for(profile),
        "enabled": True,
        "profile": profile,
        "domain_key": domain_key,
        "input_seed": int(seed),
        "concentration_seed": int(profile_seed),
        "applied_to_concentrations": False,
        "applied_to_spectra": False,
        "transform_params": {},
    }
    rule = _domain_rule_for(profile, domain_key)
    if rule is None:
        audit_base["reason"] = (
            f"domain {domain_key!r} has no remediation rule under profile "
            f"{profile!r}; concentrations unchanged"
        )
        return concentrations, audit_base

    component_keys = list(record.component_keys)
    alpha_table: dict[str, float] = rule["alphas"]
    alphas = np.array(
        [float(alpha_table.get(key, 1.0)) for key in component_keys],
        dtype=float,
    )
    rng = np.random.default_rng(profile_seed)
    new_concentrations = np.asarray(
        rng.dirichlet(alphas, size=concentrations.shape[0]),
        dtype=float,
    )
    audit_base["applied_to_concentrations"] = True
    audit_base["transform_params"] = {
        "composition_rule": rule["composition_rule"],
        "composition_source": rule["composition_source"],
        "alphas": {key: float(alpha) for key, alpha in zip(component_keys, alphas, strict=True)},
        "alpha_sum": float(alphas.sum()),
        "n_samples": int(concentrations.shape[0]),
        "n_components": int(len(component_keys)),
    }
    audit_base["reason"] = "domain_centered_tight_dirichlet_applied"
    return new_concentrations, audit_base


def _apply_r2c_spectra_remediation(
    X: np.ndarray,
    *,
    wavelengths: np.ndarray,
    audit: dict[str, Any],
    record: PriorConfigRecord,
    profile: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a domain-specific mechanistic spectra remediation.

    Applies a multiplicative optical-path / scatter scale per sample. When
    the rule sets ``smoothing_fwhm_nm`` (R2d/R2g powder rules), the spectra are
    additionally convolved row-wise with a fixed Gaussian instrumental kernel
    of that FWHM (in nm) before the multiplicative scale is applied. This
    reduces over-sharp synthetic derivatives while preserving the row-mean
    structure. Constants are mechanistic; never derived from real spectra.
    """
    _validate_r2c_profile(profile)
    audit_out = dict(audit)
    transform_params = dict(audit_out.get("transform_params") or {})
    rule = _domain_rule_for(profile, record.domain_key)
    if rule is None:
        audit_out["transform_params"] = transform_params
        return X, audit_out

    spectra_seed = _profile_seed(f"r2c_spectra:{profile}", seed)
    rng = np.random.default_rng(spectra_seed)
    X_in = np.asarray(X, dtype=float)
    wl = np.asarray(wavelengths, dtype=float)

    smoothing_fwhm_nm = rule.get("smoothing_fwhm_nm")
    if smoothing_fwhm_nm is not None and X_in.shape[1] > 1:
        median_step = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
        median_step = median_step if median_step > 0 else 1.0
        sigma_bins = max(0.5, (float(smoothing_fwhm_nm) / 2.354820045) / median_step)
        kernel = _gaussian_kernel(sigma_bins)
        X_smoothed = _convolve_rows(X_in, kernel)
        transform_params.update({
            "smoothing_fwhm_nm": float(smoothing_fwhm_nm),
            "smoothing_sigma_bins": float(sigma_bins),
            "smoothing_kernel_size": int(kernel.size),
            "smoothing_median_step_nm": float(median_step),
        })
    else:
        X_smoothed = X_in

    if rule.get("spectra_rule") == "strawberry_puree_transflectance_residual_readout":
        route = _r2r_fruitpuree_route(record)
        if route is None:
            msg = (
                "R2r FruitPuree readout requires explicit bench-only FruitPuree "
                "route provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        low, high = rule["path_factor_range"]
        residual_scales = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        baseline_low, baseline_high = rule["additive_baseline_range"]
        baselines = np.asarray(
            rng.uniform(baseline_low, baseline_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        slope_low, slope_high = rule["scatter_slope_absorbance_range"]
        slopes = np.asarray(
            rng.uniform(slope_low, slope_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        water_low, water_high = rule["water_band_absorbance_range"]
        water_bands = np.asarray(
            rng.uniform(water_low, water_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        sugar_low, sugar_high = rule["sugar_solids_band_absorbance_range"]
        sugar_bands = np.asarray(
            rng.uniform(sugar_low, sugar_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        if wl.size > 1:
            wl_span = max(float(wl.max() - wl.min()), 1.0)
            wl_norm = (wl - float(wl.mean())) / wl_span
        else:
            wl_norm = np.zeros_like(wl, dtype=float)
        water_center = float(rule["water_band_center_nm"])
        water_width = float(rule["water_band_width_nm"])
        sugar_center = float(rule["sugar_solids_band_center_nm"])
        sugar_width = float(rule["sugar_solids_band_width_nm"])
        water_profile = np.exp(-0.5 * ((wl - water_center) / water_width) ** 2)
        sugar_profile = np.exp(-0.5 * ((wl - sugar_center) / sugar_width) ** 2)
        row_centered = X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)
        X_out = (
            baselines[:, None]
            + slopes[:, None] * wl_norm[None, :]
            + water_bands[:, None] * water_profile[None, :]
            + sugar_bands[:, None] * sugar_profile[None, :]
            - residual_scales[:, None] * row_centered
        )
        clip_low, clip_high = rule["output_clip_absorbance"]
        X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(residual_scales.min()),
            "path_factor_max": float(residual_scales.max()),
            "additive_baseline_range": [
                float(baseline_low),
                float(baseline_high),
            ],
            "additive_baseline_min": float(baselines.min()),
            "additive_baseline_max": float(baselines.max()),
            "scatter_slope_absorbance_range": [
                float(slope_low),
                float(slope_high),
            ],
            "scatter_slope_absorbance_min": float(slopes.min()),
            "scatter_slope_absorbance_max": float(slopes.max()),
            "water_band_absorbance_range": [float(water_low), float(water_high)],
            "water_band_absorbance_min": float(water_bands.min()),
            "water_band_absorbance_max": float(water_bands.max()),
            "sugar_solids_band_absorbance_range": [
                float(sugar_low),
                float(sugar_high),
            ],
            "sugar_solids_band_absorbance_min": float(sugar_bands.min()),
            "sugar_solids_band_absorbance_max": float(sugar_bands.max()),
            "water_band_center_nm": water_center,
            "water_band_width_nm": water_width,
            "sugar_solids_band_center_nm": sugar_center,
            "sugar_solids_band_width_nm": sugar_width,
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "provenance_source": rule["provenance_source"],
            "matrix_source": rule["matrix_source"],
            "scatter_source": rule["scatter_source"],
            "fruitpuree_readout_route_source": str(route.get("source", "unknown")),
            "fruitpuree_readout_route_marker": str(
                route.get("route_marker", "unknown")
            ),
            "fruitpuree_readout_route_non_oracle": bool(
                route.get("non_oracle", False)
            ),
            "fruitpuree_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "fruitpuree_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
            "output_clip_absorbance": [float(clip_low), float(clip_high)],
        })
    elif rule.get("spectra_rule") == "micro_path_fuel_ch_overtone_contrast_readout":
        route = _r2s_diesel_route(record) if profile in R2S_REMEDIATION_PROFILES else None
        if profile in R2S_REMEDIATION_PROFILES and route is None:
            msg = (
                "R2s DIESEL readout requires explicit bench-only DIESEL route "
                "provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        low, high = rule["path_factor_range"]
        path_factors = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        contrast_low, contrast_high = rule["feature_contrast_range"]
        feature_contrasts = np.asarray(
            rng.uniform(contrast_low, contrast_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        gain_low, gain_high = rule["ch_overtone_gain_range"]
        ch_gains = np.asarray(
            rng.uniform(gain_low, gain_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        continuum_fwhm_nm = float(rule["continuum_smoothing_fwhm_nm"])
        median_step = float(np.median(np.diff(wl))) if wl.size > 1 else 1.0
        median_step = median_step if median_step > 0 else 1.0
        continuum_sigma_bins = max(
            0.5,
            (continuum_fwhm_nm / 2.354820045) / median_step,
        )
        continuum_kernel = _gaussian_kernel(continuum_sigma_bins)
        continuum = _convolve_rows(X_in, continuum_kernel)
        feature_residual = X_in - continuum

        centers = np.asarray(rule["ch_overtone_centers_nm"], dtype=float)
        width_nm = float(rule["ch_overtone_width_nm"])
        ch_profile = np.zeros_like(wl, dtype=float)
        for center in centers:
            ch_profile += np.exp(-0.5 * ((wl - center) / width_nm) ** 2)
        if ch_profile.size and float(ch_profile.max()) > 0.0:
            ch_profile = ch_profile / float(ch_profile.max())
        path_profile = 1.0 + ch_gains[:, None] * ch_profile[None, :]

        X_out = (
            continuum * path_factors[:, None] * path_profile
            + feature_residual * feature_contrasts[:, None]
        )
        additive_baseline_range = rule["additive_baseline_range"]
        offset_low, offset_high = additive_baseline_range
        offsets = np.asarray(
            rng.uniform(offset_low, offset_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        X_out = X_out + offsets[:, None]
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(path_factors.min()),
            "path_factor_max": float(path_factors.max()),
            "feature_contrast_range": [float(contrast_low), float(contrast_high)],
            "feature_contrast_min": float(feature_contrasts.min()),
            "feature_contrast_max": float(feature_contrasts.max()),
            "continuum_smoothing_fwhm_nm": continuum_fwhm_nm,
            "continuum_smoothing_sigma_bins": float(continuum_sigma_bins),
            "continuum_smoothing_kernel_size": int(continuum_kernel.size),
            "continuum_smoothing_median_step_nm": float(median_step),
            "ch_overtone_centers_nm": [float(center) for center in centers],
            "ch_overtone_width_nm": width_nm,
            "ch_overtone_gain_range": [float(gain_low), float(gain_high)],
            "ch_overtone_gain_min": float(ch_gains.min()),
            "ch_overtone_gain_max": float(ch_gains.max()),
            "additive_baseline_range": [float(offset_low), float(offset_high)],
            "additive_baseline_min": float(offsets.min()),
            "additive_baseline_max": float(offsets.max()),
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "contrast_source": rule["contrast_source"],
            "provenance_source": rule.get("provenance_source"),
            "output_clip_absorbance": [
                float(clip_low),
                None if clip_high is None else float(clip_high),
            ],
        })
        if route is not None:
            transform_params.update({
                "diesel_readout_route_source": str(route.get("source", "unknown")),
                "diesel_readout_route_marker": str(route.get("route_marker", "unknown")),
                "diesel_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "diesel_readout_route_real_stat_capture": bool(
                    route.get("real_stat_capture", True)
                ),
                "diesel_readout_route_thresholds_modified": bool(
                    route.get("thresholds_modified", True)
                ),
            })
    elif (
        rule.get("spectra_rule")
        == "milk_emulsion_scatter_inverse_transflectance_readout"
    ):
        route = _r2m_milk_route(record)
        if route is None:
            msg = (
                "R2m milk readout requires explicit bench-only MILK route "
                "provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        variant = _r2m_milk_readout_variant(record)
        low, high = rule["path_factor_range"]
        path_factors = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        exponent_low, exponent_high = rule["scatter_exponent_range"]
        scatter_exponents = np.asarray(
            rng.uniform(exponent_low, exponent_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        offset_low, offset_high = rule["detector_offset_range"]
        detector_offsets = np.asarray(
            rng.uniform(offset_low, offset_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        dynamic_key = (
            "fullrange_detector_dynamic_range"
            if variant == "fullrange"
            else "detector_dynamic_range"
        )
        dynamic_low, dynamic_high = rule[dynamic_key]
        detector_dynamics = np.asarray(
            rng.uniform(dynamic_low, dynamic_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        clip_key = (
            "fullrange_output_clip_intensity"
            if variant == "fullrange"
            else "output_clip_intensity"
        )
        clip_low, clip_high = rule[clip_key]
        wl_ref = float(wl.min()) if wl.size else 1100.0
        wl_ref = wl_ref if wl_ref > 0.0 else 1100.0
        scatter_profile = np.power(
            np.clip(wl[None, :] / wl_ref, 1e-6, None),
            -scatter_exponents[:, None],
        )
        effective_absorbance = np.clip(X_smoothed, 0.0, None) * path_factors[:, None]
        X_out = (
            detector_offsets[:, None]
            + detector_dynamics[:, None]
            * scatter_profile
            * np.power(10.0, -effective_absorbance)
        )
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(path_factors.min()),
            "path_factor_max": float(path_factors.max()),
            "milk_readout_variant": variant,
            "scatter_exponent_range": [float(exponent_low), float(exponent_high)],
            "scatter_exponent_min": float(scatter_exponents.min()),
            "scatter_exponent_max": float(scatter_exponents.max()),
            "detector_offset_range": [float(offset_low), float(offset_high)],
            "detector_offset_min": float(detector_offsets.min()),
            "detector_offset_max": float(detector_offsets.max()),
            "detector_dynamic_range": [float(dynamic_low), float(dynamic_high)],
            "detector_dynamic_min": float(detector_dynamics.min()),
            "detector_dynamic_max": float(detector_dynamics.max()),
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "scatter_source": rule["scatter_source"],
            "provenance_source": "exp09_dataset_token_milk_route",
            "milk_readout_route_source": str(route.get("source", "unknown")),
            "milk_readout_route_marker": str(route.get("route_marker", "unknown")),
            "milk_readout_route_non_oracle": bool(route.get("non_oracle", False)),
            "milk_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "milk_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
            "output_clip_intensity": [float(clip_low), float(clip_high)],
        })
    elif rule.get("spectra_rule") == "cloudy_berry_percent_transmittance_readout":
        path_low, path_high = rule["absorbance_path_factor_range"]
        path_factors = np.asarray(
            rng.uniform(path_low, path_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        turbidity_low, turbidity_high = rule["turbidity_offset_percent_range"]
        turbidity_offsets = np.asarray(
            rng.uniform(turbidity_low, turbidity_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        baseline = float(rule["detector_baseline_percent"])
        dynamic = float(rule["detector_dynamic_percent"])
        clip_low, clip_high = rule["output_clip_percent"]
        effective_absorbance = np.clip(X_smoothed, 0.0, None) * path_factors[:, None]
        X_out = baseline + dynamic * np.power(10.0, -effective_absorbance)
        X_out = np.clip(X_out + turbidity_offsets[:, None], clip_low, clip_high)
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "absorbance_path_factor_range": [float(path_low), float(path_high)],
            "absorbance_path_factor_min": float(path_factors.min()),
            "absorbance_path_factor_max": float(path_factors.max()),
            "detector_baseline_percent": baseline,
            "detector_dynamic_percent": dynamic,
            "turbidity_offset_percent_range": [
                float(turbidity_low),
                float(turbidity_high),
            ],
            "turbidity_offset_percent_min": float(turbidity_offsets.min()),
            "turbidity_offset_percent_max": float(turbidity_offsets.max()),
            "output_clip_percent": [float(clip_low), float(clip_high)],
        })
    elif (
        rule.get("spectra_rule")
        == "fermented_beer_turbid_cuvette_absorbance_readout"
    ):
        route = _r2o_beer_route(record)
        if route is None:
            msg = (
                "R2o beer readout requires explicit bench-only BEER route "
                "provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        low, high = rule["path_factor_range"]
        path_factors = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        haze_low, haze_high = rule["haze_absorbance_baseline_range"]
        haze_baselines = np.asarray(
            rng.uniform(haze_low, haze_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        slope_low, slope_high = rule["haze_slope_absorbance_range"]
        haze_slopes = np.asarray(
            rng.uniform(slope_low, slope_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        carb_low, carb_high = rule["carbonation_residual_absorbance_range"]
        carbonation = np.asarray(
            rng.uniform(carb_low, carb_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        if wl.size > 1:
            wl_span = max(float(wl.max() - wl.min()), 1.0)
            shortwave_haze = (float(wl.max()) - wl) / wl_span
        else:
            shortwave_haze = np.zeros_like(wl, dtype=float)
        carbonation_profile = np.zeros_like(wl, dtype=float)
        for center, width in ((1210.0, 42.0), (1450.0, 58.0)):
            carbonation_profile += np.exp(-0.5 * ((wl - center) / width) ** 2)
        if carbonation_profile.size and float(carbonation_profile.max()) > 0.0:
            carbonation_profile /= float(carbonation_profile.max())
        X_out = (
            haze_baselines[:, None]
            + X_smoothed * path_factors[:, None]
            + haze_slopes[:, None] * shortwave_haze[None, :]
            + carbonation[:, None] * carbonation_profile[None, :]
        )
        clip_low, clip_high = rule["output_clip_absorbance"]
        X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(path_factors.min()),
            "path_factor_max": float(path_factors.max()),
            "haze_absorbance_baseline_range": [float(haze_low), float(haze_high)],
            "haze_absorbance_baseline_min": float(haze_baselines.min()),
            "haze_absorbance_baseline_max": float(haze_baselines.max()),
            "haze_slope_absorbance_range": [float(slope_low), float(slope_high)],
            "haze_slope_absorbance_min": float(haze_slopes.min()),
            "haze_slope_absorbance_max": float(haze_slopes.max()),
            "carbonation_residual_absorbance_range": [
                float(carb_low),
                float(carb_high),
            ],
            "carbonation_residual_absorbance_min": float(carbonation.min()),
            "carbonation_residual_absorbance_max": float(carbonation.max()),
            "carbonation_residual_centers_nm": [1210.0, 1450.0],
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "provenance_source": rule["provenance_source"],
            "matrix_source": rule["matrix_source"],
            "scatter_source": rule["scatter_source"],
            "output_clip_absorbance": [float(clip_low), float(clip_high)],
            "beer_readout_route_source": str(route.get("source", "unknown")),
            "beer_readout_route_marker": str(route.get("route_marker", "unknown")),
            "beer_readout_route_non_oracle": bool(route.get("non_oracle", False)),
            "beer_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "beer_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
        })
    elif (
        rule.get("spectra_rule")
        == "phosphorus_mineral_fertilizer_albedo_residual_readout"
    ):
        route = _r2p_phosphorus_route(record)
        if route is None:
            msg = (
                "R2p phosphorus readout requires explicit bench-only PHOSPHORUS "
                "route provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        low, high = rule["path_factor_range"]
        residual_scales = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        baseline_low, baseline_high = rule["additive_baseline_range"]
        baselines = np.asarray(
            rng.uniform(baseline_low, baseline_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        centered_residual = X_smoothed - X_smoothed.mean(axis=1, keepdims=True)
        X_out = baselines[:, None] - residual_scales[:, None] * centered_residual
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(residual_scales.min()),
            "path_factor_max": float(residual_scales.max()),
            "additive_baseline_range": [
                float(baseline_low),
                float(baseline_high),
            ],
            "additive_baseline_min": float(baselines.min()),
            "additive_baseline_max": float(baselines.max()),
            "centered_residual_readout": "inverted_about_phosphate_mineral_albedo",
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "provenance_source": rule["provenance_source"],
            "matrix_source": rule["matrix_source"],
            "scatter_source": rule["scatter_source"],
            "albedo_source": rule["albedo_source"],
            "output_clip_absorbance": [
                float(clip_low),
                None if clip_high is None else float(clip_high),
            ],
            "phosphorus_readout_route_source": str(route.get("source", "unknown")),
            "phosphorus_readout_route_marker": str(
                route.get("route_marker", "unknown")
            ),
            "phosphorus_readout_route_non_oracle": bool(
                route.get("non_oracle", False)
            ),
            "phosphorus_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "phosphorus_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
        })
    elif rule.get("spectra_rule") == "lucas_ph_organic_humic_albedo_oh_readout":
        route = _r2q_lucas_ph_organic_route(record)
        if route is None:
            msg = (
                "R2q LUCAS pH Organic readout requires explicit bench-only "
                "LUCAS pH Organic route provenance; route was missing or "
                "non-compliant"
            )
            raise ValueError(msg)
        low, high = rule["path_factor_range"]
        residual_scales = np.asarray(
            rng.uniform(low, high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        baseline_low, baseline_high = rule["additive_baseline_range"]
        baselines = np.asarray(
            rng.uniform(baseline_low, baseline_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        slope_low, slope_high = rule["humic_slope_absorbance_range"]
        humic_slopes = np.asarray(
            rng.uniform(slope_low, slope_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        oh_low, oh_high = rule["oh_band_absorbance_range"]
        oh_amplitudes = np.asarray(
            rng.uniform(oh_low, oh_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        if wl.size > 1:
            wl_span = max(float(wl.max() - wl.min()), 1.0)
            shortwave_humic = (float(wl.max()) - wl) / wl_span
        else:
            shortwave_humic = np.zeros_like(wl, dtype=float)
        oh_profile = np.zeros_like(wl, dtype=float)
        oh_width = float(rule["oh_band_width_nm"])
        oh_centers = np.asarray(rule["oh_band_centers_nm"], dtype=float)
        for center in oh_centers:
            oh_profile += np.exp(-0.5 * ((wl - center) / oh_width) ** 2)
        if oh_profile.size and float(oh_profile.max()) > 0.0:
            oh_profile /= float(oh_profile.max())
        X_out = (
            baselines[:, None]
            + residual_scales[:, None] * X_smoothed
            + humic_slopes[:, None] * shortwave_humic[None, :]
            + oh_amplitudes[:, None] * oh_profile[None, :]
        )
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(residual_scales.min()),
            "path_factor_max": float(residual_scales.max()),
            "additive_baseline_range": [
                float(baseline_low),
                float(baseline_high),
            ],
            "additive_baseline_min": float(baselines.min()),
            "additive_baseline_max": float(baselines.max()),
            "humic_slope_absorbance_range": [float(slope_low), float(slope_high)],
            "humic_slope_absorbance_min": float(humic_slopes.min()),
            "humic_slope_absorbance_max": float(humic_slopes.max()),
            "oh_band_absorbance_range": [float(oh_low), float(oh_high)],
            "oh_band_absorbance_min": float(oh_amplitudes.min()),
            "oh_band_absorbance_max": float(oh_amplitudes.max()),
            "oh_band_centers_nm": [float(center) for center in oh_centers],
            "oh_band_width_nm": oh_width,
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "provenance_source": rule["provenance_source"],
            "matrix_source": rule["matrix_source"],
            "scatter_source": rule["scatter_source"],
            "albedo_source": rule["albedo_source"],
            "output_clip_absorbance": [
                float(clip_low),
                None if clip_high is None else float(clip_high),
            ],
            "lucas_ph_organic_readout_route_source": str(
                route.get("source", "unknown")
            ),
            "lucas_ph_organic_readout_route_marker": str(
                route.get("route_marker", "unknown")
            ),
            "lucas_ph_organic_readout_route_non_oracle": bool(
                route.get("non_oracle", False)
            ),
            "lucas_ph_organic_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "lucas_ph_organic_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
        })
    elif (
        rule.get("spectra_rule")
        in {
            "dried_manure_heterogeneous_scatter_patch_readout",
            "dried_manure_bounded_centered_scatter_readout",
            "dried_manure_balanced_centered_scatter_readout",
            "dried_manure_albedo_variance_centered_scatter_readout",
        }
    ):
        route = _r2n_manure21_route(record)
        if route is None:
            msg = (
                "R2t/R2u/R2v/R2w manure heterogeneity readout requires explicit bench-only "
                "MANURE21 route provenance; route was missing or non-compliant"
            )
            raise ValueError(msg)
        balanced_centered_draws = bool(rule.get("balanced_centered_draws", False))

        def _bounded_draws(low: float, high: float, size: int) -> np.ndarray:
            if not balanced_centered_draws or size <= 1:
                return np.asarray(rng.uniform(low, high, size=size), dtype=float)
            quantiles = (np.arange(size, dtype=float) + 0.5) / float(size)
            values = low + (high - low) * quantiles
            return np.asarray(values[rng.permutation(size)], dtype=float)

        low, high = rule["path_factor_range"]
        residual_scales = _bounded_draws(low, high, X_smoothed.shape[0])
        baseline_low, baseline_high = rule["additive_baseline_range"]
        baselines = _bounded_draws(baseline_low, baseline_high, X_smoothed.shape[0])
        slope_low, slope_high = rule["scatter_slope_absorbance_range"]
        scatter_slopes = _bounded_draws(slope_low, slope_high, X_smoothed.shape[0])
        moisture_low, moisture_high = rule["moisture_patch_absorbance_range"]
        moisture_amplitudes = _bounded_draws(
            moisture_low,
            moisture_high,
            X_smoothed.shape[0],
        )
        organic_low, organic_high = rule["organic_lump_absorbance_range"]
        organic_amplitudes = _bounded_draws(
            organic_low,
            organic_high,
            X_smoothed.shape[0],
        )
        mineral_low, mineral_high = rule["mineral_ash_absorbance_range"]
        mineral_amplitudes = _bounded_draws(
            mineral_low,
            mineral_high,
            X_smoothed.shape[0],
        )
        if wl.size > 1:
            wl_span = max(float(wl.max() - wl.min()), 1.0)
            wl_norm = (wl - float(wl.mean())) / wl_span
        else:
            wl_norm = np.zeros_like(wl, dtype=float)

        centering_range = rule.get("readout_centering_range_nm")
        if centering_range is None:
            center_mask = np.ones_like(wl, dtype=bool)
            center_eval_grid = wl
        else:
            center_low, center_high = centering_range
            center_mask = (wl >= float(center_low)) & (wl <= float(center_high))
            if not bool(np.any(center_mask)):
                center_mask = np.ones_like(wl, dtype=bool)
            if (
                rule.get("readout_centering_grid") == "uniform_wavenumber"
                and float(center_low) > 0.0
                and float(center_high) > float(center_low)
            ):
                eval_size = max(int(center_mask.sum()) * 2, int(center_mask.sum()), 2)
                wavenumbers = np.linspace(
                    1.0e7 / float(center_high),
                    1.0e7 / float(center_low),
                    eval_size,
                )
                center_eval_grid = np.sort(1.0e7 / wavenumbers)
            else:
                center_eval_grid = wl[center_mask]

        def _center_mean(values: np.ndarray) -> float:
            if values.size == wl.size and center_eval_grid.size:
                return float(np.interp(center_eval_grid, wl, values).mean())
            return float(values[center_mask].mean())

        def _band_profile(centers_widths: tuple[tuple[float, float], ...]) -> np.ndarray:
            profile = np.zeros_like(wl, dtype=float)
            for center, width in centers_widths:
                profile += np.exp(-0.5 * ((wl - center) / width) ** 2)
            if profile.size and float(profile.max()) > 0.0:
                profile = profile / float(profile.max())
            return profile - _center_mean(profile) if profile.size else profile

        moisture_profile = _band_profile(((1450.0, 58.0), (1940.0, 76.0)))
        organic_profile = _band_profile(((1720.0, 64.0), (2300.0, 82.0)))
        mineral_profile = _band_profile(
            ((1410.0, 50.0), (2200.0, 74.0), (2340.0, 70.0))
        )
        residual_center = np.asarray(
            [
                np.interp(center_eval_grid, wl, row).mean()
                for row in X_smoothed
            ],
            dtype=float,
        )
        centered_residual = X_smoothed - residual_center[:, None]
        X_out = (
            baselines[:, None]
            + residual_scales[:, None] * centered_residual
            + scatter_slopes[:, None] * wl_norm[None, :]
            + moisture_amplitudes[:, None] * moisture_profile[None, :]
            + organic_amplitudes[:, None] * organic_profile[None, :]
            + mineral_amplitudes[:, None] * mineral_profile[None, :]
        )
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(residual_scales.min()),
            "path_factor_max": float(residual_scales.max()),
            "additive_baseline_range": [
                float(baseline_low),
                float(baseline_high),
            ],
            "additive_baseline_min": float(baselines.min()),
            "additive_baseline_max": float(baselines.max()),
            "scatter_slope_absorbance_range": [float(slope_low), float(slope_high)],
            "scatter_slope_absorbance_min": float(scatter_slopes.min()),
            "scatter_slope_absorbance_max": float(scatter_slopes.max()),
            "moisture_patch_absorbance_range": [
                float(moisture_low),
                float(moisture_high),
            ],
            "moisture_patch_absorbance_min": float(moisture_amplitudes.min()),
            "moisture_patch_absorbance_max": float(moisture_amplitudes.max()),
            "organic_lump_absorbance_range": [float(organic_low), float(organic_high)],
            "organic_lump_absorbance_min": float(organic_amplitudes.min()),
            "organic_lump_absorbance_max": float(organic_amplitudes.max()),
            "mineral_ash_absorbance_range": [float(mineral_low), float(mineral_high)],
            "mineral_ash_absorbance_min": float(mineral_amplitudes.min()),
            "mineral_ash_absorbance_max": float(mineral_amplitudes.max()),
            "moisture_patch_band_centers_nm": [1450.0, 1940.0],
            "organic_lump_band_centers_nm": [1720.0, 2300.0],
            "mineral_ash_band_centers_nm": [1410.0, 2200.0, 2340.0],
            "heterogeneous_terms_centered": True,
            "balanced_centered_draws": balanced_centered_draws,
            "readout_centering_range_nm": (
                None
                if centering_range is None
                else [float(center_low), float(center_high)]
            ),
            "readout_centering_grid": rule.get("readout_centering_grid"),
            "constant_status": rule["constant_status"],
            "readout_space": rule["readout_space"],
            "calibration_source": rule["calibration_source"],
            "real_stat_source": rule["real_stat_source"],
            "threshold_source": rule["threshold_source"],
            "provenance_source": rule["provenance_source"],
            "matrix_source": rule["matrix_source"],
            "scatter_source": rule["scatter_source"],
            "albedo_source": rule["albedo_source"],
            "heterogeneity_source": rule["heterogeneity_source"],
            "output_clip_absorbance": [
                float(clip_low),
                None if clip_high is None else float(clip_high),
            ],
            "manure21_readout_route_source": str(route.get("source", "unknown")),
            "manure21_readout_route_marker": str(
                route.get("route_marker", "unknown")
            ),
            "manure21_readout_route_non_oracle": bool(
                route.get("non_oracle", False)
            ),
            "manure21_readout_route_real_stat_capture": bool(
                route.get("real_stat_capture", True)
            ),
            "manure21_readout_route_thresholds_modified": bool(
                route.get("thresholds_modified", True)
            ),
        })
    else:
        low, high = rule["path_factor_range"]
        factors = np.asarray(rng.uniform(low, high, size=X_smoothed.shape[0]), dtype=float)
        X_out = X_smoothed * factors[:, None]
        additive_baseline_range = rule.get("additive_baseline_range")
        if additive_baseline_range is not None:
            offset_low, offset_high = additive_baseline_range
            offsets = np.asarray(
                rng.uniform(offset_low, offset_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            X_out = X_out + offsets[:, None]

        transform_params.update({
            "spectra_rule": rule["spectra_rule"],
            "spectra_source": rule["spectra_source"],
            "path_factor_range": [float(low), float(high)],
            "path_factor_min": float(factors.min()),
            "path_factor_max": float(factors.max()),
        })
        if additive_baseline_range is not None:
            transform_params.update({
                "additive_baseline_range": [float(offset_low), float(offset_high)],
                "additive_baseline_min": float(offsets.min()),
                "additive_baseline_max": float(offsets.max()),
            })
        for key in (
            "constant_status",
            "readout_space",
            "calibration_source",
            "real_stat_source",
            "threshold_source",
            "baseline_source",
            "provenance_source",
            "matrix_source",
            "scatter_source",
            "albedo_source",
        ):
            if key in rule:
                transform_params[key] = rule[key]
        if (
            rule.get("spectra_rule")
            == "dried_manure_organic_mineral_albedo_scatter_readout"
        ):
            route = _r2n_manure21_route(record)
            if route is None:
                msg = (
                    "R2n manure readout requires explicit bench-only MANURE21 "
                    "route provenance; route was missing or non-compliant"
                )
                raise ValueError(msg)
            transform_params.update({
                "manure21_readout_route_source": str(route.get("source", "unknown")),
                "manure21_readout_route_marker": str(
                    route.get("route_marker", "unknown")
                ),
                "manure21_readout_route_non_oracle": bool(
                    route.get("non_oracle", False)
                ),
                "manure21_readout_route_real_stat_capture": bool(
                    route.get("real_stat_capture", True)
                ),
                "manure21_readout_route_thresholds_modified": bool(
                    route.get("thresholds_modified", True)
                ),
            })
        transform_params.setdefault("constant_status", "fixed_mechanistic_prior")
        transform_params.setdefault("readout_space", "uncalibrated_raw_absorbance")
        transform_params.setdefault("calibration_source", "none")
        transform_params.setdefault("real_stat_source", "none")
        transform_params.setdefault("threshold_source", "none")
    audit_out["applied_to_spectra"] = True
    audit_out["spectra_seed"] = int(spectra_seed)
    audit_out["transform_params"] = transform_params
    return X_out, audit_out


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
