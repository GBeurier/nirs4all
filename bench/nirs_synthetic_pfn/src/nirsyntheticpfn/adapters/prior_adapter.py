"""Bench-side canonicalization for ``nirs4all.synthesis.prior`` samples."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, cast

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import APPLICATION_DOMAINS, get_domain_config
from nirs4all.synthesis.instruments import INSTRUMENT_ARCHETYPES, get_instrument_archetype
from nirs4all.synthesis.measurement_modes import MeasurementMode
from nirs4all.synthesis.prior import MatrixType, NIRSPriorConfig, PriorSampler

DOMAIN_ALIASES: dict[str, str] = {
    "agriculture_grain": "agriculture_grain",
    "grain": "agriculture_grain",
    "cereal": "agriculture_grain",
    "cereals": "agriculture_grain",
    "agriculture_forage": "agriculture_forage",
    "forage": "agriculture_forage",
    "agriculture_oilseeds": "agriculture_oilseeds",
    "oilseed": "agriculture_oilseeds",
    "oilseeds": "agriculture_oilseeds",
    "agriculture_fruit": "agriculture_fruit",
    "fruit": "agriculture_fruit",
    "food_dairy": "food_dairy",
    "dairy": "food_dairy",
    "food_meat": "food_meat",
    "meat": "food_meat",
    "food_bakery": "food_bakery",
    "baking": "food_bakery",
    "bakery": "food_bakery",
    "food_chocolate": "food_chocolate",
    "chocolate": "food_chocolate",
    "beverage_juice": "beverage_juice",
    "beverages": "beverage_juice",
    "juice": "beverage_juice",
    "beverage_wine": "beverage_wine",
    "wine": "beverage_wine",
    "pharma_tablets": "pharma_tablets",
    "tablets": "pharma_tablets",
    "tablet": "pharma_tablets",
    "pharma_powder_blends": "pharma_powder_blends",
    "powders": "pharma_powder_blends",
    "powder_blends": "pharma_powder_blends",
    "pharma_raw_materials": "pharma_raw_materials",
    "liquids": "pharma_raw_materials",
    "raw_materials": "pharma_raw_materials",
    "petrochem_fuels": "petrochem_fuels",
    "fuel": "petrochem_fuels",
    "fuels": "petrochem_fuels",
    "lubricants": "petrochem_fuels",
    "petrochem_polymers": "petrochem_polymers",
    "polymers": "petrochem_polymers",
    "polymer": "petrochem_polymers",
    "environmental_water": "environmental_water",
    "water_quality": "environmental_water",
    "water": "environmental_water",
    "environmental_soil": "environmental_soil",
    "soil": "environmental_soil",
    "biomedical_tissue": "biomedical_tissue",
    "tissue": "biomedical_tissue",
    "blood": "biomedical_tissue",
    "textile_natural": "textile_natural",
    "textiles": "textile_natural",
    "textile": "textile_natural",
    "textile_synthetic": "textile_synthetic",
}

VALID_MEASUREMENT_MODES = {mode.value for mode in MeasurementMode}
VALID_MATRIX_TYPES = {matrix.value for matrix in MatrixType}
VALID_TARGET_TYPES = {"regression", "classification"}
VALID_REGRESSION_NONLINEARITIES = {"none", "mild", "moderate"}
VALID_CLASSIFICATION_SEPARATIONS = {"easy", "moderate", "hard"}

# Bench-only domain-specific component aliases for R2c sentinel remediation.
# Each (domain_key, alias_lower) entry maps a non-registered fuel/matrix label
# to an existing nirs4all component that shares its dominant NIR signature.
# Mechanistic mapping only: never reads real spectra, labels, splits, or
# targets, never modifies nirs4all/ component definitions, and always records
# the substitution under ``source_prior_config["_bench_component_aliases"]``.
BENCH_ONLY_COMPONENT_ALIASES: dict[tuple[str, str], str] = {
    # Diesel is a saturated-hydrocarbon-dominated fuel blend; its NIR signature
    # is well approximated by the existing "oil" component (long-chain C-H
    # overtones and combinations). Used by R2c petrochem_fuels remediation.
    ("petrochem_fuels", "diesel"): "oil",
}
BENCH_ONLY_COMPONENT_ALIAS_SCOPE = "bench_only_r2c_petrochem_fuels_diesel_alias"


@dataclass(frozen=True)
class PriorValidationIssue:
    """A classified validation issue for a sampled prior config."""

    reason: str
    field: str
    message: str


class PriorCanonicalizationError(ValueError):
    """Raised when a prior sample cannot become a valid record."""

    def __init__(self, issues: list[PriorValidationIssue]) -> None:
        self.issues = issues
        summary = "; ".join(f"{issue.reason}:{issue.field}" for issue in issues)
        super().__init__(summary)

    @property
    def reason_counts(self) -> dict[str, int]:
        return dict(Counter(issue.reason for issue in self.issues))


@dataclass(frozen=True)
class PriorConfigRecord:
    """Canonical, validated bench record for a sampled NIRS prior."""

    domain_key: str
    product_key: str | None
    aggregate_key: str | None
    instrument_key: str
    measurement_mode: str
    wavelength_policy: dict[str, Any]
    component_keys: tuple[str, ...]
    concentration_prior: dict[str, Any]
    nuisance_prior: dict[str, Any]
    target_prior: dict[str, Any]
    task_prior: dict[str, Any]
    random_seed: int | None
    source_prior_config: dict[str, Any] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return cast("dict[str, Any]", _to_builtin(asdict(self)))


def canonicalize_domain(domain: Any) -> str:
    """Return an ``APPLICATION_DOMAINS`` key for a prior domain alias."""
    key = str(domain)
    canonical = DOMAIN_ALIASES.get(key, key)
    if canonical not in APPLICATION_DOMAINS:
        raise PriorCanonicalizationError([
            PriorValidationIssue(
                reason="invalid_domain",
                field="domain",
                message=f"Unknown domain alias {key!r}",
            )
        ])
    return canonical


def canonicalize_prior_config(
    source: dict[str, Any],
    *,
    allow_bench_wavelength_support_override: bool = False,
) -> PriorConfigRecord:
    """Validate and canonicalize one raw ``PriorSampler`` config."""
    issues: list[PriorValidationIssue] = []

    try:
        domain_key = canonicalize_domain(source.get("domain"))
    except PriorCanonicalizationError as exc:
        issues.extend(exc.issues)
        domain_key = ""

    domain_config = None
    if domain_key:
        domain_config = get_domain_config(domain_key)

    instrument_key = str(source.get("instrument", ""))
    instrument = None
    if instrument_key not in INSTRUMENT_ARCHETYPES:
        issues.append(PriorValidationIssue(
            reason="invalid_instrument",
            field="instrument",
            message=f"Unknown instrument {instrument_key!r}",
        ))
    else:
        instrument = get_instrument_archetype(instrument_key)
        source_category = source.get("instrument_category")
        if source_category is not None and str(source_category) != instrument.category.value:
            issues.append(PriorValidationIssue(
                reason="instrument_category_mismatch",
                field="instrument_category",
                message=(
                    f"Prior category {source_category!r} does not match "
                    f"{instrument_key!r} category {instrument.category.value!r}"
                ),
            ))

    measurement_mode = str(source.get("measurement_mode", ""))
    if measurement_mode not in VALID_MEASUREMENT_MODES:
        issues.append(PriorValidationIssue(
            reason="invalid_measurement_mode",
            field="measurement_mode",
            message=f"Unknown measurement mode {measurement_mode!r}",
        ))
    elif instrument is not None and not _mode_supported_by_category(measurement_mode, instrument.category.value):
        issues.append(PriorValidationIssue(
            reason="unsupported_measurement_mode",
            field="measurement_mode",
            message=(
                f"Mode {measurement_mode!r} has zero prior weight for "
                f"instrument category {instrument.category.value!r}"
            ),
        ))

    wavelength_range = _coerce_range(source.get("wavelength_range"), "wavelength_range", issues)
    spectral_resolution = source.get("spectral_resolution")
    wavelength_policy: dict[str, Any] = {}
    if wavelength_range is not None and instrument is not None and domain_config is not None:
        domain_range, wavelength_support_audit = _wavelength_support_domain_range(
            source=source,
            default_domain_range=domain_config.wavelength_range,
            allow_bench_wavelength_support_override=allow_bench_wavelength_support_override,
            issues=issues,
        )
        wavelength_policy = _validate_wavelengths(
            wavelength_range=wavelength_range,
            spectral_resolution=spectral_resolution,
            instrument_range=instrument.wavelength_range,
            domain_range=domain_range,
            issues=issues,
        )
        wavelength_policy["bench_wavelength_support_override"] = wavelength_support_audit

    raw_components = source.get("components", ())
    aliased_components, alias_audit = _apply_bench_component_aliases(
        domain_key, raw_components
    )
    component_keys = _validate_components(aliased_components, issues)
    if domain_config is not None:
        _validate_domain_components(component_keys, domain_config, issues)
    concentration_prior = _build_concentration_prior(domain_key, component_keys)
    nuisance_prior = _validate_nuisance(source, instrument, issues)
    target_prior = _validate_target(source.get("target_config"), component_keys, issues)
    task_prior = {
        "n_samples": source.get("n_samples"),
        "matrix_type": source.get("matrix_type"),
        "domain_category": domain_config.category.value if domain_config else None,
    }

    if issues:
        raise PriorCanonicalizationError(issues)

    source_with_audit = dict(source)
    if alias_audit:
        source_with_audit["components"] = list(aliased_components)
        source_with_audit["_bench_component_aliases"] = {
            "scope": BENCH_ONLY_COMPONENT_ALIAS_SCOPE,
            "applied": True,
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "translations": alias_audit,
            "raw_components": [str(name) for name in raw_components],
        }
    return PriorConfigRecord(
        domain_key=domain_key,
        product_key=_optional_str(source.get("product_key")),
        aggregate_key=_optional_str(source.get("aggregate_key")),
        instrument_key=instrument_key,
        measurement_mode=measurement_mode,
        wavelength_policy=wavelength_policy,
        component_keys=tuple(component_keys),
        concentration_prior=concentration_prior,
        nuisance_prior=nuisance_prior,
        target_prior=target_prior,
        task_prior=task_prior,
        random_seed=_optional_int(source.get("random_state")),
        source_prior_config=_to_builtin(source_with_audit),
    )


def sample_canonical_prior(
    random_state: int | None = None,
    *,
    repair_domain_components: bool = False,
) -> PriorConfigRecord:
    """Sample one raw prior config and canonicalize it.

    The canonical sampling order is: short/raw domain alias -> canonical domain ->
    components of the canonical domain -> validated record. Production
    ``PriorSampler`` samples components before any canonicalization, so when
    ``repair_domain_components=True`` the components are re-drawn from the
    canonical domain's ``typical_components`` using the same RNG, and the repair
    is recorded in ``source_prior_config["_canonical_repairs"]``. Errors that
    cannot be repaired (e.g. unknown domain) are still raised.
    """
    sampler = PriorSampler(random_state=random_state)
    raw = sampler.sample()
    if repair_domain_components:
        raw = _repair_raw_sample(raw, sampler.rng)
    return canonicalize_prior_config(raw)


def summarize_prior_coverage(
    n_samples: int = 1000,
    random_state: int = 0,
    *,
    repair_domain_components: bool = False,
) -> dict[str, Any]:
    """Sample raw priors and summarize canonicalization coverage.

    When ``repair_domain_components=True``, components are re-sampled from the
    canonical domain after canonicalizing the domain alias; failures unrelated
    to component compatibility (e.g. unknown domains, wavelength mismatches)
    are still reported.
    """
    sampler = PriorSampler(random_state=random_state)
    valid_records: list[PriorConfigRecord] = []
    invalid_reasons: Counter[str] = Counter()
    invalid_fields: Counter[str] = Counter()
    source_domains: Counter[str] = Counter()
    repair_counts: Counter[str] = Counter()

    for _ in range(n_samples):
        sample = sampler.sample()
        source_domains[str(sample.get("domain"))] += 1
        if repair_domain_components:
            sample = _repair_raw_sample(sample, sampler.rng)
            for repair_key in sample.get("_canonical_repairs", {}):
                repair_counts[repair_key] += 1
        try:
            valid_records.append(canonicalize_prior_config(sample))
        except PriorCanonicalizationError as exc:
            for issue in exc.issues:
                invalid_reasons[issue.reason] += 1
                invalid_fields[issue.field] += 1

    return {
        "n_samples": n_samples,
        "random_state": random_state,
        "repair_domain_components": repair_domain_components,
        "valid_count": len(valid_records),
        "invalid_count": n_samples - len(valid_records),
        "source_domain_counts": dict(sorted(source_domains.items())),
        "domain_counts": _record_counter(valid_records, "domain_key"),
        "component_counts": _component_counter(valid_records),
        "instrument_counts": _record_counter(valid_records, "instrument_key"),
        "measurement_mode_counts": _record_counter(valid_records, "measurement_mode"),
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
        "invalid_field_counts": dict(sorted(invalid_fields.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
    }


def _repair_raw_sample(
    sample: dict[str, Any],
    rng: Any,
) -> dict[str, Any]:
    """Resample components from the canonical domain (in-place repair).

    Repairs only fields that production ``PriorSampler`` derives from a
    pre-canonicalization domain alias. The original raw values are preserved
    under ``_canonical_repairs`` so downstream consumers can audit what
    changed. Errors that cannot be repaired (unknown domain, missing typical
    components) leave the sample untouched and are surfaced by validation.
    """
    raw = dict(sample)
    raw_prior_config = _to_builtin(dict(sample))
    raw_domain = raw.get("domain")
    try:
        canonical_domain = canonicalize_domain(raw_domain)
    except PriorCanonicalizationError:
        return raw

    domain_config = get_domain_config(canonical_domain)
    valid_typical: list[str] = []
    for typical_name in domain_config.typical_components:
        try:
            valid_typical.append(get_component(str(typical_name)).name)
        except ValueError:
            continue

    if not valid_typical:
        return raw

    original_components = list(raw.get("components") or [])
    requested_targets = _requested_regression_targets(raw.get("target_config"))
    target_count = max(len(original_components) or 3, requested_targets or 1)
    target_count = min(target_count, len(valid_typical))
    chosen = [str(name) for name in rng.choice(valid_typical, size=target_count, replace=False)]

    repairs: dict[str, Any] = {}
    if chosen != [str(name) for name in original_components]:
        repairs["components_resampled_from_canonical_domain"] = {
            "raw_domain": str(raw_domain),
            "canonical_domain": canonical_domain,
            "original": [str(name) for name in original_components],
            "canonical": chosen,
        }
        raw["components"] = chosen

    if requested_targets is not None and requested_targets > len(chosen):
        target_config = dict(raw.get("target_config") or {})
        target_config["n_targets"] = len(chosen)
        repairs["target_n_targets_clipped_to_component_count"] = {
            "raw_domain": str(raw_domain),
            "canonical_domain": canonical_domain,
            "original": requested_targets,
            "canonical": len(chosen),
            "component_count": len(chosen),
        }
        raw["target_config"] = target_config

    if repairs:
        raw["_raw_prior_config"] = raw_prior_config
        raw["_canonical_repairs"] = repairs
    return raw


def _requested_regression_targets(target_config: Any) -> int | None:
    if not isinstance(target_config, dict) or target_config.get("type") != "regression":
        return None
    return _optional_int(target_config.get("n_targets"))


def _mode_supported_by_category(mode: str, category: str) -> bool:
    weights = NIRSPriorConfig().mode_given_category.get(category)
    if weights is None:
        return mode in {"reflectance", "transmittance"}
    return weights.get(mode, 0.0) > 0.0


def _coerce_range(
    value: Any,
    field_name: str,
    issues: list[PriorValidationIssue],
) -> tuple[float, float] | None:
    try:
        low, high = value
        low = float(low)
        high = float(high)
    except Exception:
        issues.append(PriorValidationIssue(
            reason="invalid_wavelength_range",
            field=field_name,
            message=f"{field_name} must be a two-value numeric range",
        ))
        return None

    if not low < high:
        issues.append(PriorValidationIssue(
            reason="invalid_wavelength_range",
            field=field_name,
            message=f"{field_name} must be increasing, got {(low, high)!r}",
        ))
        return None
    return low, high


def _validate_wavelengths(
    *,
    wavelength_range: tuple[float, float],
    spectral_resolution: Any,
    instrument_range: tuple[float, float],
    domain_range: tuple[float, float],
    issues: list[PriorValidationIssue],
) -> dict[str, Any]:
    wl_low, wl_high = wavelength_range
    inst_low, inst_high = instrument_range
    dom_low, dom_high = domain_range
    tol = 1e-9

    if wl_low < inst_low - tol or wl_high > inst_high + tol:
        issues.append(PriorValidationIssue(
            reason="wavelength_instrument_mismatch",
            field="wavelength_range",
            message=(
                f"Wavelength range {(wl_low, wl_high)!r} is outside "
                f"instrument range {(inst_low, inst_high)!r}"
            ),
        ))

    overlap = (max(wl_low, dom_low), min(wl_high, dom_high))
    if not overlap[0] < overlap[1]:
        issues.append(PriorValidationIssue(
            reason="wavelength_domain_mismatch",
            field="wavelength_range",
            message=(
                f"Wavelength range {(wl_low, wl_high)!r} does not overlap "
                f"domain range {(dom_low, dom_high)!r}"
            ),
        ))

    try:
        resolution = None if spectral_resolution is None else float(spectral_resolution)
    except Exception:
        issues.append(PriorValidationIssue(
            reason="invalid_spectral_resolution",
            field="spectral_resolution",
            message=f"Invalid spectral resolution {spectral_resolution!r}",
        ))
        resolution = None

    if resolution is not None and resolution <= 0:
        issues.append(PriorValidationIssue(
            reason="invalid_spectral_resolution",
            field="spectral_resolution",
            message=f"Spectral resolution must be positive, got {resolution!r}",
        ))

    return {
        "source_range_nm": [wl_low, wl_high],
        "instrument_range_nm": [float(inst_low), float(inst_high)],
        "domain_range_nm": [float(dom_low), float(dom_high)],
        "effective_range_nm": [float(overlap[0]), float(overlap[1])],
        "spectral_resolution_nm": resolution,
    }


def _wavelength_support_domain_range(
    *,
    source: dict[str, Any],
    default_domain_range: tuple[float, float],
    allow_bench_wavelength_support_override: bool,
    issues: list[PriorValidationIssue],
) -> tuple[tuple[float, float], dict[str, Any]]:
    raw_override = source.get("_bench_wavelength_support_override")
    default_range = (float(default_domain_range[0]), float(default_domain_range[1]))
    base_audit: dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "reason": "not_requested",
        "rule": "",
        "scope": "bench_only_real_grid_support",
        "default_domain_range_nm": [default_range[0], default_range[1]],
        "override_domain_range_nm": None,
        "non_oracle": True,
        "no_target_or_label": True,
        "oracle": False,
        "label_inputs_used": False,
        "target_inputs_used": False,
        "split_inputs_used": False,
        "source_oracle_used": False,
        "thresholds_modified": False,
        "metrics_modified": False,
        "covariance_enabled": False,
        "imputed": False,
        "replays_real_rows": False,
    }
    if raw_override is None:
        return default_range, base_audit
    if not isinstance(raw_override, dict) or raw_override.get("enabled") is not True:
        audit = dict(base_audit)
        audit["reason"] = "override_present_but_disabled_or_malformed"
        return default_range, audit

    source_fields = raw_override.get("source_fields", ())
    if not isinstance(source_fields, (list, tuple)):
        source_fields = ()
    audit = {
        **base_audit,
        "enabled": True,
        "reason": str(raw_override.get("reason", "")),
        "rule": str(raw_override.get("rule", "")),
        "source_fields": [str(field) for field in source_fields],
    }
    if not allow_bench_wavelength_support_override:
        audit["reason"] = "explicit_canonicalization_flag_not_enabled"
        return default_range, audit

    required_false_fields = (
        "oracle",
        "label_inputs_used",
        "target_inputs_used",
        "split_inputs_used",
        "source_oracle_used",
        "thresholds_modified",
        "metrics_modified",
        "covariance_enabled",
        "imputed",
        "replays_real_rows",
    )
    invalid_fields = [
        field_name
        for field_name in required_false_fields
        if raw_override.get(field_name) not in (False, None)
    ]
    if raw_override.get("non_oracle") is not True:
        invalid_fields.append("non_oracle")
    if raw_override.get("no_target_or_label") is not True:
        invalid_fields.append("no_target_or_label")
    if invalid_fields:
        issues.append(PriorValidationIssue(
            reason="invalid_bench_wavelength_support_override",
            field="_bench_wavelength_support_override",
            message=f"Unsafe wavelength support override audit fields: {invalid_fields!r}",
        ))
        return default_range, audit

    override_range = _coerce_range(
        raw_override.get("domain_range"),
        "_bench_wavelength_support_override.domain_range",
        issues,
    )
    if override_range is None:
        return default_range, audit

    override_range = (float(override_range[0]), float(override_range[1]))
    audit["applied"] = True
    audit["override_domain_range_nm"] = [override_range[0], override_range[1]]
    return override_range, audit


def _apply_bench_component_aliases(
    domain_key: str,
    components: Any,
) -> tuple[Any, list[dict[str, str]]]:
    """Translate bench-only domain-specific component aliases.

    Mechanistic substitution only: maps non-registered fuel/matrix labels
    (e.g. ``"diesel"`` for ``petrochem_fuels``) to existing nirs4all components
    that share the dominant NIR signature. Returns ``(translated_components,
    audit)`` where ``audit`` lists each substitution. The translation never
    reads real spectra, labels, splits, or targets.
    """
    if not isinstance(components, (list, tuple)):
        return components, []
    audit: list[dict[str, str]] = []
    translated: list[Any] = []
    for raw in components:
        name = str(raw)
        alias_key = (domain_key, name.lower())
        canonical = BENCH_ONLY_COMPONENT_ALIASES.get(alias_key)
        if canonical is None:
            translated.append(raw)
            continue
        translated.append(canonical)
        audit.append({
            "domain_key": domain_key,
            "raw_component": name,
            "canonical_component": canonical,
            "rule": "non_registered_blend_label_to_dominant_nir_component",
        })
    return translated, audit


def _validate_components(value: Any, issues: list[PriorValidationIssue]) -> list[str]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        issues.append(PriorValidationIssue(
            reason="invalid_components",
            field="components",
            message="Components must be a non-empty list",
        ))
        return []

    canonical: list[str] = []
    for raw_name in value:
        name = str(raw_name)
        if name.startswith("procedural:") and len(name) > len("procedural:"):
            issues.append(PriorValidationIssue(
                reason="unsupported_component_source",
                field="components",
                message=(
                    f"Procedural component {name!r} has no bench-side "
                    "resolver in Phase A1"
                ),
            ))
            continue
        try:
            canonical.append(get_component(name).name)
        except ValueError:
            issues.append(PriorValidationIssue(
                reason="invalid_component",
                field="components",
                message=f"Unknown component {name!r}",
            ))

    if len(set(canonical)) != len(canonical):
        issues.append(PriorValidationIssue(
            reason="duplicate_component",
            field="components",
            message=f"Duplicate components after canonicalization: {canonical!r}",
        ))

    return canonical


def _validate_domain_components(
    component_keys: list[str],
    domain_config: Any,
    issues: list[PriorValidationIssue],
) -> None:
    allowed: set[str] = set()
    for raw_name in domain_config.typical_components:
        try:
            allowed.add(get_component(str(raw_name)).name)
        except ValueError:
            continue

    mismatched = [
        component_key
        for component_key in component_keys
        if not component_key.startswith("procedural:") and component_key not in allowed
    ]
    if mismatched:
        issues.append(PriorValidationIssue(
            reason="domain_component_mismatch",
            field="components",
            message=(
                f"Components {mismatched!r} are not typical for "
                f"domain {domain_config.name!r}"
            ),
        ))


def _build_concentration_prior(domain_key: str, component_keys: list[str]) -> dict[str, Any]:
    if not domain_key:
        return {}
    domain_config = get_domain_config(domain_key)
    priors: dict[str, Any] = {}
    for component_key in component_keys:
        prior = domain_config.concentration_priors.get(component_key)
        if prior is None:
            priors[component_key] = {
                "distribution": "beta",
                "params": {"a": 2, "b": 5},
                "min_value": 0.0,
                "max_value": 1.0,
                "source": "domain_default",
            }
        else:
            priors[component_key] = {
                "distribution": prior.distribution,
                "params": dict(prior.params),
                "min_value": prior.min_value,
                "max_value": prior.max_value,
                "source": "domain",
            }
    return priors


def _validate_nuisance(
    source: dict[str, Any],
    instrument: Any,
    issues: list[PriorValidationIssue],
) -> dict[str, Any]:
    config = NIRSPriorConfig()
    matrix_type = str(source.get("matrix_type", ""))
    if matrix_type not in VALID_MATRIX_TYPES:
        issues.append(PriorValidationIssue(
            reason="invalid_matrix_type",
            field="matrix_type",
            message=f"Unknown matrix type {matrix_type!r}",
        ))

    temperature = _optional_float(source.get("temperature"))
    if temperature is None or not (config.temperature_range[0] <= temperature <= config.temperature_range[1]):
        issues.append(PriorValidationIssue(
            reason="nuisance_out_of_range",
            field="temperature",
            message=f"Temperature {source.get('temperature')!r} outside {config.temperature_range!r}",
        ))

    particle_size = _optional_float(source.get("particle_size"))
    particle_range = _particle_size_range(matrix_type)
    if particle_size is None or not (particle_range[0] <= particle_size <= particle_range[1]):
        issues.append(PriorValidationIssue(
            reason="nuisance_out_of_range",
            field="particle_size",
            message=f"Particle size {source.get('particle_size')!r} outside {particle_range!r}",
        ))

    category = instrument.category.value if instrument is not None else str(source.get("instrument_category", ""))
    noise_level = _optional_float(source.get("noise_level"))
    noise_range = _noise_level_range(category)
    if noise_level is None or not (noise_range[0] <= noise_level <= noise_range[1]):
        issues.append(PriorValidationIssue(
            reason="nuisance_out_of_range",
            field="noise_level",
            message=f"Noise level {source.get('noise_level')!r} outside {noise_range!r}",
        ))

    return {
        "matrix_type": matrix_type,
        "temperature_c": temperature,
        "particle_size_um": particle_size,
        "noise_level": noise_level,
    }


def _validate_target(
    value: Any,
    component_keys: list[str],
    issues: list[PriorValidationIssue],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        issues.append(PriorValidationIssue(
            reason="invalid_target_prior",
            field="target_config",
            message="target_config must be a dictionary",
        ))
        return {}

    target_type = str(value.get("type", ""))
    if target_type not in VALID_TARGET_TYPES:
        issues.append(PriorValidationIssue(
            reason="invalid_target_prior",
            field="target_config.type",
            message=f"Unsupported target type {target_type!r}",
        ))
        return dict(value)

    target_prior = dict(value)
    target_prior["executable_mapping"] = None
    if target_type == "regression":
        n_targets = _optional_int(value.get("n_targets"))
        nonlinearity = str(value.get("nonlinearity", ""))
        if n_targets is None or n_targets < 1 or n_targets > max(1, len(component_keys)):
            issues.append(PriorValidationIssue(
                reason="invalid_target_prior",
                field="target_config.n_targets",
                message=(
                    f"n_targets must be between 1 and the component count "
                    f"({len(component_keys)}), got {value.get('n_targets')!r}"
                ),
            ))
        if nonlinearity not in VALID_REGRESSION_NONLINEARITIES:
            issues.append(PriorValidationIssue(
                reason="invalid_target_prior",
                field="target_config.nonlinearity",
                message=f"Unsupported regression nonlinearity {nonlinearity!r}",
            ))
        target_prior["executable_mapping"] = "component_concentration_regression"
    else:
        n_classes = _optional_int(value.get("n_classes"))
        separation = str(value.get("separation", ""))
        if n_classes is None or not (2 <= n_classes <= 5):
            issues.append(PriorValidationIssue(
                reason="invalid_target_prior",
                field="target_config.n_classes",
                message=f"n_classes must be in [2, 5], got {value.get('n_classes')!r}",
            ))
        if separation not in VALID_CLASSIFICATION_SEPARATIONS:
            issues.append(PriorValidationIssue(
                reason="invalid_target_prior",
                field="target_config.separation",
                message=f"Unsupported class separation {separation!r}",
            ))
        target_prior["executable_mapping"] = "mixture_classification"

    return cast("dict[str, Any]", _to_builtin(target_prior))


def _particle_size_range(matrix_type: str) -> tuple[float, float]:
    if matrix_type == "powder":
        return 5.0, 100.0
    if matrix_type == "granular":
        return 50.0, 500.0
    if matrix_type in {"liquid", "emulsion"}:
        return 0.1, 10.0
    if matrix_type == "solid":
        return 100.0, 1000.0
    return NIRSPriorConfig().particle_size_range


def _noise_level_range(instrument_category: str) -> tuple[float, float]:
    if instrument_category == "handheld":
        return 1.0, 3.0
    if instrument_category == "embedded":
        return 1.5, 3.5
    if instrument_category == "ft_nir":
        return 0.3, 1.0
    if instrument_category == "benchtop":
        return 0.5, 1.5
    return NIRSPriorConfig().noise_level_range


def _record_counter(records: list[PriorConfigRecord], attr: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(record, attr)) for record in records).items()))


def _component_counter(records: list[PriorConfigRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.component_keys)
    return dict(sorted(counts.items()))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
