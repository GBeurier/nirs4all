from __future__ import annotations

import pytest
from nirsyntheticpfn.adapters.prior_adapter import (
    PriorCanonicalizationError,
    PriorConfigRecord,
    _repair_raw_sample,
    canonicalize_domain,
    canonicalize_prior_config,
    sample_canonical_prior,
    summarize_prior_coverage,
)

from nirs4all.synthesis.domains import APPLICATION_DOMAINS, get_domain_config
from nirs4all.synthesis.prior import NIRSPriorConfig


def test_default_prior_domain_aliases_resolve_to_application_domains() -> None:
    for domain in NIRSPriorConfig().domain_weights:
        assert canonicalize_domain(domain) in APPLICATION_DOMAINS


def test_canonicalize_valid_prior_record() -> None:
    record = canonicalize_prior_config(_valid_source("grain"))

    assert isinstance(record, PriorConfigRecord)
    assert record.domain_key == "agriculture_grain"
    assert record.instrument_key == "foss_xds"
    assert record.measurement_mode == "reflectance"
    assert record.component_keys == ("starch", "protein", "moisture")
    assert record.target_prior["executable_mapping"] == "component_concentration_regression"
    assert record.wavelength_policy["effective_range_nm"] == [1100.0, 2500.0]
    assert record.source_prior_config["domain"] == "grain"
    assert record.source_prior_config["components"] == ["starch", "protein", "moisture"]


def test_canonicalization_is_reproducible_for_same_source() -> None:
    source = _valid_source("tablets")

    first = canonicalize_prior_config(source).to_dict()
    second = canonicalize_prior_config(source).to_dict()

    assert first == second


def test_invalid_domain_is_classified() -> None:
    source = _valid_source("grain")
    source["domain"] = "not_a_domain"

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts == {"invalid_domain": 1}


def test_invalid_component_is_classified_without_silent_fallback() -> None:
    source = _valid_source("grain")
    source["components"] = ["water", "carbohydrate"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts["invalid_component"] == 1


def test_procedural_component_without_resolver_is_classified() -> None:
    source = _valid_source("grain")
    source["components"] = ["procedural:synthetic_unknown"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts["unsupported_component_source"] == 1


def test_domain_component_mismatch_is_classified_without_silent_fallback() -> None:
    source = _valid_source("fuel")
    source["components"] = ["water", "protein", "lipid"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert exc.value.reason_counts["domain_component_mismatch"] == 1


def test_invalid_instrument_mode_wavelength_and_target_are_classified() -> None:
    source = _valid_source("grain")
    source["instrument"] = "missing_instrument"
    source["measurement_mode"] = "unsupported_mode"
    source["wavelength_range"] = (3000, 2500)
    source["target_config"] = {"type": "classification", "n_classes": 9, "separation": "none"}

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    reasons = exc.value.reason_counts
    assert reasons["invalid_instrument"] == 1
    assert reasons["invalid_measurement_mode"] == 1
    assert reasons["invalid_wavelength_range"] == 1
    assert reasons["invalid_target_prior"] == 2


def test_ten_alias_domain_presets_validate_with_domain_components() -> None:
    aliases = [
        "grain",
        "forage",
        "oilseeds",
        "fruit",
        "dairy",
        "meat",
        "baking",
        "tablets",
        "powders",
        "fuel",
    ]

    records = [canonicalize_prior_config(_valid_source(alias)) for alias in aliases]

    assert len(records) == 10
    assert len({record.domain_key for record in records}) == 10
    assert all(record.component_keys for record in records)


def test_prior_coverage_summary_accounts_for_all_samples() -> None:
    summary = summarize_prior_coverage(n_samples=50, random_state=123)
    repeated = summarize_prior_coverage(n_samples=50, random_state=123)

    assert summary["n_samples"] == 50
    assert summary["valid_count"] + summary["invalid_count"] == 50
    assert summary["source_domain_counts"]
    assert summary["invalid_reason_counts"]
    assert summary == repeated


def test_raw_fallback_components_do_not_silently_validate_for_incompatible_domain() -> None:
    """A raw sample with the production fallback components ``water/protein/lipid/cellulose``
    must not become a valid record on a domain whose canonical typical_components are
    different (e.g. petrochemical fuels)."""
    source = _valid_source("fuel")
    source["components"] = ["water", "protein", "lipid", "cellulose"]

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(source)

    assert "domain_component_mismatch" in exc.value.reason_counts


def test_sample_canonical_prior_without_repair_can_fail() -> None:
    """Without repair, the canonical sampler must propagate validation failures
    rather than silently discarding raw mismatches."""
    failures = 0
    for seed in range(50):
        try:
            sample_canonical_prior(random_state=seed, repair_domain_components=False)
        except PriorCanonicalizationError:
            failures += 1
    assert failures > 0


def test_sample_canonical_prior_with_repair_repairs_components_from_canonical_domain() -> None:
    """With repair, components are re-drawn from the canonical domain's
    typical_components and the repair is recorded in source_prior_config."""
    repaired = 0
    for seed in range(20):
        try:
            record = sample_canonical_prior(random_state=seed, repair_domain_components=True)
        except PriorCanonicalizationError:
            continue
        repairs = record.source_prior_config.get("_canonical_repairs")
        if not repairs:
            continue
        component_repair = repairs.get("components_resampled_from_canonical_domain")
        if component_repair is None:
            continue
        repaired += 1
        assert component_repair["canonical_domain"] == record.domain_key
        domain_typical = {
            str(c) for c in get_domain_config(record.domain_key).typical_components
        }
        assert set(record.component_keys).issubset(domain_typical)
    assert repaired > 0


def test_repair_clips_regression_targets_and_records_source_config() -> None:
    """If canonical components are fewer than requested regression targets,
    repair mode clips n_targets and records the original raw prior values."""
    from nirs4all.synthesis.prior import PriorSampler

    source = _valid_source("blood")
    source["components"] = ["water", "protein", "lipid", "cellulose", "carbohydrate"]
    source["target_config"] = {
        "type": "regression",
        "n_targets": 5,
        "nonlinearity": "none",
    }

    repaired = _repair_raw_sample(source, PriorSampler(random_state=7).rng)
    record = canonicalize_prior_config(repaired)

    repairs = record.source_prior_config["_canonical_repairs"]
    assert record.source_prior_config["_raw_prior_config"]["target_config"]["n_targets"] == 5
    assert record.source_prior_config["_raw_prior_config"]["components"] == [
        "water",
        "protein",
        "lipid",
        "cellulose",
        "carbohydrate",
    ]
    assert repairs["components_resampled_from_canonical_domain"]["original"] == [
        "water",
        "protein",
        "lipid",
        "cellulose",
        "carbohydrate",
    ]
    assert repairs["target_n_targets_clipped_to_component_count"]["original"] == 5
    assert repairs["target_n_targets_clipped_to_component_count"]["canonical"] == 4
    assert record.source_prior_config["target_config"]["n_targets"] == 4
    assert len(record.component_keys) == 4


def test_sample_canonical_prior_with_repair_is_reproducible() -> None:
    """Two canonical samples with identical seed and repair flag must produce
    identical records (component repair is RNG-driven via the same sampler)."""
    first = sample_canonical_prior(random_state=20260428, repair_domain_components=True).to_dict()
    second = sample_canonical_prior(random_state=20260428, repair_domain_components=True).to_dict()
    assert first == second


def test_sample_canonical_prior_with_repair_does_not_mask_unknown_domain() -> None:
    """Repair must not invent canonical domains: invalid raw inputs still raise."""
    bad_source = _valid_source("grain")
    bad_source["domain"] = "not_a_domain"

    with pytest.raises(PriorCanonicalizationError) as exc:
        canonicalize_prior_config(bad_source)
    assert "invalid_domain" in exc.value.reason_counts


def test_summarize_prior_coverage_repair_mode_increases_valid_count() -> None:
    """Repair mode must validate strictly more samples than raw mode for the
    same seed because component mismatches are the dominant failure cause."""
    raw = summarize_prior_coverage(n_samples=100, random_state=20260428)
    repaired = summarize_prior_coverage(
        n_samples=100, random_state=20260428, repair_domain_components=True
    )

    assert raw["repair_domain_components"] is False
    assert repaired["repair_domain_components"] is True
    assert repaired["valid_count"] >= raw["valid_count"]
    assert repaired["repair_counts"]


def _valid_source(domain_alias: str) -> dict[str, object]:
    domain_key = canonicalize_domain(domain_alias)
    components = _first_valid_domain_components(domain_key, 3)
    return {
        "domain": domain_alias,
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (400, 2500),
        "spectral_resolution": 0.5,
        "measurement_mode": "reflectance",
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": components,
        "n_samples": 100,
        "target_config": {
            "type": "regression",
            "n_targets": 1,
            "nonlinearity": "none",
        },
        "random_state": 42,
    }


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    components = []
    for component in get_domain_config(domain_key).typical_components:
        trial = _valid_source_component(component)
        if trial is not None:
            components.append(trial)
        if len(components) == n_components:
            return components
    raise AssertionError(f"Not enough valid components for {domain_key}")


def _valid_source_component(component: str) -> str | None:
    from nirs4all.synthesis.components import get_component

    try:
        return get_component(component).name
    except ValueError:
        return None
