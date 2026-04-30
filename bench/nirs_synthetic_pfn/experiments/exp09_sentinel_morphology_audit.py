"""R2b sentinel morphology audit (bench-only, report-only).

Diagnostic lane that compares the current baseline synthetic prior to real
sentinel cohorts using purely morphological summary statistics in the
``uncalibrated_raw`` lane. The audit is non-gate, report-only, and never:

- fits any real-data calibration, marginal mapping, or covariance capture;
- runs PCA, adversarial AUC, or any ML/DL classifier;
- consumes labels, splits, targets, or any downstream gate feedback;
- modifies B2/B3/B4/B5 thresholds or metric definitions.

R2a live showed AUC=1.0 on simple profiles; this audit produces a reproducible
morphology snapshot to guide the next mechanistic correction. To keep the
audit's import surface free of calibration/PCA/adversarial-AUC code, this
module does NOT import ``exp02_real_synthetic_scorecards`` or
``exp08_mechanistic_sentinel_ablation``: every selection helper, deterministic
seed helper, and synthetic-baseline path is inlined locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from nirsyntheticpfn.adapters.builder_adapter import (
    ALL_REMEDIATION_PROFILES,
    R2C_REMEDIATION_PROFILES,
    R2D_REMEDIATION_PROFILES,
    R2F_REMEDIATION_PROFILES,
    R2G_REMEDIATION_PROFILES,
    R2H_REMEDIATION_PROFILES,
    R2I_REMEDIATION_PROFILES,
    R2J_REMEDIATION_PROFILES,
    R2K_REMEDIATION_PROFILES,
    R2L_REMEDIATION_PROFILES,
    R2M_REMEDIATION_PROFILES,
    R2N_REMEDIATION_PROFILES,
    R2O_REMEDIATION_PROFILES,
    R2P_REMEDIATION_PROFILES,
    R2Q_REMEDIATION_PROFILES,
    R2R_REMEDIATION_PROFILES,
    R2S_REMEDIATION_PROFILES,
    R2T_REMEDIATION_PROFILES,
    R2U_REMEDIATION_PROFILES,
    R2V_REMEDIATION_PROFILES,
    R2W_REMEDIATION_PROFILES,
    build_synthetic_dataset_run,
)
from nirsyntheticpfn.adapters.prior_adapter import canonicalize_domain, canonicalize_prior_config
from nirsyntheticpfn.evaluation.realism import (
    RealDataset,
    align_to_real_grid,
    discover_local_real_datasets,
    is_index_fallback_grid,
    load_real_spectra,
    sanitize_finite_spectra,
)

from nirs4all.synthesis.components import get_component
from nirs4all.synthesis.domains import get_domain_config

# ---------------------------------------------------------------------------
# Sentinel selection (inlined; matches R2a primary-first contract).
# ---------------------------------------------------------------------------

PRIMARY_SENTINEL_TOKENS: tuple[str, ...] = ("BEER", "DIESEL", "CORN")
SECONDARY_MILK_SENTINEL_TOKENS: tuple[str, ...] = ("MILK",)
SECONDARY_SOIL_SENTINEL_TOKENS: tuple[str, ...] = (
    "LUCAS",
    "PHOSPHORUS",
    "MANURE",
    "SOIL",
)
SECONDARY_FRUIT_SENTINEL_TOKENS: tuple[str, ...] = (
    "BERRY",
    "PEACH",
    "PLUMS",
    "FRUIT",
)
SENTINEL_PRIORITY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("primary", PRIMARY_SENTINEL_TOKENS),
    ("secondary_milk", SECONDARY_MILK_SENTINEL_TOKENS),
    ("secondary_soil", SECONDARY_SOIL_SENTINEL_TOKENS),
    ("secondary_fruit", SECONDARY_FRUIT_SENTINEL_TOKENS),
)
DEFAULT_SENTINEL_TOKENS: tuple[str, ...] = tuple(
    token for _, group in SENTINEL_PRIORITY_GROUPS for token in group
)
DEFAULT_MAX_SENTINEL_DATASETS = 8

DEFAULT_REPORT = Path("bench/nirs_synthetic_pfn/reports/r2b_sentinel_morphology_audit.md")
DEFAULT_CSV = Path("bench/nirs_synthetic_pfn/reports/r2b_sentinel_morphology_audit.csv")
R2B_AUDIT_SCOPE = "bench_only_r2b_sentinel_morphology_audit"
COMPARISON_SPACE = "uncalibrated_raw"
RATIO_EPS = 1e-12
DOMINANT_GAP_TOL = 1e-6


def _token_priority_map(tokens: Sequence[str]) -> dict[str, int]:
    group_lookup: dict[str, int] = {}
    for prio, (_name, group) in enumerate(SENTINEL_PRIORITY_GROUPS):
        for token in group:
            group_lookup[token.casefold()] = prio
    custom_priority = len(SENTINEL_PRIORITY_GROUPS)
    out: dict[str, int] = {}
    for token in tokens:
        cf = token.casefold().strip()
        if not cf:
            continue
        out[cf] = group_lookup.get(cf, custom_priority)
    return out


def _select_sentinel_datasets(
    datasets: Iterable[RealDataset],
    tokens: Sequence[str],
) -> list[RealDataset]:
    """Filter to datasets matching any token; sort by (priority, original index)."""
    priority_map = _token_priority_map(tokens)
    if not priority_map:
        return []
    decorated: list[tuple[int, int, RealDataset]] = []
    for idx, dataset in enumerate(datasets):
        haystack = " ".join(
            (dataset.source, dataset.task, dataset.database_name, dataset.dataset)
        ).casefold()
        matched = [prio for token_cf, prio in priority_map.items() if token_cf in haystack]
        if matched:
            decorated.append((min(matched), idx, dataset))
    decorated.sort(key=lambda item: (item[0], item[1]))
    return [dataset for _, _, dataset in decorated]


# ---------------------------------------------------------------------------
# Deterministic seed.
# ---------------------------------------------------------------------------


def _stable_dataset_seed(seed: int, dataset: RealDataset, purpose: str) -> int:
    key = "|".join(
        [
            str(seed),
            purpose,
            dataset.source,
            dataset.task,
            dataset.database_name,
            dataset.dataset,
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Synthetic baseline (inlined minimal preset selector + builder call).
# ---------------------------------------------------------------------------

# Default presets known to be supported by the upstream domain registry. The
# tuple is (domain_alias, target_type, target_size). The audit only needs
# spectra; targets are still requested so build_synthetic_dataset_run accepts
# the canonical config without adjustment.
_DEFAULT_PRESETS: tuple[tuple[str, str, int], ...] = (
    ("grain", "regression", 1),
    ("forage", "classification", 3),
    ("oilseeds", "regression", 1),
    ("fruit", "classification", 2),
    ("dairy", "regression", 1),
    ("meat", "classification", 3),
    ("wine", "regression", 1),
    ("juice", "regression", 1),
    ("baking", "regression", 1),
    ("tablets", "classification", 4),
    ("powders", "regression", 1),
    ("soil", "regression", 1),
    ("fuel", "classification", 2),
)
_PRESET_TARGETS: dict[str, tuple[str, int]] = {
    alias: (target_type, target_size)
    for alias, target_type, target_size in _DEFAULT_PRESETS
}
_PRESET_SOURCE_OVERRIDES: dict[str, dict[str, object]] = {
    "wine": {
        "measurement_mode": "transmittance",
        "matrix_type": "liquid",
        "particle_size": 5.0,
    },
    "juice": {
        "measurement_mode": "transmittance",
        "matrix_type": "liquid",
        "particle_size": 5.0,
    },
    "soil": {
        "measurement_mode": "reflectance",
        "matrix_type": "powder",
        "particle_size": 75.0,
    },
}

# R2c remediation token-based source overrides. Activated only when
# ``--remediation-profile`` is set. Overrides are derived from dataset/source
# tokens (DIESEL, MILK/DAIRY) and never read real spectra, labels, splits, or
# targets. Components rely on the bench-only ``BENCH_ONLY_COMPONENT_ALIASES``
# in ``prior_adapter.py`` when listing non-registered fuel labels (e.g.
# ``"diesel"``).
_R2C_TOKEN_SOURCE_OVERRIDES: tuple[tuple[tuple[str, ...], dict[str, object]], ...] = (
    (
        ("diesel",),
        {
            "components": ["diesel", "alkane", "aromatic"],
            "matrix_type": "liquid",
            "measurement_mode": "transmittance",
            "particle_size": 2.0,
        },
    ),
    (
        ("milk", "dairy"),
        {
            "components": ["water", "lipid", "casein", "lactose"],
            "matrix_type": "emulsion",
            "measurement_mode": "transflectance",
            "particle_size": 2.0,
        },
    ),
)

# R2d additional token-based overrides for BEER (beverage_wine domain) and
# CORN (agriculture_grain domain). Components are restricted to entries that
# resolve via the upstream component registry (no bench-only alias is required
# for these tokens). Activated only when ``--remediation-profile`` selects an
# r2d profile id; r2c continues to apply only the DIESEL/MILK overrides.
_R2D_EXTRA_TOKEN_SOURCE_OVERRIDES: tuple[
    tuple[tuple[str, ...], dict[str, object]], ...
] = (
    (
        ("beer",),
        {
            "components": ["water", "ethanol", "glucose", "fructose", "glycerol"],
            "matrix_type": "liquid",
            "measurement_mode": "transmittance",
            "particle_size": 2.0,
        },
    ),
    (
        ("corn",),
        {
            "components": ["starch", "protein", "moisture", "lipid", "cellulose"],
            "matrix_type": "powder",
            "measurement_mode": "reflectance",
            "particle_size": 75.0,
        },
    ),
)

# R2f additional token-based overrides for BERRY/juice sentinel rows.
# Components are restricted to valid beverage_juice entries. FruitPuree/puree
# intentionally stay unremediated until a separate puree-specific rule exists.
# Activated only for r2f profile ids; R2d intentionally remains unchanged.
_R2F_EXTRA_TOKEN_SOURCE_OVERRIDES: tuple[
    tuple[tuple[str, ...], dict[str, object]], ...
] = (
    (
        ("berry", "juice"),
        {
            "components": [
                "water",
                "glucose",
                "fructose",
                "sucrose",
                "citric_acid",
                "malic_acid",
                "carotenoid",
            ],
            "matrix_type": "liquid",
            "measurement_mode": "transmittance",
            "particle_size": 2.0,
        },
    ),
)

# R2g additional token-based overrides for soil sentinels. Components are
# restricted to valid environmental_soil entries. Activated only for r2g
# profile ids; R2f and earlier profiles intentionally keep soil routing stable.
_R2G_EXTRA_TOKEN_SOURCE_OVERRIDES: tuple[
    tuple[tuple[str, ...], dict[str, object]], ...
] = (
    (
        ("lucas", "phosphorus", "soil"),
        {
            "components": [
                "moisture",
                "carbonates",
                "kaolinite",
                "gypsum",
                "cellulose",
                "lignin",
                "protein",
            ],
            "matrix_type": "powder",
            "measurement_mode": "reflectance",
            "particle_size": 75.0,
        },
    ),
)

# R2h additional token-based overrides for BERRY juice sentinels. The builder
# handles the apparent percent-transmittance readout. FruitPuree remains
# excluded by the effective-profile routing below.
_R2H_EXTRA_TOKEN_SOURCE_OVERRIDES: tuple[
    tuple[tuple[str, ...], dict[str, object]], ...
] = (
    (
        ("berry",),
        {
            "components": [
                "water",
                "glucose",
                "fructose",
                "sucrose",
                "citric_acid",
                "malic_acid",
                "carotenoid",
            ],
            "matrix_type": "liquid",
            "measurement_mode": "transmittance",
            "particle_size": 10.0,
        },
    ),
)

# R2i FruitPuree-only override. This intentionally switches the synthetic
# source from beverage_juice to agriculture_fruit so the builder can use fruit
# tissue components and a paste/transflectance optical rule instead of the R2f
# clear-juice or R2h BERRY percent readout paths.
_R2I_FRUITPUREE_TOKEN_SOURCE_OVERRIDES: tuple[
    tuple[tuple[str, ...], dict[str, object]], ...
] = (
    (
        ("fruitpuree", "puree"),
        {
            "domain": "fruit",
            "components": [
                "water",
                "glucose",
                "fructose",
                "sucrose",
                "cellulose",
                "starch",
                "malic_acid",
                "citric_acid",
                "carotenoid",
            ],
            "matrix_type": "paste",
            "measurement_mode": "transflectance",
            "particle_size": 35.0,
            "wavelength_range": (900, 1100),
        },
    ),
)


def _r2c_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2c source overrides matching the dataset's textual tokens.

    Token matching uses the dataset's ``source``/``task``/``database_name``/
    ``dataset`` fields only. Never reads real spectra, labels, splits, or
    targets. The returned dict is empty when no token rule applies.
    """
    text, tokens = _dataset_text_and_tokens(dataset)
    for keywords, override in _R2C_TOKEN_SOURCE_OVERRIDES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                return dict(override)
    return {}


def _r2d_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2d source overrides matching the dataset's textual tokens.

    Superset of ``_r2c_token_source_overrides`` that additionally maps BEER
    and CORN datasets to ``beverage_wine``/``agriculture_grain`` source
    configs supported by the R2d ``_R2D_DOMAIN_RULES`` table. Same audit
    contract: token matching only, no real spectra/labels/splits/targets read.
    """
    r2c_override = _r2c_token_source_overrides(dataset)
    if r2c_override:
        return r2c_override
    text, tokens = _dataset_text_and_tokens(dataset)
    for keywords, override in _R2D_EXTRA_TOKEN_SOURCE_OVERRIDES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                return dict(override)
    return {}


def _r2f_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2f token overrides, extending R2d with berry/juice sentinels."""
    text, tokens = _dataset_text_and_tokens(dataset)
    r2d_override = _r2d_token_source_overrides(dataset)
    if r2d_override:
        return r2d_override
    if _has_puree_source_marker(text, tokens):
        return {}
    for keywords, override in _R2F_EXTRA_TOKEN_SOURCE_OVERRIDES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                return dict(override)
    return {}


def _r2g_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2g token overrides, extending R2f with soil sentinels."""
    text, tokens = _dataset_text_and_tokens(dataset)
    for keywords, override in _R2G_EXTRA_TOKEN_SOURCE_OVERRIDES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                return dict(override)
    r2f_override = _r2f_token_source_overrides(dataset)
    if r2f_override:
        return r2f_override
    if _has_puree_source_marker(text, tokens):
        return {}
    return {}


def _has_r2h_berry_marker(text: str, tokens: set[str]) -> bool:
    return "berry" in tokens or "berry" in text


def _r2h_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2h token overrides, extending R2g with BERRY juice sentinels."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2g_soil_marker(text, tokens):
        return _r2g_token_source_overrides(dataset)
    if _has_puree_source_marker(text, tokens):
        return {}
    for keywords, override in _R2H_EXTRA_TOKEN_SOURCE_OVERRIDES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                return dict(override)
    return _r2g_token_source_overrides(dataset)


def _r2i_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2i token overrides with FruitPuree separate from BERRY juice."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2g_soil_marker(text, tokens):
        return _r2g_token_source_overrides(dataset)
    if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
        dataset
    ):
        for keywords, override in _R2I_FRUITPUREE_TOKEN_SOURCE_OVERRIDES:
            for keyword in keywords:
                if keyword in tokens or keyword in text:
                    return dict(override)
    if _has_r2h_berry_marker(text, tokens):
        return _r2h_token_source_overrides(dataset)
    return _r2g_token_source_overrides(dataset)


def _has_r2j_diesel_marker(text: str, tokens: set[str]) -> bool:
    del text
    return "diesel" in tokens


def _has_r2l_lucas_marker(text: str, tokens: set[str]) -> bool:
    return "lucas" in tokens or "lucas" in text


def _has_r2m_milk_marker(text: str, tokens: set[str]) -> bool:
    return "milk" in tokens or "milk" in text or "dairy" in tokens or "dairy" in text


def _has_r2n_manure21_marker(text: str, tokens: set[str]) -> bool:
    return "manure21" in tokens or "manure21" in text


def _has_r2o_beer_marker(text: str, tokens: set[str]) -> bool:
    return "beer" in tokens or "beer" in text


def _has_r2p_phosphorus_marker(text: str, tokens: set[str]) -> bool:
    return "phosphorus" in tokens or "phosphorus" in text


def _has_r2q_lucas_ph_organic_marker(text: str, tokens: set[str]) -> bool:
    return (
        ("lucas" in tokens or "lucas" in text)
        and ("ph" in tokens or "ph" in text)
        and ("organic" in tokens or "organic" in text)
    )


def _r2m_milk_variant(text: str, tokens: set[str]) -> str:
    if "labels" in tokens or "kenstone70" in tokens or "kenstone70" in text:
        return "fullrange"
    return "shortwave"


def _r2j_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2j/R2k token overrides; only DIESEL changes vs R2i."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2j_diesel_marker(text, tokens):
        return _r2c_token_source_overrides(dataset)
    return _r2i_token_source_overrides(dataset)


def _r2l_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2l token overrides; only LUCAS rows use the R2l soil readout."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2l_lucas_marker(text, tokens):
        overrides = _r2g_token_source_overrides(dataset)
        overrides["_r2l_lucas_soil_route"] = {
            "enabled": True,
            "route_marker": "lucas",
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2j_token_source_overrides(dataset)


def _r2m_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2m token overrides; only MILK rows use the dairy readout."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2m_milk_marker(text, tokens):
        overrides = _r2c_token_source_overrides(dataset)
        overrides["_r2m_milk_readout_route"] = {
            "enabled": True,
            "route_marker": "milk",
            "variant": _r2m_milk_variant(text, tokens),
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2l_token_source_overrides(dataset)


def _r2n_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2n token overrides; only MANURE21 rows use the manure readout."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2n_manure21_marker(text, tokens):
        overrides: dict[str, object] = {
            "domain": "soil",
            "components": [
                "moisture",
                "cellulose",
                "lignin",
                "protein",
                "carbonates",
                "gypsum",
                "kaolinite",
            ],
            "matrix_type": "powder",
            "measurement_mode": "reflectance",
            "particle_size": 90.0,
            "_r2n_manure21_readout_route": {
                "enabled": True,
                "route_marker": "manure21",
                "source": "exp09_dataset_token",
                "non_oracle": True,
                "no_target_or_label": True,
                "real_stat_capture": False,
                "thresholds_modified": False,
            },
        }
        return overrides
    return _r2m_token_source_overrides(dataset)


def _r2o_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2o token overrides; only BEER rows use the beer readout."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2o_beer_marker(text, tokens):
        overrides = _r2d_token_source_overrides(dataset)
        overrides["_r2o_beer_readout_route"] = {
            "enabled": True,
            "route_marker": "beer",
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2n_token_source_overrides(dataset)


def _r2p_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2p token overrides; only PHOSPHORUS rows use the phosphate readout."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2p_phosphorus_marker(text, tokens):
        overrides = _r2g_token_source_overrides(dataset)
        overrides["_r2p_phosphorus_readout_route"] = {
            "enabled": True,
            "route_marker": "phosphorus",
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2o_token_source_overrides(dataset)


def _r2q_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2q overrides; only LUCAS pH Organic rows use humic topsoil."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2q_lucas_ph_organic_marker(text, tokens):
        overrides = _r2g_token_source_overrides(dataset)
        overrides["_r2q_lucas_ph_organic_readout_route"] = {
            "enabled": True,
            "route_marker": "lucas_ph_organic",
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2p_token_source_overrides(dataset)


def _r2r_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2r overrides, changing only explicitly marked FruitPuree rows."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
        dataset
    ):
        overrides = _r2i_token_source_overrides(dataset)
        if overrides:
            overrides["_r2r_fruitpuree_readout_route"] = {
                "enabled": True,
                "route_marker": "fruitpuree",
                "source": "exp09_dataset_token",
                "non_oracle": True,
                "no_target_or_label": True,
                "real_stat_capture": False,
                "thresholds_modified": False,
            }
            return overrides
    return _r2q_token_source_overrides(dataset)


def _r2s_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2s overrides, changing only explicitly marked DIESEL rows."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2j_diesel_marker(text, tokens):
        overrides = _r2c_token_source_overrides(dataset)
        overrides["_r2s_diesel_readout_route"] = {
            "enabled": True,
            "route_marker": "diesel",
            "source": "exp09_dataset_token",
            "non_oracle": True,
            "no_target_or_label": True,
            "real_stat_capture": False,
            "thresholds_modified": False,
        }
        return overrides
    return _r2r_token_source_overrides(dataset)


def _r2t_token_source_overrides(dataset: RealDataset) -> dict[str, object]:
    """Return R2t overrides, changing only explicitly marked MANURE21 rows."""
    text, tokens = _dataset_text_and_tokens(dataset)
    if _has_r2n_manure21_marker(text, tokens):
        return _r2n_token_source_overrides(dataset)
    return _r2s_token_source_overrides(dataset)


def _remediation_token_source_overrides(
    dataset: RealDataset, profile: str
) -> dict[str, object]:
    """Dispatch token-based source overrides per remediation profile."""
    if profile in R2W_REMEDIATION_PROFILES:
        return _r2t_token_source_overrides(dataset)
    if profile in R2U_REMEDIATION_PROFILES:
        return _r2t_token_source_overrides(dataset)
    if profile in R2V_REMEDIATION_PROFILES:
        return _r2t_token_source_overrides(dataset)
    if profile in R2T_REMEDIATION_PROFILES:
        return _r2t_token_source_overrides(dataset)
    if profile in R2S_REMEDIATION_PROFILES:
        return _r2s_token_source_overrides(dataset)
    if profile in R2R_REMEDIATION_PROFILES:
        return _r2r_token_source_overrides(dataset)
    if profile in R2Q_REMEDIATION_PROFILES:
        return _r2q_token_source_overrides(dataset)
    if profile in R2P_REMEDIATION_PROFILES:
        return _r2p_token_source_overrides(dataset)
    if profile in R2O_REMEDIATION_PROFILES:
        return _r2o_token_source_overrides(dataset)
    if profile in R2N_REMEDIATION_PROFILES:
        return _r2n_token_source_overrides(dataset)
    if profile in R2M_REMEDIATION_PROFILES:
        return _r2m_token_source_overrides(dataset)
    if profile in R2L_REMEDIATION_PROFILES:
        return _r2l_token_source_overrides(dataset)
    if profile in R2J_REMEDIATION_PROFILES + R2K_REMEDIATION_PROFILES:
        return _r2j_token_source_overrides(dataset)
    if profile in R2I_REMEDIATION_PROFILES:
        return _r2i_token_source_overrides(dataset)
    if profile in R2H_REMEDIATION_PROFILES:
        return _r2h_token_source_overrides(dataset)
    if profile in R2G_REMEDIATION_PROFILES:
        return _r2g_token_source_overrides(dataset)
    if profile in R2F_REMEDIATION_PROFILES:
        return _r2f_token_source_overrides(dataset)
    if profile in R2D_REMEDIATION_PROFILES:
        return _r2d_token_source_overrides(dataset)
    return _r2c_token_source_overrides(dataset)


def _effective_remediation_profile_for_dataset(
    dataset: RealDataset,
    profile: str | None,
) -> str | None:
    """Return the remediation profile that should reach the synthetic builder."""
    if profile in R2W_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2n_manure21_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2s_sentinel_matrix_v1",
        )
    if profile in R2V_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2n_manure21_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2s_sentinel_matrix_v1",
        )
    if profile in R2U_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2n_manure21_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2s_sentinel_matrix_v1",
        )
    if profile in R2T_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2n_manure21_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2s_sentinel_matrix_v1",
        )
    if profile in R2S_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2j_diesel_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2r_sentinel_matrix_v1",
        )
    if profile in R2R_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
            dataset
        ):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2q_sentinel_matrix_v1",
        )
    if profile in R2Q_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2q_lucas_ph_organic_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2p_sentinel_matrix_v1",
        )
    if profile in R2P_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2p_phosphorus_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2o_sentinel_matrix_v1",
        )
    if profile in R2O_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2o_beer_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2n_sentinel_matrix_v1",
        )
    if profile in R2N_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2n_manure21_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2m_sentinel_matrix_v1",
        )
    if profile in R2M_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2m_milk_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2l_sentinel_matrix_v1",
        )
    if profile in R2L_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2l_lucas_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2k_sentinel_matrix_v1",
        )
    if profile in R2J_REMEDIATION_PROFILES + R2K_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2j_diesel_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2i_sentinel_matrix_v1",
        )
    if profile in R2I_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_r2g_soil_marker(text, tokens):
            return _effective_remediation_profile_for_dataset(
                dataset,
                "r2g_sentinel_matrix_v1",
            )
        if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
            dataset
        ):
            return profile
        if _has_r2h_berry_marker(text, tokens):
            return _effective_remediation_profile_for_dataset(
                dataset,
                "r2h_sentinel_matrix_v1",
            )
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2g_sentinel_matrix_v1",
        )
    if profile in R2H_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
            dataset
        ):
            return None
        if _has_r2g_soil_marker(text, tokens):
            return _effective_remediation_profile_for_dataset(
                dataset,
                "r2g_sentinel_matrix_v1",
            )
        if _has_r2h_berry_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2g_sentinel_matrix_v1",
        )
    if profile in R2G_REMEDIATION_PROFILES:
        text, tokens = _dataset_text_and_tokens(dataset)
        if _has_puree_source_marker(text, tokens) and not _r2d_token_source_overrides(
            dataset
        ):
            return None
        if _has_r2g_soil_marker(text, tokens):
            return profile
        return _effective_remediation_profile_for_dataset(
            dataset,
            "r2f_sentinel_matrix_v1",
        )
    if profile not in R2F_REMEDIATION_PROFILES:
        return profile
    text, tokens = _dataset_text_and_tokens(dataset)
    if not _has_puree_source_marker(text, tokens):
        return profile
    if _r2d_token_source_overrides(dataset):
        return profile
    return None


# Matrix-first preset rules (mirrors the bench rules from the gate path so the
# sentinel rows map to the same spectroscopic preset family they would in the
# B2 audit, but without importing the gate path).
_MATRIX_FIRST_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("milk_dairy_matrix", "dairy", ("milk", "dairy", "cheese")),
    ("diesel_fuel_matrix", "fuel", ("diesel", "fuel")),
    ("tablet_pharma_matrix", "tablets", ("tablet", "pharma", "escitalopram")),
    ("beer_beverage_matrix", "wine", ("beer",)),
    ("berry_puree_beverage_matrix", "juice", ("fruitpuree", "puree", "juice", "berry")),
    ("leaf_plant_matrix", "forage", ("leaftraits", "darkresp", "ecosis", "arabidopsis", "leaf")),
    ("soil_mineral_matrix", "soil", ("lucas", "soil", "phosphorus", "quartz", "incombustible")),
    ("cassava_starch_root_proxy", "grain", ("cassava",)),
    ("corn_grain_matrix", "grain", ("corn",)),
    ("grain_cereal_matrix", "grain", ("rice", "wheat", "barley", "cereal", "amylose", "grain")),
    ("meat_matrix", "meat", ("beef", "meat", "marbling", "impurity")),
    ("oilseed_matrix", "oilseeds", ("colza", "oilseed")),
    ("baking_matrix", "baking", ("biscuit", "baking")),
    ("fruit_solid_matrix", "fruit", ("fruit", "peach", "plum", "grape", "pistacia")),
    ("powder_matrix", "powders", ("powder",)),
)
_DEFAULT_PRESET = "grain"


def _dataset_text_and_tokens(dataset: RealDataset) -> tuple[str, set[str]]:
    text = " ".join(
        [dataset.source, dataset.task, dataset.database_name, dataset.dataset]
    ).lower()
    return text, set(re.findall(r"[a-z0-9]+", text))


def _has_puree_source_marker(text: str, tokens: set[str]) -> bool:
    return (
        "fruitpuree" in tokens
        or "puree" in tokens
        or "fruitpuree" in text
        or "puree" in text
    )


def _has_r2g_soil_marker(text: str, tokens: set[str]) -> bool:
    return any(
        keyword in tokens or keyword in text
        for keyword in ("lucas", "phosphorus", "soil")
    )


def select_synthetic_preset_for_dataset(dataset: RealDataset) -> str:
    """Return the matrix-first preset alias for ``dataset``; default ``grain``."""
    text, tokens = _dataset_text_and_tokens(dataset)
    available = set(_PRESET_TARGETS)
    for _rule_id, preset, keywords in _MATRIX_FIRST_RULES:
        for keyword in keywords:
            if keyword in tokens or keyword in text:
                if preset in available:
                    return preset
                break
    return _DEFAULT_PRESET


def _first_valid_domain_components(domain_key: str, n_components: int) -> list[str]:
    components: list[str] = []
    for component in get_domain_config(domain_key).typical_components:
        try:
            components.append(get_component(str(component)).name)
        except ValueError:
            continue
        if len(components) == n_components:
            return components
    raise ValueError(f"Not enough executable components for {domain_key}")


def _preset_source(
    preset: str,
    *,
    seed: int,
    extra_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    target_type, target_size = _PRESET_TARGETS[preset]
    domain_key = canonicalize_domain(preset)
    components = _first_valid_domain_components(domain_key, max(3, target_size))
    target_config: dict[str, object]
    if target_type == "classification":
        target_config = {
            "type": "classification",
            "n_classes": target_size,
            "separation": "moderate",
        }
    else:
        target_config = {
            "type": "regression",
            "n_targets": target_size,
            "nonlinearity": "none",
        }
    source: dict[str, object] = {
        "domain": preset,
        "domain_category": "research",
        "instrument": "foss_xds",
        "instrument_category": "benchtop",
        "wavelength_range": (400, 2500),
        "spectral_resolution": 4.0,
        "measurement_mode": "reflectance",
        "matrix_type": "solid",
        "temperature": 25.0,
        "particle_size": 150.0,
        "noise_level": 1.0,
        "components": components,
        "n_samples": 100,
        "target_config": target_config,
        "random_state": seed,
    }
    source.update(_PRESET_SOURCE_OVERRIDES.get(preset, {}))
    if extra_overrides:
        source.update(extra_overrides)
    return source


def _build_baseline_synthetic_run(
    *,
    dataset: RealDataset,
    preset: str,
    n_samples: int,
    seed: int,
    remediation_profile: str | None = None,
) -> Any:
    """Build the baseline (or remediated) synthetic run for ``dataset``.

    Uses the upstream ``build_synthetic_dataset_run`` adapter on a
    canonicalized preset config; no per-dataset wavelength override, no
    calibration, no real-data capture. ``align_to_real_grid`` interpolates
    the resulting spectra onto the real grid downstream. When
    ``remediation_profile`` is set, bench-only overrides derived from
    dataset tokens are applied to the source config before canonicalization
    and the effective per-dataset profile is forwarded to the builder.
    """
    run_seed = _stable_dataset_seed(seed, dataset, f"r2b:on_demand_synthetic:{preset}")
    effective_remediation_profile = _effective_remediation_profile_for_dataset(
        dataset,
        remediation_profile,
    )
    extra_overrides: dict[str, object] | None = None
    if effective_remediation_profile is not None:
        extra_overrides = (
            _remediation_token_source_overrides(dataset, effective_remediation_profile)
            or None
        )
    source = _preset_source(preset, seed=run_seed, extra_overrides=extra_overrides)
    record = canonicalize_prior_config(source)
    return build_synthetic_dataset_run(
        record,
        n_samples=n_samples,
        random_seed=run_seed,
        remediation_profile=effective_remediation_profile,
    )


# ---------------------------------------------------------------------------
# Morphology metrics.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MorphologyRow:
    """One R2b morphology row for a sentinel dataset in uncalibrated_raw lane."""

    status: str  # "compared" | "blocked"
    source: str
    task: str
    dataset: str
    synthetic_preset: str
    effective_matrix_route: str | None
    comparison_space: str
    n_real_samples: int
    n_synthetic_samples: int
    n_wavelengths: int
    wavelength_min: float | None
    wavelength_max: float | None
    real_global_mean: float | None
    synthetic_global_mean: float | None
    global_mean_delta: float | None
    real_global_std: float | None
    synthetic_global_std: float | None
    global_std_ratio: float | None
    log10_global_std_ratio: float | None
    real_amplitude_p50: float | None
    synthetic_amplitude_p50: float | None
    amplitude_p50_ratio: float | None
    log10_amplitude_p50_ratio: float | None
    real_derivative_std_p50: float | None
    synthetic_derivative_std_p50: float | None
    derivative_std_p50_ratio: float | None
    log10_derivative_std_p50_ratio: float | None
    mean_curve_corr: float | None
    inverted_mean_curve_corr: float | None
    morphology_gap_score: float | None
    dominant_morphology_gap: str
    audit_oracle: bool
    audit_label_inputs_used: bool
    audit_target_inputs_used: bool
    audit_split_inputs_used: bool
    audit_source_oracle_used: bool
    audit_learned: bool
    audit_real_stat_capture: bool
    audit_thresholds_modified: bool
    audit_metrics_modified: bool
    audit_imputed: bool
    audit_replays_real_rows: bool
    audit_scope: str
    remediation_profile: str | None
    r2c_remediation_enabled: bool
    r2c_remediation_domain_key: str | None
    r2c_remediation_concentrations_applied: bool
    r2c_remediation_spectra_applied: bool
    r2c_remediation_spectra_rule: str | None
    r2c_remediation_composition_source: str | None
    r2c_remediation_spectra_source: str | None
    r2c_remediation_provenance_source: str | None
    r2c_remediation_route_variant: str | None
    r2c_remediation_constant_status: str | None
    r2c_remediation_readout_space: str | None
    r2c_remediation_calibration_source: str | None
    r2c_remediation_real_stat_source: str | None
    r2c_remediation_threshold_source: str | None
    blocked_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _audit_fields(scope: str = R2B_AUDIT_SCOPE) -> dict[str, Any]:
    return {
        "audit_oracle": False,
        "audit_label_inputs_used": False,
        "audit_target_inputs_used": False,
        "audit_split_inputs_used": False,
        "audit_source_oracle_used": False,
        "audit_learned": False,
        "audit_real_stat_capture": False,
        "audit_thresholds_modified": False,
        "audit_metrics_modified": False,
        "audit_imputed": False,
        "audit_replays_real_rows": False,
        "audit_scope": scope,
    }


def _audit_scope_for_profile(remediation_profile: str | None) -> str:
    if remediation_profile is None:
        return R2B_AUDIT_SCOPE
    family = remediation_profile.split("_", maxsplit=1)[0].casefold()
    return f"bench_only_{family}_sentinel_morphology_audit"


def _audit_label_for_profile(remediation_profile: str | None) -> str:
    if remediation_profile is None:
        return "R2b"
    family = remediation_profile.split("_", maxsplit=1)[0].casefold()
    return f"{family[:2].upper()}{family[2:]}"


def _remediation_fields_disabled(remediation_profile: str | None) -> dict[str, Any]:
    return {
        "remediation_profile": remediation_profile,
        "effective_matrix_route": None,
        "r2c_remediation_enabled": False,
        "r2c_remediation_domain_key": None,
        "r2c_remediation_concentrations_applied": False,
        "r2c_remediation_spectra_applied": False,
        "r2c_remediation_spectra_rule": None,
        "r2c_remediation_composition_source": None,
        "r2c_remediation_spectra_source": None,
        "r2c_remediation_provenance_source": None,
        "r2c_remediation_route_variant": None,
        "r2c_remediation_constant_status": None,
        "r2c_remediation_readout_space": None,
        "r2c_remediation_calibration_source": None,
        "r2c_remediation_real_stat_source": None,
        "r2c_remediation_threshold_source": None,
    }


def _remediation_fields_from_metadata(
    *,
    remediation_profile: str | None,
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    audit = (metadata or {}).get("r2c_mechanistic_remediation") or {}
    transform_params = audit.get("transform_params") or {}
    return {
        "remediation_profile": remediation_profile,
        "effective_matrix_route": _effective_matrix_route_from_metadata(
            audit=audit,
            transform_params=transform_params,
        ),
        "r2c_remediation_enabled": bool(audit.get("enabled", False)),
        "r2c_remediation_domain_key": audit.get("domain_key"),
        "r2c_remediation_concentrations_applied": bool(
            audit.get("applied_to_concentrations", False)
        ),
        "r2c_remediation_spectra_applied": bool(
            audit.get("applied_to_spectra", False)
        ),
        "r2c_remediation_spectra_rule": transform_params.get("spectra_rule"),
        "r2c_remediation_composition_source": transform_params.get("composition_source"),
        "r2c_remediation_spectra_source": transform_params.get("spectra_source"),
        "r2c_remediation_provenance_source": (
            transform_params.get("provenance_source")
            or transform_params.get("milk_readout_route_source")
            or transform_params.get("baseline_source")
            or transform_params.get("contrast_source")
            or transform_params.get("scatter_source")
        ),
        "r2c_remediation_route_variant": transform_params.get("milk_readout_variant"),
        "r2c_remediation_constant_status": transform_params.get("constant_status"),
        "r2c_remediation_readout_space": transform_params.get("readout_space"),
        "r2c_remediation_calibration_source": transform_params.get("calibration_source"),
        "r2c_remediation_real_stat_source": transform_params.get("real_stat_source"),
        "r2c_remediation_threshold_source": transform_params.get("threshold_source"),
    }


def _effective_matrix_route_from_metadata(
    *,
    audit: dict[str, Any],
    transform_params: dict[str, Any],
) -> str | None:
    """Return a report label for the matrix/readout route the builder applied."""
    if not bool(audit.get("enabled", False)):
        return None

    spectra_rule = transform_params.get("spectra_rule")
    domain_key = audit.get("domain_key")
    if (
        spectra_rule == "dried_manure_organic_mineral_albedo_scatter_readout"
        and transform_params.get("manure21_readout_route_marker") == "manure21"
    ):
        return "manure_organic_mineral_matrix"
    if (
        spectra_rule
        in {
            "dried_manure_heterogeneous_scatter_patch_readout",
            "dried_manure_bounded_centered_scatter_readout",
            "dried_manure_balanced_centered_scatter_readout",
            "dried_manure_albedo_variance_centered_scatter_readout",
        }
        and transform_params.get("manure21_readout_route_marker") == "manure21"
    ):
        return "manure_organic_mineral_matrix"
    if spectra_rule == "milk_emulsion_scatter_inverse_transflectance_readout":
        return "milk_dairy_matrix"
    if spectra_rule == "lucas_mineral_albedo_absorbance_floor_scatter_readout":
        return "lucas_mineral_organic_soil_matrix"
    if (
        spectra_rule == "phosphorus_mineral_fertilizer_albedo_residual_readout"
        and transform_params.get("phosphorus_readout_route_marker") == "phosphorus"
    ):
        return "phosphorus_mineral_soil_matrix"
    if (
        spectra_rule == "lucas_ph_organic_humic_albedo_oh_readout"
        and transform_params.get("lucas_ph_organic_readout_route_marker")
        == "lucas_ph_organic"
    ):
        return "lucas_ph_organic_humic_soil_matrix"
    if spectra_rule == "diffuse_powder_smoothing_and_scatter_compression":
        return "soil_mineral_matrix" if domain_key == "environmental_soil" else None
    if spectra_rule == "micro_path_fuel_ch_overtone_contrast_readout":
        return "diesel_fuel_matrix"
    if spectra_rule == "micro_path_fuel_transmission_absorbance_floor":
        return "diesel_fuel_matrix"
    if (
        spectra_rule == "fermented_beer_turbid_cuvette_absorbance_readout"
        and transform_params.get("beer_readout_route_marker") == "beer"
    ):
        return "beer_fermented_liquid_matrix"
    if spectra_rule == "long_liquid_optical_path_scale":
        return "beer_beverage_matrix"
    if spectra_rule == "powder_reflectance_smoothing_and_scatter":
        return "corn_grain_matrix" if domain_key == "agriculture_grain" else None
    if spectra_rule == "cloudy_berry_percent_transmittance_readout":
        return "berry_juice_matrix"
    if spectra_rule == "strawberry_puree_transflectance_residual_readout":
        return "fruit_puree_matrix"
    if spectra_rule == "semi_solid_fruit_puree_short_path_scatter_smoothing":
        return "fruit_puree_matrix"
    if domain_key is not None:
        return str(domain_key)
    return None


def _row_remediation_profile(
    dataset: RealDataset,
    requested_profile: str | None,
) -> str | None:
    """Return the per-row remediation profile reported from builder intent."""
    return _effective_remediation_profile_for_dataset(dataset, requested_profile)


def _per_row_amplitude(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.array([], dtype=float)
    high = np.percentile(X, 95.0, axis=1)
    low = np.percentile(X, 5.0, axis=1)
    return np.asarray(high - low, dtype=float)


def _per_row_derivative_std(X: np.ndarray) -> np.ndarray:
    if X.shape[1] < 2:
        return np.zeros(X.shape[0], dtype=float)
    diffs = np.diff(X, axis=1)
    return np.asarray(diffs.std(axis=1, ddof=0), dtype=float)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size != b.size or a.size < 2:
        return None
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    a_norm = float(np.linalg.norm(a_centered))
    b_norm = float(np.linalg.norm(b_centered))
    if a_norm <= 0.0 or b_norm <= 0.0:
        return None
    return float(np.dot(a_centered, b_centered) / (a_norm * b_norm))


def _log10_ratio(numerator: float, denominator: float) -> float:
    return float(np.log10((numerator + RATIO_EPS) / (denominator + RATIO_EPS)))


def compute_morphology_metrics(
    real_X: np.ndarray,
    synthetic_X: np.ndarray,
    wavelengths: np.ndarray,
) -> dict[str, Any]:
    """Compute morphology summary metrics on aligned real and synthetic spectra."""
    real = np.asarray(real_X, dtype=float)
    synth = np.asarray(synthetic_X, dtype=float)
    wl = np.asarray(wavelengths, dtype=float).ravel()
    if real.ndim != 2 or synth.ndim != 2:
        raise ValueError("expected 2D real/synthetic spectra")
    if real.shape[1] != synth.shape[1]:
        raise ValueError("real and synthetic must share the wavelength axis")

    real_global_mean = float(real.mean()) if real.size else None
    syn_global_mean = float(synth.mean()) if synth.size else None
    real_global_std = float(real.std(ddof=0)) if real.size else None
    syn_global_std = float(synth.std(ddof=0)) if synth.size else None

    global_mean_delta = (
        None
        if real_global_mean is None or syn_global_mean is None
        else float(syn_global_mean - real_global_mean)
    )

    if real_global_std is None or syn_global_std is None:
        global_std_ratio = None
        log10_global_std_ratio = None
    else:
        global_std_ratio = float(
            (syn_global_std + RATIO_EPS) / (real_global_std + RATIO_EPS)
        )
        log10_global_std_ratio = _log10_ratio(syn_global_std, real_global_std)

    real_amp = _per_row_amplitude(real)
    syn_amp = _per_row_amplitude(synth)
    real_amp_p50 = float(np.median(real_amp)) if real_amp.size else None
    syn_amp_p50 = float(np.median(syn_amp)) if syn_amp.size else None
    if real_amp_p50 is None or syn_amp_p50 is None:
        amplitude_ratio = None
        log10_amplitude_ratio = None
    else:
        amplitude_ratio = float((syn_amp_p50 + RATIO_EPS) / (real_amp_p50 + RATIO_EPS))
        log10_amplitude_ratio = _log10_ratio(syn_amp_p50, real_amp_p50)

    real_dstd = _per_row_derivative_std(real)
    syn_dstd = _per_row_derivative_std(synth)
    real_dstd_p50 = float(np.median(real_dstd)) if real_dstd.size else None
    syn_dstd_p50 = float(np.median(syn_dstd)) if syn_dstd.size else None
    if real_dstd_p50 is None or syn_dstd_p50 is None:
        derivative_ratio = None
        log10_derivative_ratio = None
    else:
        derivative_ratio = float(
            (syn_dstd_p50 + RATIO_EPS) / (real_dstd_p50 + RATIO_EPS)
        )
        log10_derivative_ratio = _log10_ratio(syn_dstd_p50, real_dstd_p50)

    real_mean_curve = real.mean(axis=0) if real.size else np.array([], dtype=float)
    syn_mean_curve = synth.mean(axis=0) if synth.size else np.array([], dtype=float)
    mean_corr = (
        _safe_pearson(real_mean_curve, syn_mean_curve)
        if real_mean_curve.size and syn_mean_curve.size
        else None
    )
    inverted_corr = -mean_corr if mean_corr is not None else None

    normalized_mean_shift: float | None
    if (
        global_mean_delta is None
        or real_global_std is None
    ):
        normalized_mean_shift = None
    else:
        normalized_mean_shift = float(
            global_mean_delta / (real_global_std + RATIO_EPS)
        )

    score_terms: list[float] = []
    for value in (
        log10_global_std_ratio,
        log10_amplitude_ratio,
        log10_derivative_ratio,
        normalized_mean_shift,
    ):
        if value is not None:
            score_terms.append(abs(value))
    if mean_corr is not None:
        score_terms.append(max(0.0, 1.0 - mean_corr))
    morphology_gap_score = float(sum(score_terms)) if score_terms else None

    dominant = _dominant_gap(
        log10_global_std_ratio=log10_global_std_ratio,
        log10_amplitude_ratio=log10_amplitude_ratio,
        log10_derivative_ratio=log10_derivative_ratio,
        normalized_mean_shift=normalized_mean_shift,
        mean_corr=mean_corr,
    )

    return {
        "n_wavelengths": int(wl.size),
        "wavelength_min": float(wl.min()) if wl.size else None,
        "wavelength_max": float(wl.max()) if wl.size else None,
        "real_global_mean": real_global_mean,
        "synthetic_global_mean": syn_global_mean,
        "global_mean_delta": global_mean_delta,
        "real_global_std": real_global_std,
        "synthetic_global_std": syn_global_std,
        "global_std_ratio": global_std_ratio,
        "log10_global_std_ratio": log10_global_std_ratio,
        "real_amplitude_p50": real_amp_p50,
        "synthetic_amplitude_p50": syn_amp_p50,
        "amplitude_p50_ratio": amplitude_ratio,
        "log10_amplitude_p50_ratio": log10_amplitude_ratio,
        "real_derivative_std_p50": real_dstd_p50,
        "synthetic_derivative_std_p50": syn_dstd_p50,
        "derivative_std_p50_ratio": derivative_ratio,
        "log10_derivative_std_p50_ratio": log10_derivative_ratio,
        "mean_curve_corr": mean_corr,
        "inverted_mean_curve_corr": inverted_corr,
        "morphology_gap_score": morphology_gap_score,
        "dominant_morphology_gap": dominant,
    }


def _dominant_gap(
    *,
    log10_global_std_ratio: float | None,
    log10_amplitude_ratio: float | None,
    log10_derivative_ratio: float | None,
    normalized_mean_shift: float | None,
    mean_corr: float | None,
) -> str:
    """Pick the dominant morphology gap label by largest scored magnitude.

    Each candidate contributes a non-negative magnitude on a comparable scale:

    - log-ratio magnitudes for variance / amplitude / derivative_std,
    - absolute normalized mean shift (delta over real std),
    - mean curve inversion magnitude ``-mean_corr`` (only when ``mean_corr < 0``).

    The label of the largest magnitude wins. Ties within ``DOMINANT_GAP_TOL``
    yield ``"mixed"``. ``"none"`` is returned when no metric is available.
    Mean-curve inversion competes with the other gaps; it never overrides a
    larger log-ratio.
    """
    candidates: list[tuple[str, float]] = []
    if log10_global_std_ratio is not None:
        label = "variance_over" if log10_global_std_ratio > 0 else "variance_under"
        candidates.append((label, abs(log10_global_std_ratio)))
    if log10_amplitude_ratio is not None:
        label = "amplitude_over" if log10_amplitude_ratio > 0 else "amplitude_under"
        candidates.append((label, abs(log10_amplitude_ratio)))
    if log10_derivative_ratio is not None:
        label = "derivative_over" if log10_derivative_ratio > 0 else "derivative_under"
        candidates.append((label, abs(log10_derivative_ratio)))
    if normalized_mean_shift is not None:
        candidates.append(("mean_shift", abs(normalized_mean_shift)))
    if mean_corr is not None and mean_corr < 0.0:
        candidates.append(("mean_curve_inversion", float(-mean_corr)))

    if not candidates:
        return "none"
    max_value = max(value for _, value in candidates)
    if max_value <= DOMINANT_GAP_TOL:
        return "mixed"
    leaders = [
        label for label, value in candidates if max_value - value <= DOMINANT_GAP_TOL
    ]
    if len(leaders) > 1:
        return "mixed"
    return leaders[0]


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------


def run_audit(
    *,
    root: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seed: int,
    sentinel_tokens: Sequence[str] | None = None,
    remediation_profile: str | None = None,
) -> dict[str, Any]:
    """Run the R2b morphology audit over the selected sentinel cohorts."""
    if remediation_profile is not None and remediation_profile not in ALL_REMEDIATION_PROFILES:
        raise ValueError(
            f"unknown remediation profile {remediation_profile!r}; "
            f"valid profiles are {list(ALL_REMEDIATION_PROFILES)}"
        )
    audit_scope = _audit_scope_for_profile(remediation_profile)
    tokens = (
        tuple(DEFAULT_SENTINEL_TOKENS)
        if sentinel_tokens is None
        else tuple(sentinel_tokens)
    )
    real_datasets, _ = discover_local_real_datasets(root)
    if max_sentinel_datasets <= 0:
        sentinel_candidates = list(real_datasets)
        selected = list(real_datasets)
    else:
        sentinel_candidates = _select_sentinel_datasets(real_datasets, tokens)
        selected = sentinel_candidates[:max_sentinel_datasets]

    rows: list[MorphologyRow] = []
    if not selected:
        return {
            "status": "blocked_no_real_data",
            "rows": rows,
            "real_runnable_count": len(real_datasets),
            "real_sentinel_candidate_count": len(sentinel_candidates),
            "real_selected_count": 0,
            "sentinel_tokens": list(tokens),
            "remediation_profile": remediation_profile,
        }

    for dataset in selected:
        preset = select_synthetic_preset_for_dataset(dataset)
        row_remediation_profile = _row_remediation_profile(dataset, remediation_profile)
        try:
            real_X_raw, real_wl_raw = load_real_spectra(dataset, root=root)
            if is_index_fallback_grid(real_wl_raw):
                rows.append(
                    _blocked_row(
                        dataset=dataset,
                        preset=preset,
                        blocked_reason="wavelength_grid_unknown: real wavelengths could not be parsed",
                        remediation_profile=row_remediation_profile,
                        audit_scope=audit_scope,
                    )
                )
                continue
            sanitized_real, sanitized_wl, _, real_blocked = sanitize_finite_spectra(
                real_X_raw, real_wl_raw, side="real"
            )
            if real_blocked is not None or sanitized_real is None or sanitized_wl is None:
                rows.append(
                    _blocked_row(
                        dataset=dataset,
                        preset=preset,
                        blocked_reason=f"non_finite_spectra: {real_blocked}",
                        remediation_profile=row_remediation_profile,
                        audit_scope=audit_scope,
                    )
                )
                continue
            real_X = _downsample_rows(
                sanitized_real,
                max_rows=max_real_samples,
                random_state=_stable_dataset_seed(seed, dataset, "r2b:real_downsample"),
            )
            real_wl = sanitized_wl

            synthetic_run = _build_baseline_synthetic_run(
                dataset=dataset,
                preset=preset,
                n_samples=n_synthetic_samples,
                seed=seed,
                remediation_profile=remediation_profile,
            )
            synth_X = np.asarray(synthetic_run.X, dtype=float)
            synth_wl = np.asarray(synthetic_run.wavelengths, dtype=float)

            sanitized_syn, sanitized_syn_wl, _, syn_blocked = sanitize_finite_spectra(
                synth_X, synth_wl, side="synthetic"
            )
            if syn_blocked is not None or sanitized_syn is None or sanitized_syn_wl is None:
                rows.append(
                    _blocked_row(
                        dataset=dataset,
                        preset=preset,
                        blocked_reason=f"non_finite_spectra_synthetic: {syn_blocked}",
                        remediation_profile=row_remediation_profile,
                        synthetic_metadata=synthetic_run.metadata,
                        audit_scope=audit_scope,
                    )
                )
                continue
            synth_X = _downsample_rows(
                sanitized_syn,
                max_rows=max_real_samples,
                random_state=_stable_dataset_seed(seed, dataset, "r2b:syn_downsample"),
            )
            synth_wl = sanitized_syn_wl

            real_aligned, syn_aligned, aligned_wl = align_to_real_grid(
                real_X, real_wl, synth_X, synth_wl
            )
            metrics = compute_morphology_metrics(real_aligned, syn_aligned, aligned_wl)

            rows.append(
                MorphologyRow(
                    status="compared",
                    source=dataset.source,
                    task=dataset.task,
                    dataset=f"{dataset.database_name}/{dataset.dataset}",
                    synthetic_preset=preset,
                    comparison_space=COMPARISON_SPACE,
                    n_real_samples=int(real_aligned.shape[0]),
                    n_synthetic_samples=int(syn_aligned.shape[0]),
                    n_wavelengths=metrics["n_wavelengths"],
                    wavelength_min=metrics["wavelength_min"],
                    wavelength_max=metrics["wavelength_max"],
                    real_global_mean=metrics["real_global_mean"],
                    synthetic_global_mean=metrics["synthetic_global_mean"],
                    global_mean_delta=metrics["global_mean_delta"],
                    real_global_std=metrics["real_global_std"],
                    synthetic_global_std=metrics["synthetic_global_std"],
                    global_std_ratio=metrics["global_std_ratio"],
                    log10_global_std_ratio=metrics["log10_global_std_ratio"],
                    real_amplitude_p50=metrics["real_amplitude_p50"],
                    synthetic_amplitude_p50=metrics["synthetic_amplitude_p50"],
                    amplitude_p50_ratio=metrics["amplitude_p50_ratio"],
                    log10_amplitude_p50_ratio=metrics["log10_amplitude_p50_ratio"],
                    real_derivative_std_p50=metrics["real_derivative_std_p50"],
                    synthetic_derivative_std_p50=metrics["synthetic_derivative_std_p50"],
                    derivative_std_p50_ratio=metrics["derivative_std_p50_ratio"],
                    log10_derivative_std_p50_ratio=metrics["log10_derivative_std_p50_ratio"],
                    mean_curve_corr=metrics["mean_curve_corr"],
                    inverted_mean_curve_corr=metrics["inverted_mean_curve_corr"],
                    morphology_gap_score=metrics["morphology_gap_score"],
                    dominant_morphology_gap=metrics["dominant_morphology_gap"],
                    **_audit_fields(audit_scope),
                    **_remediation_fields_from_metadata(
                        remediation_profile=row_remediation_profile,
                        metadata=synthetic_run.metadata,
                    ),
                    blocked_reason="",
                )
            )
        except Exception as exc:  # noqa: BLE001 - bench-only diagnostic surfacing
            rows.append(
                _blocked_row(
                    dataset=dataset,
                    preset=preset,
                    blocked_reason=f"{type(exc).__name__}: {exc}",
                    remediation_profile=row_remediation_profile,
                    audit_scope=audit_scope,
                )
            )

    compared = [row for row in rows if row.status == "compared"]
    status = "done" if compared else "blocked_no_successful_comparisons"
    return {
        "status": status,
        "rows": rows,
        "real_runnable_count": len(real_datasets),
        "real_sentinel_candidate_count": len(sentinel_candidates),
        "real_selected_count": len(selected),
        "sentinel_tokens": list(tokens),
        "remediation_profile": remediation_profile,
    }


def _blocked_row(
    *,
    dataset: RealDataset,
    preset: str,
    blocked_reason: str,
    remediation_profile: str | None = None,
    synthetic_metadata: dict[str, Any] | None = None,
    audit_scope: str = R2B_AUDIT_SCOPE,
) -> MorphologyRow:
    if synthetic_metadata is not None:
        remediation_fields = _remediation_fields_from_metadata(
            remediation_profile=remediation_profile,
            metadata=synthetic_metadata,
        )
    else:
        remediation_fields = _remediation_fields_disabled(remediation_profile)
    return MorphologyRow(
        status="blocked",
        source=dataset.source,
        task=dataset.task,
        dataset=f"{dataset.database_name}/{dataset.dataset}",
        synthetic_preset=preset,
        comparison_space=COMPARISON_SPACE,
        n_real_samples=0,
        n_synthetic_samples=0,
        n_wavelengths=0,
        wavelength_min=None,
        wavelength_max=None,
        real_global_mean=None,
        synthetic_global_mean=None,
        global_mean_delta=None,
        real_global_std=None,
        synthetic_global_std=None,
        global_std_ratio=None,
        log10_global_std_ratio=None,
        real_amplitude_p50=None,
        synthetic_amplitude_p50=None,
        amplitude_p50_ratio=None,
        log10_amplitude_p50_ratio=None,
        real_derivative_std_p50=None,
        synthetic_derivative_std_p50=None,
        derivative_std_p50_ratio=None,
        log10_derivative_std_p50_ratio=None,
        mean_curve_corr=None,
        inverted_mean_curve_corr=None,
        morphology_gap_score=None,
        dominant_morphology_gap="none",
        **_audit_fields(audit_scope),
        **remediation_fields,
        blocked_reason=blocked_reason,
    )


def _downsample_rows(X: np.ndarray, *, max_rows: int, random_state: int) -> np.ndarray:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(X.shape[0], size=max_rows, replace=False)
    indices.sort()
    return np.asarray(X[indices], dtype=float)


def _csv_fieldnames() -> list[str]:
    return [field.name for field in fields(MorphologyRow)]


def write_csv(rows: list[MorphologyRow], path: Path) -> None:
    """Write rows to ``path``; always emits a stable header even when empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(
    *,
    result: dict[str, Any],
    report_path: Path,
    csv_path: Path,
    n_synthetic_samples: int,
    max_real_samples: int,
    max_sentinel_datasets: int,
    seed: int,
    sentinel_tokens: Sequence[str] | None = None,
    remediation_profile: str | None = None,
) -> str:
    rows: list[MorphologyRow] = result["rows"]
    tokens = (
        list(sentinel_tokens)
        if sentinel_tokens is not None
        else list(result.get("sentinel_tokens", DEFAULT_SENTINEL_TOKENS))
    )
    if remediation_profile is None:
        remediation_profile = result.get("remediation_profile")
    audit_label = _audit_label_for_profile(remediation_profile)
    audit_scope = _audit_scope_for_profile(remediation_profile)
    compared = [row for row in rows if row.status == "compared"]
    blocked = [row for row in rows if row.status == "blocked"]
    remediation_arg = (
        f" --remediation-profile {remediation_profile}"
        if remediation_profile is not None
        else ""
    )
    command = (
        "PYTHONPATH=bench/nirs_synthetic_pfn/src "
        "python bench/nirs_synthetic_pfn/experiments/exp09_sentinel_morphology_audit.py "
        f"--n-synthetic-samples {n_synthetic_samples} "
        f"--max-real-samples {max_real_samples} "
        f"--max-sentinel-datasets {max_sentinel_datasets} "
        f"--sentinel-tokens {','.join(tokens)} "
        f"--seed {seed}"
        f"{remediation_arg}"
    )

    lines: list[str] = [
        f"# {audit_label} Sentinel Morphology Audit",
        "",
        "## Scope and Non-Gate Disclaimer",
        "",
        "- Bench-only, report-only diagnostic lane. Comparison space is `uncalibrated_raw` only.",
        "- This audit does NOT establish any B2/B3/B4/B5 pass and does not modify any gate threshold or metric.",
        "- no calibration fitted, captured, or applied (no marginal, covariance, or quantile mapping).",
        "- no PCA/covariance capture from real data; no adversarial AUC; no ML/DL.",
        "- no labels, splits, targets, or downstream feedback are consulted.",
        "- Reporting note: `synthetic_preset` is the selector preset used to seed the canonical synthetic source; `effective_matrix_route` is the builder-reported remediation matrix/readout route when metadata is present.",
        "",
        "## Command",
        "",
        f"`{command}`",
        "",
        "## Outputs",
        "",
        f"- Markdown: `{report_path}`",
        f"- CSV: `{csv_path}`",
        "",
        "## Config",
        "",
        f"- Seed: {seed}",
        f"- Synthetic samples per dataset: {n_synthetic_samples}",
        f"- Real samples per dataset cap: {max_real_samples}",
        f"- Sentinel dataset cap: {max_sentinel_datasets if max_sentinel_datasets > 0 else 'all runnable rows'}",
        f"- Sentinel tokens: `{', '.join(tokens)}`"
        + ("" if max_sentinel_datasets > 0 else " (ignored when cap <= 0)"),
        f"- Remediation profile: `{remediation_profile if remediation_profile is not None else 'none'}`",
        "",
    ]
    if remediation_profile in R2H_REMEDIATION_PROFILES:
        lines.extend(
            [
                "## R2h Constant Provenance",
                "",
                "- R2h applies only to BERRY-routed `beverage_juice` rows; non-BERRY rows keep the R2g/R2f/R2d/R2c rule table behavior.",
                "- The cloudy-berry constants are fixed mechanistic optical priors for raw instrument axes reported as apparent percent-transmittance or intensity, not a calibration layer.",
                "- No real sentinel spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, or downstream result was read to set these constants.",
                "- The spectra mechanism is Beer-Lambert-style absorbance to apparent transmission/intensity: generated absorbance is multiplied by a fixed prior path-factor draw, mapped through `10^-A`, then placed on a bounded detector percent/intensity axis with a broadband turbidity offset.",
                "- The readout is exposed as `cloudy_berry_percent_transmittance_readout` in `readout/rule`; the builder audit records `constant_status=fixed_mechanistic_prior`, `readout_space=apparent_percent_transmittance_intensity`, `calibration_source=none`, `real_stat_source=none`, and `threshold_source=none`.",
                "",
                "## Validity and Risk Notes",
                "",
                "- Valid only as a bench-only raw readout hypothesis for cloudy berry juice datasets whose instrument axis is already percent-transmittance or intensity-like.",
                "- Amplitude remains unvalidated: the BERRY rows still show `amplitude_over`, so these constants are not a scientific pass and not a B2/B3/B4/B5 gate.",
                "- The report is single-seed diagnostic evidence. Any candidate follow-up requires repeated seeds, robust/adversarial checks, real-vs-real context, and explicit failure notes before gate reconsideration.",
                "- This path must remain free of PCA/statistical capture, real-fit calibration, ML/DL, threshold tuning, and downstream feedback.",
                "",
            ]
        )
    if remediation_profile in R2I_REMEDIATION_PROFILES:
        lines.extend(
            [
                "## R2i FruitPuree Provenance",
                "",
                "- R2i routes BERRY rows through the existing R2h BERRY readout, soil rows through R2g, and other non-soil sentinels through R2f.",
                "- FruitPuree/puree rows are separate: the source config is switched to `agriculture_fruit` tissue components with `matrix_type=paste` and `measurement_mode=transflectance`.",
                "- The FruitPuree spectra rule is `semi_solid_fruit_puree_short_path_scatter_smoothing`; it does not reuse `cloudy_berry_percent_transmittance_readout`.",
                "- The builder audit records `constant_status=fixed_mechanistic_prior`, `readout_space=semi_solid_puree_raw_absorbance`, `calibration_source=none`, `real_stat_source=none`, and `threshold_source=none`.",
                "- This is a single-seed diagnostic profile only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2J_REMEDIATION_PROFILES:
        lines.extend(
            [
                "## R2j DIESEL Provenance",
                "",
                "- R2j inherits R2i routing and changes only DIESEL/petrochem fuel rows; FruitPuree remains R2i, BERRY remains R2h, soil remains R2g, and other non-soil sentinels remain R2f.",
                "- The DIESEL spectra rule is `micro_path_fuel_transmission_absorbance_floor`: generated absorbance is compressed by a fixed Beer-Lambert micro-path factor and a small detector/dark-current absorbance floor is added.",
                "- Constants are fixed fuel optical/readout priors: `path_factor_range=[0.03, 0.05]`, `additive_baseline_range=[0.0005, 0.002]`, `readout_space=micro_path_raw_absorbance`.",
                "- No real DIESEL spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                "- Risk note: this rule is a wavelength-uniform multiplicative compression plus a row-uniform detector floor; when DIESEL rows report `derivative_under`, treat that as possible loss of spectral structure, not evidence of realistic fuel morphology.",
                "- This is a single-seed diagnostic profile only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2K_REMEDIATION_PROFILES:
        diesel_rows = [
            row
            for row in compared
            if row.synthetic_preset == "fuel" or "diesel" in row.dataset.casefold()
        ]
        diesel_derivative_under = sum(
            1
            for row in diesel_rows
            if row.dominant_morphology_gap == "derivative_under"
        )
        lines.extend(
            [
                "## R2k DIESEL Provenance",
                "",
                "- R2k inherits R2i routing and changes only DIESEL/petrochem fuel rows; FruitPuree remains R2i, BERRY remains R2h, soil remains R2g, and other non-soil sentinels remain R2f.",
                "- The DIESEL spectra rule is `micro_path_fuel_ch_overtone_contrast_readout`: generated fuel absorbance is split into a broad continuum and residual CH-band structure.",
                "- The continuum receives a fixed micro-path Beer-Lambert factor, while residual spectral structure keeps a fixed hydrocarbon feature-contrast factor and a wavelength-dependent CH overtone path perturbation.",
                "- Constants are fixed petrochemical optical priors: `path_factor_range=[0.055, 0.085]`, `feature_contrast_range=[0.24, 0.34]`, `continuum_smoothing_fwhm_nm=96.0`, `ch_overtone_centers_nm=[1150.0, 1210.0, 1390.0, 1460.0, 1720.0]`, `additive_baseline_range=[0.0005, 0.002]`, `readout_space=micro_path_ch_overtone_raw_absorbance`.",
                "- No real DIESEL spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                "- Tradeoff note: R2j reduced the DIESEL gap more aggressively with a wavelength-uniform micro-path compression, but its repeated-seed audit reported `derivative_under=9/9`; R2k accepts a higher DIESEL gap to preserve derivative structure better under a fixed CH overtone prior.",
                f"- Derivative-under warning: DIESEL rows with `derivative_under` = {diesel_derivative_under}/{len(diesel_rows)}. If this is all compared DIESEL rows, R2k must be treated as `needs-review`, not `done`.",
                "- This is a single-seed diagnostic profile only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2L_REMEDIATION_PROFILES:
        lucas_rows = [
            row
            for row in compared
            if "lucas" in row.dataset.casefold()
        ]
        phosphorus_rows = [
            row
            for row in compared
            if "phosphorus" in row.dataset.casefold()
        ]
        phosphorus_on_r2g = sum(
            1
            for row in phosphorus_rows
            if row.remediation_profile == "r2g_sentinel_matrix_v1"
        )
        lines.extend(
            [
                "## R2l LUCAS Soil Provenance",
                "",
                "- R2l inherits R2k routing, then changes only LUCAS-marked soil rows in this audit path; PHOSPHORUS rows are deliberately routed back to R2g.",
                "- The LUCAS spectra rule is `lucas_mineral_albedo_absorbance_floor_scatter_readout`: generated soil absorbance is smoothed, compressed as a diffuse-scatter residual, and placed on a fixed mineral-albedo apparent absorbance floor.",
                "- Constants are fixed optical priors: `additive_baseline_range=[0.30103, 0.30103]` from `A=-log10(0.5)`, `path_factor_range=[0.20, 0.25]`, `smoothing_fwhm_nm=24.0`, `readout_space=lucas_raw_soil_apparent_absorbance`.",
                "- No real LUCAS/PHOSPHORUS spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- PHOSPHORUS preservation check: rows reported on R2g = {phosphorus_on_r2g}/{len(phosphorus_rows)}.",
                f"- LUCAS diagnostic rows under R2l = {len(lucas_rows)}. This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2M_REMEDIATION_PROFILES:
        milk_rows = [
            row
            for row in compared
            if "milk" in row.dataset.casefold()
        ]
        non_milk_rows = [
            row
            for row in compared
            if "milk" not in row.dataset.casefold()
        ]
        changed_non_milk = sum(
            1
            for row in non_milk_rows
            if row.remediation_profile == "r2m_sentinel_matrix_v1"
        )
        milk_inversions = sum(
            1
            for row in milk_rows
            if row.dominant_morphology_gap == "mean_curve_inversion"
        )
        milk_variance_under = sum(
            1
            for row in milk_rows
            if row.dominant_morphology_gap == "variance_under"
        )
        lines.extend(
            [
                "## R2m MILK Dairy Provenance",
                "",
                "- R2m inherits R2l routing and changes only MILK/`food_dairy` rows in this audit path; non-MILK rows remain on their R2l effective routes.",
                "- The MILK spectra rule is `milk_emulsion_scatter_inverse_transflectance_readout`: generated milk absorbance is smoothed, clipped to non-negative attenuation, multiplied by a fixed short transflectance path factor, and exposed through `10^-A` on a fat-globule scatter continuum.",
                "- Constants are fixed dairy optical/readout priors: `path_factor_range=[0.55, 0.85]`, `smoothing_fwhm_nm=18.0`, `scatter_exponent_range=[1.0, 1.6]`, `detector_offset_range=[0.0, 0.04]`, `readout_space=milk_raw_transflectance_intensity`.",
                "- The three `Milk_*_1224_KS` regression rows use the shortwave detector-gain prior; `labels_kenstone70_strat` uses the compact full-range detector-gain prior because its token route represents the full-range readout where the 1940 nm water band is in scope.",
                "- The shortwave/full-range branch is selected only from the explicit bench-only MILK dataset-token route marker recorded in builder metadata; invalid or missing route provenance falls back to inherited R2l routing.",
                "- No real MILK spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- MILK diagnostic rows under R2m = {len(milk_rows)}; rows still dominated by mean-curve inversion = {milk_inversions}/{len(milk_rows)}.",
                f"- Remaining MILK `variance_under` rows = {milk_variance_under}/{len(milk_rows)}; this remains a reported failure mode, not a pass.",
                f"- Non-MILK rows accidentally reported on R2m = {changed_non_milk}/{len(non_milk_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2N_REMEDIATION_PROFILES:
        manure_rows = [
            row
            for row in compared
            if "manure21" in row.dataset.casefold()
        ]
        non_manure_rows = [
            row
            for row in compared
            if "manure21" not in row.dataset.casefold()
        ]
        changed_non_manure = sum(
            1
            for row in non_manure_rows
            if row.remediation_profile == "r2n_sentinel_matrix_v1"
        )
        manure_mean_shift = sum(
            1
            for row in manure_rows
            if row.dominant_morphology_gap == "mean_shift"
        )
        manure_amplitude_under = sum(
            1
            for row in manure_rows
            if row.dominant_morphology_gap == "amplitude_under"
        )
        lines.extend(
            [
                "## R2n MANURE21 Manure Provenance",
                "",
                "- R2n inherits R2m routing and changes only explicit MANURE21-marked dried/ground manure rows in this audit path; LUCAS remains R2l, PHOSPHORUS remains R2g, and MILK remains R2m.",
                "- The MANURE21 spectra rule is `dried_manure_organic_mineral_albedo_scatter_readout`: generated organic-mineral powder absorbance is smoothed, compressed as a diffuse-scatter residual, and placed on a fixed dark organic-albedo apparent absorbance continuum.",
                "- The composition route uses residual moisture, cellulose, lignin, protein, carbonates, gypsum, and kaolinite; it deliberately avoids the CORN starch-grain prior and does not apply the LUCAS mineral-albedo soil readout by domain alone.",
                "- Reporting clarity: MANURE21 rows can keep `synthetic_preset=grain` as the selector fallback while reporting `effective_matrix_route=manure_organic_mineral_matrix` for the actual R2n builder route.",
                "- Constants are fixed manure optical/matrix priors: `additive_baseline_range=[0.60, 0.78]`, `path_factor_range=[0.30, 0.42]`, `smoothing_fwhm_nm=20.0`, `readout_space=dried_ground_manure_raw_apparent_absorbance`.",
                "- The route is selected only from the explicit bench-only MANURE21 dataset-token marker recorded in builder metadata; invalid or missing route provenance falls back to inherited R2m routing.",
                "- No real MANURE21 spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- MANURE21 diagnostic rows under R2n = {len(manure_rows)}; rows still dominated by `mean_shift` = {manure_mean_shift}/{len(manure_rows)}.",
                f"- Remaining MANURE21 `amplitude_under` rows = {manure_amplitude_under}/{len(manure_rows)}; this is the current failure mode, not a pass.",
                f"- Non-MANURE rows accidentally reported on R2n = {changed_non_manure}/{len(non_manure_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2O_REMEDIATION_PROFILES:
        beer_rows = [
            row
            for row in compared
            if "beer" in row.dataset.casefold()
        ]
        non_beer_rows = [
            row
            for row in compared
            if "beer" not in row.dataset.casefold()
        ]
        changed_non_beer = sum(
            1
            for row in non_beer_rows
            if row.remediation_profile == "r2o_sentinel_matrix_v1"
        )
        beer_mean_shift = sum(
            1
            for row in beer_rows
            if row.dominant_morphology_gap == "mean_shift"
        )
        beer_variance_under = sum(
            1
            for row in beer_rows
            if row.dominant_morphology_gap == "variance_under"
        )
        lines.extend(
            [
                "## R2o BEER Fermented-Liquid Provenance",
                "",
                "- R2o inherits R2n routing and changes only explicit BEER-marked fermented liquid rows in this audit path; BERRY remains R2h, FruitPuree remains R2i, DIESEL remains R2k, LUCAS remains R2l, MILK remains R2m, MANURE21 remains R2n, and PHOSPHORUS remains R2g.",
                "- The BEER spectra rule is `fermented_beer_turbid_cuvette_absorbance_readout`: generated beer absorbance is smoothed, multiplied by a fixed long cuvette path factor, and exposed on a raw apparent absorbance axis with fixed haze/carbonation broadband attenuation.",
                "- This route is specific to fermented beer liquid; it does not reuse the BERRY percent-transmittance/intensity readout or the FruitPuree paste/transflectance readout.",
                "- Constants are fixed beer optical/readout priors: `path_factor_range=[1.75, 2.35]`, `haze_absorbance_baseline_range=[1.75, 2.15]`, `haze_slope_absorbance_range=[0.06, 0.18]`, `carbonation_residual_absorbance_range=[0.0, 0.05]`, `smoothing_fwhm_nm=10.0`, `readout_space=fermented_beer_raw_apparent_absorbance`.",
                "- The route is selected only from the explicit bench-only BEER dataset-token marker recorded in builder metadata; invalid or missing route provenance falls back to inherited R2n routing.",
                "- No real BEER spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- BEER diagnostic rows under R2o = {len(beer_rows)}; rows still dominated by `mean_shift` = {beer_mean_shift}/{len(beer_rows)}.",
                f"- Remaining BEER `variance_under` rows = {beer_variance_under}/{len(beer_rows)}; this remains a reported failure mode, not a pass.",
                f"- Non-BEER rows accidentally reported on R2o = {changed_non_beer}/{len(non_beer_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2P_REMEDIATION_PROFILES:
        phosphorus_rows = [
            row
            for row in compared
            if "phosphorus" in row.dataset.casefold()
        ]
        non_phosphorus_rows = [
            row
            for row in compared
            if "phosphorus" not in row.dataset.casefold()
        ]
        changed_non_phosphorus = sum(
            1
            for row in non_phosphorus_rows
            if row.remediation_profile == "r2p_sentinel_matrix_v1"
        )
        phosphorus_mean_shift = sum(
            1
            for row in phosphorus_rows
            if row.dominant_morphology_gap == "mean_shift"
        )
        phosphorus_inversion = sum(
            1
            for row in phosphorus_rows
            if row.dominant_morphology_gap == "mean_curve_inversion"
        )
        phosphorus_derivative_over = sum(
            1
            for row in phosphorus_rows
            if row.dominant_morphology_gap == "derivative_over"
        )
        lines.extend(
            [
                "## R2p PHOSPHORUS Mineral-Soil Provenance",
                "",
                "- R2p inherits R2o routing and changes only explicit PHOSPHORUS-marked mineral/phosphate soil rows in this audit path; BEER remains R2o, BERRY remains R2h, FruitPuree remains R2i, DIESEL remains R2k, LUCAS remains R2l, MILK remains R2m, and MANURE21 remains R2n.",
                "- The PHOSPHORUS spectra rule is `phosphorus_mineral_fertilizer_albedo_residual_readout`: generated mineral-soil absorbance is smoothed, centered row-wise, inverted as an apparent diffuse-albedo residual, and placed around a fixed phosphate/mineral albedo continuum.",
                "- This route deliberately does not reuse the LUCAS R2l continuum: LUCAS uses a darker fixed topsoil floor `A=-log10(0.5)=0.30103` plus residual compression, while PHOSPHORUS keeps phosphate/mineral powder amplitude around a brighter fixed albedo continuum `A=-log10(0.63)`.",
                "- Constants are fixed mineral/phosphate optical priors: `additive_baseline_range=[0.195, 0.210]`, `path_factor_range=[0.95, 1.05]`, `smoothing_fwhm_nm=24.0`, `readout_space=phosphorus_raw_mineral_soil_apparent_absorbance`.",
                "- The route is selected only from the explicit bench-only PHOSPHORUS dataset-token marker recorded in builder metadata; invalid or missing route provenance falls back to inherited R2o routing.",
                "- No real PHOSPHORUS spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- PHOSPHORUS diagnostic rows under R2p = {len(phosphorus_rows)}; rows still dominated by `mean_shift` = {phosphorus_mean_shift}/{len(phosphorus_rows)}.",
                f"- Remaining PHOSPHORUS `derivative_over` rows = {phosphorus_derivative_over}/{len(phosphorus_rows)}; this is the current residual morphology failure mode, not a pass.",
                f"- Remaining PHOSPHORUS `mean_curve_inversion` rows = {phosphorus_inversion}/{len(phosphorus_rows)}; this remains diagnostic evidence, not a pass.",
                f"- Non-PHOSPHORUS rows accidentally reported on R2p = {changed_non_phosphorus}/{len(non_phosphorus_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2Q_REMEDIATION_PROFILES:
        lucas_ph_organic_rows = [
            row
            for row in compared
            if "lucas" in row.dataset.casefold()
            and "ph" in row.dataset.casefold()
            and "organic" in row.dataset.casefold()
        ]
        other_lucas_rows = [
            row
            for row in compared
            if "lucas" in row.dataset.casefold()
            and row not in lucas_ph_organic_rows
        ]
        phosphorus_rows = [
            row
            for row in compared
            if "phosphorus" in row.dataset.casefold()
        ]
        non_target_r2q = sum(
            1
            for row in compared
            if row not in lucas_ph_organic_rows
            and row.remediation_profile == "r2q_sentinel_matrix_v1"
        )
        target_mean_shift = sum(
            1
            for row in lucas_ph_organic_rows
            if row.dominant_morphology_gap == "mean_shift"
        )
        target_derivative_over = sum(
            1
            for row in lucas_ph_organic_rows
            if row.dominant_morphology_gap == "derivative_over"
        )
        target_variance_under = sum(
            1
            for row in lucas_ph_organic_rows
            if row.dominant_morphology_gap == "variance_under"
        )
        target_weak_or_negative_corr = sum(
            1
            for row in lucas_ph_organic_rows
            if row.mean_curve_corr is not None and row.mean_curve_corr < 0.2
        )
        lines.extend(
            [
                "## R2q LUCAS pH Organic Provenance",
                "",
                "- R2q inherits R2p routing and changes only explicit LUCAS pH Organic rows in this audit path; general LUCAS remains R2l, PHOSPHORUS remains R2p, BEER remains R2o, BERRY remains R2h, FruitPuree remains R2i, DIESEL remains R2k, MILK remains R2m, and MANURE21 remains R2n.",
                "- The LUCAS pH Organic spectra rule is `lucas_ph_organic_humic_albedo_oh_readout`: generated organic-topsoil absorbance is smoothed, kept as a bounded diffuse residual, placed on a fixed darker humic albedo continuum, and given weak broad OH bands around 1450 and 1940 nm.",
                "- This route is deliberately narrower than the R2l LUCAS mineral-organic topsoil continuum; it is selected only when LUCAS, pH, and Organic tokens are all present.",
                "- Constants are fixed organic-soil optical priors: `additive_baseline_range=[0.405, 0.455]`, `path_factor_range=[0.22, 0.32]`, `humic_slope_absorbance_range=[0.015, 0.045]`, `oh_band_absorbance_range=[0.005, 0.025]`, `smoothing_fwhm_nm=24.0`, `readout_space=lucas_ph_organic_raw_soil_apparent_absorbance`.",
                "- No real LUCAS pH Organic spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- LUCAS pH Organic diagnostic rows under R2q = {len(lucas_ph_organic_rows)}; rows still dominated by `mean_shift` = {target_mean_shift}/{len(lucas_ph_organic_rows)}.",
                f"- LUCAS pH Organic rows dominated by `variance_under` = {target_variance_under}/{len(lucas_ph_organic_rows)}; this remains a residual morphology gap, not a pass.",
                f"- LUCAS pH Organic rows with weak/negative `mean_curve_corr` (<0.2) = {target_weak_or_negative_corr}/{len(lucas_ph_organic_rows)}; correlation weakness is reported rather than optimized.",
                f"- LUCAS pH Organic rows dominated by `derivative_over` = {target_derivative_over}/{len(lucas_ph_organic_rows)}; if this rises materially in repeated seeds, R2q should be treated as `needs-review`.",
                f"- Other LUCAS rows preserved on R2l = {sum(1 for row in other_lucas_rows if row.remediation_profile == 'r2l_sentinel_matrix_v1')}/{len(other_lucas_rows)}.",
                f"- PHOSPHORUS rows preserved on R2p = {sum(1 for row in phosphorus_rows if row.remediation_profile == 'r2p_sentinel_matrix_v1')}/{len(phosphorus_rows)}.",
                f"- Non-target rows accidentally reported on R2q = {non_target_r2q}/{len(compared) - len(lucas_ph_organic_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2R_REMEDIATION_PROFILES:
        fruitpuree_rows = [
            row for row in compared if "fruitpuree" in row.dataset.casefold()
        ]
        non_target_r2r = sum(
            1
            for row in compared
            if row not in fruitpuree_rows
            and row.remediation_profile == "r2r_sentinel_matrix_v1"
        )
        target_mean_shift = sum(
            1
            for row in fruitpuree_rows
            if row.dominant_morphology_gap == "mean_shift"
        )
        target_derivative_over = sum(
            1
            for row in fruitpuree_rows
            if row.dominant_morphology_gap == "derivative_over"
        )
        target_variance_over = sum(
            1
            for row in fruitpuree_rows
            if row.dominant_morphology_gap == "variance_over"
        )
        target_amplitude_under = sum(
            1
            for row in fruitpuree_rows
            if row.dominant_morphology_gap == "amplitude_under"
        )
        lines.extend(
            [
                "## R2r FruitPuree Provenance",
                "",
                "- R2r inherits R2q routing and changes only explicit FruitPuree/puree rows in this audit path; BERRY remains R2h, beverage_juice remains on inherited juice routing, DIESEL remains R2k, LUCAS remains R2l/R2q, MILK remains R2m, MANURE21 remains R2n, BEER remains R2o, and PHOSPHORUS remains R2p.",
                "- The FruitPuree spectra rule is `strawberry_puree_transflectance_residual_readout`, distinct from `cloudy_berry_percent_transmittance_readout` and from the R2f clear-juice readout.",
                "- Constants are fixed semi-solid puree optical priors for water/sugar/cell-wall solids, seed/skin/pectin-like scatter, short transflectance path, and a raw residual absorbance readout.",
                "- No real FruitPuree spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- FruitPuree diagnostic rows under R2r = {len(fruitpuree_rows)}; rows still dominated by `mean_shift` = {target_mean_shift}/{len(fruitpuree_rows)}.",
                f"- FruitPuree rows still dominated by `amplitude_under` = {target_amplitude_under}/{len(fruitpuree_rows)}; amplitude transfer remains residual and non-gate.",
                f"- FruitPuree rows dominated by `derivative_over` = {target_derivative_over}/{len(fruitpuree_rows)}; if this appears in repeated seeds, R2r should be treated as `needs-review`.",
                f"- FruitPuree rows dominated by `variance_over` = {target_variance_over}/{len(fruitpuree_rows)}; variance transfer remains diagnostic, not a pass.",
                f"- Non-target rows accidentally reported on R2r = {non_target_r2r}/{len(compared) - len(fruitpuree_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2S_REMEDIATION_PROFILES:
        diesel_rows = [
            row
            for row in compared
            if row.synthetic_preset == "fuel" or "diesel" in row.dataset.casefold()
        ]
        non_target_r2s = sum(
            1
            for row in compared
            if row not in diesel_rows
            and row.remediation_profile == "r2s_sentinel_matrix_v1"
        )
        diesel_mean_shift = sum(
            1 for row in diesel_rows if row.dominant_morphology_gap == "mean_shift"
        )
        diesel_derivative_under = sum(
            1
            for row in diesel_rows
            if row.dominant_morphology_gap == "derivative_under"
        )
        diesel_derivative_over = sum(
            1 for row in diesel_rows if row.dominant_morphology_gap == "derivative_over"
        )
        diesel_variance_over = sum(
            1 for row in diesel_rows if row.dominant_morphology_gap == "variance_over"
        )
        lines.extend(
            [
                "## R2s DIESEL Provenance",
                "",
                "- R2s inherits R2r routing and changes only explicit DIESEL/petrochem fuel rows in this audit path; BERRY remains R2h, FruitPuree remains R2r, LUCAS remains R2l/R2q, MILK remains R2m, MANURE21 remains R2n, BEER remains R2o, and PHOSPHORUS remains R2p.",
                "- The DIESEL spectra rule remains `micro_path_fuel_ch_overtone_contrast_readout`: generated fuel absorbance is split into a broad continuum and residual CH-band structure.",
                "- Relative to R2k, R2s uses a shorter blank-referenced micro-path continuum while keeping fixed CH-overtone residual contrast, so the broadband absorbance floor is reduced without returning to R2j's wavelength-uniform derivative compression.",
                "- Constants are fixed petrochemical optical priors: `path_factor_range=[0.03, 0.045]`, `feature_contrast_range=[0.24, 0.34]`, `continuum_smoothing_fwhm_nm=96.0`, `ch_overtone_gain_range=[0.12, 0.20]`, `additive_baseline_range=[0.0005, 0.002]`, `readout_space=blank_referenced_micro_path_ch_overtone_raw_absorbance`.",
                "- The route is selected only from the explicit bench-only DIESEL dataset-token marker recorded in builder metadata; invalid or missing route provenance falls back to inherited R2r/R2k routing.",
                "- No real DIESEL spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                f"- DIESEL diagnostic rows under R2s = {len(diesel_rows)}; rows still dominated by `mean_shift` = {diesel_mean_shift}/{len(diesel_rows)}.",
                f"- DIESEL rows dominated by `derivative_under` = {diesel_derivative_under}/{len(diesel_rows)}; if this returns on all DIESEL rows, R2s must be treated as `needs-review`.",
                f"- DIESEL rows dominated by `derivative_over` = {diesel_derivative_over}/{len(diesel_rows)} and `variance_over` = {diesel_variance_over}/{len(diesel_rows)}; these are monitored failure modes, not a pass.",
                f"- Non-target rows accidentally reported on R2s = {non_target_r2s}/{len(compared) - len(diesel_rows)}.",
                "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                "",
            ]
        )
    if remediation_profile in R2T_REMEDIATION_PROFILES:
        manure_rows = [
            row
            for row in compared
            if "manure21" in row.dataset.casefold()
            or row.effective_matrix_route == "manure_organic_mineral_matrix"
        ]
        non_target_r2t = sum(
            1
            for row in compared
            if row not in manure_rows
            and row.remediation_profile == "r2t_sentinel_matrix_v1"
        )
        if manure_rows:
            manure_mean_shift = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "mean_shift"
            )
            manure_amplitude_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "amplitude_under"
            )
            manure_variance_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_over"
            )
            manure_derivative_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "derivative_over"
            )
            lines.extend(
                [
                    "## R2t MANURE21 Provenance",
                    "",
                    "- R2t inherits R2s routing and changes only explicit MANURE21 dried/ground manure rows in this audit path; DIESEL remains R2s, BERRY remains R2h, FruitPuree remains R2r, LUCAS remains R2l/R2q, MILK remains R2m, BEER remains R2o, and PHOSPHORUS remains R2p.",
                    "- The MANURE21 spectra rule is `dried_manure_heterogeneous_scatter_patch_readout`: R2n's dark organic/mineral albedo is kept while fixed particle-size scatter, moisture patch, organic lump, and mineral ash broad-band terms are added.",
                    "- No real MANURE21 spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                    f"- MANURE21 diagnostic rows under R2t = {len(manure_rows)}; rows still dominated by `mean_shift` = {manure_mean_shift}/{len(manure_rows)}.",
                    f"- MANURE21 rows still dominated by `amplitude_under` = {manure_amplitude_under}/{len(manure_rows)}; amplitude transfer remains residual and non-gate.",
                    f"- MANURE21 rows dominated by `variance_over` = {manure_variance_over}/{len(manure_rows)} and `derivative_over` = {manure_derivative_over}/{len(manure_rows)}; systematic over-correction should be treated as `needs-review`.",
                    f"- Non-target rows accidentally reported on R2t = {non_target_r2t}/{len(compared) - len(manure_rows)}.",
                    "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                    "",
                ]
            )
    if remediation_profile in R2U_REMEDIATION_PROFILES:
        manure_rows = [
            row
            for row in compared
            if "manure21" in row.dataset.casefold()
            or row.effective_matrix_route == "manure_organic_mineral_matrix"
        ]
        non_target_r2u = sum(
            1
            for row in compared
            if row not in manure_rows
            and row.remediation_profile == "r2u_sentinel_matrix_v1"
        )
        if manure_rows:
            manure_mean_shift = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "mean_shift"
            )
            manure_amplitude_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "amplitude_under"
            )
            manure_variance_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_under"
            )
            manure_variance_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_over"
            )
            manure_derivative_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "derivative_over"
            )
            lines.extend(
                [
                    "## R2u MANURE21 Provenance",
                    "",
                    "- R2u inherits R2s routing and changes only explicit MANURE21 dried/ground manure rows in this audit path; R2t remains available but is not the baseline profile for this report.",
                    "- The MANURE21 spectra rule is `dried_manure_bounded_centered_scatter_readout`: R2n's dark organic/mineral albedo range is kept while fixed centered residual, particle-size scatter, moisture patch, organic lump, and mineral ash broad-band terms increase amplitude without a global continuum lift.",
                    "- No real MANURE21 spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                    f"- MANURE21 diagnostic rows under R2u = {len(manure_rows)}; rows still dominated by `mean_shift` = {manure_mean_shift}/{len(manure_rows)}.",
                    f"- MANURE21 rows still dominated by `amplitude_under` = {manure_amplitude_under}/{len(manure_rows)}; amplitude transfer remains residual and non-gate.",
                    f"- MANURE21 rows still dominated by `variance_under` = {manure_variance_under}/{len(manure_rows)}; variance transfer remains residual and non-gate.",
                    f"- MANURE21 rows dominated by `variance_over` = {manure_variance_over}/{len(manure_rows)} and `derivative_over` = {manure_derivative_over}/{len(manure_rows)}; systematic over-correction should be treated as `needs-review`.",
                    f"- Non-target rows accidentally reported on R2u = {non_target_r2u}/{len(compared) - len(manure_rows)}.",
                    "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                    "",
                ]
            )
    if remediation_profile in R2V_REMEDIATION_PROFILES:
        manure_rows = [
            row
            for row in compared
            if "manure21" in row.dataset.casefold()
            or row.effective_matrix_route == "manure_organic_mineral_matrix"
        ]
        non_target_r2v = sum(
            1
            for row in compared
            if row not in manure_rows
            and row.remediation_profile == "r2v_sentinel_matrix_v1"
        )
        if manure_rows:
            manure_mean_shift = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "mean_shift"
            )
            manure_amplitude_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "amplitude_under"
            )
            manure_variance_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_under"
            )
            manure_variance_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_over"
            )
            manure_derivative_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "derivative_over"
            )
            lines.extend(
                [
                    "## R2v MANURE21 Provenance",
                    "",
                    "- R2v inherits R2s routing and changes only explicit MANURE21 dried/ground manure rows in this audit path; R2t/R2u remain available but are not the baseline profile for this report.",
                    "- The MANURE21 spectra rule is `dried_manure_balanced_centered_scatter_readout`: R2u's fixed dark organic/mineral continuum expectation is kept while bounded balanced draws remove seed-dependent continuum drift.",
                    "- Variability is carried by centered residual, particle-size scatter, moisture patch, organic lump, and mineral ash broad-band terms; these terms are wavelength-centered and do not add a row-uniform lift/downshift.",
                    "- No real MANURE21 spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                    f"- MANURE21 diagnostic rows under R2v = {len(manure_rows)}; rows still dominated by `mean_shift` = {manure_mean_shift}/{len(manure_rows)}.",
                    f"- MANURE21 rows still dominated by `amplitude_under` = {manure_amplitude_under}/{len(manure_rows)} and `variance_under` = {manure_variance_under}/{len(manure_rows)}; residual under-transfer remains non-gate.",
                    f"- MANURE21 rows dominated by `variance_over` = {manure_variance_over}/{len(manure_rows)} and `derivative_over` = {manure_derivative_over}/{len(manure_rows)}; systematic over-correction should be treated as `needs-review`.",
                    f"- Non-target rows accidentally reported on R2v = {non_target_r2v}/{len(compared) - len(manure_rows)}.",
                    "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                    "",
                ]
            )
    if remediation_profile in R2W_REMEDIATION_PROFILES:
        manure_rows = [
            row
            for row in compared
            if "manure21" in row.dataset.casefold()
            or row.effective_matrix_route == "manure_organic_mineral_matrix"
        ]
        non_target_r2w = sum(
            1
            for row in compared
            if row not in manure_rows
            and row.remediation_profile == "r2w_sentinel_matrix_v1"
        )
        if manure_rows:
            manure_mean_shift = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "mean_shift"
            )
            manure_amplitude_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "amplitude_under"
            )
            manure_variance_under = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_under"
            )
            manure_variance_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "variance_over"
            )
            manure_derivative_over = sum(
                1
                for row in manure_rows
                if row.dominant_morphology_gap == "derivative_over"
            )
            lines.extend(
                [
                    "## R2w MANURE21 Provenance",
                    "",
                    "- R2w inherits R2s routing and changes only explicit MANURE21 dried/ground manure rows in this audit path; R2t/R2u/R2v remain available but are not the baseline profile for this report.",
                    "- The MANURE21 spectra rule is `dried_manure_albedo_variance_centered_scatter_readout`: R2v's balanced centered residual/scatter readout is kept while a wider fixed dark organic/mineral albedo prior transfers more dried-manure cup heterogeneity.",
                    "- Variability is carried by balanced albedo draws plus centered residual, particle-size scatter, moisture patch, organic lump, and mineral ash broad-band terms; no row-specific real statistic is captured.",
                    "- No real MANURE21 spectra, marginal statistics, covariance/PCA structure, quantiles, labels, targets, splits, adversarial AUC, morphology gap score, thresholds, or downstream result was read to set these constants.",
                    f"- MANURE21 diagnostic rows under R2w = {len(manure_rows)}; rows still dominated by `mean_shift` = {manure_mean_shift}/{len(manure_rows)}.",
                    f"- MANURE21 rows still dominated by `amplitude_under` = {manure_amplitude_under}/{len(manure_rows)} and `variance_under` = {manure_variance_under}/{len(manure_rows)}; residual under-transfer remains non-gate.",
                    f"- MANURE21 rows dominated by `variance_over` = {manure_variance_over}/{len(manure_rows)} and `derivative_over` = {manure_derivative_over}/{len(manure_rows)}; systematic over-correction should be treated as `needs-review`.",
                    f"- Non-target rows accidentally reported on R2w = {non_target_r2w}/{len(compared) - len(manure_rows)}.",
                    "- This is single-seed diagnostic evidence only; it is not a B2/B3/B4/B5 gate and does not claim scientific validation.",
                    "",
                ]
            )
    lines.extend(
        [
            "## Summary",
            "",
            f"- Status: `{result['status']}`",
            f"- Real runnable rows discovered: {result['real_runnable_count']}",
            f"- Real sentinel candidates after token filter: {result.get('real_sentinel_candidate_count', 0)}",
            f"- Real rows selected: {result['real_selected_count']}",
            f"- Compared rows: {len(compared)}",
            f"- Blocked rows: {len(blocked)}",
            "",
            "## Morphology Metrics (uncalibrated_raw)",
            "",
            "| dataset | preset selector | effective matrix route | n real | n syn | log10 std ratio | log10 amp p50 ratio | log10 deriv std p50 ratio | mean curve corr | gap score | dominant gap | remediation | readout/rule | composition source | spectra source | provenance source | route variant | constant status | readout space | calibration source | real stat source | threshold source | rem conc | rem spec | status |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|:-:|:-:|---|",
        ]
    )
    for row in rows:
        remediation_label = row.remediation_profile if row.remediation_profile is not None else "none"
        effective_matrix_route_label = row.effective_matrix_route or "none"
        spectra_rule_label = (
            row.r2c_remediation_spectra_rule
            if row.r2c_remediation_spectra_rule is not None
            else "none"
        )
        composition_source_label = row.r2c_remediation_composition_source or "none"
        spectra_source_label = row.r2c_remediation_spectra_source or "none"
        provenance_source_label = row.r2c_remediation_provenance_source or "none"
        route_variant_label = row.r2c_remediation_route_variant or "none"
        constant_status_label = row.r2c_remediation_constant_status or "none"
        readout_space_label = row.r2c_remediation_readout_space or "none"
        calibration_source_label = row.r2c_remediation_calibration_source or "none"
        real_stat_source_label = row.r2c_remediation_real_stat_source or "none"
        threshold_source_label = row.r2c_remediation_threshold_source or "none"
        lines.append(
            f"| `{row.dataset}` | `{row.synthetic_preset}` | "
            f"`{effective_matrix_route_label}` | "
            f"{row.n_real_samples} | {row.n_synthetic_samples} | "
            f"{_fmt(row.log10_global_std_ratio)} | {_fmt(row.log10_amplitude_p50_ratio)} | "
            f"{_fmt(row.log10_derivative_std_p50_ratio)} | {_fmt(row.mean_curve_corr)} | "
            f"{_fmt(row.morphology_gap_score)} | `{row.dominant_morphology_gap}` | "
            f"`{remediation_label}` | "
            f"`{spectra_rule_label}` | "
            f"`{composition_source_label}` | "
            f"`{spectra_source_label}` | "
            f"`{provenance_source_label}` | "
            f"`{route_variant_label}` | "
            f"`{constant_status_label}` | "
            f"`{readout_space_label}` | "
            f"`{calibration_source_label}` | "
            f"`{real_stat_source_label}` | "
            f"`{threshold_source_label}` | "
            f"{'Y' if row.r2c_remediation_concentrations_applied else '-'} | "
            f"{'Y' if row.r2c_remediation_spectra_applied else '-'} | "
            f"`{row.status}` |"
        )

    lines.extend(
        [
            "",
            "## Audit Flags (every row)",
            "",
            "- `oracle=false`",
            "- `label_inputs_used=false`",
            "- `target_inputs_used=false`",
            "- `split_inputs_used=false`",
            "- `source_oracle_used=false`",
            "- `learned=false`",
            "- `real_stat_capture=false`",
            "- `thresholds_modified=false`",
            "- `metrics_modified=false`",
            "- `imputed=false`",
            "- `replays_real_rows=false`",
            f"- `audit_scope={audit_scope}`",
            "",
            "## Decision",
            "",
            (
                "Report-only morphology snapshot; this lane is non-gate and does not establish any "
                "B2/B3/B4/B5 pass. Dominant-gap labels and gap scores are diagnostic input for the next "
                "mechanistic correction only."
            ),
            "",
            "## Raw Summary JSON",
            "",
            "```json",
            json.dumps(
                {
                    "status": result["status"],
                    "comparison_space": COMPARISON_SPACE,
                    "real_runnable_count": result["real_runnable_count"],
                    "real_sentinel_candidate_count": result.get(
                        "real_sentinel_candidate_count", 0
                    ),
                    "real_selected_count": result["real_selected_count"],
                    "sentinel_tokens": tokens,
                    "remediation_profile": remediation_profile,
                    "r2c_remediation_concentrations_applied_count": sum(
                        1 for row in rows if row.r2c_remediation_concentrations_applied
                    ),
                    "r2c_remediation_spectra_applied_count": sum(
                        1 for row in rows if row.r2c_remediation_spectra_applied
                    ),
                    "compared_row_count": len(compared),
                    "blocked_row_count": len(blocked),
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _parse_sentinel_tokens(raw: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("at least one sentinel token must be provided")
    return tokens


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "bench" / "nirs_synthetic_pfn").is_dir():
            return candidate
    raise RuntimeError(f"could not locate repo root from {here}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-synthetic-samples", type=int, default=64)
    parser.add_argument("--max-real-samples", type=int, default=64)
    parser.add_argument(
        "--max-sentinel-datasets",
        type=int,
        default=DEFAULT_MAX_SENTINEL_DATASETS,
        help=(
            "Cap on sentinel datasets after priority-based selection. "
            "0 (or any non-positive value) means every runnable local cohort row, "
            "with no token filter or priority sort applied."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260430)
    parser.add_argument(
        "--sentinel-tokens",
        type=str,
        default=",".join(DEFAULT_SENTINEL_TOKENS),
        help=(
            "Comma-separated case-insensitive tokens used to select sentinel datasets "
            "by matching against `source`, `task`, `database_name`, or `dataset`. "
            "Only applied when `--max-sentinel-datasets > 0`."
        ),
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--remediation-profile",
        type=str,
        default=None,
        choices=[None, *ALL_REMEDIATION_PROFILES],
        help=(
            "Optional bench-only remediation profile. R2c profile applies "
            "DIESEL/MILK rules; R2d additionally applies BEER/CORN rules; "
            "R2f additionally applies beverage_juice rules for BERRY/juice rows; "
            "R2g additionally applies environmental_soil rules for LUCAS/"
            "PHOSPHORUS/SOIL rows; R2h additionally applies a BERRY/juice "
            "apparent percent-transmittance readout while preserving R2g "
            "routing for non-BERRY rows; R2i keeps R2h/R2g/R2f routing and "
            "adds a separate FruitPuree semi-solid puree rule; R2j keeps R2i "
            "routing and changes only DIESEL/petrochem fuels to a micro-path "
            "raw absorbance readout; R2k keeps R2i routing and changes only "
            "DIESEL/petrochem fuels to a micro-path CH overtone contrast "
            "readout that preserves derivative structure better than R2j. "
            "R2l keeps R2k routing and changes only LUCAS-marked soil rows. "
            "R2m keeps R2l routing and changes only MILK/food_dairy rows to "
            "a fat-globule scatter inverse-transflectance raw intensity "
            "readout. R2n keeps R2m routing and changes only MANURE21-marked "
            "dried/ground manure rows to an organic-mineral albedo diffuse "
            "reflectance readout. R2o keeps R2n routing and changes only "
            "BEER-marked rows to a fermented-liquid turbid cuvette apparent "
            "absorbance readout. R2p keeps R2o routing and changes only "
            "PHOSPHORUS-marked rows to a mineral/phosphate albedo residual "
            "readout. R2q keeps R2p routing and changes only LUCAS pH "
            "Organic-marked rows to a humic organic-topsoil albedo/OH "
            "readout. R2r keeps R2q routing and changes only FruitPuree/"
            "puree-marked rows to a strawberry puree transflectance residual "
            "readout. R2s keeps R2r routing and changes only DIESEL/petrochem "
            "fuel rows to a shorter blank-referenced micro-path CH overtone "
            "readout. R2t keeps R2s routing and changes only MANURE21 dried/"
            "ground manure rows to a heterogeneous scatter/patch readout. "
            "R2u/R2v/R2w keep R2s routing and change only MANURE21 rows with "
            "centered manure albedo/scatter readouts. "
            "Each profile re-biases composition with a tight Dirichlet and "
            "applies a mechanistic spectra transform (optical-path scale and, "
            "for CORN/SOIL, instrumental Gaussian smoothing). Audit recorded per "
            "row. Default: no remediation (path unchanged)."
        ),
    )
    args = parser.parse_args()

    sentinel_tokens = _parse_sentinel_tokens(args.sentinel_tokens)
    root = _repo_root()
    result = run_audit(
        root=root,
        n_synthetic_samples=args.n_synthetic_samples,
        max_real_samples=args.max_real_samples,
        max_sentinel_datasets=args.max_sentinel_datasets,
        seed=args.seed,
        sentinel_tokens=sentinel_tokens,
        remediation_profile=args.remediation_profile,
    )
    write_csv(result["rows"], args.csv)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        render_markdown(
            result=result,
            report_path=args.report,
            csv_path=args.csv,
            n_synthetic_samples=args.n_synthetic_samples,
            max_real_samples=args.max_real_samples,
            max_sentinel_datasets=args.max_sentinel_datasets,
            seed=args.seed,
            sentinel_tokens=sentinel_tokens,
            remediation_profile=args.remediation_profile,
        ),
        encoding="utf-8",
    )
    print(args.report)
    print(args.csv)


__all__ = [
    "ALL_REMEDIATION_PROFILES",
    "COMPARISON_SPACE",
    "DEFAULT_CSV",
    "DEFAULT_MAX_SENTINEL_DATASETS",
    "DEFAULT_REPORT",
    "DEFAULT_SENTINEL_TOKENS",
    "MorphologyRow",
    "PRIMARY_SENTINEL_TOKENS",
    "R2B_AUDIT_SCOPE",
    "R2C_REMEDIATION_PROFILES",
    "R2D_REMEDIATION_PROFILES",
    "R2F_REMEDIATION_PROFILES",
    "R2G_REMEDIATION_PROFILES",
    "R2H_REMEDIATION_PROFILES",
    "R2I_REMEDIATION_PROFILES",
    "R2J_REMEDIATION_PROFILES",
    "R2K_REMEDIATION_PROFILES",
    "R2L_REMEDIATION_PROFILES",
    "R2M_REMEDIATION_PROFILES",
    "R2N_REMEDIATION_PROFILES",
    "R2O_REMEDIATION_PROFILES",
    "R2P_REMEDIATION_PROFILES",
    "R2Q_REMEDIATION_PROFILES",
    "R2R_REMEDIATION_PROFILES",
    "R2S_REMEDIATION_PROFILES",
    "R2T_REMEDIATION_PROFILES",
    "R2U_REMEDIATION_PROFILES",
    "R2V_REMEDIATION_PROFILES",
    "R2W_REMEDIATION_PROFILES",
    "SECONDARY_FRUIT_SENTINEL_TOKENS",
    "SECONDARY_MILK_SENTINEL_TOKENS",
    "SECONDARY_SOIL_SENTINEL_TOKENS",
    "SENTINEL_PRIORITY_GROUPS",
    "compute_morphology_metrics",
    "main",
    "render_markdown",
    "run_audit",
    "select_synthetic_preset_for_dataset",
    "write_csv",
]


if __name__ == "__main__":
    main()
