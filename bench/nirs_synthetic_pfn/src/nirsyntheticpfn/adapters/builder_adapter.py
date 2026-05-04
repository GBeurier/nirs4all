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

# R2x sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicit MANURE21 rows. R2x keeps R2w's mean-neutral
# balanced albedo/scatter readout, adds a slightly wider centered albedo
# dispersion, and uses coarser particulate smoothing so variance can rise
# without making derivative-over the dominant repeated-seed failure mode.
R2X_REMEDIATION_PROFILES: tuple[str, ...] = ("r2x_sentinel_matrix_v1",)

# R2y sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicit MANURE21 rows. R2y restarts from R2w's
# centered albedo/scatter readout and adds only softer low-frequency,
# mean-neutral dispersion to reduce residual under-transfer without using
# statistical capture.
R2Y_REMEDIATION_PROFILES: tuple[str, ...] = ("r2y_sentinel_matrix_v1",)

# R2z sentinel matrix remediation profiles. Bench-only opt-in extension of
# R2s that changes only explicit MANURE21 rows. R2z starts from R2w's
# conservative centered albedo/scatter route and adds mean-neutral
# compositional heterogeneity plus smooth low-frequency readout variability.
R2Z_REMEDIATION_PROFILES: tuple[str, ...] = ("r2z_sentinel_matrix_v1",)

# R3a CORN matrix remediation profile. Bench-only opt-in extension after the
# R2* sequence. R3a changes only explicitly CORN-marked grain/powder records;
# every non-CORN record inherits the conservative R2w path so BEER, DIESEL,
# MANURE21, MILK, soil, and fruit sentinels keep their established routes.
R3A_REMEDIATION_PROFILES: tuple[str, ...] = ("r3a_corn_matrix_v1",)

# R3b CORN matrix remediation profile. Bench-only opt-in extension of R3a that
# keeps the centered corn-powder albedo readout but increases mechanistic
# particle path dispersion under coarser smoothing to recover band amplitude
# without returning to R2f/R2w derivative-dominant powder spectra.
R3B_REMEDIATION_PROFILES: tuple[str, ...] = ("r3b_corn_matrix_v1",)

# R3c DIESEL matrix remediation profile. Bench-only opt-in extension of R3b
# that changes only explicitly DIESEL-marked petrochem fuel records. Every
# non-DIESEL record inherits the R3b context unchanged.
R3C_REMEDIATION_PROFILES: tuple[str, ...] = ("r3c_diesel_matrix_v1",)

# R3d DIESEL matrix remediation profile. Bench-only opt-in extension of R3c
# that changes only explicitly DIESEL-marked petrochem fuel records. Every
# non-DIESEL record inherits the R3c context unchanged.
R3D_REMEDIATION_PROFILES: tuple[str, ...] = ("r3d_diesel_matrix_v1",)

# R3e DIESEL matrix remediation profile. Bench-only diagnostic-only opt-in
# extension of R3d that changes only explicitly DIESEL-marked petrochem fuel
# records. Every non-DIESEL record inherits the R3d context unchanged.
R3E_REMEDIATION_PROFILES: tuple[str, ...] = ("r3e_diesel_matrix_v1",)

# R3f DIESEL matrix remediation profile. Bench-only diagnostic-only opt-in
# extension of R3e that changes only explicitly DIESEL-marked petrochem fuel
# records. Every non-DIESEL record inherits the R3e context unchanged.
R3F_REMEDIATION_PROFILES: tuple[str, ...] = ("r3f_diesel_matrix_v1",)

# R3g DIESEL matrix remediation profile. Bench-only diagnostic-only opt-in
# extension of R3f that changes only explicitly DIESEL-marked petrochem fuel
# records. It keeps R3f's lowered micro-path level and adds only a fixed
# mean-neutral hydrocarbon CH-band envelope from general NIR spectroscopy.
R3G_REMEDIATION_PROFILES: tuple[str, ...] = ("r3g_diesel_matrix_v1",)

# R4a DIESEL basis remediation profile. Bench-only diagnostic-only opt-in
# extension that inherits R3d (NOT R3e/R3f/R3g) for every non-DIESEL row and
# changes only explicitly DIESEL-marked petrochem fuel records. R4a keeps
# R3d's micro-path continuum and detector offset, drops the 1720 nm CH center
# (out-of-support for the 750-1550 nm DIESEL real basis observed in the audit),
# widens the remaining CH overtones, lowers the CH gain, damps the residual
# inside the over-structured 1100-1500 nm region (without inversion), and adds
# a short-continuum hydrocarbon scatter hump centered at 975 nm restricted to
# the 750-1550 nm support. Constants are mechanistic priors only; no real
# spectra/statistics, labels, targets, splits, AUC, or thresholds are consulted.
R4A_REMEDIATION_PROFILES: tuple[str, ...] = ("r4a_diesel_basis_v1",)

# R4b DIESEL derivative-restore remediation profile. Bench-only diagnostic-only
# opt-in extension that inherits R3d (NOT R4a) for every non-DIESEL row and
# changes only explicitly DIESEL-marked petrochem fuel records. R4b keeps R3d's
# micro-path continuum and detector offset, drops the 1720 nm CH center, but
# uses CH overtones that are less smoothed than R4a (narrower width, slightly
# higher gain) and weaker, narrower damping windows so the first-derivative
# structure is restored without falling back to R3d's over-structured residual.
# A short-continuum hydrocarbon hump is kept on the 750-1550 nm support but is
# narrower and lower-amplitude than R4a so it lifts the level without removing
# derivative signal. Constants are mechanistic priors only; no real
# spectra/statistics, labels, targets, splits, AUC, or thresholds are consulted.
R4B_REMEDIATION_PROFILES: tuple[str, ...] = ("r4b_diesel_derivative_restore_v1",)

# R4c DIESEL balanced-derivative remediation profile. Bench-only diagnostic-only
# opt-in extension that inherits R3d (NOT R4a/R4b) for every non-DIESEL row and
# changes only explicitly DIESEL-marked petrochem fuel records. R4c keeps R3d's
# micro-path continuum and detector offset and the same support-only CH overtone
# centers as R4a/R4b (drops 1720 nm), but tightens the first-derivative balance
# beyond R4b: a narrower CH overtone width (36 nm vs R4b 38 nm) and slightly
# higher gain (0.092-0.155 vs R4b 0.085-0.145) sharpen the CH bands, while
# weaker / narrower residual damping windows ((1180, 46, 0.60),
# (1425, 54, 0.70) at strength 0.05-0.15 vs R4b (1180, 52, 0.75),
# (1425, 62, 0.85) at 0.10-0.22) preserve more first-derivative structure inside
# the 1100-1500 nm region. A narrower lower-amplitude short-continuum hydrocarbon
# hump (72 nm width, 0.00010-0.00032 amplitude) is kept centered at 975 nm on the
# 750-1550 nm support so the 900-1050 nm level is still lifted without flattening
# the derivative. Constants are mechanistic priors only; no real
# spectra/statistics, labels, targets, splits, AUC, or thresholds are consulted.
R4C_REMEDIATION_PROFILES: tuple[str, ...] = ("r4c_diesel_balanced_derivative_v1",)

# R5 DIESEL readout-space remediation profiles. Bench-only diagnostic-only
# opt-in extensions that inherit R3d for every non-DIESEL row (NOT R4a/R4b/R4c)
# and inherit the full R4c CH overtone / damping / hump / path / feature
# contrast configuration on explicitly DIESEL-marked petrochem fuel records.
# The ONLY difference between the three R5 profiles and R4c is the final
# spectral readout space:
#   * R5a keeps absorbance (A -> A) and is byte-identical to R4c on the same
#     seed and explicit DIESEL route;
#   * R5b maps absorbance to transmittance (A -> 10**-A) clipped to [0, 1];
#   * R5c maps absorbance to blank-referenced intensity (A -> 1 - 10**-A)
#     clipped to [0, 1].
# Constants are mechanistic priors only; no real spectra/statistics, labels,
# targets, splits, AUC, thresholds, calibration, or downstream feedback are
# consulted. R5 is diagnostic-only; not promoted over R3d, not a B2/B3/B4/B5
# gate, and does not authorize any nirs4all integration.
R5A_REMEDIATION_PROFILES: tuple[str, ...] = ("r5a_diesel_absorbance_readout_v1",)
R5B_REMEDIATION_PROFILES: tuple[str, ...] = ("r5b_diesel_transmittance_readout_v1",)
R5C_REMEDIATION_PROFILES: tuple[str, ...] = ("r5c_diesel_blank_referenced_intensity_v1",)

# R6a DIESEL centered-hydrocarbon-shape remediation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# (NOT R4a/R4b/R4c/R5a/R5b/R5c) and changes only explicitly DIESEL-marked
# petrochem fuel records that carry the explicit R6a shape route. R6a reuses
# the full R4c absorbance pipeline byte-for-byte (path, baseline, CH overtones,
# damping, hump) and only adds a small fixed mean-neutral hydrocarbon CH-band
# shape envelope on the 750-1550 nm support: outside the support the envelope
# is zero, inside the support it is normalized to peak 1 and recentered to
# zero mean over the support so it adds shape without shifting the support
# mean. Constants are mechanistic priors only; never derived from real spectra,
# labels, targets, splits, calibration, real statistics, AUC, thresholds, or
# downstream feedback. R6a is diagnostic-only; not promoted over R3d, not a
# B2/B3/B4/B5 gate, and does not authorize any nirs4all integration.
R6A_REMEDIATION_PROFILES: tuple[str, ...] = ("r6a_diesel_centered_hydrocarbon_shape_v1",)

# R7a DIESEL support-centered residual transfer remediation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a) and changes only explicitly DIESEL-marked
# petrochem fuel records that carry the explicit R7a residual route. R7a reuses
# the R4a-like absorbance base (R3d micro-path continuum and detector offset,
# support-only CH overtone centers 1150/1210/1390/1460 nm, width 46 nm, gain
# 0.055-0.105, R4a damping windows, 975 nm short-continuum hump) and adds a
# bounded support-centered residual transfer step: the synthetic hydrocarbon
# residual ``X_in - continuum`` is masked to the 750-1550 nm DIESEL real basis
# support, row-centered on the support so its support-mean is zero by
# construction, scaled by a fixed bounded draw in [0.08, 0.18], and added to
# the R4a base before the final non-negative absorbance clip. The final clip
# rule, clip fraction, and pre/post-clip min/max are recorded in the audit
# trail. Constants are mechanistic priors only; no real spectra, statistics,
# PCA, covariance, quantiles, ML/DL output, labels, targets, splits, AUC,
# calibration, or threshold tuning is consulted. R7a is bench-only
# diagnostic-only; not promoted over R3d, not a B2/B3/B4/B5 gate, and does not
# authorize any nirs4all integration.
R7A_REMEDIATION_PROFILES: tuple[str, ...] = ("r7a_diesel_support_centered_residual_transfer_v1",)

# R8a DIESEL mean-preserving micro-path modulation remediation profile.
# Bench-only diagnostic-only opt-in extension that inherits R3d for every
# non-DIESEL row (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a) and changes only
# explicitly DIESEL-marked petrochem fuel records that carry the explicit
# R8a micro-path modulation route. R8a reuses the R4a-like absorbance base
# (R3d micro-path continuum and detector offset, support-only CH overtone
# centers 1150/1210/1390/1460 nm at width 46 nm and gain 0.055-0.105, R4a
# damping windows and strength range, 975 nm short-continuum hump on the
# 750-1550 nm support). After the R4a base reaches non-negative absorbance
# through the standard R4a final clip, R8a applies a bounded multiplicative
# modulation on the 750-1550 nm support derived from the synthetic
# hydrocarbon residual ``X_in - continuum``: the residual is masked to the
# support, row-centered on the support, normalized by a robust synthetic
# scale (p95 of |residual| with a numerical epsilon), bounded to a
# dimensionless shape in [-1, 1], and the support of the base is multiplied
# by ``exp(strength * shape)`` with a fixed bounded strength draw; the
# support row mean is then exactly preserved by a multiplicative
# renormalization. Outside the support the readout is identically unchanged
# from the R4a base. A final non-negative guard clip is recorded for audit
# but is expected to be a no-op because the base is non-negative and the
# multiplicative modulation is positive. Constants are mechanistic priors
# only; no real spectra, statistics, PCA, covariance, quantiles, ML/DL
# output, labels, targets, splits, AUC, calibration, or threshold tuning is
# consulted. R8a is bench-only diagnostic-only; not promoted over R3d, not
# a B2/B3/B4/B5 gate, and does not authorize any nirs4all integration.
R8A_REMEDIATION_PROFILES: tuple[str, ...] = ("r8a_diesel_mean_preserving_micro_path_modulation_v1",)

# R8b DIESEL R4c-base mean-preserving micro-path modulation profile.
# Bench-only diagnostic-only opt-in extension that inherits R3d for every
# non-DIESEL row (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a) and changes only
# explicitly DIESEL-marked petrochem fuel records that carry the explicit
# R8b micro-path modulation route. R8b reuses the R4c balanced-derivative
# absorbance base byte-for-byte, then applies the same bounded support
# mean-preserving multiplicative modulation mechanism as R8a. Constants are
# mechanistic priors only; no real spectra, statistics, PCA, covariance,
# quantiles, ML/DL output, labels, targets, splits, AUC, calibration, or
# threshold tuning is consulted. R8b is bench-only diagnostic-only; not
# promoted over R3d, not a B2/B3/B4/B5 gate, and does not authorize any
# nirs4all integration.
R8B_REMEDIATION_PROFILES: tuple[str, ...] = ("r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1",)

# R9b DIESEL support-level mechanistic intercept profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a/R8b) and changes only explicitly
# DIESEL-marked petrochem fuel records that carry the explicit R9b support
# intercept route. R9b reuses the R4c balanced-derivative absorbance base
# byte-for-byte (path, baseline, CH overtones, damping, hump, R4c
# non-negative output clip) and then adds a single small fixed mechanistic
# absorbance intercept on the 750-1550 nm DIESEL real basis support; outside
# the support the readout is identically equal to the R4c base. The intercept
# value is a pre-declared constant set from generic optical
# reference/blank-cell/detector support-level prior knowledge, not derived
# from R9a/R9b mean-shift deltas, real spectra, marginal statistics, PCA,
# covariance, quantiles, ML/DL output, labels, targets, splits, AUC,
# morphology gap scores, calibration, or threshold tuning. The constant is
# small enough that it neither triggers the R4c non-negative output clip
# (R4c output is non-negative and the intercept is positive) nor smooths
# derivatives the way the R4a basis profile does (a constant added on the
# support has zero first-derivative inside the support and a single one-bin
# step at the support boundary that is unrelated to the support-interior
# derivative structure that the morphology audit measures). R9b is bench-only
# diagnostic-only; not promoted over R3d, not a B2/B3/B4/B5 gate, and does
# not authorize any nirs4all integration.
R9B_REMEDIATION_PROFILES: tuple[str, ...] = ("r9b_diesel_support_intercept_v1",)

# R9c DIESEL support-level shape mechanism profile. Bench-only diagnostic-only
# opt-in extension that inherits R3d for every non-DIESEL row (NOT
# R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a/R8b/R9b) and changes only explicitly
# DIESEL-marked petrochem fuel records that carry the explicit R9c support
# shape route. R9c reuses the R3d micro-path / baseline / CH-overtone /
# additive-baseline / non-negative output clip pipeline byte-for-byte (so the
# pre-shape base output equals the R3d output on the same seed and DIESEL
# route) and only adds, AFTER the R3d output clip and ON the 750-1550 nm
# DIESEL real basis support window only, a fixed mechanistic shape modulation:
#
#   * a sum of Gaussian CH overtone bands at 1150/1210/1390/1460 nm with
#     per-band widths (40, 40, 44, 48) nm and a small bounded gain draw in
#     [0.075, 0.135] (selective hydrocarbon broadening);
#   * two support-localized damping windows centered at 1180 nm (width 56,
#     weight 0.55) and 1425 nm (width 72, weight 0.85) modulated by a fixed
#     bounded damping strength draw in [0.14, 0.28];
#   * a small fixed support hump at 975 nm (width 84) with a bounded amplitude
#     draw in [0.00018, 0.00048].
#
# Outside the support the readout is byte-identically equal to the R3d base.
# Constants are PRE-DECLARED MECHANISTIC CONSTANTS pulled from a general
# liquid-hydrocarbon NIR prior (Goddu/Workman/Burns NIR overtone tables); they
# are NOT chosen from any R9a or R9b mean-shift residual delta, NOT fitted to
# real spectra, marginal statistics, PCA loadings, quantiles, ML/DL output,
# labels, targets, splits, AUC, morphology gap scores, thresholds, or
# downstream feedback. R9c is bench-only diagnostic-only; it is NOT promoted
# over R3d, NOT a B2/B3/B4/B5 gate, and does NOT authorize any nirs4all
# integration.
R9C_REMEDIATION_PROFILES: tuple[str, ...] = ("r9c_diesel_selective_ch_bandwidth_damping_v1",)

# R9d DIESEL energy-normalized mean-neutral support redistribution profile.
# Bench-only diagnostic-only opt-in extension that inherits R3d for every
# non-DIESEL row (NOT R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a/R8b/R9b/R9c) and
# changes only explicitly DIESEL-marked petrochem fuel records that carry the
# explicit R9d support redistribution route. R9d reuses the R3d micro-path /
# baseline / CH-overtone / additive-baseline / non-negative output clip
# pipeline byte-for-byte (so the pre-redistribution base output equals the
# R3d output on the same seed and DIESEL route) and only applies, AFTER the
# R3d output clip and ON the 750-1550 nm DIESEL real basis support window
# only, a multiplicative ``exp(strength * shape)`` modulation where ``shape``
# is a fixed mean-neutral max-abs-normalized basis built from a sum of
# Gaussian CH overtone bands at 1150/1210/1390/1460 nm with per-band widths
# (40, 40, 44, 48) nm, and the per-row ``strength`` is bounded in
# [0.035, 0.095]. After the multiplication the support of each row is
# multiplicatively renormalized so that the post-redistribution support mean
# equals the pre-redistribution support mean exactly within numerical
# tolerance. Outside the support the readout is byte-identically equal to
# the R3d base. R9d adds NO scalar offset and adds NO positive-area lift on
# the support; the support delta is mean-neutral by construction. Constants
# are PRE-DECLARED MECHANISTIC CONSTANTS pulled from a general
# liquid-hydrocarbon NIR energy redistribution prior; they are NOT chosen
# from any R9a/R9b/R9c mean-shift residual delta, NOT fitted to real
# spectra, marginal statistics, PCA loadings, quantiles, ML/DL output,
# labels, targets, splits, AUC, morphology gap scores, thresholds, or
# downstream feedback. R9d is bench-only diagnostic-only; it is NOT promoted
# over R3d, NOT a B2/B3/B4/B5 gate, and does NOT authorize any nirs4all
# integration.
R9D_REMEDIATION_PROFILES: tuple[str, ...] = ("r9d_diesel_energy_normalized_support_redistribution_v1",)

# R9e DIESEL pathlength/reference attenuation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# and changes only explicitly DIESEL-marked petrochem fuel records carrying
# the explicit R9e reference attenuation route. R9e reuses the R3d
# micro-path / baseline / CH-overtone / additive-baseline / non-negative output
# clip pipeline byte-for-byte, then applies a positive row-wise multiplicative
# attenuation factor in [0.970, 0.985] only on the fixed 750-1550 nm support.
# Off-support cells remain byte-identical to R3d. No additional clip, offset,
# R9d shape, support-mean renormalization, or readout transform is applied.
R9E_REMEDIATION_PROFILES: tuple[str, ...] = ("r9e_diesel_pathlength_reference_attenuation_v1",)

# R9f DIESEL pre-offset pathlength/reference attenuation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# and changes only explicitly DIESEL-marked petrochem fuel records carrying
# the explicit R9f pre-offset reference attenuation route. R9f applies the
# same positive row-wise support-only factor range as R9e, but only to the
# Beer-Lambert continuum/path component before the additive detector offset
# and before the existing R3d output clip. It does not attenuate feature
# residuals, additive baseline/offsets, readout transforms, R9d shape,
# support-mean renormalization, or a negative intercept.
R9F_REMEDIATION_PROFILES: tuple[str, ...] = ("r9f_diesel_pre_offset_pathlength_reference_attenuation_v1",)

# R9h DIESEL support-CH-center/drop-1720 isolation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# and for non-compliant DIESEL rows. On explicit DIESEL petrochem rows carrying
# the R9h support-CH-center route, it keeps the full R3d micro-path / baseline
# / CH-width / CH-gain / feature contrast / readout / output-clip pipeline and
# changes ONLY the CH overtone centers to the 750-1550 nm support centers,
# dropping the 1720 nm band. It adds no damping windows, no 975 nm continuum
# hump, no support intercept/shape/redistribution/attenuation, no readout
# transform, and no additional guard clip.
R9H_REMEDIATION_PROFILES: tuple[str, ...] = ("r9h_diesel_support_ch_center_drop1720_isolation_v1",)

# R9i DIESEL CH width/gain isolation profile. Bench-only diagnostic-only
# opt-in extension that inherits R3d for every non-DIESEL row and for
# non-compliant DIESEL rows. On explicit DIESEL petrochem rows carrying the
# R9i CH width/gain route, it keeps the full R3d micro-path / baseline /
# CH-center / feature contrast / readout / output-clip pipeline and changes
# ONLY the scalar CH overtone width and gain range. It adds no damping windows,
# no 975 nm continuum hump, no support intercept/shape/redistribution/
# attenuation, no readout transform, and no additional guard clip.
R9I_REMEDIATION_PROFILES: tuple[str, ...] = ("r9i_diesel_ch_width_gain_isolation_v1",)

# R9j DIESEL residual damping-only isolation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# and for non-compliant DIESEL rows. On explicit DIESEL petrochem rows carrying
# the R9j residual damping route, it keeps the full R3d micro-path / baseline /
# CH-center / CH-width / CH-gain / feature contrast / readout / output-clip
# pipeline and changes ONLY the residual damping windows and damping strength
# range. It adds no 975 nm continuum hump, no support intercept/shape/
# redistribution/attenuation, no readout transform, and no additional guard
# clip.
R9J_REMEDIATION_PROFILES: tuple[str, ...] = ("r9j_diesel_residual_damping_isolation_v1",)

# R9k DIESEL continuum-hump-only isolation profile. Bench-only
# diagnostic-only opt-in extension that inherits R3d for every non-DIESEL row
# and for non-compliant DIESEL rows. On explicit DIESEL petrochem rows carrying
# the R9k continuum-hump route, it keeps the full R3d micro-path / baseline /
# CH-center / CH-width / CH-gain / feature contrast / readout / output-clip
# pipeline and changes ONLY the R4c 975 nm continuum hump. It adds no residual
# damping windows, no R9e/R9f attenuation, no support intercept/shape/
# redistribution, no readout transform, and no additional guard clip.
R9K_REMEDIATION_PROFILES: tuple[str, ...] = ("r9k_diesel_continuum_hump_isolation_v1",)

# R9l DIESEL residual-damping + clean attenuation controlled-combination
# profile. Bench-only diagnostic-only opt-in extension that inherits R3d for
# every non-DIESEL row and for non-compliant DIESEL rows. On explicit DIESEL
# petrochem rows carrying the R9l route, it combines exactly the R9j residual
# damping constants with the R9e clean support-only attenuation stage. It keeps
# R3d CH centers/width/gain, adds no continuum hump, no pre-offset attenuation,
# no support intercept/shape/redistribution, no readout transform, and no
# additional guard clip.
R9L_REMEDIATION_PROFILES: tuple[str, ...] = ("r9l_diesel_residual_damping_clean_attenuation_v1",)

# R9m DIESEL width/gain + residual-damping + clean attenuation final
# controlled-combination profile. Bench-only diagnostic-only opt-in extension
# that inherits R3d for every non-DIESEL row and for non-compliant DIESEL rows.
# On explicit DIESEL petrochem rows carrying the R9m route, it combines exactly
# the R9i CH width/gain constants, R9j residual damping constants, and R9e
# clean support-only attenuation stage. It keeps R3d CH centers including
# 1720.0, adds no continuum hump, no pre-offset attenuation, no support
# intercept/shape/redistribution, no readout transform, and no additional guard
# clip.
R9M_REMEDIATION_PROFILES: tuple[str, ...] = ("r9m_diesel_width_gain_damping_clean_attenuation_v1",)

# P2a DIESEL row-level pathlength/reference profile. Bench-only diagnostic-only
# opt-in Palier 2 extension that inherits R3d for every non-DIESEL row and for
# non-compliant DIESEL rows. On explicit DIESEL petrochem rows carrying the P2a
# route, it keeps the full R3d micro-path / baseline / CH-center / CH-width /
# CH-gain / feature contrast / readout / output-clip pipeline and changes ONLY
# the row-level pathlength/reference scale after the R3d output clip. Unlike
# R9e/R9l/R9m, the factor is applied to the full row over every generated
# wavelength, not only the 750-1550 nm support. It adds no damping, CH retune,
# continuum hump, support-only correction, readout transform, offset, or
# additional guard clip.
P2A_REMEDIATION_PROFILES: tuple[str, ...] = ("p2a_diesel_row_pathlength_reference_v1",)

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
    *R2X_REMEDIATION_PROFILES,
    *R2Y_REMEDIATION_PROFILES,
    *R2Z_REMEDIATION_PROFILES,
    *R3A_REMEDIATION_PROFILES,
    *R3B_REMEDIATION_PROFILES,
    *R3C_REMEDIATION_PROFILES,
    *R3D_REMEDIATION_PROFILES,
    *R3E_REMEDIATION_PROFILES,
    *R3F_REMEDIATION_PROFILES,
    *R3G_REMEDIATION_PROFILES,
    *R4A_REMEDIATION_PROFILES,
    *R4B_REMEDIATION_PROFILES,
    *R4C_REMEDIATION_PROFILES,
    *R5A_REMEDIATION_PROFILES,
    *R5B_REMEDIATION_PROFILES,
    *R5C_REMEDIATION_PROFILES,
    *R6A_REMEDIATION_PROFILES,
    *R7A_REMEDIATION_PROFILES,
    *R8A_REMEDIATION_PROFILES,
    *R8B_REMEDIATION_PROFILES,
    *R9B_REMEDIATION_PROFILES,
    *R9C_REMEDIATION_PROFILES,
    *R9D_REMEDIATION_PROFILES,
    *R9E_REMEDIATION_PROFILES,
    *R9F_REMEDIATION_PROFILES,
    *R9H_REMEDIATION_PROFILES,
    *R9I_REMEDIATION_PROFILES,
    *R9J_REMEDIATION_PROFILES,
    *R9K_REMEDIATION_PROFILES,
    *R9L_REMEDIATION_PROFILES,
    *R9M_REMEDIATION_PROFILES,
    *P2A_REMEDIATION_PROFILES,
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
            (f"classification datasets need at least one sample per declared class, got n_samples={resolved_n_samples}, n_classes={target_config['n_classes']}"),
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
            "reason": ("SyntheticNIRSGenerator.generate_from_concentrations expects mixture fractions whose rows sum to approximately 1.0."),
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

    When set to ``"r2x_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only explicit MANURE21 records use a coarser particulate
    albedo-dispersion readout that keeps R2w's centered continuum expectation
    while reducing residual variance-under without derivative-over dominance.

    When set to ``"r2y_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only explicit MANURE21 records use a soft low-frequency
    centered albedo-dispersion readout derived from R2w's conservative route.

    When set to ``"r2z_sentinel_matrix_v1"``, R2s inheritance is kept for all
    non-MANURE21 rows. Only explicit MANURE21 records use R2w's conservative
    centered albedo envelope with mean-neutral compositional heterogeneity and
    smooth low-frequency readout variability.

    When set to ``"r3a_corn_matrix_v1"``, R2w inheritance is kept for all
    non-CORN rows. Only explicit CORN records use a grain-powder diffuse
    reflectance/readout prior with a fixed apparent-absorbance baseline,
    centered residual transfer, and coarse particle-size smoothing.

    When set to ``"r3b_corn_matrix_v1"``, the same CORN-only route is used
    with larger fixed particle path dispersion and broader smoothing to recover
    band amplitude while keeping the continuum and first derivative bounded.

    When set to ``"r3c_diesel_matrix_v1"``, R3b inheritance is kept for every
    non-DIESEL row. Only explicit DIESEL records use a lower-offset
    blank-referenced CH-overtone micro-path readout derived from R2s, with
    bounded amplitude damping and no real-stat capture.

    When set to ``"r3d_diesel_matrix_v1"``, R3c inheritance is kept for every
    non-DIESEL row. Only explicit DIESEL records use the same fixed CH-overtone
    readout with a shorter continuum path and lower detector offset to reduce
    broadband absorbance shift while keeping residual contrast active.

    When set to ``"r3e_diesel_matrix_v1"``, R3d inheritance is kept for every
    non-DIESEL row. Only explicit DIESEL records use the same fixed CH-overtone
    readout with a minimal continuum path, near-zero detector offset, and
    retained residual CH contrast. This is diagnostic-only and is not promoted
    over R3d.

    When set to ``"r3f_diesel_matrix_v1"``, R3e inheritance is kept for every
    non-DIESEL row. Only explicit DIESEL records keep the lowered blank
    reference from R3e while restoring R3d residual contrast and the fixed R3d
    CH-overtone path perturbation. This is diagnostic-only and is not promoted
    over R3d.

    When set to ``"r3g_diesel_matrix_v1"``, R3f inheritance is kept for every
    non-DIESEL row. Only explicit DIESEL records add a fixed mean-neutral
    hydrocarbon CH-band envelope to the R3f micro-path readout. This is
    diagnostic-only and is not promoted over R3d.

    When set to ``"r4a_diesel_basis_v1"``, R3d inheritance is kept for every
    non-DIESEL row (NOT R3e/R3f/R3g). Only explicit DIESEL records use the same
    R3d micro-path continuum and detector offset, drop the 1720 nm CH center,
    widen the remaining CH overtones at lower gain, damp the residual inside
    the over-structured 1100-1500 nm region, and add a short-continuum
    hydrocarbon scatter hump centered at 975 nm restricted to the 750-1550 nm
    DIESEL real basis support. This is diagnostic-only and is not promoted over
    R3d.

    When set to ``"r4b_diesel_derivative_restore_v1"``, R3d inheritance is kept
    for every non-DIESEL row (NOT R4a). Only explicit DIESEL records use the
    same R3d micro-path continuum and detector offset and the same support-only
    CH overtone center set as R4a (no 1720 nm), but with a narrower CH overtone
    width and slightly higher gain so the first-derivative structure is
    restored, weaker and narrower residual damping so the 1100-1500 nm
    derivative is no longer flattened, and a narrower lower-amplitude
    short-continuum hydrocarbon hump centered at 975 nm restricted to the
    750-1550 nm support. This is diagnostic-only and is not promoted over R3d.

    When set to ``"r4c_diesel_balanced_derivative_v1"``, R3d inheritance is kept
    for every non-DIESEL row (NOT R4a/R4b). Only explicit DIESEL records use the
    same R3d micro-path continuum and detector offset and the same support-only
    CH overtone center set as R4a/R4b (no 1720 nm), but with a narrower CH
    overtone width than R4b (36 nm vs 38 nm) and slightly higher gain
    (0.092-0.155 vs 0.085-0.145), weaker and narrower residual damping windows
    inside the 1100-1500 nm region so more first-derivative structure is kept,
    and a narrower lower-amplitude short-continuum hydrocarbon hump centered at
    975 nm restricted to the 750-1550 nm support so the 900-1050 nm level is
    still lifted without flattening the derivative. This is diagnostic-only and
    is not promoted over R3d, and does not authorize any nirs4all integration.

    When set to ``"r5a_diesel_absorbance_readout_v1"``,
    ``"r5b_diesel_transmittance_readout_v1"``, or
    ``"r5c_diesel_blank_referenced_intensity_v1"``, R3d inheritance is kept for
    every non-DIESEL row (NOT R4a/R4b/R4c). On records that carry the explicit
    matching DIESEL route the full R4c CH overtone / damping / hump / path /
    feature-contrast pipeline is applied byte-for-byte, with the same RNG
    sequence as R4c, and only the final spectral readout space is varied:
    R5a keeps absorbance, R5b returns transmittance ``10**-A`` clipped to
    [0, 1], R5c returns blank-referenced intensity ``1 - 10**-A`` clipped to
    [0, 1]. R5 is bench-only diagnostic-only; not promoted over R3d, not a
    B2/B3/B4/B5 gate, and does not authorize any nirs4all integration.

    When set to ``"r6a_diesel_centered_hydrocarbon_shape_v1"``, R3d
    inheritance is kept for every non-DIESEL row (NOT R4a/R4b/R4c/R5a/R5b/R5c).
    On records that carry the explicit DIESEL shape route the full R4c
    absorbance pipeline is applied byte-for-byte (with the same RNG seed source
    so target draws and intermediate R4c draws are aligned with R4c/R5a) and a
    small fixed mean-neutral hydrocarbon CH-band shape envelope is added on
    the 750-1550 nm support: outside the support the envelope is identically
    zero, inside the support it is normalized to peak 1 and recentered to zero
    mean over the support so the support-mean of the addition is zero by
    construction. Constants are mechanistic priors only; no real spectra,
    statistics, PCA, covariance, quantiles, ML/DL output, labels, targets,
    splits, AUC, calibration, or threshold tuning is consulted. R6a is
    bench-only diagnostic-only; not promoted over R3d, not a B2/B3/B4/B5 gate,
    and does not authorize any nirs4all integration.

    When set to ``"r7a_diesel_support_centered_residual_transfer_v1"``, R3d
    inheritance is kept for every non-DIESEL row (NOT
    R4a/R4b/R4c/R5a/R5b/R5c/R6a). On records that carry the explicit DIESEL
    residual route the R4a-like absorbance base is applied (R3d micro-path
    continuum and detector offset, support-only CH overtone centers
    1150/1210/1390/1460 nm at width 46 nm and gain 0.055-0.105, R4a damping
    windows and strength range, 975 nm short-continuum hydrocarbon hump on the
    750-1550 nm support) and a bounded support-centered residual transfer is
    added before the final absorbance clip: the synthetic hydrocarbon residual
    ``X_in - continuum`` is masked to the 750-1550 nm DIESEL real basis
    support, row-centered on the support so its support-mean is zero by
    construction, scaled by a fixed bounded draw in [0.08, 0.18], and added to
    the R4a base. A final non-negative absorbance clip is then applied; the
    audit trail records the clip rule, clip fraction, and pre/post-clip
    min/max so any non-zero clip activity is observable. Constants are
    mechanistic priors only; no real spectra, statistics, PCA, covariance,
    quantiles, ML/DL output, labels, targets, splits, AUC, calibration, or
    threshold tuning is consulted. R7a is bench-only diagnostic-only; not
    promoted over R3d, not a B2/B3/B4/B5 gate, and does not authorize any
    nirs4all integration.

    When set to ``"r8a_diesel_mean_preserving_micro_path_modulation_v1"``,
    R3d inheritance is kept for every non-DIESEL row (NOT
    R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a). On records that carry the explicit
    DIESEL micro-path modulation route the R4a-like absorbance base is
    applied (R3d micro-path continuum and detector offset, support-only CH
    overtone centers 1150/1210/1390/1460 nm at width 46 nm and gain
    0.055-0.105, R4a damping windows and strength range, 975 nm
    short-continuum hydrocarbon hump on the 750-1550 nm support) and then,
    AFTER the R4a base goes through its standard non-negative absorbance
    clip, the support of the base is multiplied by a bounded mean-preserving
    multiplicative modulation derived from the synthetic hydrocarbon
    residual ``X_in - continuum``: the residual is masked to the 750-1550 nm
    support, row-centered on the support so its support-mean is zero,
    normalized by a robust synthetic-only scale (p95 of |residual| with a
    numerical epsilon), bounded to a dimensionless shape in [-1, 1], and
    applied as ``X_support *= exp(strength * shape)`` with a bounded
    mechanistic strength draw. The support row mean is then exactly
    preserved by a multiplicative renormalization. Outside the support the
    readout is identically equal to the R4a base. A final non-negative
    guard clip is recorded in the audit trail (clip rule, clip fraction,
    pre/post-modulation min/max) but is expected to be a no-op because the
    base is non-negative and the modulation is positive. Constants are
    mechanistic priors only; no real spectra, statistics, PCA, covariance,
    quantiles, ML/DL output, labels, targets, splits, AUC, calibration, or
    threshold tuning is consulted. R8a is bench-only diagnostic-only; not
    promoted over R3d, not a B2/B3/B4/B5 gate, and does not authorize any
    nirs4all integration.

    When set to
    ``"r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1"``, R3d
    inheritance is kept for every non-DIESEL row (NOT
    R4a/R4b/R4c/R5a/R5b/R5c/R6a/R7a/R8a). On records that carry the explicit
    DIESEL R8b micro-path modulation route the full R4c balanced-derivative
    absorbance base is applied byte-for-byte, then the same bounded
    support-mean-preserving multiplicative modulation mechanism as R8a is
    applied after the base non-negative clip. This is bench-only
    diagnostic-only; not promoted over R3d, not a B2/B3/B4/B5 gate, and does
    not authorize any nirs4all integration.
    """
    builder_config = prior_to_builder_config(
        record,
        n_samples=n_samples,
        random_seed=random_seed,
    )
    rng = np.random.default_rng(builder_config["random_state"])
    concentrations = _sample_concentrations(record, rng, builder_config["n_samples"])

    effective_remediation_profile = _effective_builder_remediation_profile(remediation_profile, record) if remediation_profile is not None else None
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
        failures.append(
            {
                "reason": "shape_mismatch",
                "field": "concentrations",
                "message": str(concentrations.shape),
            }
        )
    concentration_row_sums = np.sum(concentrations, axis=1)
    if not np.isfinite(concentrations).all():
        failures.append(
            {
                "reason": "non_finite",
                "field": "concentrations",
                "message": "concentrations contain non-finite values",
            }
        )
    if not np.allclose(concentration_row_sums, 1.0, rtol=1e-9, atol=1e-9):
        failures.append(
            {
                "reason": "concentration_row_sum_mismatch",
                "field": "concentrations",
                "message": (f"row-normalized concentrations must sum to 1.0; observed range={(float(np.min(concentration_row_sums)), float(np.max(concentration_row_sums)))}"),
            }
        )
    if not np.isfinite(X).all():
        failures.append({"reason": "non_finite", "field": "X", "message": "spectra contain non-finite values"})
    if not np.isfinite(y).all():
        failures.append({"reason": "non_finite", "field": "y", "message": "target contains non-finite values"})
    if not np.isfinite(wavelengths).all() or not np.all(np.diff(wavelengths) > 0):
        failures.append(
            {
                "reason": "invalid_wavelengths",
                "field": "wavelengths",
                "message": "wavelengths must be finite and strictly increasing",
            }
        )
    expected_low, expected_high = builder_config["features"]["wavelength_range"]
    if wavelengths[0] < expected_low - 1e-9 or wavelengths[-1] > expected_high + 1e-9:
        failures.append(
            {
                "reason": "wavelength_range_mismatch",
                "field": "wavelengths",
                "message": f"{(wavelengths[0], wavelengths[-1])} outside {(expected_low, expected_high)}",
            }
        )
    if target["type"] == "classification":
        observed = set(np.unique(y).astype(int).tolist())
        expected = set(range(target["n_classes"]))
        if observed != expected:
            failures.append(
                {
                    "reason": "invalid_class_labels",
                    "field": "y",
                    "message": f"observed={sorted(observed)}, expected={sorted(expected)}",
                }
            )
    else:
        target_min, target_max = target["range"]
        if float(np.min(y)) < target_min - 1e-9 or float(np.max(y)) > target_max + 1e-9:
            failures.append(
                {
                    "reason": "target_range_mismatch",
                    "field": "y",
                    "message": f"target outside {(target_min, target_max)}",
                }
            )

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "unsupported_fields": [],
        "adapter_notes": [("measurement_mode is passed to SyntheticNIRSGenerator and preserved in metadata; A2 contract checks do not validate mode-specific optical physics.")],
        "checks": {
            "shape": not any(f["reason"] == "shape_mismatch" for f in failures),
            "finite": bool(np.isfinite(X).all() and np.isfinite(y).all()),
            "wavelengths_monotonic": bool(np.all(np.diff(wavelengths) > 0)),
            "target_contract": not any(f["reason"] in {"invalid_class_labels", "target_range_mismatch"} for f in failures),
            "concentrations_row_normalized": not any(f["reason"] == "concentration_row_sum_mismatch" for f in failures),
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
    return cast(
        "dict[str, Any]",
        _to_builtin(
            {
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
            }
        ),
    )


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
        raise ValueError(f"n_samples ({concentrations.shape[0]}) must be >= n_classes ({n_classes})")
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
    raise PriorDatasetAdapterError(
        {
            "status": "failed",
            "record": {
                "domain_key": record.domain_key,
                "instrument_key": record.instrument_key,
                "measurement_mode": record.measurement_mode,
                "target_prior": record.target_prior,
            },
            "failures": [{"reason": reason, "field": "prior_config", "message": message}],
        }
    )


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
        raise ValueError(f"unknown R2a mechanistic profile {profile!r}; valid profiles are {list(R2A_MECHANISTIC_PROFILES)}")
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
        baseline = amp_linear[:, None] * normalized_wl[None, :] + amp_quad[:, None] * (normalized_wl[None, :] ** 2)
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
        "reason": ("fixed mechanistic approximation determined by profile name and seed; no real data, labels, targets, splits, or AUC consulted"),
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

_R2X_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2x_sentinel_matrix_remediation",
}

_R2Y_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2y_sentinel_matrix_remediation",
}

_R2Z_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r2z_sentinel_matrix_remediation",
}

_R3A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3a_corn_matrix_remediation",
}

_R3B_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3b_corn_matrix_remediation",
}

_R3C_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3c_diesel_matrix_remediation",
}

_R3D_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3d_diesel_matrix_remediation",
}

_R3E_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3e_diesel_matrix_remediation",
}

_R3F_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3f_diesel_matrix_remediation",
}

_R3G_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r3g_diesel_matrix_remediation",
}

_R4A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r4a_diesel_basis_remediation",
}

_R4B_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r4b_diesel_derivative_restore_remediation",
}

_R4C_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r4c_diesel_balanced_derivative_remediation",
}

_R5A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r5a_diesel_absorbance_readout_remediation",
}

_R5B_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r5b_diesel_transmittance_readout_remediation",
}

_R5C_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r5c_diesel_blank_referenced_intensity_remediation",
}

_R6A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r6a_diesel_centered_hydrocarbon_shape_remediation",
}

_R7A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r7a_diesel_support_centered_residual_transfer_remediation",
}

_R8A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r8a_diesel_mean_preserving_micro_path_modulation",
}

_R8B_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r8b_diesel_r4c_base_mean_preserving_micro_path_modulation",
}

_R9B_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9b_diesel_support_intercept_remediation",
}

_R9C_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9c_diesel_selective_ch_bandwidth_damping_remediation",
}

_R9D_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": ("bench_only_r9d_diesel_energy_normalized_support_redistribution_remediation"),
}

_R9E_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9e_diesel_pathlength_reference_attenuation_remediation",
}

_R9F_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": ("bench_only_r9f_diesel_pre_offset_pathlength_reference_attenuation_remediation"),
}

_R9H_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": ("bench_only_r9h_diesel_support_ch_center_drop1720_isolation_remediation"),
}

_R9I_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9i_diesel_ch_width_gain_isolation_remediation",
}

_R9J_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9j_diesel_residual_damping_isolation_remediation",
}

_R9K_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_r9k_diesel_continuum_hump_isolation_remediation",
}

_R9L_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": ("bench_only_r9l_diesel_residual_damping_clean_attenuation_remediation"),
}

_R9M_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": ("bench_only_r9m_diesel_width_gain_damping_clean_attenuation_remediation"),
}

_P2A_AUDIT_BASE: dict[str, Any] = {
    **_R2C_AUDIT_BASE,
    "scope": "bench_only_p2a_diesel_row_pathlength_reference_remediation",
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
        "additive_baseline_range": (_R2Q_LUCAS_PH_ORGANIC_HUMIC_ALBEDO_ABSORBANCE_RANGE),
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
        "sugar_solids_band_absorbance_range": (_R2R_FRUIT_PUREE_SUGAR_SOLIDS_BAND_RANGE),
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
        "spectra_source": ("beer_lambert_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"),
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
        "spectra_source": ("fixed_dark_organic_albedo_plus_particle_scatter_moisture_mineral_lumps"),
        "smoothing_fwhm_nm": _R2T_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2T_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2T_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2T_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2T_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2T_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "heterogeneity_source": ("fixed_dried_manure_particle_size_moisture_organic_mineral_lump_prior"),
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
        "spectra_source": ("fixed_dark_organic_albedo_plus_centered_particle_scatter_bands"),
        "smoothing_fwhm_nm": _R2U_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2U_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2U_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2U_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2U_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2U_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "heterogeneity_source": ("fixed_dried_manure_centered_particle_moisture_mineral_scatter_prior"),
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
        "spectra_source": ("fixed_dark_organic_albedo_plus_balanced_centered_particle_scatter_bands"),
        "smoothing_fwhm_nm": _R2V_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2V_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2V_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2V_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2V_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2V_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": ("fixed_dried_manure_balanced_centered_particle_moisture_mineral_scatter_prior"),
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
        "spectra_source": ("fixed_dark_organic_albedo_variance_plus_balanced_centered_particle_scatter_bands"),
        "smoothing_fwhm_nm": _R2W_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2W_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2W_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2W_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2W_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2W_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": ("fixed_dried_manure_albedo_variance_centered_particle_moisture_mineral_scatter_prior"),
        "scatter_source": "fixed_dried_manure_balanced_centered_scatter_prior",
        "albedo_source": "fixed_wide_dark_organic_mineral_albedo_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2x keeps the albedo midpoint used by R2w but widens the balanced cup-scale
# albedo dispersion slightly and applies coarser fixed particulate smoothing.
# These are mechanistic constants only; no sentinel metrics or real statistics
# are read to set them.
_R2X_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.80, 1.00)
_R2X_MANURE21_SMOOTHING_FWHM_NM: float = 26.0
_R2X_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.70,
    1.02,
)
_R2X_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.16, 0.16)
_R2X_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.14)
_R2X_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.13)
_R2X_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.10, 0.10)

_R2X_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2X_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_coarse_albedo_dispersion_centered_readout",
        "spectra_source": ("fixed_coarse_dark_organic_albedo_dispersion_plus_centered_particle_scatter_bands"),
        "smoothing_fwhm_nm": _R2X_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2X_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2X_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2X_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2X_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2X_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": ("fixed_dried_manure_coarse_albedo_dispersion_centered_particle_moisture_mineral_scatter_prior"),
        "scatter_source": "fixed_dried_manure_coarse_centered_scatter_prior",
        "albedo_source": "fixed_wide_dark_organic_mineral_albedo_dispersion_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2y restarts from R2w rather than pushing R2x. The continuum midpoint stays
# fixed and centered, the albedo spread matches R2x's bounded range while the
# residual path/slopes stay softer, and the fixed particulate smoothing is
# low-frequency so variance/amplitude transfer does not rely on sharper
# derivative structure.
_R2Y_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.82, 1.02)
_R2Y_MANURE21_SMOOTHING_FWHM_NM: float = 30.0
_R2Y_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.70,
    1.02,
)
_R2Y_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.15, 0.15)
_R2Y_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.14)
_R2Y_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.13)
_R2Y_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.10, 0.10)

_R2Y_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "path_factor_range": _R2Y_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_soft_low_frequency_albedo_dispersion_centered_readout",
        "spectra_source": ("fixed_soft_low_frequency_dark_organic_albedo_dispersion_plus_centered_particle_scatter_bands"),
        "smoothing_fwhm_nm": _R2Y_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2Y_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2Y_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2Y_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2Y_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2Y_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "heterogeneity_source": ("fixed_dried_manure_soft_low_frequency_albedo_dispersion_centered_particle_moisture_mineral_scatter_prior"),
        "scatter_source": "fixed_dried_manure_soft_low_frequency_centered_scatter_prior",
        "albedo_source": "fixed_soft_wide_dark_organic_mineral_albedo_dispersion_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

# R2z keeps R2w's albedo envelope and balanced centering, but shifts part of
# the extra variance transfer into the dried-manure composition prior. Scaling
# all MANURE21 alphas by the same factor preserves the textbook component
# means while lowering concentration, which increases inter-sample
# organic/mineral heterogeneity without reading real MANURE21 statistics.
_R2Z_MANURE21_ALPHA_CONCENTRATION_SCALE: float = 0.72
_R2Z_MANURE21_ALPHAS: dict[str, float] = {name: value * _R2Z_MANURE21_ALPHA_CONCENTRATION_SCALE for name, value in _R2N_MANURE21_ALPHAS.items()}
_R2Z_MANURE21_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.84, 1.02)
_R2Z_MANURE21_SMOOTHING_FWHM_NM: float = 32.0
_R2Z_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE: tuple[float, float] = (
    0.71,
    1.01,
)
_R2Z_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE: tuple[float, float] = (-0.10, 0.10)
_R2Z_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.12)
_R2Z_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE: tuple[float, float] = (0.00, 0.11)
_R2Z_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE: tuple[float, float] = (-0.08, 0.08)

_R2Z_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2S_DOMAIN_RULES,
    "environmental_soil": {
        **_R2N_DOMAIN_RULES["environmental_soil"],
        "alphas": _R2Z_MANURE21_ALPHAS,
        "path_factor_range": _R2Z_MANURE21_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "dried_manure_compositional_heterogeneity_centered_readout",
        "spectra_source": ("fixed_mean_neutral_compositional_heterogeneity_plus_smooth_centered_scatter_bands"),
        "smoothing_fwhm_nm": _R2Z_MANURE21_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R2Z_MANURE21_ORGANIC_ALBEDO_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R2Z_MANURE21_PARTICLE_SCATTER_SLOPE_RANGE,
        "moisture_patch_absorbance_range": _R2Z_MANURE21_MOISTURE_PATCH_ABSORBANCE_RANGE,
        "organic_lump_absorbance_range": _R2Z_MANURE21_ORGANIC_LUMP_ABSORBANCE_RANGE,
        "mineral_ash_absorbance_range": _R2Z_MANURE21_MINERAL_ASH_ABSORBANCE_RANGE,
        "balanced_centered_draws": True,
        "readout_centering_range_nm": (1100.0, 2500.0),
        "readout_centering_grid": "uniform_wavenumber",
        "composition_heterogeneity": "mean_neutral_dirichlet_concentration_scaled",
        "composition_alpha_concentration_scale": _R2Z_MANURE21_ALPHA_CONCENTRATION_SCALE,
        "heterogeneity_source": ("fixed_dried_manure_mean_neutral_compositional_heterogeneity_smooth_centered_scatter_prior"),
        "scatter_source": "fixed_dried_manure_smooth_low_frequency_centered_scatter_prior",
        "albedo_source": "fixed_wide_dark_organic_mineral_albedo_prior_r2w_envelope",
        "output_clip_absorbance": (0.0, None),
    },
}

# R3a CORN grain/powder readout. CORN repeated audits show a low apparent
# absorbance continuum with over-energetic derivatives under the inherited R2f
# powder smoothing route. The readout below treats generated grain absorbance
# as a centered chemical residual observed against a fixed corn-meal diffuse
# reflectance albedo, then applies broad particle-size smoothing and weak
# starch/moisture bands. Constants are optical/material priors only.
_R3A_CORN_GRAIN_ALPHAS: dict[str, float] = {
    "starch": 18.0,
    "moisture": 2.8,
    "protein": 2.2,
    "lipid": 0.9,
    "cellulose": 0.7,
    "gluten": 0.6,
    "hemicellulose": 1.0,
    "dietary_fiber": 0.5,
}
_R3A_CORN_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.22, 0.34)
_R3A_CORN_SMOOTHING_FWHM_NM: float = 42.0
_R3A_CORN_ALBEDO_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.34,
    0.43,
)
_R3A_CORN_SCATTER_SLOPE_ABSORBANCE_RANGE: tuple[float, float] = (-0.04, 0.04)
_R3A_CORN_MOISTURE_BAND_ABSORBANCE_RANGE: tuple[float, float] = (0.0, 0.025)
_R3A_CORN_STARCH_BAND_ABSORBANCE_RANGE: tuple[float, float] = (0.0, 0.025)

_R3B_CORN_RESIDUAL_PATH_FACTOR_RANGE: tuple[float, float] = (0.90, 1.35)
_R3B_CORN_SMOOTHING_FWHM_NM: float = 140.0
_R3B_CORN_SCATTER_SLOPE_ABSORBANCE_RANGE: tuple[float, float] = (-0.012, 0.012)
_R3B_CORN_MOISTURE_BAND_ABSORBANCE_RANGE: tuple[float, float] = (0.035, 0.105)
_R3B_CORN_STARCH_BAND_ABSORBANCE_RANGE: tuple[float, float] = (0.035, 0.105)

_R3A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2W_DOMAIN_RULES,
    "agriculture_grain": {
        "alphas": _R3A_CORN_GRAIN_ALPHAS,
        "path_factor_range": _R3A_CORN_RESIDUAL_PATH_FACTOR_RANGE,
        "composition_rule": "tight_dirichlet_corn_grain_powder_centered",
        "spectra_rule": "corn_powder_albedo_baseline_smoothing_readout",
        "composition_source": "textbook_corn_grain_powder_composition",
        "spectra_source": ("fixed_corn_meal_diffuse_reflectance_albedo_plus_particle_smoothing"),
        "smoothing_fwhm_nm": _R3A_CORN_SMOOTHING_FWHM_NM,
        "additive_baseline_range": _R3A_CORN_ALBEDO_BASELINE_ABSORBANCE_RANGE,
        "scatter_slope_absorbance_range": _R3A_CORN_SCATTER_SLOPE_ABSORBANCE_RANGE,
        "moisture_band_absorbance_range": _R3A_CORN_MOISTURE_BAND_ABSORBANCE_RANGE,
        "starch_band_absorbance_range": _R3A_CORN_STARCH_BAND_ABSORBANCE_RANGE,
        "constant_status": "fixed_mechanistic_prior",
        "readout_space": "corn_powder_raw_apparent_absorbance",
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_corn_route",
        "matrix_source": "fixed_corn_grain_powder_prior",
        "scatter_source": "fixed_particle_size_diffuse_reflectance_smoothing_prior",
        "albedo_source": "fixed_corn_meal_reflectance_albedo_prior",
        "output_clip_absorbance": (0.0, None),
    },
}

_R3B_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R2W_DOMAIN_RULES,
    "agriculture_grain": {
        **_R3A_DOMAIN_RULES["agriculture_grain"],
        "path_factor_range": _R3B_CORN_RESIDUAL_PATH_FACTOR_RANGE,
        "spectra_rule": "corn_powder_albedo_path_dispersion_smoothing_readout",
        "spectra_source": ("fixed_corn_meal_albedo_plus_coarse_particle_path_dispersion_smoothing"),
        "smoothing_fwhm_nm": _R3B_CORN_SMOOTHING_FWHM_NM,
        "scatter_slope_absorbance_range": _R3B_CORN_SCATTER_SLOPE_ABSORBANCE_RANGE,
        "moisture_band_absorbance_range": _R3B_CORN_MOISTURE_BAND_ABSORBANCE_RANGE,
        "starch_band_absorbance_range": _R3B_CORN_STARCH_BAND_ABSORBANCE_RANGE,
        "scatter_source": ("fixed_coarse_particle_size_path_dispersion_smoothing_prior"),
    },
}

_R3C_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.024,
    0.036,
)
_R3C_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE: tuple[float, float] = (0.22, 0.31)
_R3C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.11, 0.18)

_R3C_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3B_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R2S_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R3C_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "feature_contrast_range": _R3C_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE,
        "ch_overtone_gain_range": _R3C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "spectra_source": ("beer_lambert_low_offset_blank_referenced_micro_path_with_fixed_ch_overtone_contrast"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r3c"),
    },
}

_R3D_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.010,
    0.018,
)
_R3D_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.00005,
    0.00035,
)

_R3D_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3C_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3C_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R3D_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "additive_baseline_range": _R3D_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE,
        "spectra_source": ("beer_lambert_ultra_short_blank_referenced_micro_path_with_low_detector_offset_and_fixed_ch_overtone_contrast"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r3d"),
    },
}

_R3E_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.004,
    0.009,
)
_R3E_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.0,
    0.00010,
)
_R3E_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE: tuple[float, float] = (0.20, 0.28)

_R3E_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R3E_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "additive_baseline_range": _R3E_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE,
        "feature_contrast_range": _R3E_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE,
        "spectra_source": ("beer_lambert_minimal_blank_referenced_micro_path_with_near_zero_detector_offset_and_fixed_ch_residual_contrast"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r3e"),
    },
}

_R3F_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.009,
    0.016,
)
_R3F_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.0,
    0.00012,
)

_R3F_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3E_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3E_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R3F_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "additive_baseline_range": _R3F_PETROCHEM_FUELS_BASELINE_ABSORBANCE_RANGE,
        "feature_contrast_range": _R3C_PETROCHEM_FUELS_FEATURE_CONTRAST_RANGE,
        "ch_overtone_gain_range": _R3C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "spectra_source": ("beer_lambert_short_blank_referenced_micro_path_with_low_detector_offset_and_restored_ch_residual_contrast"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r3f"),
    },
}

_R3G_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE: tuple[float, float] = (
    0.003,
    0.007,
)
_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.0005,
    0.0010,
)
_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_WIDTHS_NM: tuple[float, ...] = (
    30.0,
    34.0,
    42.0,
    46.0,
)
_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_WEIGHTS: tuple[float, ...] = (
    0.65,
    1.00,
    0.55,
    0.72,
)
_R3G_PETROCHEM_FUELS_CONTINUUM_SLOPE_RANGE: tuple[float, float] = (
    -0.00025,
    0.00010,
)

_R3G_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3F_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3F_DOMAIN_RULES["petrochem_fuels"],
        "path_factor_range": _R3G_PETROCHEM_FUELS_CONTINUUM_PATH_FACTOR_RANGE,
        "fixed_envelope_absorbance_range": (_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_ABSORBANCE_RANGE),
        "fixed_envelope_centers_nm": (_R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_CENTERS_NM),
        "fixed_envelope_widths_nm": _R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_WIDTHS_NM,
        "fixed_envelope_weights": _R3G_PETROCHEM_FUELS_FIXED_ENVELOPE_WEIGHTS,
        "continuum_slope_absorbance_range": (_R3G_PETROCHEM_FUELS_CONTINUUM_SLOPE_RANGE),
        "spectra_source": ("beer_lambert_short_blank_referenced_micro_path_with_fixed_hydrocarbon_ch_band_envelope"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_envelope_prior_explicit_diesel_route_r3g"),
    },
}

# R4a DIESEL basis constants. Path/baseline ranges match R3d byte-for-byte
# (R4a does not lower or restore the broadband level). The CH overtone centers
# drop the 1720 nm band (out of the 750-1550 nm real DIESEL support observed in
# the diagnostic audit), the CH overtone width is widened, and the gain range
# is lowered so the 1100-1500 nm residual is no longer over-structured. A
# shortwave hydrocarbon scatter hump centered at 975 nm with a 90 nm width is
# added, restricted to the 750-1550 nm support window, to lift energy in the
# 900-1050 nm region without disturbing the 1550 nm and longer support. Two
# residual damping windows centered at 1180 nm and 1425 nm attenuate the
# residual contrast inside the over-structured CH region without inverting it.
_R4A_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R4A_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM: float = 46.0
_R4A_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.055, 0.105)
_R4A_PETROCHEM_FUELS_DAMPING_WINDOWS_NM: tuple[tuple[float, float, float], ...] = (
    (1180.0, 70.0, 1.0),
    (1425.0, 85.0, 1.0),
)
_R4A_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE: tuple[float, float] = (0.30, 0.50)
_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM: float = 975.0
_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM: float = 90.0
_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE: tuple[float, float] = (
    0.00025,
    0.00065,
)
_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)

_R4A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_centers_nm": _R4A_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
        "ch_overtone_width_nm": _R4A_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R4A_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "damping_windows_nm": _R4A_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R4A_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "continuum_hump_center_nm": (_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM),
        "continuum_hump_width_nm": _R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM,
        "continuum_hump_amplitude_range": (_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE),
        "continuum_hump_support_nm": (_R4A_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM),
        "spectra_source": ("r4a_diesel_basis_v1_short_continuum_with_damped_residual_and_short_continuum_hydrocarbon_hump"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r4a"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

# R4b DIESEL derivative-restore constants. Path/baseline ranges match R3d
# byte-for-byte (R4b does not lower or restore the broadband level). The CH
# overtone centers match R4a (drop the 1720 nm band, support-only on 750-1550
# nm), but the CH overtone width is narrower than R4a (38 nm vs 46 nm) and the
# gain range is slightly higher (0.085-0.145 vs 0.055-0.105) so the first-
# derivative structure is restored without inverting it. The two residual
# damping windows centered at 1180 nm and 1425 nm are kept inside the
# over-structured CH region but with weaker individual weights (0.75/0.85),
# narrower widths (52/62 nm vs 70/85 nm), and a lower strength range
# (0.10-0.22 vs 0.30-0.50) so the 1100-1500 nm derivative is preserved. A
# narrower lower-amplitude short-continuum hydrocarbon hump (75 nm width,
# 0.00010-0.00035 amplitude) is kept on the 750-1550 nm support to lift
# the 900-1050 nm level without flattening the derivative.
_R4B_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R4B_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM: float = 38.0
_R4B_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.085, 0.145)
_R4B_PETROCHEM_FUELS_DAMPING_WINDOWS_NM: tuple[tuple[float, float, float], ...] = (
    (1180.0, 52.0, 0.75),
    (1425.0, 62.0, 0.85),
)
_R4B_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE: tuple[float, float] = (0.10, 0.22)
_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM: float = 975.0
_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM: float = 75.0
_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE: tuple[float, float] = (
    0.00010,
    0.00035,
)
_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)

_R4B_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_centers_nm": _R4B_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
        "ch_overtone_width_nm": _R4B_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R4B_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "damping_windows_nm": _R4B_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R4B_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "continuum_hump_center_nm": (_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM),
        "continuum_hump_width_nm": _R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM,
        "continuum_hump_amplitude_range": (_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE),
        "continuum_hump_support_nm": (_R4B_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM),
        "spectra_source": ("r4b_diesel_derivative_restore_v1_short_continuum_with_narrow_residual_damping_and_low_amplitude_hydrocarbon_hump"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r4b"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

# R4c DIESEL balanced-derivative constants. Path/baseline ranges match R3d
# byte-for-byte (R4c does not lower or restore the broadband level). The CH
# overtone centers match R4a/R4b (drop 1720 nm, support-only on 750-1550 nm),
# but the CH overtone width is narrower than R4b (36 nm vs 38 nm) and the gain
# range is slightly higher (0.092-0.155 vs 0.085-0.145), tightening the
# first-derivative balance while keeping the broadband level. The two residual
# damping windows are kept inside the over-structured 1100-1500 nm region but
# at narrower widths and weaker weights than R4b (52->46, 62->54, 0.75->0.60,
# 0.85->0.70) and a lower strength range (0.05-0.15 vs 0.10-0.22) so more of
# the first-derivative structure is preserved. A narrower lower-amplitude
# short-continuum hydrocarbon hump (72 nm width, 0.00010-0.00032 amplitude) is
# kept on the 750-1550 nm support to lift the 900-1050 nm level without
# flattening the derivative.
_R4C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R4C_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM: float = 36.0
_R4C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.092, 0.155)
_R4C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM: tuple[tuple[float, float, float], ...] = (
    (1180.0, 46.0, 0.60),
    (1425.0, 54.0, 0.70),
)
_R4C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE: tuple[float, float] = (0.05, 0.15)
_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM: float = 975.0
_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM: float = 72.0
_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE: tuple[float, float] = (
    0.00010,
    0.00032,
)
_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)

_R4C_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_centers_nm": _R4C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
        "ch_overtone_width_nm": _R4C_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R4C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "damping_windows_nm": _R4C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R4C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "continuum_hump_center_nm": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM),
        "continuum_hump_width_nm": _R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM,
        "continuum_hump_amplitude_range": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE),
        "continuum_hump_support_nm": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM),
        "spectra_source": ("r4c_diesel_balanced_derivative_v1_short_continuum_with_narrow_residual_damping_and_low_amplitude_hydrocarbon_hump"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r4c"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

# R5 DIESEL readout-space rules. Each profile inherits the full R4c petrochem
# fuels rule (composition, CH overtones, damping, hump, path, feature contrast)
# byte-for-byte and only varies the final readout space. ``readout_space_transform``
# is consumed by ``_apply_r2c_spectra_remediation`` after the absorbance pipeline
# completes. Constants are mechanistic priors only; never derived from real
# spectra/labels/targets/splits/PCA/AUC/thresholds.
_R5A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "readout_space_transform": "absorbance",
        "readout_space": "uncalibrated_raw_absorbance",
        "spectra_source": ("r5a_diesel_absorbance_readout_v1_inherits_r4c_balanced_derivative"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r5a"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

_R5B_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "readout_space_transform": "transmittance",
        "readout_space": "uncalibrated_raw_transmittance",
        "spectra_source": ("r5b_diesel_transmittance_readout_v1_inherits_r4c_balanced_derivative"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r5b"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

_R5C_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "readout_space_transform": "blank_referenced_intensity",
        "readout_space": "uncalibrated_raw_blank_referenced_intensity",
        "spectra_source": ("r5c_diesel_blank_referenced_intensity_v1_inherits_r4c_balanced_derivative"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_route_r5c"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_route",
    },
}

# R6a DIESEL centered hydrocarbon shape constants. The shape envelope reuses
# textbook hydrocarbon CH overtone bands from general NIR spectroscopy
# (1150/1210/1390/1460 nm), all four lying inside the 750-1550 nm support so
# the addition is identically zero outside the support and zero-mean over the
# support by construction. Widths/weights are mechanistic priors (no real
# spectra/PCA/quantiles consulted). The amplitude range is intentionally small
# so R6a only adds a faint shape on top of the R4c absorbance pipeline. R4c's
# rule keys (path, baseline, CH overtones, damping, hump) are inherited
# byte-for-byte.
_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_WIDTHS_NM: tuple[float, ...] = (
    30.0,
    34.0,
    42.0,
    46.0,
)
_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_WEIGHTS: tuple[float, ...] = (
    0.65,
    1.00,
    0.55,
    0.72,
)
_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)
_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_ABSORBANCE_RANGE: tuple[float, float] = (
    0.00020,
    0.00050,
)

_R6A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "shape_envelope_centers_nm": (_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_CENTERS_NM),
        "shape_envelope_widths_nm": (_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_WIDTHS_NM),
        "shape_envelope_weights": (_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_WEIGHTS),
        "shape_envelope_support_nm": (_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_SUPPORT_NM),
        "shape_envelope_absorbance_range": (_R6A_PETROCHEM_FUELS_SHAPE_ENVELOPE_ABSORBANCE_RANGE),
        "spectra_source": ("r6a_diesel_centered_hydrocarbon_shape_v1_inherits_r4c_balanced_derivative_with_zero_mean_support_envelope"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_shape_route_r6a"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_shape_route",
    },
}

# R7a DIESEL support-centered residual transfer constants. The R7a base reuses
# the full R4a petrochem_fuels rule byte-for-byte (R3d micro-path continuum
# and detector offset, support-only CH overtone centers 1150/1210/1390/1460 nm
# at width 46 nm and gain 0.055-0.105, R4a damping windows ((1180, 70, 1.0),
# (1425, 85, 1.0)) at strength 0.30-0.50, and a 975 nm short-continuum hump on
# the 750-1550 nm support). On top of the R4a base R7a adds a bounded
# support-centered residual transfer: the synthetic hydrocarbon residual
# ``X_in - continuum`` is masked to the 750-1550 nm DIESEL real basis support,
# row-centered on the support so its support-mean is zero by construction,
# scaled by a fixed bounded draw in [0.08, 0.18], and added to the R4a base
# before the final non-negative absorbance clip.
_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_RANGE: tuple[float, float] = (0.08, 0.18)
_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)
_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_SOURCE: str = "fixed_synthetic_hydrocarbon_residual_transfer_prior"

_R7A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4A_DOMAIN_RULES["petrochem_fuels"],
        "support_centered_residual_transfer_range": (_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_RANGE),
        "support_centered_residual_transfer_support_nm": (_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_SUPPORT_NM),
        "support_centered_residual_transfer_source": (_R7A_PETROCHEM_FUELS_RESIDUAL_TRANSFER_SOURCE),
        "spectra_source": ("r7a_diesel_support_centered_residual_transfer_v1_inherits_r4a_basis_with_support_centered_residual_transfer_before_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_residual_route_r7a"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": "exp09_dataset_token_diesel_residual_transfer_route",
    },
}

# R8a DIESEL mean-preserving micro-path modulation constants. The R8a base
# reuses the full R4a petrochem_fuels rule byte-for-byte (R3d micro-path
# continuum and detector offset, support-only CH overtone centers
# 1150/1210/1390/1460 nm at width 46 nm and gain 0.055-0.105, R4a damping
# windows ((1180, 70, 1.0), (1425, 85, 1.0)) at strength 0.30-0.50, and a
# 975 nm short-continuum hump on the 750-1550 nm support). On top of the R4a
# base (after the standard non-negative R4a final clip) R8a applies a
# bounded mean-preserving multiplicative modulation derived from the
# synthetic hydrocarbon residual ``X_in - continuum``: the residual is
# masked to the 750-1550 nm support, row-centered on the support, normalized
# by a robust synthetic-only scale (``p95_abs`` with a small numerical
# epsilon), bounded to a dimensionless shape in [-1, 1], and applied as
# ``X_support *= exp(strength * shape)`` with a bounded mechanistic strength
# draw in [0.10, 0.30]. The support row mean is exactly preserved by a
# multiplicative renormalization. Outside the support the readout is
# identically equal to the R4a base.
_R8A_PETROCHEM_FUELS_MODULATION_STRENGTH_RANGE: tuple[float, float] = (0.10, 0.30)
_R8A_PETROCHEM_FUELS_MODULATION_SUPPORT_NM: tuple[float, float] = (750.0, 1550.0)
_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_METHOD: str = "p95_abs"
_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_EPSILON: float = 1.0e-9
_R8A_PETROCHEM_FUELS_MODULATION_SHAPE_CLIP: tuple[float, float] = (-1.0, 1.0)
_R8A_PETROCHEM_FUELS_MODULATION_SOURCE: str = "synthetic_internal_residual_only"

_R8A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4A_DOMAIN_RULES["petrochem_fuels"],
        "support_centered_micro_path_modulation_strength_range": (_R8A_PETROCHEM_FUELS_MODULATION_STRENGTH_RANGE),
        "support_centered_micro_path_modulation_support_nm": (_R8A_PETROCHEM_FUELS_MODULATION_SUPPORT_NM),
        "support_centered_micro_path_modulation_normalization": (_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_METHOD),
        "support_centered_micro_path_modulation_normalization_epsilon": (_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_EPSILON),
        "support_centered_micro_path_modulation_shape_clip": (_R8A_PETROCHEM_FUELS_MODULATION_SHAPE_CLIP),
        "support_centered_micro_path_modulation_source": (_R8A_PETROCHEM_FUELS_MODULATION_SOURCE),
        "spectra_source": ("r8a_diesel_mean_preserving_micro_path_modulation_v1_inherits_r4a_basis_with_mean_preserving_multiplicative_modulation_after_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_micro_path_modulation_route_r8a"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp09_dataset_token_diesel_micro_path_modulation_route"),
    },
}

# R8b keeps the R8a modulation contract but swaps the pre-modulation base from
# R4a to R4c so the balanced-derivative content is retained.
_R8B_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "support_centered_micro_path_modulation_strength_range": (_R8A_PETROCHEM_FUELS_MODULATION_STRENGTH_RANGE),
        "support_centered_micro_path_modulation_support_nm": (_R8A_PETROCHEM_FUELS_MODULATION_SUPPORT_NM),
        "support_centered_micro_path_modulation_normalization": (_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_METHOD),
        "support_centered_micro_path_modulation_normalization_epsilon": (_R8A_PETROCHEM_FUELS_MODULATION_NORMALIZATION_EPSILON),
        "support_centered_micro_path_modulation_shape_clip": (_R8A_PETROCHEM_FUELS_MODULATION_SHAPE_CLIP),
        "support_centered_micro_path_modulation_source": (_R8A_PETROCHEM_FUELS_MODULATION_SOURCE),
        "spectra_source": ("r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1_inherits_r4c_balanced_derivative_with_mean_preserving_multiplicative_modulation_after_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_micro_path_modulation_route_r8b"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp09_dataset_token_diesel_r8b_micro_path_modulation_route"),
    },
}

# R9b DIESEL support-level mechanistic intercept constants. The R9b base
# reuses the full R4c petrochem_fuels rule byte-for-byte (R3d micro-path
# continuum and detector offset, R4c CH overtone centers / width / gain,
# R4c damping windows / strength, R4c 975 nm short-continuum hump on the
# 750-1550 nm support, R4c additive baseline range, R4c output clip). After
# the R4c absorbance pipeline reaches the R4c non-negative output clip, R9b
# adds a single small fixed mechanistic absorbance intercept on the
# 750-1550 nm DIESEL real basis support; outside the support the readout is
# identically equal to the R4c base.
#
# The intercept value is a PRE-DECLARED MECHANISTIC CONSTANT representing a
# generic detector reference / blank-cell baseline support-level offset in
# absorbance space. Typical near-IR transmission cells with a blank-cell or
# detector dark-current reference exhibit a small residual absorbance
# baseline of order 1e-3 to 5e-3 absorbance units after blank referencing;
# the R9b constant is the lower end of that mechanistic range so that:
#   * It does NOT trigger the R4c non-negative output clip (R4c output is
#     non-negative and the intercept is positive: ``X_support += intercept``
#     with ``intercept > 0`` keeps the lower bound non-negative by
#     construction).
#   * It does NOT smooth derivatives the way the R4a basis profile does:
#     adding a constant to a contiguous block has zero first-derivative
#     inside the block and a single one-bin step at the support boundary,
#     so support-interior derivative structure is mathematically unchanged.
#   * It is independent of any audit measurement: the value is set from
#     general optical reference-cell prior knowledge BEFORE any R9a or R9b
#     mean-shift audit row is computed, and is NOT tuned to close the R9a
#     residual mean-shift gap.
#
# Constants are mechanistic priors only; no real spectra, statistics, PCA,
# covariance, quantiles, ML/DL output, labels, targets, splits, AUC,
# morphology gap scores, calibration, or threshold tuning is consulted.
_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_ABSORBANCE: float = 0.002
_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SUPPORT_NM: tuple[float, float] = (
    750.0,
    1550.0,
)
_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SOURCE: str = "fixed_blank_cell_detector_support_level_intercept_prior"

_R9B_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R4C_DOMAIN_RULES["petrochem_fuels"],
        "support_intercept_absorbance": (_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_ABSORBANCE),
        "support_intercept_support_nm": (_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SUPPORT_NM),
        "support_intercept_source": (_R9B_PETROCHEM_FUELS_SUPPORT_INTERCEPT_SOURCE),
        "spectra_source": ("r9b_diesel_support_intercept_v1_inherits_r4c_balanced_derivative_with_fixed_support_only_blank_cell_detector_intercept_after_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_support_intercept_route_r9b"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp11_dataset_token_diesel_r9b_support_intercept_route"),
    },
}

# R9c DIESEL support-level shape mechanism constants. The base path/baseline/
# clip pipeline is inherited from R3d byte-for-byte; R9c only adds, on the
# 750-1550 nm DIESEL real basis support and AFTER the R3d non-negative output
# clip, a fixed mechanistic shape modulation. Centers are the support-only CH
# overtone bands (1720 nm intentionally excluded) used by R4a/R4b/R4c, but
# with NEW per-band widths sourced from a general liquid-hydrocarbon NIR
# overtone prior (not from any R9a/R9b mean-shift delta). Damping windows,
# damping strength, hump and gain ranges are pre-declared mechanistic
# constants too. R9c does NOT use the R4c balanced-derivative base, the R8a/
# R8b mean-preserving multiplicative modulation, the R9b support intercept,
# any readout-space transform, any residual transfer mechanism, or any scalar
# offset.
_R9C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
# Per-band Gaussian widths (nm). The first two CH-stretch overtones are
# narrower than the longer-wavelength overtones in liquid hydrocarbon NIR
# spectra; the values below come from a general liquid-hydrocarbon NIR prior
# and are NOT chosen from any R9a/R9b residual delta or audit metric.
_R9C_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM: tuple[float, ...] = (
    40.0,
    40.0,
    44.0,
    48.0,
)
_R9C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (0.075, 0.135)
_R9C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM: tuple[tuple[float, float, float], ...] = (
    (1180.0, 56.0, 0.55),
    (1425.0, 72.0, 0.85),
)
_R9C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE: tuple[float, float] = (0.14, 0.28)
_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM: float = 975.0
_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM: float = 84.0
_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE: tuple[float, float] = (
    0.00018,
    0.00048,
)
_R9C_PETROCHEM_FUELS_SUPPORT_NM: tuple[float, float] = (750.0, 1550.0)
_R9C_PETROCHEM_FUELS_CONSTANTS_SOURCE: str = "predeclared_general_liquid_hydrocarbon_nir_prior"

_R9C_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        # R9c support-shape-only modulation keys. They are NOT consumed by the
        # R4a/R4b/R4c CH-overtone/damping/hump branches above (those branches
        # gate on ``damping_windows_nm`` / ``continuum_hump_center_nm`` keys
        # at the rule top level, which R9c does NOT add); they are consumed
        # by the dedicated R9c block at the end of the spectra pipeline.
        "support_shape_centers_nm": _R9C_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM,
        "support_shape_widths_nm": _R9C_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM,
        "support_shape_gain_range": _R9C_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "support_shape_damping_windows_nm": (_R9C_PETROCHEM_FUELS_DAMPING_WINDOWS_NM),
        "support_shape_damping_strength_range": (_R9C_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE),
        "support_shape_hump_center_nm": (_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM),
        "support_shape_hump_width_nm": (_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM),
        "support_shape_hump_amplitude_range": (_R9C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE),
        "support_shape_support_nm": _R9C_PETROCHEM_FUELS_SUPPORT_NM,
        "support_shape_mechanism": ("selective_ch_bandwidth_damping_support_shape_only"),
        "support_shape_constants_source": (_R9C_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "spectra_source": ("r9c_diesel_selective_ch_bandwidth_damping_v1_inherits_r3d_micro_path_with_fixed_support_only_ch_band_damping_and_hump_after_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_support_shape_route_r9c"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp12_dataset_token_diesel_r9c_support_shape_route"),
    },
}

# R9d DIESEL energy-normalized mean-neutral support redistribution constants.
# The base path/baseline/clip pipeline is inherited from R3d byte-for-byte;
# R9d only applies, on the 750-1550 nm DIESEL real basis support and AFTER
# the R3d non-negative output clip, a multiplicative ``exp(strength * shape)``
# modulation where ``shape`` is a fixed mean-neutral max-abs-normalized
# basis. Centers/widths come from a general liquid-hydrocarbon NIR energy
# redistribution prior; they are NOT chosen from any R9a/R9b/R9c mean-shift
# delta and NOT fitted to real spectra. R9d does NOT use the R4c balanced-
# derivative base, the R8a/R8b mean-preserving multiplicative modulation
# (different base, different shape derivation, different normalization), the
# R9b support intercept, the R9c additive support shape, any readout-space
# transform, any residual transfer mechanism, or any scalar offset.
_R9D_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)
_R9D_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM: tuple[float, ...] = (
    40.0,
    40.0,
    44.0,
    48.0,
)
_R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE: tuple[float, float] = (
    0.035,
    0.095,
)
_R9D_PETROCHEM_FUELS_SHAPE_CLIP: tuple[float, float] = (-1.0, 1.0)
_R9D_PETROCHEM_FUELS_RENORM_EPSILON: float = 1e-12
_R9D_PETROCHEM_FUELS_SUPPORT_NM: tuple[float, float] = (750.0, 1550.0)
_R9D_PETROCHEM_FUELS_SHAPE_NORMALIZATION: str = "max_abs"
_R9D_PETROCHEM_FUELS_CONSTANTS_SOURCE: str = "predeclared_general_liquid_hydrocarbon_nir_energy_redistribution_prior"

_R9D_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        # R9d support-redistribution-only modulation keys. They are NOT
        # consumed by the R4a/R4b/R4c CH-overtone/damping/hump branches and
        # are NOT consumed by the R9c support_shape branch (R9c uses keys
        # prefixed ``support_shape_*``); they are consumed by the dedicated
        # R9d block at the end of the spectra pipeline.
        "support_redistribution_centers_nm": (_R9D_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM),
        "support_redistribution_widths_nm": (_R9D_PETROCHEM_FUELS_CH_OVERTONE_WIDTHS_NM),
        "support_redistribution_log_strength_range": (_R9D_PETROCHEM_FUELS_LOG_REDISTRIBUTION_STRENGTH_RANGE),
        "support_redistribution_shape_clip": (_R9D_PETROCHEM_FUELS_SHAPE_CLIP),
        "support_redistribution_renorm_epsilon": (_R9D_PETROCHEM_FUELS_RENORM_EPSILON),
        "support_redistribution_support_nm": (_R9D_PETROCHEM_FUELS_SUPPORT_NM),
        "support_redistribution_shape_normalization": (_R9D_PETROCHEM_FUELS_SHAPE_NORMALIZATION),
        "support_redistribution_mechanism": ("energy_normalized_mean_neutral_support_redistribution"),
        "support_redistribution_constants_source": (_R9D_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "spectra_source": ("r9d_diesel_energy_normalized_support_redistribution_v1_inherits_r3d_micro_path_with_fixed_mean_neutral_max_abs_normalized_support_only_multiplicative_redistribution_after_final_clip"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_support_redistribution_route_r9d"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp13_dataset_token_diesel_r9d_support_redistribution_route"),
    },
}

# R9e DIESEL pathlength/reference attenuation constants. The base
# path/baseline/clip pipeline is inherited from R3d byte-for-byte; R9e only
# applies, on the 750-1550 nm support and AFTER the R3d non-negative output
# clip, a positive row-wise multiplicative attenuation factor. Constants are
# pre-declared generic blank/reference pathlength attenuation priors, not
# fitted to real spectra or audit metrics.
_R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE: tuple[float, float] = (
    0.970,
    0.985,
)
_R9E_PETROCHEM_FUELS_SUPPORT_NM: tuple[float, float] = (750.0, 1550.0)
_R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE: str = "predeclared_generic_blank_reference_pathlength_attenuation_prior"

_R9E_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "support_reference_attenuation_factor_range": (_R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE),
        "support_reference_attenuation_support_nm": (_R9E_PETROCHEM_FUELS_SUPPORT_NM),
        "support_reference_attenuation_constants_source": (_R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "support_reference_attenuation_mechanism": ("positive_pathlength_reference_attenuation_support_only"),
        "spectra_source": ("r9e_diesel_pathlength_reference_attenuation_v1_inherits_r3d_micro_path_with_fixed_support_only_multiplicative_attenuation_after_final_clip"),
        "contrast_source": ("fixed_blank_reference_pathlength_attenuation_prior_explicit_diesel_reference_attenuation_route_r9e"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp15_dataset_token_diesel_r9e_reference_attenuation_route"),
    },
}

_R9F_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE: tuple[float, float] = _R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE
_R9F_PETROCHEM_FUELS_SUPPORT_NM: tuple[float, float] = _R9E_PETROCHEM_FUELS_SUPPORT_NM
_R9F_PETROCHEM_FUELS_CONSTANTS_SOURCE: str = _R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE

_R9F_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "pre_offset_reference_attenuation_factor_range": (_R9F_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE),
        "pre_offset_reference_attenuation_support_nm": (_R9F_PETROCHEM_FUELS_SUPPORT_NM),
        "pre_offset_reference_attenuation_constants_source": (_R9F_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "pre_offset_reference_attenuation_mechanism": ("positive_pathlength_reference_attenuation_continuum_path_component_only"),
        "spectra_source": ("r9f_diesel_pre_offset_pathlength_reference_attenuation_v1_inherits_r3d_micro_path_with_fixed_support_only_multiplicative_attenuation_on_continuum_path_component_before_offset_and_clip"),
        "contrast_source": ("fixed_blank_reference_pathlength_attenuation_prior_explicit_diesel_pre_offset_reference_attenuation_route_r9f"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp16_dataset_token_diesel_r9f_pre_offset_reference_attenuation_route"),
    },
}

_R9H_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM: tuple[float, ...] = (
    1150.0,
    1210.0,
    1390.0,
    1460.0,
)

_R9H_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_centers_nm": (_R9H_PETROCHEM_FUELS_CH_OVERTONE_CENTERS_NM),
        "spectra_source": ("r9h_diesel_support_ch_center_drop1720_isolation_v1_inherits_r3d_micro_path_with_only_support_ch_centers_and_1720_nm_dropped"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_support_ch_center_route_r9h"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp18_dataset_token_diesel_r9h_support_ch_center_route"),
    },
}

_R9I_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM: float = 36.0
_R9I_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE: tuple[float, float] = (
    0.092,
    0.155,
)

_R9I_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_width_nm": _R9I_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R9I_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "spectra_source": ("r9i_diesel_ch_width_gain_isolation_v1_inherits_r3d_micro_path_with_only_ch_width_and_gain_changed"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_prior_explicit_diesel_ch_width_gain_route_r9i"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp19_dataset_token_diesel_r9i_ch_width_gain_route"),
    },
}

_R9J_PETROCHEM_FUELS_DAMPING_WINDOWS_NM: tuple[tuple[float, float, float], ...] = (
    (1180.0, 46.0, 0.60),
    (1425.0, 54.0, 0.70),
)
_R9J_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE: tuple[float, float] = (0.05, 0.15)

_R9J_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "damping_windows_nm": _R9J_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R9J_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "spectra_source": ("r9j_diesel_residual_damping_isolation_v1_inherits_r3d_micro_path_with_only_residual_damping_changed"),
        "contrast_source": ("fixed_hydrocarbon_residual_damping_prior_explicit_diesel_residual_damping_route_r9j"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp20_dataset_token_diesel_r9j_residual_damping_route"),
    },
}

_R9K_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "continuum_hump_center_nm": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_CENTER_NM),
        "continuum_hump_width_nm": _R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_WIDTH_NM,
        "continuum_hump_amplitude_range": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_AMPLITUDE_RANGE),
        "continuum_hump_support_nm": (_R4C_PETROCHEM_FUELS_CONTINUUM_HUMP_SUPPORT_NM),
        "spectra_source": ("r9k_diesel_continuum_hump_isolation_v1_inherits_r3d_micro_path_with_only_continuum_hump_changed"),
        "contrast_source": ("fixed_hydrocarbon_continuum_hump_prior_explicit_diesel_continuum_hump_route_r9k"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp21_dataset_token_diesel_r9k_continuum_hump_route"),
    },
}

_R9L_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "damping_windows_nm": _R9J_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R9J_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "support_reference_attenuation_factor_range": (_R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE),
        "support_reference_attenuation_support_nm": (_R9E_PETROCHEM_FUELS_SUPPORT_NM),
        "support_reference_attenuation_constants_source": (_R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "support_reference_attenuation_mechanism": ("positive_pathlength_reference_attenuation_support_only"),
        "spectra_source": ("r9l_diesel_residual_damping_clean_attenuation_v1_inherits_r3d_micro_path_with_r9j_residual_damping_and_r9e_clean_support_attenuation"),
        "contrast_source": ("fixed_hydrocarbon_residual_damping_prior_plus_blank_reference_pathlength_attenuation_prior_explicit_diesel_r9l_route"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp22_dataset_token_diesel_r9l_residual_damping_clean_attenuation_route"),
    },
}

_R9M_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "ch_overtone_width_nm": _R9I_PETROCHEM_FUELS_CH_OVERTONE_WIDTH_NM,
        "ch_overtone_gain_range": _R9I_PETROCHEM_FUELS_CH_OVERTONE_GAIN_RANGE,
        "damping_windows_nm": _R9J_PETROCHEM_FUELS_DAMPING_WINDOWS_NM,
        "damping_strength_range": _R9J_PETROCHEM_FUELS_DAMPING_STRENGTH_RANGE,
        "support_reference_attenuation_factor_range": (_R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE),
        "support_reference_attenuation_support_nm": (_R9E_PETROCHEM_FUELS_SUPPORT_NM),
        "support_reference_attenuation_constants_source": (_R9E_PETROCHEM_FUELS_CONSTANTS_SOURCE),
        "support_reference_attenuation_mechanism": ("positive_pathlength_reference_attenuation_support_only"),
        "spectra_source": ("r9m_diesel_width_gain_damping_clean_attenuation_v1_inherits_r3d_micro_path_with_r9i_width_gain_r9j_residual_damping_and_r9e_clean_support_attenuation"),
        "contrast_source": ("fixed_hydrocarbon_ch_overtone_width_gain_prior_plus_residual_damping_prior_plus_blank_reference_pathlength_attenuation_prior_explicit_diesel_r9m_route"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp23_dataset_token_diesel_r9m_width_gain_damping_clean_attenuation_route"),
    },
}

# P2a row-level pathlength/reference constants. The factor range deliberately
# reuses the pre-declared R9e generic blank/reference attenuation prior so P2a
# tests only the row-level application semantics, not a retuned amplitude.
_P2A_PETROCHEM_FUELS_ROW_REFERENCE_FACTOR_RANGE: tuple[float, float] = (
    _R9E_PETROCHEM_FUELS_REFERENCE_ATTENUATION_FACTOR_RANGE
)
_P2A_PETROCHEM_FUELS_CONSTANTS_SOURCE: str = (
    "predeclared_generic_blank_reference_row_pathlength_distribution_prior"
)

_P2A_DOMAIN_RULES: dict[str, dict[str, Any]] = {
    **_R3D_DOMAIN_RULES,
    "petrochem_fuels": {
        **_R3D_DOMAIN_RULES["petrochem_fuels"],
        "row_pathlength_reference_factor_range": (
            _P2A_PETROCHEM_FUELS_ROW_REFERENCE_FACTOR_RANGE
        ),
        "row_pathlength_reference_constants_source": (
            _P2A_PETROCHEM_FUELS_CONSTANTS_SOURCE
        ),
        "row_pathlength_reference_mechanism": (
            "positive_row_level_pathlength_reference_attenuation_full_wavelength_row"
        ),
        "spectra_source": ("p2a_diesel_row_pathlength_reference_v1_inherits_r3d_micro_path_with_fixed_full_row_multiplicative_reference_attenuation_after_final_clip"),
        "contrast_source": ("fixed_blank_reference_row_pathlength_prior_explicit_diesel_p2a_route"),
        "calibration_source": "none",
        "real_stat_source": "none",
        "threshold_source": "none",
        "provenance_source": ("exp25_dataset_token_diesel_p2a_row_pathlength_reference_route"),
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
    "r2x_sentinel_matrix_v1": _R2X_DOMAIN_RULES,
    "r2y_sentinel_matrix_v1": _R2Y_DOMAIN_RULES,
    "r2z_sentinel_matrix_v1": _R2Z_DOMAIN_RULES,
    "r3a_corn_matrix_v1": _R3A_DOMAIN_RULES,
    "r3b_corn_matrix_v1": _R3B_DOMAIN_RULES,
    "r3c_diesel_matrix_v1": _R3C_DOMAIN_RULES,
    "r3d_diesel_matrix_v1": _R3D_DOMAIN_RULES,
    "r3e_diesel_matrix_v1": _R3E_DOMAIN_RULES,
    "r3f_diesel_matrix_v1": _R3F_DOMAIN_RULES,
    "r3g_diesel_matrix_v1": _R3G_DOMAIN_RULES,
    "r4a_diesel_basis_v1": _R4A_DOMAIN_RULES,
    "r4b_diesel_derivative_restore_v1": _R4B_DOMAIN_RULES,
    "r4c_diesel_balanced_derivative_v1": _R4C_DOMAIN_RULES,
    "r5a_diesel_absorbance_readout_v1": _R5A_DOMAIN_RULES,
    "r5b_diesel_transmittance_readout_v1": _R5B_DOMAIN_RULES,
    "r5c_diesel_blank_referenced_intensity_v1": _R5C_DOMAIN_RULES,
    "r6a_diesel_centered_hydrocarbon_shape_v1": _R6A_DOMAIN_RULES,
    "r7a_diesel_support_centered_residual_transfer_v1": _R7A_DOMAIN_RULES,
    "r8a_diesel_mean_preserving_micro_path_modulation_v1": _R8A_DOMAIN_RULES,
    "r8b_diesel_r4c_base_mean_preserving_micro_path_modulation_v1": (_R8B_DOMAIN_RULES),
    "r9b_diesel_support_intercept_v1": _R9B_DOMAIN_RULES,
    "r9c_diesel_selective_ch_bandwidth_damping_v1": _R9C_DOMAIN_RULES,
    "r9d_diesel_energy_normalized_support_redistribution_v1": (_R9D_DOMAIN_RULES),
    "r9e_diesel_pathlength_reference_attenuation_v1": _R9E_DOMAIN_RULES,
    "r9f_diesel_pre_offset_pathlength_reference_attenuation_v1": (_R9F_DOMAIN_RULES),
    "r9h_diesel_support_ch_center_drop1720_isolation_v1": _R9H_DOMAIN_RULES,
    "r9i_diesel_ch_width_gain_isolation_v1": _R9I_DOMAIN_RULES,
    "r9j_diesel_residual_damping_isolation_v1": _R9J_DOMAIN_RULES,
    "r9k_diesel_continuum_hump_isolation_v1": _R9K_DOMAIN_RULES,
    "r9l_diesel_residual_damping_clean_attenuation_v1": _R9L_DOMAIN_RULES,
    "r9m_diesel_width_gain_damping_clean_attenuation_v1": _R9M_DOMAIN_RULES,
    "p2a_diesel_row_pathlength_reference_v1": _P2A_DOMAIN_RULES,
}


def _validate_r2c_profile(profile: str) -> None:
    if profile not in ALL_REMEDIATION_PROFILES:
        raise ValueError(f"unknown remediation profile {profile!r}; valid profiles are {list(ALL_REMEDIATION_PROFILES)}")


def _is_r2l_lucas_soil_record(record: PriorConfigRecord) -> bool:
    route = record.source_prior_config.get("_r2l_lucas_soil_route")
    if not isinstance(route, dict):
        return False
    return route.get("enabled") is True and route.get("route_marker") == "lucas" and route.get("non_oracle") is True and route.get("real_stat_capture") is False


def _r2m_milk_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r2m_milk_readout_route")
    if not isinstance(route, dict):
        return None
    if route.get("enabled") is True and route.get("route_marker") == "milk" and route.get("non_oracle") is True and route.get("real_stat_capture") is False:
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


def _r3c_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r3c_diesel_readout_route")
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


def _is_r3c_diesel_record(record: PriorConfigRecord) -> bool:
    return _r3c_diesel_route(record) is not None


def _r3d_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r3d_diesel_readout_route")
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


def _is_r3d_diesel_record(record: PriorConfigRecord) -> bool:
    return _r3d_diesel_route(record) is not None


def _r3e_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r3e_diesel_readout_route")
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


def _is_r3e_diesel_record(record: PriorConfigRecord) -> bool:
    return _r3e_diesel_route(record) is not None


def _r3f_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r3f_diesel_readout_route")
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


def _is_r3f_diesel_record(record: PriorConfigRecord) -> bool:
    return _r3f_diesel_route(record) is not None


def _r3g_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r3g_diesel_readout_route")
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


def _is_r3g_diesel_record(record: PriorConfigRecord) -> bool:
    return _r3g_diesel_route(record) is not None


def _r4a_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r4a_diesel_readout_route")
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


def _is_r4a_diesel_record(record: PriorConfigRecord) -> bool:
    return _r4a_diesel_route(record) is not None


def _r4b_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r4b_diesel_readout_route")
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


def _is_r4b_diesel_record(record: PriorConfigRecord) -> bool:
    return _r4b_diesel_route(record) is not None


def _r4c_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r4c_diesel_readout_route")
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


def _is_r4c_diesel_record(record: PriorConfigRecord) -> bool:
    return _r4c_diesel_route(record) is not None


def _r5a_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r5a_diesel_readout_route")
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


def _is_r5a_diesel_record(record: PriorConfigRecord) -> bool:
    return _r5a_diesel_route(record) is not None


def _r5b_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r5b_diesel_readout_route")
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


def _is_r5b_diesel_record(record: PriorConfigRecord) -> bool:
    return _r5b_diesel_route(record) is not None


def _r5c_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r5c_diesel_readout_route")
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


def _is_r5c_diesel_record(record: PriorConfigRecord) -> bool:
    return _r5c_diesel_route(record) is not None


def _r6a_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r6a_diesel_shape_route")
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


def _is_r6a_diesel_record(record: PriorConfigRecord) -> bool:
    return _r6a_diesel_route(record) is not None


def _r7a_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r7a_diesel_residual_route")
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


def _is_r7a_diesel_record(record: PriorConfigRecord) -> bool:
    return _r7a_diesel_route(record) is not None


def _r8a_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r8a_diesel_micro_path_route")
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


def _is_r8a_diesel_record(record: PriorConfigRecord) -> bool:
    return _r8a_diesel_route(record) is not None


def _r8b_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r8b_diesel_micro_path_route")
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


def _is_r8b_diesel_record(record: PriorConfigRecord) -> bool:
    return _r8b_diesel_route(record) is not None


def _r9b_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9b_diesel_support_intercept_route")
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


def _is_r9b_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9b_diesel_route(record) is not None


def _r9c_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9c_diesel_support_shape_route")
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


def _is_r9c_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9c_diesel_route(record) is not None


def _r9d_diesel_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9d_diesel_support_redistribution_route")
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


def _is_r9d_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9d_diesel_route(record) is not None


def _r9e_diesel_reference_attenuation_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9e_diesel_reference_attenuation_route")
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


def _is_r9e_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9e_diesel_reference_attenuation_route(record) is not None


def _r9f_diesel_pre_offset_reference_attenuation_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9f_diesel_pre_offset_reference_attenuation_route")
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


def _is_r9f_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9f_diesel_pre_offset_reference_attenuation_route(record) is not None


def _r9h_diesel_support_ch_center_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9h_diesel_support_ch_center_route")
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


def _is_r9h_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9h_diesel_support_ch_center_route(record) is not None


def _r9i_diesel_ch_width_gain_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9i_diesel_ch_width_gain_route")
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


def _is_r9i_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9i_diesel_ch_width_gain_route(record) is not None


def _r9j_diesel_residual_damping_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9j_diesel_residual_damping_route")
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


def _is_r9j_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9j_diesel_residual_damping_route(record) is not None


def _r9k_diesel_continuum_hump_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9k_diesel_continuum_hump_route")
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


def _is_r9k_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9k_diesel_continuum_hump_route(record) is not None


def _r9l_diesel_residual_damping_clean_attenuation_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9l_diesel_residual_damping_clean_attenuation_route")
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


def _is_r9l_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9l_diesel_residual_damping_clean_attenuation_route(record) is not None


def _r9m_diesel_width_gain_damping_clean_attenuation_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_r9m_diesel_width_gain_damping_clean_attenuation_route")
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


def _is_r9m_diesel_record(record: PriorConfigRecord) -> bool:
    return _r9m_diesel_width_gain_damping_clean_attenuation_route(record) is not None


def _p2a_diesel_row_pathlength_reference_route(
    record: PriorConfigRecord,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get("_p2a_diesel_row_pathlength_reference_route")
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


def _is_p2a_diesel_record(record: PriorConfigRecord) -> bool:
    return _p2a_diesel_row_pathlength_reference_route(record) is not None


def _seed_source_profile(profile: str) -> str:
    """Resolve the profile id used to derive deterministic RNG seeds.

    R5 profiles inherit the entire R4c absorbance pipeline; they only swap the
    final readout space. To guarantee that R5a is byte-identical to R4c on the
    same seed, and that R5b/R5c apply the readout transform on top of the same
    intermediate absorbance draw, every R5 profile delegates seed derivation to
    the R4c profile id.

    R6a inherits the full R4c absorbance pipeline byte-for-byte and only adds
    a small fixed zero-mean hydrocarbon shape envelope on top, so it must use
    the same RNG seed source as R4c/R5a so that target draws are identical and
    the R4c portion of the spectra RNG sequence stays aligned with R4c/R5a.

    R8a inherits the full R4a absorbance pipeline byte-for-byte (including the
    standard R4a non-negative final clip) and then applies a mean-preserving
    multiplicative modulation on top, so it must use the same RNG seed source
    as R4a so the R4a portion of the spectra RNG sequence (path/contrast/CH/
    damping/hump/offset draws) stays byte-identical to R4a. The R8a-specific
    modulation strength draw is consumed AFTER the R4a portion, leaving R4a
    byte-aligned with R8a's pre-modulation base.

    R8b has the same modulation step as R8a but inherits the full R4c
    balanced-derivative absorbance pipeline as its pre-modulation base, so it
    delegates seed derivation to R4c.

    R9b inherits the full R4c balanced-derivative absorbance pipeline as its
    pre-intercept base and only adds a fixed mechanistic absorbance constant
    on the support after the R4c output clip. The intercept is a single
    pre-declared constant that consumes no RNG draw, so delegating to R4c
    keeps the R4c portion of the spectra RNG sequence byte-identical to R4c
    (every R4c per-row draw is reproduced before the R9b additive step).

    R9c inherits the full R3d micro-path / baseline / CH-overtone / clip
    pipeline as its pre-shape base (NOT the R4c balanced-derivative base),
    and adds support-only Gaussian band sums, support-localized damping
    windows, and a small support hump after the R3d non-negative output
    clip. Delegating seed derivation to R3d keeps the R3d portion of the
    spectra RNG sequence byte-identical to R3d on the same seed (every R3d
    per-row draw is reproduced before the R9c additive shape draws).

    R9d inherits the full R3d micro-path / baseline / CH-overtone / clip
    pipeline as its pre-redistribution base (NOT the R4c balanced-derivative
    base) and applies a multiplicative ``exp(strength * shape)`` energy-
    normalized mean-neutral support-only redistribution on top, after which
    each row support is multiplicatively renormalized to preserve the
    pre-redistribution support mean. Delegating seed derivation to R3d
    keeps the R3d portion of the spectra RNG sequence byte-identical to
    R3d on the same seed (every R3d per-row draw is reproduced before the
    R9d per-row strength draw is consumed).

    R9e inherits the same full R3d pipeline and only applies a positive
    support-only attenuation factor after the R3d output clip. Delegating seed
    derivation to R3d keeps the R3d portion byte-identical before the R9e
    factor draw is consumed.

    R9i inherits the full R3d pipeline and changes only the scalar CH overtone
    width and gain range. Delegating seed derivation to R3d preserves the R3d
    draw order and changes only that deterministic width/gain transformation.

    R9j inherits the full R3d pipeline and changes only residual damping
    windows / strength. Delegating seed derivation to R3d preserves the R3d
    draw order before the damping strength draw is consumed.

    R9k inherits the full R3d pipeline and changes only the 975 nm continuum
    hump. Delegating seed derivation to R3d preserves the R3d draw order; the
    R9k hump amplitude draw uses a dedicated profile seed so later R3d draws
    stay aligned.

    R9l inherits the full R3d pipeline and combines only the R9j residual
    damping constants with the R9e support-only attenuation. Delegating seed
    derivation to R3d preserves the R3d draw order; the damping draw uses the
    existing R9j dedicated stream and the attenuation draw uses the same stage
    as R9e.

    R9m inherits the full R3d pipeline, changes only to the R9i width/gain
    constants, and combines the same R9j damping stream with the same R9e clean
    attenuation stage. Delegating seed derivation to R3d keeps it a controlled
    Palier 1 diagnostic combination on top of R3d.
    """
    if profile in (R5A_REMEDIATION_PROFILES + R5B_REMEDIATION_PROFILES + R5C_REMEDIATION_PROFILES + R6A_REMEDIATION_PROFILES):
        return "r4c_diesel_balanced_derivative_v1"
    if profile in R8A_REMEDIATION_PROFILES:
        return "r4a_diesel_basis_v1"
    if profile in R8B_REMEDIATION_PROFILES:
        return "r4c_diesel_balanced_derivative_v1"
    if profile in R9B_REMEDIATION_PROFILES:
        return "r4c_diesel_balanced_derivative_v1"
    if profile in R9C_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9D_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9E_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9F_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9H_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9I_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9J_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9K_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9L_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in R9M_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    if profile in P2A_REMEDIATION_PROFILES:
        return "r3d_diesel_matrix_v1"
    return profile


def _r3a_corn_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    return _corn_readout_route(record, "_r3a_corn_readout_route")


def _r3b_corn_route(record: PriorConfigRecord) -> dict[str, Any] | None:
    return _corn_readout_route(record, "_r3b_corn_readout_route")


def _corn_readout_route(
    record: PriorConfigRecord,
    route_key: str,
) -> dict[str, Any] | None:
    route = record.source_prior_config.get(route_key)
    if not isinstance(route, dict):
        return None
    if (
        route.get("enabled") is True
        and route.get("route_marker") == "corn"
        and route.get("non_oracle") is True
        and route.get("no_target_or_label") is True
        and route.get("real_stat_capture") is False
        and route.get("thresholds_modified") is False
    ):
        return route
    return None


def _is_r3a_corn_record(record: PriorConfigRecord) -> bool:
    return _r3a_corn_route(record) is not None


def _is_r3b_corn_record(record: PriorConfigRecord) -> bool:
    return _r3b_corn_route(record) is not None


def _effective_builder_remediation_profile(
    profile: str,
    record: PriorConfigRecord,
) -> str:
    _validate_r2c_profile(profile)
    domain_key = record.domain_key
    if profile in P2A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_p2a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9M_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9m_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9L_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9l_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9K_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9k_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9J_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9j_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9I_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9i_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9H_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9h_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9F_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9f_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9E_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9e_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9D_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9d_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9C_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9c_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R9B_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r9b_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R8B_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r8b_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R8A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r8a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R7A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r7a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R6A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r6a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R5C_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r5c_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R5B_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r5b_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R5A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r5a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R4C_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r4c_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R4B_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r4b_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R4A_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r4a_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R3G_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r3g_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3f_diesel_matrix_v1",
            record,
        )
    if profile in R3F_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r3f_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3e_diesel_matrix_v1",
            record,
        )
    if profile in R3E_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r3e_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3d_diesel_matrix_v1",
            record,
        )
    if profile in R3D_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r3d_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3c_diesel_matrix_v1",
            record,
        )
    if profile in R3C_REMEDIATION_PROFILES:
        if domain_key == "petrochem_fuels" and _is_r3c_diesel_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r3b_corn_matrix_v1",
            record,
        )
    if profile in R3B_REMEDIATION_PROFILES:
        if domain_key == "agriculture_grain" and _is_r3b_corn_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2w_sentinel_matrix_v1",
            record,
        )
    if profile in R3A_REMEDIATION_PROFILES:
        if domain_key == "agriculture_grain" and _is_r3a_corn_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2w_sentinel_matrix_v1",
            record,
        )
    if profile in R2Z_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2Y_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
    if profile in R2X_REMEDIATION_PROFILES:
        if domain_key == "environmental_soil" and _is_r2n_manure21_record(record):
            return profile
        return _effective_builder_remediation_profile(
            "r2s_sentinel_matrix_v1",
            record,
        )
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
        if domain_key == "environmental_soil" and _is_r2q_lucas_ph_organic_record(record):
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
    if profile in R2J_REMEDIATION_PROFILES + R2K_REMEDIATION_PROFILES and domain_key != "petrochem_fuels":
        return "r2i_sentinel_matrix_v1"
    return profile


def _audit_base_for(profile: str) -> dict[str, Any]:
    if profile in R9M_REMEDIATION_PROFILES:
        return _R9M_AUDIT_BASE
    if profile in P2A_REMEDIATION_PROFILES:
        return _P2A_AUDIT_BASE
    if profile in R9L_REMEDIATION_PROFILES:
        return _R9L_AUDIT_BASE
    if profile in R9K_REMEDIATION_PROFILES:
        return _R9K_AUDIT_BASE
    if profile in R9J_REMEDIATION_PROFILES:
        return _R9J_AUDIT_BASE
    if profile in R9I_REMEDIATION_PROFILES:
        return _R9I_AUDIT_BASE
    if profile in R9H_REMEDIATION_PROFILES:
        return _R9H_AUDIT_BASE
    if profile in R9F_REMEDIATION_PROFILES:
        return _R9F_AUDIT_BASE
    if profile in R9E_REMEDIATION_PROFILES:
        return _R9E_AUDIT_BASE
    if profile in R9D_REMEDIATION_PROFILES:
        return _R9D_AUDIT_BASE
    if profile in R9C_REMEDIATION_PROFILES:
        return _R9C_AUDIT_BASE
    if profile in R9B_REMEDIATION_PROFILES:
        return _R9B_AUDIT_BASE
    if profile in R8B_REMEDIATION_PROFILES:
        return _R8B_AUDIT_BASE
    if profile in R8A_REMEDIATION_PROFILES:
        return _R8A_AUDIT_BASE
    if profile in R7A_REMEDIATION_PROFILES:
        return _R7A_AUDIT_BASE
    if profile in R6A_REMEDIATION_PROFILES:
        return _R6A_AUDIT_BASE
    if profile in R5C_REMEDIATION_PROFILES:
        return _R5C_AUDIT_BASE
    if profile in R5B_REMEDIATION_PROFILES:
        return _R5B_AUDIT_BASE
    if profile in R5A_REMEDIATION_PROFILES:
        return _R5A_AUDIT_BASE
    if profile in R4C_REMEDIATION_PROFILES:
        return _R4C_AUDIT_BASE
    if profile in R4B_REMEDIATION_PROFILES:
        return _R4B_AUDIT_BASE
    if profile in R4A_REMEDIATION_PROFILES:
        return _R4A_AUDIT_BASE
    if profile in R3G_REMEDIATION_PROFILES:
        return _R3G_AUDIT_BASE
    if profile in R3F_REMEDIATION_PROFILES:
        return _R3F_AUDIT_BASE
    if profile in R3E_REMEDIATION_PROFILES:
        return _R3E_AUDIT_BASE
    if profile in R3D_REMEDIATION_PROFILES:
        return _R3D_AUDIT_BASE
    if profile in R3C_REMEDIATION_PROFILES:
        return _R3C_AUDIT_BASE
    if profile in R3B_REMEDIATION_PROFILES:
        return _R3B_AUDIT_BASE
    if profile in R3A_REMEDIATION_PROFILES:
        return _R3A_AUDIT_BASE
    if profile in R2Z_REMEDIATION_PROFILES:
        return _R2Z_AUDIT_BASE
    if profile in R2Y_REMEDIATION_PROFILES:
        return _R2Y_AUDIT_BASE
    if profile in R2X_REMEDIATION_PROFILES:
        return _R2X_AUDIT_BASE
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
    profile_seed = _profile_seed(f"r2c_concentrations:{_seed_source_profile(profile)}", seed)
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
        audit_base["reason"] = f"domain {domain_key!r} has no remediation rule under profile {profile!r}; concentrations unchanged"
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

    spectra_seed = _profile_seed(f"r2c_spectra:{_seed_source_profile(profile)}", seed)
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
        transform_params.update(
            {
                "smoothing_fwhm_nm": float(smoothing_fwhm_nm),
                "smoothing_sigma_bins": float(sigma_bins),
                "smoothing_kernel_size": int(kernel.size),
                "smoothing_median_step_nm": float(median_step),
            }
        )
    else:
        X_smoothed = X_in

    if rule.get("spectra_rule") == "strawberry_puree_transflectance_residual_readout":
        route = _r2r_fruitpuree_route(record)
        if route is None:
            msg = "R2r FruitPuree readout requires explicit bench-only FruitPuree route provenance; route was missing or non-compliant"
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
        X_out = baselines[:, None] + slopes[:, None] * wl_norm[None, :] + water_bands[:, None] * water_profile[None, :] + sugar_bands[:, None] * sugar_profile[None, :] - residual_scales[:, None] * row_centered
        clip_low, clip_high = rule["output_clip_absorbance"]
        X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update(
            {
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
                "fruitpuree_readout_route_marker": str(route.get("route_marker", "unknown")),
                "fruitpuree_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "fruitpuree_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "fruitpuree_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                "output_clip_absorbance": [float(clip_low), float(clip_high)],
            }
        )
    elif rule.get("spectra_rule") == "micro_path_fuel_ch_overtone_contrast_readout":
        route = None
        if profile in R2S_REMEDIATION_PROFILES:
            route = _r2s_diesel_route(record)
        elif profile in R3C_REMEDIATION_PROFILES:
            route = _r3c_diesel_route(record)
        elif profile in R3D_REMEDIATION_PROFILES:
            route = _r3d_diesel_route(record)
        elif profile in R3E_REMEDIATION_PROFILES:
            route = _r3e_diesel_route(record)
        elif profile in R3F_REMEDIATION_PROFILES:
            route = _r3f_diesel_route(record)
        elif profile in R3G_REMEDIATION_PROFILES:
            route = _r3g_diesel_route(record)
        elif profile in R4C_REMEDIATION_PROFILES:
            route = _r4c_diesel_route(record)
        elif profile in R4B_REMEDIATION_PROFILES:
            route = _r4b_diesel_route(record)
        elif profile in R4A_REMEDIATION_PROFILES:
            route = _r4a_diesel_route(record)
        elif profile in R5A_REMEDIATION_PROFILES:
            route = _r5a_diesel_route(record)
        elif profile in R5B_REMEDIATION_PROFILES:
            route = _r5b_diesel_route(record)
        elif profile in R5C_REMEDIATION_PROFILES:
            route = _r5c_diesel_route(record)
        elif profile in R6A_REMEDIATION_PROFILES:
            route = _r6a_diesel_route(record)
        elif profile in R7A_REMEDIATION_PROFILES:
            route = _r7a_diesel_route(record)
        elif profile in R8A_REMEDIATION_PROFILES:
            route = _r8a_diesel_route(record)
        elif profile in R8B_REMEDIATION_PROFILES:
            route = _r8b_diesel_route(record)
        elif profile in R9B_REMEDIATION_PROFILES:
            route = _r9b_diesel_route(record)
        elif profile in R9C_REMEDIATION_PROFILES:
            route = _r9c_diesel_route(record)
        elif profile in R9D_REMEDIATION_PROFILES:
            route = _r9d_diesel_route(record)
        elif profile in R9E_REMEDIATION_PROFILES:
            route = _r9e_diesel_reference_attenuation_route(record)
        elif profile in R9F_REMEDIATION_PROFILES:
            route = _r9f_diesel_pre_offset_reference_attenuation_route(record)
        elif profile in R9H_REMEDIATION_PROFILES:
            route = _r9h_diesel_support_ch_center_route(record)
        elif profile in R9I_REMEDIATION_PROFILES:
            route = _r9i_diesel_ch_width_gain_route(record)
        elif profile in R9J_REMEDIATION_PROFILES:
            route = _r9j_diesel_residual_damping_route(record)
        elif profile in R9K_REMEDIATION_PROFILES:
            route = _r9k_diesel_continuum_hump_route(record)
        elif profile in R9L_REMEDIATION_PROFILES:
            route = _r9l_diesel_residual_damping_clean_attenuation_route(record)
        elif profile in R9M_REMEDIATION_PROFILES:
            route = _r9m_diesel_width_gain_damping_clean_attenuation_route(record)
        elif profile in P2A_REMEDIATION_PROFILES:
            route = _p2a_diesel_row_pathlength_reference_route(record)
        if (
            profile
            in R2S_REMEDIATION_PROFILES
            + R3C_REMEDIATION_PROFILES
            + R3D_REMEDIATION_PROFILES
            + R3E_REMEDIATION_PROFILES
            + R3F_REMEDIATION_PROFILES
            + R3G_REMEDIATION_PROFILES
            + R4A_REMEDIATION_PROFILES
            + R4B_REMEDIATION_PROFILES
            + R4C_REMEDIATION_PROFILES
            + R5A_REMEDIATION_PROFILES
            + R5B_REMEDIATION_PROFILES
            + R5C_REMEDIATION_PROFILES
            + R6A_REMEDIATION_PROFILES
            + R7A_REMEDIATION_PROFILES
            + R8A_REMEDIATION_PROFILES
            + R8B_REMEDIATION_PROFILES
            + R9B_REMEDIATION_PROFILES
            + R9C_REMEDIATION_PROFILES
            + R9D_REMEDIATION_PROFILES
            + R9E_REMEDIATION_PROFILES
            + R9F_REMEDIATION_PROFILES
            + R9H_REMEDIATION_PROFILES
            + R9I_REMEDIATION_PROFILES
            + R9J_REMEDIATION_PROFILES
            + R9K_REMEDIATION_PROFILES
            + R9L_REMEDIATION_PROFILES
            + R9M_REMEDIATION_PROFILES
            + P2A_REMEDIATION_PROFILES
            and route is None
        ):
            if profile in R6A_REMEDIATION_PROFILES:
                msg = "DIESEL shape remediation requires explicit bench-only DIESEL shape-route provenance; route was missing or non-compliant"
            elif profile in R7A_REMEDIATION_PROFILES:
                msg = "DIESEL residual transfer remediation requires explicit bench-only DIESEL residual-route provenance; route was missing or non-compliant"
            elif profile in R8A_REMEDIATION_PROFILES + R8B_REMEDIATION_PROFILES:
                msg = "DIESEL micro-path modulation remediation requires explicit bench-only DIESEL micro-path modulation route provenance; route was missing or non-compliant"
            elif profile in R9B_REMEDIATION_PROFILES:
                msg = "DIESEL support intercept remediation requires explicit bench-only DIESEL support intercept route provenance; route was missing or non-compliant"
            elif profile in R9C_REMEDIATION_PROFILES:
                msg = "DIESEL support shape remediation requires explicit bench-only DIESEL support shape route provenance; route was missing or non-compliant"
            elif profile in R9D_REMEDIATION_PROFILES:
                msg = "DIESEL support redistribution remediation requires explicit bench-only DIESEL support redistribution route provenance; route was missing or non-compliant"
            elif profile in R9E_REMEDIATION_PROFILES:
                msg = "DIESEL reference attenuation remediation requires explicit bench-only DIESEL reference attenuation route provenance; route was missing or non-compliant"
            elif profile in R9F_REMEDIATION_PROFILES:
                msg = "DIESEL pre-offset reference attenuation remediation requires explicit bench-only DIESEL pre-offset reference attenuation route provenance; route was missing or non-compliant"
            elif profile in R9H_REMEDIATION_PROFILES:
                msg = "DIESEL support CH-center/drop-1720 isolation remediation requires explicit bench-only DIESEL support CH-center route provenance; route was missing or non-compliant"
            elif profile in R9I_REMEDIATION_PROFILES:
                msg = "DIESEL CH width/gain isolation remediation requires explicit bench-only DIESEL CH width/gain route provenance; route was missing or non-compliant"
            elif profile in R9J_REMEDIATION_PROFILES:
                msg = "DIESEL residual damping isolation remediation requires explicit bench-only DIESEL residual damping route provenance; route was missing or non-compliant"
            elif profile in R9K_REMEDIATION_PROFILES:
                msg = "DIESEL continuum hump isolation remediation requires explicit bench-only DIESEL continuum hump route provenance; route was missing or non-compliant"
            elif profile in R9L_REMEDIATION_PROFILES:
                msg = "DIESEL residual damping clean attenuation remediation requires explicit bench-only DIESEL R9l route provenance; route was missing or non-compliant"
            elif profile in R9M_REMEDIATION_PROFILES:
                msg = "DIESEL width/gain damping clean attenuation remediation requires explicit bench-only DIESEL R9m route provenance; route was missing or non-compliant"
            elif profile in P2A_REMEDIATION_PROFILES:
                msg = "DIESEL row pathlength/reference remediation requires explicit bench-only DIESEL P2a route provenance; route was missing or non-compliant"
            else:
                msg = "DIESEL readout requires explicit bench-only DIESEL route provenance; route was missing or non-compliant"
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

        continuum_path_component = continuum * path_factors[:, None] * path_profile
        feature_residual_component = feature_residual * feature_contrasts[:, None]
        X_out = continuum_path_component + feature_residual_component
        if "fixed_envelope_absorbance_range" in rule:
            envelope_centers = np.asarray(
                rule["fixed_envelope_centers_nm"],
                dtype=float,
            )
            envelope_widths = np.asarray(
                rule["fixed_envelope_widths_nm"],
                dtype=float,
            )
            envelope_weights = np.asarray(
                rule["fixed_envelope_weights"],
                dtype=float,
            )
            fixed_envelope = np.zeros_like(wl, dtype=float)
            for center, width, weight in zip(
                envelope_centers,
                envelope_widths,
                envelope_weights,
                strict=True,
            ):
                fixed_envelope += weight * np.exp(-0.5 * ((wl - center) / width) ** 2)
            if fixed_envelope.size and float(fixed_envelope.max()) > 0.0:
                fixed_envelope = fixed_envelope / float(fixed_envelope.max())
                fixed_envelope = fixed_envelope - float(fixed_envelope.mean())
            env_low, env_high = rule["fixed_envelope_absorbance_range"]
            envelope_amplitudes = np.asarray(
                rng.uniform(env_low, env_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            if wl.size > 1:
                wl_span = max(float(wl.max() - wl.min()), 1.0)
                centered_wl = (wl - float(wl.mean())) / wl_span
            else:
                centered_wl = np.zeros_like(wl, dtype=float)
            slope_low, slope_high = rule["continuum_slope_absorbance_range"]
            continuum_slopes = np.asarray(
                rng.uniform(slope_low, slope_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            X_out = X_out + envelope_amplitudes[:, None] * fixed_envelope[None, :] + continuum_slopes[:, None] * centered_wl[None, :]
        if "damping_windows_nm" in rule:
            damp_low, damp_high = rule["damping_strength_range"]
            damping_rng = (
                np.random.default_rng(
                    _profile_seed(
                        "r2c_residual_damping:r9j_diesel_residual_damping_isolation_v1",
                        seed,
                    )
                )
                if profile in (R9J_REMEDIATION_PROFILES + R9L_REMEDIATION_PROFILES + R9M_REMEDIATION_PROFILES)
                else rng
            )
            damping_strengths = np.asarray(
                damping_rng.uniform(damp_low, damp_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            damping_profile = np.zeros_like(wl, dtype=float)
            for damp_center, damp_width, damp_weight in rule["damping_windows_nm"]:
                damping_profile = np.maximum(
                    damping_profile,
                    float(damp_weight) * np.exp(-0.5 * ((wl - float(damp_center)) / float(damp_width)) ** 2),
                )
            X_out = X_out - (feature_residual * feature_contrasts[:, None] * damping_strengths[:, None] * damping_profile[None, :])
        if "continuum_hump_center_nm" in rule:
            hump_low, hump_high = rule["continuum_hump_amplitude_range"]
            hump_rng = (
                np.random.default_rng(
                    _profile_seed(
                        "r2c_continuum_hump:r9k_diesel_continuum_hump_isolation_v1",
                        seed,
                    )
                )
                if profile in R9K_REMEDIATION_PROFILES
                else rng
            )
            hump_amplitudes = np.asarray(
                hump_rng.uniform(hump_low, hump_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            hump_center = float(rule["continuum_hump_center_nm"])
            hump_width = float(rule["continuum_hump_width_nm"])
            hump_support_low, hump_support_high = rule["continuum_hump_support_nm"]
            hump_profile = np.exp(-0.5 * ((wl - hump_center) / hump_width) ** 2)
            hump_in_support = ((wl >= float(hump_support_low)) & (wl <= float(hump_support_high))).astype(float)
            hump_profile = hump_profile * hump_in_support
            X_out = X_out + hump_amplitudes[:, None] * hump_profile[None, :]
        additive_baseline_range = rule["additive_baseline_range"]
        offset_low, offset_high = additive_baseline_range
        offsets = np.asarray(
            rng.uniform(offset_low, offset_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        pre_offset_reference_attenuation_active = "pre_offset_reference_attenuation_factor_range" in rule
        if pre_offset_reference_attenuation_active:
            r9f_factor_low, r9f_factor_high = rule["pre_offset_reference_attenuation_factor_range"]
            r9f_factors = np.asarray(
                rng.uniform(
                    r9f_factor_low,
                    r9f_factor_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            r9f_support_low, r9f_support_high = rule["pre_offset_reference_attenuation_support_nm"]
            r9f_support_mask = (wl >= float(r9f_support_low)) & (wl <= float(r9f_support_high))
            r9f_n_support = int(r9f_support_mask.sum())
            if r9f_n_support > 0:
                r9f_continuum_path_before_block = continuum_path_component[:, r9f_support_mask].copy()
                r9f_continuum_path_after = continuum_path_component.copy()
                r9f_continuum_path_after[:, r9f_support_mask] = r9f_continuum_path_before_block * r9f_factors[:, None]
            else:
                r9f_continuum_path_before_block = np.zeros((X_smoothed.shape[0], 0), dtype=float)
                r9f_continuum_path_after = continuum_path_component
            r9f_pre_offset_before = X_out
            X_out = r9f_continuum_path_after + feature_residual_component
            r9f_min_before_attenuation = float(r9f_pre_offset_before.min())
            r9f_max_before_attenuation = float(r9f_pre_offset_before.max())
            r9f_min_after_attenuation = float(X_out.min())
            r9f_max_after_attenuation = float(X_out.max())
            if r9f_n_support > 0:
                r9f_continuum_path_support_after = r9f_continuum_path_after[:, r9f_support_mask]
                r9f_support_mean_before = r9f_pre_offset_before[:, r9f_support_mask].mean(axis=1)
                r9f_support_mean_after = X_out[:, r9f_support_mask].mean(axis=1)
                r9f_component_ratio = np.divide(
                    r9f_continuum_path_support_after,
                    r9f_continuum_path_before_block,
                    out=np.ones_like(r9f_continuum_path_before_block),
                    where=r9f_continuum_path_before_block != 0.0,
                )
                r9f_component_nonzero_ratio = r9f_component_ratio[r9f_continuum_path_before_block != 0.0]
                r9f_support_mean_ratio = np.divide(
                    r9f_support_mean_after,
                    np.maximum(r9f_support_mean_before, 1e-300),
                )
            else:
                r9f_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
                r9f_support_mean_after = r9f_support_mean_before
                r9f_component_ratio = np.ones((X_smoothed.shape[0], 0), dtype=float)
                r9f_component_nonzero_ratio = np.asarray([], dtype=float)
                r9f_support_mean_ratio = np.ones(X_smoothed.shape[0], dtype=float)
        X_out = X_out + offsets[:, None]
        residual_transfer_active = "support_centered_residual_transfer_range" in rule
        if residual_transfer_active:
            # R7a support-centered residual transfer. The synthetic hydrocarbon
            # residual ``feature_residual`` is masked to the rule support,
            # row-centered on the support so its support-mean is zero by
            # construction, scaled by a fixed bounded draw, and added to the
            # R4a base before the final non-negative clip.
            transfer_support_low, transfer_support_high = rule["support_centered_residual_transfer_support_nm"]
            transfer_support_mask = (wl >= float(transfer_support_low)) & (wl <= float(transfer_support_high))
            masked_transfer = feature_residual.copy()
            if (~transfer_support_mask).any():
                masked_transfer[:, ~transfer_support_mask] = 0.0
            if int(transfer_support_mask.sum()) > 0:
                support_block = masked_transfer[:, transfer_support_mask]
                support_means = support_block.mean(axis=1, keepdims=True)
                masked_transfer[:, transfer_support_mask] = support_block - support_means
            transfer_low, transfer_high = rule["support_centered_residual_transfer_range"]
            transfer_scales = np.asarray(
                rng.uniform(transfer_low, transfer_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            X_out = X_out + transfer_scales[:, None] * masked_transfer
            final_min_before_clip = float(X_out.min())
            final_max_before_clip = float(X_out.max())
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_clipped = np.clip(X_out, float(clip_low), None)
        else:
            X_clipped = np.clip(X_out, float(clip_low), float(clip_high))
        if residual_transfer_active:
            n_clipped = int(np.count_nonzero(X_clipped != X_out))
            total_cells = int(X_out.size)
            final_clip_fraction = n_clipped / float(total_cells) if total_cells > 0 else 0.0
            final_min_after_clip = float(X_clipped.min())
            final_max_after_clip = float(X_clipped.max())
        X_out = X_clipped
        modulation_active = "support_centered_micro_path_modulation_strength_range" in rule
        if modulation_active:
            # R8a mean-preserving micro-path modulation. The R4a base has just
            # been clipped to non-negative absorbance above; this block
            # multiplies the support of the base by exp(strength * shape)
            # where ``shape`` is a dimensionless bounded normalization of the
            # synthetic hydrocarbon residual ``feature_residual`` masked and
            # row-centered on the rule support, scaled by a robust
            # synthetic-only norm with a numerical epsilon, and clipped to
            # ``shape_clip``. The support row mean is then exactly preserved
            # by a multiplicative renormalization. Outside the support the
            # readout is identically equal to the R4a base.
            mod_support_low, mod_support_high = rule["support_centered_micro_path_modulation_support_nm"]
            mod_support_mask = (wl >= float(mod_support_low)) & (wl <= float(mod_support_high))
            mod_n_support = int(mod_support_mask.sum())
            mod_residual = feature_residual.copy()
            if (~mod_support_mask).any():
                mod_residual[:, ~mod_support_mask] = 0.0
            if mod_n_support > 0:
                mod_support_block = mod_residual[:, mod_support_mask]
                mod_residual[:, mod_support_mask] = mod_support_block - mod_support_block.mean(axis=1, keepdims=True)
            mod_norm_method = str(rule["support_centered_micro_path_modulation_normalization"])
            mod_norm_eps = float(rule["support_centered_micro_path_modulation_normalization_epsilon"])
            if mod_n_support > 0:
                support_residual = mod_residual[:, mod_support_mask]
                if mod_norm_method == "p95_abs":
                    mod_scale_per_row = np.percentile(np.abs(support_residual), 95.0, axis=1)
                elif mod_norm_method == "rms":
                    mod_scale_per_row = np.sqrt(np.mean(support_residual**2, axis=1))
                else:
                    msg = f"Unknown R8a normalization method {mod_norm_method!r}"
                    raise ValueError(msg)
            else:
                mod_scale_per_row = np.zeros(X_smoothed.shape[0], dtype=float)
            mod_scale_safe = np.maximum(mod_scale_per_row.astype(float), mod_norm_eps)
            mod_shape = mod_residual / mod_scale_safe[:, None]
            mod_shape_clip_low, mod_shape_clip_high = rule["support_centered_micro_path_modulation_shape_clip"]
            mod_shape = np.clip(
                mod_shape,
                float(mod_shape_clip_low),
                float(mod_shape_clip_high),
            )
            mod_strength_low, mod_strength_high = rule["support_centered_micro_path_modulation_strength_range"]
            mod_strengths = np.asarray(
                rng.uniform(
                    mod_strength_low,
                    mod_strength_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            mod_modulation = np.exp(mod_strengths[:, None] * mod_shape)
            if mod_n_support > 0:
                mod_support_mean_before = X_out[:, mod_support_mask].mean(axis=1)
            else:
                mod_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
            X_modulated = X_out * mod_modulation
            if mod_n_support > 0:
                mod_support_mean_after_mod = X_modulated[:, mod_support_mask].mean(axis=1)
                mod_ratio = np.where(
                    mod_support_mean_after_mod > mod_norm_eps,
                    mod_support_mean_before / np.maximum(mod_support_mean_after_mod, mod_norm_eps),
                    1.0,
                )
                X_modulated[:, mod_support_mask] = X_modulated[:, mod_support_mask] * mod_ratio[:, None]
                mod_support_mean_after_renorm = X_modulated[:, mod_support_mask].mean(axis=1)
            else:
                mod_ratio = np.ones(X_smoothed.shape[0], dtype=float)
                mod_support_mean_after_renorm = mod_support_mean_before
            mod_min_before_guard = float(X_modulated.min())
            mod_max_before_guard = float(X_modulated.max())
            if clip_high is None:
                X_guarded = np.clip(X_modulated, float(clip_low), None)
            else:
                X_guarded = np.clip(X_modulated, float(clip_low), float(clip_high))
            mod_n_guard_clipped = int(np.count_nonzero(X_guarded != X_modulated))
            mod_total_cells = int(X_modulated.size)
            mod_guard_clip_fraction = mod_n_guard_clipped / float(mod_total_cells) if mod_total_cells > 0 else 0.0
            mod_min_after_guard = float(X_guarded.min())
            mod_max_after_guard = float(X_guarded.max())
            X_out = X_guarded
        if "shape_envelope_absorbance_range" in rule:
            # R6a hydrocarbon shape envelope. Drawn after every R4c RNG draw so
            # the R4c portion of the spectra RNG sequence stays byte-identical
            # to R4c/R5a on the same seed. The envelope is built from fixed
            # mechanistic CH-overtone bands, masked to the rule support, and
            # recentered to zero mean over the support so the addition does
            # not shift the support mean by construction.
            shape_centers = np.asarray(rule["shape_envelope_centers_nm"], dtype=float)
            shape_widths = np.asarray(rule["shape_envelope_widths_nm"], dtype=float)
            shape_weights = np.asarray(rule["shape_envelope_weights"], dtype=float)
            shape_support_low, shape_support_high = rule["shape_envelope_support_nm"]
            shape_low, shape_high = rule["shape_envelope_absorbance_range"]
            shape_amplitudes = np.asarray(
                rng.uniform(shape_low, shape_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            shape_profile = np.zeros_like(wl, dtype=float)
            for shape_center, shape_width, shape_weight in zip(
                shape_centers,
                shape_widths,
                shape_weights,
                strict=True,
            ):
                shape_profile += float(shape_weight) * np.exp(-0.5 * ((wl - float(shape_center)) / float(shape_width)) ** 2)
            if shape_profile.size and float(shape_profile.max()) > 0.0:
                shape_profile = shape_profile / float(shape_profile.max())
            shape_support_mask = (wl >= float(shape_support_low)) & (wl <= float(shape_support_high))
            shape_profile = shape_profile * shape_support_mask.astype(float)
            if int(shape_support_mask.sum()) > 0:
                shape_support_mean = float(shape_profile[shape_support_mask].mean())
                shape_profile[shape_support_mask] = shape_profile[shape_support_mask] - shape_support_mean
            # Outside the support the addition is identically zero by the mask
            # multiplication above.
            X_out = X_out + shape_amplitudes[:, None] * shape_profile[None, :]
        support_intercept_active = "support_intercept_absorbance" in rule
        if support_intercept_active:
            # R9b support-level mechanistic intercept. Applied after the R4c
            # absorbance pipeline reaches its non-negative output clip and on
            # the explicit DIESEL support window only. The intercept is a
            # single pre-declared mechanistic constant pulled from the rule
            # dictionary; it is NOT computed from any audit residual and
            # consumes no RNG draw, so the R4c portion of the spectra RNG
            # sequence above stays byte-identical to R4c on the same seed.
            # Off-support cells are left identically equal to the R4c base.
            r9b_intercept = float(rule["support_intercept_absorbance"])
            r9b_support_low, r9b_support_high = rule["support_intercept_support_nm"]
            r9b_support_mask = (wl >= float(r9b_support_low)) & (wl <= float(r9b_support_high))
            r9b_n_support = int(r9b_support_mask.sum())
            if r9b_n_support > 0:
                r9b_support_mean_before = X_out[:, r9b_support_mask].mean(axis=1)
            else:
                r9b_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
            r9b_min_before_guard = float(X_out.min())
            r9b_max_before_guard = float(X_out.max())
            X_intercepted = X_out.copy()
            if r9b_n_support > 0:
                X_intercepted[:, r9b_support_mask] = X_intercepted[:, r9b_support_mask] + r9b_intercept
            if clip_high is None:
                X_guarded_r9b = np.clip(X_intercepted, float(clip_low), None)
            else:
                X_guarded_r9b = np.clip(X_intercepted, float(clip_low), float(clip_high))
            r9b_n_guard_clipped = int(np.count_nonzero(X_guarded_r9b != X_intercepted))
            r9b_total_cells = int(X_intercepted.size)
            r9b_guard_clip_fraction = r9b_n_guard_clipped / float(r9b_total_cells) if r9b_total_cells > 0 else 0.0
            r9b_min_after_guard = float(X_guarded_r9b.min())
            r9b_max_after_guard = float(X_guarded_r9b.max())
            X_out = X_guarded_r9b
            if r9b_n_support > 0:
                r9b_support_mean_after = X_out[:, r9b_support_mask].mean(axis=1)
            else:
                r9b_support_mean_after = r9b_support_mean_before
        support_shape_active = "support_shape_centers_nm" in rule
        if support_shape_active:
            # R9c support-level shape mechanism. Applied after the R3d
            # absorbance pipeline reaches its non-negative output clip and on
            # the explicit DIESEL support window only. The shape mechanism is
            # built from PRE-DECLARED MECHANISTIC CONSTANTS (general liquid-
            # hydrocarbon NIR overtone prior); it is NOT a calibration, NOT a
            # real-stat capture, NOT a PCA loading, and NOT chosen from any
            # R9a / R9b mean-shift residual delta. Off-support cells are left
            # identically equal to the R3d base (multiplicative support mask
            # below). The shape consists of three additive components:
            #
            #   * a sum of Gaussian CH overtone bands at the rule centers
            #     with the rule per-band widths, scaled by a bounded gain
            #     draw, and selectively damped near the rule damping windows
            #     by a bounded damping strength draw;
            #   * a small Gaussian support hump, scaled by a bounded
            #     amplitude draw.
            #
            # The R9c-specific draws (gain, damping strength, hump amplitude)
            # are consumed AFTER the R3d offsets draw, leaving the R3d
            # portion of the spectra RNG sequence byte-identical to R3d on
            # the same seed.
            r9c_centers = np.asarray(rule["support_shape_centers_nm"], dtype=float)
            r9c_widths = np.asarray(rule["support_shape_widths_nm"], dtype=float)
            if r9c_widths.shape != r9c_centers.shape:
                msg = f"R9c support_shape_widths_nm must have one entry per support_shape_centers_nm; got {r9c_widths.shape!r} vs {r9c_centers.shape!r}"
                raise ValueError(msg)
            r9c_support_low, r9c_support_high = rule["support_shape_support_nm"]
            r9c_support_mask = (wl >= float(r9c_support_low)) & (wl <= float(r9c_support_high))
            r9c_n_support = int(r9c_support_mask.sum())
            r9c_gain_low, r9c_gain_high = rule["support_shape_gain_range"]
            r9c_gains = np.asarray(
                rng.uniform(r9c_gain_low, r9c_gain_high, size=X_smoothed.shape[0]),
                dtype=float,
            )
            r9c_damp_low, r9c_damp_high = rule["support_shape_damping_strength_range"]
            r9c_damping_strengths = np.asarray(
                rng.uniform(
                    r9c_damp_low,
                    r9c_damp_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            r9c_hump_low, r9c_hump_high = rule["support_shape_hump_amplitude_range"]
            r9c_hump_amplitudes = np.asarray(
                rng.uniform(
                    r9c_hump_low,
                    r9c_hump_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            r9c_ch_profile = np.zeros_like(wl, dtype=float)
            for r9c_center, r9c_width in zip(r9c_centers, r9c_widths, strict=True):
                r9c_ch_profile += np.exp(-0.5 * ((wl - float(r9c_center)) / float(r9c_width)) ** 2)
            r9c_support_mask_float = r9c_support_mask.astype(float)
            r9c_ch_profile = r9c_ch_profile * r9c_support_mask_float
            r9c_damping_profile = np.zeros_like(wl, dtype=float)
            for r9c_dc, r9c_dw, r9c_dweight in rule["support_shape_damping_windows_nm"]:
                r9c_damping_profile = np.maximum(
                    r9c_damping_profile,
                    float(r9c_dweight) * np.exp(-0.5 * ((wl - float(r9c_dc)) / float(r9c_dw)) ** 2),
                )
            r9c_hump_center = float(rule["support_shape_hump_center_nm"])
            r9c_hump_width = float(rule["support_shape_hump_width_nm"])
            r9c_hump_profile = np.exp(-0.5 * ((wl - r9c_hump_center) / r9c_hump_width) ** 2) * r9c_support_mask_float
            r9c_attenuation = 1.0 - r9c_damping_strengths[:, None] * r9c_damping_profile[None, :]
            r9c_ch_signal = r9c_gains[:, None] * r9c_ch_profile[None, :] * r9c_attenuation
            r9c_hump_signal = r9c_hump_amplitudes[:, None] * r9c_hump_profile[None, :]
            r9c_delta = r9c_ch_signal + r9c_hump_signal
            if (~r9c_support_mask).any():
                # Defensive: support_mask multiplication on the band/hump
                # profiles already zeroes the off-support contribution; this
                # explicit zero keeps off-support cells byte-identical to the
                # R3d base even under floating-point pathologies.
                r9c_delta[:, ~r9c_support_mask] = 0.0
            if r9c_n_support > 0:
                r9c_support_mean_before = X_out[:, r9c_support_mask].mean(axis=1)
            else:
                r9c_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
            r9c_min_before_guard = float((X_out + r9c_delta).min())
            r9c_max_before_guard = float((X_out + r9c_delta).max())
            X_shaped = X_out + r9c_delta
            if clip_high is None:
                X_guarded_r9c = np.clip(X_shaped, float(clip_low), None)
            else:
                X_guarded_r9c = np.clip(X_shaped, float(clip_low), float(clip_high))
            r9c_n_guard_clipped = int(np.count_nonzero(X_guarded_r9c != X_shaped))
            r9c_total_cells = int(X_shaped.size)
            r9c_guard_clip_fraction = r9c_n_guard_clipped / float(r9c_total_cells) if r9c_total_cells > 0 else 0.0
            r9c_min_after_guard = float(X_guarded_r9c.min())
            r9c_max_after_guard = float(X_guarded_r9c.max())
            X_out = X_guarded_r9c
            if r9c_n_support > 0:
                r9c_support_mean_after = X_out[:, r9c_support_mask].mean(axis=1)
            else:
                r9c_support_mean_after = r9c_support_mean_before
        support_redistribution_active = "support_redistribution_centers_nm" in rule
        if support_redistribution_active:
            # R9d energy-normalized mean-neutral support redistribution.
            # Applied after the R3d absorbance pipeline reaches its
            # non-negative output clip and on the explicit DIESEL support
            # window only. The shape is built from PRE-DECLARED MECHANISTIC
            # CONSTANTS (general liquid-hydrocarbon NIR energy redistribution
            # prior); it is NOT a calibration, NOT a real-stat capture, NOT
            # a PCA loading, NOT chosen from any R9a/R9b/R9c mean-shift
            # residual delta, and is built without reading real data or
            # marginal statistics. The shape is constructed once from the
            # rule centers/widths, mean-subtracted on the support so that
            # its support mean is zero by construction, normalized to
            # max-abs == 1 on the support, and clipped to ``shape_clip``.
            # Off-support shape values are exactly zero so that
            # ``exp(strength * shape) == 1`` byte-identically off-support.
            #
            # For each row, ``strength`` is drawn uniformly from the rule
            # range and ``X_out[:, support] *= exp(strength * shape)`` is
            # applied. The support of each row is then multiplicatively
            # renormalized so that the post-redistribution support mean
            # equals the pre-redistribution support mean exactly within
            # numerical tolerance; an epsilon guard avoids divide-by-zero
            # when the post-redistribution row support mean is at the
            # numerical floor. The R9d-specific per-row strength draw is
            # consumed AFTER the R3d offsets draw, leaving the R3d portion
            # of the spectra RNG sequence byte-identical to R3d on the
            # same seed.
            r9d_centers = np.asarray(rule["support_redistribution_centers_nm"], dtype=float)
            r9d_widths = np.asarray(rule["support_redistribution_widths_nm"], dtype=float)
            if r9d_widths.shape != r9d_centers.shape:
                msg = f"R9d support_redistribution_widths_nm must have one entry per support_redistribution_centers_nm; got {r9d_widths.shape!r} vs {r9d_centers.shape!r}"
                raise ValueError(msg)
            r9d_support_low, r9d_support_high = rule["support_redistribution_support_nm"]
            r9d_support_mask = (wl >= float(r9d_support_low)) & (wl <= float(r9d_support_high))
            r9d_n_support = int(r9d_support_mask.sum())
            r9d_strength_low, r9d_strength_high = rule["support_redistribution_log_strength_range"]
            r9d_strengths = np.asarray(
                rng.uniform(
                    r9d_strength_low,
                    r9d_strength_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            r9d_shape_clip_low, r9d_shape_clip_high = rule["support_redistribution_shape_clip"]
            r9d_renorm_eps = float(rule["support_redistribution_renorm_epsilon"])
            r9d_normalization = str(rule["support_redistribution_shape_normalization"])
            r9d_basis = np.zeros_like(wl, dtype=float)
            for r9d_center, r9d_width in zip(r9d_centers, r9d_widths, strict=True):
                r9d_basis += np.exp(-0.5 * ((wl - float(r9d_center)) / float(r9d_width)) ** 2)
            r9d_shape = np.zeros_like(wl, dtype=float)
            if r9d_n_support > 0:
                r9d_basis_support_values = r9d_basis[r9d_support_mask]
                r9d_basis_support_mean = float(r9d_basis_support_values.mean())
                r9d_shape[r9d_support_mask] = r9d_basis_support_values - r9d_basis_support_mean
            if r9d_normalization == "max_abs":
                r9d_max_abs = float(np.max(np.abs(r9d_shape))) if r9d_shape.size else 0.0
            elif r9d_normalization == "p95_abs":
                r9d_max_abs = float(np.percentile(np.abs(r9d_shape), 95.0)) if r9d_shape.size else 0.0
            else:
                msg = f"Unknown R9d shape normalization {r9d_normalization!r}"
                raise ValueError(msg)
            if r9d_max_abs > 0.0:
                r9d_shape = r9d_shape / r9d_max_abs
            r9d_shape = np.clip(
                r9d_shape,
                float(r9d_shape_clip_low),
                float(r9d_shape_clip_high),
            )
            # Off-support shape MUST be exactly zero so that
            # ``exp(strength * 0) == 1`` byte-identically off-support and
            # the multiplicative redistribution leaves the off-support cells
            # bit-equal to the R3d base.
            if (~r9d_support_mask).any():
                r9d_shape[~r9d_support_mask] = 0.0
            r9d_shape_has_positive_lobe = bool((r9d_shape > 0.0).any())
            r9d_shape_has_negative_lobe = bool((r9d_shape < 0.0).any())
            r9d_shape_support_mean = float(r9d_shape[r9d_support_mask].mean()) if r9d_n_support > 0 else 0.0
            if r9d_n_support > 0:
                r9d_support_mean_before = X_out[:, r9d_support_mask].mean(axis=1)
            else:
                r9d_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
            r9d_factor = np.exp(r9d_strengths[:, None] * r9d_shape[None, :])
            X_redistributed = X_out * r9d_factor
            if r9d_n_support > 0:
                r9d_support_mean_after_factor = X_redistributed[:, r9d_support_mask].mean(axis=1)
                r9d_renorm_safe_denom = np.where(
                    np.abs(r9d_support_mean_after_factor) > r9d_renorm_eps,
                    r9d_support_mean_after_factor,
                    r9d_renorm_eps,
                )
                r9d_renorm_ratio = np.where(
                    np.isfinite(r9d_support_mean_before) & (np.abs(r9d_support_mean_after_factor) > r9d_renorm_eps),
                    r9d_support_mean_before / r9d_renorm_safe_denom,
                    1.0,
                )
                X_redistributed[:, r9d_support_mask] = X_redistributed[:, r9d_support_mask] * r9d_renorm_ratio[:, None]
            X_out = X_redistributed
            if r9d_n_support > 0:
                r9d_support_mean_after = X_out[:, r9d_support_mask].mean(axis=1)
                r9d_support_mean_abs_error_max = float(np.max(np.abs(r9d_support_mean_after - r9d_support_mean_before)))
            else:
                r9d_support_mean_after = r9d_support_mean_before
                r9d_support_mean_abs_error_max = 0.0
        support_reference_attenuation_active = "support_reference_attenuation_factor_range" in rule
        if support_reference_attenuation_active:
            # R9e support-only reference/pathlength attenuation. Applied
            # after the R3d absorbance pipeline reaches its non-negative
            # output clip. The factor is positive, row-wise, and bounded by a
            # pre-declared generic blank/reference pathlength prior. It does
            # not add an offset, perform an extra clip, reuse the R9d shape,
            # renormalize the support mean, or transform readout space.
            r9e_factor_low, r9e_factor_high = rule["support_reference_attenuation_factor_range"]
            r9e_factors = np.asarray(
                rng.uniform(
                    r9e_factor_low,
                    r9e_factor_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            r9e_support_low, r9e_support_high = rule["support_reference_attenuation_support_nm"]
            r9e_support_mask = (wl >= float(r9e_support_low)) & (wl <= float(r9e_support_high))
            r9e_n_support = int(r9e_support_mask.sum())
            if r9e_n_support > 0:
                r9e_support_mean_before = X_out[:, r9e_support_mask].mean(axis=1)
            else:
                r9e_support_mean_before = np.zeros(X_smoothed.shape[0], dtype=float)
            r9e_min_before_attenuation = float(X_out.min())
            r9e_max_before_attenuation = float(X_out.max())
            X_attenuated = X_out.copy()
            if r9e_n_support > 0:
                r9e_support_before_block = X_out[:, r9e_support_mask].copy()
                X_attenuated[:, r9e_support_mask] = r9e_support_before_block * r9e_factors[:, None]
            else:
                r9e_support_before_block = np.zeros((X_smoothed.shape[0], 0), dtype=float)
            X_out = X_attenuated
            r9e_min_after_attenuation = float(X_out.min())
            r9e_max_after_attenuation = float(X_out.max())
            if r9e_n_support > 0:
                r9e_support_mean_after = X_out[:, r9e_support_mask].mean(axis=1)
                r9e_support_ratio = np.divide(
                    X_out[:, r9e_support_mask],
                    r9e_support_before_block,
                    out=np.ones_like(r9e_support_before_block),
                    where=r9e_support_before_block != 0.0,
                )
                r9e_support_nonzero_ratio = r9e_support_ratio[r9e_support_before_block != 0.0]
                r9e_support_mean_ratio = np.divide(
                    r9e_support_mean_after,
                    np.maximum(r9e_support_mean_before, 1e-300),
                )
            else:
                r9e_support_mean_after = r9e_support_mean_before
                r9e_support_ratio = np.ones((X_smoothed.shape[0], 0), dtype=float)
                r9e_support_nonzero_ratio = np.asarray([], dtype=float)
                r9e_support_mean_ratio = np.ones(X_smoothed.shape[0], dtype=float)
        row_pathlength_reference_active = "row_pathlength_reference_factor_range" in rule
        if row_pathlength_reference_active:
            # P2a row-level pathlength/reference attenuation. Applied after
            # the R3d absorbance pipeline reaches its non-negative output
            # clip, before any audit-side wavelength alignment. This is a
            # full-row mechanism: every generated wavelength receives the
            # same positive row-specific reference/pathlength factor, so the
            # branch is not support-only and has no off-support no-op region.
            p2a_factor_low, p2a_factor_high = rule["row_pathlength_reference_factor_range"]
            p2a_factors = np.asarray(
                rng.uniform(
                    p2a_factor_low,
                    p2a_factor_high,
                    size=X_smoothed.shape[0],
                ),
                dtype=float,
            )
            p2a_row_mean_before = X_out.mean(axis=1)
            p2a_min_before_attenuation = float(X_out.min())
            p2a_max_before_attenuation = float(X_out.max())
            X_before_p2a = X_out.copy()
            X_out = X_out * p2a_factors[:, None]
            p2a_row_mean_after = X_out.mean(axis=1)
            p2a_min_after_attenuation = float(X_out.min())
            p2a_max_after_attenuation = float(X_out.max())
            p2a_ratio = np.divide(
                X_out,
                X_before_p2a,
                out=np.ones_like(X_before_p2a),
                where=X_before_p2a != 0.0,
            )
            p2a_nonzero_ratio = p2a_ratio[X_before_p2a != 0.0]
            p2a_row_mean_ratio = np.divide(
                p2a_row_mean_after,
                np.maximum(p2a_row_mean_before, 1e-300),
            )
        transform_params.update(
            {
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
            }
        )
        if "fixed_envelope_absorbance_range" in rule:
            transform_params.update(
                {
                    "fixed_envelope_absorbance_range": [
                        float(env_low),
                        float(env_high),
                    ],
                    "fixed_envelope_absorbance_min": float(envelope_amplitudes.min()),
                    "fixed_envelope_absorbance_max": float(envelope_amplitudes.max()),
                    "fixed_envelope_centers_nm": [float(center) for center in envelope_centers],
                    "fixed_envelope_widths_nm": [float(width) for width in envelope_widths],
                    "fixed_envelope_weights": [float(weight) for weight in envelope_weights],
                    "fixed_envelope_centered": True,
                    "continuum_slope_absorbance_range": [
                        float(slope_low),
                        float(slope_high),
                    ],
                    "continuum_slope_absorbance_min": float(continuum_slopes.min()),
                    "continuum_slope_absorbance_max": float(continuum_slopes.max()),
                }
            )
        if "damping_windows_nm" in rule:
            transform_params.update(
                {
                    "damping_windows_nm": [[float(c), float(w), float(weight)] for c, w, weight in rule["damping_windows_nm"]],
                    "damping_strength_range": [
                        float(damp_low),
                        float(damp_high),
                    ],
                    "damping_strength_min": float(damping_strengths.min()),
                    "damping_strength_max": float(damping_strengths.max()),
                }
            )
        if "continuum_hump_center_nm" in rule:
            transform_params.update(
                {
                    "continuum_hump_center_nm": float(hump_center),
                    "continuum_hump_width_nm": float(hump_width),
                    "continuum_hump_amplitude_range": [
                        float(hump_low),
                        float(hump_high),
                    ],
                    "continuum_hump_amplitude_min": float(hump_amplitudes.min()),
                    "continuum_hump_amplitude_max": float(hump_amplitudes.max()),
                    "continuum_hump_support_nm": [
                        float(hump_support_low),
                        float(hump_support_high),
                    ],
                }
            )
        if route is not None:
            if profile in R6A_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_shape_route_source": str(route.get("source", "unknown")),
                        "diesel_shape_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_shape_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_shape_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_shape_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R7A_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_residual_transfer_route_source": str(route.get("source", "unknown")),
                        "diesel_residual_transfer_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_residual_transfer_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_residual_transfer_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_residual_transfer_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R8A_REMEDIATION_PROFILES + R8B_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_micro_path_modulation_route_source": str(route.get("source", "unknown")),
                        "diesel_micro_path_modulation_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_micro_path_modulation_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_micro_path_modulation_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_micro_path_modulation_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9B_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_support_intercept_route_source": str(route.get("source", "unknown")),
                        "diesel_support_intercept_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_support_intercept_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_support_intercept_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_support_intercept_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9C_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_support_shape_route_source": str(route.get("source", "unknown")),
                        "diesel_support_shape_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_support_shape_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_support_shape_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_support_shape_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9D_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_support_redistribution_route_source": str(route.get("source", "unknown")),
                        "diesel_support_redistribution_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_support_redistribution_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_support_redistribution_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_support_redistribution_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9E_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_reference_attenuation_route_source": str(route.get("source", "unknown")),
                        "diesel_reference_attenuation_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_reference_attenuation_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_reference_attenuation_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_reference_attenuation_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9F_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_pre_offset_reference_attenuation_route_source": str(route.get("source", "unknown")),
                        "diesel_pre_offset_reference_attenuation_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_pre_offset_reference_attenuation_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_pre_offset_reference_attenuation_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_pre_offset_reference_attenuation_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
            elif profile in R9H_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_support_ch_center_route_source": str(route.get("source", "unknown")),
                        "diesel_support_ch_center_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_support_ch_center_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_support_ch_center_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_support_ch_center_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "support_ch_center_drop1720_isolation_route_key": ("_r9h_diesel_support_ch_center_route"),
                        "support_ch_center_drop1720_isolation_only": True,
                        "support_ch_center_drop1720_isolation_calibration": False,
                        "support_ch_center_drop1720_isolation_uses_real_stats": False,
                        "support_ch_center_drop1720_isolation_uses_pca": False,
                        "support_ch_center_drop1720_isolation_captures_noise": False,
                        "support_ch_center_drop1720_isolation_uses_labels": False,
                        "support_ch_center_drop1720_isolation_uses_targets": False,
                        "support_ch_center_drop1720_isolation_uses_splits": False,
                        "support_ch_center_drop1720_isolation_uses_ml": False,
                        "support_ch_center_drop1720_isolation_uses_dl": False,
                        "support_ch_center_drop1720_isolation_mutates_thresholds": False,
                        "support_ch_center_drop1720_isolation_mutates_metrics": False,
                        "support_ch_center_drop1720_isolation_adds_damping": False,
                        "support_ch_center_drop1720_isolation_adds_continuum_hump": False,
                        "support_ch_center_drop1720_isolation_adds_support_intercept": False,
                        "support_ch_center_drop1720_isolation_adds_support_shape": False,
                        "support_ch_center_drop1720_isolation_adds_redistribution": False,
                        "support_ch_center_drop1720_isolation_adds_attenuation": False,
                        "support_ch_center_drop1720_isolation_readout_transform": False,
                        "support_ch_center_drop1720_isolation_extra_guard_clip": False,
                    }
                )
            elif profile in R9I_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_ch_width_gain_route_source": str(route.get("source", "unknown")),
                        "diesel_ch_width_gain_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_ch_width_gain_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_ch_width_gain_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_ch_width_gain_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_ch_width_gain_isolation_route_key": ("_r9i_diesel_ch_width_gain_route"),
                        "diesel_ch_width_gain_isolation_only": True,
                        "diesel_ch_width_gain_isolation_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_ch_width_gain_isolation_calibration": False,
                        "diesel_ch_width_gain_isolation_uses_real_stats": False,
                        "diesel_ch_width_gain_isolation_uses_pca": False,
                        "diesel_ch_width_gain_isolation_captures_noise": False,
                        "diesel_ch_width_gain_isolation_uses_labels": False,
                        "diesel_ch_width_gain_isolation_uses_targets": False,
                        "diesel_ch_width_gain_isolation_uses_splits": False,
                        "diesel_ch_width_gain_isolation_uses_ml": False,
                        "diesel_ch_width_gain_isolation_uses_dl": False,
                        "diesel_ch_width_gain_isolation_mutates_thresholds": False,
                        "diesel_ch_width_gain_isolation_mutates_metrics": False,
                        "diesel_ch_width_gain_isolation_changes_ch_centers": False,
                        "diesel_ch_width_gain_isolation_adds_damping": False,
                        "diesel_ch_width_gain_isolation_adds_continuum_hump": False,
                        "diesel_ch_width_gain_isolation_adds_support_intercept": False,
                        "diesel_ch_width_gain_isolation_adds_support_shape": False,
                        "diesel_ch_width_gain_isolation_adds_redistribution": False,
                        "diesel_ch_width_gain_isolation_adds_attenuation": False,
                        "diesel_ch_width_gain_isolation_readout_transform": False,
                        "diesel_ch_width_gain_isolation_extra_guard_clip": False,
                    }
                )
            elif profile in R9J_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_residual_damping_route_source": str(route.get("source", "unknown")),
                        "diesel_residual_damping_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_residual_damping_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_residual_damping_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_residual_damping_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_residual_damping_isolation_route_key": ("_r9j_diesel_residual_damping_route"),
                        "diesel_residual_damping_isolation_only": True,
                        "diesel_residual_damping_isolation_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_residual_damping_isolation_calibration": False,
                        "diesel_residual_damping_isolation_uses_real_stats": False,
                        "diesel_residual_damping_isolation_uses_pca": False,
                        "diesel_residual_damping_isolation_captures_noise": False,
                        "diesel_residual_damping_isolation_uses_labels": False,
                        "diesel_residual_damping_isolation_uses_targets": False,
                        "diesel_residual_damping_isolation_uses_splits": False,
                        "diesel_residual_damping_isolation_uses_ml": False,
                        "diesel_residual_damping_isolation_uses_dl": False,
                        "diesel_residual_damping_isolation_mutates_thresholds": False,
                        "diesel_residual_damping_isolation_mutates_metrics": False,
                        "diesel_residual_damping_isolation_changes_ch_centers": False,
                        "diesel_residual_damping_isolation_changes_ch_width_gain": False,
                        "diesel_residual_damping_isolation_adds_damping": True,
                        "diesel_residual_damping_isolation_adds_continuum_hump": False,
                        "diesel_residual_damping_isolation_adds_support_intercept": False,
                        "diesel_residual_damping_isolation_adds_support_shape": False,
                        "diesel_residual_damping_isolation_adds_redistribution": False,
                        "diesel_residual_damping_isolation_adds_attenuation": False,
                        "diesel_residual_damping_isolation_readout_transform": False,
                        "diesel_residual_damping_isolation_extra_guard_clip": False,
                    }
                )
            elif profile in R9K_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_continuum_hump_route_source": str(route.get("source", "unknown")),
                        "diesel_continuum_hump_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_continuum_hump_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_continuum_hump_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_continuum_hump_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_continuum_hump_isolation_route_key": ("_r9k_diesel_continuum_hump_route"),
                        "diesel_continuum_hump_isolation_only": True,
                        "diesel_continuum_hump_isolation_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_continuum_hump_isolation_calibration": False,
                        "diesel_continuum_hump_isolation_uses_real_stats": False,
                        "diesel_continuum_hump_isolation_uses_pca": False,
                        "diesel_continuum_hump_isolation_captures_noise": False,
                        "diesel_continuum_hump_isolation_uses_labels": False,
                        "diesel_continuum_hump_isolation_uses_targets": False,
                        "diesel_continuum_hump_isolation_uses_splits": False,
                        "diesel_continuum_hump_isolation_uses_ml": False,
                        "diesel_continuum_hump_isolation_uses_dl": False,
                        "diesel_continuum_hump_isolation_mutates_thresholds": False,
                        "diesel_continuum_hump_isolation_mutates_metrics": False,
                        "diesel_continuum_hump_isolation_changes_ch_centers": False,
                        "diesel_continuum_hump_isolation_changes_ch_width_gain": False,
                        "diesel_continuum_hump_isolation_adds_damping": False,
                        "diesel_continuum_hump_isolation_adds_continuum_hump": True,
                        "diesel_continuum_hump_isolation_adds_support_intercept": False,
                        "diesel_continuum_hump_isolation_adds_support_shape": False,
                        "diesel_continuum_hump_isolation_adds_redistribution": False,
                        "diesel_continuum_hump_isolation_adds_attenuation": False,
                        "diesel_continuum_hump_isolation_readout_transform": False,
                        "diesel_continuum_hump_isolation_extra_guard_clip": False,
                    }
                )
            elif profile in R9L_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_residual_damping_clean_attenuation_route_source": str(route.get("source", "unknown")),
                        "diesel_residual_damping_clean_attenuation_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_residual_damping_clean_attenuation_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_residual_damping_clean_attenuation_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_residual_damping_clean_attenuation_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_residual_damping_clean_attenuation_route_key": ("_r9l_diesel_residual_damping_clean_attenuation_route"),
                        "diesel_residual_damping_clean_attenuation_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_residual_damping_clean_attenuation_calibration": False,
                        "diesel_residual_damping_clean_attenuation_uses_real_stats": False,
                        "diesel_residual_damping_clean_attenuation_uses_pca": False,
                        "diesel_residual_damping_clean_attenuation_captures_noise": False,
                        "diesel_residual_damping_clean_attenuation_uses_labels": False,
                        "diesel_residual_damping_clean_attenuation_uses_targets": False,
                        "diesel_residual_damping_clean_attenuation_uses_splits": False,
                        "diesel_residual_damping_clean_attenuation_uses_ml": False,
                        "diesel_residual_damping_clean_attenuation_uses_dl": False,
                        "diesel_residual_damping_clean_attenuation_mutates_thresholds": False,
                        "diesel_residual_damping_clean_attenuation_mutates_metrics": False,
                        "diesel_residual_damping_clean_attenuation_changes_ch_centers": False,
                        "diesel_residual_damping_clean_attenuation_changes_ch_width_gain": False,
                        "diesel_residual_damping_clean_attenuation_adds_damping": True,
                        "diesel_residual_damping_clean_attenuation_adds_clean_attenuation": True,
                        "diesel_residual_damping_clean_attenuation_adds_continuum_hump": False,
                        "diesel_residual_damping_clean_attenuation_adds_pre_offset_attenuation": False,
                        "diesel_residual_damping_clean_attenuation_adds_support_intercept": False,
                        "diesel_residual_damping_clean_attenuation_adds_support_shape": False,
                        "diesel_residual_damping_clean_attenuation_adds_redistribution": False,
                        "diesel_residual_damping_clean_attenuation_readout_transform": False,
                        "diesel_residual_damping_clean_attenuation_extra_guard_clip": False,
                        "diesel_residual_damping_clean_attenuation_integration": False,
                        "diesel_residual_damping_clean_attenuation_no_calibration": True,
                        "diesel_residual_damping_clean_attenuation_no_real_stats": True,
                        "diesel_residual_damping_clean_attenuation_no_pca": True,
                        "diesel_residual_damping_clean_attenuation_no_noise_capture": True,
                        "diesel_residual_damping_clean_attenuation_no_ml_dl": True,
                        "diesel_residual_damping_clean_attenuation_no_labels_targets_splits": True,
                        "diesel_residual_damping_clean_attenuation_no_threshold_metric_mutation": True,
                    }
                )
            elif profile in R9M_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_width_gain_damping_clean_attenuation_route_source": str(route.get("source", "unknown")),
                        "diesel_width_gain_damping_clean_attenuation_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_width_gain_damping_clean_attenuation_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_width_gain_damping_clean_attenuation_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_width_gain_damping_clean_attenuation_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_width_gain_damping_clean_attenuation_route_key": ("_r9m_diesel_width_gain_damping_clean_attenuation_route"),
                        "diesel_width_gain_damping_clean_attenuation_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_width_gain_damping_clean_attenuation_calibration": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_real_stats": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_pca": False,
                        "diesel_width_gain_damping_clean_attenuation_captures_noise": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_labels": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_targets": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_splits": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_ml": False,
                        "diesel_width_gain_damping_clean_attenuation_uses_dl": False,
                        "diesel_width_gain_damping_clean_attenuation_mutates_thresholds": False,
                        "diesel_width_gain_damping_clean_attenuation_mutates_metrics": False,
                        "diesel_width_gain_damping_clean_attenuation_changes_ch_centers": False,
                        "diesel_width_gain_damping_clean_attenuation_changes_ch_width_gain": True,
                        "diesel_width_gain_damping_clean_attenuation_adds_damping": True,
                        "diesel_width_gain_damping_clean_attenuation_adds_clean_attenuation": True,
                        "diesel_width_gain_damping_clean_attenuation_adds_continuum_hump": False,
                        "diesel_width_gain_damping_clean_attenuation_adds_pre_offset_attenuation": False,
                        "diesel_width_gain_damping_clean_attenuation_adds_support_intercept": False,
                        "diesel_width_gain_damping_clean_attenuation_adds_support_shape": False,
                        "diesel_width_gain_damping_clean_attenuation_adds_redistribution": False,
                        "diesel_width_gain_damping_clean_attenuation_readout_transform": False,
                        "diesel_width_gain_damping_clean_attenuation_extra_guard_clip": False,
                        "diesel_width_gain_damping_clean_attenuation_integration": False,
                        "diesel_width_gain_damping_clean_attenuation_no_calibration": True,
                        "diesel_width_gain_damping_clean_attenuation_no_real_stats": True,
                        "diesel_width_gain_damping_clean_attenuation_no_pca": True,
                        "diesel_width_gain_damping_clean_attenuation_no_noise_capture": True,
                        "diesel_width_gain_damping_clean_attenuation_no_ml_dl": True,
                        "diesel_width_gain_damping_clean_attenuation_no_labels_targets_splits": True,
                        "diesel_width_gain_damping_clean_attenuation_no_threshold_metric_mutation": True,
                    }
                )
            elif profile in P2A_REMEDIATION_PROFILES:
                transform_params.update(
                    {
                        "diesel_row_pathlength_reference_route_source": str(route.get("source", "unknown")),
                        "diesel_row_pathlength_reference_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_row_pathlength_reference_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_row_pathlength_reference_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_row_pathlength_reference_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                        "diesel_row_pathlength_reference_route_key": ("_p2a_diesel_row_pathlength_reference_route"),
                        "diesel_row_pathlength_reference_seed_source_profile": ("r3d_diesel_matrix_v1"),
                        "diesel_row_pathlength_reference_calibration": False,
                        "diesel_row_pathlength_reference_uses_real_stats": False,
                        "diesel_row_pathlength_reference_uses_pca": False,
                        "diesel_row_pathlength_reference_captures_noise": False,
                        "diesel_row_pathlength_reference_uses_labels": False,
                        "diesel_row_pathlength_reference_uses_targets": False,
                        "diesel_row_pathlength_reference_uses_splits": False,
                        "diesel_row_pathlength_reference_uses_ml": False,
                        "diesel_row_pathlength_reference_uses_dl": False,
                        "diesel_row_pathlength_reference_mutates_thresholds": False,
                        "diesel_row_pathlength_reference_mutates_metrics": False,
                        "diesel_row_pathlength_reference_changes_ch_centers": False,
                        "diesel_row_pathlength_reference_changes_ch_width_gain": False,
                        "diesel_row_pathlength_reference_adds_damping": False,
                        "diesel_row_pathlength_reference_adds_continuum_hump": False,
                        "diesel_row_pathlength_reference_adds_support_only_correction": False,
                        "diesel_row_pathlength_reference_readout_transform": False,
                        "diesel_row_pathlength_reference_extra_guard_clip": False,
                        "diesel_row_pathlength_reference_integration": False,
                        "diesel_row_pathlength_reference_no_calibration": True,
                        "diesel_row_pathlength_reference_no_real_stats": True,
                        "diesel_row_pathlength_reference_no_pca": True,
                        "diesel_row_pathlength_reference_no_noise_capture": True,
                        "diesel_row_pathlength_reference_no_ml_dl": True,
                        "diesel_row_pathlength_reference_no_labels_targets_splits": True,
                        "diesel_row_pathlength_reference_no_threshold_metric_mutation": True,
                    }
                )
            else:
                transform_params.update(
                    {
                        "diesel_readout_route_source": str(route.get("source", "unknown")),
                        "diesel_readout_route_marker": str(route.get("route_marker", "unknown")),
                        "diesel_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                        "diesel_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                        "diesel_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                    }
                )
        if "support_centered_residual_transfer_range" in rule:
            transform_params.update(
                {
                    "support_centered_residual_transfer_range": [
                        float(transfer_low),
                        float(transfer_high),
                    ],
                    "support_centered_residual_transfer_min": float(transfer_scales.min()),
                    "support_centered_residual_transfer_max": float(transfer_scales.max()),
                    "support_centered_residual_transfer_support_nm": [
                        float(transfer_support_low),
                        float(transfer_support_high),
                    ],
                    "support_centered_residual_transfer_centering": ("row_center_on_support_zero_outside"),
                    "support_centered_residual_transfer_source": str(
                        rule.get(
                            "support_centered_residual_transfer_source",
                            "fixed_synthetic_hydrocarbon_residual_transfer_prior",
                        )
                    ),
                    "support_centered_residual_transfer_application_stage": ("before_final_clip_after_r4a_base"),
                    "final_min_absorbance_before_clip": float(final_min_before_clip),
                    "final_max_absorbance_before_clip": float(final_max_before_clip),
                    "final_min_absorbance_after_clip": float(final_min_after_clip),
                    "final_max_absorbance_after_clip": float(final_max_after_clip),
                    "final_clip_rule": ("nonnegative_lower_bound_no_upper_bound" if clip_high is None else "nonnegative_with_upper_bound"),
                    "final_clip_fraction": float(final_clip_fraction),
                }
            )
        if "support_centered_micro_path_modulation_strength_range" in rule:
            transform_params.update(
                {
                    "support_centered_micro_path_modulation_strength_range": [
                        float(mod_strength_low),
                        float(mod_strength_high),
                    ],
                    "support_centered_micro_path_modulation_strength_min": float(mod_strengths.min()),
                    "support_centered_micro_path_modulation_strength_max": float(mod_strengths.max()),
                    "support_centered_micro_path_modulation_support_nm": [
                        float(mod_support_low),
                        float(mod_support_high),
                    ],
                    "support_centered_micro_path_modulation_normalization": (mod_norm_method),
                    "support_centered_micro_path_modulation_normalization_epsilon": (float(mod_norm_eps)),
                    "support_centered_micro_path_modulation_shape_clip": [
                        float(mod_shape_clip_low),
                        float(mod_shape_clip_high),
                    ],
                    "support_centered_micro_path_modulation_shape_min": float(mod_shape.min()),
                    "support_centered_micro_path_modulation_shape_max": float(mod_shape.max()),
                    "support_centered_micro_path_modulation_modulation_min": float(mod_modulation.min()),
                    "support_centered_micro_path_modulation_modulation_max": float(mod_modulation.max()),
                    "support_centered_micro_path_modulation_centering": ("row_center_on_support_zero_outside"),
                    "support_centered_micro_path_modulation_application_stage": ("after_base_nonnegative_clip"),
                    "support_centered_micro_path_modulation_normalization_source": ("synthetic_internal_residual_only"),
                    "support_centered_micro_path_modulation_source": str(
                        rule.get(
                            "support_centered_micro_path_modulation_source",
                            "synthetic_internal_residual_only",
                        )
                    ),
                    "support_centered_micro_path_modulation_support_mean_before_min": float(mod_support_mean_before.min()),
                    "support_centered_micro_path_modulation_support_mean_before_max": float(mod_support_mean_before.max()),
                    "support_centered_micro_path_modulation_support_mean_after_min": float(mod_support_mean_after_renorm.min()),
                    "support_centered_micro_path_modulation_support_mean_after_max": float(mod_support_mean_after_renorm.max()),
                    "support_centered_micro_path_modulation_support_mean_abs_delta_max": float(np.max(np.abs(mod_support_mean_after_renorm - mod_support_mean_before))),
                    "support_centered_micro_path_modulation_min_before_guard_clip": (float(mod_min_before_guard)),
                    "support_centered_micro_path_modulation_max_before_guard_clip": (float(mod_max_before_guard)),
                    "support_centered_micro_path_modulation_min_after_guard_clip": (float(mod_min_after_guard)),
                    "support_centered_micro_path_modulation_max_after_guard_clip": (float(mod_max_after_guard)),
                    "support_centered_micro_path_modulation_guard_clip_rule": ("nonnegative_lower_bound_no_upper_bound" if clip_high is None else "nonnegative_with_upper_bound"),
                    "support_centered_micro_path_modulation_guard_clip_fraction": (float(mod_guard_clip_fraction)),
                    "support_centered_micro_path_modulation_n_support_bins": (int(mod_n_support)),
                }
            )
        if "shape_envelope_absorbance_range" in rule:
            transform_params.update(
                {
                    "shape_envelope_centers_nm": [float(c) for c in shape_centers],
                    "shape_envelope_widths_nm": [float(w) for w in shape_widths],
                    "shape_envelope_weights": [float(w) for w in shape_weights],
                    "shape_envelope_support_nm": [
                        float(shape_support_low),
                        float(shape_support_high),
                    ],
                    "shape_envelope_absorbance_range": [
                        float(shape_low),
                        float(shape_high),
                    ],
                    "shape_envelope_absorbance_min": float(shape_amplitudes.min()),
                    "shape_envelope_absorbance_max": float(shape_amplitudes.max()),
                    "shape_envelope_zero_mean_on_support": True,
                    "shape_envelope_application_stage": "after_r4c_output_clip",
                    "output_clip_absorbance_applies_to": ("r4c_pipeline_before_shape_envelope"),
                    "shape_envelope_final_output_clip_absorbance": None,
                    "shape_envelope_final_min_absorbance": float(X_out.min()),
                    "shape_envelope_final_max_absorbance": float(X_out.max()),
                }
            )
        if support_intercept_active:
            transform_params.update(
                {
                    "support_intercept_absorbance": float(r9b_intercept),
                    "support_intercept_support_nm": [
                        float(r9b_support_low),
                        float(r9b_support_high),
                    ],
                    "support_intercept_n_support_bins": int(r9b_n_support),
                    "support_intercept_source": str(
                        rule.get(
                            "support_intercept_source",
                            "fixed_blank_cell_detector_support_level_intercept_prior",
                        )
                    ),
                    "support_intercept_application_stage": "after_r4c_output_clip",
                    "support_intercept_off_support_unchanged": True,
                    "support_intercept_value_origin": ("pre_declared_mechanistic_constant_not_audit_delta"),
                    "support_intercept_support_mean_before_min": float(r9b_support_mean_before.min()),
                    "support_intercept_support_mean_before_max": float(r9b_support_mean_before.max()),
                    "support_intercept_support_mean_after_min": float(r9b_support_mean_after.min()),
                    "support_intercept_support_mean_after_max": float(r9b_support_mean_after.max()),
                    "support_intercept_support_mean_delta_min": float((r9b_support_mean_after - r9b_support_mean_before).min()),
                    "support_intercept_support_mean_delta_max": float((r9b_support_mean_after - r9b_support_mean_before).max()),
                    "support_intercept_min_before_guard_clip": float(r9b_min_before_guard),
                    "support_intercept_max_before_guard_clip": float(r9b_max_before_guard),
                    "support_intercept_min_after_guard_clip": float(r9b_min_after_guard),
                    "support_intercept_max_after_guard_clip": float(r9b_max_after_guard),
                    "support_intercept_guard_clip_rule": ("nonnegative_lower_bound_no_upper_bound" if clip_high is None else "nonnegative_with_upper_bound"),
                    "support_intercept_guard_clip_fraction": float(r9b_guard_clip_fraction),
                }
            )
        if support_shape_active:
            transform_params.update(
                {
                    "support_shape_centers_nm": [float(c) for c in r9c_centers],
                    "support_shape_widths_nm": [float(w) for w in r9c_widths],
                    "support_shape_gain_range": [
                        float(r9c_gain_low),
                        float(r9c_gain_high),
                    ],
                    "support_shape_gain_min": float(r9c_gains.min()),
                    "support_shape_gain_max": float(r9c_gains.max()),
                    "support_shape_damping_windows_nm": [[float(c), float(w), float(weight)] for c, w, weight in rule["support_shape_damping_windows_nm"]],
                    "support_shape_damping_strength_range": [
                        float(r9c_damp_low),
                        float(r9c_damp_high),
                    ],
                    "support_shape_damping_strength_min": float(r9c_damping_strengths.min()),
                    "support_shape_damping_strength_max": float(r9c_damping_strengths.max()),
                    "support_shape_hump_center_nm": float(r9c_hump_center),
                    "support_shape_hump_width_nm": float(r9c_hump_width),
                    "support_shape_hump_amplitude_range": [
                        float(r9c_hump_low),
                        float(r9c_hump_high),
                    ],
                    "support_shape_hump_amplitude_min": float(r9c_hump_amplitudes.min()),
                    "support_shape_hump_amplitude_max": float(r9c_hump_amplitudes.max()),
                    "support_shape_support_nm": [
                        float(r9c_support_low),
                        float(r9c_support_high),
                    ],
                    "support_shape_n_support_bins": int(r9c_n_support),
                    "support_shape_application_stage": "after_r3d_output_clip",
                    "support_shape_off_support_unchanged": True,
                    "support_shape_mechanism": str(
                        rule.get(
                            "support_shape_mechanism",
                            "selective_ch_bandwidth_damping_support_shape_only",
                        )
                    ),
                    "support_shape_constants_source": str(
                        rule.get(
                            "support_shape_constants_source",
                            "predeclared_general_liquid_hydrocarbon_nir_prior",
                        )
                    ),
                    "support_shape_value_origin": ("predeclared_mechanistic_constants_not_audit_delta"),
                    "support_shape_calibration": False,
                    "support_shape_uses_real_stats": False,
                    "support_shape_uses_pca": False,
                    "support_shape_captures_noise": False,
                    "support_shape_uses_labels": False,
                    "support_shape_uses_targets": False,
                    "support_shape_uses_splits": False,
                    "support_shape_uses_ml": False,
                    "support_shape_uses_dl": False,
                    "support_shape_mutates_thresholds": False,
                    "support_shape_mutates_metrics": False,
                    "support_shape_adds_offset": False,
                    "support_shape_only": True,
                    "support_shape_support_mean_before_min": float(r9c_support_mean_before.min()),
                    "support_shape_support_mean_before_max": float(r9c_support_mean_before.max()),
                    "support_shape_support_mean_after_min": float(r9c_support_mean_after.min()),
                    "support_shape_support_mean_after_max": float(r9c_support_mean_after.max()),
                    "support_shape_min_before_guard_clip": float(r9c_min_before_guard),
                    "support_shape_max_before_guard_clip": float(r9c_max_before_guard),
                    "support_shape_min_after_guard_clip": float(r9c_min_after_guard),
                    "support_shape_max_after_guard_clip": float(r9c_max_after_guard),
                    "support_shape_guard_clip_rule": ("nonnegative_lower_bound_no_upper_bound" if clip_high is None else "nonnegative_with_upper_bound"),
                    "support_shape_guard_clip_fraction": float(r9c_guard_clip_fraction),
                }
            )
        if support_redistribution_active:
            transform_params.update(
                {
                    "support_redistribution_centers_nm": [float(c) for c in r9d_centers],
                    "support_redistribution_widths_nm": [float(w) for w in r9d_widths],
                    "support_redistribution_log_strength_range": [
                        float(r9d_strength_low),
                        float(r9d_strength_high),
                    ],
                    "support_redistribution_strength_min": float(r9d_strengths.min()),
                    "support_redistribution_strength_max": float(r9d_strengths.max()),
                    "support_redistribution_shape_clip": [
                        float(r9d_shape_clip_low),
                        float(r9d_shape_clip_high),
                    ],
                    "support_redistribution_renorm_epsilon": float(r9d_renorm_eps),
                    "support_redistribution_support_nm": [
                        float(r9d_support_low),
                        float(r9d_support_high),
                    ],
                    "support_redistribution_n_support_bins": int(r9d_n_support),
                    "support_redistribution_application_stage": ("after_r3d_output_clip"),
                    "support_redistribution_off_support_unchanged": True,
                    "support_redistribution_mechanism": str(
                        rule.get(
                            "support_redistribution_mechanism",
                            "energy_normalized_mean_neutral_support_redistribution",
                        )
                    ),
                    "support_redistribution_constants_source": str(
                        rule.get(
                            "support_redistribution_constants_source",
                            "predeclared_general_liquid_hydrocarbon_nir_energy_redistribution_prior",
                        )
                    ),
                    "support_redistribution_shape_normalization": str(r9d_normalization),
                    "support_redistribution_value_origin": ("predeclared_mechanistic_constants_not_audit_delta"),
                    "support_redistribution_shape_support_mean": float(r9d_shape_support_mean),
                    "support_redistribution_shape_has_positive_lobe": bool(r9d_shape_has_positive_lobe),
                    "support_redistribution_shape_has_negative_lobe": bool(r9d_shape_has_negative_lobe),
                    "support_redistribution_mean_neutral": True,
                    "support_redistribution_energy_normalized": True,
                    "support_redistribution_calibration": False,
                    "support_redistribution_uses_real_stats": False,
                    "support_redistribution_uses_pca": False,
                    "support_redistribution_captures_noise": False,
                    "support_redistribution_uses_labels": False,
                    "support_redistribution_uses_targets": False,
                    "support_redistribution_uses_splits": False,
                    "support_redistribution_uses_ml": False,
                    "support_redistribution_uses_dl": False,
                    "support_redistribution_mutates_thresholds": False,
                    "support_redistribution_mutates_metrics": False,
                    "support_redistribution_adds_offset": False,
                    "support_redistribution_only": True,
                    "support_redistribution_support_mean_before_min": float(r9d_support_mean_before.min()),
                    "support_redistribution_support_mean_before_max": float(r9d_support_mean_before.max()),
                    "support_redistribution_support_mean_after_min": float(r9d_support_mean_after.min()),
                    "support_redistribution_support_mean_after_max": float(r9d_support_mean_after.max()),
                    "support_redistribution_support_mean_abs_error_max": float(r9d_support_mean_abs_error_max),
                }
            )
        if support_reference_attenuation_active:
            transform_params.update(
                {
                    "support_reference_attenuation_factor_range": [
                        float(r9e_factor_low),
                        float(r9e_factor_high),
                    ],
                    "support_reference_attenuation_factor_min": float(r9e_factors.min()),
                    "support_reference_attenuation_factor_max": float(r9e_factors.max()),
                    "support_reference_attenuation_support_nm": [
                        float(r9e_support_low),
                        float(r9e_support_high),
                    ],
                    "support_reference_attenuation_n_support_bins": int(r9e_n_support),
                    "support_reference_attenuation_application_stage": ("after_r3d_output_clip"),
                    "support_reference_attenuation_off_support_unchanged": True,
                    "support_reference_attenuation_mechanism": str(
                        rule.get(
                            "support_reference_attenuation_mechanism",
                            "positive_pathlength_reference_attenuation_support_only",
                        )
                    ),
                    "support_reference_attenuation_constants_source": str(
                        rule.get(
                            "support_reference_attenuation_constants_source",
                            "predeclared_generic_blank_reference_pathlength_attenuation_prior",
                        )
                    ),
                    "support_reference_attenuation_value_origin": ("predeclared_mechanistic_constants_not_audit_delta"),
                    "support_reference_attenuation_calibration": False,
                    "support_reference_attenuation_uses_real_stats": False,
                    "support_reference_attenuation_uses_pca": False,
                    "support_reference_attenuation_captures_noise": False,
                    "support_reference_attenuation_uses_labels": False,
                    "support_reference_attenuation_uses_targets": False,
                    "support_reference_attenuation_uses_splits": False,
                    "support_reference_attenuation_uses_ml": False,
                    "support_reference_attenuation_uses_dl": False,
                    "support_reference_attenuation_mutates_thresholds": False,
                    "support_reference_attenuation_mutates_metrics": False,
                    "support_reference_attenuation_adds_offset": False,
                    "support_reference_attenuation_no_additional_clip": True,
                    "support_reference_attenuation_uses_r9d_shape": False,
                    "support_reference_attenuation_renormalizes_support_mean": False,
                    "support_reference_attenuation_readout_transform": False,
                    "support_reference_attenuation_only": (profile in R9E_REMEDIATION_PROFILES),
                    "support_reference_attenuation_support_mean_before_min": float(r9e_support_mean_before.min()),
                    "support_reference_attenuation_support_mean_before_max": float(r9e_support_mean_before.max()),
                    "support_reference_attenuation_support_mean_after_min": float(r9e_support_mean_after.min()),
                    "support_reference_attenuation_support_mean_after_max": float(r9e_support_mean_after.max()),
                    "support_reference_attenuation_support_ratio_min": float(r9e_support_nonzero_ratio.min() if r9e_support_nonzero_ratio.size else r9e_factors.min()),
                    "support_reference_attenuation_support_ratio_max": float(r9e_support_nonzero_ratio.max() if r9e_support_nonzero_ratio.size else r9e_factors.max()),
                    "support_reference_attenuation_support_mean_ratio_min": float(r9e_support_mean_ratio.min()),
                    "support_reference_attenuation_support_mean_ratio_max": float(r9e_support_mean_ratio.max()),
                    "support_reference_attenuation_min_before_attenuation": float(r9e_min_before_attenuation),
                    "support_reference_attenuation_max_before_attenuation": float(r9e_max_before_attenuation),
                    "support_reference_attenuation_min_after_attenuation": float(r9e_min_after_attenuation),
                    "support_reference_attenuation_max_after_attenuation": float(r9e_max_after_attenuation),
                    "support_reference_attenuation_guard_clip_fraction": 0.0,
                    "support_reference_attenuation_route_key": (
                        "_r9m_diesel_width_gain_damping_clean_attenuation_route"
                        if profile in R9M_REMEDIATION_PROFILES
                        else ("_r9l_diesel_residual_damping_clean_attenuation_route" if profile in R9L_REMEDIATION_PROFILES else "_r9e_diesel_reference_attenuation_route")
                    ),
                }
            )
        if pre_offset_reference_attenuation_active:
            transform_params.update(
                {
                    "support_reference_attenuation_factor_range": [
                        float(r9f_factor_low),
                        float(r9f_factor_high),
                    ],
                    "support_reference_attenuation_factor_min": float(r9f_factors.min()),
                    "support_reference_attenuation_factor_max": float(r9f_factors.max()),
                    "support_reference_attenuation_support_nm": [
                        float(r9f_support_low),
                        float(r9f_support_high),
                    ],
                    "support_reference_attenuation_n_support_bins": int(r9f_n_support),
                    "support_reference_attenuation_application_stage": ("before_additive_baseline_and_output_clip_on_continuum_path_component"),
                    "support_reference_attenuation_component_only": True,
                    "support_reference_attenuation_offset_unchanged": True,
                    "support_reference_attenuation_feature_residual_unchanged": True,
                    "support_reference_attenuation_off_support_unchanged": True,
                    "support_reference_attenuation_mechanism": str(
                        rule.get(
                            "pre_offset_reference_attenuation_mechanism",
                            "positive_pathlength_reference_attenuation_continuum_path_component_only",
                        )
                    ),
                    "support_reference_attenuation_constants_source": str(
                        rule.get(
                            "pre_offset_reference_attenuation_constants_source",
                            "predeclared_generic_blank_reference_pathlength_attenuation_prior",
                        )
                    ),
                    "support_reference_attenuation_value_origin": ("predeclared_mechanistic_constants_not_audit_delta"),
                    "support_reference_attenuation_calibration": False,
                    "support_reference_attenuation_uses_real_stats": False,
                    "support_reference_attenuation_uses_pca": False,
                    "support_reference_attenuation_captures_noise": False,
                    "support_reference_attenuation_uses_labels": False,
                    "support_reference_attenuation_uses_targets": False,
                    "support_reference_attenuation_uses_splits": False,
                    "support_reference_attenuation_uses_ml": False,
                    "support_reference_attenuation_uses_dl": False,
                    "support_reference_attenuation_mutates_thresholds": False,
                    "support_reference_attenuation_mutates_metrics": False,
                    "support_reference_attenuation_adds_offset": False,
                    "support_reference_attenuation_no_additional_clip": True,
                    "support_reference_attenuation_uses_r9d_shape": False,
                    "support_reference_attenuation_renormalizes_support_mean": False,
                    "support_reference_attenuation_readout_transform": False,
                    "support_reference_attenuation_only": True,
                    "support_reference_attenuation_component_ratio_min": float(r9f_component_nonzero_ratio.min() if r9f_component_nonzero_ratio.size else r9f_factors.min()),
                    "support_reference_attenuation_component_ratio_max": float(r9f_component_nonzero_ratio.max() if r9f_component_nonzero_ratio.size else r9f_factors.max()),
                    "support_reference_attenuation_support_ratio_min": float(r9f_component_nonzero_ratio.min() if r9f_component_nonzero_ratio.size else r9f_factors.min()),
                    "support_reference_attenuation_support_ratio_max": float(r9f_component_nonzero_ratio.max() if r9f_component_nonzero_ratio.size else r9f_factors.max()),
                    "support_reference_attenuation_support_mean_before_min": float(r9f_support_mean_before.min()),
                    "support_reference_attenuation_support_mean_before_max": float(r9f_support_mean_before.max()),
                    "support_reference_attenuation_support_mean_after_min": float(r9f_support_mean_after.min()),
                    "support_reference_attenuation_support_mean_after_max": float(r9f_support_mean_after.max()),
                    "support_reference_attenuation_support_mean_ratio_min": float(r9f_support_mean_ratio.min()),
                    "support_reference_attenuation_support_mean_ratio_max": float(r9f_support_mean_ratio.max()),
                    "support_reference_attenuation_min_before_attenuation": float(r9f_min_before_attenuation),
                    "support_reference_attenuation_max_before_attenuation": float(r9f_max_before_attenuation),
                    "support_reference_attenuation_min_after_attenuation": float(r9f_min_after_attenuation),
                    "support_reference_attenuation_max_after_attenuation": float(r9f_max_after_attenuation),
                    "support_reference_attenuation_guard_clip_fraction": 0.0,
                    "support_reference_attenuation_route_key": ("_r9f_diesel_pre_offset_reference_attenuation_route"),
                }
            )
        if row_pathlength_reference_active:
            transform_params.update(
                {
                    "row_pathlength_reference_factor_range": [
                        float(p2a_factor_low),
                        float(p2a_factor_high),
                    ],
                    "row_pathlength_reference_factor_min": float(p2a_factors.min()),
                    "row_pathlength_reference_factor_max": float(p2a_factors.max()),
                    "row_pathlength_reference_n_wavelengths": int(wl.size),
                    "row_pathlength_reference_application_stage": ("after_r3d_output_clip_before_audit_alignment"),
                    "row_pathlength_reference_applies_to": ("full_generated_wavelength_row"),
                    "row_pathlength_reference_off_support_unchanged": False,
                    "row_pathlength_reference_support_only": False,
                    "row_pathlength_reference_mechanism": str(
                        rule.get(
                            "row_pathlength_reference_mechanism",
                            "positive_row_level_pathlength_reference_attenuation_full_wavelength_row",
                        )
                    ),
                    "row_pathlength_reference_constants_source": str(
                        rule.get(
                            "row_pathlength_reference_constants_source",
                            "predeclared_generic_blank_reference_row_pathlength_distribution_prior",
                        )
                    ),
                    "row_pathlength_reference_value_origin": ("predeclared_mechanistic_constants_not_audit_delta"),
                    "row_pathlength_reference_calibration": False,
                    "row_pathlength_reference_uses_real_stats": False,
                    "row_pathlength_reference_uses_pca": False,
                    "row_pathlength_reference_captures_noise": False,
                    "row_pathlength_reference_uses_labels": False,
                    "row_pathlength_reference_uses_targets": False,
                    "row_pathlength_reference_uses_splits": False,
                    "row_pathlength_reference_uses_ml": False,
                    "row_pathlength_reference_uses_dl": False,
                    "row_pathlength_reference_mutates_thresholds": False,
                    "row_pathlength_reference_mutates_metrics": False,
                    "row_pathlength_reference_adds_offset": False,
                    "row_pathlength_reference_no_additional_clip": True,
                    "row_pathlength_reference_readout_transform": False,
                    "row_pathlength_reference_row_mean_before_min": float(p2a_row_mean_before.min()),
                    "row_pathlength_reference_row_mean_before_max": float(p2a_row_mean_before.max()),
                    "row_pathlength_reference_row_mean_after_min": float(p2a_row_mean_after.min()),
                    "row_pathlength_reference_row_mean_after_max": float(p2a_row_mean_after.max()),
                    "row_pathlength_reference_row_ratio_min": float(p2a_nonzero_ratio.min() if p2a_nonzero_ratio.size else p2a_factors.min()),
                    "row_pathlength_reference_row_ratio_max": float(p2a_nonzero_ratio.max() if p2a_nonzero_ratio.size else p2a_factors.max()),
                    "row_pathlength_reference_row_mean_ratio_min": float(p2a_row_mean_ratio.min()),
                    "row_pathlength_reference_row_mean_ratio_max": float(p2a_row_mean_ratio.max()),
                    "row_pathlength_reference_min_before_attenuation": float(p2a_min_before_attenuation),
                    "row_pathlength_reference_max_before_attenuation": float(p2a_max_before_attenuation),
                    "row_pathlength_reference_min_after_attenuation": float(p2a_min_after_attenuation),
                    "row_pathlength_reference_max_after_attenuation": float(p2a_max_after_attenuation),
                    "row_pathlength_reference_guard_clip_fraction": 0.0,
                    "row_pathlength_reference_route_key": ("_p2a_diesel_row_pathlength_reference_route"),
                }
            )
        readout_space_transform = rule.get("readout_space_transform")
        if readout_space_transform is not None:
            # R5 readout-space remediation: keep the full R4c absorbance
            # pipeline above and only swap the final readout space. The
            # absorbance X_out is non-negative (clipped to [clip_low, ...] with
            # clip_low >= 0 in the R4c rule) so 10**-A in (0, 1] and
            # 1 - 10**-A in [0, 1).
            r5_clip = (0.0, 1.0)
            if readout_space_transform == "absorbance":
                # R5a: identity. X_out remains in absorbance space.
                pass
            elif readout_space_transform == "transmittance":
                X_out = np.clip(np.power(10.0, -X_out), r5_clip[0], r5_clip[1])
            elif readout_space_transform == "blank_referenced_intensity":
                X_out = np.clip(1.0 - np.power(10.0, -X_out), r5_clip[0], r5_clip[1])
            else:
                msg = f"Unknown readout_space_transform {readout_space_transform!r}"
                raise ValueError(msg)
            transform_params["readout_space_transform"] = readout_space_transform
            transform_params["readout_space_transform_clip"] = [
                float(r5_clip[0]),
                float(r5_clip[1]),
            ]
    elif rule.get("spectra_rule") == "milk_emulsion_scatter_inverse_transflectance_readout":
        route = _r2m_milk_route(record)
        if route is None:
            msg = "R2m milk readout requires explicit bench-only MILK route provenance; route was missing or non-compliant"
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
        dynamic_key = "fullrange_detector_dynamic_range" if variant == "fullrange" else "detector_dynamic_range"
        dynamic_low, dynamic_high = rule[dynamic_key]
        detector_dynamics = np.asarray(
            rng.uniform(dynamic_low, dynamic_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        clip_key = "fullrange_output_clip_intensity" if variant == "fullrange" else "output_clip_intensity"
        clip_low, clip_high = rule[clip_key]
        wl_ref = float(wl.min()) if wl.size else 1100.0
        wl_ref = wl_ref if wl_ref > 0.0 else 1100.0
        scatter_profile = np.power(
            np.clip(wl[None, :] / wl_ref, 1e-6, None),
            -scatter_exponents[:, None],
        )
        effective_absorbance = np.clip(X_smoothed, 0.0, None) * path_factors[:, None]
        X_out = detector_offsets[:, None] + detector_dynamics[:, None] * scatter_profile * np.power(10.0, -effective_absorbance)
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update(
            {
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
                "milk_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "milk_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                "output_clip_intensity": [float(clip_low), float(clip_high)],
            }
        )
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
        transform_params.update(
            {
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
            }
        )
    elif rule.get("spectra_rule") == "fermented_beer_turbid_cuvette_absorbance_readout":
        route = _r2o_beer_route(record)
        if route is None:
            msg = "R2o beer readout requires explicit bench-only BEER route provenance; route was missing or non-compliant"
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
        X_out = haze_baselines[:, None] + X_smoothed * path_factors[:, None] + haze_slopes[:, None] * shortwave_haze[None, :] + carbonation[:, None] * carbonation_profile[None, :]
        clip_low, clip_high = rule["output_clip_absorbance"]
        X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update(
            {
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
                "beer_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "beer_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
            }
        )
    elif rule.get("spectra_rule") == "phosphorus_mineral_fertilizer_albedo_residual_readout":
        route = _r2p_phosphorus_route(record)
        if route is None:
            msg = "R2p phosphorus readout requires explicit bench-only PHOSPHORUS route provenance; route was missing or non-compliant"
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
        transform_params.update(
            {
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
                "phosphorus_readout_route_marker": str(route.get("route_marker", "unknown")),
                "phosphorus_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "phosphorus_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "phosphorus_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
            }
        )
    elif rule.get("spectra_rule") in {
        "corn_powder_albedo_baseline_smoothing_readout",
        "corn_powder_albedo_path_dispersion_smoothing_readout",
    }:
        is_r3b_readout = rule.get("spectra_rule") == "corn_powder_albedo_path_dispersion_smoothing_readout"
        route = _r3b_corn_route(record) if is_r3b_readout else _r3a_corn_route(record)
        if route is None:
            profile_label = "R3b" if is_r3b_readout else "R3a"
            msg = f"{profile_label} CORN readout requires explicit bench-only CORN route provenance; route was missing or non-compliant"
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
        scatter_slopes = np.asarray(
            rng.uniform(slope_low, slope_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        moisture_low, moisture_high = rule["moisture_band_absorbance_range"]
        moisture_bands = np.asarray(
            rng.uniform(moisture_low, moisture_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        starch_low, starch_high = rule["starch_band_absorbance_range"]
        starch_bands = np.asarray(
            rng.uniform(starch_low, starch_high, size=X_smoothed.shape[0]),
            dtype=float,
        )
        if wl.size > 1:
            wl_span = max(float(wl.max() - wl.min()), 1.0)
            wl_norm = (wl - float(wl.mean())) / wl_span
        else:
            wl_norm = np.zeros_like(wl, dtype=float)

        def _centered_band(
            centers_widths: tuple[tuple[float, float], ...],
        ) -> np.ndarray:
            profile = np.zeros_like(wl, dtype=float)
            for center, width in centers_widths:
                profile += np.exp(-0.5 * ((wl - center) / width) ** 2)
            if profile.size and float(profile.max()) > 0.0:
                profile = profile / float(profile.max())
            return profile - float(profile.mean()) if profile.size else profile

        moisture_profile = _centered_band(((1450.0, 62.0), (1940.0, 82.0)))
        starch_profile = _centered_band(((1210.0, 50.0), (2100.0, 92.0)))
        centered_residual = X_smoothed - np.mean(X_smoothed, axis=1, keepdims=True)
        X_out = (
            baselines[:, None]
            + residual_scales[:, None] * centered_residual
            + scatter_slopes[:, None] * wl_norm[None, :]
            + moisture_bands[:, None] * moisture_profile[None, :]
            + starch_bands[:, None] * starch_profile[None, :]
        )
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update(
            {
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
                "moisture_band_absorbance_range": [
                    float(moisture_low),
                    float(moisture_high),
                ],
                "moisture_band_absorbance_min": float(moisture_bands.min()),
                "moisture_band_absorbance_max": float(moisture_bands.max()),
                "starch_band_absorbance_range": [float(starch_low), float(starch_high)],
                "starch_band_absorbance_min": float(starch_bands.min()),
                "starch_band_absorbance_max": float(starch_bands.max()),
                "moisture_band_centers_nm": [1450.0, 1940.0],
                "starch_band_centers_nm": [1210.0, 2100.0],
                "centered_residual_readout": ("corn_powder_albedo_path_dispersion_centered_residual" if is_r3b_readout else "corn_powder_albedo_baseline_centered_residual"),
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
                "corn_readout_route_source": str(route.get("source", "unknown")),
                "corn_readout_route_marker": str(route.get("route_marker", "unknown")),
                "corn_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "corn_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "corn_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
            }
        )
    elif rule.get("spectra_rule") == "lucas_ph_organic_humic_albedo_oh_readout":
        route = _r2q_lucas_ph_organic_route(record)
        if route is None:
            msg = "R2q LUCAS pH Organic readout requires explicit bench-only LUCAS pH Organic route provenance; route was missing or non-compliant"
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
        X_out = baselines[:, None] + residual_scales[:, None] * X_smoothed + humic_slopes[:, None] * shortwave_humic[None, :] + oh_amplitudes[:, None] * oh_profile[None, :]
        clip_low, clip_high = rule["output_clip_absorbance"]
        if clip_high is None:
            X_out = np.clip(X_out, float(clip_low), None)
        else:
            X_out = np.clip(X_out, float(clip_low), float(clip_high))
        transform_params.update(
            {
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
                "lucas_ph_organic_readout_route_source": str(route.get("source", "unknown")),
                "lucas_ph_organic_readout_route_marker": str(route.get("route_marker", "unknown")),
                "lucas_ph_organic_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "lucas_ph_organic_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "lucas_ph_organic_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
            }
        )
    elif rule.get("spectra_rule") in {
        "dried_manure_heterogeneous_scatter_patch_readout",
        "dried_manure_bounded_centered_scatter_readout",
        "dried_manure_balanced_centered_scatter_readout",
        "dried_manure_albedo_variance_centered_scatter_readout",
        "dried_manure_coarse_albedo_dispersion_centered_readout",
        "dried_manure_soft_low_frequency_albedo_dispersion_centered_readout",
        "dried_manure_compositional_heterogeneity_centered_readout",
    }:
        route = _r2n_manure21_route(record)
        if route is None:
            msg = "R2t/R2u/R2v/R2w/R2x/R2y/R2z manure heterogeneity readout requires explicit bench-only MANURE21 route provenance; route was missing or non-compliant"
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
            if rule.get("readout_centering_grid") == "uniform_wavenumber" and float(center_low) > 0.0 and float(center_high) > float(center_low):
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
        mineral_profile = _band_profile(((1410.0, 50.0), (2200.0, 74.0), (2340.0, 70.0)))
        residual_center = np.asarray(
            [np.interp(center_eval_grid, wl, row).mean() for row in X_smoothed],
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
        transform_params.update(
            {
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
                "readout_centering_range_nm": (None if centering_range is None else [float(center_low), float(center_high)]),
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
                "manure21_readout_route_marker": str(route.get("route_marker", "unknown")),
                "manure21_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                "manure21_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                "manure21_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
            }
        )
        if "composition_alpha_concentration_scale" in rule:
            transform_params.update(
                {
                    "composition_heterogeneity": rule["composition_heterogeneity"],
                    "composition_alpha_concentration_scale": float(rule["composition_alpha_concentration_scale"]),
                }
            )
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

        transform_params.update(
            {
                "spectra_rule": rule["spectra_rule"],
                "spectra_source": rule["spectra_source"],
                "path_factor_range": [float(low), float(high)],
                "path_factor_min": float(factors.min()),
                "path_factor_max": float(factors.max()),
            }
        )
        if additive_baseline_range is not None:
            transform_params.update(
                {
                    "additive_baseline_range": [float(offset_low), float(offset_high)],
                    "additive_baseline_min": float(offsets.min()),
                    "additive_baseline_max": float(offsets.max()),
                }
            )
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
        if rule.get("spectra_rule") == "dried_manure_organic_mineral_albedo_scatter_readout":
            route = _r2n_manure21_route(record)
            if route is None:
                msg = "R2n manure readout requires explicit bench-only MANURE21 route provenance; route was missing or non-compliant"
                raise ValueError(msg)
            transform_params.update(
                {
                    "manure21_readout_route_source": str(route.get("source", "unknown")),
                    "manure21_readout_route_marker": str(route.get("route_marker", "unknown")),
                    "manure21_readout_route_non_oracle": bool(route.get("non_oracle", False)),
                    "manure21_readout_route_real_stat_capture": bool(route.get("real_stat_capture", True)),
                    "manure21_readout_route_thresholds_modified": bool(route.get("thresholds_modified", True)),
                }
            )
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
