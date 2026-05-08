"""Phase 0 baseline benchmark runner — Ridge, PLS, nicon, decon.

Usage::

    PYTHONPATH=bench/nicon_v2:bench/AOM_v0:bench/AOM_v0/Ridge \\
      python bench/nicon_v2/benchmarks/run_baseline_benchmark.py \\
        --workspace bench/nicon_v2/benchmark_runs/smoke \\
        --cohort smoke --variants smoke --seed 0

Resumability: rows already present with ``status="OK"`` are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Make the package importable when run as a script from the repo root.
_HERE = Path(__file__).resolve()
_BENCH_ROOT = _HERE.parent.parent          # bench/nicon_v2
for path in (str(_BENCH_ROOT), str(_BENCH_ROOT.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch  # noqa: E402  (after sys.path setup)

from nicon_v2 import CODE_VERSION  # noqa: E402
from nicon_v2.datasets import (  # noqa: E402
    DatasetSpec,
    EXTENDED_SMOKE_DATASETS,
    SMOKE_DATASETS,
    load_cohort_manifest,
    load_dataset,
)
from nicon_v2.metrics import bias as bias_metric  # noqa: E402
from nicon_v2.metrics import mae, r2, relative_rmsep, rmse  # noqa: E402
from nicon_v2.models.baseline import (  # noqa: E402
    PLSBaseline,
    RidgeBaseline,
    build_decon_torch,
    build_nicon_torch,
    count_parameters,
    cuda_peak_mb,
)
from nicon_v2.models.searched_baseline import SearchedPLS, SearchedRidge  # noqa: E402
from nicon_v2.augmentation import (  # noqa: E402
    AugmentationPlan,
    BjerrumConfig,
    CMixupConfig,
)
from nicon_v2.models.v1a_minimal_repair import (  # noqa: E402
    build_nicon_v1a,
    build_nicon_v1a_activation_only,
    build_nicon_v1a_head_only,
)
from nicon_v2.models.v1b_concat_aug import build_nicon_v1b  # noqa: E402
from nicon_v2.models.v1c_gap_backbone import build_nicon_v1c  # noqa: E402
from nicon_v2.models.v2_aom_cnn import build_nicon_v2a  # noqa: E402
from nicon_v2.models.stacking import StackedRegressor, StackingConfig  # noqa: E402
from nicon_v2.training import (  # noqa: E402
    StandardXProcessor,
    StandardYProcessor,
    TrainConfig,
    pick_device,
    predict_torch_regressor,
    set_global_seed,
    train_torch_regressor,
)


RESULT_COLUMNS = [
    "dataset_group",
    "dataset",
    "task",
    "n_train",
    "n_test",
    "n_features",
    "variant",
    "model_version",
    "seed",
    "cv_fold",
    "cv_protocol",
    "status",
    "error_message",
    "rmsep",
    "mae",
    "r2",
    "bias",
    "ref_rmse_pls",
    "ref_rmse_paper_ridge",
    "ref_rmse_tabpfn_raw",
    "ref_rmse_tabpfn_opt",
    "ref_rmse_cnn",
    "ref_rmse_catboost",
    "ref_rmse_aom_ridge_curated_best",
    "relative_rmsep_vs_pls",
    "relative_rmsep_vs_paper_ridge",
    "relative_rmsep_vs_tabpfn_raw",
    "relative_rmsep_vs_tabpfn_opt",
    "relative_rmsep_vs_cnn",
    "relative_rmsep_vs_aom_ridge_curated_best",
    "fit_time_s",
    "predict_time_s",
    "total_params",
    "peak_vram_mb",
    "hyperparams_json",
    "python_version",
    "torch_version",
    "cuda_version",
    "git_sha",
    "host",
    # R21 — shrinkage CV diagnostics. NaN / empty for variants without shrinkage.
    "shrinkage_s_star",
    "catastrophic",
]


@dataclass
class Variant:
    label: str
    family: str        # "ridge" | "pls" | "nicon" | "decon"
    extra: dict = field(default_factory=dict)


SMOKE_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NICON-baseline", family="nicon"),
    Variant("DECON-baseline", family="decon"),
)

PHASE1A_VARIANTS: tuple[Variant, ...] = SMOKE_VARIANTS + (
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("NiconV1a-activation-only", family="nicon_v1a_act"),
    Variant("NiconV1a-baseline", family="nicon_v1a"),
)

# Phase 1b ablation: V1a-head-only baseline + concat-deriv only + concat+Bjerrum + concat+Bjerrum+C-Mixup.
PHASE1B_VARIANTS: tuple[Variant, ...] = (
    # Phase 1a accepted control:
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Phase 1b ablation cells:
    Variant("NiconV1b-concat-only",     family="nicon_v1b", extra={"bjerrum": False, "cmixup": False, "vanilla_mixup": False}),
    Variant("NiconV1b-concat-bjerrum",  family="nicon_v1b", extra={"bjerrum": True,  "cmixup": False, "vanilla_mixup": False}),
    Variant("NiconV1b-concat-cmixup",   family="nicon_v1b", extra={"bjerrum": True,  "cmixup": True,  "vanilla_mixup": False}),
    Variant("NiconV1b-concat-mixup",    family="nicon_v1b", extra={"bjerrum": True,  "cmixup": False, "vanilla_mixup": True}),
)

# Phase 5 / H12 stacking: V1c + PLS via Ridge meta.
PHASE_STACK_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": False, "vanilla_mixup": False}),
    Variant("Stack-Ridge-PLS",        family="stack",
            extra={"_base": ("ridge", "pls")}),
    Variant("Stack-Ridge-PLS-V1c",    family="stack",
            extra={"_base": ("ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
    Variant("Stack-Ridge-PLS-V1aHead",family="stack",
            extra={"_base": ("ridge", "pls", "v1a_head")}),
)

# Phase 5b: stacks using AOM-Ridge (84 % wins vs paper Ridge) as a base / instead of Ridge.
PHASE_STACK_AOM_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("AOMRidge-base", family="stack", extra={"_base": ("aom_ridge",)}),
    Variant("Stack-AOMRidge-PLS",     family="stack", extra={"_base": ("aom_ridge", "pls")}),
    Variant("Stack-AOMRidge-PLS-V1c", family="stack",
            extra={"_base": ("aom_ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
    Variant("Stack-AOMRidge-Ridge-PLS-V1c", family="stack",
            extra={"_base": ("aom_ridge", "ridge", "pls", "v1c"), "_cnn_use_bjerrum": True}),
)

# Phase 2 / round 5: AOM-superblock CNN (NiconV2A) ablation set.
PHASE_V2A_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NICON-baseline", family="nicon"),
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True}),
    # V2A — bare strict-linear AOM bank, frozen ops:
    Variant("V2A-compact-frozen",  family="nicon_v2a", extra={"bank": "compact",  "trainable_ops": False}),
    Variant("V2A-extended-frozen", family="nicon_v2a", extra={"bank": "extended", "trainable_ops": False}),
    # V2A + Bjerrum aug (pre-branch, per Codex F5):
    Variant("V2A-extended-frozen-bjerrum", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": False, "bjerrum": True}),
    # V2A + CNN-only extras (SNV/MSC/Gaussian) + Bjerrum aug:
    Variant("V2A-full-frozen-bjerrum", family="nicon_v2a",
            extra={"bank": "full", "trainable_ops": False, "bjerrum": True}),
    # V2B — learnable operators with L2-from-init regulariser:
    Variant("V2B-extended-trainable", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True}),
)


# Round 6 (Codex F1 + F2 + F7): V2B with reg fix + V2C/V2D variants on representative cohort.
PHASE_V2_R6_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NICON-baseline", family="nicon"),
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True}),
    # V2B baseline (ops trainable, reg λ=1e-3, F1 fix now active in train loop)
    Variant("V2B-extended-trainable",  family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True,  "operator_reg_lambda": 1e-3, "bjerrum": True}),
    # V2B-noreg ablation: confirm whether the F1 fix changes behaviour
    Variant("V2B-extended-no-reg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True,  "operator_reg_lambda": 0.0,  "bjerrum": True}),
    # V2C — V2B + branch-level SE (Codex F2)
    Variant("V2C-branchSE",        family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True,  "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True}),
    # V2D — V2B + dilated trunk (1, 2, 4)  (Codex F7-rank-1)
    Variant("V2D-dilated",         family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True,  "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "block_type": "dilated", "dilations": (1, 2, 4)}),
    # V2E — V2C + V2D combined
    Variant("V2E-branchSE-dilated", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True,  "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True, "block_type": "dilated", "dilations": (1, 2, 4)}),
)


# Round 7 — based on round 6 findings, drop the reg term and freeze O(p^2)
# matrix operators while keeping convolutional AOM kernels trainable.
PHASE_V2_R7_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round 7 production candidate: V2C branch-SE with no L2-to-init operator pull.
    Variant("V2C-branchSE-no-reg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True}),
    # V2G: convolutional AOM kernels trainable, Detrend/Whittaker matrices frozen.
    Variant("V2G-FrozenMatrix", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "matrix_trainable_ops": False,
                   "operator_reg_lambda": 0.0, "bjerrum": True, "branch_se": True}),
    # Round 6 controls:
    Variant("V2C-branchSE", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True}),
    Variant("V2E-branchSE-dilated", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True, "block_type": "dilated", "dilations": (1, 2, 4)}),
    # round 7 ablations:
    # (a) drop the L2-from-init regulariser entirely (Codex Q1 option):
    Variant("V2C-noreg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True}),
    Variant("V2E-noreg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "block_type": "dilated", "dilations": (1, 2, 4)}),
    # (b) wider trunk: 64/96/128 instead of 32/64/96:
    Variant("V2C-wide", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True, "trunk_channels": (64, 96, 128), "trunk_kernels": (7, 5, 3)}),
    # (c) deeper trunk: 4 blocks for n_train ≥ 500
    Variant("V2C-deeper", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 1e-3, "bjerrum": True,
                   "branch_se": True, "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3)}),
)


# Round 8 — V2H-LowRankMatrix variants, plus the round-7 winners as controls.
# Codex round 7 finding: trainable matrices add value but explode params; V2H
# uses A ≈ U V^T with rank=8/16/32 to keep flexibility at O(p·k) instead of O(p²).
PHASE_V2_R8_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-7 winners as controls:
    Variant("V2C-noreg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True}),
    Variant("V2E-noreg", family="nicon_v2a",
            extra={"bank": "extended", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "block_type": "dilated", "dilations": (1, 2, 4)}),
    # V2H — low-rank trainable matrix ops (rank=8/16/32):
    Variant("V2H-lowrank-r8",  family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 8}),
    Variant("V2H-lowrank-r16", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 16}),
    Variant("V2H-lowrank-r32", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32}),
    # V2H + dilated trunk:
    Variant("V2H-lowrank-dilated", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "block_type": "dilated", "dilations": (1, 2, 4)}),
)


# Round 9 — V2J stacking + V2I ensemble (Codex round 8 ranking V2J > V2I).
# V2J: V2H-r32 OOF stacked with AOM-Ridge and PLS through a Ridge meta.
# V2I: 3-seed V2H-r32 ensemble averaged at predict time.
PHASE_V2_R9_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # round-8 production controls:
    Variant("V2H-lowrank-r32", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32}),
    Variant("V2H-lowrank-r16", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 16}),
    # V2J — V2H-r32 stacked with AOM-Ridge + PLS (Ridge meta):
    Variant("V2J-Stack-V2H-AOMRidge-PLS", family="stack",
            extra={"_base": ("v2h", "aom_ridge", "pls")}),
    Variant("V2J-Stack-V2H-AOMRidge",     family="stack",
            extra={"_base": ("v2h", "aom_ridge")}),
    Variant("V2J-Stack-V2H-PLS",          family="stack",
            extra={"_base": ("v2h", "pls")}),
)


# Round 10 — pure-CNN ablation (user direction "stop stacking, focus on the deep model").
# Builds on V2H-lowrank-r32 (round 8 production CNN). All variants are stand-alone CNNs.
PHASE_V2_R10_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-8 production CNN (control):
    Variant("V2H-lowrank-r32", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32}),
    # V2L — learnable RMS scale per branch (replaces fixed first-batch fit):
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # V2O — multi-kernel stem (parallel 3,5,7,9):
    Variant("V2O-multikernelStem", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "multi_kernel_stem": True,
                   "multi_kernel_kernels": (3, 5, 7, 9)}),
    # V2P — wavelength self-attention head (replaces GAP+Linear):
    Variant("V2P-attnHead", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "head_type": "attn", "attn_heads": 4}),
    # V2L + V2O combined:
    Variant("V2LO-rms+multikernel", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True, "multi_kernel_stem": True}),
    # V2L + V2P combined:
    Variant("V2LP-rms+attn", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True, "head_type": "attn"}),
    # V2OP — multi-kernel stem + attention head:
    Variant("V2OP-multikernel+attn", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "multi_kernel_stem": True, "head_type": "attn"}),
)


# Round 11 — Codex round-9 priorities (V2M deeper + V2L mechanism diagnostic).
# V2L is the round-10 production CNN. Round 11 keeps V2L's learnable RMS as
# the base and adds (a) V2M deeper trunk, (b) V2L mechanism diagnostics to
# disentangle "branch calibration" from "free-param optimisation".
PHASE_V2_R11_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-10 production CNN (control):
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # Frozen-RMS V2H control (Codex diagnostic 1: inverse-RMS frozen):
    Variant("V2H-frozenRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": False}),
    # V2L diagnostic — tied global learnable scale (single shared param across branches):
    Variant("V2L-tiedGlobalRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True, "tied_global_rms": True}),
    # V2L diagnostic — per-branch learnable scale init=1 (no inverse-RMS data init):
    Variant("V2L-perbranchInit1", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True, "rms_init_mode": "unit"}),
    # V2M-DeeperConditional — 4-block trunk (run unconditionally on representative;
    # the runner could later gate this by p ≥ 1024 ∧ n_train ≥ 500):
    Variant("V2M-deeper", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3)}),
    # V2M + dilated trunk to test multi-scale on the deeper backbone:
    Variant("V2M-deeper-dilated", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "block_type": "dilated", "dilations": (1, 2, 4)}),
)


# Round 12 — diversification beyond pure CNN (Codex round-9 ceiling diagnosis).
# Includes V2L + V2M-deeper as the round-11 controls, plus three architectural
# extensions: V3 (Transformer trunk), V6 (knowledge distillation from AOM-PLS),
# V7 (test-time Bjerrum augmentation). All keep a single-model predict path.
PHASE_V2_R12_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-11 production CNN control:
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # Round-11 best (V2M-deeper) — second control:
    Variant("V2M-deeper", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3)}),
    # V3 — AOM-Transformer trunk (1 conv block + 2-layer transformer encoder):
    Variant("V3-AOMTransformer", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_type": "hybrid_transformer",
                   "transformer_d_model": 64, "transformer_heads": 4,
                   "transformer_layers": 2, "transformer_ff": 128}),
    # V6 — knowledge distillation from AOM-PLS (teacher fitted per fold):
    Variant("V6-Distill-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "distill_lambda": 0.3, "distill_teacher": "aompls_compact"}),
    # V6b — same distillation with the deeper V2M trunk:
    Variant("V6-Distill-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_compact"}),
    # V7 — test-time augmentation (K=5 Bjerrum copies averaged):
    Variant("V7-TTA-V2L", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "tta_k": 5, "tta_bjerrum": True}),
)


# Round 13 — Codex round-10 GO direction: stronger teachers + SWA training recipe.
# Round 12 showed marginal-positive distillation (V6, +0.5-0.9 % improvement vs V2L);
# Codex predicts V6b (extended bank) + SWA together can close 5 pp of the
# AOM-Ridge-best gap. NO-GO on new pure-arch variants per Codex round-10.
PHASE_V2_R13_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-12 controls (re-run for paired comparison on the same seed):
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V6-Distill-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_compact"}),
    # V6b — stronger teacher: AOM-PLS extended bank, max_components=25, cv=10:
    Variant("V6b-DistillExtended-V2L", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended"}),
    Variant("V6b-DistillExtended-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended"}),
    # SWA — pure SWA on V2L trunk (no distillation):
    Variant("V2L-SWA", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "use_swa": True, "swa_start_frac": 0.75}),
    # V6b + SWA — combine the two GO directions:
    Variant("V6b-DistillExtended-SWA-V2L", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended",
                   "use_swa": True, "swa_start_frac": 0.75}),
)


# Round 14 — multi-seed validation of round-13's directional 6/10 wins.
# Slimmed to 4 variants (Ridge, PLS, V2L control, V6b-V2M best). Run with
# `--seeds 1 2 3 4` to extend the round-13 seed-0 results into a 5-seed cohort
# (200 paired observations).
PHASE_V2_R14_MULTISEED: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V6b-DistillExtended-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended"}),
)


# Round 15 — LUCAS-pretrained backbone fine-tune.
# Last hypothesis: pretrained foundation-model transfer from LUCAS-SOC
# (~5000 spectra, log1p(SOC), V2A backbone). The pretrained checkpoint loads
# length-invariant params (conv kernels, branch SE, RMS norms, head Linear);
# length-dependent operators (LowRank Detrend/Whittaker U/V matrices, MSC
# means) are silently re-init at the target p.
LUCAS_CHECKPOINT_5K = "bench/nicon_v2/checkpoints/lucas_v2l_5k.pt"

PHASE_V2_R15_LUCAS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    # Round-12 production CNN control:
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # V2M-deeper control (matches the LUCAS-pretrained trunk for fair comparison):
    Variant("V2M-deeper", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3)}),
    # V2M with LUCAS-pretrained backbone (no distillation, just transfer):
    Variant("V2M-LucasPretrained", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "pretrained_path": LUCAS_CHECKPOINT_5K}),
    # V2M with LUCAS-pretrained backbone + V6b distillation (combined):
    Variant("V6b-LucasPretrained-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended",
                   "pretrained_path": LUCAS_CHECKPOINT_5K}),
)


# Round 17 — exhaustive NN ecosystem sweep (user direction, post-publication).
# Priority 1: B (heteroscedastic Student-t head), C (PLS-residual hybrid),
# H (KAN trunk), I (Conformer trunk). Priority 2: D, F, J, N.
PHASE_V2_R17_PRIORITY1: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # B — heteroscedastic Student-t head
    Variant("V2L-StudentT", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "head_out_dim": 2, "loss_type": "studentt", "studentt_df": 5.0}),
    Variant("V2L-Huber", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "loss_type": "huber", "huber_delta": 1.0}),
    # C — PLS-residual hybrid: CNN learns residuals over AOM-PLS / AOM-Ridge-PLS
    Variant("V2L-Residual-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended"}),
    Variant("V2L-Residual-AOMRidgePLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aomridgepls"}),
    # H — KAN trunk
    Variant("V2L-KAN", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_type": "kan", "transformer_d_model": 64, "transformer_layers": 2}),
    # I — Conformer trunk
    Variant("V2L-Conformer", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_type": "conformer", "transformer_d_model": 64, "transformer_heads": 4,
                   "transformer_layers": 2}),
    # C-boost variants — CNN gets AOM_pred as boost signal at the head; predicts y_true (no residual)
    Variant("V2L-Boost-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aompls_extended"}),
    Variant("V2L-Boost-AOMRidgePLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aomridgepls"}),
)


# Round 20 — final publication variant set with OOF residuals.
# After Codex round-13 review found in-sample-validation contamination,
# OOF residuals are now the default (`pls_residual_oof=True`). This is the
# clean variant set to ship.
PHASE_V2_R20_FINAL: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V2L-Residual-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended", "pls_residual_oof": True, "pls_oof_n_folds": 5}),
    Variant("V2L-Boost-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aompls_extended",
                   "pls_boost_oof": True, "pls_oof_n_folds": 5}),
)


# Round 21 — multiseed validation of V2L-Residual-AOMPLS with shrinkage CV.
# Adds inner-CV shrinkage selection (s ∈ {0, 0.25, 0.5, 0.75, 1.0}) on the
# val partition, with s=0 acting as a do-no-harm teacher-only fallback. The
# `catastrophic` flag is set when final test rmse exceeds the teacher-only
# rmse by more than `shrinkage_catastrophic_threshold` (default +50 %).
# 39 datasets × 5 seeds × 1 variant. Baselines reused from r20.
PHASE_V2_R21_MULTISEED: tuple[Variant, ...] = (
    Variant("V2L-Residual-AOMPLS-shrinkage", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended",
                   "pls_residual_oof": True, "pls_oof_n_folds": 5,
                   "shrinkage_cv": True,
                   "shrinkage_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
                   "shrinkage_catastrophic_threshold": 0.5}),
)


# Round 19 — leakage diagnostic and OOF residual fix (Codex round-13 finding).
# Compares the round-17/18 implementation (in-sample residuals; default
# `pls_residual_oof=False` here for the diagnostic) vs OOF residuals
# (`pls_residual_oof=True`, the new default in `_run_torch_cnn`).
PHASE_V2_R19_OOF_DIAGNOSTIC: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # R17/R18 reference (leaky teacher fit; explicit OOF off for fair compare):
    Variant("V2L-Residual-AOMPLS-leaky", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended", "pls_residual_oof": False}),
    # R19 OOF fix (default):
    Variant("V2L-Residual-AOMPLS-OOF", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended", "pls_residual_oof": True, "pls_oof_n_folds": 5}),
    # Same comparison for boost mode:
    Variant("V2L-Boost-AOMPLS-leaky", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aompls_extended", "pls_boost_oof": False}),
    Variant("V2L-Boost-AOMPLS-OOF", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aompls_extended", "pls_boost_oof": True, "pls_oof_n_folds": 5}),
)


# Round 18 — multi-seed validation of round-17's V2L-Residual-AOMPLS signal.
# Slimmed to 4 variants for tractability; 5 seeds × 4 × 10 = 200 paired obs.
PHASE_V2_R18_RESIDUAL_MULTISEED: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V2L-Residual-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "pls_residual_teacher": "aompls_extended"}),
    Variant("V2L-Boost-AOMPLS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "boost_signal_dim": 1, "pls_boost_teacher": "aompls_extended"}),
)


# Round 17 priority-2 — D (MoE head) and F (auxiliary PLS-scores multi-task).
PHASE_V2_R17_PRIORITY2: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    # D — Mixture-of-Experts head (K=2)
    Variant("V2L-MoE-K2", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "head_type": "moe", "moe_num_experts": 2}),
    Variant("V2L-MoE-K4", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "head_type": "moe", "moe_num_experts": 4}),
    # F — auxiliary multi-task with PLS scores
    Variant("V2L-AuxY-PLS5", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "aux_head_dim": 5, "aux_target_kind": "pls_scores",
                   "aux_n_components": 5, "aux_lambda": 0.3}),
    Variant("V2L-AuxY-PLS10", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "aux_head_dim": 10, "aux_target_kind": "pls_scores",
                   "aux_n_components": 10, "aux_lambda": 0.3}),
)


# Publication — V2L production CNN + key ablations on the curated cohort.
# After 16 rounds of architecture/training experiments, V2L-learnableRMS is
# the production CNN (multi-seed validated, ties or beats every other CNN
# variant). V6b-LUCAS-V2M is included as the LUCAS-pretrained ablation.
PHASE_PUBLICATION: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V6b-LucasPretrained-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended",
                   "pretrained_path": LUCAS_CHECKPOINT_5K}),
)


# Round 16 — multi-seed validation of round-15's V6b-LUCAS pretrained signal.
# Slimmed to 4 variants (Ridge, PLS, V2L, V6b-LUCAS-V2M; V2M-deeper as second
# CNN control). Run with `--seeds 1 2 3 4` to extend round-15 seed-0 results
# into a 5-seed cohort (250 paired observations).
PHASE_V2_R16_LUCAS_MULTISEED: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("V2L-learnableRMS", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True}),
    Variant("V2M-deeper", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3)}),
    Variant("V6b-LucasPretrained-V2M", family="nicon_v2a",
            extra={"bank": "extended_lowrank", "trainable_ops": True, "operator_reg_lambda": 0.0, "bjerrum": True,
                   "branch_se": True, "lowrank_rank": 32, "learnable_rms": True,
                   "trunk_channels": (32, 64, 96, 128), "trunk_kernels": (7, 5, 3, 3),
                   "distill_lambda": 0.3, "distill_teacher": "aompls_extended",
                   "pretrained_path": LUCAS_CHECKPOINT_5K}),
)


# Phase 1d: cartesian-preprocessing-search Ridge / PLS (mirror tabpfn paper baseline).
PHASE_SEARCHED_VARIANTS: tuple[Variant, ...] = (
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("SearchedRidge", family="searched_ridge"),
    Variant("SearchedPLS",   family="searched_pls"),
    Variant("Stack-SearchedRidge-SearchedPLS", family="stack",
            extra={"_base": ("searched_ridge", "searched_pls")}),
    Variant("Stack-SearchedRidge-SearchedPLS-V1c", family="stack",
            extra={"_base": ("searched_ridge", "searched_pls", "v1c"), "_cnn_use_bjerrum": True}),
)


# Phase 1c: GAP backbone alone, GAP+concat-deriv, GAP+concat+Bjerrum+C-Mixup, plus norm A/B (LayerNorm/BatchNorm/GroupNorm).
PHASE1C_VARIANTS: tuple[Variant, ...] = (
    # Anchor controls:
    Variant("Ridge-baseline", family="ridge"),
    Variant("PLS-baseline", family="pls"),
    Variant("NiconV1a-head-only", family="nicon_v1a_head"),
    Variant("NICON-baseline", family="nicon"),
    # Phase 1c GAP backbone — bare:
    Variant("NiconV1c-bare-LN", family="nicon_v1c", extra={"norm": "layer", "use_concat_derivatives": False}),
    Variant("NiconV1c-bare-BN", family="nicon_v1c", extra={"norm": "batch", "use_concat_derivatives": False}),
    Variant("NiconV1c-bare-GN", family="nicon_v1c", extra={"norm": "group", "use_concat_derivatives": False}),
    # GAP + concat-deriv (Phase 1b H5 lifted onto the new backbone):
    Variant("NiconV1c-concat", family="nicon_v1c", extra={"norm": "layer", "use_concat_derivatives": True}),
    # GAP + concat + Bjerrum:
    Variant("NiconV1c-concat-bjerrum", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": False, "vanilla_mixup": False}),
    # GAP + concat + Bjerrum + C-Mixup (the kitchen sink):
    Variant("NiconV1c-concat-bjerrum-cmixup", family="nicon_v1c",
            extra={"norm": "layer", "use_concat_derivatives": True, "bjerrum": True, "cmixup": True, "vanilla_mixup": False}),
)


# ---------------------------------------------------------------------------
# Result storage helpers
# ---------------------------------------------------------------------------


def write_predictions_parquet(
    workspace: Path,
    variant: str,
    dataset: str,
    seed: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Path | None:
    """Write per-sample predictions to a parquet (Codex F14).

    Falls back silently to CSV if pyarrow is missing.
    """
    pred_dir = workspace / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    safe_dataset = dataset.replace("/", "_")
    safe_variant = variant.replace("/", "_")
    parquet_path = pred_dir / f"{safe_variant}__{safe_dataset}__seed{seed}.parquet"
    df = pd.DataFrame(
        {
            "sample_id": np.arange(len(y_true), dtype=np.int64),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "fold": np.full(len(y_true), -1, dtype=np.int64),
            "seed": np.full(len(y_true), seed, dtype=np.int64),
            "variant": np.full(len(y_true), variant, dtype=object),
            "dataset": np.full(len(y_true), dataset, dtype=object),
        }
    )
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ValueError):
        # pyarrow may be missing; fall back to CSV with the same path stem.
        csv_path = parquet_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        return csv_path


def existing_done_rows(csv_path: Path) -> set[tuple[str, str, int]]:
    """Return the set of ``(dataset, variant, seed)`` rows already finished with status=OK."""
    if not csv_path.is_file():
        return set()
    done: set[tuple[str, str, int]] = set()
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("status", "")).upper() == "OK":
                try:
                    done.add(
                        (
                            str(row["dataset"]),
                            str(row["variant"]),
                            int(row["seed"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    return done


def open_result_writer(csv_path: Path) -> tuple[csv.DictWriter, "object"]:
    """Open the CSV in append mode, writing the header if file does not exist."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file() or csv_path.stat().st_size == 0
    fh = csv_path.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
        fh.flush()
    return writer, fh


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------


def _run_ridge(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = RidgeBaseline(seed=seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams, total_params=spec.n_features, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_stack(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int, extra: dict | None) -> dict:
    set_global_seed(seed)
    extra = extra or {}
    base = tuple(extra.get("_base", ("ridge", "pls", "v1c")))
    cfg = StackingConfig(
        base_learners=base,
        seed=seed,
        cnn_use_bjerrum=bool(extra.get("_cnn_use_bjerrum", True)),
    )
    model = StackedRegressor(cfg=cfg)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(
        spec, pred, y_test, fit_time, pred_time, model.hyperparams,
        total_params=len(base), peak_vram_mb=cuda_peak_mb() if torch.cuda.is_available() else float("nan"),
    )
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_pls(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = PLSBaseline(seed=seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    t0 = time.time()
    pred = model.predict(X_test)
    pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams, total_params=model.selected_n_components_, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_searched_ridge(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = SearchedRidge(seed=seed)
    t0 = time.time(); model.fit(X_train, y_train); fit_time = time.time() - t0
    t0 = time.time(); pred = model.predict(X_test); pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams,
                      total_params=spec.n_features, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _run_searched_pls(spec: DatasetSpec, X_train, y_train, X_test, y_test, seed: int) -> dict:
    set_global_seed(seed)
    model = SearchedPLS(seed=seed)
    t0 = time.time(); model.fit(X_train, y_train); fit_time = time.time() - t0
    t0 = time.time(); pred = model.predict(X_test); pred_time = time.time() - t0
    row = _result_row(spec, pred, y_test, fit_time, pred_time, model.hyperparams,
                      total_params=model.selected_n_components_, peak_vram_mb=float("nan"))
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _fit_distill_teacher_predictions(
    teacher_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> np.ndarray:
    """V6 — fit a fold-local teacher on (X_train, y_train) and return predictions on X_train.

    Predictions are in the *original y scale*; the caller standardises them
    via the student's StandardYProcessor before passing to the trainer.

    Available teachers (in order of expected strength on NIRS regression):
    * ``"pls"`` — sklearn PLSRegression, ~15 components, no operator search
    * ``"aompls_compact"`` — AOM-PLS (default bank ~77 ops), ``max_components=15``
    * ``"aompls_extended"`` — AOM-PLS (extended bank ~80 ops), ``max_components=25``
    * ``"popplsr_extended"`` — POP-PLS (per-component selection on extended bank,
      ``max_components=25``); usually the stronger PLS-family teacher
    """
    if teacher_name == "aompls_compact":
        from aompls.estimators import AOMPLSRegressor  # type: ignore
        teacher = AOMPLSRegressor(
            n_components="auto", max_components=15, engine="simpls_covariance",
            operator_bank="default", criterion="cv", cv=5, random_state=seed,
        )
        teacher.fit(X_train, y_train)
        return np.asarray(teacher.predict(X_train), dtype=float).ravel()
    if teacher_name == "aompls_extended":
        from aompls.estimators import AOMPLSRegressor  # type: ignore
        teacher = AOMPLSRegressor(
            n_components="auto", max_components=20, engine="simpls_covariance",
            operator_bank="extended", criterion="cv", cv=5, random_state=seed,
        )
        teacher.fit(X_train, y_train)
        return np.asarray(teacher.predict(X_train), dtype=float).ravel()
    if teacher_name == "popplsr_extended":
        from aompls.estimators import POPPLSRegressor  # type: ignore
        teacher = POPPLSRegressor(
            n_components="auto", max_components=20, engine="simpls_covariance",
            operator_bank="extended", criterion="cv", cv=5, random_state=seed,
        )
        teacher.fit(X_train, y_train)
        return np.asarray(teacher.predict(X_train), dtype=float).ravel()
    if teacher_name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        teacher = PLSRegression(n_components=min(15, X_train.shape[0] - 1, X_train.shape[1]))
        teacher.fit(X_train, y_train)
        return np.asarray(teacher.predict(X_train), dtype=float).ravel()
    raise ValueError(f"unknown distill_teacher {teacher_name!r}; expected "
                     f"'pls' | 'aompls_compact' | 'aompls_extended' | 'popplsr_extended'")


def _build_pls_residual_teacher_oof(
    name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int, n_folds: int = 5,
) -> tuple[object, np.ndarray]:
    """R19 — out-of-fold teacher predictions for clean residual / boost training.

    Codex round-13 review found that the teacher was previously fit on the
    full outer ``(X_train, y_train)``, so the residual `r = y - z(X)` for
    every train row was in-sample for the teacher. The CNN's val loss was
    therefore computed against artificially small residuals, contaminating
    early stopping (no test-set leakage, but val curve quality compromised).

    This OOF variant fits the teacher in a 5-fold CV: each row's `z[i]` is
    predicted by a teacher fit on the OTHER 4 folds. Plus a final teacher
    fit on all of `X_train` for test-time predictions.

    Returns ``(final_teacher, z_train_oof)`` where ``final_teacher.predict``
    can be called on `X_test` to obtain test-time AOM predictions, and
    ``z_train_oof`` is a (n,) array of out-of-fold predictions for the
    training set (used to compute `r = y_train - z_train_oof`).
    """
    from sklearn.model_selection import KFold
    n = X_train.shape[0]
    z_oof = np.zeros(n, dtype=float)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_idx, va_idx in kf.split(np.arange(n)):
        teacher_k = _build_pls_residual_teacher(name, X_train[tr_idx], y_train[tr_idx], seed)
        z_oof[va_idx] = np.asarray(teacher_k.predict(X_train[va_idx]), dtype=float).ravel()
    final_teacher = _build_pls_residual_teacher(name, X_train, y_train, seed)
    return final_teacher, z_oof


def _build_pls_residual_teacher(name: str, X_train: np.ndarray, y_train: np.ndarray, seed: int):
    """R17 C — return a fitted teacher object exposing ``.predict(X) -> np.ndarray``.

    Teachers:
    * ``"aompls_extended"`` — AOM-PLS extended bank, max_components=20, cv=5.
    * ``"aomridgepls"`` — AOM-Ridge-PLS with the production CV wrapper.
    * ``"pls"`` — sklearn PLSRegression with auto component selection.
    """
    if name == "aompls_extended":
        from aompls.estimators import AOMPLSRegressor  # type: ignore
        teacher = AOMPLSRegressor(
            n_components="auto", max_components=20, engine="simpls_covariance",
            operator_bank="extended", criterion="cv", cv=5, random_state=seed,
        )
        teacher.fit(X_train, y_train)
        return teacher
    if name == "aomridgepls":
        from aomridge.aom_ridge_pls import AOMRidgePLSCV  # type: ignore
        teacher = AOMRidgePLSCV(random_state=seed)
        teacher.fit(X_train, y_train)
        return teacher
    if name == "pls":
        from sklearn.cross_decomposition import PLSRegression
        teacher = PLSRegression(n_components=min(15, X_train.shape[0] - 1, X_train.shape[1]))
        teacher.fit(X_train, y_train)
        return teacher
    raise ValueError(f"unknown pls_residual_teacher {name!r}; expected aompls_extended | aomridgepls | pls")


def _predict_with_tta(
    model: torch.nn.Module,
    X_test_s: np.ndarray,
    device: torch.device,
    tta_k: int,
    seed: int,
) -> np.ndarray:
    """V7 — average predictions over K Bjerrum-augmented copies of X_test_s.

    The augmentation amplitudes mirror the train-time defaults so test-time
    augmentation explores the same neighbourhood the model trained on.
    """
    from nicon_v2.augmentation import BjerrumAugmenter, BjerrumConfig, _per_dataset_amplitude

    n, p = X_test_s.shape
    sigma_unit = _per_dataset_amplitude(X_test_s)
    bjer = BjerrumAugmenter(BjerrumConfig(enabled=True), sigma_unit=sigma_unit,
                            seq_len=p, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 100)
    X_t = torch.from_numpy(X_test_s.reshape(n, 1, p)).float().to(device)

    model.eval()
    accum = np.zeros(n, dtype=float)
    with torch.no_grad():
        # Always include the un-augmented forward pass as one of the K copies.
        accum += predict_torch_regressor(model, X_test_s, device=device)
        for _ in range(tta_k - 1):
            x_aug = bjer(X_t, gen).detach().cpu().numpy().reshape(n, p)
            accum += predict_torch_regressor(model, x_aug, device=device)
    return accum / float(tta_k)


def _build_aug_plan(extra: dict) -> AugmentationPlan | None:
    """Translate variant.extra options into an AugmentationPlan, or None if no aug."""
    bjerrum_enabled = bool(extra.get("bjerrum", False))
    cmixup_enabled = bool(extra.get("cmixup", False))
    vanilla_mixup = bool(extra.get("vanilla_mixup", False))
    if not (bjerrum_enabled or cmixup_enabled or vanilla_mixup):
        return None
    bjer_cfg = BjerrumConfig(enabled=bjerrum_enabled)
    cmix_cfg = CMixupConfig(
        enabled=(cmixup_enabled or vanilla_mixup),
        sigma_y=(float("inf") if vanilla_mixup else None),
    )
    return AugmentationPlan(bjerrum=bjer_cfg, cmixup=cmix_cfg)


def _run_torch_cnn(
    spec: DatasetSpec,
    X_train,
    y_train,
    X_test,
    y_test,
    seed: int,
    family: str,
    extra_options: dict | None = None,
) -> dict:
    set_global_seed(seed)
    device = pick_device("auto")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    x_proc = StandardXProcessor().fit(X_train)
    y_proc = StandardYProcessor().fit(y_train)
    X_train_s = x_proc.transform(X_train)
    X_test_s = x_proc.transform(X_test)
    y_train_s = y_proc.transform(y_train)

    n_features = X_train_s.shape[1]
    builder_map = {
        "nicon": build_nicon_torch,
        "decon": build_decon_torch,
        "nicon_v1a": build_nicon_v1a,
        "nicon_v1a_head": build_nicon_v1a_head_only,
        "nicon_v1a_act": build_nicon_v1a_activation_only,
        "nicon_v1b": build_nicon_v1b,
        "nicon_v1c": build_nicon_v1c,
        "nicon_v2a": build_nicon_v2a,
    }
    builder = builder_map[family]
    _NON_BUILDER_KEYS = {
        "bjerrum", "cmixup", "vanilla_mixup",
        "distill_lambda", "distill_teacher",
        "tta_k", "tta_bjerrum",
        "use_swa", "swa_start_frac", "swa_lr",
        "epochs",
        # R17 — training-side knobs propagated via TrainConfig (not builder).
        "loss_type", "studentt_df", "huber_delta",
        # R17 C — PLS-residual hybrid (handled by a wrapper, not the builder).
        "pls_residual_teacher",
        # R17 F — auxiliary multi-task target (handled by the runner, builder
        # only needs `aux_head_dim`).
        "aux_target_kind", "aux_n_components", "aux_lambda",
        # R17 C-boost — boost teacher (handled by runner, builder needs `boost_signal_dim`).
        "pls_boost_teacher",
        # R19 — OOF teacher prediction toggle (Codex round-13 fix).
        "pls_residual_oof", "pls_boost_oof", "pls_oof_n_folds",
        # R21 — shrinkage CV for V2L-Residual-AOMPLS (do-no-harm gate).
        "shrinkage_cv", "shrinkage_grid", "shrinkage_catastrophic_threshold",
    }
    builder_params = {k: v for k, v in (extra_options or {}).items()
                      if not k.startswith("_") and k not in _NON_BUILDER_KEYS}
    model = builder((1, n_features), params=builder_params).to(device)
    # NiconV2A needs branch-statistics fitting (MSC reference, RMSBranchNorm).
    if family == "nicon_v2a" and hasattr(model, "fit_branches"):
        model.fit_branches(torch.from_numpy(X_train_s.reshape(-1, 1, n_features)).float().to(device))
    config = TrainConfig(seed=seed, device=device.type, batch_size=min(32, max(8, X_train_s.shape[0] // 8)))
    # R13 — optional epoch budget override + SWA knobs.
    if extra_options:
        if "epochs" in extra_options:
            config.epochs = int(extra_options["epochs"])
        if extra_options.get("use_swa", False):
            config.use_swa = True
            config.swa_start_frac = float(extra_options.get("swa_start_frac", 0.75))
            if "swa_lr" in extra_options:
                config.swa_lr = float(extra_options["swa_lr"])
        # R17 B — loss-type override.
        if "loss_type" in extra_options:
            config.loss_type = str(extra_options["loss_type"])
        if "studentt_df" in extra_options:
            config.studentt_df = float(extra_options["studentt_df"])
        if "huber_delta" in extra_options:
            config.huber_delta = float(extra_options["huber_delta"])

    # Phase 1b: optional augmentation hooks (Bjerrum / C-Mixup).
    if extra_options:
        plan = _build_aug_plan(extra_options)
        if plan is not None:
            bjer, cmix, sigma_y = plan.build(X_train_s, y_train_s, seq_len=n_features, device=device)
            if plan.bjerrum.enabled:
                config.augmenter = bjer
            if plan.cmixup.enabled:
                config.cmixup = cmix
                config.cmixup_sigma_y = sigma_y

    # V6 — knowledge distillation from a training-only teacher (e.g. AOM-PLS).
    distill_lambda = float((extra_options or {}).get("distill_lambda", 0.0))
    distill_teacher_fit_time = 0.0
    if distill_lambda > 0.0:
        teacher_name = str((extra_options or {}).get("distill_teacher", "aompls_compact"))
        t_t = time.time()
        teacher_pred_raw = _fit_distill_teacher_predictions(teacher_name, X_train, y_train, seed)
        distill_teacher_fit_time = time.time() - t_t
        config.teacher_predictions = y_proc.transform(teacher_pred_raw)
        config.distill_lambda = distill_lambda

    # R17 F — auxiliary multi-task target (e.g. PLS projection coefficients).
    aux_kind = str((extra_options or {}).get("aux_target_kind", "")).strip()
    aux_n_components = int((extra_options or {}).get("aux_n_components", 5))
    aux_lambda_val = float((extra_options or {}).get("aux_lambda", 0.0))
    if aux_lambda_val > 0.0 and aux_kind:
        if aux_kind == "pls_scores":
            from sklearn.cross_decomposition import PLSRegression
            n_comp = min(aux_n_components, X_train.shape[0] - 1, X_train.shape[1])
            pls_aux = PLSRegression(n_components=n_comp)
            pls_aux.fit(X_train, y_train)
            # Scores: T_train = (X − x_mean) @ x_rotations_.  sklearn exposes the
            # mean as ``_x_mean`` (private attribute, stable across versions).
            x_mean = getattr(pls_aux, "_x_mean", None)
            if x_mean is None:
                x_mean = X_train.mean(axis=0)
            X_centered = np.asarray(X_train, dtype=float) - np.asarray(x_mean, dtype=float)
            scores_train = (X_centered @ pls_aux.x_rotations_).astype(np.float32)
            # Standardise per-component for stable optimisation.
            mu = scores_train.mean(axis=0); std = scores_train.std(axis=0) + 1e-12
            scores_train = (scores_train - mu) / std
            config.aux_targets = scores_train
            config.aux_lambda = aux_lambda_val
        else:
            raise ValueError(f"unknown aux_target_kind {aux_kind!r}; expected pls_scores")

    # R17 C — PLS-residual hybrid: fit a PLS/AOM-PLS teacher, train CNN on residuals,
    # add teacher prediction back at predict time.
    # R19 — OOF mode (default ON, Codex round-13 fix): teacher predictions on
    # X_train are computed via 5-fold CV so each row's residual is OUT of the
    # teacher's training set. Test predictions use a final teacher fit on all
    # of X_train. Disable via `pls_residual_oof: False` for the leakage diagnostic.
    pls_residual_teacher_name = (extra_options or {}).get("pls_residual_teacher")
    pls_residual_teacher = None
    pls_residual_fit_time = 0.0
    use_oof_residual = bool((extra_options or {}).get("pls_residual_oof", True))
    use_oof_boost = bool((extra_options or {}).get("pls_boost_oof", True))
    oof_n_folds = int((extra_options or {}).get("pls_oof_n_folds", 5))
    if pls_residual_teacher_name:
        t_t = time.time()
        if use_oof_residual:
            pls_residual_teacher, z_train_raw = _build_pls_residual_teacher_oof(
                str(pls_residual_teacher_name), X_train, y_train, seed, n_folds=oof_n_folds,
            )
        else:
            pls_residual_teacher = _build_pls_residual_teacher(
                str(pls_residual_teacher_name), X_train, y_train, seed,
            )
            z_train_raw = np.asarray(pls_residual_teacher.predict(X_train), dtype=float).ravel()
        pls_residual_fit_time = time.time() - t_t
        residual_train = np.asarray(y_train, dtype=float).ravel() - z_train_raw
        y_proc = StandardYProcessor().fit(residual_train)
        y_train_s = y_proc.transform(residual_train)

    # R17 C-boost — feature-stacking PLS-residual: pass AOM_pred as boost signal to
    # the head; train CNN to predict y_true given (X, AOM_pred). No residual subtraction.
    pls_boost_teacher_name = (extra_options or {}).get("pls_boost_teacher")
    pls_boost_teacher = None
    pls_boost_fit_time = 0.0
    boost_test_signal = None
    if pls_boost_teacher_name:
        t_t = time.time()
        if use_oof_boost:
            pls_boost_teacher, z_train_raw = _build_pls_residual_teacher_oof(
                str(pls_boost_teacher_name), X_train, y_train, seed, n_folds=oof_n_folds,
            )
        else:
            pls_boost_teacher = _build_pls_residual_teacher(
                str(pls_boost_teacher_name), X_train, y_train, seed,
            )
            z_train_raw = np.asarray(pls_boost_teacher.predict(X_train), dtype=float).ravel()
        z_test_raw = np.asarray(pls_boost_teacher.predict(X_test), dtype=float).ravel()
        pls_boost_fit_time = time.time() - t_t
        # Standardise boost signal using y_proc so it lives on the same scale as
        # the targets the network sees during training.
        config.boost_signals = y_proc.transform(z_train_raw).astype(np.float32)
        boost_test_signal = y_proc.transform(z_test_raw).astype(np.float32)

    t0 = time.time()
    model, info = train_torch_regressor(model, X_train_s, y_train_s, config)
    fit_time = time.time() - t0 + distill_teacher_fit_time + pls_residual_fit_time + pls_boost_fit_time
    t0 = time.time()
    # V7 — test-time Bjerrum augmentation: K forward passes over augmented copies.
    tta_k = int((extra_options or {}).get("tta_k", 1))
    if tta_k > 1 and bool((extra_options or {}).get("tta_bjerrum", True)):
        if boost_test_signal is not None:
            raise NotImplementedError("TTA + boost_signal combination not implemented")
        pred_scaled = _predict_with_tta(model, X_test_s, device, tta_k=tta_k, seed=seed)
    else:
        pred_scaled = predict_torch_regressor(
            model, X_test_s, device=device, boost_signals=boost_test_signal,
        )
    pred_time = time.time() - t0
    pred = y_proc.inverse_transform(pred_scaled)
    # R17 C — add the teacher's test-time prediction back to obtain the final ŷ.
    # R21 — optional shrinkage CV: replace the implicit s=1 add-back with
    # `pred = z_test + s* * nn_residual_test`, where ``s*`` is selected by
    # held-out RMSE on the same val partition used for early-stopping
    # (re-derived deterministically from the seed). ``s = 0`` is always in
    # the grid as a teacher-only fallback.
    s_star: float | None = None
    inner_cv_rmse_per_s: dict[str, float] | None = None
    catastrophic_flag: bool = False
    teacher_test_rmse_for_diag: float | None = None
    if pls_residual_teacher is not None:
        z_test_raw = np.asarray(pls_residual_teacher.predict(X_test), dtype=float).ravel()
        shrinkage_enabled = bool((extra_options or {}).get("shrinkage_cv", False))
        if shrinkage_enabled:
            grid_raw = (extra_options or {}).get(
                "shrinkage_grid", (0.0, 0.25, 0.5, 0.75, 1.0)
            )
            grid = tuple(float(s) for s in grid_raw)
            n_train_local = X_train_s.shape[0]
            rng_split = np.random.default_rng(seed)
            shuffled = rng_split.permutation(n_train_local)
            n_val_local = max(1, int(round(config.val_fraction * n_train_local)))
            val_idx_local = shuffled[:n_val_local]
            # CNN inference on the held-out val partition (residual-scale).
            X_val_s = X_train_s[val_idx_local]
            nn_val_scaled = predict_torch_regressor(model, X_val_s, device=device)
            nn_val_residual = y_proc.inverse_transform(nn_val_scaled)
            y_val_orig = np.asarray(y_train, dtype=float).ravel()[val_idx_local]
            z_val = np.asarray(z_train_raw, dtype=float).ravel()[val_idx_local]
            inner_cv_rmse_per_s = {}
            for s in grid:
                rmse_s = float(
                    np.sqrt(np.mean((y_val_orig - (z_val + s * nn_val_residual)) ** 2))
                )
                inner_cv_rmse_per_s[f"{s:.2f}"] = rmse_s
            s_star = float(min(grid, key=lambda s: inner_cv_rmse_per_s[f"{s:.2f}"]))
            pred = z_test_raw + s_star * pred
            # Catastrophic-loss diagnostic: compare final test RMSE to the
            # teacher-only test RMSE. Threshold defaults to +50 % per
            # B_PLAN_2026-05.md §2.1, overridable via ``shrinkage_catastrophic_threshold``.
            cat_thresh = float(
                (extra_options or {}).get("shrinkage_catastrophic_threshold", 0.5)
            )
            teacher_test_rmse_for_diag = float(
                np.sqrt(np.mean((np.asarray(y_test, dtype=float).ravel() - z_test_raw) ** 2))
            )
            final_rmse = float(
                np.sqrt(np.mean((np.asarray(y_test, dtype=float).ravel() - pred) ** 2))
            )
            if teacher_test_rmse_for_diag > 0:
                catastrophic_flag = bool(
                    (final_rmse / teacher_test_rmse_for_diag - 1.0) > cat_thresh
                )
        else:
            pred = pred + z_test_raw

    hp = {
        "model": family,
        "epochs": config.epochs,
        "patience": config.patience,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "device": str(device),
        "best_epoch": info["best_epoch"],
        "best_val_loss_scaled": info["best_val_loss"],
    }
    if s_star is not None:
        hp["shrinkage_s_star"] = s_star
        hp["shrinkage_inner_cv_rmse_per_s"] = inner_cv_rmse_per_s
        hp["shrinkage_teacher_test_rmse"] = teacher_test_rmse_for_diag
    row = _result_row(
        spec, pred, y_test, fit_time, pred_time, hp,
        total_params=count_parameters(model),
        peak_vram_mb=cuda_peak_mb(),
    )
    if s_star is not None:
        row["catastrophic"] = bool(catastrophic_flag)
        row["shrinkage_s_star"] = float(s_star)
    row["_pred"] = np.asarray(pred, dtype=float).ravel()
    row["_y_test"] = np.asarray(y_test, dtype=float).ravel()
    return row


def _result_row(
    spec: DatasetSpec,
    pred: np.ndarray,
    y_test: np.ndarray,
    fit_time: float,
    pred_time: float,
    hyperparams: dict,
    *,
    total_params: int | float,
    peak_vram_mb: float,
) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    y_test = np.asarray(y_test, dtype=float).ravel()
    rmsep_value = rmse(y_test, pred)
    return {
        "rmsep": rmsep_value,
        "mae": mae(y_test, pred),
        "r2": r2(y_test, pred),
        "bias": bias_metric(y_test, pred),
        "fit_time_s": fit_time,
        "predict_time_s": pred_time,
        "total_params": int(total_params) if not isinstance(total_params, float) or not np.isnan(total_params) else 0,
        "peak_vram_mb": peak_vram_mb,
        "hyperparams_json": json.dumps(hyperparams, default=str),
        "ref_rmse_pls": spec.ref_rmse_pls,
        "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
        "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
        "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
        "ref_rmse_cnn": spec.ref_rmse_cnn,
        "ref_rmse_catboost": spec.ref_rmse_catboost,
        "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best,
        "relative_rmsep_vs_pls": relative_rmsep(rmsep_value, spec.ref_rmse_pls),
        "relative_rmsep_vs_paper_ridge": relative_rmsep(rmsep_value, spec.ref_rmse_paper_ridge),
        "relative_rmsep_vs_tabpfn_raw": relative_rmsep(rmsep_value, spec.ref_rmse_tabpfn_raw),
        "relative_rmsep_vs_tabpfn_opt": relative_rmsep(rmsep_value, spec.ref_rmse_tabpfn_opt),
        "relative_rmsep_vs_cnn": relative_rmsep(rmsep_value, spec.ref_rmse_cnn),
        "relative_rmsep_vs_aom_ridge_curated_best": relative_rmsep(rmsep_value, spec.ref_rmse_aom_ridge_curated_best),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def get_git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_BENCH_ROOT.parent.parent),
            capture_output=True, text=True, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def run_one(
    variant: Variant,
    spec: DatasetSpec,
    seed: int,
) -> dict:
    """Run a single (variant, dataset, seed) and return a result row dict (no metadata)."""
    X_train, y_train, X_test, y_test = load_dataset(spec)
    if variant.family == "ridge":
        result = _run_ridge(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "pls":
        result = _run_pls(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "searched_ridge":
        result = _run_searched_ridge(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "searched_pls":
        result = _run_searched_pls(spec, X_train, y_train, X_test, y_test, seed)
    elif variant.family == "stack":
        result = _run_stack(spec, X_train, y_train, X_test, y_test, seed, extra=variant.extra)
    elif variant.family in ("nicon", "decon", "nicon_v1a", "nicon_v1a_head", "nicon_v1a_act", "nicon_v1b", "nicon_v1c", "nicon_v2a"):
        result = _run_torch_cnn(
            spec, X_train, y_train, X_test, y_test, seed,
            family=variant.family, extra_options=variant.extra,
        )
    else:
        raise ValueError(f"unknown variant family: {variant.family!r}")
    return result


def filter_specs(all_specs: list[DatasetSpec], cohort: str, only: list[str] | None) -> list[DatasetSpec]:
    if only:
        keep = set(only)
        return [s for s in all_specs if s.dataset in keep]
    return all_specs


def main() -> int:
    parser = argparse.ArgumentParser(description="nicon_v2 baseline benchmark runner")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--cohort", default="smoke",
                        choices=["smoke", "extended_smoke", "representative", "curated", "full"])
    parser.add_argument("--variants", default="smoke",
                        choices=["smoke", "phase1a", "phase1b", "phase1c", "stack", "stack_aom", "searched",
                                 "v2a", "v2_r6", "v2_r7", "v2_r8", "v2_r9", "v2_r10", "v2_r11", "v2_r12",
                                 "v2_r13", "v2_r14_multiseed", "v2_r15_lucas",
                                 "v2_r16_lucas_multiseed", "publication",
                                 "v2_r17_priority1", "v2_r17_priority2",
                                 "v2_r18_residual_multiseed",
                                 "v2_r19_oof_diagnostic", "v2_r20_final",
                                 "v2_r21_multiseed"])
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="run multiple seeds (overrides --seed)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to dataset names")
    parser.add_argument("--skip-cnn", action="store_true", help="skip nicon/decon variants (fast smoke)")
    args = parser.parse_args()

    workspace: Path = args.workspace
    csv_path = workspace / "results.csv"
    workspace.mkdir(parents=True, exist_ok=True)

    specs = load_cohort_manifest(args.cohort)
    specs = filter_specs(specs, args.cohort, args.only)

    if args.variants == "smoke":
        variants = list(SMOKE_VARIANTS)
    elif args.variants == "phase1a":
        variants = list(PHASE1A_VARIANTS)
    elif args.variants == "phase1b":
        variants = list(PHASE1B_VARIANTS)
    elif args.variants == "phase1c":
        variants = list(PHASE1C_VARIANTS)
    elif args.variants == "stack":
        variants = list(PHASE_STACK_VARIANTS)
    elif args.variants == "stack_aom":
        variants = list(PHASE_STACK_AOM_VARIANTS)
    elif args.variants == "searched":
        variants = list(PHASE_SEARCHED_VARIANTS)
    elif args.variants == "v2a":
        variants = list(PHASE_V2A_VARIANTS)
    elif args.variants == "v2_r6":
        variants = list(PHASE_V2_R6_VARIANTS)
    elif args.variants == "v2_r7":
        variants = list(PHASE_V2_R7_VARIANTS)
    elif args.variants == "v2_r8":
        variants = list(PHASE_V2_R8_VARIANTS)
    elif args.variants == "v2_r9":
        variants = list(PHASE_V2_R9_VARIANTS)
    elif args.variants == "v2_r10":
        variants = list(PHASE_V2_R10_VARIANTS)
    elif args.variants == "v2_r11":
        variants = list(PHASE_V2_R11_VARIANTS)
    elif args.variants == "v2_r12":
        variants = list(PHASE_V2_R12_VARIANTS)
    elif args.variants == "v2_r13":
        variants = list(PHASE_V2_R13_VARIANTS)
    elif args.variants == "v2_r14_multiseed":
        variants = list(PHASE_V2_R14_MULTISEED)
    elif args.variants == "v2_r15_lucas":
        variants = list(PHASE_V2_R15_LUCAS)
    elif args.variants == "v2_r16_lucas_multiseed":
        variants = list(PHASE_V2_R16_LUCAS_MULTISEED)
    elif args.variants == "publication":
        variants = list(PHASE_PUBLICATION)
    elif args.variants == "v2_r17_priority1":
        variants = list(PHASE_V2_R17_PRIORITY1)
    elif args.variants == "v2_r17_priority2":
        variants = list(PHASE_V2_R17_PRIORITY2)
    elif args.variants == "v2_r18_residual_multiseed":
        variants = list(PHASE_V2_R18_RESIDUAL_MULTISEED)
    elif args.variants == "v2_r19_oof_diagnostic":
        variants = list(PHASE_V2_R19_OOF_DIAGNOSTIC)
    elif args.variants == "v2_r20_final":
        variants = list(PHASE_V2_R20_FINAL)
    elif args.variants == "v2_r21_multiseed":
        variants = list(PHASE_V2_R21_MULTISEED)
    else:
        raise ValueError(f"unknown variants set: {args.variants!r}")
    if args.skip_cnn:
        variants = [v for v in variants if v.family in ("ridge", "pls")]
    seeds = args.seeds if args.seeds else [args.seed]

    done = existing_done_rows(csv_path)
    writer, fh = open_result_writer(csv_path)
    git_sha = get_git_sha()
    cuda_version = torch.version.cuda or ""

    try:
      for current_seed in seeds:
        t_start = time.time()  # initialise even if all rows skip (resume case)
        for spec in specs:
            for variant in variants:
                key = (spec.dataset, variant.label, current_seed)
                if key in done:
                    print(f"skip [done] {variant.label} on {spec.dataset} seed={current_seed}", flush=True)
                    continue
                row: dict = {
                    "dataset_group": spec.database_name,
                    "dataset": spec.dataset,
                    "task": "regression",
                    "n_train": spec.n_train,
                    "n_test": spec.n_test,
                    "n_features": spec.n_features,
                    "variant": variant.label,
                    "model_version": CODE_VERSION,
                    "seed": current_seed,
                    "cv_fold": -1,
                    "cv_protocol": "predefined",
                    "status": "OK",
                    "error_message": "",
                    "rmsep": float("nan"),
                    "mae": float("nan"),
                    "r2": float("nan"),
                    "bias": float("nan"),
                    "fit_time_s": float("nan"),
                    "predict_time_s": float("nan"),
                    "total_params": 0,
                    "peak_vram_mb": float("nan"),
                    "hyperparams_json": "",
                    "ref_rmse_pls": spec.ref_rmse_pls,
                    "ref_rmse_paper_ridge": spec.ref_rmse_paper_ridge,
                    "ref_rmse_tabpfn_raw": spec.ref_rmse_tabpfn_raw,
                    "ref_rmse_tabpfn_opt": spec.ref_rmse_tabpfn_opt,
                    "ref_rmse_cnn": spec.ref_rmse_cnn,
                    "ref_rmse_catboost": spec.ref_rmse_catboost,
                    "ref_rmse_aom_ridge_curated_best": spec.ref_rmse_aom_ridge_curated_best,
                    "relative_rmsep_vs_pls": None,
                    "relative_rmsep_vs_paper_ridge": None,
                    "relative_rmsep_vs_tabpfn_raw": None,
                    "relative_rmsep_vs_tabpfn_opt": None,
                    "relative_rmsep_vs_cnn": None,
                    "relative_rmsep_vs_aom_ridge_curated_best": None,
                    "python_version": sys.version.split(" ")[0],
                    "torch_version": torch.__version__,
                    "cuda_version": cuda_version,
                    "git_sha": git_sha,
                    "host": platform.node(),
                }
                t_start = time.time()
                try:
                    out = run_one(variant, spec, current_seed)
                    pred_arr = out.pop("_pred", None)
                    y_test_arr = out.pop("_y_test", None)
                    row.update(out)
                    if pred_arr is not None and y_test_arr is not None:
                        write_predictions_parquet(
                            workspace, variant.label, spec.dataset, current_seed,
                            y_test_arr, pred_arr,
                        )
                    print(
                        f"OK    {variant.label:<22}  {spec.dataset:<35}  rmse={row['rmsep']:.4f}  "
                        f"fit={row['fit_time_s']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:  # benchmark must keep running on per-row errors
                    row["status"] = "ERROR"
                    row["error_message"] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"ERROR {variant.label:<18}  {spec.dataset:<35}  {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                writer.writerow(row)
                fh.flush()
        print(f"done in {time.time() - t_start:.1f}s; rows written to {csv_path}")
    finally:
        fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
