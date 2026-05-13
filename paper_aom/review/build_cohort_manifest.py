"""Build the canonical AOM paper cohort manifest (CSV + Markdown).

This script consolidates the AOM_v0 regression and classification cohorts into a
single per-(dataset, task) manifest used by every downstream table and figure in
the Talanta submission. It also writes a human-readable Markdown summary with
the denominators that the manuscript and supplement must cite.

Sources read by default:

* ``bench/AOM_v0/benchmarks/cohort_regression.csv``         (61 regression rows)
* ``bench/AOM_v0/benchmarks/cohort_classification.csv``     (17 classification rows + skipped)
* ``bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv``
* ``bench/scenarios/runs/best_current_full57_seed0/results.csv``
* ``bench/AOM_v0/benchmark_runs/full/results.csv``          (broad AOM-PLS grid, 7888 rows)
* ``bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv``
* ``bench/master_results_classif.csv``                       (PLS-DA / TabPFN baselines)
* ``bench/1_master_results.csv``                             (PLS / Ridge / TabPFN regression baselines)

Outputs:

* ``paper_aom/review/cohort_manifest.csv``  -- one row per (dataset, task) with all P0 columns
* ``paper_aom/review/cohort_manifest.md``   -- summary, denominators, per-domain table

Example::

    python paper_aom/review/build_cohort_manifest.py \\
        --out paper_aom/review/cohort_manifest.csv \\
        --md  paper_aom/review/cohort_manifest.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Default source paths (resolved relative to the repository root).
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_REG_COHORT = _REPO_ROOT / "bench/AOM_v0/benchmarks/cohort_regression.csv"
_DEFAULT_CLF_COHORT = _REPO_ROOT / "bench/AOM_v0/benchmarks/cohort_classification.csv"
_DEFAULT_OUT_CSV = _REPO_ROOT / "paper_aom/review/cohort_manifest.csv"
_DEFAULT_OUT_MD = _REPO_ROOT / "paper_aom/review/cohort_manifest.md"

_DEFAULT_SOURCES = (
    _REPO_ROOT / "bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv",
    _REPO_ROOT / "bench/scenarios/runs/best_current_full57_seed0/results.csv",
    _REPO_ROOT / "bench/AOM_v0/benchmark_runs/full/results.csv",
    _REPO_ROOT / "bench/AOM_v0/Ridge/benchmark_runs/classification_all17/results.csv",
    _REPO_ROOT / "bench/master_results_classif.csv",
    _REPO_ROOT / "bench/1_master_results.csv",
)

# --------------------------------------------------------------------------- #
# Domain group mapping (database_name -> human-readable family).
#
# Keep "other" as bucket for unmapped database_name values. The mapping was
# derived from the cohort CSVs and the paper narrative; new families can be
# added without changing downstream code.
# --------------------------------------------------------------------------- #

_DOMAIN_GROUPS: dict[str, str] = {
    "ALPINE": "leaf-physiology",
    "AMYLOSE": "crop-grain",
    "BEEFMARBLING": "meat-quality",
    "BEEF_Impurity": "meat-quality",
    "BEER": "beverage",
    "BERRY": "fruit-quality",
    "BISCUIT": "food-product",
    "COFFEE_orig": "beverage",
    "COFFEE_sp": "beverage",
    "COLZA": "crop-seed",
    "CORN": "crop-grain",
    "Cassava": "crop-tuber",
    "DIESEL": "petroleum",
    "DarkResp": "leaf-physiology",
    "ECOSIS_LeafTraits": "leaf-physiology",
    "FUSARIUM": "plant-disease",
    "FruitPuree": "fruit-quality",
    "GRAPEVINES": "leaf-physiology",
    "GRAPEVINE_LeafTraits": "leaf-physiology",
    "IncombustibleMaterial": "industrial",
    "LUCAS": "soil-eu",
    "MALARIA": "biomedical",
    "MANURE21": "soil-amendment",
    "MILK": "dairy",
    "PEACH": "fruit-quality",
    "PHOSPHORUS": "leaf-physiology",
    "PISTACIA": "plant-id",
    "PLUMS": "fruit-quality",
    "QUARTZ": "mineral",
    "TABLET": "pharmaceutical",
    "WOOD_density": "wood-product",
    "Wood_Sustainability": "wood-product",
    "ARABIDOPSIS_CEFE": "plant-id",
}

# --------------------------------------------------------------------------- #
# Split type recognition. Order matters: more specific markers first.
# --------------------------------------------------------------------------- #

_SPLIT_PATTERNS: Sequence[tuple[str, str]] = (
    ("NocitaKS", "NocitaKS"),
    ("block2deg", "spxyG_block2deg"),
    ("byCultivar", "spxyG_byCultivar"),
    ("byCultivar", "spxyG70_30_byCultivar"),
    ("spxyG70_30", "spxyG70_30"),
    ("spxyG", "spxyG"),
    ("YbasedSplit", "YbasedSplit"),
    ("YbaseSplit", "YbaseSplit"),
    ("Ybase", "YbaseSplit"),
    ("Zheng", "ZhengChenPelegYbaseSplit"),
    ("Maia", "Maia"),
    ("Holland", "Holland"),
    ("AlJowder", "AlJowder"),
    ("Davrieux", "Davrieux"),
    ("Olale", "Olale"),
    ("Zhao", "Zhao"),
    ("Liu", "LiuRandom"),
    ("Bagnall", "Bagnall"),
    ("CIAT", "CIAT"),
    ("kenstone70", "kenstone70_strat"),
    ("classStrat", "grp70_30_classStrat"),
    ("scoreQ", "grp70_30_scoreQ"),
    ("grp70_30", "grp70_30"),
    ("grpStrat70_30", "grpStrat70_30"),
    ("groupSampleID", "groupSampleID_stratDateVar"),
    ("SPXY_strat", "SPXY_strat"),
    ("spxy70", "spxy70"),
    ("SPXY", "SPXY"),
    ("spxy", "spxy"),
    ("RandomSplit", "RandomSplit"),
    ("Random", "Random"),
    ("KS", "KS"),
    ("CBtestSite", "external_site_CB"),
    ("GTtestSite", "external_site_GT"),
    ("XSBNtestSite", "external_site_XSBN"),
    ("woOutlier", "woOutlier"),
    ("wOutlier", "wOutlier"),
    ("bySpecimen", "bySpecimen"),
)

# --------------------------------------------------------------------------- #
# Allowability gates.
# --------------------------------------------------------------------------- #

_TABPFN_MAX_FEATURES = 500
"""TabPFN sample-then-truncate path: PFN backbone trained on <=500 features."""

_TABPFN_MAX_N_TRAIN = 10000
"""TabPFN v1 inference cost grows quadratically; 10k train rows is the
practical ceiling on a single GPU (see TabPFN paper, Sec. 3.4)."""

_AOMRIDGE_GLOBAL_NXP_THRESHOLD = 1e8
"""AOMRidge-global builds K = X X^T or full Gram blocks; with 62 GB RAM on the
benchmark host an upper bound of ``n_features * n_train < 1e8`` keeps the
single-machine path under ~30 GB peak resident set. Above that the variant
should be routed through the local/MKL configurations."""


# --------------------------------------------------------------------------- #
# Source identification: which canonical_name / model / variant maps to which
# manifest "has_*" flag.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CandidateRule:
    """Match a candidate run row by canonical_name / model / variant prefixes."""

    column: str
    needles: tuple[str, ...]
    case_insensitive: bool = True


_REG_HAS_FLAGS: dict[str, tuple[CandidateRule, ...]] = {
    "has_pls": (
        CandidateRule("canonical_name", ("PLS-tuned-cv5",)),
        CandidateRule("model", ("PLS-standard-numpy", "PLS")),
    ),
    "has_ridge": (
        CandidateRule("canonical_name", ("Ridge-tuned-cv5",)),
        CandidateRule("model", ("Ridge",)),
    ),
    "has_aom_pls": (
        CandidateRule("canonical_name", ("AOM-PLS-compact-numpy", "ASLS-AOM-compact-cv5-numpy")),
        CandidateRule("model", ("AOM-compact-cv5-numpy", "ASLS-AOM-compact-cv5-numpy", "nirs4all-AOM-PLS-default")),
    ),
    "has_aom_ridge": (
        CandidateRule(
            "canonical_name",
            (
                "AOMRidge-global-compact-none",
                "AOMRidge-global-compact-snv",
                "AOMRidge-Local-compact-knn50",
                "AOMRidge-Blender-headline-spxy3",
                "AOMRidge-AutoSelect-headline-spxy3",
                "AOMRidge-MultiBranchMKL-compact-shrink03",
                "AOMRidge-Local-compact-knn-sweep",
            ),
        ),
    ),
    "has_tabpfn_raw": (
        CandidateRule("model", ("TabPFN-Raw",)),
    ),
    "has_tabpfn_hpo": (
        CandidateRule("model", ("TabPFN-opt",)),
    ),
}

_CLF_HAS_FLAGS: dict[str, tuple[CandidateRule, ...]] = {
    "has_pls_da": (
        CandidateRule("model", ("PLS-DA",)),
    ),
    "has_aom_pls_da": (
        CandidateRule(
            "variant",
            (
                "AOM-PLS-DA-global-nipals-adjoint",
                "POP-PLS-DA-nipals-adjoint",
                "AOM-PLS-DA-global-simpls-covariance",
                "POP-PLS-DA-simpls-covariance",
                "PLS-DA-standard",
            ),
        ),
    ),
    "has_aom_ridge_cls": (
        CandidateRule(
            "variant",
            (
                "AOMRidgeCls-global-compact",
                "AOMRidgeCls-active-compact",
                "AOMRidgeCls-branch_global-compact",
                "AOMRidgeCls-mkl-compact",
                "AOMRidgeCls-superblock-compact",
            ),
        ),
    ),
    "has_tabpfn_raw": (
        CandidateRule("model", ("TabPFN-Raw",)),
    ),
    "has_tabpfn_hpo": (
        CandidateRule("model", ("TabPFN-opt",)),
    ),
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, returning ``None`` if it does not exist."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _split_type_from_name(dataset: str) -> str:
    """Derive a coarse split-type label from the dataset folder name."""
    name = dataset
    for needle, label in _SPLIT_PATTERNS:
        if needle in name:
            return label
    return "unspecified"


def _domain_group(database_name: str) -> str:
    """Map a database_name to a coarse domain bucket."""
    return _DOMAIN_GROUPS.get(database_name, "other")


def _response_or_trait(database_name: str, dataset: str) -> str:
    """Derive a short trait/response token from the dataset folder name."""
    base = dataset
    for sep in ("_KS", "_SPXY", "_spxy", "_RandomSplit", "_Random", "_Maia", "_kenstone70",
                "_grpStrat70", "_grp70", "_groupSample", "_Olale", "_Zhao", "_Liu",
                "_NocitaKS", "_Davrieux", "_Bagnall", "_Holland", "_AlJowder",
                "_byCultivar", "_block2deg", "_Yb", "_Zheng", "_classStrat",
                "_scoreQ", "_bySpecimen"):
        idx = base.find(sep)
        if idx > 0:
            base = base[:idx]
            break
    return base or database_name


def _matches_any(row: pd.Series, rule: CandidateRule) -> bool:
    """Return True if ``row[rule.column]`` matches any needle in ``rule``."""
    if rule.column not in row.index:
        return False
    val = row[rule.column]
    if not isinstance(val, str):
        return False
    haystack = val.casefold() if rule.case_insensitive else val
    for needle in rule.needles:
        nd = needle.casefold() if rule.case_insensitive else needle
        if nd == haystack:
            return True
    return False


_SUCCESS_STATUSES = {"ok", "partial"}


def _datasets_with_candidate(
    sources: Iterable[pd.DataFrame],
    rules: tuple[CandidateRule, ...],
    success_only: bool = True,
) -> set[str]:
    """Return the set of dataset names that have at least one success row
    matching any of the supplied candidate rules across the given source frames.

    "Success" includes both ``status == "ok"`` and ``status == "partial"``;
    the latter is used by the master baselines (PLS / Ridge / PLS-DA) where the
    HPO finished but with a non-default fold count, and we still want those
    datasets to count as "has a result for this candidate"."""
    hits: set[str] = set()
    for df in sources:
        if df is None or df.empty:
            continue
        cols = set(df.columns)
        if "dataset" not in cols:
            continue
        sub = df
        if success_only and "status" in cols:
            sub = sub[sub["status"].astype(str).str.lower().isin(_SUCCESS_STATUSES)]
        if sub.empty:
            continue
        for rule in rules:
            if rule.column not in cols:
                continue
            mask = sub.apply(lambda r, rule=rule: _matches_any(r, rule), axis=1)
            hits.update(sub.loc[mask, "dataset"].astype(str).tolist())
    return hits


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #


def build_manifest(
    reg_cohort_path: Path,
    clf_cohort_path: Path,
    source_paths: Sequence[Path],
) -> pd.DataFrame:
    """Build the merged (regression + classification) manifest DataFrame."""
    reg = pd.read_csv(reg_cohort_path)
    clf = pd.read_csv(clf_cohort_path)
    sources = [_read_csv_if_exists(p) for p in source_paths]

    # Identify which datasets have which candidate.
    has_sets_reg: dict[str, set[str]] = {
        flag: _datasets_with_candidate(sources, rules) for flag, rules in _REG_HAS_FLAGS.items()
    }
    has_sets_clf: dict[str, set[str]] = {
        flag: _datasets_with_candidate(sources, rules) for flag, rules in _CLF_HAS_FLAGS.items()
    }

    rows: list[dict[str, object]] = []

    # ----- Regression rows ------------------------------------------------- #
    for _, r in reg.iterrows():
        dataset = str(r["dataset"])
        database = str(r["database_name"])
        status_raw = str(r["status"]) if pd.notna(r.get("status")) else "skipped"
        n_train = r.get("n_train")
        n_test = r.get("n_test")
        p = r.get("p")
        reason = r.get("reason")
        reason_str = "" if pd.isna(reason) else str(reason)

        n_train_i = int(n_train) if pd.notna(n_train) else None
        n_test_i = int(n_test) if pd.notna(n_test) else None
        p_i = int(p) if pd.notna(p) else None

        p_over_n = float(p_i) / float(n_train_i) if (p_i is not None and n_train_i) else None

        tabpfn_allowed = bool(
            p_i is not None
            and n_train_i is not None
            and p_i <= _TABPFN_MAX_FEATURES
            and n_train_i <= _TABPFN_MAX_N_TRAIN
        )
        aomridge_global_allowed = bool(
            p_i is not None
            and n_train_i is not None
            and (float(p_i) * float(n_train_i)) < _AOMRIDGE_GLOBAL_NXP_THRESHOLD
        )

        is_quartz = database.upper() == "QUARTZ"
        included = status_raw == "ok"

        rows.append(
            {
                "dataset": dataset,
                "task": "regression",
                "source_family": database,
                "source_run": "bench/AOM_v0/benchmarks/cohort_regression.csv",
                "domain_group": _domain_group(database),
                "response_or_trait": _response_or_trait(database, dataset),
                "split_type": _split_type_from_name(dataset),
                "n_train": n_train_i,
                "n_test": n_test_i,
                "n_features": p_i,
                "p_over_n_train": p_over_n,
                "has_pls": dataset in has_sets_reg.get("has_pls", set()),
                "has_ridge": dataset in has_sets_reg.get("has_ridge", set()),
                "has_aom_pls": dataset in has_sets_reg.get("has_aom_pls", set()),
                "has_aom_ridge": dataset in has_sets_reg.get("has_aom_ridge", set()),
                "has_pls_da": False,
                "has_aom_pls_da": False,
                "has_aom_ridge_cls": False,
                "has_tabpfn_raw": dataset in has_sets_reg.get("has_tabpfn_raw", set()),
                "has_tabpfn_hpo": dataset in has_sets_reg.get("has_tabpfn_hpo", set()),
                "tabpfn_allowed": tabpfn_allowed,
                "aomridge_global_allowed": aomridge_global_allowed,
                "status_in_primary_analysis": "include" if included else "exclude",
                "exclusion_reason": (
                    "denominator_near_zero_pairwise" if is_quartz else (reason_str if not included else "")
                ),
            }
        )

    # ----- Classification rows --------------------------------------------- #
    for _, r in clf.iterrows():
        dataset = str(r["dataset"])
        database = str(r["database_name"])
        status_raw = str(r["status"]) if pd.notna(r.get("status")) else "skipped"
        n_train = r.get("n_train")
        n_test = r.get("n_test")
        p = r.get("p")
        reason = r.get("reason")
        reason_str = "" if pd.isna(reason) else str(reason)

        n_train_i = int(n_train) if pd.notna(n_train) else None
        n_test_i = int(n_test) if pd.notna(n_test) else None
        p_i = int(p) if pd.notna(p) else None

        p_over_n = float(p_i) / float(n_train_i) if (p_i is not None and n_train_i) else None

        tabpfn_allowed = bool(
            p_i is not None
            and n_train_i is not None
            and p_i <= _TABPFN_MAX_FEATURES
            and n_train_i <= _TABPFN_MAX_N_TRAIN
        )
        aomridge_global_allowed = bool(
            p_i is not None
            and n_train_i is not None
            and (float(p_i) * float(n_train_i)) < _AOMRIDGE_GLOBAL_NXP_THRESHOLD
        )

        included = status_raw == "ok"

        rows.append(
            {
                "dataset": dataset,
                "task": "classification",
                "source_family": database,
                "source_run": "bench/AOM_v0/benchmarks/cohort_classification.csv",
                "domain_group": _domain_group(database),
                "response_or_trait": _response_or_trait(database, dataset),
                "split_type": _split_type_from_name(dataset),
                "n_train": n_train_i,
                "n_test": n_test_i,
                "n_features": p_i,
                "p_over_n_train": p_over_n,
                "has_pls": False,
                "has_ridge": False,
                "has_aom_pls": False,
                "has_aom_ridge": False,
                "has_pls_da": dataset in has_sets_clf.get("has_pls_da", set()),
                "has_aom_pls_da": dataset in has_sets_clf.get("has_aom_pls_da", set()),
                "has_aom_ridge_cls": dataset in has_sets_clf.get("has_aom_ridge_cls", set()),
                "has_tabpfn_raw": dataset in has_sets_clf.get("has_tabpfn_raw", set()),
                "has_tabpfn_hpo": dataset in has_sets_clf.get("has_tabpfn_hpo", set()),
                "tabpfn_allowed": tabpfn_allowed,
                "aomridge_global_allowed": aomridge_global_allowed,
                "status_in_primary_analysis": "include" if included else "exclude",
                "exclusion_reason": "" if included else reason_str,
            }
        )

    columns = [
        "dataset",
        "task",
        "source_family",
        "source_run",
        "domain_group",
        "response_or_trait",
        "split_type",
        "n_train",
        "n_test",
        "n_features",
        "p_over_n_train",
        "has_pls",
        "has_ridge",
        "has_aom_pls",
        "has_aom_ridge",
        "has_pls_da",
        "has_aom_pls_da",
        "has_aom_ridge_cls",
        "has_tabpfn_raw",
        "has_tabpfn_hpo",
        "tabpfn_allowed",
        "aomridge_global_allowed",
        "status_in_primary_analysis",
        "exclusion_reason",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df


# --------------------------------------------------------------------------- #
# Markdown writer
# --------------------------------------------------------------------------- #


def _format_md_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a Markdown table from headers + rows."""
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def write_markdown(df: pd.DataFrame, out_md: Path) -> None:
    """Write the cohort manifest summary as Markdown."""
    reg = df[df["task"] == "regression"]
    clf = df[df["task"] == "classification"]

    reg_inc = reg[reg["status_in_primary_analysis"] == "include"]
    clf_inc = clf[clf["status_in_primary_analysis"] == "include"]

    n_reg_total = len(reg)
    n_reg_inc = len(reg_inc)
    n_clf_total = len(clf)
    n_clf_inc = len(clf_inc)

    quartz_rows = reg[reg["source_family"].str.upper() == "QUARTZ"]
    n_quartz = len(quartz_rows)

    n_tabpfn_allowed_reg = int(reg_inc["tabpfn_allowed"].sum())
    n_aomridge_allowed_reg = int(reg_inc["aomridge_global_allowed"].sum())
    n_tabpfn_allowed_clf = int(clf_inc["tabpfn_allowed"].sum())

    # Paired denominators per candidate (regression). The "paired vs PLS"
    # row says: how many included datasets have BOTH has_pls AND has_<X>.
    has_pls_set = set(reg_inc.loc[reg_inc["has_pls"], "dataset"])
    has_ridge_set = set(reg_inc.loc[reg_inc["has_ridge"], "dataset"])
    paired_rows: list[tuple[str, int, str, str]] = [
        (
            "full regression cohort",
            n_reg_inc,
            "status==ok in cohort_regression.csv",
            "RMSEP (absolute)",
        ),
        (
            "paired AOM-PLS vs PLS",
            len(has_pls_set & set(reg_inc.loc[reg_inc["has_aom_pls"], "dataset"])),
            "has_pls AND has_aom_pls",
            "RMSEP / RMSEP_PLS",
        ),
        (
            "paired AOM-Ridge vs Ridge",
            len(has_ridge_set & set(reg_inc.loc[reg_inc["has_aom_ridge"], "dataset"])),
            "has_ridge AND has_aom_ridge",
            "RMSEP / RMSEP_Ridge",
        ),
        (
            "paired AOM-PLS vs TabPFN-HPO",
            len(set(reg_inc.loc[reg_inc["has_tabpfn_hpo"], "dataset"]) & set(reg_inc.loc[reg_inc["has_aom_pls"], "dataset"])),
            "has_tabpfn_hpo AND has_aom_pls",
            "RMSEP / RMSEP_TabPFN-opt",
        ),
        (
            "TabPFN-allowed regression subset",
            n_tabpfn_allowed_reg,
            f"n_features<={_TABPFN_MAX_FEATURES} AND n_train<={_TABPFN_MAX_N_TRAIN}",
            "balanced ratio or absolute RMSEP",
        ),
        (
            "AOM-Ridge-global allowed subset",
            n_aomridge_allowed_reg,
            f"n_features*n_train<{int(_AOMRIDGE_GLOBAL_NXP_THRESHOLD):,}",
            "RMSEP (single-machine path)",
        ),
        ("full classification cohort", n_clf_inc, "status==ok in cohort_classification.csv", "balanced accuracy, macro-F1, log-loss, ECE"),
        (
            "TabPFN-allowed classification subset",
            n_tabpfn_allowed_clf,
            f"n_features<={_TABPFN_MAX_FEATURES} AND n_train<={_TABPFN_MAX_N_TRAIN}",
            "balanced accuracy",
        ),
    ]

    # Per-domain table.
    dom_rows: list[tuple[str, int, str]] = []
    for dom, g in reg_inc.groupby("domain_group"):
        mean_pn = g["p_over_n_train"].dropna().mean()
        dom_rows.append((dom, len(g), f"{mean_pn:.2f}" if pd.notna(mean_pn) else ""))
    dom_rows.sort(key=lambda row: (-row[1], row[0]))

    md = []
    md.append("# AOM paper cohort manifest")
    md.append("")
    md.append("Auto-generated by `paper_aom/review/build_cohort_manifest.py`.")
    md.append("Do not edit by hand. Re-run the script to refresh.")
    md.append("")
    md.append("## Summary counts")
    md.append("")
    md.append(f"- Regression cohort: **{n_reg_inc} included / {n_reg_total} total** (all rows with `status==ok` in `cohort_regression.csv`).")
    md.append(f"- Classification cohort: **{n_clf_inc} included / {n_clf_total} total** (`status==ok` in `cohort_classification.csv`).")
    md.append(f"- QUARTZ datasets present: **{n_quartz}** (kept in absolute-error tables; excluded from pairwise ratios -- see rule below).")
    md.append(f"- TabPFN-allowed regression datasets: **{n_tabpfn_allowed_reg}** (n_features <= {_TABPFN_MAX_FEATURES} AND n_train <= {_TABPFN_MAX_N_TRAIN}).")
    md.append(f"- TabPFN-allowed classification datasets: **{n_tabpfn_allowed_clf}**.")
    md.append(f"- AOM-Ridge-global-allowed regression datasets: **{n_aomridge_allowed_reg}** (n_features * n_train < {int(_AOMRIDGE_GLOBAL_NXP_THRESHOLD):,}).")
    md.append("")
    md.append("## Denominators per analysis")
    md.append("")
    md.append(_format_md_table(
        ["analysis", "included_n", "denominator_rule", "primary_metric"],
        paired_rows,
    ))
    md.append("")
    md.append("## Per-domain breakdown (regression, included)")
    md.append("")
    md.append(_format_md_table(
        ["domain_group", "n_datasets", "mean_p_over_n_train"],
        dom_rows,
    ))
    md.append("")
    md.append("## QUARTZ treatment rule")
    md.append("")
    md.append(
        "The `QUARTZ_spxy70` dataset has reference RMSE near 3.4e-9 (essentially zero). "
        "Pairwise ratios such as RMSEP/RMSEP_PLS therefore blow up by ~6 orders of magnitude "
        "for any non-trivial candidate. Rule:"
    )
    md.append("")
    md.append("- **keep** QUARTZ rows in absolute RMSEP / failure tables;")
    md.append("- **exclude** QUARTZ rows from pairwise ratio summaries (median, IQR, win-counts);")
    md.append("- the manifest column `exclusion_reason` is set to `denominator_near_zero_pairwise` for QUARTZ; `status_in_primary_analysis` stays `include`.")
    md.append("")
    md.append("## Inclusion / exclusion rules")
    md.append("")
    md.append("1. `status_in_primary_analysis = include` iff the source cohort row has `status==ok`; the manifest preserves the original `reason` in `exclusion_reason` otherwise.")
    md.append("2. The full regression cohort denominator is 61 (all `ok` rows). The historical 57-dataset headline is derived by also requiring `has_pls AND has_aom_pls` -- this lives in the paired tables above.")
    md.append("3. Pairwise tables vs Ridge / TabPFN use the corresponding has_* intersection; each caption must cite the denominator from this manifest.")
    md.append("4. `tabpfn_allowed = (n_features <= 500) AND (n_train <= 10000)`. Source: TabPFN paper Sec. 3.4 (sample-then-truncate path; PFN backbone trained on <=500 features; v1 inference cost is quadratic).")
    md.append("5. `aomridge_global_allowed = (n_features * n_train < 1e8)`. The AOMRidge-global path materialises full Gram blocks; with 62 GB RAM on the benchmark host (Linux 6.6, WSL2, 64 GB total), 1e8 keeps peak RSS under ~30 GB. Datasets above the threshold must be routed through Local/MKL variants.")
    md.append("6. Classification rows for which a result CSV is missing (e.g. `COFFEE_sp/Species_56_Bagnall`) carry `exclusion_reason` from the cohort CSV (e.g. `missing files: ...`).")
    md.append("")
    md.append("## Source CSVs scanned for `has_*` flags")
    md.append("")
    for src in _DEFAULT_SOURCES:
        rel = src.relative_to(_REPO_ROOT) if src.exists() else src
        md.append(f"- `{rel}` {'(found)' if src.exists() else '(missing)'}")
    md.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg-cohort", default=str(_DEFAULT_REG_COHORT), help="Path to cohort_regression.csv")
    parser.add_argument("--clf-cohort", default=str(_DEFAULT_CLF_COHORT), help="Path to cohort_classification.csv")
    parser.add_argument("--sources", default=None, help="Comma-separated source CSVs to scan for has_* flags")
    parser.add_argument("--out", default=str(_DEFAULT_OUT_CSV), help="Output manifest CSV path")
    parser.add_argument("--md", default=str(_DEFAULT_OUT_MD), help="Output Markdown summary path")
    args = parser.parse_args(argv)

    if args.sources is None:
        source_paths: Sequence[Path] = _DEFAULT_SOURCES
    else:
        source_paths = tuple(Path(s.strip()) for s in args.sources.split(",") if s.strip())

    df = build_manifest(Path(args.reg_cohort), Path(args.clf_cohort), source_paths)

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    write_markdown(df, Path(args.md))

    print(f"[manifest] wrote {len(df)} rows -> {out_csv}")
    print(f"[manifest] regression={int((df['task']=='regression').sum())} classification={int((df['task']=='classification').sum())}")
    print(f"[manifest] included={int((df['status_in_primary_analysis']=='include').sum())}")
    print(f"[manifest] tabpfn_allowed={int(df['tabpfn_allowed'].sum())} aomridge_global_allowed={int(df['aomridge_global_allowed'].sum())}")
    print(f"[manifest] summary md -> {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
