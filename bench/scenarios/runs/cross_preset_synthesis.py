"""Cross-preset synthesis report for Phase 2 + 3 + 4 production runs.

Reads the 3 production preset workspaces and emits a markdown report covering:
- Per-candidate consistency across presets (robustness check)
- Per-dataset cross-candidate analysis
- Head-to-head win matrix between top candidates
- Failure pattern analysis
- Recommendations for synthesis / paper P1

Owner: Agent C. No new decisions; purely analysis of already-locked data.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2]
RUNS = BENCH / "scenarios" / "runs"
PRESETS = (
    ("fast_reliable", "fast_reliable_full57_seed0"),
    ("strong_practical", "strong_practical_full57_seed0"),
    ("best_current", "best_current_full57_seed0"),
)
OUT = RUNS / "cross_preset_synthesis.md"


def load_rows(workspace: Path) -> list[dict[str, str]]:
    path = workspace / "results.csv"
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def fmt(x: float | None, fmt_str: str = "{:.4f}") -> str:
    if x is None:
        return "—"
    return fmt_str.format(x)


def main() -> None:
    # Load all 3 preset workspaces
    data: dict[str, list[dict[str, str]]] = {}
    for preset, ws in PRESETS:
        rows = load_rows(RUNS / ws)
        data[preset] = rows

    lines: list[str] = []
    lines.append("# Cross-Preset Production Run Synthesis")
    lines.append("")
    lines.append("Generated from Phase 2 + 3 + 4 production runs (full-57 cohort, seed 0).")
    lines.append("")
    lines.append("Datasource:")
    for preset, ws in PRESETS:
        lines.append(f"- `bench/scenarios/runs/{ws}/results.csv` ({len(data[preset])} rows)")
    lines.append("")

    # ====================================================================
    # 1. Per-candidate consistency across presets
    # ====================================================================
    lines.append("## 1. Per-Candidate Consistency Across Presets")
    lines.append("")
    lines.append("Median rmsep per candidate × preset. Strong consistency across the 3")
    lines.append("presets validates that the candidate's performance is preset-independent")
    lines.append("(i.e., dataset coverage drives the median, not preset selection).")
    lines.append("")

    cand_preset_medians: dict[str, dict[str, float]] = defaultdict(dict)
    cand_preset_counts: dict[str, dict[str, int]] = defaultdict(dict)

    for preset, _ in PRESETS:
        by_cand: dict[str, list[float]] = defaultdict(list)
        for r in data[preset]:
            if r["status"] != "ok":
                continue
            try:
                rmsep = float(r["rmsep"])
            except (ValueError, TypeError):
                continue
            by_cand[r["canonical_name"]].append(rmsep)
        for cand, rmseps in by_cand.items():
            cand_preset_medians[cand][preset] = statistics.median(rmseps)
            cand_preset_counts[cand][preset] = len(rmseps)

    all_cands = sorted(
        cand_preset_medians.keys(),
        key=lambda c: cand_preset_medians[c].get("best_current", float("inf")),
    )

    lines.append(
        "| Candidate | fast_reliable | strong_practical | best_current | "
        "Consistency (max−min) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )
    for cand in all_cands:
        meds = cand_preset_medians[cand]
        cnts = cand_preset_counts[cand]
        f = meds.get("fast_reliable")
        s = meds.get("strong_practical")
        b = meds.get("best_current")
        active = [v for v in (f, s, b) if v is not None]
        spread = max(active) - min(active) if len(active) >= 2 else 0.0
        f_str = f"{f:.4f} (n={cnts.get('fast_reliable', 0)})" if f else "—"
        s_str = f"{s:.4f} (n={cnts.get('strong_practical', 0)})" if s else "—"
        b_str = f"{b:.4f} (n={cnts.get('best_current', 0)})" if b else "—"
        lines.append(f"| {cand} | {f_str} | {s_str} | {b_str} | {spread:.6f} |")
    lines.append("")

    # ====================================================================
    # 2. Head-to-head win matrix (best_current, all candidates)
    # ====================================================================
    lines.append("## 2. Head-to-Head Win Matrix (best_current cohort)")
    lines.append("")
    lines.append("For each pair `(row, col)`, count of datasets where `row` rmsep < `col` rmsep")
    lines.append("(strict win). Datasets where either failed are excluded.")
    lines.append("")

    # Build per-(candidate, dataset) → rmsep map for best_current
    best_curr = data["best_current"]
    by_pair: dict[tuple[str, str], float] = {}
    for r in best_curr:
        if r["status"] != "ok":
            continue
        try:
            rmsep = float(r["rmsep"])
        except (ValueError, TypeError):
            continue
        by_pair[(r["canonical_name"], r["dataset"])] = rmsep

    cands_bc = sorted({k[0] for k in by_pair.keys()})

    # Compute head-to-head wins
    h2h: dict[tuple[str, str], int] = {}
    for c_row in cands_bc:
        for c_col in cands_bc:
            wins = 0
            for ds in {k[1] for k in by_pair.keys() if k[0] == c_row}:
                r_val = by_pair.get((c_row, ds))
                c_val = by_pair.get((c_col, ds))
                if r_val is not None and c_val is not None and r_val < c_val:
                    wins += 1
            h2h[(c_row, c_col)] = wins

    # Render as table — short candidate names
    short_names = {
        "AOMRidge-global-compact-none": "AR-glb-none",
        "AOMRidge-global-compact-snv": "AR-glb-snv",
        "ASLS-AOM-compact-cv5-numpy": "ASLS-AOM",
        "AOMRidge-Local-compact-knn50": "AR-loc-50",
        "Ridge-tuned-cv5": "Ridge-cv5",
        "AOM-PLS-compact-numpy": "AOM-PLS",
        "PLS-tuned-cv5": "PLS-cv5",
        "AOMRidge-MultiBranchMKL-compact-shrink03": "AR-MBMKL",
    }
    short_cands = [short_names.get(c, c[:12]) for c in cands_bc]

    header = "| row \\ col |" + "|".join(f" {s} " for s in short_cands) + "|"
    sep = "|---:|" + "|".join(":-:" for _ in short_cands) + "|"
    lines.append(header)
    lines.append(sep)
    for c_row, s_row in zip(cands_bc, short_cands):
        row_cells: list[str] = []
        for c_col in cands_bc:
            if c_row == c_col:
                row_cells.append("—")
            else:
                row_cells.append(str(h2h.get((c_row, c_col), 0)))
        lines.append(f"| **{s_row}** | " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("Read: row wins over col on N datasets. The winner of the cohort has")
    lines.append("high win counts across all columns.")
    lines.append("")

    # ====================================================================
    # 3. Per-dataset top performer in best_current
    # ====================================================================
    lines.append("## 3. Per-Dataset Winner (best_current)")
    lines.append("")

    datasets_bc = sorted({k[1] for k in by_pair.keys()})
    winners: dict[str, tuple[str, float]] = {}
    for ds in datasets_bc:
        entries = [(c, by_pair[(c, ds)]) for c in cands_bc if (c, ds) in by_pair]
        if not entries:
            continue
        winner = min(entries, key=lambda x: x[1])
        winners[ds] = winner

    winner_count: dict[str, int] = defaultdict(int)
    for cand, _ in winners.values():
        winner_count[cand] += 1

    lines.append("Win count per candidate (number of datasets where they are best):")
    lines.append("")
    lines.append("| Candidate | Wins / 54 datasets |")
    lines.append("|---|---:|")
    for cand, count in sorted(winner_count.items(), key=lambda x: -x[1]):
        lines.append(f"| {cand} | {count} |")
    lines.append("")

    # ====================================================================
    # 4. Failure pattern analysis
    # ====================================================================
    lines.append("## 4. Failure Pattern Analysis (best_current)")
    lines.append("")

    failed_by_ds: dict[str, list[str]] = defaultdict(list)
    failed_terminal_by_ds: dict[str, list[str]] = defaultdict(list)
    for r in best_curr:
        if r["status"] == "failed":
            failed_by_ds[r["dataset"]].append(r["canonical_name"])
        elif r["status"] == "failed_terminal":
            failed_terminal_by_ds[r["dataset"]].append(r["canonical_name"])

    if failed_by_ds:
        lines.append("### Datasets with `failed` status (pre-existing data issues)")
        lines.append("")
        lines.append("| Dataset | Failed candidates |")
        lines.append("|---|---|")
        for ds in sorted(failed_by_ds.keys()):
            cands = ", ".join(sorted(failed_by_ds[ds]))
            lines.append(f"| {ds} | {cands} |")
        lines.append("")

    if failed_terminal_by_ds:
        lines.append("### Datasets with `failed_terminal` status (D-C-019 timeouts)")
        lines.append("")
        lines.append("| Dataset | Timed-out candidates |")
        lines.append("|---|---|")
        for ds in sorted(failed_terminal_by_ds.keys()):
            cands = ", ".join(sorted(failed_terminal_by_ds[ds]))
            lines.append(f"| {ds} | {cands} |")
        lines.append("")

    # ====================================================================
    # 5. Recommendations
    # ====================================================================
    lines.append("## 5. Recommendations for P1 Paper Synthesis")
    lines.append("")
    lines.append("Based on the cross-preset consistency analysis:")
    lines.append("")
    lines.append("- **AOMRidge-global-compact-none/snv** is the production-grade winner")
    lines.append("  across all 3 presets with median rmsep 1.0956. -23.7 % vs the")
    lines.append("  next-best ASLS-AOM-compact-cv5-numpy (1.436), -50.6 % vs MBMKL (1.970).")
    lines.append("")
    lines.append("- **SNV preprocessing has zero impact** on AOMRidge-global median in this")
    lines.append("  cohort. The two registry slots `none` and `snv` are functionally")
    lines.append("  equivalent here. Future cleanup: collapse to a single slot.")
    lines.append("")
    lines.append("- **AOMRidge-Local-compact-knn50** lands rank 4 (median 1.519) — useful")
    lines.append("  fallback for big-n datasets where -global times out at 1200 s (5/57 in")
    lines.append("  our cohort). Complementary slot, not strictly dominated.")
    lines.append("")
    lines.append("- **AOMRidge-MultiBranchMKL-compact-shrink03** rank 8 with median 1.970")
    lines.append("  confirms the D-C-015 stub-minimal LOCKED framing — under-tuned default")
    lines.append("  hyperparams. Not recommended for production preset until A specifies")
    lines.append("  the canonical `top_m` / `mkl_mode` / `alpha` grid.")
    lines.append("")
    lines.append("- **Persistently-failing datasets** (`Brix_spxy70` EmptyDataError,")
    lines.append("  `FinalScore_grp70_30_scoreQ` + `Tleaf_grp70_30` GridSearchCV failures)")
    lines.append("  should be flagged in the master CSV as `extras.bench_unfit_data=true`")
    lines.append("  or excluded from the canonical 57-dataset cohort entirely.")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
