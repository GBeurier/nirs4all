"""bench/build_run_dashboard.py.

Build a self-contained HTML dashboard from a harness run's results.csv.

Input: a workspace directory written by `bench/harness/run_benchmark.py`,
expected to contain a `results.csv` with the unified schema declared in
`bench.harness.run_benchmark.RESULT_FIELDS`.

Output (under the same workspace):
  * `dashboard.html` — single-file HTML, no external deps. Includes
    aggregate counts, per-candidate leaderboard, model × dataset RMSEP
    heatmap, runtime summary, and a per-error failure breakdown.
  * `dashboard_data.json` — the raw aggregations the HTML embeds; useful
    for downstream tooling.

CLI:

    python bench/build_run_dashboard.py <workspace>

Status: SKELETON, sub-decision under D-C-006 (harness contract). Codex
must validate the dashboard's aggregation conventions:
  * leaderboard sort key = median RMSEP across (status=ok) rows;
  * heatmap value = best RMSEP per (canonical_name, dataset);
  * failure rollup = top error_message prefixes (truncated at 60 chars).

DECISION_PENDING_CODEX_REVIEW (D-C-013).
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench/build_run_dashboard.py",
        description="Render a self-contained HTML dashboard from a harness results.csv.",
    )
    parser.add_argument(
        "workspace",
        type=Path,
        help="Workspace directory containing results.csv (and optional stats.json).",
    )
    parser.add_argument(
        "--out-html",
        type=Path,
        default=None,
        help="Override output HTML path. Default: <workspace>/dashboard.html.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Override output JSON path. Default: <workspace>/dashboard_data.json.",
    )
    return parser.parse_args(argv)


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"results.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(value: str | None) -> float | None:
    if not value:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def aggregate(rows: list[dict[str, str]]) -> dict[str, Any]:
    by_status: Counter[str] = Counter()
    by_canonical: dict[str, list[float]] = defaultdict(list)
    by_canonical_runtime: dict[str, list[float]] = defaultdict(list)
    by_canonical_datasets: dict[str, set[str]] = defaultdict(set)
    by_canonical_class: dict[str, str] = {}
    rmsep_matrix: dict[tuple[str, str], float] = {}
    failures: list[dict[str, str]] = []
    error_counts: Counter[str] = Counter()
    datasets_all: set[str] = set()

    for row in rows:
        status = row.get("status", "")
        by_status[status] += 1
        canonical = row.get("canonical_name", "")
        dataset = row.get("dataset", "")
        if dataset:
            datasets_all.add(dataset)
        if status == "ok":
            rmsep = _f(row.get("rmsep"))
            ft = _f(row.get("fit_time_s"))
            if canonical and rmsep is not None:
                by_canonical[canonical].append(rmsep)
                by_canonical_class.setdefault(canonical, row.get("model_class", ""))
                if dataset:
                    by_canonical_datasets[canonical].add(dataset)
                    key = (canonical, dataset)
                    if key not in rmsep_matrix or rmsep < rmsep_matrix[key]:
                        rmsep_matrix[key] = rmsep
            if canonical and ft is not None:
                by_canonical_runtime[canonical].append(ft)
        elif status == "failed":
            err = (row.get("error_message") or "")[:60]
            failures.append(
                {
                    "canonical_name": canonical,
                    "dataset": dataset,
                    "error_message": row.get("error_message", ""),
                }
            )
            error_counts[_classify_error(err)] += 1

    leaderboard: list[dict[str, Any]] = []
    for canonical, rmseps in by_canonical.items():
        runtimes = by_canonical_runtime.get(canonical, [])
        leaderboard.append(
            {
                "canonical_name": canonical,
                "model_class": by_canonical_class.get(canonical, ""),
                "n_datasets": len(by_canonical_datasets[canonical]),
                "median_rmsep": statistics.median(rmseps),
                "min_rmsep": min(rmseps),
                "max_rmsep": max(rmseps),
                "median_fit_time_s": statistics.median(runtimes) if runtimes else None,
                "max_fit_time_s": max(runtimes) if runtimes else None,
            }
        )
    leaderboard.sort(key=lambda r: r["median_rmsep"])

    canonicals = [row["canonical_name"] for row in leaderboard]
    datasets_sorted = sorted(datasets_all)
    heatmap_rows: list[list[float | None]] = []
    for canonical in canonicals:
        row_vals: list[float | None] = []
        for ds in datasets_sorted:
            row_vals.append(rmsep_matrix.get((canonical, ds)))
        heatmap_rows.append(row_vals)

    return {
        "by_status": dict(by_status),
        "leaderboard": leaderboard,
        "heatmap": {
            "canonicals": canonicals,
            "datasets": datasets_sorted,
            "rmsep": heatmap_rows,
        },
        "failures": {
            "total": sum(error_counts.values()),
            "by_class": dict(error_counts.most_common()),
            "samples": failures[:30],
        },
        "n_datasets": len(datasets_all),
        "n_candidates": len(by_canonical),
    }


def _classify_error(err: str) -> str:
    if not err:
        return "<empty>"
    head = err.split(":", 1)[0]
    return head.strip() or err[:30]


def build_meta(rows: list[dict[str, str]], workspace: Path) -> dict[str, Any]:
    presets = {row.get("preset") for row in rows if row.get("preset")}
    cohorts = {row.get("cohort") for row in rows if row.get("cohort")}
    seeds = {row.get("seed") for row in rows if row.get("seed") not in (None, "")}
    hosts = {row.get("host") for row in rows if row.get("host")}
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "workspace": str(workspace),
        "n_rows": len(rows),
        "presets": sorted(p for p in presets if p),
        "cohorts": sorted(c for c in cohorts if c),
        "seeds": sorted(seeds),
        "hosts": sorted(h for h in hosts if h),
    }


_HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>nirs4all bench run dashboard — {title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:24px;color:#222}}
h1{{font-size:18px;margin:0 0 4px}}
h2{{font-size:14px;margin:18px 0 6px;color:#444}}
.meta{{font-size:12px;color:#666;margin-bottom:18px}}
table{{border-collapse:collapse;font-size:12px;margin-bottom:14px}}
th,td{{border:1px solid #ddd;padding:4px 8px;text-align:right}}
th{{background:#f5f5f5;text-align:center}}
td.text{{text-align:left}}
td.center{{text-align:center}}
.heatmap td{{font-size:10px;padding:2px 4px;min-width:38px}}
.legend{{font-size:11px;color:#666;margin-top:4px}}
.failures{{font-family:monospace;font-size:11px}}
.failures pre{{margin:0;white-space:pre-wrap}}
.status-ok{{color:#07710e}}
.status-failed{{color:#a01a1a}}
.status-skipped{{color:#888}}
.status-dry_run{{color:#246a8a}}
.status-probe{{color:#246a8a}}
</style></head><body>
<h1>nirs4all bench run dashboard</h1>
<div class="meta">{meta_block}</div>
{status_block}
{leaderboard_block}
{heatmap_block}
{failures_block}
<div class="legend">DECISION_PENDING_CODEX_REVIEW (D-C-013). Generated by <code>bench/build_run_dashboard.py</code>.</div>
</body></html>
"""


def render_html(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    title = (meta["presets"][0] if meta["presets"] else "run") + " × " + (
        meta["cohorts"][0] if meta["cohorts"] else "cohort"
    )

    meta_lines = [
        f"<b>Generated</b>: {html.escape(meta['generated_at'])}",
        f"<b>Workspace</b>: <code>{html.escape(meta['workspace'])}</code>",
        f"<b>Presets</b>: {html.escape(', '.join(meta['presets']) or '—')}",
        f"<b>Cohorts</b>: {html.escape(', '.join(meta['cohorts']) or '—')}",
        f"<b>Seeds</b>: {html.escape(', '.join(meta['seeds']) or '—')}",
        f"<b>Hosts</b>: {html.escape(', '.join(meta['hosts']) or '—')}",
        f"<b>Rows</b>: {meta['n_rows']} | candidates: {payload['n_candidates']} | datasets: {payload['n_datasets']}",
    ]
    meta_block = " · ".join(meta_lines)

    status_rows = "".join(
        f"<tr><td class='text'>{html.escape(s or '<empty>')}</td>"
        f"<td>{c}</td></tr>"
        for s, c in payload["by_status"].items()
    )
    status_block = (
        "<h2>Status counts</h2><table><tr><th>Status</th><th>Count</th></tr>"
        f"{status_rows}</table>"
    )

    leaderboard_rows: list[str] = []
    for entry in payload["leaderboard"]:
        cn = html.escape(entry["canonical_name"])
        cls = html.escape(entry["model_class"])
        med_rmsep = f"{entry['median_rmsep']:.4f}"
        rng = f"[{entry['min_rmsep']:.4f}, {entry['max_rmsep']:.4f}]"
        med_ft = (
            f"{entry['median_fit_time_s']:.2f}" if entry["median_fit_time_s"] is not None else "—"
        )
        max_ft = (
            f"{entry['max_fit_time_s']:.2f}" if entry["max_fit_time_s"] is not None else "—"
        )
        leaderboard_rows.append(
            f"<tr><td class='text'>{cn}</td><td class='text'>{cls}</td>"
            f"<td>{entry['n_datasets']}</td><td>{med_rmsep}</td>"
            f"<td>{rng}</td><td>{med_ft}</td><td>{max_ft}</td></tr>"
        )
    leaderboard_block = (
        "<h2>Leaderboard (status=ok rows; sorted by median RMSEP, lower is better)</h2>"
        "<table><tr><th>Candidate</th><th>Class</th><th>n_datasets</th>"
        "<th>median rmsep</th><th>[min, max]</th><th>median fit s</th><th>max fit s</th></tr>"
        f"{''.join(leaderboard_rows)}</table>"
    )

    heatmap = payload["heatmap"]
    heatmap_block = "<h2>RMSEP heatmap (model × dataset)</h2>"
    if not heatmap["canonicals"] or not heatmap["datasets"]:
        heatmap_block += "<p>No status=ok rows; heatmap empty.</p>"
    else:
        # Color scale: green for low (good), red for high (bad). Per-dataset
        # column normalisation to give every dataset the same dynamic range.
        cells = [list(row) for row in heatmap["rmsep"]]
        col_min: list[float | None] = [None] * len(heatmap["datasets"])
        col_max: list[float | None] = [None] * len(heatmap["datasets"])
        for row in cells:
            for j, v in enumerate(row):
                if v is None:
                    continue
                if col_min[j] is None or v < col_min[j]:
                    col_min[j] = v
                if col_max[j] is None or v > col_max[j]:
                    col_max[j] = v

        def cell_html(value: float | None, j: int) -> str:
            if value is None:
                return "<td>—</td>"
            mn = col_min[j]
            mx = col_max[j]
            if mn is None or mx is None or mx == mn:
                bg = "#ffffff"
            else:
                t = (value - mn) / (mx - mn)
                # green=0, yellow=0.5, red=1
                if t < 0.5:
                    r = int(255 * (2 * t))
                    g = 200
                else:
                    r = 255
                    g = int(200 * (2 * (1 - t)))
                bg = f"rgb({r},{g},80)"
            return f"<td style='background:{bg}'>{value:.3f}</td>"

        header = "<tr><th>Candidate</th>" + "".join(
            f"<th>{html.escape(d)}</th>" for d in heatmap["datasets"]
        ) + "</tr>"
        body = ""
        for cn, row in zip(heatmap["canonicals"], cells, strict=False):
            body += f"<tr><td class='text'>{html.escape(cn)}</td>" + "".join(
                cell_html(v, j) for j, v in enumerate(row)
            ) + "</tr>"
        heatmap_block += f"<table class='heatmap'>{header}{body}</table>"
        heatmap_block += "<div class='legend'>Per-column min/max normalisation; lower (greener) = better RMSEP within that dataset.</div>"

    failures = payload["failures"]
    failures_block = (
        f"<h2>Failures ({failures['total']} rows)</h2>"
    )
    if failures["total"] == 0:
        failures_block += "<p>No failures.</p>"
    else:
        rows = "".join(
            f"<tr><td class='text'>{html.escape(cls)}</td><td>{count}</td></tr>"
            for cls, count in failures["by_class"].items()
        )
        failures_block += (
            "<table><tr><th>Error class</th><th>Count</th></tr>"
            f"{rows}</table>"
        )
        if failures["samples"]:
            samples = "<br>".join(
                f"<code>{html.escape(s['canonical_name'])}</code> on "
                f"<code>{html.escape(s['dataset'])}</code>: "
                f"{html.escape(s['error_message'])[:140]}"
                for s in failures["samples"]
            )
            failures_block += (
                "<details class='failures'><summary>First 30 failure messages</summary>"
                f"<pre>{samples}</pre></details>"
            )

    return _HTML_TEMPLATE.format(
        title=html.escape(title),
        meta_block=meta_block,
        status_block=status_block,
        leaderboard_block=leaderboard_block,
        heatmap_block=heatmap_block,
        failures_block=failures_block,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace: Path = args.workspace
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    csv_path = workspace / "results.csv"
    rows = load_results(csv_path)
    payload = aggregate(rows)
    payload["meta"] = build_meta(rows, workspace)

    json_path = args.out_json or (workspace / "dashboard_data.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    html_path = args.out_html or (workspace / "dashboard.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"Wrote {html_path} ({html_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
