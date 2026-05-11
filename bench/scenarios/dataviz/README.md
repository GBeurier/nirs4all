# Benchmark Dataviz

Comprehensive cross-preset dashboard for nirs4all benchmark production runs.

## Files

- `build_dashboard.py` — aggregator that reads the 4 preset workspaces +
  master CSV and emits `dashboard_data.json`.
- `dashboard_data.json` — generated data blob (~80 KB).
- `dashboard.html` — single-file dashboard, Plotly via CDN, loads
  `dashboard_data.json` via fetch.

## Usage

```bash
# Refresh data after a new preset run lands
python3 bench/scenarios/dataviz/build_dashboard.py

# Open in browser (any HTTP server works ; needs to be served because
# of the fetch() for the JSON):
python3 -m http.server -d bench/scenarios/dataviz 8000
# → http://localhost:8000/dashboard.html
```

## Sections

1. KPI row (total fits, success rate, best candidate, coverage)
2. Per-preset leaderboards (fast_reliable, strong_practical, best_current,
   exhaustive_research) — tabbed view
3. Model family aggregation (bars + coverage)
4. Per-dataset performance heatmap (normalised by row, sortable by n_train /
   n_features / best_rmsep)
5. Performance vs n_train scatter (log-log, per-candidate lines)
6. Preprocessing influence (none vs SNV delta)
7. Speed vs accuracy Pareto scatter
8. Head-to-head win matrix (best_current 8×8)
9. MoE / Stack performance breakdown
10. Cross-preset robustness (grouped bars showing consistency)
11. Failure & timeout patterns (per dataset)
12. TabPFN reference data (paper-master_pivot legacy box plots)
13. Synthesis & recommendations (text)

## Refresh cadence

Re-run `build_dashboard.py` after any new preset workspace lands or master CSV
is rebuilt. The HTML is static and doesn't need regeneration.

## Phase 5 in-flight handling

If `bench/scenarios/runs/exhaustive_research_full57_seed0/results.csv` exists
but has fewer than the planned 31×57=1767 fits (estimated), the page will show
the partial leaderboard for that preset. Re-run the aggregator periodically as
Phase 5 progresses.
