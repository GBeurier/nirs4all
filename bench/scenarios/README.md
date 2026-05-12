# bench/scenarios/

Auto-generated manifests describing the four benchmark presets defined in `bench/PLAN_REPRISE_2026-05.md` §9. Each JSON file is consumed by `bench/harness/run_benchmark.py`; do not hand-edit them.

Master CSV SHA256: `b27ea6f52b45e2568fb0c6912f535565f678d8b3e4f28af70dc2b86ae201ab5d`
Generator: `bench/export_benchmark_scenarios.py`
Generated on: 2026-05-12

## Presets

| Preset | Description | Members | Penalised |
|---|---|---:|---:|
| `fast_reliable` | ≤30 s per dataset; safe linear baselines, no NN, no TabPFN. | 3 | 0 |
| `strong_practical` | ≤10 min per dataset; AOM-PLS + AOM-Ridge ladder, no TabPFN-HPO yet. | 8 | 0 |
| `best_current` | ≤2 h per dataset; TabPFN-HPO-preprocessing + AOM-Ridge blender. | 9 | 2 |
| `exhaustive_research` | ≤12 h per dataset; adds NN residual, super-learner, AutoSelect. | 12 | 5 |

## Codex review

Every manifest carries `codex_review_status: DECISION_PENDING_CODEX_REVIEW` until the registry, the penalty thresholds, and the candidate ordering are validated through `bench/SYNC.md`.

## Schema

The exporter follows the schema documented at the top of `bench/scenarios/model_registry.yaml` and the manifest layout described in `bench/export_benchmark_scenarios.py`.
