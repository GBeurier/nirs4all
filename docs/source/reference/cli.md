# CLI Reference

The `nirs4all` command currently provides diagnostics, configuration validation, dataset inspection, workspace queries, and artifact maintenance.

:::{important}
This repository does not expose a native `nirs4all run` training command yet. Use `nirs4all.run(...)` from Python, or call that Python API from R/Julia/JavaScript wrappers as shown in {doc}`public_interfaces`.
:::

## Root Commands

```bash
nirs4all --version
nirs4all --test-install
nirs4all --test-integration
```

| Command | Purpose |
| --- | --- |
| `--version` | Print the installed package version. |
| `--test-install` | Print dependency/install diagnostics. |
| `--test-integration` | Run the packaged sample-data integration smoke test. |

## Config Commands

```bash
nirs4all config validate pipeline.yaml --type pipeline --check-imports
nirs4all config validate dataset.yaml --type dataset
nirs4all config schema pipeline
nirs4all config schema dataset
```

| Command | Options | Purpose |
| --- | --- | --- |
| `config validate <config_file>` | `--type pipeline|dataset|auto`, `--check-files`, `--no-check-files`, `--check-imports` | Validate JSON/YAML config files. |
| `config schema pipeline|dataset` | none | Print the JSON schema. |

## Dataset Commands

```bash
nirs4all dataset validate dataset.yaml
nirs4all dataset validate dataset.yaml --format json
nirs4all dataset inspect dataset.yaml --detect
nirs4all dataset export dataset.yaml --format json --output normalized_dataset.json
nirs4all dataset diff dataset_a.yaml dataset_b.yaml
```

| Command | Options | Purpose |
| --- | --- | --- |
| `dataset validate <config_file>` | `--check-files`, `--no-check-files`, `--verbose`, `--format text|json` | Validate dataset configuration or folder path. |
| `dataset inspect <config_file>` | `--detect` | Show sources, task type, aggregation, and optionally detected delimiter/header/signal parameters. |
| `dataset export <config_file>` | `--output`, `--format yaml|json` | Write normalized dataset config. |
| `dataset diff <config1> <config2>` | `--format text|json` | Compare two normalized dataset configs. |

## Workspace Commands

```bash
nirs4all workspace init workspace
nirs4all workspace list-runs --workspace workspace
nirs4all workspace query-best --workspace workspace --metric test_score -n 5
nirs4all workspace filter --workspace workspace --dataset wheat --test-score 0.50
nirs4all workspace stats --workspace workspace --metric test_score
nirs4all workspace list-library --workspace workspace
```

| Command | Options | Purpose |
| --- | --- | --- |
| `workspace init <path>` | none | Create a workspace directory. |
| `workspace list-runs` | `--workspace` | List recorded runs. |
| `workspace query-best` | `--workspace`, `--dataset`, `--metric`, `-n`, `--ascending` | Query top prediction rows by metric. |
| `workspace filter` | `--workspace`, `--dataset`, `--test-score`, `--train-score`, `--val-score` | Filter prediction rows. |
| `workspace stats` | `--workspace`, `--metric` | Show score statistics. |
| `workspace list-library` | `--workspace` | List saved library templates/items. |

## Artifact Commands

```bash
nirs4all artifacts list-orphaned --workspace workspace
nirs4all artifacts cleanup --workspace workspace
nirs4all artifacts cleanup --workspace workspace --force --verbose
nirs4all artifacts stats --workspace workspace
nirs4all artifacts purge --workspace workspace --dataset wheat --force --yes
```

| Command | Options | Purpose |
| --- | --- | --- |
| `artifacts list-orphaned` | `--workspace`, `--dataset` | List binary artifacts not referenced by manifests. |
| `artifacts cleanup` | `--workspace`, `--dataset`, `--force`, `--verbose` | Dry-run by default; with `--force`, delete orphaned artifacts. |
| `artifacts stats` | `--workspace`, `--dataset` | Show storage and deduplication statistics. |
| `artifacts purge` | `--workspace`, `--dataset`, `--force`, `--yes` | Delete all artifacts for one dataset. |

## Config-First Workflow

```bash
# 1. Validate the portable files
nirs4all dataset validate dataset.yaml
nirs4all config validate pipeline.yaml --type pipeline --check-imports

# 2. Run through the Python API
python - <<'PY'
import nirs4all

result = nirs4all.run("pipeline.yaml", "dataset.yaml", name="cli_wrapped_run")
print(result.best_score)
result.export("exports/cli_wrapped_run.n4a")
PY

# 3. Inspect the workspace if the run archived predictions
nirs4all workspace query-best --workspace workspace -n 5
```

## Runtime Environment

```bash
N4A_ENGINE=dag-ml python train.py
N4A_NATIVE_RESULTS=./nirs4all_results python train.py
```

See {doc}`public_interfaces` for engine behavior and language wrapper patterns.
