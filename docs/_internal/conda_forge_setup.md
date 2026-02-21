# Conda-forge & Docker Release Setup

This guide explains how to set up automated publishing to conda-forge and GitHub Container Registry (GHCR) for nirs4all releases.

## Release Flow Overview

A single GitHub Release triggers the entire distribution pipeline:

```
GitHub Release (tag vX.Y.Z)
  └─ publish.yml
       ├─ run-tests (matrix: ubuntu/windows × py3.11/3.13)
       ├─ build-docs
       ├─ build → publish-pypi (PyPI, ~5 min)
       └─ publish-docker (GHCR, ~5 min, parallel with build)
  └─ conda-forge bot (automatic, ~6-12h after PyPI)
       └─ version-bump PR → automerge → conda-forge channel
```

## Accounts & Credentials

### What you need

| Service | Account required | Credentials |
|---------|-----------------|-------------|
| **PyPI** | Yes (existing) | OIDC trusted publishing (no token needed) |
| **GHCR (Docker)** | No (uses GitHub) | Built-in `GITHUB_TOKEN` (automatic) |
| **conda-forge** | No (uses GitHub) | None — conda-forge CI handles uploads |

**No additional accounts or secrets are required** beyond the existing GitHub and PyPI setup.

## Docker / GHCR Setup

Docker publishing uses the built-in `GITHUB_TOKEN` provided by GitHub Actions. No additional configuration is needed.

The `publish-docker` job in `publish.yml`:
1. Builds a multi-stage Docker image from the `Dockerfile`
2. Pushes to `ghcr.io/<owner>/nirs4all:<version>` and `ghcr.io/<owner>/nirs4all:latest`
3. Uses GitHub Actions cache for faster builds

### Verify GHCR is enabled

GitHub Packages (which hosts GHCR) is enabled by default on public repositories. To verify:
1. Go to repository Settings → Actions → General
2. Under "Workflow permissions", ensure "Read and write permissions" is selected

### Test locally

```bash
docker build -t nirs4all .
docker run nirs4all -c "import nirs4all; print(nirs4all.__version__)"
docker run -v $(pwd):/workspace nirs4all my_script.py
```

## Conda-forge Setup

conda-forge uses its own build infrastructure — you cannot push packages directly from GitHub Actions. Instead, a feedstock repository is created under the `conda-forge` GitHub organization, and a bot automatically detects new PyPI releases and updates the feedstock.

### Current status (2025-02-21)

**Local build: PASSED** — all recipes build and test successfully with `conda-build`.

#### Dependency availability on conda-forge

| Package | conda-forge | Notes |
|---------|:-----------:|-------|
| numpy, pandas, scipy, scikit-learn | YES | Standard scientific stack |
| polars, pyarrow, python-duckdb, h5py | YES | Data processing |
| umap-learn, pywavelets, pybaselines | YES | Signal processing |
| optuna, matplotlib-base, seaborn, shap | YES | Optimization, viz, explainability |
| joblib, jsonschema, pyyaml, packaging, pydantic | YES | Utilities |
| kennard-stone | YES | NIRS-specific splitting |
| **ikpls** | **NO** | Core dep in v0.8.0+. Has sdist. Depends on `cvmatrix` (also missing). |
| **cvmatrix** | **NO** | Dependency of ikpls. Has sdist. |
| **pyopls** | **NO** | Core dep in v0.8.0+. Has sdist. License file missing from sdist (fetched from GitHub). |
| **trendfitter** | **NO** | Core dep in v0.8.0+. No sdist. Recipe uses GitHub source. |

**PyPI → conda-forge name differences:**
- `duckdb` → `python-duckdb`
- `PyWavelets` → `pywavelets`
- `matplotlib` → `matplotlib-base`

### Step 1: Submit missing dependency recipes

Recipes for all missing dependencies are in `conda-forge/staged-recipes/`. They must be submitted **before** the nirs4all recipe (conda-forge will reject nirs4all if deps are missing).

**Submission order** (respecting dependency chain):

1. **Batch 1** (no inter-dependencies — submit simultaneously):
   - `pyopls` — from PyPI sdist + GitHub LICENSE
   - `trendfitter` — from GitHub source archive
   - `cvmatrix` — from PyPI sdist

2. **Batch 2** (depends on cvmatrix):
   - `ikpls` — from PyPI sdist, depends on cvmatrix

3. **Batch 3** (depends on all above):
   - `nirs4all` — main package

For each, fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes), create `recipes/<package>/meta.yaml`, and open a PR. Multiple recipes can be submitted as separate PRs simultaneously.

**Important notes:**
- `pyopls` LICENSE is missing from the PyPI sdist. The recipe fetches it from GitHub. This is an accepted pattern on conda-forge.

### Step 2: Submit nirs4all recipe

Once all dependencies are available on conda-forge:

1. Fork [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes)
2. Copy `conda-forge/meta.yaml` to `recipes/nirs4all/meta.yaml`
3. If submitting a newer version, update `version` and `sha256` from `pypi.org/project/nirs4all/#files`
4. Open a Pull Request
5. conda-forge CI will build and validate on Linux, macOS, and Windows
6. A reviewer will review and merge (timeline: days to weeks)
7. `nirs4all-feedstock` is automatically created under `github.com/conda-forge/`
8. Accept the maintainer invitation

### Step 3: Enable automerge

After the feedstock is created and you have maintainer access:

1. Clone the feedstock:
   ```bash
   git clone https://github.com/conda-forge/nirs4all-feedstock
   cd nirs4all-feedstock
   ```

2. Edit `conda-forge.yml` to add:
   ```yaml
   bot:
     automerge: true
   ```

3. Rerender:
   ```bash
   conda-smithy rerender
   ```
   Or comment `@conda-forge-admin, please rerender` on any feedstock PR.

4. Commit and push.

### How ongoing releases work

After initial setup, the process is fully automated:

1. You create a GitHub Release (tag `vX.Y.Z`)
2. `publish.yml` publishes to PyPI
3. Within ~6 hours, the [regro-cf-autotick-bot](https://github.com/regro-cf-autotick-bot) detects the new PyPI version
4. The bot opens a PR on `nirs4all-feedstock` updating `version` and `sha256`
5. Azure Pipelines builds and tests the package
6. If automerge is enabled and CI passes, the PR auto-merges
7. The package becomes available via `conda install -c conda-forge nirs4all`

Total time from release to conda-forge availability: **~6-12 hours**.

### Local testing

```bash
# Build all recipes locally (including deps)
conda-build conda-forge/ --no-anaconda-upload --channel local

# Test install in a fresh environment
conda create -n test-nirs4all -y -c local -c conda-forge python=3.13 nirs4all
conda activate test-nirs4all
python -c "import nirs4all; print(nirs4all.__version__)"
nirs4all --help
pip check
conda deactivate && conda env remove -n test-nirs4all -y
```

### Useful bot commands

Comment these on any PR or issue in the feedstock repository:

| Command | Effect |
|---------|--------|
| `@conda-forge-admin, please rerender` | Regenerate CI configuration |
| `@conda-forge-admin, please rerun bot` | Re-trigger the autotick bot on a PR |
| `@conda-forge-admin, please lint` | Re-run the recipe linter |
| `@conda-forge-admin, please add noarch: python` | Convert to noarch build |

### Troubleshooting

**Bot not picking up new version:**
- Check if there are 3+ open version-bump PRs on the feedstock (the bot stops at 3). Close or merge stale ones.
- You can always manually update the feedstock recipe and open your own PR.

**Missing dependencies on conda-forge:**
- Submit recipes for missing deps to `staged-recipes` first, using the same grayskull workflow.
- The nirs4all recipe cannot be merged until all its run dependencies exist on conda-forge.

**CI failures on feedstock:**
- Check the Azure Pipelines logs linked from the PR.
- Common issues: dependency name mismatches (PyPI vs conda-forge names), version pins too tight.
