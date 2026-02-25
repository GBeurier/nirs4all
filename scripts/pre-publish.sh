#!/usr/bin/env bash
# scripts/pre-publish.sh
# Mirrors .github/workflows/pre-publish.yml locally.
#
# Usage:
#   ./scripts/pre-publish.sh [OPTIONS]
#
# Options:
#   --skip-tests        Skip the pytest suite
#   --skip-docs         Skip Sphinx documentation build
#   --skip-examples     Skip examples runner
#   --skip-build        Skip package build / twine check
#   --only STEP         Run only one step: ruff | mypy | tests | docs | examples | build
#   --examples-cat CAT  Example categories to run (default: "user developer reference")
#   --python PYTHON     Python interpreter to use (default: python3)
#   --docker            Run inside a clean ubuntu:24.04 Docker container (closest to GitHub Actions)
#   -h, --help          Show this help

set -euo pipefail

ALL_ARGS=("$@")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[pre-publish]${RESET} $*"; }
success() { echo -e "${GREEN}[pre-publish]${RESET} $*"; }
warn()    { echo -e "${YELLOW}[pre-publish]${RESET} $*"; }
err()     { echo -e "${RED}[pre-publish]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; \
            echo -e "${BOLD}${CYAN}  $*${RESET}"; \
            echo -e "${BOLD}${CYAN}══════════════════════════════════════════════════════${RESET}"; }

# ──────────────────────────────────────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────────────────────────────────────

SKIP_TESTS=false
SKIP_DOCS=false
SKIP_EXAMPLES=false
SKIP_BUILD=false
ONLY_STEP=""
EXAMPLES_CATS="user developer reference"
PYTHON="${PYTHON:-python3}"
USE_DOCKER=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tests)     SKIP_TESTS=true; shift ;;
    --skip-docs)      SKIP_DOCS=true; shift ;;
    --skip-examples)  SKIP_EXAMPLES=true; shift ;;
    --skip-build)     SKIP_BUILD=true; shift ;;
    --only)           ONLY_STEP="$2"; shift 2 ;;
    --examples-cat)   EXAMPLES_CATS="$2"; shift 2 ;;
    --python)         PYTHON="$2"; shift 2 ;;
    --docker)         USE_DOCKER=true; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^[^#]/p' "$0" | head -n -1 | sed 's/^# \{0,3\}//'
      exit 0 ;;
    *) err "Unknown option: $1"; exit 1 ;;
  esac
done

# ──────────────────────────────────────────────────────────────────────────────
# Resolve repo root (the directory that contains pyproject.toml)
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
  err "Could not locate pyproject.toml under $REPO_ROOT"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Docker mode – re-run this very script inside a clean ubuntu:24.04 container
# ──────────────────────────────────────────────────────────────────────────────

if $USE_DOCKER; then
  info "Docker mode: running inside ubuntu:24.04 container"

  if ! command -v docker &>/dev/null; then
    err "Docker not found. Install Docker first or run without --docker."
    exit 1
  fi

  # Forward all original flags except --docker
  FORWARD_ARGS=()
  for arg in "${ALL_ARGS[@]}"; do
    [[ "$arg" != "--docker" ]] && FORWARD_ARGS+=("$arg")
  done

  exec docker run --rm \
    --cpus=8 \
    -v "$REPO_ROOT:/workspace:ro" \
    ubuntu:24.04 \
    bash -c "
      set -e
      export LOKY_MAX_CPU_COUNT=8
      export OMP_NUM_THREADS=2
      export OPENBLAS_NUM_THREADS=2
      export MKL_NUM_THREADS=2
      echo '[docker] Installing system packages ...'
      apt-get update -qq && apt-get install -y -qq bash python3 python3-pip python3-venv git rsync 2>&1 | tail -1

      echo '[docker] Copying repo to writable /build ...'
      rsync -a --info=progress2 \
        --exclude '.venv' --exclude '.git' --exclude '__pycache__' \
        --exclude 'node_modules' --exclude '.ruff_cache' \
        --exclude '.pytest_cache' --exclude 'dist' --exclude '*.egg-info' \
        --exclude 'bench' --exclude 'examples/workspace' \
        /workspace/ /build/

      echo '[docker] Setting up venv ...'
      python3 -m venv /opt/venv
      export PATH=/opt/venv/bin:\$PATH
      cd /build
      bash scripts/pre-publish.sh ${FORWARD_ARGS[*]:-}
    "
fi

# ──────────────────────────────────────────────────────────────────────────────
# State tracking
# ──────────────────────────────────────────────────────────────────────────────

declare -A STEP_RESULT   # "pass" | "fail" | "skip"
declare -A STEP_LOG      # path to log file

TMPDIR_LOGS="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOGS"' EXIT

run_step() {
  local name="$1"; shift
  local logfile="$TMPDIR_LOGS/${name}.log"
  STEP_LOG[$name]="$logfile"

  header "$name"
  if "$@" 2>&1 | tee "$logfile"; then
    STEP_RESULT[$name]="pass"
    success "$name — PASSED"
  else
    STEP_RESULT[$name]="fail"
    err "$name — FAILED  (full log: $logfile)"
  fi
}

skip_step() {
  local name="$1"
  STEP_RESULT[$name]="skip"
  warn "$name — SKIPPED"
}

should_run() {
  local name="$1"
  [[ -z "$ONLY_STEP" || "$ONLY_STEP" == "$name" ]]
}

# ──────────────────────────────────────────────────────────────────────────────
# Steps
# ──────────────────────────────────────────────────────────────────────────────

cd "$REPO_ROOT"

# ── 1. Ruff ──────────────────────────────────────────────────────────────────
if should_run ruff; then
  run_step ruff bash -c "
    $PYTHON -m pip install --quiet --upgrade ruff
    ruff check .
  "
else
  skip_step ruff
fi

# ── 2. Mypy ──────────────────────────────────────────────────────────────────
if should_run mypy; then
  run_step mypy bash -c "
    $PYTHON -m pip install --quiet --upgrade pip setuptools wheel
    $PYTHON -m pip install --quiet numpy -r requirements-test.txt
    $PYTHON -m pip install --quiet -e . --no-deps
    $PYTHON -m pip install --quiet mypy types-PyYAML
    mypy nirs4all
  "
else
  skip_step mypy
fi

# ── 3. Tests ─────────────────────────────────────────────────────────────────
if should_run tests; then
  if $SKIP_TESTS; then
    skip_step tests
  else
    run_step tests bash -c "
      $PYTHON -m pip install --quiet --upgrade pip setuptools wheel
      $PYTHON -m pip install --quiet numpy -r requirements-test.txt
      $PYTHON -m pip install --quiet -e . --no-deps

      echo '--- Serial subset (known write-contention) ---'
      $PYTHON -m pytest -v \
        tests/integration/pipeline/test_merge_mixed.py \
        tests/integration/pipeline/test_merge_per_branch.py \
        --cov=nirs4all --cov-append

      echo '--- Parallel remainder ---'
      $PYTHON -m pytest -v -n 8 --dist worksteal tests/ \
        --ignore=tests/integration/pipeline/test_merge_mixed.py \
        --ignore=tests/integration/pipeline/test_merge_per_branch.py \
        --cov=nirs4all --cov-append --cov-report=xml
    "
  fi
else
  skip_step tests
fi

# ── 4. Documentation ─────────────────────────────────────────────────────────
if should_run docs; then
  if $SKIP_DOCS; then
    skip_step docs
  else
    run_step docs bash -c "
      $PYTHON -m pip install --quiet --upgrade pip setuptools wheel
      $PYTHON -m pip install --quiet -r docs/readthedocs.requirements.txt
      $PYTHON -m pip install --quiet -e .
      cd docs
      sphinx-build -b html source _build/html --keep-going
    "
  fi
else
  skip_step docs
fi

# ── 5. Examples ──────────────────────────────────────────────────────────────
if should_run examples; then
  if $SKIP_EXAMPLES; then
    skip_step examples
  else
    run_step examples bash -c "
      $PYTHON -m pip install --quiet --upgrade pip setuptools wheel
      $PYTHON -m pip install --quiet numpy
      $PYTHON -m pip install --quiet -r requirements-examples.txt
      $PYTHON -m pip install --quiet -e .
      cd examples
      chmod +x run.sh

      # Run categories in parallel (mirrors CI matrix strategy)
      pids=()
      cats=($EXAMPLES_CATS)
      cat_logs=()
      for cat in \"\${cats[@]}\"; do
        logfile=\"\$(mktemp)\"
        cat_logs+=(\"\$logfile\")
        echo \"--- Launching category: \$cat (parallel) ---\"
        ./run.sh -c \"\$cat\" >\"\$logfile\" 2>&1 &
        pids+=(\$!)
      done

      # Wait for all and collect results
      failed=()
      for i in \"\${!pids[@]}\"; do
        if ! wait \"\${pids[\$i]}\"; then
          failed+=(\"\${cats[\$i]}\")
        fi
        echo \"\"
        echo \"=== Output: \${cats[\$i]} ===\"
        cat \"\${cat_logs[\$i]}\"
        rm -f \"\${cat_logs[\$i]}\"
      done

      if [ \${#failed[@]} -gt 0 ]; then
        echo \"FAILED categories: \${failed[*]}\"
        exit 1
      fi
    "
  fi
else
  skip_step examples
fi

# ── 6. Package build ─────────────────────────────────────────────────────────
if should_run build; then
  if $SKIP_BUILD; then
    skip_step build
  else
    run_step build bash -c "
      $PYTHON -m pip install --quiet --upgrade build twine
      $PYTHON -m build --sdist --wheel --outdir dist
      $PYTHON -m twine check dist/*
      # Smoke-test the wheel in a temp venv
      $PYTHON -m venv /tmp/nirs4all_wheel_test
      /tmp/nirs4all_wheel_test/bin/pip install --quiet dist/*.whl
      /tmp/nirs4all_wheel_test/bin/python -c \"import nirs4all; print(f'nirs4all {nirs4all.__version__} installed OK')\"
      rm -rf /tmp/nirs4all_wheel_test
    "
  fi
else
  skip_step build
fi

# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║           PRE-PUBLISH VALIDATION SUMMARY                      ║${RESET}"
echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════╣${RESET}"

ALL_PASS=true
ORDERED_STEPS=(ruff mypy tests docs examples build)
declare -A STEP_LABELS=([ruff]="Ruff          " [mypy]="Mypy          " [tests]="Tests         " [docs]="Documentation " [examples]="Examples      " [build]="Package Build ")

for step in "${ORDERED_STEPS[@]}"; do
  result="${STEP_RESULT[$step]:-skip}"
  label="${STEP_LABELS[$step]}"
  case "$result" in
    pass) echo -e "${BOLD}║${RESET}  ${label} ${GREEN}✅ PASSED${RESET}                                   ${BOLD}║${RESET}" ;;
    fail) echo -e "${BOLD}║${RESET}  ${label} ${RED}❌ FAILED${RESET}                                   ${BOLD}║${RESET}"
          ALL_PASS=false ;;
    skip) echo -e "${BOLD}║${RESET}  ${label} ${YELLOW}⏭  SKIPPED${RESET}                                  ${BOLD}║${RESET}" ;;
  esac
done

echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════╣${RESET}"

if $ALL_PASS; then
  echo -e "${BOLD}║  ${GREEN}🎉 Ready to publish! Create your release now.${RESET}${BOLD}              ║${RESET}"
  echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════╝${RESET}"
  exit 0
else
  echo -e "${BOLD}║  ${RED}⚠️  Fix issues above before creating a release.${RESET}${BOLD}              ║${RESET}"
  echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════╝${RESET}"
  # Print failed log paths
  for step in "${ORDERED_STEPS[@]}"; do
    if [[ "${STEP_RESULT[$step]:-}" == "fail" ]]; then
      err "Log for $step: ${STEP_LOG[$step]}"
    fi
  done
  exit 1
fi
