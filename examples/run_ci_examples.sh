#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Local CI Examples Runner
# =============================================================================
# Mimics GitHub Actions workflow locally with output validation.
#
# Usage: ./run_ci_examples.sh [OPTIONS]
#
# Options:
#   -c CATEGORY  Category: user, developer, reference, all (default: all)
#   -j JOBS      Parallel workers for example execution (default: 1)
#   -p           Generate plots (pass --plots to examples)
#   -s           Strict mode: fail on warning/invalid patterns (default: true)
#   -v           Verbose: show example output on warnings/failures
#   -k           Keep going: don't stop on first failure (always implied for -j > 1)
#   -h           Show help
#
# Notes:
#   - Runs examples through `ci_example_launcher.py` with fast CI mode enabled.
#   - Each example uses an isolated workspace directory to avoid cross-talk.
# =============================================================================

CATEGORY="all"
STRICT=1
VERBOSE=0
KEEP_GOING=0
PLOTS=0
JOBS="${NIRS4ALL_CI_JOBS:-1}"
FAST_MODE="${NIRS4ALL_EXAMPLE_FAST:-1}"

format_duration() {
    local total_seconds="$1"
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

show_help() {
    cat <<'EOF'
Usage: ./run_ci_examples.sh [OPTIONS]

Options:
  -c CATEGORY  Category: user, developer, reference, all (default: all)
  -j JOBS      Parallel workers for example execution (default: 1)
  -p           Generate plots (pass --plots to examples)
  -s           Strict mode: fail on warning/invalid patterns (default: true)
  -v           Verbose: show example output on warnings/failures
  -k           Keep going: don't stop on first failure
  -h           Show help

Examples:
  ./run_ci_examples.sh
  ./run_ci_examples.sh -c user -j 4
  ./run_ci_examples.sh -c all -j 6 -k
  ./run_ci_examples.sh -c user -p
  ./run_ci_examples.sh -c user -p        # With plot generation
EOF
    exit 0
}

while getopts "c:j:psvkh" opt; do
    case "$opt" in
        c) CATEGORY="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        p) PLOTS=1 ;;
        s) STRICT=1 ;;
        v) VERBOSE=1 ;;
        k) KEEP_GOING=1 ;;
        h) show_help ;;
        *) echo "Usage: $0 [-c category] [-j jobs] [-p] [-s] [-v] [-k] [-h]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [ "$JOBS" -lt 1 ]; then
    echo "Invalid -j value '$JOBS'. Using 1." >&2
    JOBS=1
fi

if [ "$JOBS" -gt 1 ] && [ "$KEEP_GOING" -eq 0 ]; then
    # In parallel mode, workers are already in flight; stop-on-first-failure is
    # not practical, so keep-going behavior is implied.
    KEEP_GOING=1
fi

# =============================================================================
# Error Patterns to Detect
# =============================================================================

CRITICAL_PATTERNS=(
    "Traceback (most recent call last)"
    "Error:"
    "ValueError:"
    "TypeError:"
    "KeyError:"
    "ModuleNotFoundError:"
    "ImportError:"
    "AttributeError:"
    "RuntimeError:"
    "AssertionError:"
    "FAILED:"
    "VALIDATION FAILED:"
)

WARNING_PATTERNS=(
    "MSE: N/A"
    "R²: N/A"
    "RMSE: N/A"
    "Results: N/A"
    "Accuracy: N/A"
    "Accuracy: nan%"
    "R²: nan"
    "Best R²: nan"
    "RMSE = nan"
    "nan,"
    "NaN,"
)

INVALID_RESULT_PATTERNS=(
    ": 0 samples"
    ": 0 predictions"
    "No predictions"
    "Empty result"
)

# =============================================================================
# Paths and Runtime Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${SCRIPT_DIR}/workspace/ci_output"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"
SUMMARY_FILE="${RUN_DIR}/summary.txt"
ERRORS_FILE="${RUN_DIR}/errors.txt"
STATUS_DIR="${RUN_DIR}/status"
WORKSPACES_DIR="${RUN_DIR}/workspaces"

VENV_DIR="${WORKSPACE_ROOT}/.venv"
PYTHON_BIN=""

if [ -x "${VENV_DIR}/bin/python" ]; then
    echo "Using virtual environment Python: ${VENV_DIR}/bin/python"
    PYTHON_BIN="${VENV_DIR}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    echo "Virtual environment not found at ${VENV_DIR}; using system python3."
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    echo "Virtual environment not found at ${VENV_DIR}; using system python."
    PYTHON_BIN="$(command -v python)"
else
    echo "Error: no Python interpreter found." >&2
    exit 1
fi

LAUNCHER="${SCRIPT_DIR}/ci_example_launcher.py"
if [ ! -f "$LAUNCHER" ]; then
    echo "Error: CI launcher not found at ${LAUNCHER}" >&2
    exit 1
fi

mkdir -p "$RUN_DIR" "$STATUS_DIR" "$WORKSPACES_DIR"

echo "CI Examples Runner - Local Validation" | tee "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "Timestamp: $(date)" | tee -a "$SUMMARY_FILE"
echo "Category: $CATEGORY" | tee -a "$SUMMARY_FILE"
echo "Strict mode: $STRICT" | tee -a "$SUMMARY_FILE"
echo "Jobs: $JOBS" | tee -a "$SUMMARY_FILE"
echo "Plots: $PLOTS" | tee -a "$SUMMARY_FILE"
echo "Fast mode: $FAST_MODE" | tee -a "$SUMMARY_FILE"
echo "Output dir: $RUN_DIR" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# =============================================================================
# Output Validation
# =============================================================================

check_output() {
    local output_file="$1"
    local has_critical=0
    local has_warning=0
    local has_invalid=0
    local issues=()

    for pattern in "${CRITICAL_PATTERNS[@]}"; do
        if grep -qF "$pattern" "$output_file" 2>/dev/null; then
            has_critical=1
            issues+=("CRITICAL: Found '$pattern'")
        fi
    done

    for pattern in "${WARNING_PATTERNS[@]}"; do
        if grep -qE "$pattern" "$output_file" 2>/dev/null; then
            has_warning=1
            issues+=("WARNING: Found '$pattern'")
        fi
    done

    for pattern in "${INVALID_RESULT_PATTERNS[@]}"; do
        if grep -qF "$pattern" "$output_file" 2>/dev/null; then
            has_invalid=1
            issues+=("INVALID: Found '$pattern'")
        fi
    done

    if [ "$has_critical" -eq 1 ]; then
        echo "2"
        for issue in "${issues[@]}"; do
            echo "$issue"
        done
    elif [ "$has_warning" -eq 1 ] || [ "$has_invalid" -eq 1 ]; then
        echo "1"
        for issue in "${issues[@]}"; do
            echo "$issue"
        done
    else
        echo "0"
    fi
}

# =============================================================================
# Example Selection
# =============================================================================

cd "$SCRIPT_DIR"
# shellcheck source=example_inventory.sh
source "$SCRIPT_DIR/example_inventory.sh"

selected_examples=()
case "$CATEGORY" in
    user)      selected_examples=("${user_examples[@]}") ;;
    developer) selected_examples=("${developer_examples[@]}") ;;
    reference) selected_examples=("${reference_examples[@]}") ;;
    all)       selected_examples=("${user_examples[@]}" "${developer_examples[@]}" "${reference_examples[@]}") ;;
    *)
        echo "Error: Unknown category '$CATEGORY'. Valid: user, developer, reference, all" >&2
        exit 1
        ;;
esac

filtered_examples=()
for ex in "${selected_examples[@]}"; do
    if [ -f "$ex" ]; then
        filtered_examples+=("$ex")
    fi
done
selected_examples=("${filtered_examples[@]}")

if [ "${#selected_examples[@]}" -eq 0 ]; then
    echo "No examples selected." | tee -a "$SUMMARY_FILE"
    exit 1
fi

echo "Examples to run: ${#selected_examples[@]}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# =============================================================================
# Execution
# =============================================================================

is_resource_exclusive_example() {
    case "$1" in
        developer/03_deep_learning/*|user/04_models/U06_tabpfn_nirs.py) return 0 ;;
        *) return 1 ;;
    esac
}

run_example_worker() {
    local idx="$1"
    local example="$2"
    local total="$3"

    local example_name output_file status_file workspace_dir
    example_name=$(basename "$example" .py)
    output_file="${RUN_DIR}/$(printf "%03d_%s.log" "$idx" "$example_name")"
    status_file="${STATUS_DIR}/$(printf "%03d.status" "$idx")"
    workspace_dir="${WORKSPACES_DIR}/$(printf "%03d_%s" "$idx" "$example_name")"

    mkdir -p "$workspace_dir"

    local launcher_args=()
    if [ "$PLOTS" -eq 1 ]; then launcher_args+=("--plots"); fi

    local startTime endTime duration exitCode start_iso end_iso
    startTime=$(date +%s)
    start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    exitCode=0
    if is_resource_exclusive_example "$example"; then
        CUDA_VISIBLE_DEVICES="${NIRS4ALL_CI_CUDA_VISIBLE_DEVICES:-0}" \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        TF_FORCE_GPU_ALLOW_GROWTH=true \
        NIRS4ALL_EXAMPLE_FAST="$FAST_MODE" \
        NIRS4ALL_WORKSPACE="$workspace_dir" \
        "$PYTHON_BIN" "$LAUNCHER" "$example" "${launcher_args[@]}" > "$output_file" 2>&1 || exitCode=$?
    else
        NIRS4ALL_EXAMPLE_FAST="$FAST_MODE" \
        NIRS4ALL_WORKSPACE="$workspace_dir" \
        "$PYTHON_BIN" "$LAUNCHER" "$example" "${launcher_args[@]}" > "$output_file" 2>&1 || exitCode=$?
    fi
    endTime=$(date +%s)
    duration=$((endTime - startTime))
    end_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    printf "%s|%s|%s|%s|%s|%s|%s\n" "$idx" "$example" "$output_file" "$exitCode" "$duration" "$start_iso" "$end_iso" > "$status_file"
    echo "DONE   [${idx}/${total}] ${end_iso} :: ${example} (${duration}s, exit=${exitCode})"
}

GLOBAL_START_EPOCH=$(date +%s)
GLOBAL_START_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
TOTAL_EXAMPLES="${#selected_examples[@]}"

if [ "$JOBS" -gt 1 ]; then
    for i in "${!selected_examples[@]}"; do
        idx=$((i + 1))
        example="${selected_examples[$i]}"

        if is_resource_exclusive_example "$example"; then
            wait
            launch_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
            echo "LAUNCH [${idx}/${TOTAL_EXAMPLES}] ${launch_iso} :: ${example} (exclusive resources)"
            run_example_worker "$idx" "$example" "$TOTAL_EXAMPLES"
            continue
        fi

        while [ "$(jobs -pr | wc -l)" -ge "$JOBS" ]; do
            sleep 0.2
        done

        launch_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "LAUNCH [${idx}/${TOTAL_EXAMPLES}] ${launch_iso} :: ${example}"
        run_example_worker "$idx" "$example" "$TOTAL_EXAMPLES" &
    done
    wait
else
    for i in "${!selected_examples[@]}"; do
        idx=$((i + 1))
        example="${selected_examples[$i]}"
        launch_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
        echo "LAUNCH [${idx}/${TOTAL_EXAMPLES}] ${launch_iso} :: ${example}"
        run_example_worker "$idx" "$example" "$TOTAL_EXAMPLES"
    done
fi

# =============================================================================
# Validation and Summary
# =============================================================================

passed=0
skipped=0
failed=0
warnings=0
skipped_examples=()
failed_examples=()
warning_examples=()

for i in "${!selected_examples[@]}"; do
    idx=$((i + 1))
    example="${selected_examples[$i]}"
    status_file="${STATUS_DIR}/$(printf "%03d.status" "$idx")"

    if [ ! -f "$status_file" ]; then
        echo "Running: $example ... FAILED (missing status file)"
        failed=$((failed + 1))
        failed_examples+=("$example (missing status file)")
        continue
    fi

    IFS='|' read -r _ meta_example output_file exitCode duration start_iso end_iso < "$status_file"

    echo -n "Running: $example ... "

    if [ "$exitCode" -ne 0 ]; then
        echo "FAILED (exit code: $exitCode, ${duration}s, done: ${end_iso})"
        failed=$((failed + 1))
        failed_examples+=("$example (exit code: $exitCode)")
        {
            echo ""
            echo "=== $example (exit code: $exitCode) ==="
            cat "$output_file"
        } >> "$ERRORS_FILE"

        if [ "$VERBOSE" -eq 1 ]; then
            echo "--- Output ---"
            cat "$output_file"
            echo "--- End Output ---"
        fi

        if [ "$KEEP_GOING" -eq 0 ] && [ "$JOBS" -eq 1 ]; then
            echo ""
            echo "Stopping on first failure. Use -k to keep going."
            break
        fi
        continue
    fi

    # The launcher emits this marker only after matching an exact, declared
    # precondition (for example a TabPFN licence exception). Some execution
    # lanes log the caught traceback before it crosses back to the launcher,
    # so classify the trusted marker before generic traceback scanning.
    if grep -q '^\[SKIP\]' "$output_file" 2>/dev/null; then
        echo "SKIPPED (${duration}s, done: ${end_iso})"
        skipped=$((skipped + 1))
        skipped_examples+=("$example")
        continue
    fi

    check_result=$(check_output "$output_file")
    status=$(echo "$check_result" | head -1)
    issues=$(echo "$check_result" | tail -n +2)

    case "$status" in
        0)
            echo "OK (${duration}s, done: ${end_iso})"
            passed=$((passed + 1))
            ;;
        1)
            echo "WARNING (${duration}s, done: ${end_iso})"
            warnings=$((warnings + 1))
            warning_examples+=("$example")
            if [ "$STRICT" -eq 1 ]; then
                failed=$((failed + 1))
                failed_examples+=("$example (warnings detected)")
            fi
            {
                echo ""
                echo "=== $example (warnings) ==="
                echo "$issues"
                echo "--- Relevant output ---"
                for pattern in "${WARNING_PATTERNS[@]}" "${INVALID_RESULT_PATTERNS[@]}"; do
                    grep -n "$pattern" "$output_file" 2>/dev/null || true
                done
            } >> "$ERRORS_FILE"

            if [ "$VERBOSE" -eq 1 ]; then
                echo "  Issues:"
                echo "$issues" | sed 's/^/    /'
            fi

            if [ "$STRICT" -eq 1 ] && [ "$KEEP_GOING" -eq 0 ] && [ "$JOBS" -eq 1 ]; then
                echo ""
                echo "Stopping on warning (strict mode). Use -k to keep going."
                break
            fi
            ;;
        2)
            echo "CRITICAL (${duration}s, done: ${end_iso})"
            failed=$((failed + 1))
            failed_examples+=("$example (critical error)")
            {
                echo ""
                echo "=== $example (critical) ==="
                cat "$output_file"
            } >> "$ERRORS_FILE"

            if [ "$VERBOSE" -eq 1 ]; then
                echo "--- Output ---"
                cat "$output_file"
                echo "--- End Output ---"
            fi

            if [ "$KEEP_GOING" -eq 0 ] && [ "$JOBS" -eq 1 ]; then
                echo ""
                echo "Stopping on critical error. Use -k to keep going."
                break
            fi
            ;;
    esac
done

GLOBAL_END_EPOCH=$(date +%s)
GLOBAL_END_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GLOBAL_DURATION=$((GLOBAL_END_EPOCH - GLOBAL_START_EPOCH))

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "CI VALIDATION SUMMARY" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Total examples: ${#selected_examples[@]}" | tee -a "$SUMMARY_FILE"
echo "Passed: $passed" | tee -a "$SUMMARY_FILE"
echo "Skipped (declared precondition): $skipped" | tee -a "$SUMMARY_FILE"
echo "Warnings: $warnings" | tee -a "$SUMMARY_FILE"
echo "Failed: $failed" | tee -a "$SUMMARY_FILE"
echo "Started: $GLOBAL_START_ISO" | tee -a "$SUMMARY_FILE"
echo "Finished: $GLOBAL_END_ISO" | tee -a "$SUMMARY_FILE"
echo "Elapsed: ${GLOBAL_DURATION}s ($(format_duration "$GLOBAL_DURATION"))" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

if [ "${#failed_examples[@]}" -gt 0 ]; then
    echo "FAILED EXAMPLES:" | tee -a "$SUMMARY_FILE"
    for ex in "${failed_examples[@]}"; do
        echo "  X $ex" | tee -a "$SUMMARY_FILE"
    done
    echo "" | tee -a "$SUMMARY_FILE"
fi

if [ "${#warning_examples[@]}" -gt 0 ]; then
    echo "EXAMPLES WITH WARNINGS:" | tee -a "$SUMMARY_FILE"
    for ex in "${warning_examples[@]}"; do
        echo "  ! $ex" | tee -a "$SUMMARY_FILE"
    done
    echo "" | tee -a "$SUMMARY_FILE"
fi

if [ "${#skipped_examples[@]}" -gt 0 ]; then
    echo "EXAMPLES SKIPPED BY DECLARED PRECONDITION:" | tee -a "$SUMMARY_FILE"
    for ex in "${skipped_examples[@]}"; do
        echo "  - $ex" | tee -a "$SUMMARY_FILE"
    done
    echo "" | tee -a "$SUMMARY_FILE"
fi

echo "Detailed logs: $RUN_DIR" | tee -a "$SUMMARY_FILE"
if [ -f "$ERRORS_FILE" ] && [ -s "$ERRORS_FILE" ]; then
    echo "Errors file: $ERRORS_FILE" | tee -a "$SUMMARY_FILE"
fi

if [ "$failed" -gt 0 ]; then
    echo ""
    echo "X CI validation FAILED"
    exit 1
fi

echo ""
echo "OK CI validation PASSED"
exit 0
