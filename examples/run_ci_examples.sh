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
  -s           Strict mode: fail on warning/invalid patterns (default: true)
  -v           Verbose: show example output on warnings/failures
  -k           Keep going: don't stop on first failure
  -h           Show help

Examples:
  ./run_ci_examples.sh
  ./run_ci_examples.sh -c user -j 4
  ./run_ci_examples.sh -c all -j 6 -k
EOF
    exit 0
}

while getopts "c:j:svkh" opt; do
    case "$opt" in
        c) CATEGORY="$OPTARG" ;;
        j) JOBS="$OPTARG" ;;
        s) STRICT=1 ;;
        v) VERBOSE=1 ;;
        k) KEEP_GOING=1 ;;
        h) show_help ;;
        *) echo "Usage: $0 [-c category] [-j jobs] [-s] [-v] [-k] [-h]"; exit 1 ;;
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

user_examples=(
    "user/01_getting_started/U01_hello_world.py"
    "user/01_getting_started/U02_basic_regression.py"
    "user/01_getting_started/U03_basic_classification.py"
    "user/01_getting_started/U04_visualization.py"
    "user/02_data_handling/U01_flexible_inputs.py"
    "user/02_data_handling/U02_multi_datasets.py"
    "user/02_data_handling/U03_multi_source.py"
    "user/02_data_handling/U04_wavelength_handling.py"
    "user/02_data_handling/U05_synthetic_data.py"
    "user/02_data_handling/U06_synthetic_advanced.py"
    "user/03_preprocessing/U01_preprocessing_basics.py"
    "user/03_preprocessing/U02_feature_augmentation.py"
    "user/03_preprocessing/U03_sample_augmentation.py"
    "user/03_preprocessing/U04_signal_conversion.py"
    "user/04_models/U01_multi_model.py"
    "user/04_models/U02_hyperparameter_tuning.py"
    "user/04_models/U03_stacking_ensembles.py"
    "user/04_models/U04_pls_variants.py"
    "user/05_cross_validation/U01_cv_strategies.py"
    "user/05_cross_validation/U02_group_splitting.py"
    "user/05_cross_validation/U03_sample_filtering.py"
    "user/05_cross_validation/U04_aggregation.py"
    "user/06_deployment/U01_save_load_predict.py"
    "user/06_deployment/U02_export_bundle.py"
    "user/06_deployment/U03_workspace_management.py"
    "user/06_deployment/U04_sklearn_integration.py"
    "user/07_explainability/U01_shap_basics.py"
    "user/07_explainability/U02_shap_sklearn.py"
    "user/07_explainability/U03_feature_selection.py"
    "user/05_cross_validation/U05_tagging_analysis.py"
    "user/05_cross_validation/U06_exclusion_strategies.py"
)

developer_examples=(
    "developer/01_advanced_pipelines/D01_branching_basics.py"
    "developer/01_advanced_pipelines/D02_branching_advanced.py"
    "developer/01_advanced_pipelines/D03_merge_basics.py"
    "developer/01_advanced_pipelines/D04_merge_sources.py"
    "developer/01_advanced_pipelines/D05_meta_stacking.py"
    "developer/02_generators/D01_generator_syntax.py"
    "developer/02_generators/D02_generator_advanced.py"
    "developer/02_generators/D03_generator_iterators.py"
    "developer/02_generators/D04_nested_generators.py"
    "developer/02_generators/D05_synthetic_custom_components.py"
    "developer/02_generators/D06_synthetic_testing.py"
    "developer/02_generators/D07_synthetic_wavenumber_procedural.py"
    "developer/02_generators/D08_synthetic_application_domains.py"
    "developer/02_generators/D09_synthetic_instruments.py"
    "developer/02_generators/D10_synthetic_custom_components.py"
    "developer/02_generators/D11_synthetic_testing.py"
    "developer/02_generators/D12_synthetic_environmental.py"
    "developer/02_generators/D13_synthetic_validation.py"
    "developer/02_generators/D14_synthetic_fitter.py"
    "developer/03_deep_learning/D01_pytorch_models.py"
    "developer/03_deep_learning/D02_jax_models.py"
    "developer/03_deep_learning/D03_tensorflow_models.py"
    "developer/03_deep_learning/D04_framework_comparison.py"
    "developer/04_transfer_learning/D01_transfer_analysis.py"
    "developer/04_transfer_learning/D02_retrain_modes.py"
    "developer/04_transfer_learning/D03_pca_geometry.py"
    "developer/05_advanced_features/D01_metadata_branching.py"
    "developer/05_advanced_features/D02_concat_transform.py"
    "developer/05_advanced_features/D03_repetition_transform.py"
    "developer/06_internals/D01_session_workflow.py"
    "developer/06_internals/D02_custom_controllers.py"
    "developer/01_advanced_pipelines/D06_separation_branches.py"
    "developer/01_advanced_pipelines/D07_value_mapping.py"
    "developer/06_internals/D03_cache_performance.py"
)

reference_examples=(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
)

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

    local startTime endTime duration exitCode start_iso end_iso
    startTime=$(date +%s)
    start_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    exitCode=0
    NIRS4ALL_EXAMPLE_FAST="$FAST_MODE" \
    NIRS4ALL_WORKSPACE="$workspace_dir" \
    "$PYTHON_BIN" "$LAUNCHER" "$example" > "$output_file" 2>&1 || exitCode=$?
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
failed=0
warnings=0
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
