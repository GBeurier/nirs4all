#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# Local CI Examples Runner
# =============================================================================
# Mimics the GitHub Actions workflow locally for testing examples
# with strict output validation to catch silent errors.
#
# Usage: ./run_ci_examples.sh [OPTIONS]
#
# Options:
#   -c CATEGORY  Category: user, developer, reference, all (default: all)
#   -q           Quick mode: skip deep learning examples
#   -s           Strict mode: fail on any warning/error pattern (default: true)
#   -v           Verbose: show all output (not just failures)
#   -k           Keep going: don't stop on first failure
#   -h           Show this help message
#
# Examples:
#   ./run_ci_examples.sh                 # Run all examples with strict validation
#   ./run_ci_examples.sh -c user         # Run only user examples
#   ./run_ci_examples.sh -c user -q      # Quick mode (no DL)
#   ./run_ci_examples.sh -v              # Verbose output
# =============================================================================

CATEGORY="all"
QUICK=0
STRICT=1
VERBOSE=0
KEEP_GOING=0

show_help() {
    head -23 "$0" | tail -18 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

while getopts "c:qsvkh" opt; do
    case "$opt" in
        c) CATEGORY="$OPTARG" ;;
        q) QUICK=1 ;;
        s) STRICT=1 ;;
        v) VERBOSE=1 ;;
        k) KEEP_GOING=1 ;;
        h) show_help ;;
        *) echo "Usage: $0 [-c category] [-q] [-s] [-v] [-k] [-h]"; exit 1 ;;
    esac
done
shift $((OPTIND-1))

# =============================================================================
# Error Patterns to Detect
# =============================================================================
# These patterns indicate silent failures that should cause the CI to fail.

# Critical error patterns (always fail)
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

# Warning patterns that indicate potential issues
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

# Patterns that are expected library warnings (not errors)
# These are intentional warnings from nirs4all library
# EXPECTED_WARNINGS=(
#     "[!] WARNING: Using test set as validation set"
#     "[!] Failed to calculate pearson_r"
# )

# Patterns that indicate empty/invalid results
INVALID_RESULT_PATTERNS=(
    ": 0 samples"
    ": 0 predictions"
    "No predictions"
    "Empty result"
)

# =============================================================================
# Output Directories
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${SCRIPT_DIR}/workspace/ci_output"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_DIR="${OUTPUT_DIR}/run_${TIMESTAMP}"

# Activate virtual environment
VENV_DIR="${WORKSPACE_ROOT}/.venv"
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment: $VENV_DIR"
    source "${VENV_DIR}/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_DIR"
    echo "Using system Python instead."
fi

mkdir -p "$RUN_DIR"

# Summary file
SUMMARY_FILE="${RUN_DIR}/summary.txt"
ERRORS_FILE="${RUN_DIR}/errors.txt"

echo "CI Examples Runner - Local Validation" | tee "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "Timestamp: $(date)" | tee -a "$SUMMARY_FILE"
echo "Category: $CATEGORY" | tee -a "$SUMMARY_FILE"
echo "Quick mode: $QUICK" | tee -a "$SUMMARY_FILE"
echo "Strict mode: $STRICT" | tee -a "$SUMMARY_FILE"
echo "Output dir: $RUN_DIR" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# =============================================================================
# Check Output for Errors
# =============================================================================

check_output() {
    local output_file="$1"
    local example_name="$2"
    local has_critical=0
    local has_warning=0
    local has_invalid=0
    local issues=()

    # Check critical patterns
    for pattern in "${CRITICAL_PATTERNS[@]}"; do
        if grep -qF "$pattern" "$output_file" 2>/dev/null; then
            has_critical=1
            issues+=("CRITICAL: Found '$pattern'")
        fi
    done

    # Check warning patterns
    for pattern in "${WARNING_PATTERNS[@]}"; do
        if grep -qE "$pattern" "$output_file" 2>/dev/null; then
            has_warning=1
            issues+=("WARNING: Found '$pattern'")
        fi
    done

    # Check invalid result patterns
    for pattern in "${INVALID_RESULT_PATTERNS[@]}"; do
        if grep -qF "$pattern" "$output_file" 2>/dev/null; then
            has_invalid=1
            issues+=("INVALID: Found '$pattern'")
        fi
    done

    # Return status: 0=ok, 1=warning, 2=critical
    if [ $has_critical -eq 1 ]; then
        echo "2"
        for issue in "${issues[@]}"; do
            echo "$issue"
        done
    elif [ $has_warning -eq 1 ] || [ $has_invalid -eq 1 ]; then
        echo "1"
        for issue in "${issues[@]}"; do
            echo "$issue"
        done
    else
        echo "0"
    fi
}

# =============================================================================
# Build Example List (delegate to run.sh logic)
# =============================================================================

cd "$SCRIPT_DIR"

# Build command arguments
RUN_ARGS=()
if [ "$QUICK" -eq 1 ]; then
    RUN_ARGS+=("-q")
fi

case "$CATEGORY" in
    user)      RUN_ARGS+=("-c" "user") ;;
    developer) RUN_ARGS+=("-c" "developer") ;;
    reference) RUN_ARGS+=("-c" "reference") ;;
    all)       RUN_ARGS+=("-c" "all") ;;
    *)
        echo "Error: Unknown category '$CATEGORY'. Valid: user, developer, reference, all" >&2
        exit 1
        ;;
esac

# Get example list from run.sh categories (simplified version)
# We'll extract the logic from run.sh

# User examples
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
)

reference_examples=(
    "reference/R01_pipeline_syntax.py"
    "reference/R02_generator_reference.py"
    "reference/R03_all_keywords.py"
    "reference/R04_legacy_api.py"
)

# Deep learning examples to skip in quick mode
dl_examples=(
    "D01_pytorch_models.py"
    "D02_jax_models.py"
    "D03_tensorflow_models.py"
    "D04_framework_comparison.py"
)

is_dl_example() {
    local example="$1"
    local basename
    basename=$(basename "$example")
    for dl in "${dl_examples[@]}"; do
        if [[ "$basename" == "$dl" ]]; then
            return 0
        fi
    done
    return 1
}

# Build selected examples list
selected_examples=()

case "$CATEGORY" in
    user)
        selected_examples=("${user_examples[@]}")
        ;;
    developer)
        selected_examples=("${developer_examples[@]}")
        ;;
    reference)
        selected_examples=("${reference_examples[@]}")
        ;;
    all)
        selected_examples=("${user_examples[@]}" "${developer_examples[@]}" "${reference_examples[@]}")
        ;;
esac

# Filter to existing files only
filtered_examples=()
for ex in "${selected_examples[@]}"; do
    if [ -f "$ex" ]; then
        # Apply quick mode filter
        if [ "$QUICK" -eq 1 ] && is_dl_example "$ex"; then
            continue
        fi
        filtered_examples+=("$ex")
    fi
done

selected_examples=("${filtered_examples[@]}")

echo "Examples to run: ${#selected_examples[@]}" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# =============================================================================
# Run Examples with Validation
# =============================================================================

passed=0
failed=0
warnings=0
failed_examples=()
warning_examples=()

for example in "${selected_examples[@]}"; do
    example_name=$(basename "$example" .py)
    output_file="${RUN_DIR}/${example_name}.log"

    echo -n "Running: $example ... "

    # Run the example and capture output
    startTime=$(date +%s)
    exitCode=0
    python "$example" > "$output_file" 2>&1 || exitCode=$?
    endTime=$(date +%s)
    duration=$((endTime - startTime))

    # Check exit code first
    if [ $exitCode -ne 0 ]; then
        echo "FAILED (exit code: $exitCode, ${duration}s)"
        failed=$((failed + 1))
        failed_examples+=("$example (exit code: $exitCode)")
        echo "" >> "$ERRORS_FILE"
        echo "=== $example (exit code: $exitCode) ===" >> "$ERRORS_FILE"
        cat "$output_file" >> "$ERRORS_FILE"

        if [ "$VERBOSE" -eq 1 ]; then
            echo "--- Output ---"
            cat "$output_file"
            echo "--- End Output ---"
        fi

        if [ "$KEEP_GOING" -eq 0 ]; then
            echo ""
            echo "Stopping on first failure. Use -k to keep going."
            break
        fi
        continue
    fi

    # Check output for error patterns
    check_result=$(check_output "$output_file" "$example_name")
    status=$(echo "$check_result" | head -1)
    issues=$(echo "$check_result" | tail -n +2)

    case $status in
        0)
            echo "OK (${duration}s)"
            passed=$((passed + 1))
            ;;
        1)
            echo "WARNING (${duration}s)"
            warnings=$((warnings + 1))
            warning_examples+=("$example")
            if [ "$STRICT" -eq 1 ]; then
                failed=$((failed + 1))
                failed_examples+=("$example (warnings detected)")
            fi
            echo "" >> "$ERRORS_FILE"
            echo "=== $example (warnings) ===" >> "$ERRORS_FILE"
            echo "$issues" >> "$ERRORS_FILE"
            echo "--- Relevant output ---" >> "$ERRORS_FILE"
            # Extract lines with issues
            for pattern in "${WARNING_PATTERNS[@]}" "${INVALID_RESULT_PATTERNS[@]}"; do
                grep -n "$pattern" "$output_file" 2>/dev/null >> "$ERRORS_FILE" || true
            done

            if [ "$VERBOSE" -eq 1 ]; then
                echo "  Issues:"
                echo "$issues" | sed 's/^/    /'
            fi

            if [ "$STRICT" -eq 1 ] && [ "$KEEP_GOING" -eq 0 ]; then
                echo ""
                echo "Stopping on warning (strict mode). Use -k to keep going."
                break
            fi
            ;;
        2)
            echo "CRITICAL (${duration}s)"
            failed=$((failed + 1))
            failed_examples+=("$example (critical error)")
            echo "" >> "$ERRORS_FILE"
            echo "=== $example (critical) ===" >> "$ERRORS_FILE"
            cat "$output_file" >> "$ERRORS_FILE"

            if [ "$VERBOSE" -eq 1 ]; then
                echo "--- Output ---"
                cat "$output_file"
                echo "--- End Output ---"
            fi

            if [ "$KEEP_GOING" -eq 0 ]; then
                echo ""
                echo "Stopping on critical error. Use -k to keep going."
                break
            fi
            ;;
    esac
done

# =============================================================================
# Summary Report
# =============================================================================

echo "" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "CI VALIDATION SUMMARY" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Total examples: ${#selected_examples[@]}" | tee -a "$SUMMARY_FILE"
echo "Passed: $passed" | tee -a "$SUMMARY_FILE"
echo "Warnings: $warnings" | tee -a "$SUMMARY_FILE"
echo "Failed: $failed" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

if [ ${#failed_examples[@]} -gt 0 ]; then
    echo "FAILED EXAMPLES:" | tee -a "$SUMMARY_FILE"
    for ex in "${failed_examples[@]}"; do
        echo "  ✗ $ex" | tee -a "$SUMMARY_FILE"
    done
    echo "" | tee -a "$SUMMARY_FILE"
fi

if [ ${#warning_examples[@]} -gt 0 ]; then
    echo "EXAMPLES WITH WARNINGS:" | tee -a "$SUMMARY_FILE"
    for ex in "${warning_examples[@]}"; do
        echo "  ⚠ $ex" | tee -a "$SUMMARY_FILE"
    done
    echo "" | tee -a "$SUMMARY_FILE"
fi

echo "Detailed logs: $RUN_DIR" | tee -a "$SUMMARY_FILE"
if [ -f "$ERRORS_FILE" ] && [ -s "$ERRORS_FILE" ]; then
    echo "Errors file: $ERRORS_FILE" | tee -a "$SUMMARY_FILE"
fi

# Exit with appropriate code
if [ $failed -gt 0 ]; then
    echo ""
    echo "❌ CI validation FAILED"
    exit 1
else
    echo ""
    echo "✅ CI validation PASSED"
    exit 0
fi
