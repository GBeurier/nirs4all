#!/usr/bin/env bash
set -uo pipefail

# Usage: ./run.sh [-i index] [-b begin] [-e end] [-n name] [-l] [-p] [-s]
# Note: -e removed to continue on example failures

INDEX=0
BEGIN=0
END=0
NAME=""
LOG=0
PLOT=0
SHOW=0

while getopts "i:b:e:n:lps" opt; do
  case "$opt" in
    i) INDEX="$OPTARG" ;;
    b) BEGIN="$OPTARG" ;;
    e) END="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    l) LOG=1 ;;
    p) PLOT=1 ;;
    s) SHOW=1 ;;
    *) echo "Usage: $0 [-i index] [-b begin] [-e end] [-n name] [-l] [-p] [-s]"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

examples=(
  "Q1_classif.py"
  "Q1_regression.py"
  "Q2_groupsplit.py"
  "Q2_multimodel.py"
  "Q2B_force_group.py"
  "Q3_finetune.py"
  "Q4_multidatasets.py"
  "Q5_predict.py"
  "Q5_predict_NN.py"
  "Q6_multisource.py"
  "Q7_discretization.py"
  "Q8_shap.py"
  "Q9_acp_spread.py"
  "Q10_resampler.py"
  "Q11_flexible_inputs.py"
  "Q12_sample_augmentation.py"
  "Q13_nm_headers.py"
  "Q14_workspace.py"
  "Q15_jax_models.py"
  "Q16_pytorch_models.py"
  "Q17_nicon_comparison.py"
  "Q18_stacking.py"
  "Q19_pls_methods.py"
  "Q21_feature_selection.py"
  "Q22_concat_transform.py"
  "Q23_generator_syntax.py"
  "Q23b_generator.py"
  "Q24_generator_advanced.py"
  "Q25_complex_pipeline_pls.py"
  "Q26_nested_or_preprocessing.py"
  "Q27_transfer_analysis.py"
  "Q28_sample_filtering.py"
  "Q29_signal_conversion.py"
  "Q30_branching.py"
  "Q31_outlier_branching.py"
  "Q32_export_bundle.py"
  "Q33_retrain_transfer.py"
  "Q34_aggregation.py"
  "Q35_metadata_branching.py",
  "Q36_repetition_transform.py",
  "Q_meta_stacking.py"
  "Q_complex_all_keywords.py"
  "Q_feature_augmentation_modes.py"
  "Q_merge_branches.py"
  "Q_merge_sources.py"
  "baseline_sota.py"
  "X0_pipeline_sample.py"
  "X1_metadata.py"
  "X2_sample_augmentation.py"
  # "X3_hiba_full.py"
  "X4_features.py"
)

# Setup logging if requested
logFile=""
if [ "$LOG" -eq 1 ]; then
  logFile="$(pwd)/log.txt"
  echo "Logging enabled: $logFile"
  printf "=================================================\n" > "$logFile"
  printf "Log started at: %s\n" "$(date)" >> "$logFile"
  printf "=================================================\n\n" >> "$logFile"
fi

# Determine which examples to run based on parameters
paramCount=0
if [ "${INDEX:-0}" -gt 0 ]; then paramCount=$((paramCount+1)); fi
if [ "${BEGIN:-0}" -gt 0 ]; then paramCount=$((paramCount+1)); fi
if [ "${END:-0}" -gt 0 ]; then paramCount=$((paramCount+1)); fi
if [ -n "${NAME}" ]; then paramCount=$((paramCount+1)); fi

if [ "$paramCount" -gt 1 ]; then
  echo "Error: Specify only one of -Index, -Begin, or -Name." >&2
  exit 1
fi

selectedExamples=("${examples[@]}")
if [ "${INDEX:-0}" -gt 0 ]; then
  if [ "$INDEX" -lt 1 ] || [ "$INDEX" -gt "${#examples[@]}" ]; then
    echo "Error: Index $INDEX is out of range. Valid range is 1..${#examples[@]}." >&2
    exit 1
  fi
  idx=$((INDEX-1))
  selectedExamples=("${examples[$idx]}")
  echo "Running single example #$INDEX: ${selectedExamples[0]}"
elif [ "${BEGIN:-0}" -gt 0 ]; then
  if [ "$BEGIN" -lt 1 ] || [ "$BEGIN" -gt "${#examples[@]}" ]; then
    echo "Error: Start index $BEGIN is out of range. Valid range is 1..${#examples[@]}." >&2
    exit 1
  fi
  startIndex=$((BEGIN-1))
  selectedExamples=("${examples[@]:$startIndex}")
  echo "Running all examples starting from #$BEGIN (count: ${#selectedExamples[@]})"
elif [ -n "$NAME" ]; then
  matched=""
  shopt -s nocasematch
  for ex in "${examples[@]}"; do
    if [[ "$ex" == $NAME ]]; then
      matched="$ex"
      break
    fi
  done
  shopt -u nocasematch
  if [ -z "$matched" ]; then
    echo "Error: Example '$NAME' not found in the list." >&2
    echo "Available examples:"
    for ex in "${examples[@]}"; do
      echo "  $ex"
    done
    exit 1
  fi
  selectedExamples=("$matched")
  echo "Running example by name: $matched"
fi

# Disable emojis only when logging to avoid encoding issues with file output
if [ "$LOG" -eq 1 ]; then
  export DISABLE_EMOJI=1
  echo "Emojis disabled for logging"
else
  unset DISABLE_EMOJI 2>/dev/null || true
fi

# Track failed examples
failedExamples=()

# SEQUENTIAL
for example in "${selectedExamples[@]}"; do
  if [ -f "$example" ]; then
    startTime=$(date +%s)
    echo "########################################"
    echo "Launch: $example"
    echo "########################################"

    args=()
    if [ "$PLOT" -eq 1 ]; then args+=("--plots"); fi
    if [ "$SHOW" -eq 1 ]; then args+=("--show"); fi

    exitCode=0
    if [ -n "$logFile" ]; then
      printf "===============================================\n" >> "$logFile"
      printf "Starting: %s at %s\n" "$example" "$(date)" >> "$logFile"
      printf "===============================================\n" >> "$logFile"

      python "$example" "${args[@]}" 2>&1 | tee -a "$logFile" || exitCode=$?

      printf "Finished: %s at %s (exit code: %d)\n\n" "$example" "$(date)" "$exitCode" >> "$logFile"
    else
      python "$example" "${args[@]}" || exitCode=$?
    fi

    # Track failed examples
    if [ "$exitCode" -ne 0 ]; then
      failedExamples+=("$example (exit code: $exitCode)")
      echo "FAILED: $example with exit code $exitCode"
    fi

    endTime=$(date +%s)
    duration=$((endTime - startTime))
    hours=$((duration/3600))
    mins=$(((duration%3600)/60))
    secs=$((duration%60))
    printf "Finished running: %s (Duration: %02d:%02d:%02d)\n" "$example" "$hours" "$mins" "$secs"
    echo "########################################"
  fi
done

# Summary of failed examples
echo ""
echo "========================================"
echo "EXECUTION SUMMARY"
echo "========================================"
echo "Total examples run: ${#selectedExamples[@]}"
echo "Failed: ${#failedExamples[@]}"
echo "Passed: $((${#selectedExamples[@]} - ${#failedExamples[@]}))"

if [ ${#failedExamples[@]} -gt 0 ]; then
  echo ""
  echo "FAILED EXAMPLES:"
  for failed in "${failedExamples[@]}"; do
    echo "  - $failed"
  done
  if [ -n "$logFile" ]; then
    echo "" >> "$logFile"
    echo "========================================" >> "$logFile"
    echo "FAILED EXAMPLES:" >> "$logFile"
    for failed in "${failedExamples[@]}"; do
      echo "  - $failed" >> "$logFile"
    done
  fi
  exit 1
else
  echo ""
  echo "All examples passed successfully!"
fi
