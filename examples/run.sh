#!/usr/bin/env bash
set -uo pipefail

# =============================================================================
# nirs4all Examples Runner
# =============================================================================
# Usage: ./run.sh [OPTIONS]
#
# Options:
#   -i INDEX     Run single example by index (1-based)
#   -b BEGIN     Start from this index (1-based)
#   -e END       End at this index (1-based)
#   -n NAME      Run by name pattern (glob, e.g., "U01*.py")
#   -c CATEGORY  Category: user, developer, reference, all (default: all)
#   -l           Enable logging to log.txt
#   -p           Generate plots
#   -s           Show plots interactively
#   -h           Show this help message
#
# Examples:
#   ./run.sh                     # Run all examples
#   ./run.sh -c user             # Run only User path examples
#   ./run.sh -i 1                # Run first example
#   ./run.sh -n "U01*.py"        # Run by name pattern
#   ./run.sh -l -p               # Enable logging and plots
# =============================================================================

INDEX=0
BEGIN=0
END=0
NAME=""
CATEGORY="all"
LOG=0
PLOT=0
SHOW=0

show_help() {
  head -30 "$0" | tail -25 | sed 's/^# //' | sed 's/^#//'
  exit 0
}

while getopts "i:b:e:n:c:lpsh" opt; do
  case "$opt" in
    i) INDEX="$OPTARG" ;;
    b) BEGIN="$OPTARG" ;;
    e) END="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    c) CATEGORY="$OPTARG" ;;
    l) LOG=1 ;;
    p) PLOT=1 ;;
    s) SHOW=1 ;;
    h) show_help ;;
    *) echo "Usage: $0 [-i index] [-b begin] [-e end] [-n name] [-c category] [-l] [-p] [-s] [-h]"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

# =============================================================================
# Example Definitions by Category
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# shellcheck source=example_inventory.sh
source "$SCRIPT_DIR/example_inventory.sh"

# =============================================================================
# Build Examples List
# =============================================================================

# Function to filter existing examples only
filter_existing() {
  local -n arr=$1
  local filtered=()
  for ex in "${arr[@]}"; do
    if [ -f "$ex" ]; then
      filtered+=("$ex")
    fi
  done
  arr=("${filtered[@]}")
}

# Build selected examples based on category
build_examples_list() {
  local examples=()

  case "$CATEGORY" in
    user)
      examples=("${user_examples[@]}")
      ;;
    developer)
      examples=("${developer_examples[@]}")
      ;;
    reference)
      examples=("${reference_examples[@]}")
      ;;
    all)
      examples=("${user_examples[@]}" "${developer_examples[@]}" "${reference_examples[@]}")
      ;;
    *)
      echo "Error: Unknown category '$CATEGORY'. Valid: user, developer, reference, all" >&2
      exit 1
      ;;
  esac

  # Filter to existing files only
  local existing=()
  for ex in "${examples[@]}"; do
    if [ -f "$ex" ]; then
      existing+=("$ex")
    fi
  done

  echo "${existing[@]}"
}

# Get examples list
IFS=' ' read -ra selectedExamples <<< "$(build_examples_list)"

if [ ${#selectedExamples[@]} -eq 0 ]; then
  echo "No examples found for category '$CATEGORY'."
  echo "Make sure examples exist in the expected locations."
  exit 1
fi

echo "Category: $CATEGORY"
echo "Found ${#selectedExamples[@]} example(s)"

# =============================================================================
# Setup Logging
# =============================================================================

logFile=""
if [ "$LOG" -eq 1 ]; then
  logFile="$(pwd)/log.txt"
  echo "Logging enabled: $logFile"
  printf "=================================================\n" > "$logFile"
  printf "Log started at: %s\n" "$(date)" >> "$logFile"
  printf "Category: %s\n" "$CATEGORY" >> "$logFile"
  printf "=================================================\n\n" >> "$logFile"
fi

# =============================================================================
# Handle Selection Parameters (-i, -b, -e, -n)
# =============================================================================

# Validate mutually exclusive options
# -i (single index) and -n (name pattern) are exclusive
# -b and -e can be used together (range) but not with -i or -n
paramCount=0
if [ "${INDEX:-0}" -gt 0 ]; then paramCount=$((paramCount+1)); fi
if [ "${BEGIN:-0}" -gt 0 ] || [ "${END:-0}" -gt 0 ]; then paramCount=$((paramCount+1)); fi
if [ -n "${NAME}" ]; then paramCount=$((paramCount+1)); fi

if [ "$paramCount" -gt 1 ]; then
  echo "Error: Specify only one of -i (Index), -b/-e (Range), or -n (Name)." >&2
  exit 1
fi

if [ "${INDEX:-0}" -gt 0 ]; then
  if [ "$INDEX" -lt 1 ] || [ "$INDEX" -gt "${#selectedExamples[@]}" ]; then
    echo "Error: Index $INDEX is out of range. Valid range is 1..${#selectedExamples[@]}." >&2
    exit 1
  fi
  idx=$((INDEX-1))
  selectedExamples=("${selectedExamples[$idx]}")
  echo "Running single example #$INDEX: ${selectedExamples[0]}"
elif [ "${BEGIN:-0}" -gt 0 ]; then
  if [ "$BEGIN" -lt 1 ] || [ "$BEGIN" -gt "${#selectedExamples[@]}" ]; then
    echo "Error: Start index $BEGIN is out of range. Valid range is 1..${#selectedExamples[@]}." >&2
    exit 1
  fi
  startIndex=$((BEGIN-1))
  if [ "${END:-0}" -gt 0 ]; then
    if [ "$END" -lt "$BEGIN" ] || [ "$END" -gt "${#selectedExamples[@]}" ]; then
      echo "Error: End index $END is invalid. Valid range is $BEGIN..${#selectedExamples[@]}." >&2
      exit 1
    fi
    endIndex=$((END-1))
    selectedExamples=("${selectedExamples[@]:$startIndex:$((endIndex-startIndex+1))}")
  else
    selectedExamples=("${selectedExamples[@]:$startIndex}")
  fi
  echo "Running examples from #$BEGIN (count: ${#selectedExamples[@]})"
elif [ -n "$NAME" ]; then
  matched=()
  shopt -s nocasematch
  for ex in "${selectedExamples[@]}"; do
    basename_ex=$(basename "$ex")
    if [[ "$basename_ex" == $NAME ]] || [[ "$ex" == *"$NAME"* ]]; then
      matched+=("$ex")
    fi
  done
  shopt -u nocasematch
  if [ ${#matched[@]} -eq 0 ]; then
    echo "Error: No example matching '$NAME' found." >&2
    echo "Available examples:"
    for ex in "${selectedExamples[@]}"; do
      echo "  $ex"
    done
    exit 1
  fi
  selectedExamples=("${matched[@]}")
  echo "Running ${#selectedExamples[@]} example(s) matching: $NAME"
fi

# =============================================================================
# Run Examples
# =============================================================================

# Disable emojis when logging to avoid encoding issues
if [ "$LOG" -eq 1 ]; then
  export DISABLE_EMOJI=1
  echo "Emojis disabled for logging"
else
  unset DISABLE_EMOJI 2>/dev/null || true
fi

# Track failed examples
failedExamples=()
passedExamples=()

for example in "${selectedExamples[@]}"; do
  if [ -f "$example" ]; then
    startTime=$(date +%s)
    echo ""
    echo "########################################"
    echo "Launch: $example"
    echo "########################################"

    args=()
    case "$example" in
      */U07_multimodal.py|*/U08_multimodal_targets.py|*/U09_multimodal_missing_sources.py|*/U10_multimodal_late_tuning.py|*/U11_multimodal_data_provider.py)
        # These demonstrations emit JSON and archives, with durable search state.
        # A fresh directory keeps repeated runner invocations independent.
        artifact_dir=$(mktemp -d "${TMPDIR:-/tmp}/nirs4all-multimodal.XXXXXX")
        args+=("--output" "$artifact_dir")
        echo "Artifacts: $artifact_dir"
        ;;
      *)
        if [ "$PLOT" -eq 1 ]; then args+=("--plots"); fi
        if [ "$SHOW" -eq 1 ]; then args+=("--show"); fi
        ;;
    esac

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

    endTime=$(date +%s)
    duration=$((endTime - startTime))
    hours=$((duration/3600))
    mins=$(((duration%3600)/60))
    secs=$((duration%60))

    if [ "$exitCode" -ne 0 ]; then
      failedExamples+=("$example (exit code: $exitCode)")
      echo "FAILED: $example with exit code $exitCode"
    else
      passedExamples+=("$example")
    fi

    printf "Finished: %s (Duration: %02d:%02d:%02d)\n" "$example" "$hours" "$mins" "$secs"
    echo "########################################"
  else
    echo "SKIPPED (not found): $example"
  fi
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "========================================"
echo "EXECUTION SUMMARY"
echo "========================================"
echo "Category: $CATEGORY"
echo "Total examples run: ${#selectedExamples[@]}"
echo "Passed: ${#passedExamples[@]}"
echo "Failed: ${#failedExamples[@]}"

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
