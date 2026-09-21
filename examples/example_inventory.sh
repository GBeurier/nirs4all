#!/usr/bin/env bash

# Canonical example inventory shared by the interactive and CI runners.
# Qualification scripts are release-audit utilities, not user tutorials.
shopt -s globstar nullglob

user_examples=()
for example in user/**/U*.py; do
  if [[ "$example" != *_qualification.py ]]; then
    user_examples+=("$example")
  fi
done

developer_examples=(developer/**/D*.py)
reference_examples=(reference/R*.py)

unset example
shopt -u globstar nullglob
