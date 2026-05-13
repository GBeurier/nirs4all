#!/usr/bin/env bash
# Mirror cpp/include/ → r/aompls/inst/include/ (idempotent).
# Run before `R CMD build r/aompls` so the package ships standalone.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$HERE/cpp/include"
DST="$HERE/r/aompls/inst/include"

rm -rf "$DST/aompls" "$DST/Eigen"
mkdir -p "$DST"
cp -r "$SRC/aompls" "$DST/aompls"
cp -r "$SRC/Eigen" "$DST/Eigen"

echo "Synced cpp/include/{aompls,Eigen} → $DST"
