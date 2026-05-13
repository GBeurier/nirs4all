#!/usr/bin/env bash
# Build the JavaScript / WASM bundle. Requires emsdk (https://emscripten.org).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INCLUDE="$HERE/../cpp/include"
DIST="$HERE/dist"
mkdir -p "$DIST"
emcc -std=c++17 -O3 \
     -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE \
     -I "$INCLUDE" \
     "$HERE/src/aompls_wasm.cpp" \
     -lembind \
     -sMODULARIZE=1 \
     -sEXPORT_ES6=1 \
     -sENVIRONMENT=node,web \
     -sALLOW_MEMORY_GROWTH=1 \
     -o "$DIST/aompls.mjs"
echo "Built $DIST/aompls.mjs (and aompls.wasm)"
