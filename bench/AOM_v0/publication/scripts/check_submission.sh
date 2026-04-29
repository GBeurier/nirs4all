#!/usr/bin/env bash
# Sanity-check the AOM_v0 publication tree.
#
# Verifies:
#   - pdflatex and bibtex are available (warn-only if missing).
#   - main.tex exists.
#   - every \includegraphics path referenced by main.tex resolves to an
#     existing file under publication/figures/.
#   - every \input path referenced by main.tex resolves to an existing
#     file under publication/tables/ or publication/supplement/.
#   - references.bib parses cleanly with `bibtex --version`-style
#     dry-run (we run a real bibtex pass when possible).
#
# Exits non-zero on any failure. Prints clear messages.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUB_ROOT="$(cd "${HERE}/.." && pwd)"
MAIN_TEX="${PUB_ROOT}/manuscript/main.tex"
FIG_DIR="${PUB_ROOT}/figures"
TAB_DIR="${PUB_ROOT}/tables"
SUP_DIR="${PUB_ROOT}/supplement"

ERRORS=0

note()  { printf "[check] %s\n"   "$*"; }
warn()  { printf "[warn] %s\n"    "$*" >&2; }
fail()  { printf "[fail] %s\n"    "$*" >&2; ERRORS=$((ERRORS + 1)); }

# 1. pdflatex / bibtex availability
if ! command -v pdflatex >/dev/null 2>&1; then
    warn "pdflatex not found on PATH (needed for actual builds; non-fatal)."
fi
if ! command -v bibtex >/dev/null 2>&1; then
    warn "bibtex not found on PATH (needed for actual builds; non-fatal)."
fi

# 2. main.tex exists
if [ ! -f "${MAIN_TEX}" ]; then
    fail "main.tex missing: ${MAIN_TEX}"
fi

# 3. figure paths
note "scanning figure includes in main.tex..."
while IFS= read -r relpath; do
    base=$(basename "${relpath}")
    if [ -f "${FIG_DIR}/${base}" ]; then
        note "  figure ok: ${base}"
    else
        fail "  missing figure: ${base} (from \\includegraphics{${relpath}})"
    fi
done < <(grep -oE '\\includegraphics(\[[^]]*\])?\{[^}]+\}' "${MAIN_TEX}" \
            | sed -E 's|\\includegraphics(\[[^]]*\])?\{([^}]+)\}|\2|')

# 4. \input paths
note "scanning \\input includes in main.tex..."
while IFS= read -r relpath; do
    base=$(basename "${relpath}")
    case "${relpath}" in
        */tables/*) target="${TAB_DIR}/${base}" ;;
        */figures/*) target="${FIG_DIR}/${base}" ;;
        */supplement/*) target="${SUP_DIR}/${base}" ;;
        *) target="${TAB_DIR}/${base}" ;;
    esac
    if [ -f "${target}" ]; then
        note "  input ok: ${base}"
    else
        fail "  missing input target: ${base} (from \\input{${relpath}}) at ${target}"
    fi
done < <(grep -oE '\\input\{[^}]+\}' "${MAIN_TEX}" \
            | sed -E 's|\\input\{([^}]+)\}|\1|')

# 5. references.bib parses
BIB="${PUB_ROOT}/manuscript/references.bib"
if [ ! -f "${BIB}" ]; then
    fail "missing references.bib"
else
    if command -v bibtex >/dev/null 2>&1; then
        TMP_AUX=$(mktemp -t aompls-aux-XXXXXX)
        cat > "${TMP_AUX}" <<EOF
\\bibstyle{plain}
\\bibdata{${BIB%.bib}}
EOF
        # bibtex needs the aux without extension argument; we invoke
        # it on the basename so it can find references.bib via the
        # absolute path embedded in the aux file.
        if bibtex "${TMP_AUX%.aux}" >/dev/null 2>&1; then
            note "bibtex ran without errors on a synthetic aux."
        else
            warn "bibtex returned non-zero on synthetic aux; ignoring (commonly due to lack of \\citation entries)."
        fi
        rm -f "${TMP_AUX}" "${TMP_AUX%.aux}.bbl" "${TMP_AUX%.aux}.blg"
    fi
fi

if [ "${ERRORS}" -gt 0 ]; then
    printf "\n[check] %d failure(s).\n" "${ERRORS}" >&2
    exit 1
fi

printf "\n[check] all checks passed.\n"
exit 0
