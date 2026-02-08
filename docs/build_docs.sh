#!/bin/bash
# Documentation build script for nirs4all

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================"
echo "  nirs4all Documentation Builder"
echo "================================================"
echo ""

# Parse arguments
CLEAN=false
OPEN=false
STRICT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c)
            CLEAN=true
            shift
            ;;
        --open|-o)
            OPEN=true
            shift
            ;;
        --strict|-s)
            STRICT=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --clean    Clean build directory before building"
            echo "  -o, --open     Open documentation in browser after building"
            echo "  -s, --strict   Treat warnings as errors"
            echo "  -h, --help     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                # Basic build"
            echo "  $0 --clean --open # Clean build and open in browser"
            echo "  $0 --strict       # Build with warnings as errors"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if sphinx-build is available
if ! command -v sphinx-build &> /dev/null; then
    echo "❌ Error: sphinx-build not found"
    echo ""
    echo "Install documentation dependencies:"
    echo "  pip install -r readthedocs.requirements.txt"
    exit 1
fi

# Clean build directory if requested
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf _build/
    echo "✅ Clean complete"
    echo ""
fi

# Build documentation
echo "📚 Building documentation..."
echo ""

SPHINXOPTS="-j auto"  # Parallel build

if [ "$STRICT" = true ]; then
    SPHINXOPTS="$SPHINXOPTS -W --keep-going"
    echo "⚠️  Building in STRICT mode (warnings as errors)"
    echo ""
fi

if sphinx-build -b html $SPHINXOPTS source _build/html; then
    echo ""
    echo "✅ Documentation built successfully!"
    echo ""
    echo "📂 Output directory: _build/html/"
    echo "🌐 Open: file://$SCRIPT_DIR/_build/html/index.html"
    echo ""

    # Count warnings
    if [ -f "_build/html/output.txt" ]; then
        WARNING_COUNT=$(grep -c "WARNING" "_build/html/output.txt" 2>/dev/null || echo "0")
        if [ "$WARNING_COUNT" -gt 0 ]; then
            echo "⚠️  Found $WARNING_COUNT warnings"
        fi
    fi

    # Open in browser if requested
    if [ "$OPEN" = true ]; then
        echo "🌐 Opening documentation in browser..."
        if command -v xdg-open &> /dev/null; then
            xdg-open "_build/html/index.html"
        elif command -v open &> /dev/null; then
            open "_build/html/index.html"
        else
            echo "⚠️  Could not detect browser opener (xdg-open or open)"
        fi
    fi
else
    echo ""
    echo "❌ Build failed!"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: pip install -r readthedocs.requirements.txt"
    echo "  - Syntax errors in .md or .rst files"
    echo "  - Missing referenced files"
    echo "  - Invalid Mermaid diagram syntax"
    echo ""
    exit 1
fi

echo "================================================"
