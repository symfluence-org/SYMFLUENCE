#!/bin/bash
# Version synchronization validation script
# Single source of truth: src/symfluence/symfluence_version.py
# All other version references must match.

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Extract version from single source of truth
PYTHON_VERSION=$(grep '^__version__' "$REPO_ROOT/src/symfluence/symfluence_version.py" | sed 's/.*"\([0-9.]*\)".*/\1/')

# Extract versions from publish-side manifests (the two package.json files
# that are actually used to npm-publish the binaries). Note: the root
# package-lock.json is intentionally NOT checked here — it is a
# consumer-install artifact pinning the *last published* version from the
# npm registry, not the version about to be published. Including it in
# this check creates a chicken-and-egg failure mode where the release
# workflow cannot publish version N until the lockfile resolves N, but
# the lockfile cannot resolve N until N has been published. The lockfile
# is refreshed via a follow-up `npm install --package-lock-only` after
# each publish lands. (This bit Release Binaries CI for three weeks in
# April 2026 — see the commit that introduced this comment for context.)
TOOLS_NPM_VERSION=$(grep '"version":' "$REPO_ROOT/tools/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')
NPM_VERSION=$(grep '"version":' "$REPO_ROOT/npm/package.json" | head -1 | sed 's/.*"\([0-9.]*\)".*/\1/')

echo "Checking version synchronization..."
echo "  Source of truth:"
echo "    symfluence_version.py: $PYTHON_VERSION"
echo "  Must match:"
echo "    tools/npm/package.json: $TOOLS_NPM_VERSION"
echo "    npm/package.json:       $NPM_VERSION"
echo ""

ERRORS=0

if [ "$PYTHON_VERSION" != "$TOOLS_NPM_VERSION" ]; then
    echo "❌ tools/npm/package.json ($TOOLS_NPM_VERSION) does not match ($PYTHON_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$PYTHON_VERSION" != "$NPM_VERSION" ]; then
    echo "❌ npm/package.json ($NPM_VERSION) does not match ($PYTHON_VERSION)"
    ERRORS=$((ERRORS + 1))
fi

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "❌ VERSION MISMATCH DETECTED!"
    echo ""
    echo "The single source of truth is: src/symfluence/symfluence_version.py"
    echo "Update all version references to match: $PYTHON_VERSION"
    echo ""
    exit 1
else
    echo "✓ All versions synchronized: $PYTHON_VERSION"
    exit 0
fi
