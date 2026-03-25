#!/bin/bash
#
# Create release tarball for SYMFLUENCE tools
#
# Creates:
#   - symfluence-tools-<version>-<platform>.tar.gz
#   - symfluence-tools-<version>-<platform>.tar.gz.sha256
#
# Usage:
#   ./scripts/create_release_tarball.sh <version> <platform> <staged_dir> <output_dir>
#
# Example:
#   ./scripts/create_release_tarball.sh \
#     v0.7.0 \
#     linux-x86_64 \
#     ./release/symfluence-tools \
#     ./release

set -e

# Colors
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    GREEN=''; BLUE=''; NC='';
fi

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }

# Parse arguments
VERSION="${1:-}"
PLATFORM="${2:-}"
STAGED_DIR="${3:-}"
OUTPUT_DIR="${4:-}"

# Portable realpath fallback
_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")")
    fi
}

# Resolve absolute paths
STAGED_DIR="$(_realpath "$STAGED_DIR")"
OUTPUT_DIR="$(_realpath "$OUTPUT_DIR")"

if [ ! -d "$STAGED_DIR" ]; then
    print_error "Staged directory not found: $STAGED_DIR"
    exit 1
fi

# Sanitize VERSION for use in filenames (e.g., PR refs like "22/merge")
VERSION="${VERSION//\//-}"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Tarball name
TARBALL_NAME="symfluence-tools-${VERSION}-${PLATFORM}.tar.gz"
TARBALL_PATH="$OUTPUT_DIR/$TARBALL_NAME"

print_info "Creating release tarball..."
print_info "Version: $VERSION"
print_info "Platform: $PLATFORM"
print_info "Source: $STAGED_DIR"
print_info "Output: $TARBALL_PATH"

# Create tarball
cd "$(dirname "$STAGED_DIR")"
STAGED_BASENAME="$(basename "$STAGED_DIR")"

print_info "Compressing..."
# On Windows (MSYS/MinGW), tar interprets D: as a remote host.
# --force-local fixes this but is not supported by BSD tar (macOS).
TAR_ARGS="-czf"
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*) TAR_ARGS="--force-local $TAR_ARGS" ;;
esac
tar $TAR_ARGS "$TARBALL_PATH" "$STAGED_BASENAME"

TARBALL_SIZE="$(du -h "$TARBALL_PATH" | cut -f1)"
print_success "Created tarball: $TARBALL_NAME ($TARBALL_SIZE)"

# Generate checksum
print_info "Generating SHA256 checksum..."

cd "$OUTPUT_DIR"
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$TARBALL_NAME" > "${TARBALL_NAME}.sha256"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$TARBALL_NAME" > "${TARBALL_NAME}.sha256"
else
    echo "Warning: No SHA256 tool found (sha256sum or shasum)" >&2
    exit 1
fi

CHECKSUM="$(cat "${TARBALL_NAME}.sha256")"
print_success "Generated checksum: ${TARBALL_NAME}.sha256"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Release Artifact Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "Version: $VERSION"
print_success "Platform: $PLATFORM"
print_success "Tarball: $TARBALL_NAME ($TARBALL_SIZE)"
print_success "Checksum: $CHECKSUM"
echo ""
print_info "Files created:"
ls -lh "$OUTPUT_DIR/$TARBALL_NAME"*
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
