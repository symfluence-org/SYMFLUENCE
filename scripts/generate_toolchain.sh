#!/bin/bash
#
# Generate toolchain.json metadata for SYMFLUENCE binary artifacts
#
# This script captures:
# - Tool versions (commit hashes, branches, upstream repository, and the
#   source pin recorded by `symfluence binary install` in each tool's
#   .symfluence_source.json)
# - Compiler versions (Fortran, C, C++)
# - Library versions (NetCDF, HDF5, MPI)
# - Build metadata (date, platform)
#
# Usage:
#   ./scripts/generate_toolchain.sh <installs_dir> <output_file> [platform]
#
# Example:
#   ./scripts/generate_toolchain.sh \
#     /path/to/SYMFLUENCE_data/installs \
#     /path/to/output/toolchain.json \
#     linux-x86_64

set -e

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC='';
fi

print_error()   { echo -e "${RED}Error:${NC} $1" >&2; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }

# Parse arguments
INSTALLS_DIR="${1:-}"
OUTPUT_FILE="${2:-}"
PLATFORM="${3:-}"

if [ -z "$INSTALLS_DIR" ] || [ -z "$OUTPUT_FILE" ]; then
    cat >&2 <<EOF
Usage: $0 <installs_dir> <output_file> [platform]

Arguments:
  installs_dir   Directory containing tool installations (e.g., \$SYMFLUENCE_DATA/installs)
  output_file    Path to output toolchain.json file
  platform       Platform identifier (e.g., linux-x86_64, macos-arm64)
                 Auto-detected if not provided

Example:
  $0 /path/to/installs /path/to/toolchain.json linux-x86_64
EOF
    exit 1
fi

if [ ! -d "$INSTALLS_DIR" ]; then
    print_error "Installation directory not found: $INSTALLS_DIR"
    exit 1
fi

print_info "Generating toolchain metadata..."
print_info "Installation directory: $INSTALLS_DIR"
print_info "Output file: $OUTPUT_FILE"

# Auto-detect platform if not provided
if [ -z "$PLATFORM" ]; then
    OS_TYPE="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS_TYPE" in
        Linux)
            PLATFORM="linux-${ARCH}"
            ;;
        Darwin)
            PLATFORM="macos-${ARCH}"
            ;;
        *)
            PLATFORM="unknown-${ARCH}"
            ;;
    esac

    print_info "Auto-detected platform: $PLATFORM"
fi

# Get SYMFLUENCE version from git or environment
if [ -n "${GITHUB_REF_NAME:-}" ]; then
    SYMFLUENCE_VERSION="$GITHUB_REF_NAME"
elif git rev-parse --git-dir > /dev/null 2>&1; then
    SYMFLUENCE_VERSION="$(git describe --tags --always 2>/dev/null || git rev-parse --short HEAD)"
else
    SYMFLUENCE_VERSION="unknown"
fi

# Build timestamp
BUILD_DATE="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

print_info "SYMFLUENCE version: $SYMFLUENCE_VERSION"
print_info "Build date: $BUILD_DATE"

# Helper function to get git info for a tool
get_tool_git_info() {
    local tool_dir="$1"
    local tool_name="$2"

    if [ ! -d "$tool_dir" ]; then
        echo "\"$tool_name\": {\"error\": \"not installed\"}"
        return
    fi

    cd "$tool_dir"

    local commit="unknown"
    local branch="unknown"
    local repository="unknown"
    local source_json="null"

    if [ -d ".git" ]; then
        commit="$(git rev-parse HEAD 2>/dev/null || echo "unknown")"
        branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")"
        repository="$(git config --get remote.origin.url 2>/dev/null || echo "unknown")"
    fi
    # Source record written by `symfluence binary install`: repository, the
    # pinned commit (binary_pins.json) and the commit actually built.
    if [ -f ".symfluence_source.json" ]; then
        source_json="$(cat .symfluence_source.json)"
    fi

    # Get relative executable path based on tool
    local exe_path=""
    case "$tool_name" in
        summa)
            if [ -f "bin/summa.exe" ]; then
                exe_path="bin/summa.exe"
            elif [ -f "bin/summa_sundials.exe" ]; then
                exe_path="bin/summa_sundials.exe"
            fi
            ;;
        mizuroute)
            if [ -f "route/bin/mizuRoute.exe" ]; then
                exe_path="route/bin/mizuRoute.exe"
            fi
            ;;
        fuse)
            if [ -f "bin/fuse.exe" ]; then
                exe_path="bin/fuse.exe"
            fi
            ;;
        ngen)
            if [ -f "cmake_build/ngen" ]; then
                exe_path="cmake_build/ngen"
            fi
            ;;
        hype)
            if [ -f "bin/hype" ]; then
                exe_path="bin/hype"
            fi
            ;;
        taudem)
            if [ -f "bin/pitremove" ]; then
                exe_path="bin/pitremove"
            fi
            ;;
        sundials)
            if [ -f "install/sundials/lib/libsundials_core.a" ]; then
                exe_path="install/sundials/lib/libsundials_core.a"
            elif [ -f "install/sundials/lib64/libsundials_core.a" ]; then
                exe_path="install/sundials/lib64/libsundials_core.a"
            fi
            ;;
    esac

    cat <<EOF
    "$tool_name": {
      "commit": "$commit",
      "branch": "$branch",
      "repository": "$repository",
      "source": $source_json,
      "executable": "$exe_path",
      "installed": true
    }
EOF
}

# Get compiler versions
print_info "Detecting compilers..."

FC_VERSION="unknown"
if command -v gfortran >/dev/null 2>&1; then
    FC_VERSION="$(gfortran --version 2>/dev/null | head -1 || echo "unknown")"
elif command -v ifort >/dev/null 2>&1; then
    FC_VERSION="$(ifort --version 2>/dev/null | head -1 || echo "unknown")"
fi

CC_VERSION="unknown"
if command -v gcc >/dev/null 2>&1; then
    CC_VERSION="$(gcc --version 2>/dev/null | head -1 || echo "unknown")"
elif command -v clang >/dev/null 2>&1; then
    CC_VERSION="$(clang --version 2>/dev/null | head -1 || echo "unknown")"
fi

CXX_VERSION="unknown"
if command -v g++ >/dev/null 2>&1; then
    CXX_VERSION="$(g++ --version 2>/dev/null | head -1 || echo "unknown")"
elif command -v clang++ >/dev/null 2>&1; then
    CXX_VERSION="$(clang++ --version 2>/dev/null | head -1 || echo "unknown")"
fi

print_success "Fortran: $FC_VERSION"
print_success "C: $CC_VERSION"
print_success "C++: $CXX_VERSION"

# Get library versions
print_info "Detecting libraries..."

NETCDF_VERSION="unknown"
if command -v nc-config >/dev/null 2>&1; then
    NETCDF_VERSION="$(nc-config --version 2>/dev/null || echo "unknown")"
fi

NETCDF_FORTRAN_VERSION="unknown"
if command -v nf-config >/dev/null 2>&1; then
    NETCDF_FORTRAN_VERSION="$(nf-config --version 2>/dev/null || echo "unknown")"
fi

HDF5_VERSION="unknown"
if command -v h5cc >/dev/null 2>&1; then
    HDF5_VERSION="$(h5cc -showconfig 2>/dev/null | grep "HDF5 Version" | sed 's/.*: *//' || echo "unknown")"
fi

print_success "NetCDF: $NETCDF_VERSION"
print_success "NetCDF-Fortran: $NETCDF_FORTRAN_VERSION"
print_success "HDF5: $HDF5_VERSION"

# MPI detection
print_info "Detecting MPI..."

MPI_VERSION="none"
MPI_TYPE="none"
if command -v mpirun >/dev/null 2>&1; then
    MPI_INFO="$(mpirun --version 2>&1 | head -1 || echo "")"
    if echo "$MPI_INFO" | grep -qi "Open MPI"; then
        MPI_TYPE="OpenMPI"
        MPI_VERSION="$MPI_INFO"
    elif echo "$MPI_INFO" | grep -qi "MPICH"; then
        MPI_TYPE="MPICH"
        MPI_VERSION="$MPI_INFO"
    else
        MPI_TYPE="unknown"
        MPI_VERSION="$MPI_INFO"
    fi
fi

print_success "MPI: $MPI_TYPE ($MPI_VERSION)"

# Collect tool information
print_info "Collecting tool information..."

TOOLS_JSON=""
TOOL_COUNT=0

for tool in summa mizuroute fuse ngen hype taudem sundials; do
    tool_dir="$INSTALLS_DIR/$tool"

    if [ "$tool" = "mizuroute" ]; then
        tool_dir="$INSTALLS_DIR/mizuRoute"
    elif [ "$tool" = "taudem" ]; then
        tool_dir="$INSTALLS_DIR/TauDEM"
    fi

    if [ -d "$tool_dir" ]; then
        print_success "Found $tool"
        tool_json="$(get_tool_git_info "$tool_dir" "$tool")"

        if [ $TOOL_COUNT -gt 0 ]; then
            TOOLS_JSON="${TOOLS_JSON},"
        fi
        TOOLS_JSON="${TOOLS_JSON}${tool_json}"
        TOOL_COUNT=$((TOOL_COUNT + 1))
    else
        print_info "Skipping $tool (not installed)"
    fi
done

# Every other tool built with a source record (the optional release models:
# vic, swat, prms, mhm, parflow, ...), named by the record's "tool" field.
for tool_dir in "$INSTALLS_DIR"/*/; do
    [ -f "${tool_dir}.symfluence_source.json" ] || continue
    tool="$(sed -n 's/^ *"tool": *"\([^"]*\)".*/\1/p' "${tool_dir}.symfluence_source.json" | head -1)"
    [ -n "$tool" ] || continue
    case " summa mizuroute fuse ngen hype taudem sundials " in
        *" $tool "*) continue ;;
    esac
    print_success "Found $tool (source record)"
    tool_json="$(get_tool_git_info "${tool_dir%/}" "$tool")"
    if [ $TOOL_COUNT -gt 0 ]; then
        TOOLS_JSON="${TOOLS_JSON},"
    fi
    TOOLS_JSON="${TOOLS_JSON}${tool_json}"
    TOOL_COUNT=$((TOOL_COUNT + 1))
done

# Generate JSON output
print_info "Writing toolchain.json..."

mkdir -p "$(dirname "$OUTPUT_FILE")"

cat > "$OUTPUT_FILE" <<EOF
{
  "symfluence_version": "$SYMFLUENCE_VERSION",
  "build_date": "$BUILD_DATE",
  "platform": "$PLATFORM",
  "tools": {
${TOOLS_JSON}
  },
  "compilers": {
    "fortran": "$FC_VERSION",
    "c": "$CC_VERSION",
    "cxx": "$CXX_VERSION",
    "mpi": {
      "type": "$MPI_TYPE",
      "version": "$MPI_VERSION"
    }
  },
  "libraries": {
    "netcdf": "$NETCDF_VERSION",
    "netcdf_fortran": "$NETCDF_FORTRAN_VERSION",
    "hdf5": "$HDF5_VERSION"
  },
  "generator": {
    "script": "scripts/generate_toolchain.sh",
    "hostname": "$(hostname 2>/dev/null || echo "unknown")",
    "user": "${USER:-unknown}"
  }
}
EOF

print_success "Toolchain metadata generated: $OUTPUT_FILE"

# Display summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Toolchain Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat "$OUTPUT_FILE" | python3 -m json.tool 2>/dev/null || cat "$OUTPUT_FILE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
