# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PIHM (MM-PIHM) build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from MM-PIHM GitHub releases.
Tier 2 (fallback): Build from source using CMake with SUNDIALS dependency.

MM-PIHM is the actively maintained multi-module variant of the Penn State
Integrated Hydrologic Model. Pre-compiled binaries may be available for
Linux and macOS.
"""
from __future__ import annotations

from symfluence.core.registries import R


@R.build_instructions.add('pihm')
def get_pihm_build_instructions():
    """
    Get PIHM (MM-PIHM) build/install instructions.

    Downloads pre-compiled binary from GitHub releases or builds from
    source with CMake + SUNDIALS.

    Returns:
        Dictionary with complete build configuration for PIHM.
    """
    return {
        'description': 'PIHM (Penn State Integrated Hydrologic Model / MM-PIHM)',
        'config_path_key': 'PIHM_INSTALL_PATH',
        'config_exe_key': 'PIHM_EXE',
        'default_path_suffix': 'installs/pihm/bin',
        'default_exe': 'flux-pihm',
        'repository': 'https://github.com/PSUmodeling/MM-PIHM.git',
        'branch': 'main',
        'install_dir': 'pihm',
        'build_commands': [
            r'''
# MM-PIHM Install Script for SYMFLUENCE
# Tier 1: Download pre-compiled binary from GitHub releases
# Tier 2: Build from source with CMake + SUNDIALS (fallback)

set -e

echo "=== MM-PIHM Installation Starting ==="

# Resolve INSTALL_DIR to absolute path
INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
mkdir -p "${INSTALL_DIR}/bin"

echo "Install directory: ${INSTALL_DIR}"

# Platform detection
UNAME_S=$(uname -s)
UNAME_M=$(uname -m)

case "$UNAME_S" in
    Linux)
        PLATFORM="linux"
        ;;
    Darwin)
        if [ "$UNAME_M" = "arm64" ]; then
            PLATFORM="darwin-arm64"
        else
            PLATFORM="darwin-x86_64"
        fi
        ;;
    *)
        echo "WARNING: Unknown platform $UNAME_S, will try source build"
        PLATFORM="unknown"
        ;;
esac

echo "Detected platform: $PLATFORM ($UNAME_S $UNAME_M)"

# === Tier 1: Download pre-compiled binary ===
DOWNLOAD_SUCCESS=false

if [ "$PLATFORM" != "unknown" ]; then
    echo "Attempting binary download from MM-PIHM GitHub releases..."

    _api_url="https://api.github.com/repos/PSUmodeling/MM-PIHM/releases/latest"
    _api_json=""
    if command -v curl >/dev/null 2>&1; then
        _api_json=$(curl -fsSL -H "Accept: application/vnd.github+json" "$_api_url" 2>/dev/null) || true
    fi
    if [ -z "$_api_json" ] && command -v wget >/dev/null 2>&1; then
        _api_json=$(wget -qO- --header="Accept: application/vnd.github+json" "$_api_url" 2>/dev/null) || true
    fi
    LATEST_TAG=""
    if [ -n "$_api_json" ]; then
        LATEST_TAG=$(echo "$_api_json" | python3 -c "import sys, json; print(json.load(sys.stdin).get('tag_name', ''))" 2>/dev/null) || true
    fi

    if [ -z "$LATEST_TAG" ]; then
        echo "WARNING: Could not determine latest release tag"
        LATEST_TAG="v2.0"
    fi

    echo "Latest release: $LATEST_TAG"

    DOWNLOAD_URL="https://github.com/PSUmodeling/MM-PIHM/releases/download/${LATEST_TAG}/mm-pihm-${LATEST_TAG}-${PLATFORM}.tar.gz"
    echo "Download URL: $DOWNLOAD_URL"

    TMPTAR=$(mktemp /tmp/pihm_XXXXXX.tar.gz)
    TMPEXTRACT=$(mktemp -d /tmp/pihm_extract_XXXXXX)

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ ! -s "$TMPTAR" ] && command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ -s "$TMPTAR" ]; then
        if file "$TMPTAR" | grep -q "gzip\|tar"; then
            tar xzf "$TMPTAR" -C "$TMPEXTRACT" 2>/dev/null || true

            # The adapter writes Noah-specific .ic/.lsm files. A plain PIHM
            # release binary cannot run those inputs.
            PIHM_BIN=$(find "$TMPEXTRACT" -type f -name "flux-pihm" | head -1)

            if [ -n "$PIHM_BIN" ]; then
                cp "$PIHM_BIN" "${INSTALL_DIR}/bin/pihm"
                chmod +x "${INSTALL_DIR}/bin/pihm"

                if "${INSTALL_DIR}/bin/pihm" --version >/dev/null 2>&1 || \
                   "${INSTALL_DIR}/bin/pihm" -v >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                else
                    echo "WARNING: Downloaded pihm binary cannot run on this system"
                    rm -f "${INSTALL_DIR}/bin/pihm"
                fi
            fi
        fi
    fi

    rm -f "$TMPTAR"
    rm -rf "$TMPEXTRACT"
fi

# === Tier 2: Build from source with CMake ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Binary download failed, attempting source build with CMake..."

    for tool in cc make; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "ERROR: $tool not found. Install Xcode CLT: xcode-select --install"
            exit 1
        fi
    done

    # MM-PIHM uses a plain Makefile with bundled CVODE solver
    # The repo was already cloned by the framework into INSTALL_DIR
    SRC_DIR="${INSTALL_DIR}"

    if [ ! -f "${SRC_DIR}/Makefile" ]; then
        # If not cloned yet, clone manually
        BUILD_TMPDIR=$(mktemp -d /tmp/pihm_build_XXXXXX)
        echo "Cloning MM-PIHM source..."
        git clone --depth 1 https://github.com/PSUmodeling/MM-PIHM.git "${BUILD_TMPDIR}/mm-pihm"
        SRC_DIR="${BUILD_TMPDIR}/mm-pihm"
    fi

    echo "Building MM-PIHM from source in ${SRC_DIR}..."

    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

    # MM-PIHM aliases timegm/strcasecmp to their _-prefixed MSVCRT equivalents,
    # but guards them with `#if defined(_MSC_VER)` — MSVC only. MinGW (GCC) has
    # those same MSVCRT functions yet defines _WIN32, not _MSC_VER, so the
    # aliases are skipped and time_func.c fails ("implicit declaration of
    # timegm"). Widen the guard to _WIN32 so MinGW gets them too.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            perl -0777 -pi -e 's/#if defined\(_MSC_VER\)\s*\n(# define timegm)/#if defined(_WIN32)\n$1/' \
                "${SRC_DIR}/src/include/pihm_func.h" 2>/dev/null || true
            ;;
    esac

    # Build CVODE first (bundled SUNDIALS solver), then PIHM
    # Must be sequential: PIHM needs CVODE headers installed before compiling
    cd "${SRC_DIR}"
    make clean 2>/dev/null || true
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            # MM-PIHM's `cvode` target runs `cmake ../` then a bare `make`, but
            # mingw cmake defaults to the Ninja generator (emits build.ninja, no
            # Makefile) so the bare make dies ("No targets ... no makefile
            # found", Error 2) — and CMAKE_GENERATOR alone didn't override it.
            # Replicate the cvode configure (Makefile:352) and build with
            # `cmake --build`, which works with whatever generator cmake picked.
            echo "Windows: building CVODE via cmake --build (generator-agnostic)..."
            mkdir -p cvode/instdir
            ( cd cvode/instdir && cmake \
                -DCMAKE_INSTALL_PREFIX=../instdir -DCMAKE_INSTALL_LIBDIR=lib \
                -DBUILD_SHARED_LIBS=OFF -DEXAMPLES_ENABLE_C=OFF -DEXAMPLES_INSTALL=OFF ../ \
              && cmake --build . \
              && cmake --build . --target install )
            ;;
        *)
            make cvode
            ;;
    esac
    # Build the Flux-PIHM variant (-D_NOAH_): SYMFLUENCE's preprocessor writes the
    # Noah-LSM initial-condition layout (.ic) and the .lsm file, so the binary must
    # be the Noah-enabled build. The plain `pihm` target expects a smaller .ic and
    # rejects ours with "Error in initial condition file ... size does not match".
    make flux-pihm -j "${NCPU}"

    # The Makefile places the binary in the source root (gcc -o flux-pihm ...);
    # on Windows it's flux-pihm.exe. Prefer the flux-pihm name, but tolerate builds
    # that emit a plain `pihm` name for the same target.
    PIHM_BIN=""
    for p in "${SRC_DIR}/flux-pihm" "${SRC_DIR}/flux-pihm.exe" "${SRC_DIR}/bin/flux-pihm" "${SRC_DIR}/bin/flux-pihm.exe" \
             "${SRC_DIR}/pihm" "${SRC_DIR}/pihm.exe" "${SRC_DIR}/bin/pihm" "${SRC_DIR}/bin/pihm.exe"; do
        if [ -f "$p" ]; then
            PIHM_BIN="$p"
            break
        fi
    done
    if [ -z "$PIHM_BIN" ]; then
        # Fallback: search for any executable named flux-pihm or pihm
        PIHM_BIN=$(find "${SRC_DIR}" -maxdepth 2 \( -name "flux-pihm" -o -name "flux-pihm.exe" -o -name "pihm" -o -name "pihm.exe" \) -type f 2>/dev/null | head -1)
    fi

    if [ -n "$PIHM_BIN" ]; then
        mkdir -p "${INSTALL_DIR}/bin"
        case "$PIHM_BIN" in
            *.exe) cp "$PIHM_BIN" "${INSTALL_DIR}/bin/pihm.exe" ;;
            *)     cp "$PIHM_BIN" "${INSTALL_DIR}/bin/pihm" ;;
        esac
        chmod +x "${INSTALL_DIR}/bin/pihm"* 2>/dev/null || true
    else
        echo "ERROR: Build succeeded but flux-pihm/pihm binary not found"
        find "${SRC_DIR}" -maxdepth 2 \( -name "flux-pihm*" -o -name "pihm*" \) -type f 2>/dev/null || echo "No pihm files found"
        exit 1
    fi

    # Preserve required global tables before removing a temporary source tree.
    if [ "$SRC_DIR" != "$INSTALL_DIR" ] && [ -d "${SRC_DIR}/input" ]; then
        cp -a "${SRC_DIR}/input" "${INSTALL_DIR}/input"
    fi

    # Cleanup temp dir if we used one
    if [ -n "${BUILD_TMPDIR:-}" ]; then
        rm -rf "${BUILD_TMPDIR}"
    fi
    echo "Build complete"
fi

# === Verify installation ===
# Keep the legacy pihm launcher and expose the canonical Noah-enabled name.
if [ -f "${INSTALL_DIR}/bin/pihm" ]; then
    cp "${INSTALL_DIR}/bin/pihm" "${INSTALL_DIR}/bin/flux-pihm"
elif [ -f "${INSTALL_DIR}/bin/pihm.exe" ]; then
    cp "${INSTALL_DIR}/bin/pihm.exe" "${INSTALL_DIR}/bin/flux-pihm.exe"
fi
if [ -f "${INSTALL_DIR}/bin/pihm" ]; then
    echo ""
    echo "=== MM-PIHM Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/pihm"
    "${INSTALL_DIR}/bin/pihm" -v 2>/dev/null || echo "(version check not supported)"
else
    echo "ERROR: pihm not found at ${INSTALL_DIR}/bin/pihm"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'make'],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/flux-pihm', 'bin/flux-pihm.exe'],
            'check_type': 'exists_any'
        },
        'order': 27,  # After ParFlow (26)
        'optional': True,
    }
