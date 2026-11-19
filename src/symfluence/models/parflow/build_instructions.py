# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
ParFlow build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from ParFlow GitHub releases.
Tier 2 (fallback): Build from source using CMake.

ParFlow is a parallel integrated hydrologic model. Pre-compiled
binaries are available for Linux and macOS.
"""
from __future__ import annotations

from symfluence.core.registries import R


@R.build_instructions.add('parflow')
def get_parflow_build_instructions():
    """
    Get ParFlow build/install instructions.

    Downloads pre-compiled ParFlow binary from GitHub releases.
    Falls back to CMake source build if download fails.

    Returns:
        Dictionary with complete build configuration for ParFlow.
    """
    return {
        'description': 'ParFlow (Parallel Integrated Hydrologic Model)',
        'config_path_key': 'PARFLOW_INSTALL_PATH',
        'config_exe_key': 'PARFLOW_EXE',
        'default_path_suffix': 'installs/parflow/bin',
        'default_exe': 'parflow',
        'repository': 'https://github.com/parflow/parflow.git',
        'branch': 'master',
        'install_dir': 'parflow',
        'build_commands': [
            r'''
# ParFlow Install Script for SYMFLUENCE
# Tier 1: Download pre-compiled binary from GitHub releases
# Tier 2: Build from source with CMake (fallback)

set -e

echo "=== ParFlow Installation Starting ==="

# Resolve INSTALL_DIR to an absolute path BEFORE any cd commands.
# This is critical — CMake's CMAKE_INSTALL_PREFIX must be absolute,
# otherwise `make install` targets the build dir, not the real prefix.
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
    echo "Attempting binary download from ParFlow GitHub releases..."

    # Discover latest release tag via GitHub API (try curl then wget)
    _api_url="https://api.github.com/repos/parflow/parflow/releases/latest"
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
        echo "WARNING: Could not determine latest release tag, trying default"
        LATEST_TAG="v3.12.0"
    fi

    echo "Latest release: $LATEST_TAG"

    # ParFlow releases may have platform-specific archives
    DOWNLOAD_URL="https://github.com/parflow/parflow/releases/download/${LATEST_TAG}/parflow-${LATEST_TAG}-${PLATFORM}.tar.gz"
    echo "Download URL: $DOWNLOAD_URL"

    TMPTAR=$(mktemp /tmp/parflow_XXXXXX.tar.gz)
    TMPEXTRACT=$(mktemp -d /tmp/parflow_extract_XXXXXX)

    # Try curl first, fall back to wget
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ ! -s "$TMPTAR" ] && command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ -s "$TMPTAR" ]; then
        # Verify it's actually a tar.gz file
        if file "$TMPTAR" | grep -q "gzip\|tar"; then
            tar xzf "$TMPTAR" -C "$TMPEXTRACT" 2>/dev/null || true

            # Find parflow binary
            PF_BIN=$(find "$TMPEXTRACT" -name "parflow" -type f 2>/dev/null | head -1)

            if [ -n "$PF_BIN" ]; then
                cp "$PF_BIN" "${INSTALL_DIR}/bin/parflow"
                chmod +x "${INSTALL_DIR}/bin/parflow"

                # Verify the binary actually runs (catches glibc mismatch
                # on HPC where pre-compiled binaries target newer glibc)
                if "${INSTALL_DIR}/bin/parflow" --version >/dev/null 2>&1 || \
                   "${INSTALL_DIR}/bin/parflow" -v >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                else
                    echo "WARNING: Downloaded parflow binary cannot run on this system (glibc mismatch?)"
                    echo "  Will fall back to source build"
                    rm -f "${INSTALL_DIR}/bin/parflow"
                fi
            else
                echo "WARNING: parflow binary not found in downloaded archive"
            fi
        else
            echo "WARNING: Downloaded file is not a valid archive"
        fi
    else
        echo "WARNING: Download failed or empty file"
    fi

    # Cleanup
    rm -f "$TMPTAR"
    rm -rf "$TMPEXTRACT"
fi

# === Tier 2: Build from source with CMake ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Binary download failed, attempting source build with CMake..."

    # Check dependencies
    for tool in cmake make; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "ERROR: $tool not found. Install with: brew install cmake (or apt install cmake)"
            exit 1
        fi
    done

    # Clone source into a temp directory (not inside INSTALL_DIR)
    BUILD_TMPDIR=$(mktemp -d /tmp/parflow_build_XXXXXX)
    echo "Build directory: ${BUILD_TMPDIR}"

    # Build from the checkout the installer made in INSTALL_DIR (pinned to
    # the commit in binary_pins.json), not from a fresh clone of upstream HEAD.
    if git -C "${INSTALL_DIR}" rev-parse -q --verify HEAD >/dev/null 2>&1; then
        _PF_COMMIT=$(git -C "${INSTALL_DIR}" rev-parse HEAD)
        echo "Copying ParFlow source at ${_PF_COMMIT}..."
        git clone -q --no-checkout "${INSTALL_DIR}" "${BUILD_TMPDIR}/parflow_src"
        git -C "${BUILD_TMPDIR}/parflow_src" checkout -q "${_PF_COMMIT}"
    else
        echo "Cloning ParFlow source..."
        git clone --depth 1 https://github.com/parflow/parflow.git "${BUILD_TMPDIR}/parflow_src"
    fi

    mkdir -p "${BUILD_TMPDIR}/parflow_src/build"

    # Windows (MSYS2/MinGW): ParFlow's AMPS "seq" layer unconditionally includes
    # POSIX-only <sys/times.h> (times()/struct tms/sysconf) and <sys/param.h>,
    # which MinGW lacks — the build dies at "fatal error: sys/times.h: No such
    # file". Provide header-only shims on a private include dir. ParFlow's real
    # timing uses gettimeofday (CASC_HAVE_GETTIMEOFDAY) and we pass
    # PARFLOW_ENABLE_TIMING=FALSE, so times() only needs to compile and link — a
    # clock()-based stub suffices (no <windows.h>, to avoid macro pollution).
    WIN_COMPAT_FLAG=""
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            _WC="${BUILD_TMPDIR}/parflow_src/win_compat"
            mkdir -p "${_WC}/sys"
            cat > "${_WC}/sys/times.h" <<'TIMESH'
#ifndef SF_WIN_SYS_TIMES_H
#define SF_WIN_SYS_TIMES_H
#include <time.h>
struct tms { clock_t tms_utime, tms_stime, tms_cutime, tms_cstime; };
static __inline__ clock_t times(struct tms *buf) {
    clock_t c = clock();
    if (buf) { buf->tms_utime = c; buf->tms_stime = 0;
               buf->tms_cutime = 0; buf->tms_cstime = 0; }
    return c;
}
#ifndef _SC_CLK_TCK
#define _SC_CLK_TCK 2
#endif
static __inline__ long sysconf(int name) { (void)name; return (long)CLOCKS_PER_SEC; }
#endif
TIMESH
            cat > "${_WC}/sys/param.h" <<'PARAMH'
#ifndef SF_WIN_SYS_PARAM_H
#define SF_WIN_SYS_PARAM_H
#include <limits.h>
#ifndef MAXPATHLEN
#define MAXPATHLEN 4096
#endif
#endif
PARAMH
            # Force-included compat header for POSIX functions MinGW spells
            # differently: mkdir() takes a mode arg on POSIX but only a path on
            # MinGW (_mkdir), and the S_I*USR permission bits may be absent.
            cat > "${_WC}/pf_win_compat.h" <<'PFWIN'
#ifndef PF_WIN_COMPAT_H
#define PF_WIN_COMPAT_H
#ifdef _WIN32
#include <direct.h>
#include <sys/stat.h>
#undef mkdir
#define mkdir(path, mode) _mkdir(path)
#ifndef S_IRUSR
#define S_IRUSR 0
#endif
#ifndef S_IWUSR
#define S_IWUSR 0
#endif
#ifndef S_IXUSR
#define S_IXUSR 0
#endif
#include <stdlib.h>
/* sys/param.h provides MIN/MAX on POSIX; MinGW's does not. */
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
/* drand48/srand48 are POSIX (used by ParFlow for random fields); MinGW lacks
   them. A rand()-backed stub is sufficient for the build and basic runs. */
static __inline__ double drand48(void) { return (double)rand() / ((double)RAND_MAX + 1.0); }
static __inline__ void srand48(long _s) { srand((unsigned)_s); }
#endif
#endif
PFWIN
            # mingw gcc is a native Windows program and cannot resolve the MSYS
            # "/tmp/..." path (it reads it as C:\tmp\...), so -I/-include with the
            # raw path fail ("No such file") — and the bad -include even broke
            # CMake's compiler-detection test. Convert to a Windows path.
            _WCW=$(cygpath -m "${_WC}" 2>/dev/null || echo "${_WC}")
            WIN_COMPAT_FLAG="-I${_WCW} -include ${_WCW}/pf_win_compat.h"
            echo "Windows: added sys/times.h + sys/param.h + mkdir/stat compat at ${_WCW}"
            ;;
    esac

    echo "Configuring CMake build..."
    # ParFlow source has several C99/C11 compliance issues that modern
    # Clang (16+) treats as hard errors:
    #   - int-to-pointer conversions in AMPS seq layer tests (test6.c)
    #   - implicit function declarations (seepage.c missing <string.h>)
    # We suppress these as warnings to allow the build to succeed.
    cmake -S "${BUILD_TMPDIR}/parflow_src" \
          -B "${BUILD_TMPDIR}/parflow_src/build" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DPARFLOW_AMPS_LAYER=seq \
        -DPARFLOW_ENABLE_TIMING=FALSE \
        -DPARFLOW_HAVE_CLM=OFF \
        -DCMAKE_C_FLAGS="${WIN_COMPAT_FLAG} -Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types"

    echo "Compiling..."
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "${BUILD_TMPDIR}/parflow_src/build" -j "${NCPU}"

    echo "Installing to ${INSTALL_DIR}..."
    cmake --install "${BUILD_TMPDIR}/parflow_src/build"

    # Cleanup build artifacts
    rm -rf "${BUILD_TMPDIR}"
    echo "Build artifacts cleaned up"
fi

# === Install Python dependencies ===
echo ""
echo "Installing Python dependencies (pftools, parflowio)..."
pip install pftools parflowio 2>/dev/null || \
    echo "WARNING: Could not install pftools/parflowio (optional, manual reader available)"

# === Verify installation ===
if [ -f "${INSTALL_DIR}/bin/parflow" ]; then
    echo ""
    echo "=== ParFlow Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/parflow"
    "${INSTALL_DIR}/bin/parflow" -v 2>/dev/null || echo "(version check not supported)"
else
    echo "ERROR: parflow not found at ${INSTALL_DIR}/bin/parflow"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'make'],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/parflow', 'bin/parflow.exe'],
            'check_type': 'exists_any'
        },
        'order': 26,  # After MODFLOW (25)
        'optional': True,  # Not installed by default with --install
    }
