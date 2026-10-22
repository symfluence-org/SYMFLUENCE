# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLMParFlow build instructions for SYMFLUENCE.

Tier 1 (default): Download pre-compiled binary from ParFlow GitHub releases.
Tier 2 (fallback): Build from source using CMake with -DPARFLOW_HAVE_CLM=ON.

ParFlow-CLM is ParFlow compiled with the tightly-coupled Common Land Model.
The key difference from standalone ParFlow is the CMake flag:
  -DPARFLOW_HAVE_CLM=ON  (instead of OFF)

``symfluence binary install clmparflow --patched`` instead builds ParFlow v3.15.0 from source with the SYMFLUENCE
CLM patches (``patches/README.md``: land-ice water, restart read, runtime ice albedo) and hypre (PFMG
preconditioner), with the ``mpi1`` AMPS layer when ``mpicc`` is available. Basin mode needs this build.
"""
from __future__ import annotations

from pathlib import Path

from symfluence.core.registries import R

# ParFlow release the CLM patches were written and tested against; --patched builds check it out.
PATCH_BASE_TAG = 'v3.15.0'
HYPRE_TAG = 'v2.31.0'
PATCHES = [Path(__file__).with_name('patches') / name for name in (
    'parflow3150_clm_glacier_water.patch',
    'parflow3150_clm_restart_read.patch',
    'parflow3150_clm_ice_albedo.patch',
)]


def _patched_build_command() -> str:
    """Shell step building the patched ParFlow-CLM when --patched is set; a no-op otherwise."""
    # one command per line, so set -e stops the build on any patch that does not apply
    applies = '\n'.join(f'    git apply --check "{p.as_posix()}"\n    git apply "{p.as_posix()}"\n'
                        f'    cp "{p.as_posix()}" bin/applied_{p.name}' for p in PATCHES)
    return f'''set -e
if [ "${{SYMFLUENCE_PATCHED:-0}}" = "1" ]; then
    echo "=== Patched ParFlow-CLM {PATCH_BASE_TAG} (SYMFLUENCE CLM patches) ==="
    INSTALL_DIR="$(pwd)"
    if ! git rev-parse -q --verify "refs/tags/{PATCH_BASE_TAG}" >/dev/null; then
        git fetch --depth 1 origin tag "{PATCH_BASE_TAG}"
    fi
    git checkout -q "{PATCH_BASE_TAG}"
    mkdir -p bin
{applies}
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    if command -v mpicc >/dev/null 2>&1 && command -v mpif90 >/dev/null 2>&1; then
        AMPS=mpi1; PF_CC=mpicc; PF_FC=mpif90; HYPRE_MPI=""
    else
        echo "WARNING: mpicc/mpif90 not found; building the serial (seq) layer, basin mode then runs on one rank"
        AMPS=seq; PF_CC=cc; PF_FC=gfortran; HYPRE_MPI="--without-MPI"
    fi
    HYPRE_SRC=$(mktemp -d /tmp/hypre_src_XXXXXX)
    git clone -q --depth 1 --branch "{HYPRE_TAG}" https://github.com/hypre-space/hypre.git "$HYPRE_SRC"
    (cd "$HYPRE_SRC/src" && CC=$PF_CC ./configure --prefix="${{INSTALL_DIR}}/hypre" --disable-fortran $HYPRE_MPI -q \\
        && make -j "$NCPU" >/dev/null && make install >/dev/null)
    rm -rf "$HYPRE_SRC"
    cmake -S . -B build -DCMAKE_INSTALL_PREFIX="${{INSTALL_DIR}}" -DPARFLOW_AMPS_LAYER=$AMPS \\
        -DPARFLOW_HAVE_CLM=ON -DHYPRE_ROOT="${{INSTALL_DIR}}/hypre" -DCMAKE_BUILD_TYPE=Release \\
        -DCMAKE_C_COMPILER=$PF_CC -DCMAKE_Fortran_COMPILER=$PF_FC \\
        -DCMAKE_C_FLAGS="-Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types" \\
        -DCMAKE_Fortran_FLAGS="-w -ffree-line-length-none"
    cmake --build build -j "$NCPU"
    cmake --install build
    touch bin/.symfluence_patched
fi
'''


@R.build_instructions.add('clmparflow')
def get_clmparflow_build_instructions():
    """
    Get CLMParFlow build/install instructions.

    Downloads pre-compiled ParFlow binary from GitHub releases (may not include CLM).
    Falls back to CMake source build with CLM enabled if download fails or
    CLM is not included in the binary.

    Returns:
        Dictionary with complete build configuration for CLMParFlow.
    """
    return {
        'description': 'ParFlow-CLM (ParFlow with tightly-coupled Common Land Model)',
        'config_path_key': 'CLMPARFLOW_INSTALL_PATH',
        'config_exe_key': 'CLMPARFLOW_EXE',
        'default_path_suffix': 'installs/clmparflow/bin',
        'default_exe': 'parflow',
        'repository': 'https://github.com/parflow/parflow.git',
        'branch': 'master',
        'install_dir': 'clmparflow',
        'build_commands': [
            _patched_build_command(),
            r'''
# CLMParFlow Install Script for SYMFLUENCE
# ParFlow built with -DPARFLOW_HAVE_CLM=ON for tightly-coupled CLM
# Tier 1: Download pre-compiled binary from GitHub releases
# Tier 2: Build from source with CMake + CLM (fallback)

set -e

if [ -f bin/.symfluence_patched ]; then
    echo "Patched ParFlow-CLM built; skipping the binary download / unpatched source build"
    exit 0
fi

echo "=== CLMParFlow Installation Starting ==="
echo "Building ParFlow with tightly-coupled CLM (Common Land Model)"

# Resolve INSTALL_DIR to an absolute path BEFORE any cd commands.
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
# Note: Pre-compiled binaries may not include CLM, so we verify
DOWNLOAD_SUCCESS=false

if [ "$PLATFORM" != "unknown" ]; then
    echo "Attempting binary download from ParFlow GitHub releases..."

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

    DOWNLOAD_URL="https://github.com/parflow/parflow/releases/download/${LATEST_TAG}/parflow-${LATEST_TAG}-${PLATFORM}.tar.gz"
    echo "Download URL: $DOWNLOAD_URL"

    TMPTAR=$(mktemp /tmp/clmparflow_XXXXXX.tar.gz)
    TMPEXTRACT=$(mktemp -d /tmp/clmparflow_extract_XXXXXX)

    if command -v curl >/dev/null 2>&1; then
        curl -fsSL -o "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ ! -s "$TMPTAR" ] && command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMPTAR" "$DOWNLOAD_URL" 2>/dev/null || true
    fi
    if [ -s "$TMPTAR" ]; then
        if file "$TMPTAR" | grep -q "gzip\|tar"; then
            tar xzf "$TMPTAR" -C "$TMPEXTRACT" 2>/dev/null || true
            PF_BIN=$(find "$TMPEXTRACT" -name "parflow" -type f 2>/dev/null | head -1)

            if [ -n "$PF_BIN" ]; then
                cp "$PF_BIN" "${INSTALL_DIR}/bin/parflow"
                chmod +x "${INSTALL_DIR}/bin/parflow"

                if "${INSTALL_DIR}/bin/parflow" --version >/dev/null 2>&1 || \
                   "${INSTALL_DIR}/bin/parflow" -v >/dev/null 2>&1; then
                    DOWNLOAD_SUCCESS=true
                    echo "Binary download successful"
                    echo "WARNING: Pre-compiled binary may not include CLM support."
                    echo "  If CLM features are needed, the source build (Tier 2) is recommended."
                else
                    echo "WARNING: Downloaded parflow binary cannot run on this system"
                    rm -f "${INSTALL_DIR}/bin/parflow"
                fi
            fi
        fi
    fi

    rm -f "$TMPTAR"
    rm -rf "$TMPEXTRACT"
fi

# === Tier 2: Build from source with CMake + CLM ===
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    echo ""
    echo "Building ParFlow from source with CLM support..."

    for tool in cmake make gfortran; do
        if ! command -v $tool >/dev/null 2>&1; then
            echo "ERROR: $tool not found. Install with: brew install cmake gcc (or apt install cmake gfortran)"
            exit 1
        fi
    done

    BUILD_TMPDIR=$(mktemp -d /tmp/clmparflow_build_XXXXXX)
    echo "Build directory: ${BUILD_TMPDIR}"

    echo "Cloning ParFlow source..."
    git clone --depth 1 https://github.com/parflow/parflow.git "${BUILD_TMPDIR}/parflow_src"

    mkdir -p "${BUILD_TMPDIR}/parflow_src/build"

    # Windows (MSYS2/MinGW): ParFlow's AMPS seq layer needs POSIX shims MinGW
    # lacks (sys/times.h, sys/param.h's MIN/MAX, mkdir(path,mode), drand48). Same
    # shims as the standalone parflow build; CLM=ON adds Fortran that compiles
    # fine. Paths go through cygpath -m so native mingw gcc can read them.
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
            cat > "${_WC}/pf_win_compat.h" <<'PFWIN'
#ifndef PF_WIN_COMPAT_H
#define PF_WIN_COMPAT_H
#ifdef _WIN32
#include <direct.h>
#include <sys/stat.h>
#include <stdlib.h>
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
#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif
static __inline__ double drand48(void) { return (double)rand() / ((double)RAND_MAX + 1.0); }
static __inline__ void srand48(long _s) { srand((unsigned)_s); }
#endif
#endif
PFWIN
            _WCW=$(cygpath -m "${_WC}" 2>/dev/null || echo "${_WC}")
            WIN_COMPAT_FLAG="-I${_WCW} -include ${_WCW}/pf_win_compat.h "
            echo "Windows: added ParFlow POSIX compat shims at ${_WCW}"
            ;;
    esac

    echo "Configuring CMake build with CLM=ON..."
    cmake -S "${BUILD_TMPDIR}/parflow_src" \
          -B "${BUILD_TMPDIR}/parflow_src/build" \
        -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
        -DPARFLOW_AMPS_LAYER=seq \
        -DPARFLOW_ENABLE_TIMING=FALSE \
        -DPARFLOW_HAVE_CLM=ON \
        -DCMAKE_C_FLAGS="${WIN_COMPAT_FLAG}-Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types" \
        -DCMAKE_Fortran_FLAGS="-w"

    echo "Compiling (with CLM Fortran modules)..."
    NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cmake --build "${BUILD_TMPDIR}/parflow_src/build" -j "${NCPU}"

    echo "Installing to ${INSTALL_DIR}..."
    cmake --install "${BUILD_TMPDIR}/parflow_src/build"

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
    echo "=== CLMParFlow Installation Complete ==="
    echo "Installed to: ${INSTALL_DIR}/bin/parflow"
    "${INSTALL_DIR}/bin/parflow" -v 2>/dev/null || echo "(version check not supported)"
else
    echo "ERROR: parflow not found at ${INSTALL_DIR}/bin/parflow"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['cmake', 'make', 'gfortran'],
        'test_command': '--version',
        'verify_install': {
            'file_paths': ['bin/parflow', 'bin/parflow.exe'],
            'check_type': 'exists_any'
        },
        'order': 27,  # After ParFlow (26)
        'optional': True,
    }
