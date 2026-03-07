# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA build instructions for SYMFLUENCE.

This module defines how to build SUMMA from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria
- Dependencies (requires SUNDIALS)

SUMMA (Structure for Unifying Multiple Modeling Alternatives) is a
land surface model that uses SUNDIALS for solving differential equations.
"""

import sys

from symfluence.cli.services import BuildInstructionsRegistry, get_common_build_environment


@BuildInstructionsRegistry.register('summa')
def get_summa_build_instructions():
    """
    Get SUMMA build instructions.

    SUMMA requires SUNDIALS to be installed first. The build uses CMake
    and links against NetCDF and LAPACK.

    Returns:
        Dictionary with complete build configuration for SUMMA.
    """
    common_env = get_common_build_environment()
    _libext = 'dylib' if sys.platform == 'darwin' else 'so'

    return {
        'description': 'Structure for Unifying Multiple Modeling Alternatives (with SUNDIALS)',
        'config_path_key': 'SUMMA_INSTALL_PATH',
        'config_exe_key': 'SUMMA_EXE',
        'default_path_suffix': 'installs/summa/bin',
        'default_exe': 'summa_sundials.exe',
        'repository': 'https://github.com/CH-Earth/summa.git',
        'branch': 'develop_sundials',
        'install_dir': 'summa',
        'requires': ['sundials'],
        'build_commands': [
            common_env,
            r'''
# Build SUMMA against SUNDIALS + NetCDF, leverage SUMMA's CMake-based build
set -e

# SUMMA is a serial Fortran program — do NOT use MPI compiler wrappers
# (mpicc/mpiCC) for CC/CXX. They link against libmpi_cxx which may require
# a newer libstdc++ than the system linker provides, causing:
#   "undefined reference to std::ios_base_library_init()@GLIBCXX_3.4.32"
# Instead, resolve to the underlying gcc/g++/gfortran from the same toolchain.
_resolve_non_mpi_compiler() {
    local wrapper="$1" fallback="$2"
    # If it's an MPI wrapper, extract the underlying compiler
    if command -v "$wrapper" >/dev/null 2>&1; then
        case "$(basename "$wrapper")" in
            mpicc|mpicxx|mpiCC|mpic++)
                # Try to extract underlying compiler from MPI wrapper
                local underlying
                underlying=$("$wrapper" -show 2>/dev/null | awk '{print $1}') || true
                if [ -n "$underlying" ] && command -v "$underlying" >/dev/null 2>&1; then
                    echo "$underlying"
                    return
                fi
                ;;
        esac
    fi
    echo "$fallback"
}

# Ensure all compilers are from the same GCC toolchain
if [ -n "${FC:-}" ] && [[ "$FC" != */usr/bin/* ]]; then
    # FC is from a module — derive CC/CXX from the same prefix
    _fc_dir="$(dirname "$FC")"
    if [ -x "$_fc_dir/gcc" ]; then
        export CC="$_fc_dir/gcc"
        export CXX="$_fc_dir/g++"
        echo "Using matched compiler toolchain from FC: CC=$CC"
    else
        export CC="$(_resolve_non_mpi_compiler "${CC:-gcc}" "gcc")"
        export CXX="$(_resolve_non_mpi_compiler "${CXX:-g++}" "g++")"
    fi
elif echo "${CC:-}" | grep -qE 'mpicc|mpicxx|mpiCC'; then
    export CC="$(_resolve_non_mpi_compiler "$CC" "gcc")"
    export CXX="$(_resolve_non_mpi_compiler "${CXX:-g++}" "g++")"
    echo "Resolved MPI wrappers to: CC=$CC CXX=$CXX"
fi

export SUNDIALS_DIR="$(realpath ../sundials/install/sundials)"
echo "Using SUNDIALS from: $SUNDIALS_DIR"

# Ensure NetCDF paths are set correctly for CMake
# On Windows conda, libraries live under CONDA_PREFIX/Library (not CONDA_PREFIX).
CONDA_LIB_PREFIX="${CONDA_PREFIX}"
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if [ -d "${CONDA_PREFIX}/Library/lib" ]; then
            CONDA_LIB_PREFIX="${CONDA_PREFIX}/Library"
        fi
        ;;
esac

if [ -n "$CONDA_PREFIX" ]; then
    export CMAKE_PREFIX_PATH="${CONDA_LIB_PREFIX}:${CMAKE_PREFIX_PATH:-}"
    export NETCDF="${NETCDF:-$CONDA_LIB_PREFIX}"
    export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$CONDA_LIB_PREFIX}"
    echo "Using conda NetCDF at: $NETCDF"
fi

# Validate NetCDF installation
if [ ! -f "${NETCDF}/include/netcdf.h" ] && [ ! -f "${NETCDF}/include/netcdf.inc" ]; then
    echo "WARNING: NetCDF headers not found at ${NETCDF}/include"
    echo "Available:"
    ls -la "${NETCDF}/include"/netcdf* 2>/dev/null | head -10 || true
fi

# Determine LAPACK strategy based on platform
SPECIFY_LINKS=OFF

case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        # Windows/MinGW: link directly against conda DLLs by full path.
        # MinGW's linker can link against DLLs directly.
        # NOTE: CMake uses semicolons as list separators — spaces would be
        # treated as part of a single filename and break the build.
        if [ -f "${CONDA_LIB_PREFIX}/bin/openblas.dll" ]; then
            echo "Using OpenBLAS DLL directly (Windows/MinGW)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="${CONDA_LIB_PREFIX}/bin/openblas.dll"
        elif [ -f "${CONDA_LIB_PREFIX}/bin/liblapack.dll" ]; then
            echo "Using manual LAPACK specification (Windows/MinGW)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="${CONDA_LIB_PREFIX}/bin/liblapack.dll;${CONDA_LIB_PREFIX}/bin/libblas.dll"
        else
            echo "Using manual LAPACK specification (Windows fallback)"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="-llapack;-lblas"
        fi
        ;;
    Darwin)
        echo "macOS detected - using manual LAPACK specification"
        SPECIFY_LINKS=ON
        export LIBRARY_LINKS='-llapack'
        ;;
    *)
        # HPC with OpenBLAS module loaded
        if command -v module >/dev/null 2>&1 && module list 2>&1 | grep -qi openblas; then
            echo "OpenBLAS module loaded - using auto-detection"
            SPECIFY_LINKS=OFF
        # Conda environment with OpenBLAS
        elif [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_LIB_PREFIX}/lib/libopenblas.so" -o -f "${CONDA_LIB_PREFIX}/lib/libopenblas.dylib" ]; then
            echo "Conda OpenBLAS found at ${CONDA_LIB_PREFIX}/lib - adding to cmake search path"
            SPECIFY_LINKS=OFF
            export CMAKE_PREFIX_PATH="${CONDA_LIB_PREFIX}:${CMAKE_PREFIX_PATH:-}"
            export LIBRARY_PATH="${CONDA_LIB_PREFIX}/lib:${LIBRARY_PATH:-}"
            export OPENBLAS_ROOT="$CONDA_LIB_PREFIX"
            export OpenBLAS_HOME="$CONDA_LIB_PREFIX"
        # Linux with system OpenBLAS
        elif pkg-config --exists openblas 2>/dev/null || [ -f "/usr/lib64/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ]; then
            echo "System OpenBLAS found - using auto-detection"
            SPECIFY_LINKS=OFF
        else
            # Fallback to manual LAPACK
            echo "Using manual LAPACK specification"
            SPECIFY_LINKS=ON
            export LIBRARY_LINKS="-llapack;-lblas"
        fi
        ;;
esac

# Patch SUMMA source for FIDASetMaxNumSteps integer kind.
# The Fortran interface takes integer(C_LONG) for mxsteps:
#   - Linux LP64:   C_LONG = 8 bytes -> need int(max_steps, kind=8)
#   - Windows LLP64: C_LONG = 4 bytes -> upstream int(max_steps) is correct
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        # Windows: C_LONG=4; undo kind=8 if present from a previous build
        if grep -q 'int(max_steps, kind=8)' build/source/engine/summaSolve4ida.f90 2>/dev/null; then
            echo "Patching summaSolve4ida.f90: removing kind=8 (Windows C_LONG=4)"
            sed -i 's/int(max_steps, kind=8)/int(max_steps)/g' build/source/engine/summaSolve4ida.f90
        fi
        ;;
    *)
        # Linux/macOS: C_LONG=8; add kind=8 if not already present
        if grep -q 'int(max_steps)' build/source/engine/summaSolve4ida.f90 2>/dev/null && \
           ! grep -q 'int(max_steps, kind=8)' build/source/engine/summaSolve4ida.f90 2>/dev/null; then
            echo "Patching summaSolve4ida.f90: int(max_steps) -> int(max_steps, kind=8)"
            _sed_i 's/int(max_steps)/int(max_steps, kind=8)/g' build/source/engine/summaSolve4ida.f90
        fi
        ;;
esac

# Patch uninitialized total_soil_depth in soilLiqFlx.f90.
# The develop_sundials branch declares this variable but never assigns it.
# On macOS/ARM the stack happens to contain a useful value; on Linux/gfortran
# the stack is zeroed, causing 0/0 = NaN in the Green-Ampt infiltration formula.
# Fix: compute total_soil_depth = sum(mLayerDepth) before first use.
if grep -q 'total_soil_depth' build/source/engine/soilLiqFlx.f90 2>/dev/null; then
    if ! grep -q 'total_soil_depth = ' build/source/engine/soilLiqFlx.f90 2>/dev/null; then
        echo "Patching soilLiqFlx.f90: initializing total_soil_depth"
        awk '/depthWettingFront = \(rootZoneLiq\/availCapacity\)/{print "   total_soil_depth = sum(in_surfaceFlx % mLayerDepth)"}1' \
            build/source/engine/soilLiqFlx.f90 > build/source/engine/soilLiqFlx.f90.tmp \
            && mv build/source/engine/soilLiqFlx.f90.tmp build/source/engine/soilLiqFlx.f90
    fi
fi

# On Windows, SUNDIALS is built static-only (DLLs don't export Fortran
# module symbols). Patch CMakeLists.txt to use static targets.
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if grep -q 'fida_mod_shared' build/CMakeLists.txt 2>/dev/null; then
            echo "Patching CMakeLists.txt: using static SUNDIALS targets (Windows)"
            sed -i 's/fida_mod_shared/fida_mod_static/g; s/fkinsol_mod_shared/fkinsol_mod_static/g' build/CMakeLists.txt
        fi
        # Build SUMMA library as STATIC on Windows (DLLs don't export
        # Fortran module symbols, so the exe can't link against the DLL).
        if grep -q 'add_library(summa SHARED' build/CMakeLists.txt 2>/dev/null; then
            echo "Patching CMakeLists.txt: SUMMA library SHARED -> STATIC (Windows)"
            sed -i 's/add_library(summa SHARED/add_library(summa STATIC/' build/CMakeLists.txt
        fi
        ;;
esac

rm -rf cmake_build && mkdir -p cmake_build

# Build CMAKE_PREFIX_PATH with all relevant paths
SUMMA_PREFIX_PATH="$SUNDIALS_DIR"
SUMMA_EXTRA_CMAKE=""
if [ -n "${CONDA_PREFIX:-}" ]; then
    SUMMA_PREFIX_PATH="${CONDA_LIB_PREFIX};${SUMMA_PREFIX_PATH}"
    # Pass OpenBLAS hints for SUMMA's FindOpenBLAS.cmake
    if [ -n "${OPENBLAS_ROOT:-}" ]; then
        SUMMA_EXTRA_CMAKE="-DOPENBLAS_ROOT=${OPENBLAS_ROOT} -DOpenBLAS_HOME=${OPENBLAS_ROOT}"
    fi
fi

# Collect library paths for RPATH embedding.
# On HPC (no conda), libraries come from module-provided paths that won't
# be on LD_LIBRARY_PATH at runtime. Embedding RPATH in the binary means
# it finds its libraries without any LD_LIBRARY_PATH at runtime.
SUMMA_INSTALL_LIB="$(pwd)/lib"
mkdir -p "$SUMMA_INSTALL_LIB"

SUMMA_RPATH_DIRS=""
_add_rpath() {
    local d="$1"
    if [ -d "$d" ] && echo ":${SUMMA_RPATH_DIRS}:" | grep -qv ":${d}:"; then
        SUMMA_RPATH_DIRS="${SUMMA_RPATH_DIRS:+${SUMMA_RPATH_DIRS};}${d}"
    fi
}
# SUMMA's own lib directory (libsumma lives here after install)
_add_rpath "$SUMMA_INSTALL_LIB"
# SUNDIALS
_add_rpath "$SUNDIALS_DIR/lib"
_add_rpath "$SUNDIALS_DIR/lib64"
# NetCDF
for _nc_root in "${NETCDF:-}" "${NETCDF_FORTRAN:-}"; do
    [ -n "$_nc_root" ] && _add_rpath "$_nc_root/lib" && _add_rpath "$_nc_root/lib64"
done
# HDF5
[ -n "${HDF5_ROOT:-}" ] && _add_rpath "$HDF5_ROOT/lib" && _add_rpath "$HDF5_ROOT/lib64"
# Conda
[ -n "${CONDA_PREFIX:-}" ] && _add_rpath "${CONDA_LIB_PREFIX}/lib"
# LD_LIBRARY_PATH entries (catches HPC module paths)
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    IFS=':' read -ra _ldp <<< "$LD_LIBRARY_PATH"
    for _d in "${_ldp[@]}"; do
        [ -n "$_d" ] && _add_rpath "$_d"
    done
fi
if [ -n "$SUMMA_RPATH_DIRS" ]; then
    SUMMA_EXTRA_CMAKE="$SUMMA_EXTRA_CMAKE -DCMAKE_INSTALL_RPATH=$SUMMA_RPATH_DIRS -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
    echo "RPATH: $SUMMA_RPATH_DIRS"
fi

# On x86-64 Linux with gfortran, denormalized floats can propagate to NaN
# in SUMMA's Jacobian. Build a tiny shared library (libftz.so) that sets
# the FTZ and DAZ bits in the x86 MXCSR register at load time, matching
# the default behavior of ARM FPUs (macOS) and Intel Fortran (HPCs).
# The SUMMA runner LD_PRELOADs this library at runtime.
case "$(uname -m 2>/dev/null)" in
    x86_64|amd64)
        echo "x86-64 detected: building libftz.so for denormal flush-to-zero"
        mkdir -p bin
        cat > bin/ftz_daz.c << 'CEOF'
#include <xmmintrin.h>
#include <pmmintrin.h>
__attribute__((constructor))
static void enable_ftz_daz(void) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
}
CEOF
        gcc -shared -fPIC -o bin/libftz.so bin/ftz_daz.c -msse2
        rm -f bin/ftz_daz.c
        echo "Built bin/libftz.so"
        ;;
esac

# If LDFLAGS contains -static-libgcc (set by fix_libgcc_glibc_mismatch),
# pass it to CMake so Fortran link tests succeed.
_SUMMA_LINKER_FLAGS=""
if echo "${LDFLAGS:-}" | grep -q static-libgcc; then
    _SUMMA_LINKER_FLAGS="-DCMAKE_EXE_LINKER_FLAGS=-static-libgcc -DCMAKE_SHARED_LINKER_FLAGS=-static-libgcc"
fi

cmake -S build -B cmake_build \
  -DUSE_SUNDIALS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_FLAGS_RELEASE="-O2 -DNDEBUG -finit-real=zero -finit-integer=0 -finit-logical=false" \
  -DSPECIFY_LAPACK_LINKS=$SPECIFY_LINKS \
  -DCMAKE_PREFIX_PATH="${SUMMA_PREFIX_PATH}" \
  -DSUNDIALS_ROOT="$SUNDIALS_DIR" \
  -DNETCDF_PATH="${NETCDF:-/usr}" \
  -DNETCDF_FORTRAN_PATH="${NETCDF_FORTRAN:-/usr}" \
  -DNetCDF_ROOT="${NETCDF:-/usr}" \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_Fortran_FLAGS="-ffree-form -ffree-line-length-none" \
  $_SUMMA_LINKER_FLAGS \
  $SUMMA_EXTRA_CMAKE

# Build all targets (repo scripts use 'all', not just 'summa_sundials')
cmake --build cmake_build --target all -j ${NCORES:-4}

# Stage libsumma into lib/ so the binary can find it via RPATH.
# On macOS CMake produces .dylib; on Linux .so.
case "$(uname -s)" in
    Darwin) _libext="dylib" ;;
    *)      _libext="so" ;;
esac
_libname="libsumma.$_libext"

# CMake may place it in cmake_build/lib/, cmake_build/, or lib/.
for libcandidate in \
    "cmake_build/lib/$_libname" \
    "cmake_build/$_libname" \
    "lib/$_libname"; do
    if [ -f "$libcandidate" ]; then
        if [ "$libcandidate" != "lib/$_libname" ]; then
            cp -f "$libcandidate" "lib/$_libname"
            echo "Staged: $libcandidate -> lib/$_libname"
        else
            echo "$_libname already at lib/$_libname"
        fi
        break
    fi
done
if [ ! -f "lib/$_libname" ]; then
    echo "WARNING: $_libname not found in build output, searching..."
    _found_lib=$(find cmake_build -name "libsumma.$_libext*" -type f 2>/dev/null | head -1)
    if [ -n "$_found_lib" ]; then
        cp -f "$_found_lib" "lib/$_libname"
        echo "Staged: $_found_lib -> lib/$_libname"
    else
        echo "ERROR: $_libname not found anywhere in build tree"
    fi
fi

# Stage binary into bin/ and provide standard name.
# On Windows, cmake appends .exe to the target name "summa_sundials.exe",
# producing "summa_sundials.exe.exe".  Normalise to summa_sundials.exe.
mkdir -p bin
for candidate in \
    bin/summa_sundials.exe \
    bin/summa_sundials.exe.exe \
    cmake_build/bin/summa_sundials.exe \
    cmake_build/bin/summa_sundials.exe.exe \
    cmake_build/bin/summa.exe; do
    if [ -f "$candidate" ]; then
        # Avoid error when source and destination are the same file
        # (happens when CMake outputs directly to bin/)
        if [ "$candidate" != "bin/summa_sundials.exe" ]; then
            cp -f "$candidate" bin/summa_sundials.exe
            echo "Staged: $candidate -> bin/summa_sundials.exe"
        else
            echo "Binary already at bin/summa_sundials.exe"
        fi
        cd bin
        ln -sf summa_sundials.exe summa.exe
        cd ..
        break
    fi
done
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--version',
        'verify_install': {
            'file_paths': [
                'bin/summa_sundials.exe',
                f'lib/libsumma.{_libext}',
            ],
            'check_type': 'exists'
        },
        'order': 2
    }
