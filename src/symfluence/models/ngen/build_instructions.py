# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGEN build instructions for SYMFLUENCE.

This module defines how to build NGEN from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

NGEN is the NextGen National Water Model Framework developed by NOAA/NWS.
It supports multiple BMI-compliant model modules including CFE, PET,
NOAH-OWP-Modular, and SLOTH.
"""

from symfluence.cli.services import (
    BuildInstructionsRegistry,
    get_common_build_environment,
    get_netcdf_detection,
    get_udunits2_detection_and_build,
)


@BuildInstructionsRegistry.register('ngen')
def get_ngen_build_instructions():
    """
    Get NGEN build instructions.

    NGEN uses CMake and requires Boost, NetCDF, and optionally Python
    and Fortran support for various BMI modules.

    Returns:
        Dictionary with complete build configuration for NGEN.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()
    udunits2_detect = get_udunits2_detection_and_build()

    return {
        'description': 'NextGen National Water Model Framework',
        'config_path_key': 'NGEN_INSTALL_PATH',
        'config_exe_key': 'NGEN_EXE',
        'default_path_suffix': 'installs/ngen/cmake_build',
        'default_exe': 'ngen',
        'repository': 'https://github.com/CIROH-UA/ngen',
        'branch': 'ngiab',
        'install_dir': 'ngen',
        'build_commands': [
            common_env,
            netcdf_detect,
            udunits2_detect,
            r'''
set -e
set -o pipefail  # Make pipelines return exit code of failed command, not just last command
echo "Building ngen with full BMI support (C, C++, Fortran)..."

# Detect venv Python BEFORE modifying PATH (setup-python Python must be found first)
if [ -n "$VIRTUAL_ENV" ]; then
  PYTHON_EXE="$VIRTUAL_ENV/bin/python3"
elif [ -n "$CONDA_PREFIX" ]; then
  # On Windows conda, python3 might be at Scripts/python or just python
  if [ -x "$CONDA_PREFIX/bin/python3" ]; then
    PYTHON_EXE="$CONDA_PREFIX/bin/python3"
  elif [ -x "$CONDA_PREFIX/python.exe" ]; then
    PYTHON_EXE="$CONDA_PREFIX/python.exe"
  else
    PYTHON_EXE=$(which python3 || which python)
  fi
else
  PYTHON_EXE=$(which python3 || which python)
fi
echo "Using Python: $PYTHON_EXE"
NUMPY_VERSION=$($PYTHON_EXE -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
if [ -n "$NUMPY_VERSION" ]; then
  echo "Using NumPy: $NUMPY_VERSION"
else
  echo "WARNING: numpy not available in $PYTHON_EXE"
fi

# Ensure system tools are preferred (fix for 2i2c environments)
export PATH="/usr/bin:$PATH"

# Prevent any Makefile from being auto-triggered during git operations
# Debug: show what MAKEFLAGS contains
echo "DEBUG: MAKEFLAGS before clearing: '${MAKEFLAGS:-}'"
echo "DEBUG: MAKELEVEL before clearing: '${MAKELEVEL:-}'"

# Must unset ALL make-related variables to prevent spurious make calls
unset MAKEFLAGS MAKELEVEL MAKE MFLAGS MAKEOVERRIDES GNUMAKEFLAGS 2>/dev/null || true
export MAKEFLAGS=""
export MAKELEVEL=""
export MFLAGS=""

# Disable git hooks that might trigger make
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null

# Fix for conda GCC 14: ensure libstdc++ is found
# GCC 14 from conda-forge requires explicit library path for C++ runtime
clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/lib" ]; then
    export LIBRARY_PATH="$clp/lib:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="$clp/lib:${LD_LIBRARY_PATH:-}"
    echo "Added conda lib to library paths: $clp/lib"
fi

# On Windows/MSYS2, ensure CMake uses MSYS Makefiles generator
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        export CMAKE_GENERATOR="MSYS Makefiles"
        echo "Windows detected: using MSYS Makefiles generator"
        ;;
esac

# Boost (local)
if [ ! -d "boost_1_79_0" ]; then
  echo "Fetching Boost 1.79.0..."
  (wget -q https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2 -O boost_1_79_0.tar.bz2 \
    || curl -fsSL -o boost_1_79_0.tar.bz2 https://downloads.sourceforge.net/project/boost/boost/1.79.0/boost_1_79_0.tar.bz2)
  tar -xjf boost_1_79_0.tar.bz2 && rm -f boost_1_79_0.tar.bz2
fi
export BOOST_ROOT="$(pwd)/boost_1_79_0"
export CXX=${CXX:-g++}

# Initialize ALL submodules needed for full BMI support
# Create a clean git wrapper to prevent MAKEFLAGS from triggering spurious make calls
# Also disable git hooks which may trigger make
git_clean() {
    MAKEFLAGS= MAKELEVEL= MAKE= MFLAGS= GNUMAKEFLAGS= git -c core.hooksPath=/dev/null "$@"
}

echo "Initializing submodules for ngen and external BMI modules..."
git_clean submodule update --init --recursive -- test/googletest extern/pybind11 || true
git_clean submodule update --init --recursive -- extern/cfe extern/evapotranspiration extern/sloth extern/noah-owp-modular || true

# Initialize t-route submodule for routing support
# Note: t-route triggers spurious make calls on some HPC systems, skip if it fails
echo "Initializing t-route submodule for routing..."
if ! git_clean submodule update --init -- extern/t-route 2>&1; then
    echo "WARNING: t-route submodule init failed, routing will be disabled"
fi
# Don't recursively init t-route submodules as they may trigger make
# git_clean submodule update --init --recursive -- extern/t-route || true

# Initialize iso_c_fortran_bmi for Fortran BMI support (required for NOAH-OWP)
echo "Initializing iso_c_fortran_bmi submodule..."
git_clean submodule update --init --recursive -- extern/iso_c_fortran_bmi || true

# Fallback: manually clone BMI modules if submodule init didn't populate them.
# The CIROH-UA/ngen ngiab branch doesn't register CFE/SLOTH/PET as submodules,
# so `git submodule update` silently skips them. Clone directly from NOAA-OWP.
_clone_if_missing() {
    local name="$1" url="$2" dir="$3"
    if [ ! -f "$dir/CMakeLists.txt" ]; then
        echo "$name not populated by submodule init — cloning from $url ..."
        rm -rf "$dir"
        mkdir -p "$(dirname "$dir")"
        git clone --depth 1 "$url" "$dir" 2>&1 || echo "WARNING: Failed to clone $name (non-fatal)"
    else
        echo "$name submodule OK"
    fi
}

_clone_if_missing "CFE"   "https://github.com/NOAA-OWP/cfe.git"               "extern/cfe"
_clone_if_missing "SLOTH" "https://github.com/NOAA-OWP/SLoTH.git"             "extern/sloth"
_clone_if_missing "PET"   "https://github.com/NOAA-OWP/evapotranspiration.git" "extern/evapotranspiration/evapotranspiration"
_clone_if_missing "Noah-MP" "https://github.com/NOAA-OWP/noah-owp-modular.git" "extern/noah-owp-modular"
_clone_if_missing "iso_c_fortran_bmi" "https://github.com/NOAA-OWP/iso_c_fortran_bmi.git" "extern/iso_c_fortran_bmi"

# Verify Fortran compiler
echo "Checking Fortran compiler..."
if command -v gfortran >/dev/null 2>&1; then
  export FC=$(command -v gfortran)
  # On Windows, CMake needs the full path with .exe extension
  case "$(uname -s 2>/dev/null)" in
      MSYS*|MINGW*|CYGWIN*)
          if [ -f "${FC}.exe" ]; then
              export FC="${FC}.exe"
          fi
          ;;
  esac
  echo "Using Fortran compiler: $FC"
  $FC --version | head -1
else
  echo "WARNING: gfortran not found, Fortran BMI modules will be disabled"
  export NGEN_WITH_BMI_FORTRAN=OFF
fi

rm -rf cmake_build

# On Windows/MSYS2, apply platform patches for building
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        echo "Patching CMakeLists.txt for Windows/MSYS2 build..."
        # Allow Windows platform (upstream blocks WIN32)
        sed -i.bak 's/message(FATAL_ERROR "Windows platforms are not currently supported")/set(NGEN_SHARED_LIB_EXTENSION "dll")/' CMakeLists.txt && rm -f CMakeLists.txt.bak
        # Fix ngen_multiline_message: backslashes in Windows paths cause CMake
        # string parse errors. Replace with a simple message() call.
        sed -i.bak '/macro(ngen_multiline_message/,/endmacro/c\macro(ngen_multiline_message)\n    message(STATUS "NGen configuration complete")\nendmacro()' CMakeLists.txt && rm -f CMakeLists.txt.bak
        # Fix FindUDUNITS2: on Windows, SHARED IMPORTED targets need IMPORTED_IMPLIB
        # The .lib file is the import library; set it as IMPLIB and use the DLL as LOCATION
        sed -i.bak 's/IMPORTED_LOCATION "${UDUNITS2_LIBRARY}"/IMPORTED_IMPLIB "${UDUNITS2_LIBRARY}"/' cmake/FindUDUNITS2.cmake && rm -f cmake/FindUDUNITS2.cmake.bak
        # Create a POSIX compat header for functions missing in MinGW (strsep, etc.)
        cat > mingw_posix_compat.h << 'COMPAT_EOF'
#ifndef MINGW_POSIX_COMPAT_H
#define MINGW_POSIX_COMPAT_H
#ifdef __MINGW32__
#include <string.h>
#include <stdlib.h>
static inline char *strsep(char **stringp, const char *delim) {
    char *start = *stringp;
    char *p;
    if (start == NULL) return NULL;
    p = strpbrk(start, delim);
    if (p) { *p = '\0'; *stringp = p + 1; }
    else { *stringp = NULL; }
    return start;
}
#endif
#endif
COMPAT_EOF
        # Force-include the compat header in all C compilations
        export CFLAGS=$(echo "${CFLAGS:-} -include $(pwd)/mingw_posix_compat.h" | sed 's/\\/\//g')
        export CXXFLAGS=$(echo "${CXXFLAGS:-}" | sed 's/\\/\//g')
        ;;
esac

# Debug: show environment
echo "=== Environment Debug ==="
echo "CONDA_PREFIX: ${CONDA_PREFIX:-not set}"
echo "UDUNITS2_DIR: ${UDUNITS2_DIR:-not set}"
echo "UDUNITS2_INCLUDE_DIR: ${UDUNITS2_INCLUDE_DIR:-not set}"
echo "UDUNITS2_LIBRARY: ${UDUNITS2_LIBRARY:-not set}"
echo "========================="

# Build ngen with full BMI support including Fortran
echo "Configuring ngen with BMI C, C++, and Fortran support..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
CMAKE_ARGS="$CMAKE_ARGS -DBOOST_ROOT=$BOOST_ROOT"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_SQLITE3=ON"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_C=ON"
CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_CPP=ON"

# Enable NetCDF forcing provider (avoids SIGSEGV bug in CsvPerFeatureForcingProvider)
if nc-config --libs >/dev/null 2>&1; then
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_NETCDF=ON"
  echo "NetCDF-C found ($(nc-config --version 2>/dev/null || echo 'unknown')). Enabling NetCDF forcing provider."
else
  echo "WARNING: NetCDF-C not found. CsvPerFeature forcing will be used (may crash on macOS ARM64)."
fi

# On Windows, disable tests (symlinks require admin privileges) and UDUNITS2
# (import library handling issues on MinGW)
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_TESTS=OFF"
        CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_UDUNITS=OFF"
        ;;
esac

# Fix for conda GCC 14: explicitly link against conda's libstdc++ to resolve __cxa_call_terminate
# conda's GCC 14 emits __cxa_call_terminate which only exists in its own libstdc++
EXTRA_LIBS=""
if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/lib" ]; then
    EXTRA_LIBS="-lstdc++"
    echo "Adding conda libstdc++ linker flags for GCC 14 compatibility"
fi

# Add UDUNITS2 paths if available (from detection/build snippet)
if [ -n "${UDUNITS2_INCLUDE_DIR:-}" ] && [ -n "${UDUNITS2_LIBRARY:-}" ]; then
  # On Windows, convert backslash paths to forward slashes for CMake
  _U2_DIR=$(echo "$UDUNITS2_DIR" | sed 's/\\/\//g')
  _U2_INC=$(echo "$UDUNITS2_INCLUDE_DIR" | sed 's/\\/\//g')
  _U2_LIB=$(echo "$UDUNITS2_LIBRARY" | sed 's/\\/\//g')
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_ROOT=$_U2_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_INCLUDE_DIR=$_U2_INC"
  CMAKE_ARGS="$CMAKE_ARGS -DUDUNITS2_LIBRARY=$_U2_LIB"
  # When UDUNITS2 is a static archive, GNU ld needs transitive dependencies
  # (expat, dl, m) AFTER the archive on the link line.  We pass these via
  # CMAKE_EXE_LINKER_FLAGS because UDUNITS2_LIBRARY must be a single path
  # (ngen's FindUDUNITS2.cmake uses it in IMPORTED_LOCATION).
  if echo "$_U2_LIB" | grep -q '\.a$'; then
    # On Linux, GNU ld is single-pass: transitive deps (-lexpat -ldl -lm) must
    # appear AFTER the udunits2 archive.  Use --start-group/--end-group to allow
    # circular resolution regardless of link order.
    if [ "$(uname -s)" = "Linux" ]; then
      EXTRA_LIBS="${EXTRA_LIBS:-} -Wl,--start-group -lexpat -ldl -lm -Wl,--end-group"
    else
      EXTRA_LIBS="${EXTRA_LIBS:-} -lexpat -ldl -lm"
    fi
    echo "Static UDUNITS2 detected — adding transitive deps to linker flags"
  fi

  # Also add to compiler flags (use forward-slash path)
  export CXXFLAGS="${CXXFLAGS:-} -I${_U2_INC}"
  export CFLAGS="${CFLAGS:-} -I${_U2_INC}"

  echo "Using UDUNITS2 from: $_U2_DIR"
fi

# Add extra linker flags for conda GCC 14 and expat (needed by UDUNITS2)
# IMPORTANT: CMake does NOT use LDFLAGS for target linking. We must pass these
# through CMAKE_EXE_LINKER_FLAGS. Build up all extra flags here.
EXTRA_LINK_FLAGS="${EXTRA_LIBS:-}"

# Detect UDUNITS2 from any source: explicit var, pkg-config, or system library
_HAVE_UDUNITS2=false
if [ -n "${UDUNITS2_LIBRARY:-}" ]; then
  _HAVE_UDUNITS2=true
elif pkg-config --exists udunits2 2>/dev/null; then
  _HAVE_UDUNITS2=true
elif ldconfig -p 2>/dev/null | grep -q libudunits2; then
  _HAVE_UDUNITS2=true
fi

if [ "$_HAVE_UDUNITS2" = "true" ] && [ "${UDUNITS2_FROM_HPC_MODULE:-false}" != "true" ]; then
  # UDUNITS2 depends on libexpat for XML parsing.  CMake finds libudunits2 but
  # may not propagate the transitive -lexpat dependency, causing undefined
  # references to XML_GetErrorCode / XML_ErrorString at link time.
  if [ -n "${EXPAT_LIB_DIR:-}" ] && [ -d "${EXPAT_LIB_DIR}" ]; then
    EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -L${EXPAT_LIB_DIR} -lexpat"
    # Add to LIBRARY_PATH so the linker can find it during build
    export LIBRARY_PATH="${EXPAT_LIB_DIR}:${LIBRARY_PATH:-}"
    export LD_LIBRARY_PATH="${EXPAT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    # Add to CMAKE_PREFIX_PATH so CMake can find it
    export CMAKE_PREFIX_PATH="${EXPAT_LIB_DIR%/lib}:${CMAKE_PREFIX_PATH:-}"
    echo "Using EXPAT from: ${EXPAT_LIB_DIR}"
  else
    # System expat: add -lexpat (standard on Linux, Homebrew on macOS)
    EXTRA_LINK_FLAGS="$EXTRA_LINK_FLAGS -lexpat"
    echo "Adding -lexpat for system UDUNITS2 (transitive dependency)"
  fi
elif [ "${UDUNITS2_FROM_HPC_MODULE:-false}" = "true" ]; then
  echo "Using HPC module UDUNITS2 - expat dependency handled via module rpath"
fi

# Merge conda linker flags and expat flags into CMAKE_EXE_LINKER_FLAGS
# (CMake ignores LDFLAGS for the final target link)
ALL_CMAKE_LINK_FLAGS=""
if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/lib" ]; then
    _clp_fwd=$(echo "$clp" | sed 's/\\/\//g')
    ALL_CMAKE_LINK_FLAGS="-L${_clp_fwd}/lib -lstdc++"
fi
if [ -n "$EXTRA_LINK_FLAGS" ]; then
    ALL_CMAKE_LINK_FLAGS="$ALL_CMAKE_LINK_FLAGS $EXTRA_LINK_FLAGS"
    echo "Adding extra linker flags: $EXTRA_LINK_FLAGS"
fi
# If LDFLAGS contains -static-libgcc (set by fix_libgcc_glibc_mismatch),
# add static linking flags so cmake compiler tests pass.
if echo "${LDFLAGS:-}" | grep -q static-libgcc; then
    ALL_CMAKE_LINK_FLAGS="$ALL_CMAKE_LINK_FLAGS -static-libgcc -static-libstdc++"
fi
# Build the unified CMake linker args
CMAKE_LINKER_ARGS=""
if [ -n "$ALL_CMAKE_LINK_FLAGS" ]; then
    CMAKE_LINKER_ARGS="-DCMAKE_EXE_LINKER_FLAGS='$ALL_CMAKE_LINK_FLAGS' -DCMAKE_SHARED_LINKER_FLAGS='$ALL_CMAKE_LINK_FLAGS'"
    echo "CMAKE_EXE_LINKER_FLAGS: $ALL_CMAKE_LINK_FLAGS"
fi

# Add Fortran support if compiler is available
if [ "${NGEN_WITH_BMI_FORTRAN:-ON}" = "ON" ] && [ -n "$FC" ]; then
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_BMI_FORTRAN=ON"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"

  # Configure iso_c_fortran_bmi (C wrapper for Fortran BMI modules)
  # This provides the register_bmi function that NGEN needs to load Fortran modules
  ISO_C_BMI_DIR="$(pwd)/extern/iso_c_fortran_bmi/cmake_build"
  CMAKE_ARGS="$CMAKE_ARGS -DBMI_FORTRAN_ISO_C_LIB_DIR=$ISO_C_BMI_DIR"
  CMAKE_ARGS="$CMAKE_ARGS -DBMI_FORTRAN_ISO_C_LIB_NAME=iso_c_bmi"

  echo "Enabling Fortran BMI support with iso_c_bmi wrapper"
fi

# Check NumPy version - ngen doesn't support NumPy 2.x yet
# NUMPY_VERSION was detected earlier (before PATH modification)
NUMPY_MAJOR=$(echo "${NUMPY_VERSION:-}" | cut -d. -f1)
if [ -z "$NUMPY_VERSION" ]; then
  echo "NumPy not available. Disabling Python and routing support."
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_PYTHON=OFF"
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_ROUTING=OFF"
elif [ "$NUMPY_MAJOR" -ge 2 ] 2>/dev/null; then
  echo "NumPy $NUMPY_VERSION detected (>=2.0). Disabling Python and routing support (not yet compatible with ngen)."
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_PYTHON=OFF"
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_ROUTING=OFF"
else
  # Add Python support for NumPy 1.x
  echo "NumPy $NUMPY_VERSION detected. Enabling Python and t-route routing support."
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_PYTHON=ON"
  CMAKE_ARGS="$CMAKE_ARGS -DPython_EXECUTABLE=$PYTHON_EXE"
  CMAKE_ARGS="$CMAKE_ARGS -DPython3_EXECUTABLE=$PYTHON_EXE"
  CMAKE_ARGS="$CMAKE_ARGS -DNGEN_WITH_ROUTING=ON"
fi

# Configure ngen
echo "Running CMake with args: $CMAKE_ARGS $CMAKE_LINKER_ARGS"
if eval cmake $CMAKE_ARGS $CMAKE_LINKER_ARGS -S . -B cmake_build 2>&1 | tee cmake_config.log; then
  echo "ngen configured successfully"
else
  echo "CMake configuration failed, checking log..."
  tail -30 cmake_config.log
  echo ""
  echo "Retrying with Python OFF but keeping Fortran support..."
  rm -rf cmake_build

  # Keep Fortran support in fallback - it's required for NOAH-OWP!
  FALLBACK_ARGS="-DCMAKE_BUILD_TYPE=Release"
  FALLBACK_ARGS="$FALLBACK_ARGS -DBOOST_ROOT=$BOOST_ROOT"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_PYTHON=OFF"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_ROUTING=OFF"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_SQLITE3=ON"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_C=ON"
  FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_CPP=ON"
  if nc-config --libs >/dev/null 2>&1; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_NETCDF=ON"
  fi
  case "$(uname -s 2>/dev/null)" in
      MSYS*|MINGW*|CYGWIN*)
          FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_TESTS=OFF"
          FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_UDUNITS=OFF"
          ;;
  esac

  # Conda linker flags are in CMAKE_LINKER_ARGS (set above)

  # Add UDUNITS2 paths to fallback as well
  if [ -n "${UDUNITS2_INCLUDE_DIR:-}" ] && [ -n "${UDUNITS2_LIBRARY:-}" ]; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_ROOT=$UDUNITS2_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_INCLUDE_DIR=$UDUNITS2_INCLUDE_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DUDUNITS2_LIBRARY=$UDUNITS2_LIBRARY"
  fi
  # Note: LIBRARY_PATH and CMAKE_PREFIX_PATH are already set in environment for expat

  # Keep Fortran in fallback if compiler is available
  if [ -n "$FC" ]; then
    FALLBACK_ARGS="$FALLBACK_ARGS -DNGEN_WITH_BMI_FORTRAN=ON"
    FALLBACK_ARGS="$FALLBACK_ARGS -DCMAKE_Fortran_COMPILER=$FC"

    # Include iso_c_bmi configuration in fallback
    ISO_C_BMI_DIR="$(pwd)/extern/iso_c_fortran_bmi/cmake_build"
    FALLBACK_ARGS="$FALLBACK_ARGS -DBMI_FORTRAN_ISO_C_LIB_DIR=$ISO_C_BMI_DIR"
    FALLBACK_ARGS="$FALLBACK_ARGS -DBMI_FORTRAN_ISO_C_LIB_NAME=iso_c_bmi"

    echo "Fallback: keeping Fortran BMI support with iso_c_bmi wrapper"
  fi

  eval cmake $FALLBACK_ARGS $CMAKE_LINKER_ARGS -S . -B cmake_build
fi

# Build ngen executable
echo "Building ngen..."
cmake --build cmake_build --target ngen -j ${NCORES:-4}

# Verify ngen binary (handles .exe on Windows)
if [ -x "cmake_build/ngen" ] || [ -f "cmake_build/ngen.exe" ]; then
  echo "ngen built successfully"
  NGEN_BIN="cmake_build/ngen"
  [ -f "cmake_build/ngen.exe" ] && NGEN_BIN="cmake_build/ngen.exe"
  ./$NGEN_BIN --help 2>/dev/null | head -5 || true
else
  echo "ngen binary not found"
  exit 1
fi

# ================================================================
# Build External BMI Modules (CFE, PET, SLOTH, Noah-MP)
# ================================================================
echo ""
echo "Building external BMI modules..."

# Helper: check if a shared library exists (glob-safe)
_lib_found() { ls "$@" 1>/dev/null 2>&1; }

# --- Build SLOTH (C++ module for soil/ice fractions) ---
if [ -d "extern/sloth" ] && [ -f "extern/sloth/CMakeLists.txt" ]; then
  echo "Building SLOTH..."
  (
    set +e  # don't abort main script if SLOTH build fails
    cd extern/sloth
    git_clean submodule update --init --recursive 2>/dev/null || true
    rm -rf cmake_build && mkdir -p cmake_build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -S . -B cmake_build 2>&1
    cmake --build cmake_build -j ${NCORES:-4} 2>&1
  )
  if _lib_found extern/sloth/cmake_build/libslothmodel.*; then
    echo "SLOTH built successfully"
  else
    echo "WARNING: SLOTH build failed (non-fatal)"
  fi
else
  echo "SLOTH submodule not found or empty — skipping"
fi

# --- Build CFE (C module - Conceptual Functional Equivalent) ---
if [ -d "extern/cfe" ] && [ -f "extern/cfe/CMakeLists.txt" ]; then
  echo "Building CFE..."
  (
    set +e
    cd extern/cfe
    git_clean submodule update --init --recursive 2>/dev/null || true
    rm -rf cmake_build && mkdir -p cmake_build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -S . -B cmake_build 2>&1
    cmake --build cmake_build -j ${NCORES:-4} 2>&1
  )
  if _lib_found extern/cfe/cmake_build/libcfebmi.*; then
    echo "CFE built successfully"
  else
    echo "WARNING: CFE build failed (non-fatal)"
  fi
else
  echo "CFE submodule not found or empty — skipping"
fi

# --- Build evapotranspiration/PET (C module) ---
# PET source lives one level deeper: extern/evapotranspiration/evapotranspiration/
_PET_SRC="extern/evapotranspiration/evapotranspiration"
if [ -d "$_PET_SRC" ] && [ -f "$_PET_SRC/CMakeLists.txt" ]; then
  echo "Building PET (evapotranspiration)..."
  (
    set +e
    cd "$_PET_SRC"
    git_clean submodule update --init --recursive 2>/dev/null || true
    rm -rf cmake_build && mkdir -p cmake_build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -S . -B cmake_build 2>&1
    cmake --build cmake_build -j ${NCORES:-4} 2>&1
  )
  if _lib_found "$_PET_SRC"/cmake_build/libpetbmi.*; then
    echo "PET built successfully"
  else
    echo "WARNING: PET build failed (non-fatal)"
  fi
else
  echo "PET submodule not found or empty — skipping"
fi

# --- Build iso_c_fortran_bmi (C wrapper for Fortran BMI) ---
# This must be built BEFORE Noah-MP as it provides the registration interface
if [ -d "extern/iso_c_fortran_bmi" ] && [ -f "extern/iso_c_fortran_bmi/CMakeLists.txt" ] && [ -n "$FC" ]; then
  echo "Building iso_c_fortran_bmi (C wrapper for Fortran BMI)..."
  (
    set +e
    cd extern/iso_c_fortran_bmi
    git_clean submodule update --init --recursive 2>/dev/null || true
    rm -rf cmake_build && mkdir -p cmake_build

    ISO_C_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    ISO_C_CMAKE_ARGS="$ISO_C_CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"
    ISO_C_CMAKE_ARGS="$ISO_C_CMAKE_ARGS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

    cmake $ISO_C_CMAKE_ARGS -S . -B cmake_build 2>&1
    cmake --build cmake_build -j ${NCORES:-4} 2>&1
  )
  if _lib_found extern/iso_c_fortran_bmi/cmake_build/libiso_c_bmi.*; then
    echo "iso_c_fortran_bmi built successfully"
  else
    echo "WARNING: iso_c_bmi library not found — Noah-MP may fail"
  fi
fi

# --- Build Noah-MP (noah-owp-modular, Fortran module) ---
if [ -d "extern/noah-owp-modular" ] && [ -f "extern/noah-owp-modular/CMakeLists.txt" ] && [ -n "$FC" ]; then
  echo "Building Noah-MP (noah-owp-modular, Fortran)..."
  (
    set +e
    cd extern/noah-owp-modular
    git_clean submodule update --init --recursive 2>/dev/null || true
    rm -rf cmake_build && mkdir -p cmake_build

    NOAH_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
    NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DCMAKE_Fortran_COMPILER=$FC"
    NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DNGEN_IS_MAIN_PROJECT=ON"
    NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

    if [ -n "$NETCDF_FORTRAN" ]; then
      NOAH_CMAKE_ARGS="$NOAH_CMAKE_ARGS -DNETCDF_PATH=$NETCDF_FORTRAN"
    fi

    cmake $NOAH_CMAKE_ARGS -S . -B cmake_build 2>&1
    cmake --build cmake_build -j ${NCORES:-4} 2>&1
  )
  if _lib_found extern/noah-owp-modular/cmake_build/libsurfacebmi.*; then
    echo "Noah-MP built successfully"
  else
    echo "WARNING: Noah-MP build failed (non-fatal)"
  fi
else
  if [ ! -d "extern/noah-owp-modular" ] || [ ! -f "extern/noah-owp-modular/CMakeLists.txt" ]; then
    echo "Noah-MP submodule not found or empty — skipping"
  elif [ -z "$FC" ]; then
    echo "No Fortran compiler available — skipping Noah-MP build"
  fi
fi

# ================================================================
# Install t-route (Python packages for routing)
# ================================================================
if [ -d "extern/t-route/src" ]; then
  echo ""
  echo "Installing t-route Python packages..."

  # Install python_routing_v02 (core troute routing package)
  if [ -d "extern/t-route/src/python_routing_v02" ]; then
    echo "Installing python_routing_v02 (troute)..."
    cd extern/t-route/src/python_routing_v02
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: python_routing_v02 installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install python_framework_v02 (troute framework)
  if [ -d "extern/t-route/src/python_framework_v02" ]; then
    echo "Installing python_framework_v02..."
    cd extern/t-route/src/python_framework_v02
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: python_framework_v02 installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install nwm_routing (required dependency for ngen_routing)
  if [ -d "extern/t-route/src/nwm_routing" ]; then
    echo "Installing nwm_routing..."
    cd extern/t-route/src/nwm_routing
    $PYTHON_EXE -m pip install -e . || {
      echo "WARNING: nwm_routing installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Install ngen_routing (main routing interface)
  if [ -d "extern/t-route/src/ngen_routing" ]; then
    echo "Installing ngen_routing..."
    cd extern/t-route/src/ngen_routing
    $PYTHON_EXE -m pip install -e . --no-deps || {
      echo "WARNING: ngen_routing installation failed (non-fatal)"
    }
    cd ../../../..
  fi

  # Verify installations
  if $PYTHON_EXE -c "import nwm_routing" 2>/dev/null; then
    echo "t-route nwm_routing installed successfully"
  else
    echo "t-route nwm_routing not available (non-fatal)"
  fi

  if $PYTHON_EXE -c "import ngen_routing" 2>/dev/null; then
    echo "t-route ngen_routing installed successfully"
  else
    echo "t-route ngen_routing not available (non-fatal)"
  fi
else
  echo "t-route submodule not found - routing will not be available"
fi

# ================================================================
# Fix UDUNITS2 XML path for statically-linked builds
# ================================================================
# When UDUNITS2 is statically linked, it resolves the XML database path
# relative to the executable (via dladdr): <exe_dir>/../share/udunits/udunits2.xml
# But our build installs UDUNITS2 into a udunits2/ subdirectory. Create a
# symlink so the path resolves correctly regardless of how it's computed.
if [ -d "udunits2/share/udunits" ] && [ ! -e "share/udunits" ]; then
  echo "Creating UDUNITS2 XML symlink for static-link path resolution..."
  mkdir -p share
  ln -s "$(pwd)/udunits2/share/udunits" share/udunits
  echo "UDUNITS2 XML symlink created: share/udunits -> udunits2/share/udunits"
fi

echo ""
echo "=============================================="
echo "ngen build summary:"
echo "=============================================="
echo "ngen binary: $(([ -x cmake_build/ngen ] || [ -f cmake_build/ngen.exe ]) && echo 'OK' || echo 'MISSING')"
echo "SLOTH:       $(_lib_found extern/sloth/cmake_build/libslothmodel.* && echo 'OK' || echo 'Not built')"
echo "CFE:         $(_lib_found extern/cfe/cmake_build/libcfebmi.* && echo 'OK' || echo 'Not built')"
echo "PET:         $(_lib_found extern/evapotranspiration/evapotranspiration/cmake_build/libpetbmi.* && echo 'OK' || echo 'Not built')"
echo "Noah-MP:     $(_lib_found extern/noah-owp-modular/cmake_build/libsurfacebmi.* && echo 'OK' || echo 'Not built')"
echo "t-route:     $($PYTHON_EXE -c 'import ngen_routing; print("OK")' 2>/dev/null || echo 'Not installed')"
echo "=============================================="
            '''.strip()
        ],
        'dependencies': [],
        'test_command': '--help',
        'verify_install': {
            'file_paths': ['cmake_build/ngen', 'cmake_build/ngen.exe'],
            'check_type': 'exists_any'
        },
        'order': 9
    }
