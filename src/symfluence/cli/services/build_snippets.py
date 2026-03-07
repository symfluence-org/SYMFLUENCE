# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Shared shell snippets for external tool builds.

This module contains reusable shell script fragments for detecting
system libraries (NetCDF, HDF5, GEOS, PROJ) across different platforms.
These are lightweight (no heavy dependencies) and can be safely imported
by the CLI without loading pandas, xarray, etc.
"""

from typing import Dict


def get_common_build_environment() -> str:
    """
    Get common build environment setup used across multiple tools.

    Returns:
        Shell script snippet for environment configuration.
    """
    return r'''
set -e

# ================================================================
# HPC Environment Detection and Guidance
# ================================================================
detect_hpc_environment() {
    HPC_DETECTED=false
    HPC_NAME=""

    # Check for common HPC indicators
    if [ -d "/cvmfs/soft.computecanada.ca" ] || [ -n "${CC_CLUSTER:-}" ]; then
        HPC_DETECTED=true
        HPC_NAME="Compute Canada / Digital Research Alliance"
    elif [ -n "${NERSC_HOST:-}" ]; then
        HPC_DETECTED=true
        HPC_NAME="NERSC"
    elif [ -n "${TACC_SYSTEM:-}" ]; then
        HPC_DETECTED=true
        HPC_NAME="TACC"
    elif [ -n "${PBS_O_HOST:-}" ] || [ -n "${SLURM_CLUSTER_NAME:-}" ]; then
        HPC_DETECTED=true
        HPC_NAME="HPC Cluster"
    fi

    if [ "$HPC_DETECTED" = true ]; then
        echo "=================================================="
        echo "HPC Environment Detected: $HPC_NAME"
        echo "=================================================="
        echo ""
        echo "For successful builds, ensure required modules are loaded."
        echo "Example for Compute Canada:"
        echo "  module load StdEnv/2023"
        echo "  module load gcc/12.3 cmake/3.27.7"
        echo "  module load netcdf/4.9.2 expat udunits/2.2.28"
        echo "  module load geos proj"
        echo ""
        echo "Current loaded modules (if available):"
        module list 2>/dev/null || echo "  (module command not available)"
        echo ""
    fi
}
detect_hpc_environment

# ================================================================
# 2i2c / JupyterHub Compiler Configuration
# ================================================================
# Respect pre-configured compilers for ABI compatibility with conda libraries.
# The symfluence shell script sets CC/CXX to conda compilers when available.
configure_compilers() {
    # On MSYS2/MinGW (Windows), use bare compiler names so cmake can
    # find them on PATH with the .exe extension.
    case "$(uname -s 2>/dev/null)" in
        MSYS*|MINGW*|CYGWIN*)
            if command -v gcc >/dev/null 2>&1; then
                export CC="gcc"
                export CXX="g++"
                echo "Using compilers: CC=gcc, CXX=g++ (MSYS2/MinGW)"
                return 0
            fi
            ;;
    esac

    # If CC/CXX are already set to conda compilers, trust them
    # (symfluence binary install sets these for ABI compatibility)
    if [ -n "$CC" ] && [[ "$CC" == *conda* ]]; then
        echo "Using pre-configured conda compiler: CC=$CC"
        [ -n "$CXX" ] && echo "  CXX=$CXX"
        return 0
    fi

    # If CC is already set (e.g., by user or HPC module), resolve and trust it.
    # This handles both absolute paths and bare names (e.g., CC=gcc).
    if [ -n "$CC" ]; then
        if [ -x "$CC" ]; then
            echo "Using pre-set compiler: CC=$CC"
            [ -n "$CXX" ] && echo "  CXX=$CXX"
            return 0
        fi
        # Bare name — resolve to full path
        local cc_resolved
        cc_resolved="$(command -v "$CC" 2>/dev/null)"
        if [ -n "$cc_resolved" ] && [ -x "$cc_resolved" ]; then
            export CC="$cc_resolved"
            if [ -n "$CXX" ]; then
                local cxx_resolved
                cxx_resolved="$(command -v "$CXX" 2>/dev/null)"
                [ -n "$cxx_resolved" ] && [ -x "$cxx_resolved" ] && export CXX="$cxx_resolved"
            fi
            echo "Resolved pre-set compiler: CC=$CC"
            [ -n "$CXX" ] && echo "  CXX=$CXX"
            return 0
        fi
    fi

    # Only use system compilers if explicitly requested
    if [ "${SYMFLUENCE_USE_SYSTEM_COMPILERS:-}" = "true" ]; then
        [ -x /usr/bin/gcc ] && export CC=/usr/bin/gcc
        [ -x /usr/bin/g++ ] && export CXX=/usr/bin/g++
        echo "Using system compilers (requested via SYMFLUENCE_USE_SYSTEM_COMPILERS)"
        echo "  CC=$CC, CXX=${CXX:-not set}"
        return 0
    fi

    # If no compiler is set but we're in a conda env, try to find conda compilers
    if [ -n "$CONDA_PREFIX" ] && [ -z "$CC" ]; then
        local conda_gcc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
        local conda_gxx="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        if [ -x "$conda_gcc" ]; then
            export CC="$conda_gcc"
            [ -x "$conda_gxx" ] && export CXX="$conda_gxx"
            echo "Auto-detected conda compilers: CC=$CC"
        fi
    fi

    # Report current compiler configuration
    if [ -n "$CC" ]; then
        echo "Compiler configuration: CC=$CC"
        [ -n "$CXX" ] && echo "  CXX=$CXX"
    fi
}
configure_compilers

# ================================================================
# Fortran Compiler Detection
# ================================================================
# Look for conda gfortran first (for ABI compatibility), then system gfortran
configure_fortran() {
    # On MSYS2/MinGW (Windows), use bare compiler names so that cmake
    # can find them on PATH with the .exe extension.  MSYS2's
    # "command -v" returns paths like /conda_bin/gfortran that cmake
    # cannot resolve to a Windows executable.
    case "$(uname -s 2>/dev/null)" in
        MSYS*|MINGW*|CYGWIN*)
            if command -v gfortran >/dev/null 2>&1; then
                export FC="gfortran"
                export FC_EXE="gfortran"
                echo "Using Fortran compiler: FC=gfortran (MSYS2/MinGW)"
                return 0
            fi
            ;;
    esac

    # Already set — resolve to full path if needed.
    # Handles both absolute paths (/path/to/gfortran) and bare names (gfortran).
    if [ -n "$FC" ]; then
        if [ -x "$FC" ]; then
            echo "Using FC=$FC"
            export FC_EXE="$FC"
            return 0
        fi
        # Bare name (e.g., FC=gfortran from env) — resolve via PATH
        local fc_resolved
        fc_resolved="$(command -v "$FC" 2>/dev/null)"
        if [ -n "$fc_resolved" ] && [ -x "$fc_resolved" ]; then
            export FC="$fc_resolved"
            export FC_EXE="$FC"
            echo "Resolved FC=$FC"
            return 0
        fi
    fi

    # Try conda gfortran first (from compilers package)
    if [ -n "$CONDA_PREFIX" ]; then
        local conda_fc="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gfortran"
        if [ -x "$conda_fc" ]; then
            export FC="$conda_fc"
            export FC_EXE="$FC"
            echo "Using conda Fortran compiler: FC=$FC"
            return 0
        fi
    fi

    # Fall back to gfortran on PATH, but prefer non-system over /usr/bin.
    # On HPC, 'module load gcc' adds a spack gfortran to PATH that should
    # take precedence over the ancient /usr/bin/gfortran.
    if command -v gfortran >/dev/null 2>&1; then
        local fc_path
        fc_path="$(command -v gfortran)"
        # Check if there's a non-system gfortran further down PATH
        # (command -v returns the first match; on some HPC systems /usr/bin
        # comes before the module's bin/ directory)
        if [[ "$fc_path" == "/usr/bin/gfortran" ]]; then
            local alt_fc=""
            # Search PATH for a gfortran NOT in /usr/bin
            IFS=':' read -ra _paths <<< "$PATH"
            for _p in "${_paths[@]}"; do
                if [[ "$_p" != "/usr/bin" && "$_p" != "/usr/sbin" ]] && [ -x "$_p/gfortran" ]; then
                    alt_fc="$_p/gfortran"
                    break
                fi
            done
            if [ -n "$alt_fc" ]; then
                fc_path="$alt_fc"
                echo "Preferred module gfortran over /usr/bin/gfortran"
            fi
        fi
        export FC="$fc_path"
        export FC_EXE="$FC"
        echo "Using Fortran compiler: FC=$FC"
        return 0
    fi

    # Last resort
    export FC="${FC:-gfortran}"
    export FC_EXE="$FC"
    echo "Warning: gfortran not found, set FC=$FC"
}
configure_fortran

# ================================================================
# GLIBC Compatibility Workaround
# ================================================================
# On some HPC systems (e.g., ComputeCanada StdEnv/2023), the toolchain's
# libgcc_s.so.1 was built against a newer GLIBC than is available on
# compute nodes, causing "_dl_find_object@GLIBC_2.35" link failures.
# Detect this and add -static-libgcc to avoid the dynamic libgcc_s dependency.
fix_libgcc_glibc_mismatch() {
    # Only check on Linux
    [ "$(uname -s)" = "Linux" ] || return 0

    local _test_fc="${FC:-${FC_EXE:-gfortran}}"
    if ! command -v "$_test_fc" >/dev/null 2>&1; then
        _test_fc="$(command -v gfortran 2>/dev/null)" || return 0
    fi

    # Find the libgcc_s.so.1 that the Fortran compiler would link against.
    # Strategy 1: ask the compiler directly
    local _libgcc_path=""
    _libgcc_path="$("$_test_fc" -print-file-name=libgcc_s.so.1 2>/dev/null)" || true

    # -print-file-name returns just the basename if it can't resolve it
    if [ -z "$_libgcc_path" ] || [ "$_libgcc_path" = "libgcc_s.so.1" ] || [ ! -f "$_libgcc_path" ]; then
        _libgcc_path=""
    fi

    # Strategy 2: search relative to the compiler binary
    if [ -z "$_libgcc_path" ]; then
        local _fc_real
        _fc_real="$(readlink -f "$(command -v "$_test_fc")" 2>/dev/null)" || _fc_real="$(command -v "$_test_fc")"
        local _fc_dir
        _fc_dir="$(dirname "$_fc_real")"

        # Try common gcc layouts (unquoted to allow glob expansion)
        local _candidate
        for _candidate in \
            "$_fc_dir"/../lib/libgcc_s.so.1 \
            "$_fc_dir"/../lib64/libgcc_s.so.1 \
            "$_fc_dir"/../../../lib/gcc/x86_64-pc-linux-gnu/*/libgcc_s.so.1 \
            "$_fc_dir"/../../lib/gcc/x86_64-pc-linux-gnu/*/libgcc_s.so.1; do
            if [ -f "$_candidate" ]; then
                _libgcc_path="$_candidate"
                break
            fi
        done
    fi

    # Strategy 3: known Gentoo/ComputeCanada path
    if [ -z "$_libgcc_path" ]; then
        local _candidate
        for _candidate in /cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib/gcc/x86_64-pc-linux-gnu/*/libgcc_s.so.1; do
            if [ -f "$_candidate" ]; then
                _libgcc_path="$_candidate"
                break
            fi
        done
    fi

    if [ -z "$_libgcc_path" ] || [ ! -f "$_libgcc_path" ]; then
        return 0
    fi

    # Check if libgcc_s.so.1 references GLIBC_2.35+ symbols.
    # Use strings as a universal fallback since objdump/readelf may not
    # be available or may produce unexpected output formats.
    local _needs_235=false
    if strings "$_libgcc_path" 2>/dev/null | grep -qE "GLIBC_2\.(3[5-9]|[4-9][0-9])"; then
        _needs_235=true
    elif command -v readelf >/dev/null 2>&1; then
        if readelf -V "$_libgcc_path" 2>/dev/null | grep -qE "GLIBC_2\.(3[5-9]|[4-9][0-9])"; then
            _needs_235=true
        fi
    fi

    if [ "$_needs_235" = "false" ]; then
        return 0
    fi

    # Verify the system libc doesn't actually provide GLIBC_2.35
    # (if it does, there's no mismatch and no fix needed)
    local _sys_glibc_ver=""
    _sys_glibc_ver="$(ldd --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+$')" || true
    if [ -n "$_sys_glibc_ver" ]; then
        local _major="${_sys_glibc_ver%%.*}"
        local _minor="${_sys_glibc_ver##*.}"
        # If system glibc >= 2.35, no mismatch
        if [ "$_major" -gt 2 ] 2>/dev/null || { [ "$_major" -eq 2 ] && [ "$_minor" -ge 35 ]; } 2>/dev/null; then
            return 0
        fi
    fi

    echo "Detected GLIBC incompatibility: libgcc_s.so.1 requires GLIBC_2.35+ but system has $_sys_glibc_ver"
    echo "  libgcc_s: $_libgcc_path"
    echo "  Adding -static-libgcc to avoid dynamic libgcc_s dependency"
    export LDFLAGS="-static-libgcc ${LDFLAGS:-}"
    export FFLAGS="-static-libgcc ${FFLAGS:-}"
    export FCFLAGS="-static-libgcc ${FCFLAGS:-}"
    export CMAKE_EXE_LINKER_FLAGS="-static-libgcc ${CMAKE_EXE_LINKER_FLAGS:-}"
    export CMAKE_SHARED_LINKER_FLAGS="-static-libgcc ${CMAKE_SHARED_LINKER_FLAGS:-}"
}
fix_libgcc_glibc_mismatch

# ================================================================
# Library Discovery
# ================================================================
# On Windows conda, executables and libraries live under
# CONDA_PREFIX/Library (not CONDA_PREFIX).  CONDA_LIB_PREFIX is
# the correct root for include/, lib/, bin/ lookups.
export CONDA_LIB_PREFIX="${CONDA_PREFIX}"
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        if [ -n "$CONDA_PREFIX" ] && [ -d "${CONDA_PREFIX}/Library/lib" ]; then
            export CONDA_LIB_PREFIX="${CONDA_PREFIX}/Library"
        fi
        ;;
esac

# Discover libraries - prefer conda prefix if available
configure_libraries() {
    local clp="${CONDA_LIB_PREFIX}"

    # NetCDF: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -f "$clp/bin/nc-config" ]; then
        export NETCDF="$clp"
        echo "Using conda NetCDF: $NETCDF"
    else
        export NETCDF="${NETCDF:-$(nc-config --prefix 2>/dev/null || echo /usr)}"
    fi

    # NetCDF-Fortran: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -f "$clp/bin/nf-config" ]; then
        export NETCDF_FORTRAN="$clp"
        echo "Using conda NetCDF-Fortran: $NETCDF_FORTRAN"
    else
        export NETCDF_FORTRAN="${NETCDF_FORTRAN:-$(nf-config --prefix 2>/dev/null || echo /usr)}"
    fi

    # HDF5: prefer conda installation
    if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/lib" ] && ls "$clp/lib"/*hdf5* >/dev/null 2>&1; then
        export HDF5_ROOT="$clp"
        echo "Using conda HDF5: $HDF5_ROOT"
    else
        export HDF5_ROOT="${HDF5_ROOT:-$(h5cc -showconfig 2>/dev/null | awk -F': ' "/Installation point/{print \$2}" || echo /usr)}"
    fi
}
configure_libraries

# Threads
export NCORES="${NCORES:-4}"

# ================================================================
# Portable in-place sed
# ================================================================
# macOS sed requires an explicit backup extension with -i (e.g. sed -i ''),
# while GNU sed treats the next argument as the sed expression.  This wrapper
# abstracts the difference so build scripts can use `_sed_i 's/foo/bar/' file`.
_sed_i() {
    if [ "$(uname -s)" = "Darwin" ]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}
    '''.strip()


def get_netcdf_detection() -> str:
    """
    Get reusable NetCDF detection shell snippet.

    Sets NETCDF_FORTRAN and NETCDF_C environment variables.
    Works on Linux (apt), macOS (Homebrew), conda environments, and HPC systems.

    Returns:
        Shell script snippet for NetCDF detection.
    """
    return r'''
# === NetCDF Detection (reusable snippet) ===
# Helper: check if a directory contains Fortran NetCDF files (netcdf.mod or libnetcdff)
_has_fortran_netcdf() {
    local path="$1"
    [ -d "$path/include" ] && \
    { [ -f "$path/include/netcdf.mod" ] || \
      ls "$path/lib"/libnetcdff.* >/dev/null 2>&1 || \
      ls "$path/lib64"/libnetcdff.* >/dev/null 2>&1; }
}
detect_netcdf() {
    # ── NetCDF-Fortran detection ──
    # On HPC systems with Spack, NETCDF_ROOT often points to netcdf-c only,
    # NOT the Fortran bindings. We must use Fortran-specific methods first.

    # 0. If already set and validated (e.g., by configure_libraries via nf-config), keep it
    if [ -n "${NETCDF_FORTRAN:-}" ] && _has_fortran_netcdf "${NETCDF_FORTRAN}"; then
        echo "Using pre-set NETCDF_FORTRAN at: ${NETCDF_FORTRAN}"
    else
        NETCDF_FORTRAN=""

        # 1. EasyBuild module (Compute Canada, NERSC, etc.)
        if [ -n "${EBROOTNETCDFMINFORTRAN:-}" ] && [ -d "${EBROOTNETCDFMINFORTRAN}/include" ]; then
            NETCDF_FORTRAN="${EBROOTNETCDFMINFORTRAN}"
            echo "Found HPC module NetCDF-Fortran (EasyBuild) at: ${NETCDF_FORTRAN}"

        # 2. Explicit Fortran root variable (Spack, manual HPC setup)
        elif [ -n "${NETCDF_FORTRAN_ROOT:-}" ] && [ -d "${NETCDF_FORTRAN_ROOT}/include" ]; then
            NETCDF_FORTRAN="${NETCDF_FORTRAN_ROOT}"
            echo "Found NetCDF-Fortran via NETCDF_FORTRAN_ROOT at: ${NETCDF_FORTRAN}"

        # 3. nf-config tool (most reliable for any system with netcdf-fortran installed)
        elif command -v nf-config >/dev/null 2>&1; then
            local nf_prefix
            nf_prefix="$(nf-config --prefix 2>/dev/null)"
            if [ -n "$nf_prefix" ] && [ -d "$nf_prefix" ]; then
                NETCDF_FORTRAN="$nf_prefix"
                echo "Found NetCDF-Fortran via nf-config at: ${NETCDF_FORTRAN}"
            fi

        # 4. Conda environment
        # CONDA_LIB_PREFIX is set by get_common_build_environment() and handles
        # Windows conda layout (CONDA_PREFIX/Library vs CONDA_PREFIX).
        elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_LIB_PREFIX:-$CONDA_PREFIX}/bin/nf-config" ]; then
            NETCDF_FORTRAN="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
            echo "Found conda NetCDF-Fortran at: ${NETCDF_FORTRAN}"
        fi

        # 5. Generic HPC variables - but ONLY if they actually contain Fortran files.
        #    On Spack systems, NETCDF_ROOT typically points to netcdf-c (no .mod files).
        if [ -z "${NETCDF_FORTRAN}" ]; then
            if [ -n "${EBROOTNETCDF:-}" ] && _has_fortran_netcdf "${EBROOTNETCDF}"; then
                NETCDF_FORTRAN="${EBROOTNETCDF}"
                echo "Found NetCDF-Fortran via EBROOTNETCDF at: ${NETCDF_FORTRAN}"
            elif [ -n "${NETCDF_ROOT:-}" ] && _has_fortran_netcdf "${NETCDF_ROOT}"; then
                NETCDF_FORTRAN="${NETCDF_ROOT}"
                echo "Found NetCDF-Fortran via NETCDF_ROOT at: ${NETCDF_FORTRAN}"
            elif [ -n "${NETCDF_DIR:-}" ] && _has_fortran_netcdf "${NETCDF_DIR}"; then
                NETCDF_FORTRAN="${NETCDF_DIR}"
                echo "Found NetCDF-Fortran via NETCDF_DIR at: ${NETCDF_FORTRAN}"
            fi
        fi

        # 6. Spack sibling search: if nc-config is available, look for nf-config
        #    in sibling Spack directories (netcdf-fortran alongside netcdf-c).
        if [ -z "${NETCDF_FORTRAN}" ]; then
            local _nc_bin=""
            _nc_bin="$(command -v nc-config 2>/dev/null)" || true
            if [ -n "$_nc_bin" ]; then
                # e.g. /apps/spack/.../netcdf-c/<hash>/bin/nc-config → search ../../../netcdf-fortran/*/bin/nf-config
                local _apps_dir
                _apps_dir="$(dirname "$(dirname "$(dirname "$_nc_bin")")")"
                if [ -d "$_apps_dir/netcdf-fortran" ]; then
                    for _nf_bin in "$_apps_dir"/netcdf-fortran/*/bin/nf-config; do
                        if [ -x "$_nf_bin" ]; then
                            local _nf_prefix
                            _nf_prefix="$("$_nf_bin" --prefix 2>/dev/null)"
                            if [ -n "$_nf_prefix" ] && _has_fortran_netcdf "$_nf_prefix"; then
                                NETCDF_FORTRAN="$_nf_prefix"
                                echo "Found NetCDF-Fortran via Spack sibling at: ${NETCDF_FORTRAN}"
                                break
                            fi
                        fi
                    done
                fi
            fi
        fi

        # 7. Fallback: NETCDF env var (only if it has Fortran files), system paths
        if [ -z "${NETCDF_FORTRAN}" ]; then
            if [ -n "${NETCDF:-}" ] && _has_fortran_netcdf "${NETCDF}"; then
                NETCDF_FORTRAN="${NETCDF}"
                echo "Using NETCDF env var: ${NETCDF_FORTRAN}"
            else
                # Try common locations (Homebrew, system paths)
                for try_path in /opt/homebrew/opt/netcdf-fortran /opt/homebrew/opt/netcdf \
                                /usr/local/opt/netcdf-fortran /usr/local/opt/netcdf /usr/local /usr; do
                    if _has_fortran_netcdf "$try_path"; then
                        NETCDF_FORTRAN="$try_path"
                        echo "Found NetCDF-Fortran at: $try_path"
                        break
                    fi
                done
            fi
        fi
    fi

    # ── NetCDF-C detection ──
    # For C, NETCDF_ROOT is typically correct (it usually IS the C library)
    if [ -n "${EBROOTNETCDF:-}" ] && [ -d "${EBROOTNETCDF}/lib" ]; then
        NETCDF_C="${EBROOTNETCDF}"
    elif [ -n "${NETCDF_C_ROOT:-}" ] && [ -d "${NETCDF_C_ROOT}/lib" ]; then
        NETCDF_C="${NETCDF_C_ROOT}"
    elif [ -n "${NETCDF_ROOT:-}" ] && [ -d "${NETCDF_ROOT}/lib" ]; then
        NETCDF_C="${NETCDF_ROOT}"
    elif [ -n "${NETCDF_DIR:-}" ] && [ -d "${NETCDF_DIR}/lib" ]; then
        NETCDF_C="${NETCDF_DIR}"
    elif [ -n "${CONDA_PREFIX:-}" ] && [ -f "${CONDA_LIB_PREFIX:-$CONDA_PREFIX}/bin/nc-config" ]; then
        NETCDF_C="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    elif command -v nc-config >/dev/null 2>&1; then
        local nc_prefix
        nc_prefix="$(nc-config --prefix 2>/dev/null)"
        if [ -n "$nc_prefix" ] && [ -d "$nc_prefix" ]; then
            NETCDF_C="$nc_prefix"
        fi
    elif [ -d "/opt/homebrew/opt/netcdf" ]; then
        NETCDF_C="/opt/homebrew/opt/netcdf"
    else
        NETCDF_C="${NETCDF_FORTRAN}"
    fi

    export NETCDF_FORTRAN NETCDF_C
    echo "NetCDF detection complete: NETCDF_FORTRAN=${NETCDF_FORTRAN:-not found}, NETCDF_C=${NETCDF_C:-not found}"
}
detect_netcdf
    '''.strip()


def get_hdf5_detection() -> str:
    """
    Get reusable HDF5 detection shell snippet.

    Sets HDF5_ROOT, HDF5_LIB_DIR, and HDF5_INC_DIR environment variables.
    Handles Ubuntu's hdf5/serial subdirectory structure.

    Returns:
        Shell script snippet for HDF5 detection.
    """
    return r'''
# === HDF5 Detection (reusable snippet) ===
detect_hdf5() {
    # Try h5cc config tool first
    if command -v h5cc >/dev/null 2>&1; then
        HDF5_ROOT="$(h5cc -showconfig 2>/dev/null | grep -i "Installation point" | sed 's/.*: *//' | head -n1)"
    fi

    # Try conda environment (CONDA_LIB_PREFIX handles Windows layout)
    if [ -z "$HDF5_ROOT" ] || [ ! -d "$HDF5_ROOT" ]; then
        local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
        if [ -n "$CONDA_PREFIX" ] && [ -d "$clp/lib" ] && ls "$clp/lib"/*hdf5* >/dev/null 2>&1; then
            HDF5_ROOT="$clp"
            echo "Found conda HDF5 at: $HDF5_ROOT"
        fi
    fi

    # Fallback detection
    if [ -z "$HDF5_ROOT" ] || [ ! -d "$HDF5_ROOT" ]; then
        if [ -n "$HDF5_ROOT" ] && [ -d "$HDF5_ROOT" ]; then
            : # Use existing env var
        elif command -v brew >/dev/null 2>&1 && brew --prefix hdf5 >/dev/null 2>&1; then
            HDF5_ROOT="$(brew --prefix hdf5)"
        else
            for path in /usr $HOME/.local /opt/hdf5; do
                if [ -d "$path/include" ] && [ -d "$path/lib" ]; then
                    HDF5_ROOT="$path"
                    break
                fi
            done
        fi
    fi
    HDF5_ROOT="${HDF5_ROOT:-/usr}"

    # Find lib directory (Ubuntu stores in hdf5/serial, others in lib64 or lib)
    if [ -d "${HDF5_ROOT}/lib/x86_64-linux-gnu/hdf5/serial" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib/x86_64-linux-gnu/hdf5/serial"
    elif [ -d "${HDF5_ROOT}/lib/x86_64-linux-gnu" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib/x86_64-linux-gnu"
    elif [ -d "${HDF5_ROOT}/lib64" ]; then
        HDF5_LIB_DIR="${HDF5_ROOT}/lib64"
    else
        HDF5_LIB_DIR="${HDF5_ROOT}/lib"
    fi

    # Find include directory
    if [ -d "${HDF5_ROOT}/include/hdf5/serial" ]; then
        HDF5_INC_DIR="${HDF5_ROOT}/include/hdf5/serial"
    else
        HDF5_INC_DIR="${HDF5_ROOT}/include"
    fi

    export HDF5_ROOT HDF5_LIB_DIR HDF5_INC_DIR
}
detect_hdf5
    '''.strip()


def get_netcdf_lib_detection() -> str:
    """
    Get reusable NetCDF library path detection snippet.

    Sets NETCDF_LIB_DIR and NETCDF_C_LIB_DIR for linking.
    Handles Debian/Ubuntu x86_64-linux-gnu paths and lib64 paths.

    Returns:
        Shell script snippet for NetCDF library path detection.
    """
    return r'''
# === NetCDF Library Path Detection ===
detect_netcdf_lib_paths() {
    # Find NetCDF-Fortran lib directory
    if [ -d "${NETCDF_FORTRAN}/lib/x86_64-linux-gnu" ] && \
       ls "${NETCDF_FORTRAN}/lib/x86_64-linux-gnu"/libnetcdff.* >/dev/null 2>&1; then
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib/x86_64-linux-gnu"
    elif [ -d "${NETCDF_FORTRAN}/lib64" ] && \
         ls "${NETCDF_FORTRAN}/lib64"/libnetcdff.* >/dev/null 2>&1; then
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib64"
    else
        NETCDF_LIB_DIR="${NETCDF_FORTRAN}/lib"
    fi

    # Find NetCDF-C lib directory (may differ from Fortran)
    if [ -d "${NETCDF_C}/lib/x86_64-linux-gnu" ] && \
       ls "${NETCDF_C}/lib/x86_64-linux-gnu"/libnetcdf.* >/dev/null 2>&1; then
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib/x86_64-linux-gnu"
    elif [ -d "${NETCDF_C}/lib64" ] && \
         ls "${NETCDF_C}/lib64"/libnetcdf.* >/dev/null 2>&1; then
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib64"
    else
        NETCDF_C_LIB_DIR="${NETCDF_C}/lib"
    fi

    export NETCDF_LIB_DIR NETCDF_C_LIB_DIR
}
detect_netcdf_lib_paths
    '''.strip()


def get_geos_proj_detection() -> str:
    """
    Get reusable GEOS and PROJ detection shell snippet.

    Sets GEOS_CFLAGS, GEOS_LDFLAGS, PROJ_CFLAGS, PROJ_LDFLAGS.

    Returns:
        Shell script snippet for GEOS/PROJ detection.
    """
    return r'''
# === GEOS and PROJ Detection ===
detect_geos_proj() {
    GEOS_CFLAGS="" GEOS_LDFLAGS="" PROJ_CFLAGS="" PROJ_LDFLAGS=""

    # Try geos-config tool FIRST - it returns proper flags with all dependencies
    if command -v geos-config >/dev/null 2>&1; then
        GEOS_CFLAGS="$(geos-config --cflags 2>/dev/null || true)"
        GEOS_LDFLAGS="$(geos-config --clibs 2>/dev/null || true)"
        if [ -n "$GEOS_CFLAGS" ] && [ -n "$GEOS_LDFLAGS" ]; then
            echo "GEOS found via geos-config: $GEOS_LDFLAGS"
        else
            GEOS_CFLAGS="" GEOS_LDFLAGS=""
        fi
    fi

    # Try pkg-config for GEOS if geos-config didn't work
    if [ -z "$GEOS_CFLAGS" ] && command -v pkg-config >/dev/null 2>&1; then
        if pkg-config --exists geos 2>/dev/null; then
            GEOS_CFLAGS="$(pkg-config --cflags geos 2>/dev/null || true)"
            GEOS_LDFLAGS="$(pkg-config --libs geos 2>/dev/null || true)"
            if [ -n "$GEOS_CFLAGS" ]; then
                echo "GEOS found via pkg-config"
            fi
        fi
    fi

    # Try pkg-config for PROJ (includes all dependencies like libtiff)
    if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists proj 2>/dev/null; then
        PROJ_CFLAGS="$(pkg-config --cflags proj 2>/dev/null || true)"
        PROJ_LDFLAGS="$(pkg-config --libs proj 2>/dev/null || true)"
        if [ -n "$PROJ_CFLAGS" ] && [ -n "$PROJ_LDFLAGS" ]; then
            echo "PROJ found via pkg-config: $PROJ_LDFLAGS"
        else
            PROJ_CFLAGS="" PROJ_LDFLAGS=""
        fi
    fi

    # Fall back to HPC module environment variables (EasyBuild)
    # Note: Using modules directly may miss transitive deps like libtiff
    if [ -z "$GEOS_CFLAGS" ] && [ -n "$EBROOTGEOS" ] && [ -d "$EBROOTGEOS" ]; then
        GEOS_CFLAGS="-I${EBROOTGEOS}/include"
        for libdir in "$EBROOTGEOS/lib64" "$EBROOTGEOS/lib"; do
            if [ -f "$libdir/libgeos_c.so" ] || [ -f "$libdir/libgeos_c.a" ]; then
                GEOS_LDFLAGS="-L$libdir -lgeos_c"
                break
            fi
        done
        if [ -n "$GEOS_LDFLAGS" ]; then
            echo "GEOS found via HPC module at: $EBROOTGEOS"
        fi
    fi

    if [ -z "$PROJ_CFLAGS" ] && [ -n "$EBROOTPROJ" ] && [ -d "$EBROOTPROJ" ]; then
        PROJ_CFLAGS="-I${EBROOTPROJ}/include"
        for libdir in "$EBROOTPROJ/lib64" "$EBROOTPROJ/lib"; do
            if [ -f "$libdir/libproj.so" ] || [ -f "$libdir/libproj.a" ]; then
                PROJ_LDFLAGS="-L$libdir -lproj"
                break
            fi
        done
        if [ -n "$PROJ_LDFLAGS" ]; then
            echo "PROJ found via HPC module at: $EBROOTPROJ"
            echo "WARNING: Using HPC module PROJ directly - if linking fails with libtiff errors,"
            echo "         try: module load gdal  (which provides PROJ with proper dependencies)"
        fi
    fi

    # macOS Homebrew fallback
    if [ "$(uname)" = "Darwin" ]; then
        if [ -z "$GEOS_CFLAGS" ] && command -v brew >/dev/null 2>&1; then
            GEOS_PREFIX="$(brew --prefix geos 2>/dev/null || true)"
            if [ -n "$GEOS_PREFIX" ] && [ -d "$GEOS_PREFIX" ]; then
                GEOS_CFLAGS="-I${GEOS_PREFIX}/include"
                GEOS_LDFLAGS="-L${GEOS_PREFIX}/lib -lgeos_c"
                echo "GEOS found via Homebrew"
            fi
        fi
        if [ -z "$PROJ_CFLAGS" ] && command -v brew >/dev/null 2>&1; then
            PROJ_PREFIX="$(brew --prefix proj 2>/dev/null || true)"
            if [ -n "$PROJ_PREFIX" ] && [ -d "$PROJ_PREFIX" ]; then
                PROJ_CFLAGS="-I${PROJ_PREFIX}/include"
                PROJ_LDFLAGS="-L${PROJ_PREFIX}/lib -lproj"
                echo "PROJ found via Homebrew"
            fi
        fi
    fi

    # Windows conda fallback (GEOS/PROJ from conda-forge)
    if [ -z "$GEOS_CFLAGS" ] && [ -n "$CONDA_PREFIX" ]; then
        local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
        if [ -f "$clp/lib/geos_c.lib" ] || [ -f "$clp/lib/libgeos_c.dll.a" ] || [ -f "$clp/lib/libgeos_c.so" ]; then
            GEOS_CFLAGS="-I$clp/include"
            GEOS_LDFLAGS="-L$clp/lib -lgeos_c"
            echo "GEOS found in conda at: $clp"
        fi
    fi
    if [ -z "$PROJ_CFLAGS" ] && [ -n "$CONDA_PREFIX" ]; then
        local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
        if [ -f "$clp/lib/proj.lib" ] || [ -f "$clp/lib/libproj.dll.a" ] || [ -f "$clp/lib/libproj.so" ]; then
            PROJ_CFLAGS="-I$clp/include"
            PROJ_LDFLAGS="-L$clp/lib -lproj"
            echo "PROJ found in conda at: $clp"
        fi
    fi

    # Common path fallback
    if [ -z "$GEOS_CFLAGS" ]; then
        for path in /usr/local /usr; do
            if [ -f "$path/lib/libgeos_c.so" ] || [ -f "$path/lib/libgeos_c.dylib" ]; then
                GEOS_CFLAGS="-I$path/include"
                GEOS_LDFLAGS="-L$path/lib -lgeos_c"
                echo "GEOS found in $path"
                break
            fi
        done
    fi
    if [ -z "$PROJ_CFLAGS" ]; then
        for path in /usr/local /usr; do
            if [ -f "$path/lib/libproj.so" ] || [ -f "$path/lib/libproj.dylib" ]; then
                PROJ_CFLAGS="-I$path/include"
                PROJ_LDFLAGS="-L$path/lib -lproj"
                echo "PROJ found in $path"
                break
            fi
        done
    fi

    export GEOS_CFLAGS GEOS_LDFLAGS PROJ_CFLAGS PROJ_LDFLAGS
}
detect_geos_proj
    '''.strip()


def get_udunits2_detection_and_build() -> str:
    """
    Get reusable UDUNITS2 detection and build-from-source snippet.

    Sets UDUNITS2_DIR, UDUNITS2_INCLUDE_DIR, UDUNITS2_LIBRARY environment variables.
    If UDUNITS2 is not found system-wide, builds it from source in a local directory.

    Returns:
        Shell script snippet for UDUNITS2 detection and building.
    """
    return r'''
# === UDUNITS2 Detection and Build ===
detect_or_build_udunits2() {
    UDUNITS2_FOUND=false
    EXPAT_LIB_DIR=""
    UDUNITS2_FROM_HPC_MODULE=false

    # Helper: set UDUNITS2 variables from a root directory
    _set_udunits2_from_root() {
        local root="$1"
        UDUNITS2_DIR="$root"
        UDUNITS2_INCLUDE_DIR="$root/include"
        if [ -f "$root/lib/libudunits2.so" ]; then
            UDUNITS2_LIBRARY="$root/lib/libudunits2.so"
        elif [ -f "$root/lib64/libudunits2.so" ]; then
            UDUNITS2_LIBRARY="$root/lib64/libudunits2.so"
        elif [ -f "$root/lib/libudunits2.dylib" ]; then
            UDUNITS2_LIBRARY="$root/lib/libudunits2.dylib"
        else
            UDUNITS2_LIBRARY="$root/lib/libudunits2.a"
        fi
    }

    # 1. Check HPC environment variables (EasyBuild module system)
    if [ -n "${EBROOTUDUNITS:-}" ] && [ -f "$EBROOTUDUNITS/include/udunits2.h" ]; then
        _set_udunits2_from_root "$EBROOTUDUNITS"
        echo "Found HPC module UDUNITS2 (EasyBuild) at: ${UDUNITS2_DIR}"
        UDUNITS2_FOUND=true
        UDUNITS2_FROM_HPC_MODULE=true
        if [ -n "${EBROOTEXPAT:-}" ]; then
            EXPAT_LIB_DIR="$EBROOTEXPAT/lib"
        fi
    fi

    # 2. Check Spack / generic HPC root variables
    #    On Spack systems (e.g., Anvil), UDUNITS2_ROOT or UDUNITS_ROOT is set by the module
    if [ "$UDUNITS2_FOUND" = false ]; then
        for _ud_var in UDUNITS2_ROOT UDUNITS_ROOT; do
            eval "_ud_val=\${${_ud_var}:-}"
            if [ -n "$_ud_val" ] && [ -f "$_ud_val/include/udunits2.h" ]; then
                _set_udunits2_from_root "$_ud_val"
                echo "Found UDUNITS2 via ${_ud_var} at: ${UDUNITS2_DIR}"
                UDUNITS2_FOUND=true
                UDUNITS2_FROM_HPC_MODULE=true
                break
            fi
        done
    fi

    # Check conda environment (second priority)
    # CONDA_LIB_PREFIX handles Windows conda layout (Library/ subdir)
    if [ "$UDUNITS2_FOUND" = false ] && [ -n "$CONDA_PREFIX" ]; then
        local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
        if [ -f "$clp/include/udunits2.h" ]; then
            UDUNITS2_DIR="$clp"
            UDUNITS2_INCLUDE_DIR="$clp/include"
            if [ -f "$clp/lib/libudunits2.so" ]; then
                UDUNITS2_LIBRARY="$clp/lib/libudunits2.so"
            elif [ -f "$clp/lib/libudunits2.dylib" ]; then
                UDUNITS2_LIBRARY="$clp/lib/libudunits2.dylib"
            elif [ -f "$clp/lib/udunits2.lib" ]; then
                UDUNITS2_LIBRARY="$clp/lib/udunits2.lib"
            else
                UDUNITS2_LIBRARY="$clp/lib/libudunits2.a"
            fi
            EXPAT_LIB_DIR="$clp/lib"
            echo "Found conda UDUNITS2 at: ${UDUNITS2_DIR}"
            UDUNITS2_FOUND=true
        fi
    fi

    # Try pkg-config (system install)
    if [ "$UDUNITS2_FOUND" = false ] && command -v pkg-config >/dev/null 2>&1 && pkg-config --exists udunits2 2>/dev/null; then
        UDUNITS2_DIR="$(pkg-config --variable=prefix udunits2)"
        UDUNITS2_INCLUDE_DIR="$(pkg-config --variable=includedir udunits2)"
        local udunits2_libdir="$(pkg-config --variable=libdir udunits2)"
        UDUNITS2_LIBRARY="${udunits2_libdir}/libudunits2.so"
        EXPAT_LIB_DIR="${udunits2_libdir}"
        echo "Found UDUNITS2 via pkg-config at: ${UDUNITS2_DIR}"
        UDUNITS2_FOUND=true
    fi

    # Try common system locations (including multiarch lib dirs on Debian/Ubuntu)
    if [ "$UDUNITS2_FOUND" = false ]; then
        for try_path in /usr /usr/local /opt/udunits2 $HOME/.local; do
            if [ ! -f "$try_path/include/udunits2.h" ]; then
                continue
            fi
            # Search lib/, lib/<multiarch>/, and lib64/ for the library
            _ud_lib=""
            _ud_libdir=""
            for _ldir in "$try_path/lib" "$try_path/lib/$(uname -m)-linux-gnu" "$try_path/lib64"; do
                if [ -f "$_ldir/libudunits2.so" ]; then
                    _ud_lib="$_ldir/libudunits2.so"; _ud_libdir="$_ldir"; break
                elif [ -f "$_ldir/libudunits2.dylib" ]; then
                    _ud_lib="$_ldir/libudunits2.dylib"; _ud_libdir="$_ldir"; break
                elif [ -f "$_ldir/libudunits2.a" ]; then
                    _ud_lib="$_ldir/libudunits2.a"; _ud_libdir="$_ldir"; break
                fi
            done
            if [ -n "$_ud_lib" ]; then
                UDUNITS2_DIR="$try_path"
                UDUNITS2_INCLUDE_DIR="$try_path/include"
                UDUNITS2_LIBRARY="$_ud_lib"
                EXPAT_LIB_DIR="$_ud_libdir"
                echo "Found UDUNITS2 at: $try_path (lib: $_ud_libdir)"
                UDUNITS2_FOUND=true
                break
            fi
        done
    fi

    # If not found, build from source
    if [ "$UDUNITS2_FOUND" = false ]; then
        echo "UDUNITS2 not found system-wide, building from source..."

        # Save original directory before building
        UDUNITS2_ORIGINAL_DIR="$(pwd)"

        UDUNITS2_VERSION="2.2.28"
        UDUNITS2_BUILD_DIR="${UDUNITS2_ORIGINAL_DIR}/udunits2_build"
        UDUNITS2_INSTALL_DIR="${UDUNITS2_ORIGINAL_DIR}/udunits2"

        # Check if already built locally
        if [ -f "${UDUNITS2_INSTALL_DIR}/include/udunits2.h" ] && \
           ([ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so" ] || [ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.a" ]); then
            echo "Using previously built UDUNITS2 at: ${UDUNITS2_INSTALL_DIR}"
        else
            # Download and build UDUNITS2
            mkdir -p "${UDUNITS2_BUILD_DIR}"
            cd "${UDUNITS2_BUILD_DIR}"

            if [ ! -f "udunits-${UDUNITS2_VERSION}.tar.gz" ]; then
                echo "Downloading UDUNITS2 ${UDUNITS2_VERSION}..."
                wget -q "https://downloads.unidata.ucar.edu/udunits/${UDUNITS2_VERSION}/udunits-${UDUNITS2_VERSION}.tar.gz" \
                  || curl -fsSL -o "udunits-${UDUNITS2_VERSION}.tar.gz" "https://downloads.unidata.ucar.edu/udunits/${UDUNITS2_VERSION}/udunits-${UDUNITS2_VERSION}.tar.gz"
            fi

            if [ ! -d "udunits-${UDUNITS2_VERSION}" ]; then
                echo "Extracting UDUNITS2..."
                tar -xzf "udunits-${UDUNITS2_VERSION}.tar.gz"
            fi

            cd "udunits-${UDUNITS2_VERSION}"

            # UDUNITS2 requires EXPAT for XML parsing
            EXPAT_FLAGS=""
            EXPAT_FOUND=false

            # Check HPC expat module first
            if [ -n "$EBROOTEXPAT" ] && [ -f "$EBROOTEXPAT/include/expat.h" ]; then
                echo "Found HPC module EXPAT at: $EBROOTEXPAT"
                EXPAT_FLAGS="CPPFLAGS=-I$EBROOTEXPAT/include LDFLAGS=-L$EBROOTEXPAT/lib"
                EXPAT_LIB_DIR="$EBROOTEXPAT/lib"
                export LD_LIBRARY_PATH="$EBROOTEXPAT/lib:${LD_LIBRARY_PATH:-}"
                EXPAT_FOUND=true
            fi

            # Check for expat in common locations
            if [ "$EXPAT_FOUND" = false ]; then
                for expat_path in "$CONDA_PREFIX" /usr /usr/local; do
                    if [ -n "$expat_path" ] && [ -f "$expat_path/include/expat.h" ]; then
                        echo "Found EXPAT at: $expat_path"
                        EXPAT_FLAGS="CPPFLAGS=-I$expat_path/include LDFLAGS=-L$expat_path/lib"
                        EXPAT_LIB_DIR="$expat_path/lib"
                        export LD_LIBRARY_PATH="$expat_path/lib:${LD_LIBRARY_PATH:-}"
                        EXPAT_FOUND=true
                        break
                    fi
                done
            fi

            # If EXPAT not found, build it from source
            if [ "$EXPAT_FOUND" = false ]; then
                echo "EXPAT not found, building from source..."
                EXPAT_VERSION="2.5.0"
                EXPAT_INSTALL_DIR="${UDUNITS2_ORIGINAL_DIR}/expat"

                if [ ! -f "${EXPAT_INSTALL_DIR}/lib/libexpat.a" ]; then
                    mkdir -p expat_build && cd expat_build
                    if [ ! -f "expat-${EXPAT_VERSION}.tar.gz" ]; then
                        wget -q "https://github.com/libexpat/libexpat/releases/download/R_2_5_0/expat-${EXPAT_VERSION}.tar.gz" \
                          || curl -fsSL -o "expat-${EXPAT_VERSION}.tar.gz" "https://github.com/libexpat/libexpat/releases/download/R_2_5_0/expat-${EXPAT_VERSION}.tar.gz"
                    fi
                    tar -xzf "expat-${EXPAT_VERSION}.tar.gz"
                    cd "expat-${EXPAT_VERSION}"
                    ./configure --prefix="${EXPAT_INSTALL_DIR}" --disable-shared --enable-static
                    make -j ${NCORES:-4}
                    make install
                    cd ../..
                    echo "EXPAT built successfully"
                else
                    echo "Using previously built EXPAT at: ${EXPAT_INSTALL_DIR}"
                fi

                EXPAT_FLAGS="CPPFLAGS=-I${EXPAT_INSTALL_DIR}/include LDFLAGS=-L${EXPAT_INSTALL_DIR}/lib"
                EXPAT_LIB_DIR="${EXPAT_INSTALL_DIR}/lib"
                export LD_LIBRARY_PATH="${EXPAT_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
            fi

            echo "Configuring UDUNITS2..."
            ./configure --prefix="${UDUNITS2_INSTALL_DIR}" --disable-shared --enable-static $EXPAT_FLAGS

            echo "Building UDUNITS2..."
            make -j ${NCORES:-4}

            echo "Installing UDUNITS2 to ${UDUNITS2_INSTALL_DIR}..."
            make install

            # Return to original directory
            cd "${UDUNITS2_ORIGINAL_DIR}"

            echo "UDUNITS2 built successfully"
        fi

        UDUNITS2_DIR="${UDUNITS2_INSTALL_DIR}"
        UDUNITS2_INCLUDE_DIR="${UDUNITS2_INSTALL_DIR}/include"
        if [ -f "${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so" ]; then
            UDUNITS2_LIBRARY="${UDUNITS2_INSTALL_DIR}/lib/libudunits2.so"
        else
            UDUNITS2_LIBRARY="${UDUNITS2_INSTALL_DIR}/lib/libudunits2.a"
        fi
    fi

    export UDUNITS2_DIR UDUNITS2_INCLUDE_DIR UDUNITS2_LIBRARY UDUNITS2_FROM_HPC_MODULE

    # Also set CMAKE-specific variables
    export UDUNITS2_ROOT="$UDUNITS2_DIR"
    export CMAKE_PREFIX_PATH="${UDUNITS2_DIR}:${CMAKE_PREFIX_PATH:-}"

    # Export EXPAT library path for downstream builds (needed by CMake when linking -lexpat)
    # Only needed when building UDUNITS2 from source (HPC modules handle expat via rpath)
    if [ -n "$EXPAT_LIB_DIR" ] && [ -d "$EXPAT_LIB_DIR" ]; then
        export EXPAT_LIB_DIR
        export LIBRARY_PATH="${EXPAT_LIB_DIR}:${LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH="${EXPAT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
        echo "  EXPAT_LIB_DIR: ${EXPAT_LIB_DIR}"
    fi

    echo "UDUNITS2 configuration:"
    echo "  UDUNITS2_DIR: ${UDUNITS2_DIR}"
    echo "  UDUNITS2_INCLUDE_DIR: ${UDUNITS2_INCLUDE_DIR}"
    echo "  UDUNITS2_LIBRARY: ${UDUNITS2_LIBRARY}"
    echo "  UDUNITS2_FROM_HPC_MODULE: ${UDUNITS2_FROM_HPC_MODULE}"
}
detect_or_build_udunits2
    '''.strip()


def get_bison_detection_and_build() -> str:
    """
    Get reusable bison detection and build-from-source snippet.

    Checks if bison (parser generator) is available, and if not, builds it
    from source in a local directory.

    Returns:
        Shell script snippet for bison detection and building.
    """
    return r'''
# === Bison Detection and Build ===
detect_or_build_bison() {
    BISON_FOUND=false

    # Check conda environment first (highest priority)
    local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    if [ -n "$CONDA_PREFIX" ] && ([ -x "$clp/bin/bison" ] || [ -x "$clp/bin/bison.exe" ]); then
        echo "Found conda bison: $clp/bin/bison"
        "$clp/bin/bison" --version | head -1
        export PATH="$clp/bin:$PATH"
        BISON_FOUND=true
        return 0
    fi

    # Check if bison is already available in PATH
    if command -v bison >/dev/null 2>&1; then
        echo "Found bison: $(command -v bison)"
        bison --version | head -1
        BISON_FOUND=true
        return 0
    fi

    # If not found, build from source
    echo "Bison not found system-wide, building from source..."

    # Save original directory before building
    BISON_ORIGINAL_DIR="$(pwd)"

    BISON_VERSION="3.8.2"
    BISON_BUILD_DIR="${BISON_ORIGINAL_DIR}/bison_build"
    BISON_INSTALL_DIR="${BISON_ORIGINAL_DIR}/bison"

    # Check if already built locally
    if [ -x "${BISON_INSTALL_DIR}/bin/bison" ]; then
        echo "Using previously built bison at: ${BISON_INSTALL_DIR}/bin/bison"
        export PATH="${BISON_INSTALL_DIR}/bin:$PATH"
        bison --version | head -1
        return 0
    fi

    # Download and build bison
    mkdir -p "${BISON_BUILD_DIR}"
    cd "${BISON_BUILD_DIR}"

    if [ ! -f "bison-${BISON_VERSION}.tar.xz" ]; then
        echo "Downloading bison ${BISON_VERSION}..."
        wget -q "https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.xz" \
          || curl -fsSL -o "bison-${BISON_VERSION}.tar.xz" "https://ftp.gnu.org/gnu/bison/bison-${BISON_VERSION}.tar.xz"
    fi

    if [ ! -d "bison-${BISON_VERSION}" ]; then
        echo "Extracting bison..."
        tar -xJf "bison-${BISON_VERSION}.tar.xz"
    fi

    cd "bison-${BISON_VERSION}"
    echo "Configuring bison..."
    ./configure --prefix="${BISON_INSTALL_DIR}"

    echo "Building bison..."
    make -j ${NCORES:-4}

    echo "Installing bison to ${BISON_INSTALL_DIR}..."
    make install

    # Return to original directory
    cd "${BISON_ORIGINAL_DIR}"

    # Add to PATH
    export PATH="${BISON_INSTALL_DIR}/bin:$PATH"

    echo "Bison built successfully"
    bison --version | head -1
}
detect_or_build_bison
    '''.strip()


def get_flex_detection_and_build() -> str:
    """
    Get reusable flex detection and build-from-source snippet.

    Checks if flex (lexical analyzer generator) is available, and if not,
    builds it from source in a local directory.

    Returns:
        Shell script snippet for flex detection and building.
    """
    return r'''
# === Flex Detection and Build ===
detect_or_build_flex() {
    FLEX_FOUND=false
    LIBFL_FOUND=false

    # Check conda environment first (highest priority)
    local clp="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    if [ -n "$CONDA_PREFIX" ] && ([ -x "$clp/bin/flex" ] || [ -x "$clp/bin/flex.exe" ]); then
        echo "Found conda flex: $clp/bin/flex"
        "$clp/bin/flex" --version | head -1
        export PATH="$clp/bin:$PATH"
        FLEX_FOUND=true
        # Conda flex package includes libfl
        if [ -f "$clp/lib/libfl.a" ] || [ -f "$clp/lib/libfl.so" ]; then
            echo "Found conda libfl"
            LIBFL_FOUND=true
            export FLEX_LIB_DIR="$clp/lib"
            export LDFLAGS="${LDFLAGS:-} -L${FLEX_LIB_DIR}"
            export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH:-}"
        fi
        return 0
    fi

    # Prefer Homebrew flex on macOS — Apple's /usr/bin/flex calls /usr/bin/gm4
    # which was removed in macOS Sequoia (15.x)
    if [ -x /opt/homebrew/opt/flex/bin/flex ]; then
        echo "Found Homebrew flex: /opt/homebrew/opt/flex/bin/flex"
        export PATH="/opt/homebrew/opt/flex/bin:$PATH"
        flex --version | head -1
        FLEX_FOUND=true
        # Homebrew flex includes libfl
        local brew_flex_lib="/opt/homebrew/opt/flex/lib"
        if [ -f "$brew_flex_lib/libfl.a" ] || [ -f "$brew_flex_lib/libfl.dylib" ]; then
            echo "Found Homebrew libfl"
            LIBFL_FOUND=true
            export FLEX_LIB_DIR="$brew_flex_lib"
            export LDFLAGS="${LDFLAGS:-} -L${FLEX_LIB_DIR}"
            export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH:-}"
        fi
        return 0
    fi

    # Check if flex binary is available in PATH
    if command -v flex >/dev/null 2>&1; then
        echo "Found flex: $(command -v flex)"
        flex --version | head -1
        FLEX_FOUND=true

        # Check if libfl is available for linking
        # Try multiple methods to find libfl - be specific to avoid matching libflac etc.
        # Use word boundary matching with grep
        if ldconfig -p 2>/dev/null | grep -qE 'libfl\.(so|a)'; then
            echo "System libfl found via ldconfig"
            LIBFL_FOUND=true
        fi

        # Check common system library paths
        if [ "$LIBFL_FOUND" != "true" ]; then
            for libdir in /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /lib64 /lib; do
                if [ -f "$libdir/libfl.a" ] || [ -f "$libdir/libfl.so" ]; then
                    echo "System libfl found in: $libdir"
                    LIBFL_FOUND=true
                    break
                fi
            done
        fi

        if [ "$LIBFL_FOUND" = "true" ]; then
            return 0
        else
            echo "Warning: flex found but libfl not found - will build flex from source for the library"
        fi
    fi

    # If flex or libfl not found, build from source
    if [ "$FLEX_FOUND" = "true" ]; then
        echo "Building flex from source to get libfl library..."
    else
        echo "Flex not found system-wide, building from source..."
    fi

    # Save original directory before building
    FLEX_ORIGINAL_DIR="$(pwd)"

    FLEX_VERSION="2.6.4"
    FLEX_BUILD_DIR="${FLEX_ORIGINAL_DIR}/flex_build"
    FLEX_INSTALL_DIR="${FLEX_ORIGINAL_DIR}/flex"

    # Check if already built locally
    if [ -x "${FLEX_INSTALL_DIR}/bin/flex" ]; then
        echo "Using previously built flex at: ${FLEX_INSTALL_DIR}/bin/flex"
        export PATH="${FLEX_INSTALL_DIR}/bin:$PATH"
        # Export library path for linking (LIBRARY_PATH is used by gcc at link time)
        export FLEX_LIB_DIR="${FLEX_INSTALL_DIR}/lib"
        export LDFLAGS="${LDFLAGS} -L${FLEX_LIB_DIR}"
        export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${FLEX_LIB_DIR}:${LD_LIBRARY_PATH}"
        echo "FLEX_LIB_DIR set to: ${FLEX_LIB_DIR}"
        flex --version | head -1
        return 0
    fi

    # Download and build flex
    mkdir -p "${FLEX_BUILD_DIR}"
    cd "${FLEX_BUILD_DIR}"

    if [ ! -f "flex-${FLEX_VERSION}.tar.gz" ]; then
        echo "Downloading flex ${FLEX_VERSION}..."
        wget -q "https://github.com/westes/flex/releases/download/v${FLEX_VERSION}/flex-${FLEX_VERSION}.tar.gz" \
          || curl -fsSL -o "flex-${FLEX_VERSION}.tar.gz" "https://github.com/westes/flex/releases/download/v${FLEX_VERSION}/flex-${FLEX_VERSION}.tar.gz"
    fi

    if [ ! -d "flex-${FLEX_VERSION}" ]; then
        echo "Extracting flex..."
        tar -xzf "flex-${FLEX_VERSION}.tar.gz"
    fi

    cd "flex-${FLEX_VERSION}"
    echo "Configuring flex..."
    # GCC 14+ treats implicit function declarations as errors.
    # flex 2.6.4 misc.c calls reallocarray() which needs _GNU_SOURCE on glibc
    # to be properly declared via <stdlib.h>.
    CFLAGS="${CFLAGS:-} -D_GNU_SOURCE" ./configure --prefix="${FLEX_INSTALL_DIR}"

    echo "Building flex..."
    make -j ${NCORES:-4} CFLAGS="${CFLAGS:-} -D_GNU_SOURCE"

    echo "Installing flex to ${FLEX_INSTALL_DIR}..."
    make install

    # Return to original directory
    cd "${FLEX_ORIGINAL_DIR}"

    # Add to PATH
    export PATH="${FLEX_INSTALL_DIR}/bin:$PATH"
    # Export library path for linking (LIBRARY_PATH is used by gcc at link time)
    export FLEX_LIB_DIR="${FLEX_INSTALL_DIR}/lib"
    export LDFLAGS="${LDFLAGS} -L${FLEX_LIB_DIR}"
    export LIBRARY_PATH="${FLEX_LIB_DIR}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${FLEX_LIB_DIR}:${LD_LIBRARY_PATH}"
    echo "FLEX_LIB_DIR set to: ${FLEX_LIB_DIR}"

    echo "Flex built successfully"
    flex --version | head -1
}
detect_or_build_flex
    '''.strip()


def get_all_snippets() -> Dict[str, str]:
    """
    Return all snippets as a dictionary for easy access.

    Returns:
        Dictionary mapping snippet names to their shell script content.
    """
    return {
        'common_env': get_common_build_environment(),
        'netcdf_detect': get_netcdf_detection(),
        'hdf5_detect': get_hdf5_detection(),
        'netcdf_lib_detect': get_netcdf_lib_detection(),
        'geos_proj_detect': get_geos_proj_detection(),
        'udunits2_detect_build': get_udunits2_detection_and_build(),
        'bison_detect_build': get_bison_detection_and_build(),
        'flex_detect_build': get_flex_detection_and_build(),
    }
