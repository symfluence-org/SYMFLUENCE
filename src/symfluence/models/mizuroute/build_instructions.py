# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mizuRoute build instructions for SYMFLUENCE.

This module defines how to build mizuRoute from source, including:
- Repository and branch information
- Build commands (shell scripts)
- Installation verification criteria

mizuRoute is a river network routing model developed at NCAR.
"""
from __future__ import annotations

from symfluence.core.build import (
    get_common_build_environment,
    get_netcdf_detection,
    get_safe_build_path,
)
from symfluence.core.registries import R


@R.build_instructions.add('mizuroute')
def get_mizuroute_build_instructions():
    """
    Get mizuRoute build instructions.

    mizuRoute requires NetCDF libraries. The build uses make and
    requires editing the Makefile directly.

    Returns:
        Dictionary with complete build configuration for mizuRoute.
    """
    common_env = get_common_build_environment()
    netcdf_detect = get_netcdf_detection()
    safe_build_path = get_safe_build_path()

    return {
        'description': 'Mizukami routing model for river network routing',
        'config_path_key': 'INSTALL_PATH_MIZUROUTE',
        'config_exe_key': 'EXE_NAME_MIZUROUTE',
        'default_path_suffix': 'installs/mizuRoute/route/bin',
        'default_exe': 'mizuRoute.exe',
        'repository': 'https://github.com/ESCOMP/mizuRoute.git',
        'branch': 'serial',
        'install_dir': 'mizuRoute',
        'build_commands': [
            common_env,
            netcdf_detect,
            # Relocate onto a Make/shell-safe path (spaces / '@' in the install
            # path, e.g. Google Drive) BEFORE entering the build directory.
            safe_build_path,
            r'''
# Build mizuRoute - edit Makefile directly (it doesn't use env vars)
cd route/build
mkdir -p ../bin

F_MASTER_PATH="$(cd .. && pwd)"
echo "F_MASTER: $F_MASTER_PATH/"

# Ensure NetCDF paths are set, preferring conda installation
# CONDA_LIB_PREFIX is set by the common build environment snippet
# and handles Windows conda layout (Library/ subdir).
if [ -n "$CONDA_PREFIX" ] && [ -f "${CONDA_LIB_PREFIX:-$CONDA_PREFIX}/bin/nf-config" ]; then
    export NETCDF_FORTRAN="${CONDA_LIB_PREFIX:-$CONDA_PREFIX}"
    export NETCDF_C="${NETCDF_C:-${CONDA_LIB_PREFIX:-$CONDA_PREFIX}}"
    echo "Using conda NetCDF-Fortran at: $NETCDF_FORTRAN"
fi

# Validate NetCDF was detected
if [ -z "${NETCDF_FORTRAN}" ] || [ ! -d "${NETCDF_FORTRAN}/include" ]; then
    echo "ERROR: Could not find NetCDF installation"
    echo "NETCDF_FORTRAN=${NETCDF_FORTRAN}"
    echo "CONDA_PREFIX=${CONDA_PREFIX:-not set}"
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Contents of CONDA_PREFIX/include:"
        ls -la "$CONDA_PREFIX/include" 2>/dev/null | head -10 || true
        echo "Contents of CONDA_PREFIX/bin/nf-config:"
        ls -la "$CONDA_PREFIX/bin/nf-config" 2>/dev/null || echo "nf-config not found"
    fi
    exit 1
fi

# On Windows (MSYS2/MinGW), convert backslash paths to forward slashes.
# Perl interprets \U, \e, \c etc. as escape sequences in replacement strings,
# corrupting Windows paths like C:\Users\... into C:SERS...
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        NETCDF_FORTRAN="${NETCDF_FORTRAN//\\//}"
        NETCDF_C="${NETCDF_C//\\//}"
        ;;
esac

# Edit the Makefile in-place
echo "=== Configuring Makefile ==="
perl -i -pe "s|^FC\s*=.*$|FC = gnu|" Makefile
perl -i -pe "s|^FC_EXE\s*=.*$|FC_EXE = ${FC:-gfortran}|" Makefile
perl -i -pe "s|^EXE\s*=.*$|EXE = mizuRoute.exe|" Makefile
# Pass the path via the environment and read it with $ENV{...} in a
# single-quoted perl program.  Interpolating the path directly into a
# double-quoted perl replacement string treats an '@' (e.g. in a Google
# Drive path like GoogleDrive-user@gmail.com) as an array interpolation
# and silently deletes it.  $ENV{...} is a plain value lookup, immune to that.
SYMF_F_MASTER="$F_MASTER_PATH/" perl -i -pe 's|^F_MASTER\s*=.*$|F_MASTER = $ENV{SYMF_F_MASTER}|' Makefile
perl -i -pe "s|^\s*NCDF_PATH\s*=.*$|NCDF_PATH = ${NETCDF_FORTRAN}|" Makefile
perl -i -pe "s|^isOpenMP\s*=.*$|isOpenMP = ${MIZUROUTE_OPENMP:-no}|" Makefile

# Fix LIBNETCDF for separate C/Fortran libs (e.g., macOS Homebrew, HPC with separate installs)
# Note: LIBNETCDF in mizuRoute is a multi-line definition with backslash continuation,
# so we must also remove the orphaned continuation line after replacement.
if [ "${NETCDF_C}" != "${NETCDF_FORTRAN}" ]; then
    echo "Fixing LIBNETCDF for separate C/Fortran paths"
    perl -i -pe "s|^LIBNETCDF\s*=.*$|LIBNETCDF = -L${NETCDF_FORTRAN}/lib -lnetcdff -L${NETCDF_C}/lib -lnetcdf|" Makefile
    # Remove the orphaned continuation line (starts with spaces, contains -L and NCDF_PATH)
    # Use \t-aware pattern: only match lines starting with spaces (not tabs, which are make recipes)
    perl -i -ne "print unless /^ +-L.*NCDF_PATH/" Makefile
fi

# Embed RPATH so the binary finds its libraries without LD_LIBRARY_PATH.
# Collect unique library directories from NetCDF, HDF5, and LD_LIBRARY_PATH.
# Skip entirely on Windows: PE binaries resolve DLLs via PATH/exe-dir, not
# rpath, and feeding Windows backslash paths (e.g. a conda
# C:\Miniconda\...\Library\lib) through the perl substitution below mangles them
# (\t -> tab, \L -> lowercase), producing bogus -L flags like "est/Library/lib"
# that break the link.
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        echo "Windows: skipping RPATH embedding (DLLs resolved via PATH)"
        ;;
    *)
MIZU_RPATH=""
_mizu_add_rpath() {
    local d="$1"
    if [ -d "$d" ] && echo ":${MIZU_RPATH}:" | grep -qv ":${d}:"; then
        MIZU_RPATH="${MIZU_RPATH:+${MIZU_RPATH}:}${d}"
    fi
}
for _nr in "${NETCDF_FORTRAN:-}" "${NETCDF_C:-}"; do
    [ -n "$_nr" ] && _mizu_add_rpath "$_nr/lib" && _mizu_add_rpath "$_nr/lib64"
done
[ -n "${HDF5_ROOT:-}" ] && _mizu_add_rpath "$HDF5_ROOT/lib" && _mizu_add_rpath "$HDF5_ROOT/lib64"
[ -n "${CONDA_PREFIX:-}" ] && _mizu_add_rpath "${CONDA_LIB_PREFIX:-$CONDA_PREFIX}/lib"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    IFS=':' read -ra _ldp <<< "$LD_LIBRARY_PATH"
    for _d in "${_ldp[@]}"; do [ -n "$_d" ] && _mizu_add_rpath "$_d"; done
fi
if [ -n "$MIZU_RPATH" ]; then
    # Convert colon-separated to -Wl,-rpath,dir flags
    RPATH_FLAGS=""
    IFS=':' read -ra _rps <<< "$MIZU_RPATH"
    for _rp in "${_rps[@]}"; do
        RPATH_FLAGS="$RPATH_FLAGS -Wl,-rpath,$_rp"
    done
    # Collapse multi-line LIBNETCDF into a single line first, then append RPATH.
    # The upstream Makefile uses backslash continuation:
    #   LIBNETCDF = ... \
    #               -L... -lnetcdff -lnetcdf
    # Appending to the first line would break the continuation (\ mid-line).
    perl -i -0777 -pe 's/(LIBNETCDF\s*=.*?)\s*\\\n[ \t]*/$1 /g' Makefile
    # Now safely append rpath flags to the single-line LIBNETCDF
    perl -i -pe "s|^(LIBNETCDF\s*=.*)$|\$1 $RPATH_FLAGS|" Makefile
    echo "RPATH: $MIZU_RPATH"
fi
        ;;
esac

# Build
# SYMF_MAKE_TMP (from the common build environment) passes a native
# Windows temp dir to make's children on the command line — env vars do
# not survive the Git-bash -> MSYS2-make runtime hop, and without a
# usable TMPDIR gfortran fails with "Cannot create temporary file in
# C:\Windows\".  Empty (expands to nothing) on non-Windows.
# F_MASTER also goes on the command line (as FUSE does): on the Windows
# release runner the perl $ENV{} edit above leaves it empty, and make then
# looks for the sources at a relative build/src/.
make clean || true
echo "Building mizuRoute..."
make "${SYMF_MAKE_TMP[@]}" F_MASTER="$F_MASTER_PATH/" 2>&1 | tee build.log || true

if [ -f "../bin/mizuRoute.exe" ]; then
    echo "Build successful - executable at ../bin/mizuRoute.exe"
else
    echo "ERROR: Executable not found at ../bin/mizuRoute.exe"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': [],
        'test_command': None,
        'verify_install': {
            'file_paths': ['route/bin/mizuRoute.exe'],
            'check_type': 'exists'
        },
        'order': 3
    }
