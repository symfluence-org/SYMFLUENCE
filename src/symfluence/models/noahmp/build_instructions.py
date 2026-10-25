# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Noah-MP (noah-owp-modular) build instructions for SYMFLUENCE."""
from __future__ import annotations

from symfluence.core.registries import R


@R.build_instructions.add('noahmp')
def get_noahmp_build_instructions():
    """Get Noah-MP standalone install instructions.

    noah-owp-modular is a Fortran model requiring a Fortran compiler
    and NetCDF libraries.  Build produces noah_owp_modular.exe in run/.
    """
    return {
        'description': 'Noah-MP Standalone Land Surface Model (NOAA-OWP)',
        'config_path_key': 'NOAHMP_INSTALL_PATH',
        'config_exe_key': 'NOAHMP_EXE',
        'default_path_suffix': 'installs/noah-owp-modular',
        'default_exe': 'noah_owp_modular.exe',
        'repository': 'https://github.com/NOAA-OWP/noah-owp-modular.git',
        'branch': 'main',
        'install_dir': 'noah-owp-modular',
        'build_commands': [
            r"""
# Noah-MP (noah-owp-modular) Install Script for SYMFLUENCE
set -e
echo "=== Noah-MP Installation Starting ==="

# Check for Fortran compiler
if ! command -v gfortran &>/dev/null; then
    echo "ERROR: gfortran not found. Install a Fortran compiler first."
    echo "  macOS:  brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    exit 1
fi

# Check for NetCDF
if ! command -v nf-config &>/dev/null && ! command -v nc-config &>/dev/null; then
    echo "ERROR: NetCDF-Fortran not found."
    echo "  macOS:  brew install netcdf netcdf-fortran"
    echo "  Ubuntu: sudo apt-get install libnetcdf-dev libnetcdff-dev"
    exit 1
fi

# Configure: pick the gfortran build config for the platform. (NB: the Linux
# config must be gfortran.linux — NOT pgf90.linux, which targets the PGI/NVIDIA
# compiler and breaks under gfortran.)
if [ "$(uname)" = "Darwin" ]; then
    cp config/user_build_options.macos.gfortran user_build_options 2>/dev/null \
        || cp config/user_build_options.bigsur.gfortran user_build_options 2>/dev/null \
        || { echo "WARNING: No macOS gfortran config found, trying ./configure"; ./configure 2>/dev/null || true; }
elif [ -f "config/user_build_options.gfortran.linux" ]; then
    cp config/user_build_options.gfortran.linux user_build_options
else
    ./configure 2>/dev/null || true
fi

# The shipped configs hardcode NetCDF (and sometimes compiler) paths for one
# specific machine. Rewrite the NetCDF lines from nf-config/nc-config so the
# build finds NetCDF wherever it actually lives: Debian/Ubuntu multiarch
# (/usr/lib/<arch>-linux-gnu, which ${NETCDF}/lib misses) and Homebrew's split
# netcdf / netcdf-fortran prefixes on macOS. Also normalise the compiler to the
# one on PATH (the macOS config hardcodes an Intel-Homebrew gfortran path).
if [ -f user_build_options ] && command -v nf-config >/dev/null 2>&1; then
    _NF_INC="$(nf-config --includedir 2>/dev/null)"
    _NF_LIBS="$(nf-config --flibs 2>/dev/null) $(nc-config --libs 2>/dev/null)"
    # Pass the (possibly '@'-containing) NetCDF flags as an argument (shifted
    # off @ARGV in BEGIN) to a single-quoted perl program. Interpolating them
    # into a double-quoted replacement would treat an '@' — e.g. in a
    # conda/nf-config prefix under a Google Drive mount
    # (GoogleDrive-user@gmail.com) — as an array interpolation and silently
    # delete it. Not via the environment either: on Windows, Git for Windows'
    # perl started from MSYS2's bash drops per-command env vars.
    perl -pi -e 'BEGIN { $v = shift } s|^\s*NETCDFMOD\s*=.*|NETCDFMOD = $v|' -- "-I${_NF_INC}" user_build_options
    perl -pi -e 'BEGIN { $v = shift } s|^\s*NETCDFLIB\s*=.*|NETCDFLIB = $v|' -- "${_NF_LIBS}" user_build_options
    perl -pi -e "s|^\s*COMPILERF90\s*=.*|COMPILERF90 = gfortran|" user_build_options
fi

make clean 2>/dev/null || true
make

if [ -f "run/noah_owp_modular.exe" ]; then
    echo "=== Noah-MP Installation Complete ==="
    echo "Executable: run/noah_owp_modular.exe"
else
    echo "ERROR: Build failed — noah_owp_modular.exe not found in run/"
    exit 1
fi
            """.strip()
        ],
        'dependencies': ['gfortran', 'netcdf-fortran'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['run/noah_owp_modular.exe'],
            'check_type': 'exists',
        },
        'order': 24,
        'optional': True,
    }
