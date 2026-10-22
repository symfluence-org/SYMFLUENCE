# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
WATFLOOD/CHARM build instructions for SYMFLUENCE.

WATFLOOD (CHARM — Canadian Hydrological And Routing Model) is a distributed
flood forecasting model developed at the University of Waterloo.

Tier 1: Build native binary from source (kasra-keshavarz/watflood fork with
        complete source including area_watflood module).
Tier 2: Fall back to pre-compiled Windows binary via Wine.
Tier 3: Manual installation guidance.
"""
from __future__ import annotations

from symfluence.core.build import get_common_build_environment
from symfluence.core.registries import R


@R.build_instructions.add('watflood')
def get_watflood_build_instructions():
    """Get WATFLOOD/CHARM build instructions (native CMake build)."""
    common_env = get_common_build_environment()

    return {
        'description': 'WATFLOOD/CHARM distributed flood forecasting model',
        'config_path_key': 'WATFLOOD_INSTALL_PATH',
        'config_exe_key': 'WATFLOOD_EXE',
        'default_path_suffix': 'installs/watflood/bin',
        'default_exe': 'watflood',
        'repository': 'https://github.com/kasra-keshavarz/watflood.git',
        'branch': 'main',
        'install_dir': 'watflood',
        'build_commands': [
            common_env,
            r'''
set -e

echo "=== WATFLOOD/CHARM Installation Starting ==="

INSTALL_DIR="${INSTALL_DIR:-.}"
mkdir -p "${INSTALL_DIR}/bin"
INSTALL_DIR="$(cd "${INSTALL_DIR}" && pwd)"
BIN_DIR="${INSTALL_DIR}/bin"
SRC_DIR="${INSTALL_DIR}/src"

# Resolve a Python interpreter for the in-build source patches below. On Windows
# the build runs under Git Bash, which ships no 'python3' — without this the
# EF_Module.f eof patches silently no-op ("eof injection failed (non-fatal)").
PYTHON="$(command -v python3 || command -v python || command -v py || echo python3)"
echo "Using Python for source patches: $PYTHON"

# === Check for existing native binary ===
if [ -f "${BIN_DIR}/watflood" ] || [ -f "${BIN_DIR}/watflood.exe" ]; then
    _existing="${BIN_DIR}/watflood"; [ -f "${BIN_DIR}/watflood.exe" ] && _existing="${BIN_DIR}/watflood.exe"
    file_output=$(file "${_existing}" 2>/dev/null || echo "unknown")
    if echo "$file_output" | grep -qiE "mach-o|elf|executable|PE32"; then
        echo "Found existing native binary: ${_existing}"
        echo "$file_output"
        echo "=== WATFLOOD Installation Complete (existing) ==="
        exit 0
    fi
fi

# === Windows: install the official Waterloo CHARM prebuilt (Tier 1) ===
# WATFLOOD/CHARM is a Windows-native model (Intel Fortran), so its source is
# riddled with Intel extensions gfortran rejects — cross-compiling it is a
# losing battle. Instead fetch the official University of Waterloo binary
# (CHARM64x.exe, GPL-3.0; source at github.com/watflood/model) plus the runtime
# DLLs Waterloo ships for it. Linux/macOS have no official binary and fall
# through to the source build below.
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "Windows detected — installing official Waterloo CHARM prebuilt..."
        _WF_BASE="http://www.civil.uwaterloo.ca/watflood/downloads"
        _WF_TMP="$(mktemp -d)"
        _wf_ok=1
        for _z in charm64x.zip netCDF_dll.zip; do
            if curl -fsSL --retry 3 -o "${_WF_TMP}/${_z}" "${_WF_BASE}/${_z}"; then
                ( cd "${_WF_TMP}" && unzip -oq "${_z}" ) || { echo "ERROR: unzip ${_z} failed"; _wf_ok=0; }
            else
                echo "ERROR: failed to download ${_z}"; _wf_ok=0
            fi
        done
        _charm="$(find "${_WF_TMP}" -maxdepth 2 -iname 'charm*.exe' -type f | head -1)"
        if [ "${_wf_ok}" -eq 1 ] && [ -n "${_charm}" ]; then
            mkdir -p "${BIN_DIR}"
            cp "${_charm}" "${BIN_DIR}/watflood.exe"
            # Windows resolves a program's DLLs from its own directory, so drop
            # the netCDF/HDF5/zlib + Intel/Cygwin runtime DLLs next to the exe.
            find "${_WF_TMP}" -maxdepth 2 -iname '*.dll' -type f -exec cp {} "${BIN_DIR}/" \; 2>/dev/null || true
            echo "Installed CHARM prebuilt: ${BIN_DIR}/watflood.exe (+ $(ls "${BIN_DIR}"/*.dll 2>/dev/null | wc -l | tr -d ' ') DLLs)"
            # GPL-3.0 compliance: ship a notice naming the binary's source so the
            # staging step bundles it under LICENSES/ alongside the .exe.
            cat > "${INSTALL_DIR}/LICENSE" <<'WFLIC'
WATFLOOD / CHARM (Canadian Hydrological And Routing Model)
Copyright (C) 1986-2025 N. Kouwen, University of Waterloo
Licensed under the GNU General Public License, version 3 (GPL-3.0).

This release bundles the official pre-compiled Windows binary (CHARM64x.exe)
distributed at http://www.civil.uwaterloo.ca/watflood/downloads/ .
Corresponding source code: https://github.com/watflood/model
WFLIC
            rm -rf "${_WF_TMP}"
            echo "=== WATFLOOD Installation Complete (Waterloo prebuilt) ==="
            exit 0
        fi
        echo "WARNING: CHARM prebuilt unavailable; falling back to source build"
        rm -rf "${_WF_TMP}"
        ;;
esac

# === Check for gfortran ===
if ! command -v gfortran >/dev/null 2>&1; then
    echo "ERROR: gfortran not found. Install with:"
    echo "  macOS:  brew install gcc"
    echo "  Ubuntu: sudo apt-get install gfortran"
    echo "  HPC:    module load gcc"
    exit 1
fi

# === Check for cmake ===
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake not found. Install with:"
    echo "  macOS:  brew install cmake"
    echo "  Ubuntu: sudo apt-get install cmake"
    echo "  HPC:    module load cmake"
    exit 1
fi

echo "Using gfortran: $(gfortran --version | head -1)"
echo "Using cmake: $(cmake --version | head -1)"

# === Check source exists (should be cloned by registry) ===
if [ ! -f "${SRC_DIR}/core/area_watflood.f" ]; then
    echo "Source not found at ${SRC_DIR}/core/area_watflood.f"
    echo "Checking if repository was cloned..."
    if [ ! -d "${SRC_DIR}" ]; then
        echo "ERROR: Source directory not found. Repository may not have been cloned."
        exit 1
    fi
fi

# === Apply source patches for GNU Fortran compatibility ===
echo "Applying GNU Fortran compatibility patches..."

# Normalize DOS/CRLF line endings FIRST. The upstream source ships with CRLF,
# and on Linux/macOS gfortran reads the trailing CR as part of the line — an
# invalid character that cascades into "Invalid character in name" / "Expecting
# END IF" / "Unclassifiable statement" in several common/ files (read_r2c.f,
# read_pt2.f, ...). Strip it before any other patch runs.
find "${SRC_DIR}" \( -name "*.f" -o -name "*.f90" -o -name "*.for" \) -exec perl -pi -e 's/\r//g' {} +

# Fix Intel <variable> format descriptors (not supported by gfortran). Allow
# arithmetic inside the brackets too, e.g. "<ncols-1>" in read_pt2.f, not just
# bare identifiers.
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" -o -name "*.for" | while read -r f; do
    perl -pi -e 's/<[A-Za-z0-9_][A-Za-z0-9_ +\-*\/()]*>/999/g' "$f"
done

# Fix Hollerith constant in area_watflood.f: flen='none' -> flen=999999
perl -pi -e "s/flen='none'/flen=999999/" "${SRC_DIR}/core/area_watflood.f"

# read_divert.f allocates/uses xtake/ytake/xgive/ygive (real diversion
# coordinates), but the source never declares them under implicit none — only
# the integer grid-index versions itake/jtake/igive/jgive exist (a latent bug
# present in the official source too). Declare the missing real arrays next to
# them so they're module-shared.
if ! grep -qi "allocatable :: xtake" "${SRC_DIR}/core/area_watflood.f" 2>/dev/null; then
    perl -pi -e 's/(integer\*4,\s*dimension\(:\),\s*allocatable\s*::\s*itake,jtake,igive,jgive)/$1\n      real*4, dimension(:), allocatable :: xtake,ytake,xgive,ygive\n      integer*4, dimension(:), allocatable :: gridtake,gridgive/i' "${SRC_DIR}/core/area_watflood.f"
fi

# lake_evaporation is allocated in sub.f but never declared (like the diversion
# arrays). It's a real array; add it to the real allocatable block next to its
# sibling evap_convert.
if ! grep -qi "lake_evaporation" "${SRC_DIR}/core/area_watflood.f" 2>/dev/null; then
    perl -pi -e 's/\bevap_convert,/evap_convert,lake_evaporation,/i' "${SRC_DIR}/core/area_watflood.f"
fi

# Fix STOP without a blank before its operand: STOP'msg' -> STOP 'msg', and the
# line-continued form STOP& -> STOP & (gfortran: "Blank required in STOP statement").
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" -o -name "*.for" | while read -r f; do
    perl -pi -e "s/STOP\s*'/STOP '/gi" "$f"
    perl -pi -e 's/\bSTOP&/STOP &/gi' "$f"
done

# Fix .eq./.ne. with logical operands -> .eqv./.neqv. NB: the .true. replacement
# must emit ".eqv. .true." — the previous "\.eqv\.\.\s*true\." injected a literal
# "s*" (\s is not a regex in a replacement), producing ".eqv..s*true." and a
# "Syntax error in expression" once the build reached read_flow_ef.f.
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" -o -name "*.for" | while read -r f; do
    perl -pi -e 's/\.eq\.\s*\.true\./.eqv. .true./gi' "$f"
    perl -pi -e 's/\.eq\.\s*\.false\./.eqv. .false./gi' "$f"
    perl -pi -e 's/==\s*\.false\./.eqv. .false./gi' "$f"
    perl -pi -e 's/==\s*\.true\./.eqv. .true./gi' "$f"
    # parenthesised forms, e.g. read_shed_hype.f90: ".eq.(.true.)"
    perl -pi -e 's/\.eq\.\s*\(\s*\.true\.\s*\)/.eqv.(.true.)/gi' "$f"
    perl -pi -e 's/\.eq\.\s*\(\s*\.false\.\s*\)/.eqv.(.false.)/gi' "$f"
done

# Fix integer-as-logical: if(dds_flag)then where dds_flag is integer*4. It
# appears in several built files (model/dds_code.f, common/read_evt.f, ...), so
# sweep every compiled Fortran source. Only the exact if(dds_flag)then form is
# rewritten — the genuine logicals 'dds' and 'iopt99' are deliberately untouched.
find "${SRC_DIR}/common" "${SRC_DIR}/model" -type f \( -name '*.f' -o -name '*.f90' -o -name '*.F90' -o -name '*.for' \) 2>/dev/null \
    | while read -r f; do
        perl -pi -e 's/if\s*\(\s*dds_flag\s*\)\s*then/if(dds_flag.ne.0)then/gi' "$f"
    done

# Fix character(1) firstpass used as logical in routing subroutines
for f in "${SRC_DIR}/model/rerout.f" "${SRC_DIR}/model/rules_sl.f" "${SRC_DIR}/model/rules_tl.f90"; do
    if [ -f "$f" ]; then
        perl -pi -e "s/if\(firstpass\)then/if(firstpass.eq.'y')then/g" "$f"
    fi
done

# Fix route.f: firstpass.and. -> firstpass.eq.'y'.and.
perl -pi -e "s/if\(firstpass\.and\./if(firstpass.eq.'y'.and./" "${SRC_DIR}/model/route.f" 2>/dev/null || true

# Fix sub.f90/sub.f: ssmc_firstpass=.false. -> ='n' (character(1) variable)
for f in "${SRC_DIR}/model/sub.f90" "${SRC_DIR}/model/sub.f"; do
    if [ -f "$f" ]; then
        perl -pi -e "s/ssmc_firstpass=\.false\./ssmc_firstpass='n'/" "$f"
        perl -pi -e "s/if\(\.not\.netCDFflg\.or\.dds\.eq\.0\.or\.iopt99\)then/if(.not.netCDFflg.or..not.dds.or.iopt99)then/" "$f"
        # sub.f also "use area_watflood" yet re-declares npick and
        # wfo_spec_version_number locally — gfortran flags the use-association
        # conflict and rejects the whole declaration statement, cascading into
        # ~60 "no IMPLICIT type" errors. They're declared-but-unused locals, so
        # drop them from the fixed-form continuation lines ('*'/'&' in col 6 —
        # never executable code) and let them come from the module. Use [ \t]
        # (not \s, which would eat the newline and merge continuation lines);
        # drop a continuation line left empty after the var is removed.
        perl -ni -e 'if (/^\s{0,5}[*&]/) { s/\b(?:npick|wfo_spec_version_number)\b[ \t]*,[ \t]*//gi; s/,[ \t]*\b(?:npick|wfo_spec_version_number)\b//gi; next if /^\s{0,5}[*&][ \t]*$/; } print;' "$f"
    fi
done

# Fix Intel GETARG (3-arg) -> GNU GETARG (2-arg) in CHARM.f90
if [ -f "${SRC_DIR}/model/CHARM.f90" ]; then
    perl -pi -e "s/CALL GETARG\(1, buf, status1\)/IF(IARGC().GE.1)THEN\n        CALL GETARG(1, buf)\n      ELSE\n        buf=' '\n      ENDIF/" "${SRC_DIR}/model/CHARM.f90"
    perl -pi -e 's/if\(status1\.ne\.1\)buf=.*//' "${SRC_DIR}/model/CHARM.f90"
fi

# Fix stray trailing quotes exposed by -ffixed-line-length-none. Strip a trailing
# quote ONLY when the line has an odd number of quotes (so the last one is an
# unmatched stray, e.g. lst.f's "write(...) '...Various   '   '"). A position-only
# rule wrongly strips the legitimate closing quote of long even-quote strings.
find "${SRC_DIR}" \( -name "write_tb0.f" -o -name "write_diversion_tb0.f" -o -name "lst.f" \) | while read -r f; do
    perl -pi -e "if ((tr/'//) % 2 == 1) { s/'(\s*)\$/\$1/ }" "$f"
done

# Fix integer-as-logical: if(inbsnflg(idx))then where inbsnflg is integer*4
# (appears in dds_code.f, read_resv_ef.f, synflw.f, lst.f, tracer4.f, ...).
find "${SRC_DIR}/common" "${SRC_DIR}/model" -type f \( -name '*.f' -o -name '*.f90' -o -name '*.F90' -o -name '*.for' \) 2>/dev/null \
    | while read -r f; do
        perl -pi -e 's/if\(\s*inbsnflg\(([^)]*)\)\s*\)then/if(inbsnflg($1).ne.0)then/gi' "$f"
    done

# lake_ice.f declares firstpass as LOGICAL (not the character(1) it is elsewhere),
# so its source if(firstpass.eq.'y') compares logical to character. Use it as the
# logical it is here.
if [ -f "${SRC_DIR}/model/lake_ice.f" ]; then
    perl -pi -e "s/if\(firstpass\.eq\.'y'\)then/if(firstpass)then/g" "${SRC_DIR}/model/lake_ice.f"
fi

# 'dds' is a LOGICAL in area_watflood, but several sources compare it to 0/1
# (Intel let you treat logicals as integers): dds.eq.0 -> .not.dds, dds.ne.0 -> dds.
find "${SRC_DIR}/common" "${SRC_DIR}/model" -type f \( -name '*.f' -o -name '*.f90' -o -name '*.F90' -o -name '*.for' \) 2>/dev/null \
    | while read -r f; do
        perl -pi -e 's/\bdds\.eq\.0\b/.not.dds/gi; s/\bdds\.ne\.0\b/dds/gi' "$f"
    done

# Fix EF_Module.f: CountDataLinesAfterHeader is placed after END MODULE
# which confuses gfortran.  Extract it into a separate source file instead
# of trying to squeeze it into CONTAINS (which causes host-association errors).
EF_MOD="${SRC_DIR}/common/EF_Module.f"
if [ -f "$EF_MOD" ]; then
    "$PYTHON" -c "
import re
with open('$EF_MOD') as f:
    txt = f.read()
if 'CountDataLinesAfterHeader' not in txt:
    print('  EF_Module.f: no CountDataLinesAfterHeader found')
else:
    # Case 1: function is after END MODULE -> extract to separate file
    m = re.search(r'^(\s*END\s+MODULE\s+\w+\s*\n)(.*)', txt, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if m:
        end_mod = m.group(1)
        after = m.group(2).strip()
        if after and 'CountDataLinesAfterHeader' in after:
            with open('${SRC_DIR}/common/EF_Extra.f90', 'w') as ef:
                ef.write('! Extracted from EF_Module.f for gfortran compatibility\n')
                ef.write(after + '\n')
            txt = txt[:m.start()] + end_mod
            with open('$EF_MOD', 'w') as f:
                f.write(txt)
            print('  EF_Module.f: extracted trailing function to EF_Extra.f90')
        elif 'CONTAINS' in txt.upper():
            # Case 2: function is inside CONTAINS section — valid Fortran,
            # no extraction needed.  It will see eof via host association.
            print('  EF_Module.f: CountDataLinesAfterHeader is in CONTAINS (valid)')
        else:
            print('  EF_Module.f: no trailing code after END MODULE')
    elif 'CONTAINS' in txt.upper():
        print('  EF_Module.f: CountDataLinesAfterHeader is in CONTAINS (valid)')
    else:
        print('  EF_Module.f: no END MODULE found')
" 2>/dev/null || echo "  WARNING: EF_Module.f patch failed (non-fatal)"
fi

# Fix WRITE statements with misquoted format strings
# e.g. write(un,3020)':SourceFile         ',source_file_name
# The colon inside the string confuses some compilers.  Ensure proper quoting.
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" -o -name "*.for" | while read -r f; do
    perl -pi -e "s/write\(([^)]+)\)'/write(\$1) '/gi" "$f"
done

# Fix concatenated WRITE statements in write_diversion_tb0.f / write_tb0.f:
# Two WRITE statements are run together on one line after source_file_name.
# Split them so each WRITE is on its own line (continuation-safe).
find "${SRC_DIR}" -name "write_diversion_tb0.f" -o -name "write_tb0.f" | while read -r f; do
    perl -pi -e 's/(source_file_name)\s+(write\s*\()/\1\n      \2/gi' "$f"
done

# === Create compatibility source files ===
echo "Creating compatibility stubs..."

# eof() replacement (Intel built-in)
cat > "${SRC_DIR}/core/eof_compat.f90" << 'FORTRAN_EOF'
logical function eof(unit_num)
  implicit none
  integer, intent(in) :: unit_num
  integer :: ios
  character(1) :: dummy
  read(unit_num, '(a)', iostat=ios, advance='no') dummy
  if (ios .ne. 0) then
     eof = .true.
  else
     eof = .false.
     backspace(unit_num)
  endif
  return
end function eof
FORTRAN_EOF

# Intel beepqq/sleepqq stubs
cat > "${SRC_DIR}/core/intel_compat.f90" << 'FORTRAN_EOF'
subroutine beepqq(frequency, duration)
  implicit none
  integer, intent(in) :: frequency, duration
  return
end subroutine beepqq

subroutine sleepqq(milliseconds)
  implicit none
  integer, intent(in) :: milliseconds
  call sleep(milliseconds / 1000)
  return
end subroutine sleepqq
FORTRAN_EOF

# Isotopes stub (full module excluded from build)
cat > "${SRC_DIR}/core/isotopes_stub.f90" << 'FORTRAN_EOF'
subroutine isotopes(iz, jz, time2, route_dt, iflag)
  implicit none
  integer, intent(in) :: iz, jz, iflag
  real(4), intent(in) :: time2, route_dt
  return
end subroutine isotopes
FORTRAN_EOF

# Stub the optional 2D-grid / time-series / NetCDF-update I/O routines that
# CHARM.f90 and sub.f call but that have no implementation anywhere in the
# repo (or upstream watflood/model). Argument-less external stubs link against
# any call (no explicit interface) and no-op at runtime — enough to produce a
# working executable for the core simulation. The runtime-complete binary is
# the Waterloo Windows prebuilt.
cat > "${SRC_DIR}/core/watflood_io_stubs.f90" << 'FORTRAN_EOF'
      subroutine read_2d_pcp_nc
      end subroutine
      subroutine read_2d_tmp_nc
      end subroutine
      subroutine read_swe_update
      end subroutine
      subroutine read_ts_flow_nc
      end subroutine
      subroutine read_uzs_update
      end subroutine
      subroutine write_2d_cumm_et
      end subroutine
      subroutine write_2d_flow
      end subroutine
      subroutine write_2d_grid_runoff
      end subroutine
      subroutine write_2d_swe
      end subroutine
      subroutine write_2d_uzs
      end subroutine
      subroutine write_ts_flow
      end subroutine
      subroutine write_ts_lake_inflow
      end subroutine
      subroutine write_ts_lake_levels
      end subroutine
      subroutine write_ts_lake_outflow
      end subroutine
FORTRAN_EOF

# === Declare eof() in area_watflood.f so every "use area_watflood" gets it ===
# eof() is Intel's EOF builtin; gfortran uses our eof_compat.f90 implementation.
# It MUST be declared as a module-level "logical, external" — NOT an interface
# block: with an explicit interface, gfortran misparses "do while(.not.eof(u))"
# and "if(eof(u))then" as "Unclassifiable statement" in many readers. A plain
# external declaration parses cleanly in all forms (verified on gfortran 12/15).
if ! grep -q "external :: eof" "${SRC_DIR}/core/area_watflood.f" 2>/dev/null; then
    perl -0777 -pi -e 's/(^\s*end module area_watflood)/c     eof() (Intel EOF builtin) provided by eof_compat.f90\n      logical, external :: eof\n\1/mi' "${SRC_DIR}/core/area_watflood.f"
fi

# === Add eof external declaration to EF_Module.f ===
# EF_Module's CountDataLinesAfterHeader uses Intel's EOF() builtin, which
# gfortran lacks; eof() is supplied by eof_compat.f90 (an external function).
# gfortran must be told eof is an external LOGICAL or it implicitly types it
# REAL(4) and rejects ".not.EOF(...)" ("Operand of .not. operator is REAL(4)").
# Two cases for where the declaration must go:
#   * function lives AFTER END MODULE -> declare inside the function itself.
#   * function is a module procedure (inside CONTAINS) -> declaring it in the
#     function body is fine too, but declaring it once in the module's
#     specification part (before CONTAINS) host-associates it to every contained
#     procedure. We must NOT skip this case: EF_Module does not use area_watflood,
#     so eof is otherwise undeclared here (this was the real watflood build break).
if [ -f "${SRC_DIR}/common/EF_Module.f" ] && ! grep -q "logical, external :: eof" "${SRC_DIR}/common/EF_Module.f" 2>/dev/null; then
    "$PYTHON" -c "
import re
path = '${SRC_DIR}/common/EF_Module.f'
with open(path) as f:
    txt = f.read()
if 'CountDataLinesAfterHeader' not in txt:
    raise SystemExit(0)
# Declare eof LOCALLY in the function (after its full signature line, so the
# closing paren isn't split). Local — not module scope — so 'use EF_Module'
# does not export it and clash with area_watflood's eof (ambiguous reference ->
# 'Unclassifiable statement') in readers that use both.
new, n = re.subn(
    r'(INTEGER\s+FUNCTION\s+CountDataLinesAfterHeader\s*\([^)]*\))',
    r'\1\n      logical, external :: eof',
    txt, count=1, flags=re.IGNORECASE)
if n:
    with open(path, 'w') as f:
        f.write(new)
    print('  Injected eof external declaration into CountDataLinesAfterHeader')
" 2>/dev/null || echo "  WARNING: EF_Module.f eof injection failed (non-fatal)"
fi

# (The do-while(.not.eof(...)) loops no longer need rewriting: the module-level
#  "logical, external :: eof" above makes every eof() form parse natively.)

# === Create/update CMakeLists.txt files ===
echo "Setting up CMake build system..."

cat > "${INSTALL_DIR}/CMakeLists.txt" << 'CMAKE_EOF'
cmake_minimum_required(VERSION 3.20)
project(WATFLOOD Fortran)

set(CHARM_MAJOR_VERSION 10)
set(CHARM_MINOR_VERSION 5)
set(CHARM_PATCH_VERSION 19)
set(CHARM_VERSION ${CHARM_MAJOR_VERSION}.${CHARM_MINOR_VERSION}.${CHARM_PATCH_VERSION})

IF ("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
    set(CMAKE_BUILD_TYPE "DEBUG")
ELSE()
    set(CMAKE_BUILD_TYPE "RELEASE")
ENDIF()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include)
include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
file(MAKE_DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# NetCDF-Fortran: read_2D_swe_nc.f90 does "use netcdf". Discover its module
# include dir and link flags via nf-config (Linux/macOS/MSYS2) so the model
# library compiles and the executable links. Without this the build dies at
# "Cannot open module file netcdf.mod".
find_program(NF_CONFIG NAMES nf-config)
find_program(NC_CONFIG NAMES nc-config)
if(NF_CONFIG)
    execute_process(COMMAND ${NF_CONFIG} --includedir
        OUTPUT_VARIABLE NF_INCDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND ${NF_CONFIG} --flibs
        OUTPUT_VARIABLE NF_FLIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(${NF_INCDIR})
    set(NETCDF_LINK "${NF_FLIBS}")
    if(NC_CONFIG)
        # On Homebrew, netcdf-fortran and netcdf-c live in separate prefixes:
        # nf-config --flibs names -lnetcdf but omits its -L, so the C library
        # isn't found ("ld: library 'netcdf' not found"). nc-config --libs adds
        # the C netcdf -L/-l. Harmless on Linux (already on the default path).
        execute_process(COMMAND ${NC_CONFIG} --libs
            OUTPUT_VARIABLE NC_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(NETCDF_LINK "${NF_FLIBS} ${NC_LIBS}")
    endif()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${NETCDF_LINK}")
    message(STATUS "NetCDF link: ${NETCDF_LINK}")
else()
    message(WARNING "nf-config not found — read_2D_swe_nc.f90 may fail to build")
endif()

# Legacy Fortran compat flags for GCC 10+
IF(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch -fallow-invalid-boz -ffixed-line-length-none -ffree-line-length-none -fd-lines-as-comments -fdec -fno-range-check -std=legacy -w")
ENDIF()

IF(${CMAKE_BUILD_TYPE} MATCHES "DEBUG")
    IF(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
        set(CMAKE_Fortran_FLAGS_DEBUG "${CMAKE_Fortran_FLAGS_DEBUG} -O0 -Wall -fcheck=all -fbacktrace")
    ENDIF()
ENDIF()

add_subdirectory(src)
CMAKE_EOF

cat > "${SRC_DIR}/CMakeLists.txt" << 'CMAKE_EOF'
project(CHARM Fortran)
add_subdirectory(core)
add_subdirectory(common)
add_subdirectory(model)
CMAKE_EOF

# Core CMakeLists
cat > "${SRC_DIR}/core/CMakeLists.txt" << 'CMAKE_EOF'
project(core Fortran)
set(CORE_SOURCES area17.f Areacg.f area_debug.f90 area_watflood.f eof_compat.f90 intel_compat.f90 isotopes_stub.f90 watflood_io_stubs.f90)
add_library(${PROJECT_NAME} STATIC ${CORE_SOURCES})
CMAKE_EOF

# Common CMakeLists
COMMON_SOURCES=$(cd "${SRC_DIR}/common" && ls -1 *.f *.f90 *.for 2>/dev/null | tr '\n' ';')
cat > "${SRC_DIR}/common/CMakeLists.txt" << CMAKE_EOF
project(common Fortran)
set(COMMON_SOURCES ${COMMON_SOURCES})
# Both files export read_par_parser. Static archive ordering can select the
# legacy reader, which does not initialize fpFactor used by modern routing.
if(EXISTS "\${CMAKE_CURRENT_SOURCE_DIR}/read_par_parser.f90")
    list(REMOVE_ITEM COMMON_SOURCES read_par_parser.f)
endif()
add_library(\${PROJECT_NAME} STATIC \${COMMON_SOURCES})
add_dependencies(\${PROJECT_NAME} core)
CMAKE_EOF

# Model CMakeLists — exclude iso/ and utilities/
MODEL_SOURCES=$(cd "${SRC_DIR}/model" && ls -1 *.f *.f90 *.for 2>/dev/null | grep -v "^CHARM.f90$" | tr '\n' ';')
cat > "${SRC_DIR}/model/CMakeLists.txt" << CMAKE_EOF
project(model Fortran)
set(MODEL_SOURCES ${MODEL_SOURCES})
add_library(\${PROJECT_NAME} STATIC \${MODEL_SOURCES})
add_dependencies(\${PROJECT_NAME} core common)
target_link_libraries(\${PROJECT_NAME} common)

set(PROJECT_EXECUTABLE charm)
add_executable(\${PROJECT_EXECUTABLE} CHARM.f90)
add_dependencies(\${PROJECT_EXECUTABLE} core common model)
# Link most-dependent first: model/common reference core symbols (area_watflood,
# isotopes, ...), so core must come LAST for GNU ld's single-pass resolution.
target_link_libraries(\${PROJECT_EXECUTABLE} model common core)
CMAKE_EOF

# === Build with CMake ===
echo "Building WATFLOOD..."
BUILD_DIR="${INSTALL_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

cmake -S "${INSTALL_DIR}" -B "${BUILD_DIR}" 2>&1
cmake --build "${BUILD_DIR}" -j "${NCPU}" 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: WATFLOOD build failed."
    echo "Check the build output above for errors."
    exit 1
fi

# === Install binary ===
if [ -f "${BUILD_DIR}/bin/charm" ]; then
    cp "${BUILD_DIR}/bin/charm" "${BIN_DIR}/watflood"
    chmod +x "${BIN_DIR}/watflood"
    echo ""
    echo "Binary installed: ${BIN_DIR}/watflood"
    file "${BIN_DIR}/watflood"
    echo ""
    echo "=== WATFLOOD Installation Complete ==="
    exit 0
else
    echo "ERROR: Build completed but charm binary not found."
    echo "Expected: ${BUILD_DIR}/bin/charm"
    exit 1
fi
            '''.strip()
        ],
        'dependencies': ['gfortran', 'cmake'],
        'test_command': None,
        'verify_install': {
            'file_paths': ['bin/watflood'],
            'check_type': 'exists'
        },
        'order': 24,
        'optional': True,
    }
