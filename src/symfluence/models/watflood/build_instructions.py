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

from symfluence.cli.services import BuildInstructionsRegistry, get_common_build_environment


@BuildInstructionsRegistry.register('watflood')
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

# === Check for existing native binary ===
if [ -f "${BIN_DIR}/watflood" ]; then
    file_output=$(file "${BIN_DIR}/watflood" 2>/dev/null || echo "unknown")
    if echo "$file_output" | grep -qi "mach-o\|elf\|executable"; then
        echo "Found existing native binary: ${BIN_DIR}/watflood"
        echo "$file_output"
        echo "=== WATFLOOD Installation Complete (existing) ==="
        exit 0
    fi
fi

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

# Fix Intel <variable> format descriptors (not supported by gfortran)
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" | while read -r f; do
    perl -pi -e 's/<[a-zA-Z_][a-zA-Z0-9_]*>/999/g' "$f"
done

# Fix Hollerith constant in area_watflood.f: flen='none' -> flen=999999
perl -pi -e "s/flen='none'/flen=999999/" "${SRC_DIR}/core/area_watflood.f"

# Fix STOP without space before string
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" | while read -r f; do
    perl -pi -e "s/STOP\s*'/STOP '/gi" "$f"
done

# Fix .eq./.ne. with logical operands -> .eqv./.neqv.
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" | while read -r f; do
    perl -pi -e 's/\.eq\.\s*\.true\./\.eqv\.\.\s*true\./gi' "$f"
    perl -pi -e 's/\.eq\.\s*\.false\./\.eqv\.\.false\./gi' "$f"
    perl -pi -e 's/==\s*\.false\./\.eqv\.\.false\./gi' "$f"
    perl -pi -e 's/==\s*\.true\./\.eqv\.\.true\./gi' "$f"
done

# Fix integer-as-logical: if(dds_flag)then where dds_flag is integer
perl -pi -e 's/if\(dds_flag\)then/if(dds_flag.ne.0)then/gi' "${SRC_DIR}/model/dds_code.f" 2>/dev/null || true

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
    fi
done

# Fix Intel GETARG (3-arg) -> GNU GETARG (2-arg) in CHARM.f90
if [ -f "${SRC_DIR}/model/CHARM.f90" ]; then
    perl -pi -e "s/CALL GETARG\(1, buf, status1\)/IF(IARGC().GE.1)THEN\n        CALL GETARG(1, buf)\n      ELSE\n        buf=' '\n      ENDIF/" "${SRC_DIR}/model/CHARM.f90"
    perl -pi -e 's/if\(status1\.ne\.1\)buf=.*//' "${SRC_DIR}/model/CHARM.f90"
fi

# Fix stray trailing quotes exposed by -ffixed-line-length-none
for f in "${SRC_DIR}/model/write_tb0.f" "${SRC_DIR}/model/write_diversion_tb0.f" "${SRC_DIR}/model/lst.f"; do
    if [ -f "$f" ]; then
        perl -pi -e "s/^(.{72,})'\s*$/\$1 /" "$f"
    fi
done

# Fix EF_Module.f: CountDataLinesAfterHeader is placed after END MODULE
# which confuses gfortran.  Extract it into a separate source file instead
# of trying to squeeze it into CONTAINS (which causes host-association errors).
EF_MOD="${SRC_DIR}/common/EF_Module.f"
if [ -f "$EF_MOD" ]; then
    python3 -c "
import re
with open('$EF_MOD') as f:
    txt = f.read()
if 'CountDataLinesAfterHeader' in txt:
    # Find END MODULE and everything after it
    m = re.search(r'^(\s*END\s+MODULE\s+\w+\s*\n)(.*)', txt, re.DOTALL | re.MULTILINE | re.IGNORECASE)
    if m:
        end_mod = m.group(1)
        after = m.group(2).strip()
        if after:
            # Write the trailing function(s) to a separate file
            with open('${SRC_DIR}/common/EF_Extra.f90', 'w') as ef:
                ef.write('! Extracted from EF_Module.f for gfortran compatibility\n')
                ef.write(after + '\n')
            # Remove the trailing code from the original file
            txt = txt[:m.start()] + end_mod
            with open('$EF_MOD', 'w') as f:
                f.write(txt)
            print('  EF_Module.f: extracted trailing function to EF_Extra.f90')
        else:
            print('  EF_Module.f: no trailing code after END MODULE')
    else:
        print('  EF_Module.f: no END MODULE found')
else:
    print('  EF_Module.f: no CountDataLinesAfterHeader found')
" 2>/dev/null || echo "  WARNING: EF_Module.f patch failed (non-fatal)"
fi

# Fix WRITE statements with misquoted format strings
# e.g. write(un,3020)':SourceFile         ',source_file_name
# The colon inside the string confuses some compilers.  Ensure proper quoting.
find "${SRC_DIR}" -name "*.f" -o -name "*.f90" | while read -r f; do
    perl -pi -e "s/write\(([^)]+)\)'/write(\$1) '/gi" "$f"
done

# Fix missing comma in WRITE continuation:
#   source_file_name   '   ->   source_file_name , '
# write_diversion_tb0.f line 87 has: ,source_file_name   '
# which is missing a comma before the trailing string literal.
for f in "${SRC_DIR}/model/write_diversion_tb0.f" "${SRC_DIR}/model/write_tb0.f"; do
    if [ -f "$f" ]; then
        perl -pi -e "s/(source_file_name)\s+'/\$1, '/g" "$f"
    fi
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

# === Add eof interface to area_watflood.f ===
if ! grep -q "eof_compat" "${SRC_DIR}/core/area_watflood.f" 2>/dev/null; then
    # Add eof interface before 'end module'
    perl -0777 -pi -e 's/(^\s*end module area_watflood)/c     Interface for eof() — Intel built-in, provided by eof_compat.f90\n      interface\n        logical function eof(unit_num)\n          integer, intent(in) :: unit_num\n        end function eof\n      end interface\n\1/mi' "${SRC_DIR}/core/area_watflood.f"
fi

# === Add eof external declaration to EF_Module.f ===
if ! grep -q "logical, external :: eof" "${SRC_DIR}/common/EF_Module.f" 2>/dev/null; then
    perl -pi -e 's/(INTEGER FUNCTION CountDataLinesAfterHeader)/\1\n      logical, external :: eof/' "${SRC_DIR}/common/EF_Module.f"
fi

# === Fix read_par_parser.f90 eof() usage ===
if [ -f "${SRC_DIR}/common/read_par_parser.f90" ]; then
    perl -0777 -pi -e 's/do while\(\.not\.eof\(unitNum\)\)/ios=0\n      do while(ios.eq.0)/g' "${SRC_DIR}/common/read_par_parser.f90"
fi

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

# Legacy Fortran compat flags for GCC 10+
IF(${CMAKE_Fortran_COMPILER_ID} MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -fallow-argument-mismatch -fallow-invalid-boz -ffixed-line-length-none -fd-lines-as-comments -fdec -fno-range-check -std=legacy -w")
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
set(CORE_SOURCES area17.f Areacg.f area_debug.f90 area_watflood.f eof_compat.f90 intel_compat.f90 isotopes_stub.f90)
add_library(${PROJECT_NAME} STATIC ${CORE_SOURCES})
CMAKE_EOF

# Common CMakeLists
COMMON_SOURCES=$(cd "${SRC_DIR}/common" && ls -1 *.f *.f90 2>/dev/null | tr '\n' ';')
cat > "${SRC_DIR}/common/CMakeLists.txt" << CMAKE_EOF
project(common Fortran)
set(COMMON_SOURCES ${COMMON_SOURCES})
add_library(\${PROJECT_NAME} STATIC \${COMMON_SOURCES})
add_dependencies(\${PROJECT_NAME} core)
CMAKE_EOF

# Model CMakeLists — exclude iso/ and utilities/
MODEL_SOURCES=$(cd "${SRC_DIR}/model" && ls -1 *.f *.f90 2>/dev/null | grep -v "^CHARM.f90$" | tr '\n' ';')
cat > "${SRC_DIR}/model/CMakeLists.txt" << CMAKE_EOF
project(model Fortran)
set(MODEL_SOURCES ${MODEL_SOURCES})
add_library(\${PROJECT_NAME} STATIC \${MODEL_SOURCES})
add_dependencies(\${PROJECT_NAME} core common)
target_link_libraries(\${PROJECT_NAME} common)

set(PROJECT_EXECUTABLE charm)
add_executable(\${PROJECT_EXECUTABLE} CHARM.f90)
add_dependencies(\${PROJECT_EXECUTABLE} core common model)
target_link_libraries(\${PROJECT_EXECUTABLE} core common model)
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
