#!/bin/bash
#
# Stage SYMFLUENCE release artifacts for npm distribution
#
# Creates a standardized directory structure:
#   symfluence-tools/
#     bin/          - Executables (summa, mizuroute, fuse, ngen, taudem tools)
#     share/        - Shared data files (if any)
#     LICENSES/     - License files from all tools
#     toolchain.json - Build metadata
#
# Usage:
#   ./scripts/stage_release_artifacts.sh <platform> <installs_dir> <output_dir>
#
# Example:
#   ./scripts/stage_release_artifacts.sh \
#     linux-x86_64 \
#     $SYMFLUENCE_DATA/installs \
#     ./release

set -e

# Colors
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC='';
fi

print_error()   { echo -e "${RED}Error:${NC} $1" >&2; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

# Parse arguments
PLATFORM="${1:-}"
INSTALLS_DIR="${2:-}"
OUTPUT_DIR="${3:-}"

if [ -z "$PLATFORM" ] || [ -z "$INSTALLS_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    cat >&2 <<EOF
Usage: $0 <platform> <installs_dir> <output_dir>

Arguments:
  platform       Platform identifier (e.g., linux-x86_64, macos-arm64)
  installs_dir   Directory containing tool installations
  output_dir     Directory to stage artifacts (will contain symfluence-tools/)

Example:
  $0 linux-x86_64 \$SYMFLUENCE_DATA/installs ./release
EOF
    exit 1
fi

# Resolved before the script cd's into the staging dir, so sibling scripts stay
# reachable regardless of how this one was invoked.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Portable realpath fallback (realpath not available on all systems)
_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")")
    fi
}

# Resolve absolute paths
INSTALLS_DIR="$(_realpath "$INSTALLS_DIR")"
OUTPUT_DIR="$(_realpath "$OUTPUT_DIR")"

if [ ! -d "$INSTALLS_DIR" ]; then
    print_error "Installation directory not found: $INSTALLS_DIR"
    exit 1
fi

print_info "Staging SYMFLUENCE tools for $PLATFORM"
print_info "Source: $INSTALLS_DIR"
print_info "Output: $OUTPUT_DIR"

# Create staging directory
STAGE_DIR="$OUTPUT_DIR/symfluence-tools"
mkdir -p "$STAGE_DIR"
cd "$STAGE_DIR"

# Create standard structure
mkdir -p bin lib share LICENSES

print_info "Created staging structure:"
tree -L 1 . 2>/dev/null || ls -la

# Counter for staged files
STAGED_COUNT=0

# Helper function to stage a binary
stage_binary() {
    local src="$1"
    local dest_name="$2"
    local tool_name="$3"

    if [ -f "$src" ]; then
        cp "$src" "bin/$dest_name"
        chmod +x "bin/$dest_name"
        print_success "Staged $tool_name → bin/$dest_name"
        STAGED_COUNT=$((STAGED_COUNT + 1))
        return 0
    else
        print_warning "Not found: $src"
        return 1
    fi
}

# Helper function to stage a shared library
stage_library() {
    local src="$1"
    local dest_name="$2"
    local tool_name="$3"

    if [ -f "$src" ]; then
        cp "$src" "lib/$dest_name"
        chmod +x "lib/$dest_name"
        print_success "Staged library $tool_name → lib/$dest_name"
        return 0
    else
        return 1
    fi
}

# Helper function to stage a license
stage_license() {
    local src_dir="$1"
    local tool_name="$2"

    for license_file in LICENSE LICENSE.txt LICENSE.md COPYING; do
        if [ -f "$src_dir/$license_file" ]; then
            cp "$src_dir/$license_file" "LICENSES/LICENSE-$tool_name"
            print_success "Staged license for $tool_name"
            return 0
        fi
    done

    print_warning "No license found for $tool_name in $src_dir"
    return 0  # Don't fail the build for missing licenses
}

# ============================================================================
# Stage SUMMA
# ============================================================================
print_info "Staging SUMMA..."

SUMMA_DIR="$INSTALLS_DIR/summa"
if [ -d "$SUMMA_DIR" ]; then
    # Try summa.exe first, then summa_sundials.exe
    if stage_binary "$SUMMA_DIR/bin/summa.exe" "summa" "SUMMA"; then
        :
    elif stage_binary "$SUMMA_DIR/bin/summa_sundials.exe" "summa" "SUMMA"; then
        :
    else
        print_warning "SUMMA binary not found"
    fi

    # Stage SUMMA shared libraries (required at runtime)
    if [ "$(uname)" = "Darwin" ]; then
        stage_library "$SUMMA_DIR/lib/libsumma.dylib" "libsumma.dylib" "SUMMA" || \
            print_warning "libsumma.dylib not found"
    else
        stage_library "$SUMMA_DIR/lib/libsumma.so" "libsumma.so" "SUMMA" || \
            print_warning "libsumma.so not found"
    fi

    # Stage libftz.so on Linux x86_64 (FTZ/DAZ control)
    if [ "$(uname)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ]; then
        stage_library "$SUMMA_DIR/bin/libftz.so" "libftz.so" "SUMMA" || true
    fi

    # Stage SUNDIALS shared libraries (SUMMA dependency, built from source)
    SUNDIALS_LIB="$INSTALLS_DIR/sundials/install/sundials/lib"
    if [ -d "$SUNDIALS_LIB" ]; then
        print_info "Staging SUNDIALS libraries..."
        for slib in "$SUNDIALS_LIB"/libsundials_*; do
            [ -f "$slib" ] || continue
            slib_name="$(basename "$slib")"
            # Only stage actual shared libraries (.dylib or .so), skip symlinks to avoid duplicates
            case "$slib_name" in
                *.dylib|*.so|*.so.*)
                    if [ ! -L "$slib" ]; then
                        stage_library "$slib" "$slib_name" "SUNDIALS" || true
                    else
                        # Copy symlinks too (versioned .dylib.X -> .dylib)
                        cp -P "$slib" "lib/$slib_name"
                    fi
                    ;;
            esac
        done
    else
        print_warning "SUNDIALS lib directory not found at $SUNDIALS_LIB"
    fi

    stage_license "$SUMMA_DIR" "SUMMA"
else
    print_warning "SUMMA not installed"
fi

# ============================================================================
# Stage mizuRoute
# ============================================================================
print_info "Staging mizuRoute..."

MIZU_DIR="$INSTALLS_DIR/mizuRoute"
if [ -d "$MIZU_DIR" ]; then
    # Try mizuRoute.exe first (Windows), then mizuRoute (Unix)
    if [ -f "$MIZU_DIR/route/bin/mizuRoute.exe" ]; then
        stage_binary "$MIZU_DIR/route/bin/mizuRoute.exe" "mizuroute" "mizuRoute"
    elif [ -f "$MIZU_DIR/route/bin/mizuRoute" ]; then
        stage_binary "$MIZU_DIR/route/bin/mizuRoute" "mizuroute" "mizuRoute"
    else
        print_warning "mizuRoute binary not found in $MIZU_DIR/route/bin"
    fi
    stage_license "$MIZU_DIR" "mizuRoute"
else
    print_warning "mizuRoute not installed"
fi

# ============================================================================
# Stage FUSE
# ============================================================================
print_info "Staging FUSE..."

FUSE_DIR="$INSTALLS_DIR/fuse"
if [ -d "$FUSE_DIR" ]; then
    if [ -f "$FUSE_DIR/bin/fuse.exe" ]; then
        stage_binary "$FUSE_DIR/bin/fuse.exe" "fuse" "FUSE"
    elif [ -f "$FUSE_DIR/bin/fuse" ]; then
        stage_binary "$FUSE_DIR/bin/fuse" "fuse" "FUSE"
    else
        print_warning "FUSE binary not found in $FUSE_DIR/bin"
    fi
    stage_license "$FUSE_DIR" "FUSE"
else
    print_warning "FUSE not installed"
fi

# ============================================================================
# Stage NGEN
# ============================================================================
print_info "Staging NGEN..."

NGEN_DIR="$INSTALLS_DIR/ngen"
if [ -d "$NGEN_DIR" ]; then
    if [ -f "$NGEN_DIR/cmake_build/ngen" ]; then
        stage_binary "$NGEN_DIR/cmake_build/ngen" "ngen" "NGEN"
    else
        print_warning "NGEN binary not found in $NGEN_DIR/cmake_build"
    fi

    # Stage NGEN shared libraries (BMI modules built as shared libs)
    # Use find for recursive discovery — bash globs don't expand ** without globstar
    while IFS= read -r ngen_lib; do
        [ -f "$ngen_lib" ] || continue
        ngen_lib_name="$(basename "$ngen_lib")"
        stage_library "$ngen_lib" "$ngen_lib_name" "NGEN" || true
    done < <(find "$NGEN_DIR/cmake_build" "$NGEN_DIR/extern" \( -name "*.dylib" -o -name "*.so" \) -not -type d 2>/dev/null || true)
    # Windows: ngen.exe imports libiso_c_bmi.dll (and the BMI DLLs) at load
    # time, and the loader looks next to the .exe, not in lib/.
    while IFS= read -r ngen_dll; do
        [ -f "$ngen_dll" ] || continue
        cp "$ngen_dll" "bin/$(basename "$ngen_dll")"
        print_success "Staged NGEN DLL → bin/$(basename "$ngen_dll")"
    done < <(find "$NGEN_DIR/cmake_build" "$NGEN_DIR/extern" -name "*.dll" -not -type d 2>/dev/null || true)

    stage_license "$NGEN_DIR" "NGEN"
else
    print_warning "NGEN not installed"
fi

# ============================================================================
# Stage HYPE
# ============================================================================
print_info "Staging HYPE..."

HYPE_DIR="$INSTALLS_DIR/hype"
if [ -d "$HYPE_DIR" ]; then
    # Try to stage HYPE binary (optional, may not be built on all platforms)
    stage_binary "$HYPE_DIR/bin/hype" "hype" "HYPE" || print_warning "HYPE binary not found (may not be built yet)"
    stage_license "$HYPE_DIR" "HYPE"
else
    print_warning "HYPE not installed"
fi

# ============================================================================
# Stage TauDEM
# ============================================================================
print_info "Staging TauDEM..."

TAUDEM_DIR="$INSTALLS_DIR/TauDEM"
if [ -d "$TAUDEM_DIR/bin" ]; then
    # TauDEM has multiple executables
    TAUDEM_COUNT=0
    for exe in "$TAUDEM_DIR/bin"/*; do
        if [ -x "$exe" ]; then
            exe_name="$(basename "$exe")"
            if stage_binary "$exe" "$exe_name" "TauDEM"; then
                TAUDEM_COUNT=$((TAUDEM_COUNT + 1))
            fi
        fi
    done

    if [ $TAUDEM_COUNT -gt 0 ]; then
        print_success "Staged $TAUDEM_COUNT TauDEM executables"
    fi

    stage_license "$TAUDEM_DIR" "TauDEM"
else
    print_warning "TauDEM not installed"
fi

# ============================================================================
# Stage MESH
# ============================================================================
print_info "Staging MESH..."

MESH_DIR="$INSTALLS_DIR/mesh"
if [ -d "$MESH_DIR" ]; then
    if [ -f "$MESH_DIR/bin/mesh.exe" ]; then
        stage_binary "$MESH_DIR/bin/mesh.exe" "mesh" "MESH"
    elif [ -f "$MESH_DIR/bin/mesh" ]; then
        stage_binary "$MESH_DIR/bin/mesh" "mesh" "MESH"
    else
        print_warning "MESH binary not found (may not be built yet)"
    fi
    stage_license "$MESH_DIR" "MESH"
else
    print_warning "MESH not installed"
fi

# ============================================================================
# Stage WMFire
# ============================================================================
print_info "Staging WMFire..."

WMFIRE_DIR="$INSTALLS_DIR/wmfire"
if [ -d "$WMFIRE_DIR" ]; then
    # WMFire is a shared library — stage to lib/ so @rpath resolution works
    if [ "$(uname)" = "Darwin" ] && [ -f "$WMFIRE_DIR/lib/libwmfire.dylib" ]; then
        stage_library "$WMFIRE_DIR/lib/libwmfire.dylib" "libwmfire.dylib" "WMFire"
    elif [ -f "$WMFIRE_DIR/lib/libwmfire.dll" ]; then
        stage_library "$WMFIRE_DIR/lib/libwmfire.dll" "libwmfire.dll" "WMFire"
    elif [ -f "$WMFIRE_DIR/lib/libwmfire.so" ]; then
        stage_library "$WMFIRE_DIR/lib/libwmfire.so" "libwmfire.so" "WMFire"
    else
        print_warning "WMFire binary not found (may not be built yet)"
    fi
    stage_license "$WMFIRE_DIR" "WMFire"
else
    print_warning "WMFire not installed"
fi

# ============================================================================
# Stage RHESSys
# ============================================================================
print_info "Staging RHESSys..."

RHESSYS_DIR="$INSTALLS_DIR/rhessys"
if [ -d "$RHESSYS_DIR" ]; then
    if [ -f "$RHESSYS_DIR/bin/rhessys" ]; then
        stage_binary "$RHESSYS_DIR/bin/rhessys" "rhessys" "RHESSys"
    else
        print_warning "RHESSys binary not found (may not be built yet)"
    fi
    stage_license "$RHESSYS_DIR" "RHESSys"
else
    print_warning "RHESSys not installed"
fi

# ============================================================================
# Stage NoahMP (noah-owp-modular)
# ============================================================================
print_info "Staging NoahMP..."

NOAHMP_DIR="$INSTALLS_DIR/noah-owp-modular"
if [ -d "$NOAHMP_DIR" ]; then
    if stage_binary "$NOAHMP_DIR/run/noah_owp_modular.exe" "noah_owp_modular" "NoahMP"; then
        :
    else
        print_warning "NoahMP binary not found"
    fi
    stage_license "$NOAHMP_DIR" "NoahMP"
else
    print_warning "NoahMP not installed"
fi

# ============================================================================
# Stage VIC
# ============================================================================
print_info "Staging VIC..."

VIC_DIR="$INSTALLS_DIR/vic"
if [ -d "$VIC_DIR" ]; then
    if stage_binary "$VIC_DIR/bin/vic_image.exe" "vic" "VIC"; then
        :
    elif stage_binary "$VIC_DIR/bin/vic_image" "vic" "VIC"; then
        :
    else
        print_warning "VIC binary not found"
    fi
    stage_license "$VIC_DIR" "VIC"
else
    print_warning "VIC not installed"
fi

# ============================================================================
# Stage SWAT
# ============================================================================
print_info "Staging SWAT..."

SWAT_DIR="$INSTALLS_DIR/swat"
if [ -d "$SWAT_DIR" ]; then
    if stage_binary "$SWAT_DIR/bin/swat_rel.exe" "swat" "SWAT"; then
        :
    elif stage_binary "$SWAT_DIR/bin/swat" "swat" "SWAT"; then
        :
    else
        print_warning "SWAT binary not found"
    fi
    stage_license "$SWAT_DIR" "SWAT"
else
    print_warning "SWAT not installed"
fi

# ============================================================================
# Stage PRMS
# ============================================================================
print_info "Staging PRMS..."

PRMS_DIR="$INSTALLS_DIR/prms"
if [ -d "$PRMS_DIR" ]; then
    stage_binary "$PRMS_DIR/bin/prms" "prms" "PRMS" || print_warning "PRMS binary not found"
    stage_license "$PRMS_DIR" "PRMS"
else
    print_warning "PRMS not installed"
fi

# ============================================================================
# Stage mHM
# ============================================================================
print_info "Staging mHM..."

MHM_DIR="$INSTALLS_DIR/mhm"
if [ -d "$MHM_DIR" ]; then
    stage_binary "$MHM_DIR/bin/mhm" "mhm" "mHM" || print_warning "mHM binary not found"
    stage_license "$MHM_DIR" "mHM"
else
    print_warning "mHM not installed"
fi

# ============================================================================
# Stage CRHM
# ============================================================================
print_info "Staging CRHM..."

CRHM_DIR="$INSTALLS_DIR/crhm"
if [ -d "$CRHM_DIR" ]; then
    stage_binary "$CRHM_DIR/bin/crhm" "crhm" "CRHM" || print_warning "CRHM binary not found"
    stage_license "$CRHM_DIR" "CRHM"
else
    print_warning "CRHM not installed"
fi

# ============================================================================
# Stage WRF-Hydro
# ============================================================================
print_info "Staging WRF-Hydro..."

WRFHYDRO_DIR="$INSTALLS_DIR/wrfhydro"
if [ -d "$WRFHYDRO_DIR" ]; then
    if stage_binary "$WRFHYDRO_DIR/bin/wrf_hydro.exe" "wrfhydro" "WRF-Hydro"; then
        :
    elif stage_binary "$WRFHYDRO_DIR/bin/wrfhydro" "wrfhydro" "WRF-Hydro"; then
        :
    else
        print_warning "WRF-Hydro binary not found"
    fi
    stage_license "$WRFHYDRO_DIR" "WRF-Hydro"
else
    print_warning "WRF-Hydro not installed"
fi

# ============================================================================
# Stage WATFLOOD
# ============================================================================
print_info "Staging WATFLOOD..."

WATFLOOD_DIR="$INSTALLS_DIR/watflood"
if [ -d "$WATFLOOD_DIR" ]; then
    # Try watflood.exe FIRST: under MSYS bash `[ -f bin/watflood ]` is true via
    # the .exe stat-fallback, so checking it first would stage the Windows
    # prebuilt as a bare `watflood` and skip the DLL copy below.
    if stage_binary "$WATFLOOD_DIR/bin/watflood.exe" "watflood.exe" "WATFLOOD"; then
        :
    elif stage_binary "$WATFLOOD_DIR/bin/watflood" "watflood" "WATFLOOD"; then
        :
    elif stage_binary "$WATFLOOD_DIR/bin/charm" "watflood" "WATFLOOD"; then
        :
    else
        print_warning "WATFLOOD binary not found"
    fi
    # Always bundle any DLLs shipped next to the binary (the official Waterloo
    # CHARM prebuilt carries its own netCDF/HDF5/Intel runtime DLLs, which the
    # ntldd mingw-tree bundler won't pick up). Windows resolves DLLs from the
    # executable's own directory, so place them in bin/.
    for _dll in "$WATFLOOD_DIR"/bin/*.dll; do
        if [ -f "$_dll" ]; then
            cp "$_dll" "bin/$(basename "$_dll")"
            print_success "Staged WATFLOOD DLL → bin/$(basename "$_dll")"
        fi
    done
    stage_license "$WATFLOOD_DIR" "WATFLOOD"
else
    print_warning "WATFLOOD not installed"
fi

# ============================================================================
# Stage MODFLOW
# ============================================================================
print_info "Staging MODFLOW..."

MODFLOW_DIR="$INSTALLS_DIR/modflow"
if [ -d "$MODFLOW_DIR" ]; then
    stage_binary "$MODFLOW_DIR/bin/mf6" "mf6" "MODFLOW" || print_warning "MODFLOW binary not found"
    stage_license "$MODFLOW_DIR" "MODFLOW"
else
    print_warning "MODFLOW not installed"
fi

# ============================================================================
# Stage ParFlow
# ============================================================================
print_info "Staging ParFlow..."

PARFLOW_DIR="$INSTALLS_DIR/parflow"
if [ -d "$PARFLOW_DIR" ]; then
    stage_binary "$PARFLOW_DIR/bin/parflow" "parflow" "ParFlow" \
        || stage_binary "$PARFLOW_DIR/bin/parflow.exe" "parflow.exe" "ParFlow" \
        || print_warning "ParFlow binary not found"
    stage_license "$PARFLOW_DIR" "ParFlow"
else
    print_warning "ParFlow not installed"
fi

# ============================================================================
# Stage CLM-ParFlow
# ============================================================================
print_info "Staging CLM-ParFlow..."

CLMPARFLOW_DIR="$INSTALLS_DIR/clmparflow"
if [ -d "$CLMPARFLOW_DIR" ]; then
    stage_binary "$CLMPARFLOW_DIR/bin/parflow" "parflow-clm" "CLM-ParFlow" \
        || stage_binary "$CLMPARFLOW_DIR/bin/parflow.exe" "parflow-clm.exe" "CLM-ParFlow" \
        || print_warning "CLM-ParFlow binary not found"
    stage_license "$CLMPARFLOW_DIR" "CLM-ParFlow"
else
    print_warning "CLM-ParFlow not installed"
fi

# ============================================================================
# Stage PIHM
# ============================================================================
print_info "Staging PIHM..."

PIHM_DIR="$INSTALLS_DIR/pihm"
if [ -d "$PIHM_DIR" ]; then
    stage_binary "$PIHM_DIR/bin/pihm" "pihm" "PIHM" || print_warning "PIHM binary not found"
    stage_license "$PIHM_DIR" "PIHM"
else
    print_warning "PIHM not installed"
fi

# ============================================================================
# Stage GSFLOW
# ============================================================================
print_info "Staging GSFLOW..."

GSFLOW_DIR="$INSTALLS_DIR/gsflow"
if [ -d "$GSFLOW_DIR" ]; then
    stage_binary "$GSFLOW_DIR/bin/gsflow" "gsflow" "GSFLOW" || print_warning "GSFLOW binary not found"
    stage_license "$GSFLOW_DIR" "GSFLOW"
else
    print_warning "GSFLOW not installed"
fi

# ============================================================================
# Stage CLM
# ============================================================================
print_info "Staging CLM..."

CLM_DIR="$INSTALLS_DIR/clm"
if [ -d "$CLM_DIR" ]; then
    if stage_binary "$CLM_DIR/bin/cesm.exe" "clm" "CLM"; then
        :
    elif stage_binary "$CLM_DIR/bld/cesm.exe" "clm" "CLM"; then
        :
    else
        print_warning "CLM binary not found"
    fi
    stage_license "$CLM_DIR" "CLM"
else
    print_warning "CLM not installed"
fi

# ============================================================================
# Stage MPI Runtime (for WRF-Hydro, ParFlow, CLMParFlow, TauDEM)
# ============================================================================
# If any MPI-dependent model was staged, bundle the MPI runtime so npm users
# get a fully working MPI stack without a separate OpenMPI install.

MPI_NEEDED=0
for mpi_dep_bin in wrf_hydro.exe parflow pitremove areadinf; do
    if [ -f "bin/$mpi_dep_bin" ]; then
        MPI_NEEDED=1
        break
    fi
done

if [ $MPI_NEEDED -eq 1 ]; then
    print_info "Staging MPI runtime..."

    # Find system mpirun
    MPIRUN_SRC="$(command -v mpirun 2>/dev/null || true)"
    if [ -n "$MPIRUN_SRC" ]; then
        # Resolve symlinks to get the real binary
        MPIRUN_REAL="$(_realpath "$MPIRUN_SRC")"
        MPIRUN_BINDIR="$(dirname "$MPIRUN_REAL")"
        MPIRUN_PREFIX="$(dirname "$MPIRUN_BINDIR")"

        # Stage mpirun
        if [ ! -f "bin/mpirun" ]; then
            cp -L "$MPIRUN_REAL" "bin/mpirun"
            chmod +x "bin/mpirun"
            STAGED_COUNT=$((STAGED_COUNT + 1))
            print_success "Staged mpirun (from $MPIRUN_REAL)"
        fi

        # Stage prted (OpenMPI 5.x) or orted (OpenMPI 4.x) — the daemon
        # mpirun forks for local launches
        for daemon in prted orted; do
            DAEMON_PATH="$MPIRUN_BINDIR/$daemon"
            if [ -f "$DAEMON_PATH" ] && [ ! -f "bin/$daemon" ]; then
                cp -L "$DAEMON_PATH" "bin/$daemon"
                chmod +x "bin/$daemon"
                STAGED_COUNT=$((STAGED_COUNT + 1))
                print_success "Staged $daemon (MPI daemon)"
            fi
        done

        # Copy share/openmpi/ and share/prte/ data files (help texts, ~1 MB)
        for share_subdir in openmpi prte; do
            SHARE_SRC="$MPIRUN_PREFIX/share/$share_subdir"
            if [ -d "$SHARE_SRC" ]; then
                mkdir -p "share/$share_subdir"
                cp -R "$SHARE_SRC/"* "share/$share_subdir/" 2>/dev/null || true
                print_success "Staged share/$share_subdir/ data files"
            fi
        done
    else
        print_warning "mpirun not found on PATH — MPI runtime not bundled"
        print_warning "MPI-dependent models will require a system MPI install"
    fi
else
    print_info "No MPI-dependent models staged — skipping MPI runtime"
fi

# ============================================================================
# Stage additional tools (if needed in future)
# ============================================================================

# GIStool (script only)
GISTOOL_DIR="$INSTALLS_DIR/gistool"
if [ -d "$GISTOOL_DIR" ] && [ -f "$GISTOOL_DIR/extract-gis.sh" ]; then
    print_info "Staging GIStool..."
    stage_binary "$GISTOOL_DIR/extract-gis.sh" "extract-gis.sh" "GIStool"
    stage_license "$GISTOOL_DIR" "GIStool"
fi

# Datatool (script only)
DATATOOL_DIR="$INSTALLS_DIR/datatool"
if [ -d "$DATATOOL_DIR" ] && [ -f "$DATATOOL_DIR/extract-dataset.sh" ]; then
    print_info "Staging Datatool..."
    stage_binary "$DATATOOL_DIR/extract-dataset.sh" "extract-dataset.sh" "Datatool"
    stage_license "$DATATOOL_DIR" "Datatool"
fi

# ============================================================================
# Stage udunits2 XML data (required by NGEN, SUMMA, and other models)
# ============================================================================
print_info "Staging udunits2 data files..."

UDUNITS_XML=""
# Search common locations for udunits2.xml
for udunits_candidate in \
    /opt/homebrew/share/udunits/udunits2.xml \
    /opt/homebrew/opt/udunits/share/udunits/udunits2.xml \
    /usr/share/xml/udunits/udunits2.xml \
    /usr/local/share/udunits/udunits2.xml \
    "${CONDA_PREFIX:-/dev/null}/share/udunits/udunits2.xml"; do
    if [ -f "$udunits_candidate" ]; then
        UDUNITS_XML="$udunits_candidate"
        break
    fi
done

if [ -n "$UDUNITS_XML" ]; then
    UDUNITS_SRC_DIR="$(dirname "$UDUNITS_XML")"
    mkdir -p "share/udunits"
    cp "$UDUNITS_SRC_DIR"/*.xml "share/udunits/" 2>/dev/null || true
    UDUNITS_COUNT="$(ls share/udunits/*.xml 2>/dev/null | wc -l | tr -d ' ')"
    print_success "Staged $UDUNITS_COUNT udunits2 XML files from $UDUNITS_SRC_DIR"
else
    print_warning "udunits2.xml not found — NGEN/SUMMA may fail at runtime"
    print_warning "Install udunits: brew install udunits (macOS) or apt install libudunits2-dev (Linux)"
fi

# ============================================================================
# Copy toolchain metadata
# ============================================================================
print_info "Copying toolchain metadata..."

TOOLCHAIN_SRC="$INSTALLS_DIR/toolchain.json"
if [ -f "$TOOLCHAIN_SRC" ]; then
    cp "$TOOLCHAIN_SRC" .
    print_success "Copied toolchain.json"
else
    print_error "toolchain.json not found at $TOOLCHAIN_SRC"
    print_error "Run scripts/generate_toolchain.sh first"
    exit 1
fi

# ============================================================================
# Bundle Non-System Shared Library Dependencies
# ============================================================================
print_info "Bundling non-system shared library dependencies..."

OS_BUNDLE="$(uname -s)"

if [ "$OS_BUNDLE" = "Darwin" ]; then
    # macOS: recursively discover and bundle Homebrew / non-system dylibs
    BUNDLE_ROUND=0
    BUNDLE_CHANGED=1
    while [ $BUNDLE_CHANGED -eq 1 ]; do
        BUNDLE_CHANGED=0
        BUNDLE_ROUND=$((BUNDLE_ROUND + 1))
        print_info "Dependency scan round $BUNDLE_ROUND..."

        for macho in bin/* lib/*.dylib; do
            [ -f "$macho" ] || continue
            [ -L "$macho" ] && continue
            file "$macho" | grep -q "Mach-O" || continue

            # Find non-system dependencies (anything not @rpath, /usr/lib, /System)
            DEPS="$(otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' \
                | grep -vE '^(@rpath/|@executable_path/|@loader_path/|/usr/lib/|/System/)' \
                || true)"

            for dep in $DEPS; do
                dep_basename="$(basename "$dep")"
                # Skip if already bundled
                [ -f "lib/$dep_basename" ] && continue

                if [ -f "$dep" ]; then
                    # Resolve symlinks to copy the actual file
                    cp -L "$dep" "lib/$dep_basename"
                    chmod +x "lib/$dep_basename"
                    BUNDLE_CHANGED=1
                    print_success "Bundled $dep_basename (from $dep)"
                else
                    print_warning "Dependency not found: $dep (referenced by $(basename "$macho"))"
                fi
            done

            # Also resolve @rpath/ references to files missing from lib/
            # (some Homebrew libs like libgfortran already use @rpath internally)
            RPATH_DEPS="$(otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' \
                | grep '^@rpath/' | sed 's|^@rpath/||' \
                || true)"

            for rdep_name in $RPATH_DEPS; do
                [ -f "lib/$rdep_name" ] && continue
                # Search Homebrew for this library
                found="$(find /opt/homebrew/opt /opt/homebrew/lib /usr/local/lib \
                    -name "$rdep_name" -not -type d 2>/dev/null | head -1 || true)"
                if [ -n "$found" ]; then
                    cp -L "$found" "lib/$rdep_name"
                    chmod +x "lib/$rdep_name"
                    BUNDLE_CHANGED=1
                    print_success "Bundled $rdep_name (from $found, resolved @rpath ref)"
                fi
            done
        done
    done
    print_success "Dependency bundling complete ($BUNDLE_ROUND rounds)"

elif [ "$OS_BUNDLE" = "Linux" ]; then
    # Linux: bundle the FULL recursive closure of non-system shared libraries so
    # the tarball is self-contained on a clean distro.
    #
    # The previous approach excluded everything under /usr/lib and then
    # force-bundled a hardcoded allowlist (libnetcdf, libcurl, ...). That missed
    # transitive deps that live in system paths but are NOT guaranteed present on
    # the target distro — e.g. libcurl-gnutls pulls libxml2/libssh/libldap/
    # liblber/librtmp/libpsl, none of which were bundled, so the binaries failed
    # to load on Debian/RHEL. (#156 G6 release-pipeline finding.)
    #
    # Correct policy: bundle EVERY resolved dependency except the core
    # glibc/loader set, which must always come from the host (bundling libc/libm/
    # ld-linux would break the dynamic linker). The round loop re-scans newly
    # bundled libs so the transitive closure is pulled in.
    if command -v ldd >/dev/null 2>&1; then
        # Core glibc/loader libraries — never bundle these.
        SYSTEM_LIB_RE='^(linux-vdso|ld-linux.*|libc|libm|libdl|libpthread|librt|libresolv|libutil|libnsl|libanl|libBrokenLocale|libmvec)\.so'
        BUNDLE_ROUND=0
        BUNDLE_CHANGED=1
        while [ $BUNDLE_CHANGED -eq 1 ]; do
            BUNDLE_CHANGED=0
            BUNDLE_ROUND=$((BUNDLE_ROUND + 1))
            print_info "Dependency scan round $BUNDLE_ROUND..."

            for elf in bin/* lib/*.so lib/*.so.*; do
                [ -f "$elf" ] || continue
                [ -L "$elf" ] && continue
                file "$elf" | grep -q "ELF" || continue

                # Every resolved dependency path (column 3 of "name => /path (0x..)").
                while IFS= read -r dep; do
                    [ -z "$dep" ] && continue
                    dep_basename="$(basename "$dep")"

                    # Skip the core glibc/loader libs (must come from the host).
                    if printf '%s' "$dep_basename" | grep -qE "$SYSTEM_LIB_RE"; then
                        continue
                    fi

                    [ -f "lib/$dep_basename" ] && continue
                    [ -f "$dep" ] || continue

                    cp -L "$dep" "lib/$dep_basename"
                    chmod +x "lib/$dep_basename"
                    BUNDLE_CHANGED=1
                    print_success "Bundled $dep_basename (from $dep)"
                done < <(ldd "$elf" 2>/dev/null | awk '/=>/ && $3 ~ /^\// {print $3}')
            done
        done
        print_success "Dependency bundling complete ($BUNDLE_ROUND rounds)"
    fi

elif printf '%s' "$OS_BUNDLE" | grep -qiE 'mingw|msys|cygwin'; then
    # Windows: bundle the recursive closure of non-system DLLs NEXT TO the .exe
    # files (Windows resolves DLLs from the executable's own directory), so the
    # tarball runs on a machine without MSYS2/mingw-w64 installed. ntldd -R walks
    # the dependency tree; we skip Windows system DLLs (kernel32, msvcrt, api-ms-*,
    # ...) and the MS-MPI runtime (provided by the separately-installed MS-MPI),
    # bundling everything else (libgfortran, libstdc++, libnetcdf, libgcc_s, ...).
    _ntldd="$(command -v ntldd || echo /c/msys64/mingw64/bin/ntldd.exe)"
    if [ -x "$_ntldd" ] || command -v ntldd >/dev/null 2>&1; then
        # System DLLs that always come from Windows itself — never bundle.
        WIN_SYS_DLL_RE='^(kernel32|kernelbase|ntdll|msvcrt|user32|advapi32|ws2_32|shell32|shlwapi|ole32|oleaut32|gdi32|crypt32|secur32|bcrypt|rpcrt4|sechost|combase|ucrtbase|dbghelp|version|imm32|setupapi|userenv|iphlpapi|dnsapi|winmm|comdlg32|comctl32|powrprof|psapi|wsock32|mpr|wldap32|normaliz|netapi32|pdh|wtsapi32|dwmapi|uxtheme|msmpi|api-ms-.*|ext-ms-.*)\.dll$'

        # Enumerate the images to walk by CONTENT, not by filename extension.
        #
        # This used to be `ls bin/*.exe bin/*.dll lib/*.dll`, which was wrong:
        # stage_binary() deliberately drops the .exe suffix so the staged tool
        # names match what the runners and the npm shims look up (bin/hype,
        # bin/summa, bin/fuse, ...). Every Fortran model therefore landed in
        # bin/ WITHOUT an extension and was skipped by the glob, so its imports
        # were never walked. The visible symptom was libnetcdff-7.dll never
        # being bundled at all (nothing with a .exe suffix imports it) — HYPE
        # then aborted with 0xC0000135 before main(), printing nothing, which
        # calibration recorded as a -9999 score rather than a failure.
        # `file` is not guaranteed on the Windows runner's PATH, and silently
        # classifying nothing as a PE image would silently bundle nothing, so
        # fall back to the DOS header magic.
        _is_pe() {
            if command -v file >/dev/null 2>&1; then
                file -b "$1" 2>/dev/null | grep -qE '^PE32'
            else
                [ "$(head -c 2 "$1" 2>/dev/null)" = "MZ" ]
            fi
        }
        _orig_bins=""
        for _cand in bin/* lib/*; do
            [ -f "$_cand" ] || continue
            [ -L "$_cand" ] && continue
            _is_pe "$_cand" || continue
            _orig_bins="$_orig_bins $_cand"
        done
        print_info "Walking imports of $(printf '%s' "$_orig_bins" | wc -w) PE images"

        # Direct imports of the staged images, lowercased. Used only to keep the
        # "unresolved" warning below honest: `ntldd -R` recurses THROUGH Windows
        # system DLLs and reports their optional/servicing-only components
        # (PdmUtilities.dll, HvsiFileTrust.dll, ...) as "not found" on every
        # machine. Those are irrelevant; a direct import of ours is not.
        _direct_imports=" "
        for exe in $_orig_bins; do
            for _imp in $(objdump -p "$exe" 2>/dev/null | awk '/DLL Name:/ {print tolower($3)}'); do
                case "$_direct_imports" in *" $_imp "*) ;; *) _direct_imports="$_direct_imports$_imp " ;; esac
            done
        done

        # `ntldd -R` already emits the FULL recursive closure, so a single pass
        # over the ORIGINAL staged binaries is sufficient. (A re-scanning round
        # loop is O(n^2) over GDAL's ~100-DLL closure and effectively hangs.)
        # Collect the union of resolved DLL paths first, then copy uniques once.
        _seen_dlls=" "
        _unresolved=""
        for exe in $_orig_bins; do
            [ -f "$exe" ] || continue
            # ntldd lines look like "  name.dll => C:\path\name.dll (0x...)" or
            # "  name.dll => not found". Parse both halves: the old
            # `awk '{print $3}'` turned "not found" into the literal path "not",
            # so an unresolvable dependency was silently dropped instead of
            # reported.
            while IFS='|' read -r dep_base dep_path; do
                [ -z "$dep_base" ] && continue
                case "$_seen_dlls" in *" $dep_base "*) continue ;; esac
                _seen_dlls="$_seen_dlls$dep_base "
                printf '%s' "$dep_base" | grep -qiE "$WIN_SYS_DLL_RE" && continue
                [ -f "bin/$dep_base" ] && continue

                if [ "$dep_path" = "not found" ]; then
                    _dep_lc="$(printf '%s' "$dep_base" | tr '[:upper:]' '[:lower:]')"
                    case "$_direct_imports" in
                        *" $_dep_lc "*) _unresolved="$_unresolved $dep_base" ;;
                    esac
                    continue
                fi

                dep_unix="$dep_path"
                case "$dep_path" in
                    [A-Za-z]:\\*|[A-Za-z]:/*) dep_unix="$(cygpath -u "$dep_path" 2>/dev/null || echo "$dep_path")" ;;
                esac
                [ -f "$dep_unix" ] || continue
                # Only bundle DLLs we shipped (under the mingw-w64/msys tree).
                case "$dep_unix" in
                    */msys64/*|*/mingw64/*|*/ucrt64/*) ;;
                    *) continue ;;
                esac
                cp -L "$dep_unix" "bin/$dep_base"
                chmod +x "bin/$dep_base"
                print_success "Bundled $dep_base"
            done < <("$_ntldd" -R "$exe" 2>/dev/null | tr -d '\r' \
                | sed -n 's/^[[:space:]]*\([^ ][^ ]*\) => \(.*\)$/\1|\2/p' \
                | sed 's/ (0x[0-9a-fA-F]*)$//')
        done
        if [ -n "$_unresolved" ]; then
            print_warning "Dependencies ntldd could not resolve on this runner:$_unresolved"
        fi
        print_success "DLL bundling complete"
    else
        print_warning "ntldd not found — Windows DLLs not bundled (binaries may not relocate)"
    fi

    # Hard gate: prove every packaged PE image can resolve all of its imports.
    # A missing DLL does not fail loudly on Windows — the loader aborts with
    # 0xC0000135 before main(), emitting nothing at all — so this must be a
    # static check, and it must fail the release rather than warn.
    _closure_check="$SCRIPT_DIR/check_windows_dll_closure.sh"
    if [ -f "$_closure_check" ]; then
        chmod +x "$_closure_check" 2>/dev/null || true
        bash "$_closure_check" "$STAGE_DIR/bin" "$STAGE_DIR/lib" || {
            print_error "Packaged Windows binaries have unsatisfied DLL imports (see above)"
            exit 1
        }
    else
        print_warning "check_windows_dll_closure.sh not found — skipping import-closure gate"
    fi
fi

echo ""

# ============================================================================
# Fix Binary Portability (RPATH rewriting)
# ============================================================================
print_info "Fixing binary portability (RPATH rewriting)..."

OS_TYPE="$(uname -s)"

if [ "$OS_TYPE" = "Darwin" ]; then
    # macOS: use install_name_tool to rewrite RPATHs and library references

    # Collect all Mach-O files (binaries + libraries)
    ALL_MACHO=()
    for f in bin/* lib/*.dylib; do
        [ -f "$f" ] || continue
        [ -L "$f" ] && continue  # skip symlinks
        file "$f" | grep -q "Mach-O" || continue
        ALL_MACHO+=("$f")
    done

    # Step 1: Fix RPATHs on all Mach-O files
    for macho in "${ALL_MACHO[@]}"; do
        macho_name="$(basename "$macho")"

        # Delete all existing RPATHs (they point to CI build paths)
        EXISTING_RPATHS="$(otool -l "$macho" 2>/dev/null | grep -A2 LC_RPATH | grep 'path ' | awk '{print $2}' || true)"
        if [ -n "$EXISTING_RPATHS" ]; then
            while IFS= read -r rpath; do
                install_name_tool -delete_rpath "$rpath" "$macho" 2>/dev/null || true
            done <<< "$EXISTING_RPATHS"
        fi

        # Add relocatable RPATH
        case "$macho" in
            bin/*)
                install_name_tool -add_rpath "@executable_path/../lib" "$macho" 2>/dev/null || true
                print_success "$macho_name: RPATH → @executable_path/../lib"
                ;;
            lib/*)
                install_name_tool -add_rpath "@loader_path" "$macho" 2>/dev/null || true
                print_success "$macho_name: RPATH → @loader_path"
                ;;
        esac
    done

    # Step 2: Fix install names on all libraries
    for lib in lib/*.dylib; do
        [ -f "$lib" ] || continue
        [ -L "$lib" ] && continue
        lib_name="$(basename "$lib")"
        install_name_tool -id "@rpath/$lib_name" "$lib" 2>/dev/null || true
    done

    # Step 3: Rewrite ALL non-system references in every Mach-O by basename matching
    # This catches both /opt/homebrew/opt/... and /opt/homebrew/Cellar/... paths
    for macho in "${ALL_MACHO[@]}"; do
        macho_name="$(basename "$macho")"
        REWRITTEN=0
        while IFS= read -r dep; do
            # Skip already-relocatable and system deps
            case "$dep" in
                @rpath/*|@executable_path/*|@loader_path/*|/usr/lib/*|/System/*) continue ;;
            esac
            dep_base="$(basename "$dep")"
            if [ -f "lib/$dep_base" ]; then
                install_name_tool -change "$dep" "@rpath/$dep_base" "$macho" 2>/dev/null || true
                REWRITTEN=$((REWRITTEN + 1))
            fi
        done < <(otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}')
        if [ $REWRITTEN -gt 0 ]; then
            print_success "$macho_name: rewrote $REWRITTEN references → @rpath/"
        fi
    done

    # Step 4: Re-sign all Mach-O files after path rewriting
    # install_name_tool invalidates existing code signatures. macOS Sequoia+
    # kills any process that loads a dylib with an invalid signature (SIGKILL).
    print_info "Re-signing Mach-O files (ad-hoc) after path rewriting..."
    SIGNED=0
    for macho in "${ALL_MACHO[@]}"; do
        codesign --force --sign - "$macho" 2>/dev/null || true
        SIGNED=$((SIGNED + 1))
    done
    print_success "Re-signed $SIGNED Mach-O files"

elif [ "$OS_TYPE" = "Linux" ]; then
    # Linux: use patchelf to rewrite RPATHs
    if command -v patchelf >/dev/null 2>&1; then
        for binary in bin/*; do
            [ -f "$binary" ] || continue
            binary_name="$(basename "$binary")"

            # Skip non-ELF files
            file "$binary" | grep -q "ELF" || continue

            # Set RPATH to $ORIGIN/../lib
            patchelf --set-rpath '$ORIGIN/../lib' "$binary" 2>/dev/null || true
            print_success "$binary_name: set RPATH to \$ORIGIN/../lib"
        done

        # Match versioned sonames too (libnetcdf.so.18, libnetcdff.so.7, the
        # bundled curl deps, ...). The old `lib/*.so` glob only caught unversioned
        # names like libsumma.so, leaving every bundled SYSTEM library with an
        # empty RPATH — so a transitively-needed lib (e.g. libnetcdff -> libnetcdf)
        # fell back to system paths and failed on a clean machine. They all live
        # in the same dir, so $ORIGIN is correct for each.
        for lib in lib/*.so lib/*.so.*; do
            [ -f "$lib" ] || continue
            [ -L "$lib" ] && continue
            file "$lib" | grep -q "ELF" || continue
            lib_name="$(basename "$lib")"

            patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null \
                && print_success "$lib_name: set RPATH to \$ORIGIN" \
                || print_warning "$lib_name: patchelf --set-rpath failed"
        done
    else
        print_warning "patchelf not available — skipping Linux RPATH fix"
        print_warning "Install patchelf to make binaries relocatable"
    fi
fi

# ============================================================================
# Create README
# ============================================================================
print_info "Creating README..."

cat > README.md <<'EOF'
# SYMFLUENCE Tools

Pre-built hydrological modeling tools for SYMFLUENCE.

## Contents

This archive contains compiled binaries for:
- **SUMMA**: Structure for Unifying Multiple Modeling Alternatives
- **mizuRoute**: River network routing model
- **FUSE**: Framework for Understanding Structural Errors
- **NGEN**: NextGen National Water Model Framework
- **HYPE**: Hydrological Predictions for the Environment
- **MESH**: Modelling Environmental changes of the Surface and Hydrology
- **TauDEM**: Terrain Analysis Using Digital Elevation Models
- **VIC**: Variable Infiltration Capacity model
- **SWAT**: Soil and Water Assessment Tool
- **PRMS**: Precipitation-Runoff Modeling System
- **mHM**: mesoscale Hydrologic Model
- **CRHM**: Cold Regions Hydrological Model
- **WRF-Hydro**: Weather Research and Forecasting Hydrological model
- **WATFLOOD**: Waterloo Flood forecasting model
- **MODFLOW**: USGS Modular Groundwater Flow Model (mf6)
- **ParFlow**: Parallel subsurface flow model
- **PIHM**: Penn State Integrated Hydrologic Model
- **GSFLOW**: Coupled Groundwater and Surface-water Flow model
- **CLM**: Community Land Model
- **RHESSys**: Regional Hydro-Ecologic Simulation System
- **WMFire**: Wildfire Module (shared library)

Not all models build on all platforms. Check the staging summary for availability.

## Installation

### Manual Installation

```bash
# Extract archive
tar -xzf symfluence-tools-*.tar.gz

# Add to PATH (lib/ contains shared libraries loaded via RPATH)
export PATH="$PWD/symfluence-tools/bin:$PATH"

# Verify
summa --version
```

### npm Installation (Recommended)

```bash
npm install -g symfluence
```

## System Requirements

See `toolchain.json` for build details.

### Linux
- glibc ≥ 2.39
- NetCDF ≥ 4.8
- HDF5 ≥ 1.10

### macOS
- macOS 12+ (Monterey)
- Homebrew: `netcdf netcdf-fortran hdf5`

## Toolchain

Build metadata is available in `toolchain.json`:
- Tool versions (commit hashes)
- Compiler versions
- Library versions
- Build timestamp

## Licenses

See `LICENSES/` directory for individual tool licenses.

## Support

- Issues: https://github.com/symfluence-org/SYMFLUENCE/issues
- Documentation: https://github.com/symfluence-org/SYMFLUENCE

---

🤖 Generated with SYMFLUENCE release automation
EOF

print_success "Created README.md"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Staging Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_success "Platform: $PLATFORM"
print_success "Staged binaries: $STAGED_COUNT"
print_success "Output directory: $STAGE_DIR"
echo ""
print_info "Staged binaries:"
ls -lh bin/ | tail -n +2 | awk '{printf "  %s (%s)\n", $9, $5}'
echo ""
if ls lib/* >/dev/null 2>&1; then
    print_info "Staged libraries:"
    ls -lh lib/ | tail -n +2 | awk '{printf "  %s (%s)\n", $9, $5}'
    echo ""
fi
print_info "Licenses:"
ls -1 LICENSES/ | sed 's/^/  /'
echo ""
print_info "Directory structure:"
tree -L 2 . 2>/dev/null || find . -maxdepth 2 -type f -o -type d
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $STAGED_COUNT -eq 0 ]; then
    print_error "No binaries were staged! Check installation directory."
    exit 1
fi

print_success "Staging complete!"
