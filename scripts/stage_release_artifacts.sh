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
    # WMFire is a shared library, but we'll stage it for completeness
    if [ "$(uname)" = "Darwin" ] && [ -f "$WMFIRE_DIR/lib/libwmfire.dylib" ]; then
        stage_binary "$WMFIRE_DIR/lib/libwmfire.dylib" "libwmfire.dylib" "WMFire"
    elif [ -f "$WMFIRE_DIR/lib/libwmfire.dll" ]; then
        stage_binary "$WMFIRE_DIR/lib/libwmfire.dll" "libwmfire.dll" "WMFire"
    elif [ -f "$WMFIRE_DIR/lib/libwmfire.so" ]; then
        stage_binary "$WMFIRE_DIR/lib/libwmfire.so" "libwmfire.so" "WMFire"
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
    if stage_binary "$WATFLOOD_DIR/bin/watflood" "watflood" "WATFLOOD"; then
        :
    elif stage_binary "$WATFLOOD_DIR/bin/charm" "watflood" "WATFLOOD"; then
        :
    else
        print_warning "WATFLOOD binary not found"
    fi
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
    stage_binary "$PARFLOW_DIR/bin/parflow" "parflow" "ParFlow" || print_warning "ParFlow binary not found"
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
    stage_binary "$CLMPARFLOW_DIR/bin/parflow" "parflow-clm" "CLM-ParFlow" || print_warning "CLM-ParFlow binary not found"
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
    # Linux: bundle non-system shared libraries (e.g. from conda, /opt, etc.)
    if command -v ldd >/dev/null 2>&1; then
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

                # Find non-system dependencies
                DEPS="$(ldd "$elf" 2>/dev/null | grep '=>' | awk '{print $3}' \
                    | grep -vE '^(/usr/lib|/lib/|/lib64/)' \
                    || true)"

                for dep in $DEPS; do
                    [ -z "$dep" ] && continue
                    dep_basename="$(basename "$dep")"
                    [ -f "lib/$dep_basename" ] && continue

                    if [ -f "$dep" ]; then
                        cp -L "$dep" "lib/$dep_basename"
                        chmod +x "lib/$dep_basename"
                        BUNDLE_CHANGED=1
                        print_success "Bundled $dep_basename (from $dep)"
                    fi
                done
            done
        done
        print_success "Dependency bundling complete ($BUNDLE_ROUND rounds)"
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

    # Step 2: Fix install names on all libraries, and rewrite references everywhere
    for lib in lib/*.dylib; do
        [ -f "$lib" ] || continue
        [ -L "$lib" ] && continue
        lib_name="$(basename "$lib")"

        # Get the current install name (absolute CI path)
        OLD_ID="$(otool -D "$lib" 2>/dev/null | tail -1 || true)"
        NEW_ID="@rpath/$lib_name"

        # Set install name to @rpath/libname
        install_name_tool -id "$NEW_ID" "$lib" 2>/dev/null || true

        # Update references in ALL Mach-O files (binaries AND other libraries)
        if [ -n "$OLD_ID" ] && [ "$OLD_ID" != "$NEW_ID" ]; then
            for macho in "${ALL_MACHO[@]}"; do
                install_name_tool -change "$OLD_ID" "$NEW_ID" "$macho" 2>/dev/null || true
            done
            print_success "$lib_name: install name → $NEW_ID (references updated)"
        else
            print_success "$lib_name: install name → $NEW_ID"
        fi
    done

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

        for lib in lib/*.so; do
            [ -f "$lib" ] || continue
            lib_name="$(basename "$lib")"

            patchelf --set-rpath '$ORIGIN' "$lib" 2>/dev/null || true
            print_success "$lib_name: set RPATH to \$ORIGIN"
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

- Issues: https://github.com/DarriEy/SYMFLUENCE/issues
- Documentation: https://github.com/DarriEy/SYMFLUENCE

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
