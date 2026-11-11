# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shell command payloads for infrastructure external tools."""
from __future__ import annotations

SUNDIALS_BUILD_COMMAND = r'''
# Build SUNDIALS from release tarball (shared libs OK; SUMMA will link).
set -e

SUNDIALS_VER=7.4.0

# Tool install root, e.g.  .../SYMFLUENCE_data/installs/sundials
SUNDIALS_ROOT_DIR="$(pwd)"

# Actual install prefix, consistent with default_path_suffix and SUMMA:
#   .../installs/sundials/install/sundials
SUNDIALS_PREFIX="${SUNDIALS_ROOT_DIR}/install/sundials"
mkdir -p "${SUNDIALS_PREFIX}"

rm -f "v${SUNDIALS_VER}.tar.gz" || true
curl -fsSL -o "v${SUNDIALS_VER}.tar.gz" "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz" \
  || wget -q -O "v${SUNDIALS_VER}.tar.gz" "https://github.com/LLNL/sundials/archive/refs/tags/v${SUNDIALS_VER}.tar.gz"

# Exclude doc directory (contains symlinks that fail on Windows/MSYS2)
# and examples (not needed, and CXX_parallel paths are too long for Windows).
# Allow tar errors for non-critical files (LICENSE/NOTICE on Windows).
tar -xzf "v${SUNDIALS_VER}.tar.gz" --exclude="*/doc/*" --exclude="*/examples/*" || true
# Verify core source was extracted
if [ ! -d "sundials-${SUNDIALS_VER}/src" ]; then
    echo "ERROR: SUNDIALS source extraction failed"
    exit 1
fi
cd "sundials-${SUNDIALS_VER}"

rm -rf build && mkdir build && cd build

# On Windows/MinGW, SUNDIALS DLLs do not export Fortran module procedure
# symbols (only C wrapper functions). Build static-only to avoid this issue.
# Also, Windows LLP64 uses 4-byte long, so INDEX_SIZE must be 32.
SUNDIALS_SHARED=ON
SUNDIALS_IDX_SIZE=64
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*)
        SUNDIALS_SHARED=OFF
        SUNDIALS_IDX_SIZE=32
        ;;
esac

# If LDFLAGS contains -static-libgcc (set by fix_libgcc_glibc_mismatch),
# pass it to CMake so Fortran link tests succeed.
_SUNDIALS_EXTRA_CMAKE=""
if echo "${LDFLAGS:-}" | grep -q static-libgcc; then
    _STATIC_FLAGS="-static-libgcc -static-libstdc++"
    _SUNDIALS_EXTRA_CMAKE="-DCMAKE_EXE_LINKER_FLAGS=$_STATIC_FLAGS -DCMAKE_SHARED_LINKER_FLAGS=$_STATIC_FLAGS -DCMAKE_Fortran_FLAGS=$_STATIC_FLAGS"
fi

cmake .. \
  -DBUILD_FORTRAN_MODULE_INTERFACE=ON \
  -DCMAKE_Fortran_COMPILER="$FC" \
  -DCMAKE_INSTALL_PREFIX="${SUNDIALS_PREFIX}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=$SUNDIALS_SHARED \
  -DSUNDIALS_INDEX_SIZE=$SUNDIALS_IDX_SIZE \
  -DEXAMPLES_ENABLE_C=OFF \
  -DEXAMPLES_ENABLE_CXX=OFF \
  -DEXAMPLES_ENABLE_F2003=OFF \
  -DBUILD_TESTING=OFF \
  $_SUNDIALS_EXTRA_CMAKE

cmake --build . --target install -j ${NCORES:-4}

# Debug: show where the libs landed
[ -d "${SUNDIALS_PREFIX}/lib64" ] && ls -la "${SUNDIALS_PREFIX}/lib64" | head -20 || true
[ -d "${SUNDIALS_PREFIX}/lib" ] && ls -la "${SUNDIALS_PREFIX}/lib" | head -20 || true
'''.strip()

TAUDEM_BUILD_COMMAND = r'''
# Build TauDEM from GitHub repository
set -e

# On Compute Canada HPC, OpenMPI has broken Level Zero dependency through hwloc.
# The Level Zero library doesn't exist but hwloc was built with it enabled.
# Solution: Use --allow-shlib-undefined to ignore missing symbols in shared libs.
CMAKE_MPI_FLAGS=""
if [ -d "/cvmfs/soft.computecanada.ca" ]; then
    echo "Detected Compute Canada HPC environment"

    # Use system gcc/g++ to avoid broken mpicc dependency chain
    export CC=gcc
    export CXX=g++

    # Tell linker to allow undefined symbols in shared libraries
    # This works around the missing libze_loader.so that hwloc wants
    export LDFLAGS="-Wl,--allow-shlib-undefined ${LDFLAGS:-}"

    # Tell cmake where MPI is
    MPI_ROOT=$(dirname $(dirname $(which mpicc 2>/dev/null))) || true
    if [ -n "$MPI_ROOT" ] && [ -d "$MPI_ROOT" ]; then
        echo "Found MPI at: $MPI_ROOT"
        _TAUDEM_LINK_FLAGS="-Wl,--allow-shlib-undefined"
        # Also add static lib flags if set by fix_libgcc_glibc_mismatch
        if echo "${LDFLAGS:-}" | grep -q static-libgcc; then
            _TAUDEM_LINK_FLAGS="$_TAUDEM_LINK_FLAGS -static-libgcc -static-libstdc++"
        fi
        CMAKE_MPI_FLAGS="-DMPI_HOME=$MPI_ROOT -DCMAKE_EXE_LINKER_FLAGS=$_TAUDEM_LINK_FLAGS -DCMAKE_SHARED_LINKER_FLAGS=$_TAUDEM_LINK_FLAGS"
    fi
else
    # On other systems, mpicc/mpicxx as CC/CXX works fine
    export CC=mpicc
    export CXX=mpicxx
fi

# Patch an upstream TauDEM bug (src/tiffIO.cpp): GDALGetRasterUnitType() is called
# with the dataset handle (fh) instead of the raster band handle (bandh) obtained on
# the previous line. GDAL's C API casts the handle straight to GDALRasterBand* and
# makes a virtual call, so passing a GDALDataset* dispatches through the wrong
# vtable. This happens to survive on some Linux GDAL builds but segfaults
# immediately on Windows/MSYS2 GDAL 3.x, crashing every TauDEM tool on open.
if grep -q 'GDALGetRasterUnitType(fh)' src/tiffIO.cpp; then
    sed -i.bak 's/GDALGetRasterUnitType(fh)/GDALGetRasterUnitType(bandh)/' src/tiffIO.cpp
    rm -f src/tiffIO.cpp.bak
    echo "Patched src/tiffIO.cpp: GDALGetRasterUnitType(fh) -> GDALGetRasterUnitType(bandh)"
fi

# Upstream TauDEM writes chained comparisons such as
#   for(int i=0;i<toRecv->numCoords > 0;i++)        (src/linklib.h)
# which parse as (i < numCoords) > 0 -- semantically the same loop, but Clang
# (LLVM 16+) reports -Wparentheses "chained comparison" as an error by default,
# so streamnet.cpp fails to compile and the whole build aborts. Downgrade it to
# a warning rather than editing upstream source or silencing the diagnostic.
export CXXFLAGS="${CXXFLAGS:-} -Wno-error=parentheses"

rm -rf build && mkdir -p build
cd build

# Let CMake find MPI and GDAL
cmake -S .. -B . -DCMAKE_BUILD_TYPE=Release $CMAKE_MPI_FLAGS

# Build everything plus the two tools that sometimes get skipped by default
cmake --build . -j 2
cmake --build . --target moveoutletstostreams gagewatershed -j 2 || true

echo "Staging executables..."
mkdir -p ../bin

# Detect Windows .exe extension
EXE_SUFFIX=""
case "$(uname -s 2>/dev/null)" in
    MSYS*|MINGW*|CYGWIN*) EXE_SUFFIX=".exe" ;;
esac

# Debug: show what was built
echo "Files in build directory:"
find . -type f -name "pitremove${EXE_SUFFIX}" -o -name "streamnet${EXE_SUFFIX}" -o -name "aread8${EXE_SUFFIX}" 2>/dev/null || true

# List of expected TauDEM tools (superset — some may not exist on older commits)
tools="pitremove d8flowdir d8converge dinfconverge dinfflowdir aread8 areadinf threshold
       streamnet slopearea gridnet peukerdouglas lengtharea moveoutletstostreams gagewatershed"

copied=0
for exe in $tools;
  do
  exe_name="${exe}${EXE_SUFFIX}"
  # Find by name (cross-platform: works on macOS, Linux, WSL, Windows/MSYS2)
  # Note: piping through head masks find errors, so use sequential checks
  p=""
  # Try GNU find -executable first (Linux)
  if [ -z "$p" ]; then p="$(find . -type f -executable -name "$exe_name" 2>/dev/null | head -n1 || true)"; fi
  # Try POSIX -perm /111 (macOS/BSD/Linux)
  if [ -z "$p" ]; then p="$(find . -type f -perm /111 -name "$exe_name" 2>/dev/null | head -n1 || true)"; fi
  # Fallback: find by name only (then chmod +x below handles permissions)
  if [ -z "$p" ]; then p="$(find . -type f -name "$exe_name" ! -path "*/CMakeFiles/*" 2>/dev/null | head -n1 || true)"; fi
  if [ -n "$p" ] && [ -f "$p" ]; then
    cp -f "$p" ../bin/
    chmod +x "../bin/$exe_name"
    copied=$((copied+1))
    echo "  Copied: $exe_name"
  fi
done

echo "Copied $copied executables"

# Final sanity
ls -la ../bin/ || true
if [ ! -f "../bin/pitremove${EXE_SUFFIX}" ] || [ ! -f "../bin/streamnet${EXE_SUFFIX}" ]; then
  echo "TauDEM stage failed: core binaries missing" >&2
  echo "Build directory contents:"
  find . -name "pitremove*" -o -name "streamnet*" 2>/dev/null || true
  exit 1
fi
echo "TauDEM executables staged"
'''.strip()

OPENFEWS_BUILD_COMMAND = r'''
set -e

# Delft-FEWS itself is NOT downloaded or installed here: it is distributed by
# Deltares (https://oss.deltares.nl/web/delft-fews) under its own licence.
# This step only generates the SYMFLUENCE General Adapter configuration
# template and a small launcher that starts FEWS once its jars are in lib/.

echo "Generating the SYMFLUENCE General Adapter configuration for Delft-FEWS ..."
echo "NOTE: Delft-FEWS is not installed by this command. Obtain it from Deltares:"
echo "      https://oss.deltares.nl/web/delft-fews"

# Check for Java (FEWS requires a JRE)
if ! command -v java >/dev/null 2>&1; then
    echo "WARNING: Java not found. Delft-FEWS requires Java."
fi

# Directory layout: put the Delft-FEWS distribution's jars under lib/
mkdir -p bin lib config Modules

# SYMFLUENCE General Adapter module configuration (a starting point: edit the
# --config path, the exported/imported time series sets and the ID mapping to
# match the FEWS configuration it is used in).
cat > Modules/symfluence_adapter.xml << 'ADAPTER_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<generalAdapterRun xmlns="http://www.wldelft.nl/fews" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <general>
        <description>SYMFLUENCE General Adapter</description>
        <piVersion>1.24</piVersion>
    </general>
    <activities>
        <startUpActivities>
            <purgeActivity>
                <filter>%ROOT_DIR%/toModel/*</filter>
            </purgeActivity>
            <purgeActivity>
                <filter>%ROOT_DIR%/toFews/*</filter>
            </purgeActivity>
        </startUpActivities>
        <exportActivities>
            <exportNetcdfActivity>
                <exportFile>%ROOT_DIR%/toModel/forcing.nc</exportFile>
                <cfConventions>CF-1.6</cfConventions>
            </exportNetcdfActivity>
            <exportRunFileActivity>
                <exportFile>%ROOT_DIR%/run_info.xml</exportFile>
            </exportRunFileActivity>
        </exportActivities>
        <executeActivities>
            <executeActivity>
                <description>Run SYMFLUENCE adapter</description>
                <command>
                    <executable>symfluence</executable>
                    <!-- EDIT: the config argument must point at the SYMFLUENCE
                         configuration of this model; "fews run" fails without it.
                         Exporting results also needs an ID map with explicit
                         export entries (id-map argument, e.g. discharge to Q.sim). -->
                    <arguments>fews run --run-info %ROOT_DIR%/run_info.xml --config %ROOT_DIR%/symfluence_config.yaml --format netcdf-cf</arguments>
                </command>
                <timeOut>7200000</timeOut>
            </executeActivity>
        </executeActivities>
        <importActivities>
            <importNetcdfActivity>
                <importFile>%ROOT_DIR%/toFews/output.nc</importFile>
                <cfConventions>CF-1.6</cfConventions>
            </importNetcdfActivity>
        </importActivities>
    </activities>
</generalAdapterRun>
ADAPTER_EOF

# Launcher: starts Delft-FEWS only when its jars are present.
cat > bin/fews.sh << 'LAUNCHER_EOF'
#!/usr/bin/env bash
# Delft-FEWS launcher with SYMFLUENCE support
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FEWS_HOME="$(dirname "$SCRIPT_DIR")"

# Set SYMFLUENCE CLI path if available
if command -v symfluence >/dev/null 2>&1; then
    export SYMFLUENCE_CLI="$(which symfluence)"
    echo "SYMFLUENCE CLI found: $SYMFLUENCE_CLI"
fi

# Check for Java
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: Java is required to run Delft-FEWS" >&2
    exit 1
fi

# Look for FEWS jar
FEWS_JAR="$(find "$FEWS_HOME/lib" -iname "*fews*.jar" -type f 2>/dev/null | head -1)"
if [ -z "$FEWS_JAR" ]; then
    echo "ERROR: Delft-FEWS is not installed in $FEWS_HOME/lib." >&2
    echo "Delft-FEWS must be obtained from Deltares:" >&2
    echo "  https://oss.deltares.nl/web/delft-fews" >&2
    echo "Only the SYMFLUENCE General Adapter template was generated:" >&2
    echo "  $FEWS_HOME/Modules/symfluence_adapter.xml" >&2
    exit 1
fi
echo "Starting Delft-FEWS from $FEWS_JAR"
exec java -jar "$FEWS_JAR" "$@"
LAUNCHER_EOF
chmod +x bin/fews.sh

echo "SYMFLUENCE General Adapter template: $(pwd)/Modules/symfluence_adapter.xml"
echo "Delft-FEWS itself was NOT installed; obtain it from Deltares and place its"
echo "distribution under $(pwd) (jars in lib/), or set FEWS_HOME."
'''.strip()

ENZYME_BUILD_COMMAND = r'''
echo "=== Enzyme AD Build Starting ==="

ENZYME_ROOT="$(pwd)"
OS_NAME="$(uname -s)"

# ── Detect LLVM ──────────────────────────────────────────────────
echo ""
echo "=== Detecting LLVM ==="

LLVM_DIR=""
LLVM_VERSION=""
CXX_COMPILER=""
C_COMPILER=""

if [ "$OS_NAME" = "Darwin" ]; then
    # macOS: Homebrew LLVM (generic, then versioned)
    for llvm_base in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
        if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
            CXX_COMPILER="$llvm_base/bin/clang++"
            C_COMPILER="$llvm_base/bin/clang"
            [ -d "$llvm_base/lib/cmake/llvm" ] && LLVM_DIR="$llvm_base/lib/cmake/llvm"
            LLVM_VERSION=$("$CXX_COMPILER" --version | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
            echo "Found Homebrew LLVM $LLVM_VERSION at $llvm_base"
            break
        fi
    done

    if [ -z "$CXX_COMPILER" ]; then
        for ver in 21 20 19 18 17; do
            for prefix in /opt/homebrew/opt /usr/local/opt; do
                llvm_base="$prefix/llvm@$ver"
                if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
                    CXX_COMPILER="$llvm_base/bin/clang++"
                    C_COMPILER="$llvm_base/bin/clang"
                    [ -d "$llvm_base/lib/cmake/llvm" ] && LLVM_DIR="$llvm_base/lib/cmake/llvm"
                    LLVM_VERSION="$ver"
                    echo "Found Homebrew LLVM@$ver at $llvm_base"
                    break 2
                fi
            done
        done
    fi

    # Auto-install via Homebrew if not found
    if [ -z "$CXX_COMPILER" ]; then
        if command -v brew >/dev/null 2>&1; then
            echo "No Homebrew LLVM found. Installing via: brew install llvm"
            brew install llvm 2>&1
            for llvm_base in /opt/homebrew/opt/llvm /usr/local/opt/llvm; do
                if [ -d "$llvm_base" ] && [ -x "$llvm_base/bin/clang++" ]; then
                    CXX_COMPILER="$llvm_base/bin/clang++"
                    C_COMPILER="$llvm_base/bin/clang"
                    [ -d "$llvm_base/lib/cmake/llvm" ] && LLVM_DIR="$llvm_base/lib/cmake/llvm"
                    LLVM_VERSION=$("$CXX_COMPILER" --version | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
                    echo "Installed Homebrew LLVM $LLVM_VERSION"
                    break
                fi
            done
        else
            echo "WARNING: Homebrew not found. Cannot auto-install LLVM."
        fi
    fi
else
    # Linux: llvm-config (generic, then versioned)
    if command -v llvm-config >/dev/null 2>&1; then
        LLVM_DIR="$(llvm-config --cmakedir 2>/dev/null)"
        LLVM_VERSION="$(llvm-config --version 2>/dev/null | sed -nE 's/^([0-9]+).*/\1/p')"
        for cmd in "clang++-$LLVM_VERSION" clang++; do
            if command -v "$cmd" >/dev/null 2>&1; then
                CXX_COMPILER="$(which "$cmd")"
                C_COMPILER="$(which "${cmd%%++*}")"
                break
            fi
        done
        [ -n "$CXX_COMPILER" ] && echo "Found system LLVM $LLVM_VERSION"
    fi
    if [ -z "$CXX_COMPILER" ]; then
        for ver in 21 20 19 18 17; do
            if command -v "llvm-config-$ver" >/dev/null 2>&1; then
                LLVM_DIR="$(llvm-config-$ver --cmakedir 2>/dev/null)"
                LLVM_VERSION="$ver"
                if command -v "clang++-$ver" >/dev/null 2>&1; then
                    CXX_COMPILER="$(which "clang++-$ver")"
                    C_COMPILER="$(which "clang-$ver")"
                    echo "Found system LLVM $LLVM_VERSION"
                fi
                break
            fi
        done
    fi
fi

if [ -z "$LLVM_DIR" ]; then
    echo "ERROR: LLVM cmake directory not found."
    echo "Install LLVM: brew install llvm (macOS) or apt install llvm-dev (Linux)"
    exit 1
fi

echo "LLVM cmake dir: $LLVM_DIR"
echo "LLVM version:   $LLVM_VERSION"

# ── Select branch ────────────────────────────────────────────────
echo ""
echo "=== Selecting Enzyme Branch ==="

if [ "$LLVM_VERSION" -ge 20 ] 2>/dev/null; then
    ENZYME_BRANCH="main"
else
    ENZYME_BRANCH="v$LLVM_VERSION"
    # Verify remote branch exists
    if ! git ls-remote --heads origin "$ENZYME_BRANCH" 2>/dev/null | grep -q "$ENZYME_BRANCH"; then
        echo "Branch $ENZYME_BRANCH not found, using main"
        ENZYME_BRANCH="main"
    fi
fi
echo "Using branch: $ENZYME_BRANCH"
git checkout "$ENZYME_BRANCH" 2>/dev/null || \
    git fetch --depth 1 origin "$ENZYME_BRANCH" 2>/dev/null && \
    git checkout "$ENZYME_BRANCH" 2>/dev/null || \
    echo "Staying on current branch"

# ── Build Enzyme ─────────────────────────────────────────────────
echo ""
echo "=== Building Enzyme ==="

# Use _build (not build) to avoid collision with Bazel BUILD file
# on case-insensitive filesystems (macOS HFS+/APFS)
ENZYME_BUILD_DIR="$ENZYME_ROOT/enzyme/_build"
mkdir -p "$ENZYME_BUILD_DIR"
cd "$ENZYME_BUILD_DIR"

echo "Configuring with LLVM $LLVM_VERSION..."
cmake .. \
    -DLLVM_DIR="$LLVM_DIR" \
    -DCMAKE_BUILD_TYPE=Release 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Enzyme CMake configuration failed"
    exit 1
fi

# Determine parallel jobs
if [ -n "$NPROC" ]; then
    JOBS=$NPROC
elif command -v nproc >/dev/null 2>&1; then
    JOBS=$(nproc)
elif command -v sysctl >/dev/null 2>&1; then
    JOBS=$(sysctl -n hw.ncpu)
else
    JOBS=4
fi

echo "Building with $JOBS parallel jobs..."
make -j"$JOBS" 2>&1

if [ $? -ne 0 ]; then
    echo "ERROR: Enzyme build failed"
    exit 1
fi

# ── Create predictable symlinks ──────────────────────────────────
echo ""
echo "=== Creating Library Symlinks ==="

mkdir -p "$ENZYME_ROOT/lib"
cd "$ENZYME_BUILD_DIR/Enzyme"

for lib in ClangEnzyme-*.dylib LLVMEnzyme-*.dylib ClangEnzyme-*.so LLVMEnzyme-*.so; do
    if [ -f "$lib" ]; then
        # Predictable name (e.g., ClangEnzyme.dylib) for easy discovery
        base=$(echo "$lib" | sed -E 's/-[0-9]+//')
        ln -sf "$ENZYME_BUILD_DIR/Enzyme/$lib" "$ENZYME_ROOT/lib/$base"
        ln -sf "$ENZYME_BUILD_DIR/Enzyme/$lib" "$ENZYME_ROOT/lib/$lib"
        echo "Linked: lib/$base -> $lib"
    fi
done

cd "$ENZYME_ROOT"

echo ""
echo "=== Enzyme AD Build Complete ==="
echo "Installation path: $ENZYME_ROOT"
echo "LLVM version: $LLVM_VERSION"
ls -la "$ENZYME_ROOT/lib/"
'''.strip()
