#!/usr/bin/env bash
#
# Rebuild netCDF-C for mingw-w64 WITHOUT the AWS S3 SDK and install it over the
# MSYS2 prefix.
#
# ---------------------------------------------------------------------------
# Why this exists
# ---------------------------------------------------------------------------
# MSYS2's stock `mingw-w64-x86_64-netcdf` is built with -DNETCDF_ENABLE_S3_AWS=ON
# (`libnetcdf.settings` reports "S3 Support: yes / S3 SDK: aws-sdk-cpp"). netCDF
# then registers a shutdown hook that calls Aws::ShutdownAPI at
# DLL_PROCESS_DETACH. By that point RtlExitUserProcess has already terminated the
# AWS CRT worker threads, so ShutdownAPI blocks forever in
# SleepConditionVariableSRW waiting on threads that can never respond:
#
#   main -> stop_program -> libgfortran -> msvcrt!_exit
#     -> RtlExitUserProcess -> LdrShutdownProcess          (DLL_PROCESS_DETACH)
#     -> libnetcdf.dll -> libaws-cpp-sdk-core (Aws::ShutdownAPI)
#     -> libaws-crt-cpp -> libaws-c-common
#     -> SleepConditionVariableSRW                          <- blocks forever
#
# The process gets a valid exit code but its last thread never finishes, so the
# process object never signals: GetExitCodeProcess returns the code while
# WaitForSingleObject times out forever. Every model run (SUMMA, FUSE, HYPE, ...)
# leaves an unkillable zombie holding its output NetCDF file handles, which
# poisons subsequent calibration evaluations. Only a reboot clears them.
#
# Fix: build netCDF-C with S3 off. BYTERANGE stays ON (curl-only, no AWS), so
# remote-read over plain HTTP still works; only the aws-sdk-cpp code path goes.
# The resulting DLL is a drop-in: its export table is the stock table minus the
# NC_s3sdk*/NCZ_s3finalize symbols, none of which libnetcdff-7.dll imports, so
# no model binary needs relinking. Verified below before the script exits.
#
# Everything else mirrors the MSYS2 netCDF PKGBUILD's shared build so the result
# stays feature-identical to the package it replaces (DAP2/DAP4, NCZarr, HDF5,
# plugins, standard filters).
#   Reference: https://github.com/msys2/MINGW-packages/tree/master/mingw-w64-netcdf
#
# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#   scripts/build_netcdf_no_s3_mingw.sh [work_dir]
#
# Environment:
#   SYMFLUENCE_MINGW_PREFIX
#                    mingw64 prefix to build against and install into
#                    (default: C:/msys64/mingw64). Deliberately not
#                    MINGW_PREFIX: every MSYS2/Git login shell exports that for
#                    its own toolchain (Git Bash: /ucrt64), which would
#                    silently redirect the build away from the MSYS2 prefix.
#   NETCDF_VERSION   override the netCDF-C version to build. By default the
#                    version is read from the installed MSYS2 package so the
#                    replacement always matches the ABI the rest of the prefix
#                    (netcdf-fortran, nc-config, headers) was built against.
#   NETCDF_JOBS      ninja parallelism (default: nproc)
#   NETCDF_INSTALL_PREFIX
#                    where to install the result (default: the mingw64 prefix, i.e.
#                    replace the stock package in place). Point this elsewhere
#                    to rehearse the whole build + verification without
#                    modifying a working MSYS2 installation.
#
# Must run in an MSYS2/MinGW bash with the mingw64 toolchain installed.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MINGW_PREFIX="${SYMFLUENCE_MINGW_PREFIX:-C:/msys64/mingw64}"
# Unix-style view of the same prefix, for shell file tests.
if command -v cygpath >/dev/null 2>&1; then
    MINGW_UNIX="$(cygpath -u "$MINGW_PREFIX")"
else
    MINGW_UNIX="$MINGW_PREFIX"
fi

# Where the result lands. Defaults to the mingw64 prefix (the point of the
# exercise: replace the stock package in place). Overridable so the script can be
# exercised end-to-end — including the export/ABI checks, which read the BUILD
# tree, not the install tree — without disturbing a working MSYS2 installation.
INSTALL_PREFIX="${NETCDF_INSTALL_PREFIX:-$MINGW_PREFIX}"
if command -v cygpath >/dev/null 2>&1; then
    INSTALL_UNIX="$(cygpath -u "$INSTALL_PREFIX")"
else
    INSTALL_UNIX="$INSTALL_PREFIX"
fi

WORK_DIR="${1:-${RUNNER_TEMP:-/tmp}/netcdf-noS3}"
SRC_DIR="$WORK_DIR/src"
BUILD_DIR="$WORK_DIR/build"

# mingw64/bin first (the toolchain), then MSYS2's usr/bin — the workflow runs
# under Git Bash, which does not have MSYS2's usr/bin on PATH, and `pacman` is
# needed below to learn which netCDF version the prefix currently holds.
MSYS_ROOT="$(dirname "$MINGW_UNIX")"
export PATH="$MINGW_UNIX/bin:$PATH:$MSYS_ROOT/usr/bin"

# When a Git-Bash (or MSYS2-msys) shell spawns native mingw-w64 tools, TMP/TEMP
# are frequently dropped or handed over mangled, and gcc then tries to write its
# temporaries into C:\Windows\ and dies with "Cannot create temporary file ...
# Permission denied" — which surfaces as the far more confusing "the C compiler
# is not able to compile a simple test program". Pin them to a writable dir, in
# the Windows-path form the native toolchain expects. The workflow runs its
# steps under `bash -l` (Git Bash), so this is not hypothetical.
: "${TMPDIR:=${RUNNER_TEMP:-/tmp}}"
mkdir -p "$TMPDIR"
export TMPDIR
if command -v cygpath >/dev/null 2>&1; then
    TMP="$(cygpath -w "$TMPDIR")"
else
    TMP="$TMPDIR"
fi
export TMP TEMP="$TMP"

# Be explicit about the compiler rather than relying on whatever CC the caller
# exported: the whole point is that netCDF is built by the SAME mingw-w64
# toolchain that produced the prefix it is replacing.
export CC="${CC:-$MINGW_UNIX/bin/gcc.exe}"
export CXX="${CXX:-$MINGW_UNIX/bin/g++.exe}"

say()  { printf '\n=== %s ===\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

for tool in cmake ninja gcc objdump curl tar; do
    command -v "$tool" >/dev/null 2>&1 || fail "required tool '$tool' not on PATH"
done

# ---------------------------------------------------------------------------
# 1. Pick the version to build
# ---------------------------------------------------------------------------
# Match the installed MSYS2 package exactly. netcdf-fortran, the headers and the
# .cmake/.pc files in the prefix all come from that package; building a
# different netCDF version here would leave the prefix internally inconsistent.
VER="${NETCDF_VERSION:-}"
# The release workflow runs this from Git Bash, where MSYS2's pacman is not on
# PATH; fall back to the one that sits beside the mingw64 prefix.
PACMAN="$(command -v pacman 2>/dev/null || true)"
if [ -z "$PACMAN" ] && [ -x "$MINGW_UNIX/../usr/bin/pacman.exe" ]; then
    PACMAN="$MINGW_UNIX/../usr/bin/pacman.exe"
fi
if [ -z "$VER" ] && [ -n "$PACMAN" ]; then
    # Bounded: a wedged pacman would otherwise hang here until the job's
    # timeout, with nothing on stdout to say why.
    _pacman_q() {
        if command -v timeout >/dev/null 2>&1; then
            timeout 120 "$PACMAN" -Q mingw-w64-x86_64-netcdf 2>/dev/null || true
        else
            "$PACMAN" -Q mingw-w64-x86_64-netcdf 2>/dev/null || true
        fi
    }
    VER="$(_pacman_q | awk '{print $2}' | cut -d- -f1)"
fi
# Last resort: the installed package's own pkg-config file carries the same
# upstream version.
if [ -z "$VER" ] && [ -f "$MINGW_UNIX/lib/pkgconfig/netcdf.pc" ]; then
    VER="$(sed -n 's/^Version:[[:space:]]*//p' "$MINGW_UNIX/lib/pkgconfig/netcdf.pc" | tr -d '\r')"
fi
if [ -z "$VER" ]; then
    echo "  pacman: ${PACMAN:-not found}" >&2
    echo "  pkg-config file: $MINGW_UNIX/lib/pkgconfig/netcdf.pc ($([ -f "$MINGW_UNIX/lib/pkgconfig/netcdf.pc" ] && echo present || echo missing))" >&2
    fail "could not determine netCDF version (set NETCDF_VERSION)"
fi

say "Building netCDF-C $VER without the AWS S3 SDK"
echo "build against:  $MINGW_PREFIX"
echo "INSTALL INTO:   $INSTALL_PREFIX"
echo "work dir:       $WORK_DIR"

# Announcing is not enough. This script overwrites a working netCDF in place,
# and the env-prefix meant to redirect it (`NETCDF_INSTALL_PREFIX=... script`)
# is silently dropped when the calling shell and the script's interpreter are
# different msys2-runtime flavours — which has already destroyed a developer's
# live mingw64 netCDF once. On a throwaway CI prefix that is the intent; on a
# workstation it is data loss, so require the caller to mean it.
if [ "$INSTALL_UNIX" = "$MINGW_UNIX" ] \
   && [ -z "${CI:-}${GITHUB_ACTIONS:-}" ] \
   && [ "${NETCDF_INSTALL_OVERWRITE:-}" != "1" ]; then
    fail "refusing to overwrite the live toolchain prefix $INSTALL_PREFIX.
  This replaces netCDF for everything on this machine that links it.
  Install somewhere else:   NETCDF_INSTALL_PREFIX=/some/scratch $0
  ...or say you mean it:    NETCDF_INSTALL_OVERWRITE=1 $0
  If you set NETCDF_INSTALL_PREFIX and still see this, your shell dropped it
  across the msys2 boundary — export it instead of prefixing the command,
  and check the INSTALL INTO line above."
fi
# Announced loudly because this script OVERWRITES a working netCDF installation
# by design. Note that `VAR=x scripts/build_netcdf_no_s3_mingw.sh` does NOT
# reliably pass VAR when the calling shell and the script's interpreter come
# from different msys2 runtimes (Git Bash invoking C:\msys64\usr\bin\bash, say)
# — the assignment is silently dropped and the defaults above apply. `export`
# the variable, or run the script under the same shell, if the target matters.

# ---------------------------------------------------------------------------
# 2. Record the stock DLL's ABI surface (for the drop-in checks in step 5)
# ---------------------------------------------------------------------------
PREFIX_DLL="$MINGW_UNIX/bin/libnetcdf.dll"
NETCDFF_DLL="$MINGW_UNIX/bin/libnetcdff-7.dll"
mkdir -p "$WORK_DIR"
STOCK_EXPORTS="$WORK_DIR/exports-stock.txt"
NEW_EXPORTS="$WORK_DIR/exports-nos3.txt"

# Names exported by a PE DLL, one per line.
pe_exports() {
    objdump -p "$1" \
        | sed -n '/\[Ordinal\/Name Pointer\] Table/,$p' \
        | awk '$1 ~ /^\[/ {print $NF}' \
        | sort -u
}

# Symbols a PE image imports from a given DLL, one per line.
#
# `objdump -p` import entries look like:
#     DLL Name: libnetcdf.dll
#     vma:     Ordinal  Hint  Member-Name  Bound-To
#     000d0ac0  <none>  01a6  nc__create
# The table ends at a blank line, and LATER sections (notably the .pdata
# function table) also have a hex first field with four columns — so the DLL
# context must be cleared at the blank line, and the name column must actually
# look like a symbol, or those addresses get reported as missing imports.
pe_imports_from() {
    objdump -p "$1" | awk -v want="$2" '
        /DLL Name:/ { dll = tolower($3); next }
        NF == 0     { dll = ""; next }
        dll == tolower(want) && $1 ~ /^[0-9a-fA-F]+$/ && NF >= 4 &&
            $NF ~ /^[A-Za-z_][A-Za-z0-9_@.$]*$/ { print $NF }
    ' | sort -u
}

[ -f "$PREFIX_DLL" ] || fail "stock libnetcdf.dll not found at $PREFIX_DLL (install mingw-w64-x86_64-netcdf first)"
pe_exports "$PREFIX_DLL" > "$STOCK_EXPORTS"
echo "stock libnetcdf.dll exports: $(wc -l < "$STOCK_EXPORTS")"

# ---------------------------------------------------------------------------
# 3. Fetch source
# ---------------------------------------------------------------------------
mkdir -p "$SRC_DIR"
cd "$SRC_DIR"
TARBALL="netcdf-$VER.tar.gz"
if [ ! -d "netcdf-c-$VER" ]; then
    say "Downloading netcdf-c $VER"
    curl -fL --retry 5 --retry-delay 5 -o "$TARBALL" \
        "https://github.com/Unidata/netcdf-c/archive/v$VER/$TARBALL"

    # Verified checksums for versions we have actually shipped. An unknown
    # version is not fatal (MSYS2 bumps netCDF on its own schedule and the
    # release must keep working), but it is called out in the log.
    case "$VER" in
        4.9.3) EXPECT_SHA=990f46d49525d6ab5dc4249f8684c6deeaf54de6fec63a187e9fb382cc0ffdff ;;
        *)     EXPECT_SHA="" ;;
    esac
    if [ -n "$EXPECT_SHA" ]; then
        echo "$EXPECT_SHA  $TARBALL" | sha256sum -c -
    else
        echo "::warning::No pinned sha256 for netcdf-c $VER; add one to scripts/build_netcdf_no_s3_mingw.sh"
        sha256sum "$TARBALL"
    fi

    tar xzf "$TARBALL"

    # The MSYS2 PKGBUILD applies two patches that this build needs just as much.
    # Both go through a dry-run gate so the script keeps working once a netCDF
    # release absorbs them upstream; each logs which branch it took.
    #
    #  * no-debug-libraries: netCDF's FindZip/FindBlosc/FindZstd/FindBz2 emit
    #    "debug <lib> optimized <lib>" when the debug and release libraries are
    #    the same file — which is always true on mingw-w64. The Ninja generator
    #    then links neither, and the build dies at link time with hundreds of
    #    "undefined reference to __imp_zip_*". Vendored rather than fetched from
    #    MINGW-packages master so CI does not depend on a moving reference.
    #  * hdf5-2.x compat: netCDF 4.9.3 does not compile against HDF5 >= 2.0,
    #    which MSYS2 now ships. Immutable upstream commit URL.
    say "Applying mingw-w64 build patches"
    _apply_patch() {
        local name="$1" file="$2"
        if (cd "netcdf-c-$VER" && patch -p1 --dry-run -N -i "$file" >/dev/null 2>&1); then
            (cd "netcdf-c-$VER" && patch -p1 -N -i "$file")
            echo "applied $name"
        else
            echo "$name does not apply (already upstream or not needed) — skipped"
        fi
    }

    _apply_patch no-debug-libraries \
        "$SCRIPT_DIR/patches/netcdf-mingw-no-debug-libraries.patch"

    curl -fL --retry 5 -o hdf5-2.0-compat.patch \
        "https://github.com/Unidata/netcdf-c/commit/741c4b4a.patch"
    _apply_patch hdf5-2.0-compat "$SRC_DIR/hdf5-2.0-compat.patch"
fi

# ---------------------------------------------------------------------------
# 4. Configure
# ---------------------------------------------------------------------------
# Feature flags mirror MSYS2's shared build. The ONLY deliberate difference is
# the S3 pair. Optional filter/NCZarr backends (blosc, zstd, bz2, szip, libzip)
# are auto-detected from the packages already installed in the prefix, exactly
# as the PKGBUILD does.
say "Configuring"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
export CFLAGS="${CFLAGS:-} -Wno-sign-conversion -Wno-float-conversion -Wno-incompatible-pointer-types"

MSYS2_ARG_CONV_EXCL="-DCMAKE_INSTALL_PREFIX=;-DNETCDF_WITH_PLUGIN_DIR=" \
cmake -G Ninja \
    -DCMAKE_INSTALL_PREFIX="$MINGW_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DNETCDF_ENABLE_EXAMPLES=OFF \
    -DNETCDF_ENABLE_TESTS=OFF \
    -DNETCDF_ENABLE_DAP_REMOTE_TESTS=OFF \
    -DNETCDF_ENABLE_DAP=ON \
    -DNETCDF_ENABLE_HDF5=ON \
    -DNETCDF_ENABLE_BYTERANGE=ON \
    -DNETCDF_ENABLE_LOGGING=ON \
    -DNETCDF_ENABLE_S3=OFF \
    -DNETCDF_ENABLE_S3_AWS=OFF \
    -DNETCDF_WITH_PLUGIN_DIR="$MINGW_PREFIX/lib/netcdf" \
    -Wno-author \
    "$SRC_DIR/netcdf-c-$VER"

# ---------------------------------------------------------------------------
# 5. Build, then prove the artifact before it is allowed near the prefix
# ---------------------------------------------------------------------------
say "Building"
ninja -j "${NETCDF_JOBS:-$(nproc 2>/dev/null || echo 4)}"

BUILT_DLL="$BUILD_DIR/libnetcdf.dll"
[ -f "$BUILT_DLL" ] || fail "build produced no libnetcdf.dll"

say "Verifying the built netCDF"
grep -iE '^(S3 Support|S3 SDK|Byte-Range Support|DAP[24] Support|NCZarr Support|HDF5 Support):' \
    libnetcdf.settings || fail "libnetcdf.settings has no feature summary"

if ! grep -iE '^S3 Support:[[:space:]]*no' libnetcdf.settings >/dev/null; then
    fail "built netCDF still reports S3 support"
fi
AWS_IMPORTS="$(objdump -p "$BUILT_DLL" | grep -ci 'DLL Name: libaws' || true)"
[ "$AWS_IMPORTS" = "0" ] || fail "built libnetcdf.dll still imports $AWS_IMPORTS AWS DLL(s)"
echo "built libnetcdf.dll imports no AWS DLLs"

# 5a. Export-table diff: only the S3 SDK entry points may disappear.
pe_exports "$BUILT_DLL" > "$NEW_EXPORTS"
REMOVED="$(comm -23 "$STOCK_EXPORTS" "$NEW_EXPORTS" || true)"
if [ -n "$REMOVED" ]; then
    echo "exports dropped vs stock build:"
    printf '%s\n' "$REMOVED" | sed 's/^/  - /'
    UNEXPECTED="$(printf '%s\n' "$REMOVED" | grep -vE '^(NC_s3sdk|NCZ_s3finalize)' || true)"
    [ -z "$UNEXPECTED" ] || {
        printf '%s\n' "$UNEXPECTED" | sed 's/^/  !! /' >&2
        fail "netCDF lost non-S3 exports — not a drop-in replacement"
    }
else
    echo "no exports dropped vs stock build"
fi

# 5b. The decisive check: every symbol libnetcdff-7.dll imports from
#     libnetcdf.dll must still be exported, or every Fortran model breaks.
if [ -f "$NETCDFF_DLL" ]; then
    MISSING="$(comm -23 \
        <(pe_imports_from "$NETCDFF_DLL" libnetcdf.dll) \
        "$NEW_EXPORTS" || true)"
    if [ -n "$MISSING" ]; then
        printf '%s\n' "$MISSING" | sed 's/^/  !! /' >&2
        fail "libnetcdff-7.dll imports symbols the rebuilt libnetcdf.dll no longer exports"
    fi
    echo "libnetcdff-7.dll: all $(pe_imports_from "$NETCDFF_DLL" libnetcdf.dll | wc -l) imported symbols still exported (no relink needed)"
else
    echo "::warning::libnetcdff-7.dll not present — skipped the Fortran ABI check"
fi

# ---------------------------------------------------------------------------
# 6. Install
# ---------------------------------------------------------------------------
say "Installing to $INSTALL_PREFIX"
if [ "$INSTALL_PREFIX" != "$MINGW_PREFIX" ]; then
    cmake -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX" -P cmake_install.cmake
else
    ninja install
fi

INSTALLED_SETTINGS="$INSTALL_UNIX/lib/libnetcdf.settings"
[ -f "$INSTALLED_SETTINGS" ] || fail "libnetcdf.settings not installed to $INSTALLED_SETTINGS"
grep -iE '^S3 Support:[[:space:]]*no' "$INSTALLED_SETTINGS" >/dev/null \
    || fail "installed libnetcdf.settings does not report 'S3 Support: no'"
echo "installed libnetcdf.settings: $(grep -iE '^S3 Support:' "$INSTALLED_SETTINGS")"

INSTALLED_AWS="$(objdump -p "$INSTALL_UNIX/bin/libnetcdf.dll" | grep -ci 'DLL Name: libaws' || true)"
[ "$INSTALLED_AWS" = "0" ] || fail "installed libnetcdf.dll still imports AWS DLLs"

say "netCDF $VER rebuilt without S3 and installed to $INSTALL_PREFIX"
