#!/bin/bash
#
# Verify SYMFLUENCE Binary Portability
#
# Checks binaries for:
# - Hard-coded RPATH to build directories
# - Absolute library paths
# - Missing library dependencies
# - Platform compatibility
#
# Usage:
#   ./scripts/verify_binary_portability.sh <binaries_dir>
#
# Example:
#   ./scripts/verify_binary_portability.sh ./release/symfluence-tools/bin

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

print_error()   { echo -e "${RED}✗${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info()    { echo -e "${BLUE}→${NC} $1"; }

# Portable realpath fallback
_realpath() {
    if command -v realpath >/dev/null 2>&1; then
        realpath "$1"
    else
        (cd "$(dirname "$1")" && echo "$(pwd)/$(basename "$1")")
    fi
}

# Parse arguments
BINARIES_DIR="$(_realpath "${1:-}")"

if [ -z "$BINARIES_DIR" ] || [ ! -d "$BINARIES_DIR" ]; then
    cat >&2 <<EOF
Usage: $0 <binaries_dir>
EOF
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SYMFLUENCE Binary Portability Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
print_info "Checking binaries in: $BINARIES_DIR"
echo ""

# Detect platform
OS_TYPE="$(uname -s)"
case "$OS_TYPE" in
    Linux)
        PLATFORM="Linux"
        ;;
    Darwin)
        PLATFORM="macOS"
        ;;
    MSYS*|MINGW*|CYGWIN*)
        PLATFORM="Windows"
        ;;
    *)
        print_error "Unsupported platform: $OS_TYPE"
        exit 1
        ;;
esac

print_info "Platform: $PLATFORM"
echo ""

# Find all binaries in bin/
BINARIES=()
for file in "$BINARIES_DIR"/*; do
    if [ -f "$file" ] && [ -x "$file" ]; then
        # Check if it's an ELF/Mach-O/PE binary (not a script)
        if file "$file" | grep -qE "(ELF|Mach-O|PE32|executable)"; then
            BINARIES+=("$file")
        fi
    fi
done

# Also check lib/ directory (sibling of bin/)
LIB_DIR="$(dirname "$BINARIES_DIR")/lib"
LIBRARIES=()
if [ -d "$LIB_DIR" ]; then
    for file in "$LIB_DIR"/*; do
        if [ -f "$file" ]; then
            if file "$file" | grep -qE "(ELF|Mach-O|shared object|dynamic library)"; then
                LIBRARIES+=("$file")
            fi
        fi
    done
fi

if [ ${#BINARIES[@]} -eq 0 ]; then
    print_error "No binaries found in $BINARIES_DIR"
    exit 1
fi

print_success "Found ${#BINARIES[@]} binaries to verify"
if [ ${#LIBRARIES[@]} -gt 0 ]; then
    print_success "Found ${#LIBRARIES[@]} shared libraries to verify"
fi
echo ""

# Verification results
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0
WARNING_CHECKS=0

# Track issues
RPATH_ISSUES=()
LIBRARY_ISSUES=()
PORTABILITY_ISSUES=()

# ============================================================================
# Linux Verification
# ============================================================================
if [ "$PLATFORM" = "Linux" ]; then
    print_info "Running Linux-specific checks..."
    echo ""

    for binary in "${BINARIES[@]}"; do
        binary_name="$(basename "$binary")"
        echo "━━━ Checking: $binary_name ━━━"

        # Check 1: RPATH inspection
        print_info "Checking RPATH..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v readelf >/dev/null 2>&1; then
            RPATH_OUTPUT="$(readelf -d "$binary" 2>/dev/null | grep -E "(RPATH|RUNPATH)" || true)"

            if [ -n "$RPATH_OUTPUT" ]; then
                # Check for suspicious paths
                if echo "$RPATH_OUTPUT" | grep -qE "(/home/|/tmp/|/build/|/runner/|/workspace/)"; then
                    print_error "RPATH contains build paths!"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    RPATH_ISSUES+=("$binary_name: Build path in RPATH")
                    FAILED_CHECKS=$((FAILED_CHECKS + 1))
                elif echo "$RPATH_OUTPUT" | grep -qE '\$ORIGIN'; then
                    print_success "RPATH uses \$ORIGIN (relocatable)"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                else
                    print_warning "RPATH present but may not be relocatable"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    RPATH_ISSUES+=("$binary_name: Non-relocatable RPATH")
                    WARNING_CHECKS=$((WARNING_CHECKS + 1))
                fi
            else
                print_success "No RPATH (uses system library paths)"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
            fi
        else
            print_warning "readelf not available, skipping RPATH check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        # Check 2: Library dependencies
        print_info "Checking library dependencies..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v ldd >/dev/null 2>&1; then
            LDD_OUTPUT="$(ldd "$binary" 2>/dev/null || true)"

            # Check for not found libraries
            if echo "$LDD_OUTPUT" | grep -q "not found"; then
                print_error "Missing library dependencies!"
                echo "$LDD_OUTPUT" | grep "not found" | sed 's/^/    /'
                LIBRARY_ISSUES+=("$binary_name: Missing libraries")
                FAILED_CHECKS=$((FAILED_CHECKS + 1))
            else
                # Check for absolute paths outside standard locations
                SUSPICIOUS_LIBS="$(echo "$LDD_OUTPUT" | grep -E "(/home/|/tmp/|/build/|/runner/|/workspace/)" || true)"
                if [ -n "$SUSPICIOUS_LIBS" ]; then
                    print_warning "Libraries from non-standard locations:"
                    echo "$SUSPICIOUS_LIBS" | sed 's/^/    /'
                    LIBRARY_ISSUES+=("$binary_name: Non-standard library paths")
                    WARNING_CHECKS=$((WARNING_CHECKS + 1))
                else
                    print_success "All libraries found in standard locations"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                fi
            fi
        else
            print_warning "ldd not available, skipping library check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        # Check 3: glibc version requirement
        print_info "Checking glibc version requirement..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v objdump >/dev/null 2>&1; then
            GLIBC_VERSIONS="$(objdump -T "$binary" 2>/dev/null | grep GLIBC_ | sed 's/.*GLIBC_/GLIBC_/' | sort -u || true)"

            if [ -n "$GLIBC_VERSIONS" ]; then
                MAX_GLIBC="$(echo "$GLIBC_VERSIONS" | sort -V | tail -1)"
                print_success "Max glibc: $MAX_GLIBC"

                # Check if it's too new
                if echo "$MAX_GLIBC" | grep -qE "GLIBC_2\.(3[5-9]|[4-9][0-9])"; then
                    print_success "glibc ≥ 2.35 (Ubuntu 24.04+)"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                elif echo "$MAX_GLIBC" | grep -qE "GLIBC_2\.[0-2][0-9]"; then
                    print_success "glibc compatible with older systems"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                else
                    print_warning "Unusual glibc version: $MAX_GLIBC"
                    WARNING_CHECKS=$((WARNING_CHECKS + 1))
                fi
            else
                print_warning "No glibc symbols found"
                WARNING_CHECKS=$((WARNING_CHECKS + 1))
            fi
        else
            print_warning "objdump not available, skipping glibc check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        echo ""
    done

    # Check shared libraries in lib/
    if [ ${#LIBRARIES[@]} -gt 0 ]; then
        print_info "Checking shared libraries in lib/..."
        echo ""

        for lib in "${LIBRARIES[@]}"; do
            lib_name="$(basename "$lib")"
            echo "━━━ Checking: $lib_name ━━━"

            # Check RPATH
            print_info "Checking RPATH..."
            TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

            if command -v readelf >/dev/null 2>&1; then
                RPATH_OUTPUT="$(readelf -d "$lib" 2>/dev/null | grep -E "(RPATH|RUNPATH)" || true)"

                if [ -n "$RPATH_OUTPUT" ]; then
                    if echo "$RPATH_OUTPUT" | grep -qE '\$ORIGIN'; then
                        print_success "RPATH uses \$ORIGIN (relocatable)"
                        PASSED_CHECKS=$((PASSED_CHECKS + 1))
                    elif echo "$RPATH_OUTPUT" | grep -qE "(/home/|/tmp/|/build/|/runner/|/workspace/)"; then
                        print_error "RPATH contains build paths!"
                        echo "$RPATH_OUTPUT" | sed 's/^/    /'
                        LIBRARY_ISSUES+=("$lib_name: Build path in RPATH")
                        FAILED_CHECKS=$((FAILED_CHECKS + 1))
                    else
                        print_warning "RPATH present but may not be relocatable"
                        WARNING_CHECKS=$((WARNING_CHECKS + 1))
                    fi
                else
                    print_success "No RPATH set"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                fi
            fi

            echo ""
        done
    fi

# ============================================================================
# macOS Verification
# ============================================================================
elif [ "$PLATFORM" = "Windows" ]; then
    print_info "Running Windows-specific checks (limited)..."
    echo ""

    for binary in "${BINARIES[@]}"; do
        binary_name="$(basename "$binary")"
        echo "--- Checking: $binary_name ---"

        # Check 1: File exists and is executable
        print_info "Checking file existence..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if [ -f "$binary" ]; then
            print_success "$binary_name exists"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            print_error "$binary_name not found"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        fi

        # Check 2: PE format check (if file command available)
        print_info "Checking binary format..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v file >/dev/null 2>&1; then
            FILE_TYPE="$(file "$binary")"
            if echo "$FILE_TYPE" | grep -qiE "(PE32|executable)"; then
                print_success "Valid Windows executable"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
            else
                print_warning "Unexpected file type: $FILE_TYPE"
                WARNING_CHECKS=$((WARNING_CHECKS + 1))
            fi
        else
            print_warning "file command not available, skipping format check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        echo ""
    done

elif [ "$PLATFORM" = "macOS" ]; then
    print_info "Running macOS-specific checks..."
    echo ""

    for binary in "${BINARIES[@]}"; do
        binary_name="$(basename "$binary")"
        echo "━━━ Checking: $binary_name ━━━"

        # Check 1: LC_RPATH inspection
        print_info "Checking LC_RPATH..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v otool >/dev/null 2>&1; then
            RPATH_OUTPUT="$(otool -l "$binary" 2>/dev/null | grep -A2 LC_RPATH || true)"

            if [ -n "$RPATH_OUTPUT" ]; then
                # Check for suspicious paths
                if echo "$RPATH_OUTPUT" | grep -qE "(/Users/|/tmp/|/build/|/runner/|/workspace/)"; then
                    print_error "LC_RPATH contains build paths!"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    RPATH_ISSUES+=("$binary_name: Build path in LC_RPATH")
                    FAILED_CHECKS=$((FAILED_CHECKS + 1))
                elif echo "$RPATH_OUTPUT" | grep -qE '@(executable_path|loader_path)'; then
                    print_success "LC_RPATH uses @executable_path (relocatable)"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                else
                    print_warning "LC_RPATH present but may not be relocatable"
                    echo "$RPATH_OUTPUT" | sed 's/^/    /'
                    WARNING_CHECKS=$((WARNING_CHECKS + 1))
                fi
            else
                print_success "No LC_RPATH (uses system library paths)"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
            fi
        else
            print_warning "otool not available, skipping RPATH check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        # Check 2: Library dependencies
        print_info "Checking library dependencies..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v otool >/dev/null 2>&1; then
            LIB_OUTPUT="$(otool -L "$binary" 2>/dev/null | tail -n +2 || true)"

            # Check for absolute paths outside /usr/lib and /System
            SUSPICIOUS_LIBS="$(echo "$LIB_OUTPUT" | grep -v -E "(^[[:space:]]+(/usr/lib/|/System/|@(executable_path|loader_path|rpath)))" || true)"

            if [ -n "$SUSPICIOUS_LIBS" ]; then
                # Check specifically for Homebrew or user paths
                if echo "$SUSPICIOUS_LIBS" | grep -qE "(/Users/|/opt/homebrew/|/usr/local/)"; then
                    print_error "Libraries reference non-relocatable paths (must be bundled in lib/):"
                    echo "$SUSPICIOUS_LIBS" | sed 's/^/    /'
                    LIBRARY_ISSUES+=("$binary_name: Non-relocatable library paths")
                    FAILED_CHECKS=$((FAILED_CHECKS + 1))
                else
                    print_success "All libraries use standard or relocatable paths"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                fi
            else
                print_success "All libraries use standard paths"
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
            fi
        else
            print_warning "otool not available, skipping library check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        # Check 3: macOS version requirement
        print_info "Checking macOS version requirement..."
        TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

        if command -v otool >/dev/null 2>&1; then
            VERSION_OUTPUT="$(otool -l "$binary" 2>/dev/null | grep -A3 LC_VERSION_MIN_MACOSX || otool -l "$binary" 2>/dev/null | grep -A3 LC_BUILD_VERSION || true)"

            if [ -n "$VERSION_OUTPUT" ]; then
                print_success "macOS version info found"
                echo "$VERSION_OUTPUT" | grep -E "(version|minos)" | sed 's/^/    /'
                PASSED_CHECKS=$((PASSED_CHECKS + 1))
            else
                print_warning "No macOS version requirement found"
                WARNING_CHECKS=$((WARNING_CHECKS + 1))
            fi
        else
            print_warning "otool not available, skipping version check"
            WARNING_CHECKS=$((WARNING_CHECKS + 1))
        fi

        echo ""
    done

    # Check shared libraries in lib/
    if [ ${#LIBRARIES[@]} -gt 0 ]; then
        print_info "Checking shared libraries in lib/..."
        echo ""

        for lib in "${LIBRARIES[@]}"; do
            lib_name="$(basename "$lib")"
            echo "━━━ Checking: $lib_name ━━━"

            # Check install name
            print_info "Checking install name..."
            TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

            if command -v otool >/dev/null 2>&1; then
                INSTALL_NAME="$(otool -D "$lib" 2>/dev/null | tail -1 || true)"

                if echo "$INSTALL_NAME" | grep -q "@rpath"; then
                    print_success "Install name uses @rpath (relocatable)"
                    echo "    $INSTALL_NAME"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                elif echo "$INSTALL_NAME" | grep -qE "(/Users/|/tmp/|/build/|/runner/)"; then
                    print_error "Install name contains build path!"
                    echo "    $INSTALL_NAME"
                    LIBRARY_ISSUES+=("$lib_name: Build path in install name")
                    FAILED_CHECKS=$((FAILED_CHECKS + 1))
                else
                    print_warning "Install name: $INSTALL_NAME"
                    WARNING_CHECKS=$((WARNING_CHECKS + 1))
                fi

                # Check library dependencies for non-relocatable paths
                print_info "Checking library dependencies..."
                TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

                LIB_DEPS="$(otool -L "$lib" 2>/dev/null | tail -n +2 | awk '{print $1}' || true)"
                SUSPICIOUS_LIB_DEPS="$(echo "$LIB_DEPS" | grep -vE '^(@rpath/|@executable_path/|@loader_path/|/usr/lib/|/System/)' || true)"

                if [ -n "$SUSPICIOUS_LIB_DEPS" ]; then
                    if echo "$SUSPICIOUS_LIB_DEPS" | grep -qE "(/Users/|/opt/homebrew/|/usr/local/)"; then
                        print_error "Library references non-relocatable paths (must be bundled):"
                        echo "$SUSPICIOUS_LIB_DEPS" | sed 's/^/    /'
                        LIBRARY_ISSUES+=("$lib_name: Non-relocatable dependency paths")
                        FAILED_CHECKS=$((FAILED_CHECKS + 1))
                    else
                        print_success "All dependencies use standard or relocatable paths"
                        PASSED_CHECKS=$((PASSED_CHECKS + 1))
                    fi
                else
                    print_success "All dependencies use relocatable paths"
                    PASSED_CHECKS=$((PASSED_CHECKS + 1))
                fi
            fi

            echo ""
        done
    fi
fi

# ============================================================================
# Summary
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Verification Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total checks:   $TOTAL_CHECKS"
print_success "Passed:         $PASSED_CHECKS"
print_warning "Warnings:       $WARNING_CHECKS"
print_error "Failed:         $FAILED_CHECKS"
echo ""

# Report issues
if [ ${#RPATH_ISSUES[@]} -gt 0 ]; then
    print_error "RPATH Issues:"
    for issue in "${RPATH_ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo ""
fi

if [ ${#LIBRARY_ISSUES[@]} -gt 0 ]; then
    print_warning "Library Issues:"
    for issue in "${LIBRARY_ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo ""
fi

if [ ${#PORTABILITY_ISSUES[@]} -gt 0 ]; then
    print_warning "Portability Issues:"
    for issue in "${PORTABILITY_ISSUES[@]}"; do
        echo "  - $issue"
    done
    echo ""
fi

# Exit code
if [ $FAILED_CHECKS -gt 0 ]; then
    print_error "Portability verification FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
elif [ $WARNING_CHECKS -gt 0 ]; then
    print_warning "Portability verification completed with WARNINGS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
else
    print_success "Portability verification PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
fi
