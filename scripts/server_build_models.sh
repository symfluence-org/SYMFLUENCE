#!/usr/bin/env bash
# Build the national SUMMA (glacier branch) and ParFlow-CLM executables on the landslíkan server (WSL Ubuntu 24.04).
# ADR 0015 / 0016. Run by the user (it installs software):
#
#   scp -r src/symfluence/models/summa/patches src/symfluence/models/clmparflow/patches scripts/server_build_models.sh \
#       darey@100.85.106.89:Downloads/landslikan_build/
#   ssh darey@100.85.106.89 "wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/darey/Downloads/landslikan_build/server_build_models.sh"
#
# Uses the system toolchain already present there (gcc/gfortran 13, cmake, OpenMPI 4.1, netCDF-Fortran, OpenBLAS);
# nothing is installed system-wide: everything goes to ~/landslikan_models. Same pinned sources and patches as
# `symfluence binary install summa-glacier --patched` and `... clmparflow --patched`, plus the restart-grid fix.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PREFIX="${PREFIX:-$HOME/landslikan_models}"
NCPU="${NCPU:-8}"
mkdir -p "$PREFIX"

# ---------------------------------------------------------------- SUMMA, glacier branch (native solver)
if [ ! -x "$PREFIX/summa-glacier/bin/summa.exe" ]; then
  git clone -q --branch glacier https://github.com/ashleymedin/summa.git "$PREFIX/summa-glacier"
  cd "$PREFIX/summa-glacier"
  git checkout -q b851c3df2bdc3fd91c8eeae856ce6098078a4055
  for p in glacier_diagnostics_routing.patch glacier_restart_grid.patch; do
    git apply --check "$HERE/patches/$p" && git apply "$HERE/patches/$p"
  done
  export LIBRARY_LINKS="-lopenblas"          # read by SUMMA's CMakeLists with SPECIFY_LAPACK_LINKS
  cmake -S build -B cmake_build -DUSE_SUNDIALS=OFF -DCMAKE_BUILD_TYPE=Release \
      -DSPECIFY_LAPACK_LINKS=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH="$(pwd)/cmake_build"
  cmake --build cmake_build --parallel "$NCPU"
  git rev-parse HEAD > bin/source_revision.txt
  cp "$HERE"/patches/glacier_*.patch bin/
fi
"$PREFIX/summa-glacier/bin/summa.exe" --version | head -3

# ---------------------------------------------------------------- ParFlow v3.15.0 + CLM + SYMFLUENCE patches
if [ ! -x "$PREFIX/parflow/bin/parflow" ]; then
  git clone -q --depth 1 --branch v3.15.0 https://github.com/parflow/parflow.git "$PREFIX/parflow-src"
  cd "$PREFIX/parflow-src"
  for p in parflow3150_clm_glacier_water.patch parflow3150_clm_restart_read.patch parflow3150_clm_ice_albedo.patch; do
    git apply --check "$HERE/patches/$p" && git apply "$HERE/patches/$p"
  done
  HYPRE_SRC=$(mktemp -d)
  git clone -q --depth 1 --branch v2.31.0 https://github.com/hypre-space/hypre.git "$HYPRE_SRC"
  (cd "$HYPRE_SRC/src" && CC=mpicc ./configure --prefix="$PREFIX/parflow/hypre" --disable-fortran -q \
      && make -j "$NCPU" >/dev/null && make install >/dev/null)
  rm -rf "$HYPRE_SRC"
  cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PREFIX/parflow" -DPARFLOW_AMPS_LAYER=mpi1 \
      -DPARFLOW_HAVE_CLM=ON -DHYPRE_ROOT="$PREFIX/parflow/hypre" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=mpicc -DCMAKE_Fortran_COMPILER=mpif90 \
      -DCMAKE_C_FLAGS="-Wno-int-conversion -Wno-implicit-function-declaration -Wno-incompatible-pointer-types" \
      -DCMAKE_Fortran_FLAGS="-w -ffree-line-length-none"
  cmake --build build -j "$NCPU"
  cmake --install build
  cp "$HERE"/patches/parflow3150_*.patch "$PREFIX/parflow/bin/"
fi
ls -la "$PREFIX/parflow/bin/parflow"
echo "built in $PREFIX (summa-glacier, parflow); basin.pfsol files are made on the Mac (no Tcl tools needed here)"
