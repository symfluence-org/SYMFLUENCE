Installation
============

Overview
--------
SYMFLUENCE installs via ``pip`` (the primary, recommended method); the external
model binaries are then provided by ``symfluence binary install``. Alternative
methods — npm (bundled pre-built binaries), Docker, and a source/bootstrap path
for development — are documented below and on the :doc:`docker` page. HPC
environments vary; use the module recipes on your cluster as needed.

Quick Install
-------------

**Option 1: pip (recommended)**

.. code-block:: bash

   # Install the Python framework
   pip install symfluence

   # Install the external model binaries (SUMMA, mizuRoute, FUSE, NGEN, TauDEM, ...)
   symfluence binary install

   # Verify the binaries
   symfluence binary doctor

.. note::

   **Python 3.11–3.13 only.** Key geospatial dependencies (Fiona, rasterio,
   GDAL) do not yet ship wheels for Python 3.14+. If you are on 3.14, create a
   3.11 or 3.12 environment instead (``python3.11 -m venv venv``).

   Ubuntu 26.04 ships only Python 3.14 and has no older ``python3.x`` package
   in apt, so the ``python3.11`` route does not work there. Either use Ubuntu
   24.04 (on Windows, ``wsl --install -d Ubuntu-24.04``; a plain
   ``wsl --install`` now gives 26.04), or install an interpreter with
   `uv <https://docs.astral.sh/uv/>`_ and point the bootstrap at it:

   .. code-block:: bash

      curl -LsSf https://astral.sh/uv/install.sh | sh
      uv python install 3.12
      SYMFLUENCE_PYTHON=$(uv python find 3.12) ./scripts/symfluence-bootstrap --install

**Optional extras**

The base install stays lean (~1 GB smaller than with the ML stack). Add
extras for optional feature sets:

.. code-block:: bash

   pip install "symfluence[ml]"    # PyTorch stack: LSTM and GNN models, dCoupler differentiable coupling
   pip install "symfluence[jax]"   # JAX conceptual models (Snow-17, SAC-SMA, HBV, ...) and autodiff calibration
   pip install "symfluence[sensitivity]" # Sobol and RBD-FAST sensitivity methods
   pip install "symfluence[conus404]" # HyTEST Intake catalog access for CONUS404
   pip install "symfluence[gdal]"  # GDAL Python bindings (a system GDAL install is auto-detected otherwise)

Without ``[ml]``, selecting ``HYDROLOGICAL_MODEL: LSTM`` or ``GNN`` fails with
an actionable install hint; everything else works.

.. warning::

   The optional ``[conus404]`` catalog stack currently includes ``intake``,
   which is affected by CVE-2026-33310 with no fixed upstream release. It is
   intentionally excluded from the default installation and runtime security
   audit until a fixed version is available.

**Option 2: npm (bundled pre-built binaries)**

.. code-block:: bash

   # Install globally (pre-built model binaries; see `npm/README.md` for the
   # full list — SUMMA, mizuRoute, FUSE, NGEN, TauDEM, HYPE, MESH, CRHM, mHM,
   # PRMS, SWAT, GSFLOW, RHESSys and more, subject to platform availability)
   npm install -g symfluence

   # Verify installation
   symfluence binary info

   # Check system compatibility
   symfluence binary doctor

Requirements: Linux (Ubuntu 22.04+, RHEL 9+, Debian 12+ x86_64) or macOS 12+
(Apple Silicon). System libraries NetCDF and HDF5 must be installed via your
package manager.

**Option 3: Docker**

Per-method Dockerfiles are provided for reproducible builds across all install
paths; see the :doc:`docker` page.

Development Installation
------------------------
For development or custom builds, clone the repository and use the built-in
installer:

.. code-block:: bash

   git clone https://github.com/symfluence-org/SYMFLUENCE.git
   cd SYMFLUENCE
   ./scripts/symfluence-bootstrap --install
   source venv/bin/activate

What this does:

- Sets up a Python environment — via `pixi <https://pixi.sh>`_ if it is
  installed, otherwise a ``venv/`` created with ``pip`` (Python 3.11–3.13)
- Installs Python dependencies
- Builds and validates the external model binaries (the first run can take
  a long time)
- Reuses the environment on subsequent runs

Manual Setup (Optional)
~~~~~~~~~~~~~~~~~~~~~~~
If you prefer to manage the environment yourself:

.. code-block:: bash

   uv sync --locked --extra dev-all
   uv run symfluence --help

System Prerequisites
--------------------
- Build toolchain: GCC/Clang (C/C++), gfortran; CMake >= 3.20; MPI (OpenMPI/MPICH)
- Core libs: GDAL, HDF5, NetCDF (C + Fortran), BLAS/LAPACK
- Optional tools: R, CDO (when applicable)

For the full list of system dependencies with platform-specific install commands,
see ``docs/SYSTEM_REQUIREMENTS.md``. You can verify your environment with:

.. code-block:: bash

   symfluence binary doctor

CDS API Credentials (CARRA/CERRA)
---------------------------------
If you plan to use CARRA or CERRA forcing datasets, configure CDS API credentials:

1. Register: https://cds.climate.copernicus.eu/
2. Get your Personal Access Token from your CDS user profile
3. Create ``~/.cdsapirc``:

.. code-block:: bash

   cat > ~/.cdsapirc << EOF
   url: https://cds.climate.copernicus.eu/api
   key: {PERSONAL_ACCESS_TOKEN}
   EOF

4. Restrict permissions:

.. code-block:: bash

   chmod 600 ~/.cdsapirc

Verification:

.. code-block:: python

   import cdsapi
   c = cdsapi.Client()
   print("CDS API configured successfully!")

HPC Module Recipes
------------------
Use your site's module system, then run the installer:

Anvil (Purdue RCAC):

.. code-block:: bash

   module load r/4.4.1
   module load gcc/14.2.0
   module load openmpi/4.1.6
   module load gdal/3.10.0
   module load conda/2024.09
   module load openblas/0.3.17
   module load netcdf-fortran/4.5.3
   module load udunits/2.2.28

ARC (University of Calgary):

.. code-block:: bash

   . /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
   module unuse $MODULEPATH
   module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

   module load gcc/14.2.0
   module load cmake
   module load netcdf-fortran/4.6.1
   module load netcdf-c/4.9.2
   module load openblas/0.3.27
   module load hdf5/1.14.3
   module load gdal/3.9.2
   module load netlib-lapack/3.11.0
   module load openmpi/4.1.6
   module load python/3.11.7
   module load r/4.4.1
   module load geos 

FIR (Compute Canada):

.. code-block:: bash

   module load StdEnv/2023
   module load gcc/12.3
   module load python/3.11.5
   module load gdal/3.9.1
   module load r/4.5.0
   module load cdo/2.2.2
   module load mpi4py/4.0.3
   module load netcdf-fortran/4.6.1
   module load openblas/0.3.24

Then run:

.. code-block:: bash

   ./scripts/symfluence-bootstrap --install

macOS (Intel or Apple Silicon)
------------------------------

Install Xcode tools and Homebrew:

.. code-block:: bash

   xcode-select --install
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Install core packages:

.. code-block:: bash

   brew update
   brew install cmake gcc open-mpi gdal hdf5 netcdf netcdf-fortran openblas lapack cdo r

Optional compiler pinning:

.. code-block:: bash

   export CC=$(brew --prefix)/bin/gcc-14
   export CXX=$(brew --prefix)/bin/g++-14
   export FC=$(brew --prefix)/bin/gfortran-14

Then run:

.. code-block:: bash

   ./scripts/symfluence-bootstrap --install

Verification
------------
.. code-block:: bash

   symfluence --version
   symfluence binary doctor

Troubleshooting
---------------

GDAL Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~

GDAL is the most common source of installation problems. Here are solutions for
common scenarios:

**macOS Apple Silicon (M1/M2/M3)**

The ARM architecture requires special handling:

.. code-block:: bash

   # Option 1: Homebrew (recommended)
   brew install gdal
   pip install gdal==$(gdal-config --version)

   # Option 2: Conda (if Homebrew fails)
   conda create -n symfluence python=3.11
   conda activate symfluence
   conda install -c conda-forge gdal

   # Verify architecture matches
   file $(which gdalinfo)  # Should show "arm64"

**Version Mismatch Errors**

If you see ``gdal_config_error`` or version conflicts:

.. code-block:: bash

   # Check system GDAL version
   gdal-config --version

   # Install matching Python bindings
   pip install gdal==$(gdal-config --version)

   # If that fails, try conda-forge
   conda install -c conda-forge gdal=$(gdal-config --version)

**Windows**

Native Windows installation is complex. Use conda-forge:

.. code-block:: bash

   # Create environment with all geospatial deps
   conda create -n symfluence python=3.11
   conda activate symfluence
   conda install -c conda-forge gdal geopandas rasterio netcdf4 hdf5

   # Then install symfluence
   pip install symfluence

**Linux Build Failures**

If GDAL Python bindings fail to compile:

.. code-block:: bash

   # Ensure development headers are installed
   sudo apt-get install -y libgdal-dev

   # Set include paths
   export CPLUS_INCLUDE_PATH=/usr/include/gdal
   export C_INCLUDE_PATH=/usr/include/gdal

   # Install with numpy pre-installed
   pip install numpy
   pip install gdal==$(gdal-config --version)

NetCDF/HDF5 Issues
~~~~~~~~~~~~~~~~~~

**Missing libnetcdf or libhdf5**

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt-get install -y libnetcdf-dev libhdf5-dev

   # macOS
   brew install netcdf hdf5

   # Verify
   nc-config --version
   h5cc -showconfig | head -5

**HDF5 Version Conflicts**

If you see HDF5 header/library version mismatches:

.. code-block:: bash

   # Force rebuild with system libraries
   pip uninstall h5py
   HDF5_DIR=/usr/local pip install --no-binary=h5py h5py

rpy2/R Integration Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**R Not Found**

.. code-block:: bash

   # Ensure R is in PATH
   which R
   R --version

   # On macOS, you may need to link
   export R_HOME=$(R RHOME)

**rpy2 Compilation Errors**

.. code-block:: bash

   # Install R development headers
   # Ubuntu
   sudo apt-get install r-base-dev

   # Then reinstall rpy2
   pip install --force-reinstall rpy2

Diagnostic Commands
~~~~~~~~~~~~~~~~~~~

Run these to diagnose issues:

.. code-block:: bash

   # Check all dependencies
   symfluence binary doctor

   # Verify Python environment
   python -c "from osgeo import gdal; print(gdal.__version__)"
   python -c "import netCDF4; print(netCDF4.__version__)"
   python -c "import rpy2; print(rpy2.__version__)"

   # Check system libraries
   ldconfig -p | grep -E "(gdal|netcdf|hdf5)"  # Linux
   otool -L $(python -c "from osgeo import gdal; print(gdal.__file__)")  # macOS

Next Steps
----------
- :doc:`getting_started` — your first run
- :doc:`configuration` — YAML structure and options
- :doc:`examples` — progressive tutorials
