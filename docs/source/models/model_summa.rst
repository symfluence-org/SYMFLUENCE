.. _models_summa:

=========================================
SUMMA Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

SUMMA (Structure for Unifying Multiple Modeling Alternatives) is a physically-based hydrological model designed to systematically evaluate modeling decisions. SUMMA provides a common framework to test multiple conceptual approaches for representing dominant hydrological processes.

**Key Capabilities:**

- Distributed, physically-based snow and hydrological modeling
- Multiple options for each physical process (200+ decision combinations)
- Detailed snow physics with multiple layering schemes
- Sophisticated energy balance calculations
- Flexible spatial discretization (HRUs, GRUs)
- Glacier modeling support (with glacier mode)
- Native parallel execution support (MPI/OpenMP)
- NetCDF-based input/output

**Typical Applications:**

- Snow hydrology and snowmelt processes
- Mountain/alpine hydrology
- Energy balance studies
- Process uncertainty quantification
- Hydrological model structural uncertainty analysis
- Glacier mass balance and hydrology

**Spatial Scales:** Point to regional (tested up to continental scale)

**Temporal Resolution:** Sub-hourly to daily (typically hourly)

Model Physics and Structure
===========================

Mathematical Foundation
-----------------------

SUMMA solves the mass and energy conservation equations for:

1. **Energy Balance:**

   - Surface energy balance (shortwave/longwave radiation, sensible/latent heat)
   - Snowpack energy evolution
   - Soil heat transfer

2. **Mass Balance:**

   - Snow accumulation and melt
   - Canopy interception and throughfall
   - Infiltration and saturation excess runoff
   - Unsaturated zone drainage
   - Saturated zone baseflow
   - Evapotranspiration

3. **State Variables:**

   - Snow water equivalent and density (per layer)
   - Soil moisture (per layer)
   - Soil temperature (per layer)
   - Canopy water storage
   - Aquifer storage

Spatial Discretization
----------------------

SUMMA uses a two-level hierarchy:

**Grouped Response Units (GRUs):**

- Collections of HRUs with similar characteristics
- Used for parallel execution (can run GRUs independently)
- Typically represent sub-basins or computational domains

**Hydrologic Response Units (HRUs):**

- Finest spatial element
- Homogeneous land surface characteristics
- Can be defined by elevation bands, land use, soil type, aspect, etc.
- Downslope connectivity between HRUs (if SETTINGS_SUMMA_CONNECT_HRUS = true)

Decision Options
----------------

SUMMA's unique feature is the ability to specify different model physics through decision options:

.. code-block:: yaml

   SUMMA_DECISION_OPTIONS:
     snowLayers: CLM_2010              # Multi-layer snow model
     vegeParTran: CLM_NOAHMP           # Vegetation-canopy parameterization
     stomResist: BallBerry             # Stomatal resistance
     soilStress: NoahType              # Soil moisture stress function
     spatial_gw: localColumn           # Groundwater parameterization

Over 30 decision categories with 2-5 options each allows for flexible model configuration.

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

To use SUMMA in your configuration:

.. code-block:: yaml

   HYDROLOGICAL_MODEL: SUMMA

Key Configuration Parameters
----------------------------

Installation and Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - SUMMA_INSTALL_PATH
     - default
     - Path to SUMMA installation directory
   * - SUMMA_EXE
     - summa_sundials.exe
     - SUMMA executable name
   * - SUMMA_TIMEOUT
     - 7200
     - Model execution timeout (seconds)
   * - SETTINGS_SUMMA_USE_PARALLEL_SUMMA
     - false
     - Enable parallel SUMMA execution (using specified backend)
   * - SETTINGS_SUMMA_PARALLEL_BACKEND
     - slurm
     - Parallel backend (``slurm`` or ``local``)
   * - SETTINGS_SUMMA_PARALLEL_EXE
     - summa_actors.exe
     - Parallel SUMMA executable name
   * - SETTINGS_SUMMA_PARALLEL_PATH
     - default
     - Path to parallel SUMMA executable

File Management
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - SETTINGS_SUMMA_FILEMANAGER
     - fileManager.txt
     - Master file manager
   * - SETTINGS_SUMMA_ATTRIBUTES
     - attributes.nc
     - HRU attribute file
   * - SETTINGS_SUMMA_FORCING_LIST
     - forcingFileList.txt
     - List of forcing files
   * - SETTINGS_SUMMA_COLDSTATE
     - coldState.nc
     - Initial conditions file
   * - SETTINGS_SUMMA_TRIALPARAMS
     - trialParams.nc
     - Parameter file for calibration
   * - SETTINGS_SUMMA_OUTPUT
     - outputControl.txt
     - Output variable selection

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - PARAMS_TO_CALIBRATE
     - null
     - Comma-separated list of parameters to calibrate
   * - BASIN_PARAMS_TO_CALIBRATE
     - null
     - Basin-scale parameters to calibrate
   * - CALIBRATE_DEPTH
     - false
     - Enable soil depth calibration
   * - DEPTH_TOTAL_MULT_BOUNDS
     - null
     - Bounds for total depth multiplier [min, max]
   * - DEPTH_SHAPE_FACTOR_BOUNDS
     - null
     - Bounds for depth shape factor [min, max]

Common calibration parameters:

- ``routingGammaShape``: Gamma distribution shape for routing (1.0 - 5.0)
- ``routingGammaScale``: Gamma distribution scale for routing (1.0 - 500.0)
- ``summerLAI``: Summer leaf area index (0.5 - 10.0)
- ``winterLAI``: Winter leaf area index (0.01 - 5.0)
- ``aquiferBaseflowExp``: Baseflow exponent (1.0 - 10.0)
- ``aquiferBaseflowRate``: Baseflow rate (0.001 - 1.0 mm/day)

Glacier Mode
^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - SETTINGS_SUMMA_GLACIER_MODE
     - false
     - Enable glacier preprocessing
   * - SETTINGS_SUMMA_GLACIER_ATTRIBUTES
     - attributes_glac.nc
     - Glacier-specific attributes file
   * - SETTINGS_SUMMA_GLACIER_COLDSTATE
     - coldState_glac.nc
     - Glacier initial conditions file

Parallel Execution
^^^^^^^^^^^^^^^^^^

For large-scale applications:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - SETTINGS_SUMMA_GRU_COUNT
     - 85
     - Total number of GRUs
   * - SETTINGS_SUMMA_GRU_PER_JOB
     - 5
     - GRUs processed per parallel job
   * - SETTINGS_SUMMA_CPUS_PER_TASK
     - 32
     - CPUs allocated per task (SLURM)
   * - SETTINGS_SUMMA_MEM
     - 5
     - Memory per task (GB)
   * - SETTINGS_SUMMA_TIME_LIMIT
     - 01:00:00
     - Wall time limit (HH:MM:SS)

Decision Options
^^^^^^^^^^^^^^^^

Customize model physics:

.. code-block:: yaml

   SUMMA_DECISION_OPTIONS:
     # Snow model structure
     snowLayers: CLM_2010           # Options: jrdn1991, CLM_2010

     # Vegetation and canopy
     vegeParTran: CLM_NOAHMP        # Options: Noah_BATS, CLM_NOAHMP, SSiB
     stomResist: BallBerry          # Options: BallBerry, Jarvis, simpleResistance

     # Soil moisture and stress
     num_method: itertive           # Options: iterative, non_iterative
     fDerivMeth: analytic           # Options: analytic, numerical

     # Runoff generation
     spatial_gw: localColumn        # Options: localColumn, singleBasin
     subRouting: timeDlay           # Options: timeDlay, qInstant

See SUMMA documentation for complete list of 30+ decision options.

Input File Specifications
=========================

SUMMA requires several input files, automatically generated by SYMFLUENCE:

Forcing Data (NetCDF)
---------------------

**File pattern:** ``<domain>_<year><month>.nc``

**Required variables:**

.. code-block:: text

   time        : Time coordinate (CF-compliant)
   hru         : HRU dimension

   airtemp     : Air temperature [K]
   pptrate     : Precipitation rate [kg m-2 s-1]
   windspd     : Wind speed [m s-1]
   SWRadAtm    : Downward shortwave radiation [W m-2]
   LWRadAtm    : Downward longwave radiation [W m-2]
   spechum     : Specific humidity [kg kg-1]
   airpres     : Air pressure [Pa]

Attributes File (attributes.nc)
-------------------------------

Contains HRU-specific spatial attributes:

.. code-block:: text

   hruId         : HRU identifier
   gruId         : GRU identifier
   hru2gruId     : HRU to GRU mapping
   downHRUindex  : Downslope HRU connectivity
   elevation     : Mean elevation [m]
   latitude      : Latitude [degrees]
   longitude     : Longitude [degrees]
   HRUarea       : HRU area [m²]
   tan_slope     : Tangent of topographic slope [-]
   contourLength : Contour length [m]
   slopeTypeIndex: Slope type (1=flat, 2=sloped)
   soilTypeIndex : Soil class index
   vegTypeIndex  : Vegetation class index
   mHeight       : Measurement height [m]

Trial Parameters (trialParams.nc)
---------------------------------

Calibration parameters with dimensions (hru, variables):

- Default parameter values
- Updated during calibration
- Can be HRU-specific or uniform

Output File Specifications
==========================

SUMMA outputs NetCDF files with user-defined variables.

Standard Output Files
---------------------

**File pattern:** ``<domain>_<GRU>_<run_suffix>_timestep.nc``

**Common output variables:**

.. code-block:: text

   # Snow variables
   scalarSWE              : Snow water equivalent [kg m-2]
   scalarSnowDepth        : Snow depth [m]
   scalarSnowAlbedo       : Snow albedo [-]
   mLayerTemp             : Snow layer temperature [K]
   mLayerDepth            : Snow layer depth [m]

   # Runoff variables
   scalarSurfaceRunoff    : Surface runoff [m s-1]
   scalarRainPlusMelt     : Rain plus melt reaching ground [kg m-2 s-1]
   averageInstantRunoff   : Instantaneous runoff [m s-1]
   averageRoutedRunoff    : Routed runoff [m s-1]

   # Soil variables
   mLayerVolFracWat       : Volumetric soil moisture [-]
   mLayerMatricHead       : Matric head [m]
   scalarAquiferStorage   : Aquifer storage [m]

   # Energy balance
   scalarLatHeatTotal     : Total latent heat flux [W m-2]
   scalarSenHeatTotal     : Total sensible heat flux [W m-2]
   scalarNetRadiation     : Net radiation [W m-2]

Control Output Variables
------------------------

Specify in ``outputControl.txt`` or set all outputs:

.. code-block:: bash

   # Output all variables at every timestep
   ! varName       | outFreq | instant | sum | avg | var
   *               | 1       | 1       | 0   | 0   | 0

Model-Specific Workflows
========================

Basic SUMMA Workflow
--------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   HYDROLOGICAL_MODEL: SUMMA

   # Spatial discretization
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-115.0, 51.0]
   DISCRETIZATION:
     elevation_bands:
       n_bands: 5

   # Forcing data
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2015
   FORCING_END_YEAR: 2020

   # SUMMA configuration
   SUMMA_INSTALL_PATH: /path/to/summa
   SUMMA_EXE: summa_sundials.exe

   SUMMA_DECISION_OPTIONS:
     snowLayers: CLM_2010
     vegeParTran: CLM_NOAHMP
     spatial_gw: localColumn

Run the workflow:

.. code-block:: bash

   # Full workflow
   symfluence workflow run --config config.yaml

   # Or step-by-step
   symfluence workflow steps setup_project calibrate_model --config config.yaml

Snow-Focused Application
------------------------

For detailed snow modeling:

.. code-block:: yaml

   SUMMA_DECISION_OPTIONS:
     snowLayers: CLM_2010          # Multi-layer snow (up to 5 layers)
     snowDenNew: hedAndPom         # Fresh snow density parameterization
     snowRMelt: variableLiquidHold # Variable liquid water holding
     snowCompactMethod: anderson76 # Snow compaction

   # Calibrate snow parameters
   PARAMS_TO_CALIBRATE: "Fcapil,upperBoundHead,aquiferBaseflowRate"

   # Output detailed snow variables
   # (Configure in outputControl.txt or use all variables)

Glacier Mode Workflow
---------------------

Install the experimental glacier engine explicitly (the ordinary ``summa``
target does not include glacier physics):

.. code-block:: bash

   symfluence binary install summa-glacier

This installs Ashley Medin's glacier fork at revision
``b851c3df2bdc3fd91c8eeae856ce6098078a4055`` under
``$SYMFLUENCE_DATA_DIR/installs/summa-glacier/bin/summa.exe``, separately from
ordinary SUMMA. It uses the native solver, requires CMake, NetCDF-Fortran and
OpenBLAS/LAPACK, and is excluded from default installations. Builds default to
two jobs (override with ``CMAKE_BUILD_PARALLEL_LEVEL``). ``--branch`` explicitly
overrides the pinned revision; ``--git-hash`` selects an explicit commit.

For glacier experiments, select the executable and both input files explicitly:

.. code-block:: yaml

   # Enable glacier preprocessing
   SETTINGS_SUMMA_GLACIER_MODE: true

   # Replace /path/to/SYMFLUENCE_data with your data directory
   SUMMA_INSTALL_PATH: /path/to/SYMFLUENCE_data/installs/summa-glacier/bin
   SUMMA_EXE: summa.exe

   # Glacier-specific attributes will be generated
   SETTINGS_SUMMA_GLACIER_ATTRIBUTES: attributes_glac.nc
   SETTINGS_SUMMA_GLACIER_COLDSTATE: coldState_glac.nc

   # File-manager inputs must reference the glacier files too
   SETTINGS_SUMMA_ATTRIBUTES: attributes_glac.nc
   SETTINGS_SUMMA_COLDSTATE: coldState_glac.nc

   # Use glacier-suitable decision options
   SUMMA_DECISION_OPTIONS:
     snowLayers: CLM_2010
     alb_method: varDecay      # Variable decay for glacier ice albedo

The glacier flag alone does not establish valid glacier physics. Supply and
validate glacier fractions, ice thickness/bed geometry, layer counts and phase
fractions. The pinned upstream engine has a known inactive-layer initialization
issue in its glacier mass-residual diagnostic; this target installs unmodified
upstream source. Successful execution is not evidence of water-balance closure.
Use isolated experiments and independently check precipitation, runoff,
evapotranspiration and storage before calibration.

``symfluence binary install summa-glacier --patched`` applies the inactive-layer
diagnostic initialization fix and passes the snow reservoir's own time constant
to glacier routing. It also conserves liquid mass and available pore volume
during glacier-layer resizing, sums expelled water consistently across layers,
and accounts for its outer-step duration when averaging substep fluxes.
These corrections require watershed-scale validation before calibration.
The patched wet-bulb calculation brackets the original psychrometric equation
instead of using an unconstrained Newton iteration. Invalid inputs terminate
with a nonzero exit code. SUMMA execution and calibration also validate output
timestamps against the file manager, rejecting truncated or gapped output even
when the native executable returns zero.
For glacier-only geometry updates, upland elevation is recomputed from exposed
bed-grid cells. A sub-grid land fraction with no exposed cell uses the original
HRU elevation. HRU slope and aspect remain static terrain attributes. This avoids
artificially changing land elevation to compensate for glacier thinning while
holding the original HRU mean elevation fixed. Rasterized domain areas are
normalized when their sum exceeds the polygon HRU area.
Use ``--force`` to rebuild an existing installation. The
applied patch is retained in ``bin/applied_glacier.patch`` for provenance.

For explicit semidistributed geometry, supply projected or georeferenced source
data using:

.. code-block:: yaml

   SUMMA_GLACIER_OUTLINE_PATH: /path/to/dated_outlines.gpkg
   SUMMA_GLACIER_THICKNESS_DIR: /path/to/thickness_tiles
   SUMMA_GLACIER_GRID_RESOLUTION: 100

Thickness files must match ``*thickness.tif`` and contain thickness in metres.
The preprocessor projects HRUs to a local metric grid, intersects ice outlines,
reprojects the DEM and thickness, and creates an independent glacier complex per
GRU. It preserves HRU/GRU IDs, land area and upland soil-layer counts, and writes
an input report to ``settings/SUMMA/glacier_geometry.json``. Missing thickness
or DEM coverage raises an error. This path models clean ice with fractional
land/ice domains; it does not infer debris cover or firn zones. Inventory dates,
DEM epoch and thickness uncertainty must be documented for each experiment.

Parallel SUMMA for Large Domains
--------------------------------

For continental-scale or high-resolution applications, SUMMA can run GRU subsets
in parallel through a specified backend. The ``slurm`` backend submits job arrays, while
the ``local`` backend launches multiple SUMMA subprocesses on the current machine.

SLURM backend:

.. code-block:: yaml

   # Enable parallel SUMMA
   SETTINGS_SUMMA_USE_PARALLEL_SUMMA: true
   SETTINGS_SUMMA_PARALLEL_BACKEND: slurm
   SETTINGS_SUMMA_PARALLEL_EXE: summa_actors.exe
   SETTINGS_SUMMA_PARALLEL_PATH: /path/to/parallel/summa

   # Configure parallel execution
   SETTINGS_SUMMA_GRU_COUNT: 1000      # Total GRUs
   SETTINGS_SUMMA_GRU_PER_JOB: 10      # GRUs per job
   SETTINGS_SUMMA_CPUS_PER_TASK: 32    # CPUs per task
   SETTINGS_SUMMA_MEM: 16              # GB memory per task
   SETTINGS_SUMMA_TIME_LIMIT: "04:00:00"  # 4 hours

Local backend:

.. code-block:: yaml

   # Enable parallel SUMMA with local backend
   SETTINGS_SUMMA_USE_PARALLEL_SUMMA: true
   SETTINGS_SUMMA_PARALLEL_BACKEND: local

   # Configure parallel execution
   SETTINGS_SUMMA_CPUS_PER_TASK: 8

Calibration Strategies
======================

Recommended Parameters
----------------------

**Universal parameters (all domains):**

.. code-block:: yaml

   PARAMS_TO_CALIBRATE: "routingGammaShape,routingGammaScale,aquiferBaseflowRate"

**Snow-dominated basins:**

.. code-block:: yaml

   PARAMS_TO_CALIBRATE: "routingGammaShape,routingGammaScale,aquiferBaseflowRate,Fcapil,upperBoundHead"

**Vegetation-dominated basins:**

.. code-block:: yaml

   PARAMS_TO_CALIBRATE: "routingGammaShape,routingGammaScale,summerLAI,winterSAI,heightCanopyTop"

**Arid/semi-arid basins:**

.. code-block:: yaml

   PARAMS_TO_CALIBRATE: "routingGammaShape,routingGammaScale,aquiferBaseflowExp,fieldCapacity"

Parameter Bounds
----------------

Typical ranges for common parameters:

.. code-block:: python

   # Routing parameters
   routingGammaShape:  [1.0, 5.0]
   routingGammaScale:  [1.0, 500.0]

   # Vegetation parameters
   summerLAI:          [0.5, 10.0]
   winterLAI:          [0.01, 5.0]
   heightCanopyTop:    [0.5, 30.0]  # meters

   # Soil and groundwater
   aquiferBaseflowRate: [0.001, 1.0]  # mm/day
   aquiferBaseflowExp:  [1.0, 10.0]
   fieldCapacity:       [0.05, 0.5]   # fraction
   Fcapil:              [0.01, 1.0]

Multi-Objective Calibration
---------------------------

For comprehensive performance:

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA2  # Multi-objective algorithm

   OBJECTIVE_METRICS:
     - KGE
     - NSE_log
     - RMSE

   OPTIMIZATION_MAX_ITERATIONS: 500
   OPTIMIZATION_POPULATION: 100

Known Limitations
=================

1. **Computational Cost:**

   - Detailed physics = slower execution
   - Multi-layer snow increases compute time 2-3x
   - Large domains (>1000 HRUs) benefit from parallel SUMMA

2. **Initial Conditions:**

   - Cold start requires spinup (typically 1-2 years)
   - Soil moisture initialization critical for arid regions
   - Snow states need careful initialization in seasonal snow zones

3. **Data Requirements:**

   - Requires 7 meteorological variables
   - Hourly forcing recommended for snow applications
   - Missing radiation data degrades energy balance accuracy

4. **Model Stability:**

   - Very small HRUs (<0.01 km²) can cause numerical issues
   - Extreme parameter values may cause non-convergence
   - Some decision option combinations untested

5. **Output Size:**

   - Full output can be large (GB per year for 100 HRUs)
   - Use selective output variables for storage efficiency

Troubleshooting
===============

Common Issues
-------------

**Error: "SUMMA executable not found"**

.. code-block:: yaml

   # Solution: Verify installation path and executable name
   SUMMA_INSTALL_PATH: /absolute/path/to/summa/bin
   SUMMA_EXE: summa_sundials.exe

**Error: "Simulation failed - NaN values"**

Causes and solutions:

1. Extreme parameter values → Check calibration bounds
2. Insufficient spinup → Increase SIMULATION_START_YEAR
3. Bad forcing data → Validate forcing with ``ncview`` or ``ncdump``
4. HRU connectivity issues → Set ``SETTINGS_SUMMA_CONNECT_HRUS: false``

**Error: "NetCDF dimension mismatch"**

.. code-block:: bash

   # Check forcing file structure
   ncdump -h forcing_file.nc

   # Verify HRU dimension matches attributes file
   ncdump -h attributes.nc | grep "hru ="

**Slow execution**

Solutions:

1. Enable parallel SUMMA for >100 HRUs
2. Reduce output frequency
3. Use fewer snow layers (``snowLayers: jrdn1991`` instead of ``CLM_2010``)
4. Disable HRU connectivity if not needed

**Missing output variables**

.. code-block:: bash

   # Check outputControl.txt
   cat <project_dir>/settings/SUMMA/outputControl.txt

   # Use wildcard for all variables
   echo "* | 1 | 1 | 0 | 0 | 0" > outputControl.txt

Performance Optimization
------------------------

**For faster calibration:**

1. Use simple decision options during calibration:

   .. code-block:: yaml

      SUMMA_DECISION_OPTIONS:
        snowLayers: jrdn1991      # Single-layer snow (faster)
        num_method: non_iterative # Explicit scheme

2. Reduce output variables during calibration
3. Use coarser temporal resolution for large-scale studies

**For accurate physics:**

1. Use detailed options:

   .. code-block:: yaml

      SUMMA_DECISION_OPTIONS:
        snowLayers: CLM_2010      # Multi-layer snow
        num_method: iterative     # Implicit scheme

2. Hourly forcing data
3. Full energy balance variables in output

Additional Resources
====================

**SUMMA Documentation:**

- Official docs: https://summa.readthedocs.io
- GitHub: https://github.com/NCAR/summa
- User forum: https://github.com/NCAR/summa/discussions

**SYMFLUENCE-specific:**

- :doc:`../configuration`: Full parameter reference
- :doc:`../calibration`: Calibration workflows
- :doc:`../troubleshooting`: General troubleshooting
- :doc:`../getting_started`: Quickstart tutorial

**Example Configurations:**

.. code-block:: bash

   # View example SUMMA configuration
   cat $SYMFLUENCE_ROOT/resources/config_templates/examples/config_basin_notebook.yaml

**Publications:**

- Clark et al. (2015): "A unified approach for process-based hydrologic modeling:
  1. Modeling concept" - https://doi.org/10.1002/2015WR017198
- Clark et al. (2015): "A unified approach for process-based hydrologic modeling:
  2. Model implementation and case studies" - https://doi.org/10.1002/2015WR017200
