=========================================
MESH Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

MESH (Modélisation Environmentale Communautaire - Surface and Hydrology / Community Environmental Modeling System for Surface Hydrology) is an integrated land surface-hydrological modeling system developed by Environment and Climate Change Canada (ECCC). MESH couples detailed land surface schemes with hydrological routing to provide comprehensive water and energy budget simulations.

**Key Capabilities:**

- Coupled land surface-hydrology modeling
- Multiple land surface schemes (CLASS, SVS)
- Tile-based land heterogeneity representation
- Detailed cold region processes (permafrost, organic soils)
- Lakes and wetlands
- Routing with WATROUTE or WATFLOOD
- Energy and water budgets
- Applicable from headwaters to large river basins

**Typical Applications:**

- Cold region hydrology (snow, permafrost, frozen soil)
- Wetland-dominated landscapes
- Large basin water budgets
- Climate change impact studies
- Operational flood forecasting (Canada)
- Land-atmosphere coupling studies
- Agricultural water management

**Spatial Scales:** Subcatchment (10 km²) to large basins (100,000+ km²)

**Temporal Resolution:** Sub-hourly to daily

Model Structure
===============

Land Surface Schemes
--------------------

MESH supports multiple physics options:

**CLASS (Canadian Land Surface Scheme):**

- Standard MESH land surface module
- 3-layer soil model
- Detailed snow physics
- Canopy processes
- Permafrost capable

**SVS (Soil, Vegetation, Snow):**

- Alternative to CLASS
- Multi-layer soil and snow
- Urban physics
- Glacier representation (opt-in native ice surfaces; see :doc:`model_mesh_glacier`)

Spatial Discretization
----------------------

**Grouped Response Units (GRUs):**

Similar to SUMMA, MESH uses GRUs representing subcatchments.

**Tiles within GRUs:**

Each GRU contains tiles representing land cover types:

.. code-block:: text

   GRU 1 (Subcatchment):
   ├─ Tile 1: Needleleaf Forest (40%)
   ├─ Tile 2: Cropland (35%)
   ├─ Tile 3: Wetland (15%)
   └─ Tile 4: Urban (10%)

   GRU 2:
   ├─ Tile 1: Grassland (60%)
   └─ Tile 2: Cropland (40%)

Process Representation
----------------------

**Snow:**

- Multi-layer snowpack
- Snow densification and albedo evolution
- Blowing snow sublimation
- Snow interception by canopy

**Soil:**

- 3+ layer moisture and temperature
- Frozen soil physics
- Infiltration and percolation
- Subsurface lateral flow

**Evapotranspiration:**

- Energy balance approach
- Stomatal conductance
- Soil moisture stress

**Runoff Generation:**

- Infiltration excess
- Saturation excess
- Subsurface flow
- Wetland storage and release

**Routing:**

- WATROUTE or WATFLOOD
- Channel routing
- Lake routing
- Reservoirs (optional)

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: MESH

Key Configuration Parameters
----------------------------

Installation and Execution
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - MESH_INSTALL_PATH
     - default
     - Path to MESH executable
   * - MESH_EXE
     - mesh.exe
     - MESH executable name
   * - EXPERIMENT_OUTPUT_MESH
     - default
     - Output directory

Spatial Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - MESH_SPATIAL_MODE
     - distributed
     - Spatial mode (lumped, distributed)
   * - MESH_GRU_DIM
     - default
     - GRU dimension name
   * - MESH_HRU_DIM
     - default
     - HRU (tile) dimension name

Forcing Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - MESH_FORCING_PATH
     - default
     - Path to forcing files
   * - MESH_FORCING_VARS
     - default
     - Forcing variable names
   * - MESH_FORCING_UNITS
     - default
     - Forcing units
   * - MESH_FORCING_TO_UNITS
     - default
     - Target units for conversion

Data Configuration
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - MESH_DDB_VARS
     - default
     - Drainage database variables
   * - MESH_DDB_UNITS
     - default
     - Database units
   * - MESH_DDB_MIN_VALUES
     - default
     - Minimum acceptable values
   * - MESH_DS_MAIN_ID
     - default
     - Main drainage segment ID

Input File Specifications
=========================

MESH uses NetCDF for most inputs and outputs.

Drainage Database (DDB)
-----------------------

**File:** ``MESH_drainage_database.nc``

Contains GRU and tile attributes:

.. code-block:: text

   dimensions:
     gru = 50      ! Number of subcatchments/GRUs
     tile = 5      ! Max tiles per GRU

   variables:
     int gru_id(gru)
     float gru_area(gru)          ! km²
     float gru_elevation(gru)     ! m
     float gru_slope(gru)         ! degrees
     float gru_latitude(gru)
     float gru_longitude(gru)

     int tile_type(gru, tile)     ! Land cover type
     float tile_frac(gru, tile)   ! Fraction of GRU
     float tile_slope(gru, tile)
     float tile_aspect(gru, tile)

Forcing Data (NetCDF)
---------------------

**File:** ``MESH_forcing.nc``

.. code-block:: text

   dimensions:
     time = UNLIMITED
     gru = 50

   variables:
     double time(time)
     float FSIN(time, gru)        ! Shortwave radiation [W/m²]
     float FLIN(time, gru)        ! Longwave radiation [W/m²]
     float PRE(time, gru)         ! Precipitation [kg/m²/s]
     float TA(time, gru)          ! Air temperature [K]
     float QA(time, gru)          ! Specific humidity [kg/kg]
     float UV(time, gru)          ! Wind speed [m/s]
     float PRES(time, gru)        ! Pressure [Pa]

Initial Conditions
------------------

**File:** ``MESH_initial_conditions.ini``

Text file with initial states:

.. code-block:: text

   ! Snow states
   SNO  50.0    ! Snow water equivalent [mm]
   TSNO -2.0    ! Snow temperature [°C]

   ! Soil states
   TBAR  3.5    ! Soil temperature [°C]
   THLQ  0.25   ! Liquid soil moisture [-]
   THIC  0.05   ! Frozen soil moisture [-]

Output File Specifications
==========================

Standard Output (NetCDF)
------------------------

**File:** ``MESH_output_Basin_average.nc``

Basin-averaged fluxes:

.. code-block:: text

   dimensions:
     time = UNLIMITED

   variables:
     double time(time)
     float RUNOFF(time)           ! Total runoff [mm]
     float RECHARGE(time)         ! Groundwater recharge [mm]
     float EVAP(time)             ! Evapotranspiration [mm]
     float SNO(time)              ! Snow water equivalent [mm]
     float TBAR(time)             ! Mean soil temperature [K]
     float THLQ(time)             ! Liquid soil moisture [-]

GRU-level Output (NetCDF)
-------------------------

**File:** ``MESH_output_Grid.nc``

Spatially distributed output:

.. code-block:: text

   dimensions:
     time = UNLIMITED
     gru = 50

   variables:
     float RUNOFF(time, gru)
     float SNO(time, gru)
     float EVAP(time, gru)
     float TBAR(time, gru)
     float THLQ(time, gru)

Streamflow Output (Text)
------------------------

**File:** ``MESH_streamflow_Gauge_###.txt``

.. code-block:: text

   YEAR  MONTH  DAY  HOUR  STREAMFLOW_m3s
   2015  1      1    0     45.32
   2015  1      1    1     45.18
   2015  1      1    2     45.05
   ...

Model-Specific Workflows
========================

Basic MESH Workflow
-------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: canadian_basin
   HYDROLOGICAL_MODEL: MESH

   # Domain
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-95.0, 54.0]  # Manitoba

   # Forcing
   FORCING_DATASET: RDRS  # Canadian reanalysis (recommended for Canada)
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # MESH configuration
   MESH_INSTALL_PATH: /path/to/mesh
   MESH_SPATIAL_MODE: distributed

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

Cold Region MESH Application
----------------------------

For permafrost and seasonal frost:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: MESH

   # Cold region domain (Arctic/Subarctic)
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-115.0, 67.0]  # Northwest Territories

   # Discretization with permafrost classes
   DISCRETIZATION:
     soilclass:
       include_permafrost: true
       sources: ['SoilGrids', 'PermafrostMap']
     landclass:
       sources: ['MODIS']

   # CLASS land surface scheme (better for cold regions)
   MESH_LAND_SURFACE_SCHEME: CLASS

Wetland-Dominated Basin
-----------------------

For Prairie Pothole or boreal wetlands:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: MESH

   # Wetland representation
   DISCRETIZATION:
     landclass:
       sources: ['MODIS', 'WetlandMap']
       include_wetland_class: true

   # WATFLOOD routing (handles wetlands well)
   MESH_ROUTING_SCHEME: WATFLOOD

Large Basin Application
-----------------------

For major river basins (e.g., Mackenzie, Nelson):

.. code-block:: yaml

   # config.yaml
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [...]  # 100+ subcatchments

   HYDROLOGICAL_MODEL: MESH
   MESH_SPATIAL_MODE: distributed

   # Parallel execution
   NUM_PROCESSES: 32

   # Routing essential for large domain
   MESH_ROUTING_SCHEME: WATROUTE

Calibration Strategies
======================

MESH has many parameters, but key ones for calibration:

Recommended Parameters
----------------------

**Runoff generation:**

.. code-block:: yaml

   # WATFLOOD parameters
   R2 N:  Manning's n for overland flow (0.01-0.3)
   RILL:  Rill storage (0.0-10.0 mm)

   # CLASS parameters
   ZSNL:  Limiting snow depth for albedo (0.01-0.1 m)
   ZPLG:  Maximum ponding depth (0.001-0.01 m)

**Soil parameters (per soil class):**

- Hydraulic conductivity
- Field capacity
- Wilting point
- Porosity

**Routing parameters:**

- Channel Manning's n (0.01-0.1)
- Channel geometry (if not from DEM)

**Snow parameters:**

- Snow albedo decay rate
- Fresh snow albedo

Parameter Bounds
----------------

Typical ranges for key parameters:

.. code-block:: python

   # Overland flow
   R2N:      [0.01, 0.3]       # Manning's n overland
   RILL:     [0.0, 10.0]       # Rill storage [mm]

   # Soil hydraulic
   SAND_Ksat: [1e-6, 1e-4]     # Sandy soil Ksat [m/s]
   CLAY_Ksat: [1e-7, 1e-5]     # Clay soil Ksat [m/s]

   # Channel routing
   R2CH:     [0.01, 0.1]       # Manning's n channel

   # Snow
   ZSNL:     [0.01, 0.1]       # Snow depth for albedo [m]
   ZPLS:     [0.1, 10.0]       # Max snow interception [mm]

Known Limitations
=================

1. **Complexity:**

   - Steep learning curve
   - Many input files and parameters
   - Detailed land surface data required

2. **Computational Cost:**

   - Energy balance calculations expensive
   - Tile-based approach multiplies computations
   - Large domains require significant resources

3. **Data Requirements:**

   - Detailed land cover and soil maps
   - All forcing variables needed (8 variables)
   - Lakes and wetland databases

4. **Limited Adoption Outside Canada:**

   - Primarily used in Canada
   - Documentation focused on Canadian applications
   - Less community support compared to international models

5. **Calibration Challenges:**

   - High-dimensional parameter space
   - Parameters interact complexly
   - Requires expert knowledge

Troubleshooting
===============

Common Issues
-------------

**Error: "MESH executable not found"**

.. code-block:: yaml

   # Verify installation
   MESH_INSTALL_PATH: /absolute/path/to/mesh/bin
   MESH_EXE: mesh.exe

**Error: "Drainage database format error"**

Check NetCDF structure:

.. code-block:: bash

   ncdump -h MESH_drainage_database.nc

   # Verify dimensions: gru, tile
   # Verify required variables

**Error: "Forcing variable not found"**

.. code-block:: bash

   # Check forcing NetCDF
   ncdump -h MESH_forcing.nc

   # Ensure all 7 variables: FSIN, FLIN, PRE, TA, QA, UV, PRES

**Simulation crashes**

1. **Check initial conditions** (reasonable ranges)
2. **Verify tile fractions sum to 1.0** per GRU
3. **Check forcing for missing data or extremes**
4. **Reduce timestep** if unstable

**Poor performance in cold regions**

.. code-block:: yaml

   # Use CLASS scheme
   MESH_LAND_SURFACE_SCHEME: CLASS

   # Include permafrost classes
   DISCRETIZATION:
     soilclass:
       include_permafrost: true

**Wetlands not working**

.. code-block:: yaml

   # Ensure wetland tile type defined
   DISCRETIZATION:
     landclass:
       include_wetland_class: true

   # Use WATFLOOD routing
   MESH_ROUTING_SCHEME: WATFLOOD

Performance Optimization
------------------------

**Speed up execution:**

1. Reduce number of tiles (aggregate similar classes)
2. Coarser timestep (hourly vs 30-min)
3. Simplify routing (use WATROUTE instead of WATFLOOD)
4. Parallel execution (MPI-enabled MESH)

**Improve accuracy:**

1. Detailed tile representation (more classes)
2. Finer timestep (30-min or 1-hr)
3. Include lakes explicitly
4. Use high-resolution DEM for slopes

Additional Resources
====================

**MESH Documentation:**

- Official site: https://wiki.usask.ca/display/MESH
- User Manual: https://wiki.usask.ca/display/MESH/MESH+Documentation
- GitHub: https://github.com/MESH-Model/MESH-Dev

**CLASS Documentation:**

- CLASS manual: https://cccma.gitlab.io/classic/

**Publications:**

- Pietroniro et al. (2007): "Development of the MESH modelling system for hydrological ensemble forecasting"
  https://doi.org/10.1016/j.jhydrol.2006.07.015

- Haghnegahdar et al. (2017): "Multicriteria sensitivity analysis as a diagnostic tool for understanding MESH model behavior"
  https://doi.org/10.1002/hyp.11358

**SYMFLUENCE-specific:**

- :doc:`../configuration`: MESH parameter reference
- :doc:`../calibration`: Calibration best practices
- :doc:`model_summa`: Comparison with SUMMA
- :doc:`model_mesh_glacier`: Staging native MESH glacier surfaces
- :doc:`../troubleshooting`: General troubleshooting

**Canadian Datasets:**

- RDRS (Regional Deterministic Reanalysis System): Forcing data for Canada
  https://collaboration.cmc.ec.gc.ca/cmc/cmoi/product_guide/

- CanVec: Canadian vector geospatial data
- Canadian Digital Elevation Model (CDEM)

**Training:**

- University of Saskatchewan MESH training
- ECCC workshops (for Canadian users)

**Applications:**

- Used in Canada's operational flood forecasting
- Provincial water resource management (Saskatchewan, Alberta)
- Climate change impact studies across Canada

.. toctree::
   :hidden:

   model_mesh_glacier
