=========================================
GR Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The GR (Génie Rural - Rural Engineering) model family consists of parsimonious conceptual rainfall-runoff models developed by INRAE (France). These models are renowned for their simplicity, computational efficiency, and robust performance across diverse hydroclimatic conditions.

**Key Capabilities:**

- Multiple model variants (GR4J, GR5J, GR6J) with 4-6 parameters
- CemaNeige snow module for snow-dominated basins
- Lumped and distributed spatial modes
- Fast calibration due to few parameters
- Optional routing integration (mizuRoute)
- R-based execution via rpy2 interface
- Excellent benchmark model for comparisons

**Typical Applications:**

- Rapid assessment and benchmarking
- Data-scarce regions (few parameters = less data needed)
- Operational flood forecasting
- Climate change impact studies
- Multi-site calibration
- Ensemble forecasting

**Spatial Scales:** Small catchments (10 km²) to large basins (100,000 km²)

**Temporal Resolution:** Daily (primary) or hourly

Model Variants
==============

GR4J (4 Parameters)
-------------------

**Description:** Daily lumped rainfall-runoff model with 4 parameters.

**Structure:**

- Production store (soil moisture accounting)
- Routing store (baseflow generation)
- Unit hydrographs for routing
- Groundwater exchange term

**Parameters:**

1. ``X1`` [mm]: Maximum capacity of production store (100-1500 mm)
2. ``X2`` [mm]: Groundwater exchange coefficient (-5 to +3 mm)
3. ``X3`` [mm]: One-day maximum capacity of routing store (20-500 mm)
4. ``X4`` [days]: Time base of unit hydrograph UH1 (0.5-4 days)

**Best for:** General-purpose applications, temperate climates

GR5J (5 Parameters)
-------------------

**Description:** GR4J plus inter-catchment groundwater exchange.

**Additional parameter:**

5. ``X5`` [mm]: Inter-catchment exchange threshold (0.01-3.0)

**Best for:** Karstic regions, basins with significant groundwater exchanges

GR6J (6 Parameters)
-------------------

**Description:** GR5J plus exponential store for low flows.

**Additional parameter:**

6. ``X6`` [mm]: Exponential store capacity for low flows (0.1-20)

**Best for:** Basins with complex low-flow dynamics, ephemeral streams

CemaNeige Snow Module
=====================

CemaNeige is a degree-day snow accounting module that can be coupled with any GR model.

**Structure:**

- Degree-day snowmelt equation
- Snow store for each elevation layer
- Thermal state tracking
- Separate parameters per elevation band

**Additional Parameters (per elevation band):**

1. ``CNX1``: Weighting coefficient for snowmelt (0.0-1.0)
2. ``CNX2``: Degree-day melt factor (0.0-10.0 mm/°C/day)

**Activation:**

Automatically enabled when using elevation discretization:

.. code-block:: yaml

   DISCRETIZATION:
     elevation_bands:
       n_bands: 5  # CemaNeige activated with 5 elevation zones

Forcing corrections and glacier melt (lumped, opt-in)
-----------------------------------------------------

Two optional calibration parameters correct the forcing before CemaNeige:
``PCORR`` multiplies total precipitation and ``TCORR`` adds a temperature
offset (K). Both are identity values (1 and 0) unless listed in
``GR_PARAMS_TO_CALIBRATE``. ``GR_SNOW_LAYERS`` sets the number of equal-area
CemaNeige elevation layers (airGR ``NLayers``, default 5).

airGR has no glacier process. With ``GR_GLACIER_ENABLED`` the lumped runner
calls CemaNeige and GR4J separately and adds degree-day ice melt between them:
each layer carries the glacierised fraction of its elevation band from
``GR_GLACIER_OUTLINE``, and ice melts at ``DDF * max(T_layer, 0)`` on days the
layer's snowpack is below ``GR_GLACIER_SNOWFREE_SWE`` (mm). The ice
degree-day factor is either calibrated (``ICE_DDF``) or, when
``GR_GLACIER_MB_TARGET`` (observed mean net balance, m w.e./yr over the
calibration water years) is set, solved in every run to reproduce that
balance and clipped to ``GR_GLACIER_ICE_DDF_BOUNDS``. ``GR_GLACIER_AREA_FILE``
(LamaH-Ice ``glacier_timeseries.csv``) scales the outline area to the observed
area of each water year. Ice storage is unlimited, so this is a forcing-side
workaround rather than glacier dynamics. Glacier mode needs external
parameters (the optimizer); it has no internal Michel calibration. Each run
writes ``GR_glacier_diagnostics.json`` (solved factor, simulated annual
balances) and an ``ice_melt_mm`` column in ``GR_results.csv``.

.. code-block:: yaml

   GR_PARAMS_TO_CALIBRATE: X1,X3,X4,CTG,Kf,Gratio,Albedo_diff,PCORR,TCORR
   GR_SNOW_LAYERS: 10
   GR_GLACIER_ENABLED: true
   GR_GLACIER_OUTLINE: /path/to/rgi_outline.gpkg
   GR_GLACIER_MB_TARGET: -1.907
   GR_GLACIER_AREA_FILE: /path/to/glacier_timeseries.csv

Spatial Modes
=============

The GR models in SYMFLUENCE support flexible spatial configuration:

Lumped Mode
-----------

Single HRU represents entire basin:

.. code-block:: yaml

   GR_SPATIAL_MODE: lumped

   # Basin-averaged forcing
   # 4-6 parameters total
   # Fastest execution (~seconds)

**Use case:** Quick assessments, simple basins, benchmarking

Distributed Mode
----------------

Multiple HRUs (typically elevation bands):

.. code-block:: yaml

   GR_SPATIAL_MODE: distributed

   DISCRETIZATION:
     elevation_bands:
       n_bands: 5

   # CemaNeige snow module activated
   # Each HRU has independent GR + snow parameters
   # Routing via mizuRoute

**Use case:** Snow-dominated regions, heterogeneous basins

Auto Mode
---------

Automatically infers spatial mode from domain definition:

.. code-block:: yaml

   GR_SPATIAL_MODE: auto  # Default

   # If DOMAIN_DEFINITION_METHOD = 'delineate' → distributed
   # Otherwise → lumped

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: GR

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
   * - GR_INSTALL_PATH
     - default
     - Path to GR model files (R scripts)
   * - GR_EXE
     - GR.r
     - Main R script for GR execution
   * - SETTINGS_GR_PATH
     - default
     - Working directory for GR

**Note:** GR models require R and the ``airGR`` package. SYMFLUENCE uses rpy2 for Python-R interface.

Spatial Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - GR_SPATIAL_MODE
     - auto
     - Spatial discretization (auto, lumped, distributed)
   * - GR_ROUTING_INTEGRATION
     - none
     - Enable routing (none, mizuRoute)

Model and Calibration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - GR_PARAMS_TO_CALIBRATE
     - null
     - Parameters to calibrate (auto-detected if null)
   * - SETTINGS_GR_CONTROL
     - default
     - GR control file path

Input File Specifications
=========================

GR models use CSV forcing data and R data structures.

Forcing Data (CSV)
------------------

**File (lumped):** ``<basin>_forcing_lumped.csv``

.. code-block:: text

   Date,Precip_mm,PET_mm,Temp_C
   2015-01-01,5.2,1.1,2.3
   2015-01-02,0.0,1.3,3.1
   2015-01-03,12.4,0.9,1.8
   ...

**Required columns:**

- ``Date``: YYYY-MM-DD format
- ``Precip_mm``: Precipitation [mm/day]
- ``PET_mm``: Potential evapotranspiration [mm/day]
- ``Temp_C``: Mean temperature [°C] (required for CemaNeige)

**File (distributed):** ``<basin>_forcing_hru_<HRU_ID>.csv``

Same format, one file per HRU.

Observations (CSV)
------------------

**File:** ``<basin>_streamflow.csv``

.. code-block:: text

   Date,Flow_m3s
   2015-01-01,45.3
   2015-01-02,42.1
   2015-01-03,55.8
   ...

Output File Specifications
==========================

GR outputs are CSV files with simulated fluxes and states.

Standard Output (CSV)
---------------------

**File:** ``<basin>_GR_output.csv``

.. code-block:: text

   Date,Qsim_mm,Prod,Rout,Exch,AE_mm,Precip_mm,PET_mm
   2015-01-01,1.23,234.5,45.2,-0.3,1.0,5.2,1.1
   2015-01-02,1.18,232.1,44.8,-0.2,1.2,0.0,1.3
   ...

**Output variables:**

- ``Qsim_mm``: Simulated runoff [mm/day]
- ``Prod``: Production store level [mm]
- ``Rout``: Routing store level [mm]
- ``Exch``: Groundwater exchange [mm]
- ``AE_mm``: Actual evapotranspiration [mm/day]
- ``Precip_mm``: Input precipitation [mm/day]
- ``PET_mm``: Input PET [mm/day]

CemaNeige Output (with snow module)
-----------------------------------

**Additional columns:**

.. code-block:: text

   ...,SnowPack_mm,Melt_mm,PliqAndMelt_mm
   ...,125.4,0.0,0.0
   ...,138.2,0.0,0.0
   ...,142.6,2.3,2.3

- ``SnowPack_mm``: Snow water equivalent [mm]
- ``Melt_mm``: Snowmelt [mm/day]
- ``PliqAndMelt_mm``: Liquid precipitation + melt [mm/day]

Routed Output (with mizuRoute)
------------------------------

**File:** ``<basin>_routed_streamflow.nc``

NetCDF with routed streamflow for distributed applications.

Model-Specific Workflows
========================

Basic GR4J Workflow (Lumped)
----------------------------

Simplest GR application:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: test_basin
   HYDROLOGICAL_MODEL: GR

   # Lumped mode (automatic for polygon domains)
   GR_SPATIAL_MODE: lumped
   DOMAIN_DEFINITION_METHOD: polygon
   CATCHMENT_SHP_PATH: ./basin.shp

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2015

   # Calibration
   OPTIMIZATION_ALGORITHM: DDS
   OPTIMIZATION_MAX_ITERATIONS: 1000

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

   # Output: Calibrated 4 parameters in ~5-10 minutes

Distributed GR with CemaNeige
-----------------------------

For snow-dominated mountainous basins:

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: mountain_basin
   HYDROLOGICAL_MODEL: GR

   # Distributed mode with elevation bands
   GR_SPATIAL_MODE: distributed
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-118.5, 49.2]

   # Create 5 elevation zones
   DISCRETIZATION:
     elevation_bands:
       n_bands: 5
       method: equal_area

   # CemaNeige automatically activated
   # Parameters: 4 (GR) + 2×5 (CemaNeige per band) = 14 total

   # Optional routing
   ROUTING_MODEL: mizuRoute
   GR_ROUTING_INTEGRATION: mizuRoute

   # Calibration
   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4,CNX1,CNX2"

Multi-Site GR Calibration
-------------------------

Calibrate GR on multiple basins simultaneously:

.. code-block:: yaml

   # config.yaml
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [10234, 10235, 10236]  # 3 basins

   HYDROLOGICAL_MODEL: GR
   GR_SPATIAL_MODE: distributed

   # Enable routing to aggregate sub-basins
   ROUTING_MODEL: mizuRoute

   # Calibrate using multi-site objective
   OPTIMIZATION_ALGORITHM: DDS
   OBJECTIVE_METRICS: [KGE, NSE, RMSE]

Regional GR Application
-----------------------

For large-sample studies (CAMELS-style):

.. code-block:: yaml

   # Process 100+ basins with GR4J
   # config_template.yaml
   HYDROLOGICAL_MODEL: GR
   GR_SPATIAL_MODE: lumped

   OPTIMIZATION_ALGORITHM: DDS
   OPTIMIZATION_MAX_ITERATIONS: 500

Batch process:

.. code-block:: python

   # Python script
   import pandas as pd
   from symfluence import SYMFLUENCE

   basin_list = pd.read_csv('basin_ids.csv')

   for basin_id in basin_list['id']:
       config = load_config('config_template.yaml')
       config['DOMAIN_NAME'] = f'basin_{basin_id}'
       config['CATCHMENT_SHP_PATH'] = f'./basins/{basin_id}.shp'

       workflow = SYMFLUENCE(config)
       workflow.run()

Calibration Strategies
======================

GR4J Parameters
---------------

.. code-block:: yaml

   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4"

**Recommended bounds:**

.. code-block:: python

   X1:  [100, 1500]    # Production store capacity [mm]
   X2:  [-5, 3]        # Groundwater exchange [mm]
   X3:  [20, 500]      # Routing store capacity [mm]
   X4:  [0.5, 4.0]     # Unit hydrograph time base [days]

**Physical interpretation:**

- ``X1``: Soil water storage capacity (larger = more buffering)
- ``X2``: Groundwater loss/gain (negative = loss, positive = gain)
- ``X3``: Baseflow storage (larger = slower recession)
- ``X4``: Response time (larger = slower peak)

GR4J + CemaNeige Parameters
---------------------------

.. code-block:: yaml

   # 4 GR + 2 snow parameters per elevation band
   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4,CNX1,CNX2"

**CemaNeige bounds:**

.. code-block:: python

   CNX1:  [0.0, 1.0]     # Snowmelt weighting coefficient [-]
   CNX2:  [0.0, 10.0]    # Degree-day factor [mm/°C/day]

**Typical CNX2 values:**

- Open areas: 3-6 mm/°C/day
- Forested areas: 2-4 mm/°C/day
- Glaciers: 6-10 mm/°C/day

GR5J and GR6J
-------------

Simply add extra parameters:

.. code-block:: yaml

   # GR5J (5 parameters)
   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4,X5"

.. code-block:: python

   X5:  [0.01, 3.0]      # Inter-catchment exchange threshold [mm]

.. code-block:: yaml

   # GR6J (6 parameters)
   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4,X5,X6"

.. code-block:: python

   X6:  [0.1, 20.0]      # Exponential store for low flows [mm]

Calibration Tips
----------------

1. **Start with GR4J:**

   - Simpler is often better
   - GR5J/GR6J rarely improve performance significantly
   - Extra parameters increase calibration time

2. **Use DDS algorithm:**

   - Efficient for 4-6 parameter problems
   - 500-1000 iterations usually sufficient
   - Much faster than DE or PSO

3. **Warm-up period:**

   - Use 1-2 year spinup
   - GR stores equilibrate quickly

4. **Objective function:**

   - KGE: Balanced overall performance
   - NSE: Peak flow performance
   - NSE_log: Low flow performance

5. **Multi-objective for snow:**

   .. code-block:: yaml

      OPTIMIZATION_ALGORITHM: NSGA2
      OBJECTIVE_METRICS:
        - KGE         # Streamflow fit
        - RMSE_snow   # Snow cover fit (if data available)

Known Limitations
=================

1. **Conceptual Model:**

   - No explicit physics (energy balance, etc.)
   - Parameters lack direct physical meaning
   - Limited process understanding

2. **Snow Module:**

   - Simple degree-day approach (CemaNeige)
   - No energy balance
   - Limited snow physics (no metamorphism, density evolution)
   - Rain-on-snow events may be poorly represented

3. **Daily Timestep:**

   - Primarily designed for daily data
   - Hourly variant exists but less tested
   - Sub-daily flash floods not well captured

4. **No Explicit Routing:**

   - Internal routing via unit hydrographs only
   - Distributed mode requires external routing (mizuRoute)
   - Channel processes not represented

5. **PET Dependency:**

   - Requires potential evapotranspiration as input
   - PET estimation method affects results
   - Model is sensitive to PET quality

6. **Fixed Structure:**

   - Unlike FUSE, no model structure options
   - One conceptual approach
   - Cannot test structural hypotheses

Troubleshooting
===============

Common Issues
-------------

**Error: "R/rpy2 not found"**

Solution: Install R and Python rpy2:

.. code-block:: bash

   # Install R
   # macOS:
   brew install r

   # Linux:
   sudo apt-get install r-base

   # Install rpy2
   pip install rpy2

   # Install airGR package in R
   R
   > install.packages("airGR")
   > quit()

**Error: "airGR package not found"**

.. code-block:: bash

   # Install airGR in R
   R -e "install.packages('airGR', repos='https://cran.r-project.org')"

**Error: "Invalid parameter bounds"**

Check that X2 can be negative:

.. code-block:: python

   # Correct bounds for X2
   X2: [-5, 3]  # Can be negative!

**Error: "PET missing in forcing data"**

GR requires PET calculation:

.. code-block:: yaml

   # Ensure PET is calculated from forcing data
   # SYMFLUENCE auto-calculates PET from radiation, temp, humidity

**Poor calibration performance**

1. **Check forcing data quality:**

   .. code-block:: bash

      # Verify no missing data
      python
      >>> import pandas as pd
      >>> df = pd.read_csv('basin_forcing_lumped.csv')
      >>> print(df.isnull().sum())

2. **Extend calibration period:**

   .. code-block:: yaml

      CALIBRATION_PERIOD: [2010, 2018]  # At least 5-8 years

3. **Check observation data:**

   - Ensure observations cover calibration period
   - Verify flow units (m³/s vs mm/day)
   - Check for data gaps

4. **Adjust parameter bounds:**

   .. code-block:: yaml

      # If arid basin, increase X1 upper bound
      # If flashy basin, decrease X4 upper bound

**CemaNeige not activated**

.. code-block:: yaml

   # Ensure elevation discretization is defined
   DISCRETIZATION:
     elevation_bands:
       n_bands: 5  # This activates CemaNeige

   # Verify snow parameters are included
   GR_PARAMS_TO_CALIBRATE: "X1,X2,X3,X4,CNX1,CNX2"

**Distributed mode routing issues**

.. code-block:: yaml

   # Enable routing explicitly
   ROUTING_MODEL: mizuRoute
   GR_ROUTING_INTEGRATION: mizuRoute

   # Verify network topology is created
   # Check: <project_dir>/domain/routing/network.nc

Performance Tips
----------------

**Speed up calibration:**

1. Use lumped mode (4 parameters vs 14+ for distributed)
2. Reduce calibration period (5 years minimum)
3. Use DDS algorithm (faster than DE/PSO for few parameters)
4. Reduce max iterations (500-750 often sufficient)

**Improve snow performance:**

1. Use at least 3 elevation bands (5 recommended)
2. Calibrate CNX2 per elevation band if data supports it
3. Include snow observations in calibration if available
4. Use equal-area elevation bands (better snow representation)

**Debug GR model:**

.. code-block:: bash

   # Run GR manually in R to see detailed output
   cd <project_dir>/settings/GR/
   Rscript GR.r

Additional Resources
====================

**GR Model Documentation:**

- airGR package: https://cran.r-project.org/web/packages/airGR/
- airGR vignettes: https://cran.r-project.org/web/packages/airGR/vignettes/
- INRAE Hydrology: https://webgr.inrae.fr/en/

**Publications:**

- Perrin et al. (2003): "Improvement of a parsimonious model for streamflow simulation" - GR4J
  https://doi.org/10.1016/S0022-1694(03)00225-7

- Valéry et al. (2014): "As simple as possible but not simpler: CemaNeige snow module"
  https://doi.org/10.1016/j.jhydrol.2014.04.059

- Le Moine (2008): Thesis on GR models (in French)

**SYMFLUENCE-specific:**

- :doc:`../configuration`: Full GR parameter reference
- :doc:`../calibration`: Calibration best practices
- :doc:`model_fuse`: Comparison with FUSE
- :doc:`model_summa`: Comparison with SUMMA
- :doc:`../troubleshooting`: General troubleshooting

**Example Configurations:**

.. code-block:: bash

   # List GR examples
   symfluence examples list | grep GR

   # Copy example
   symfluence examples copy bow_river_gr ./my_gr_project

**Benchmarking:**

GR4J is widely used as a benchmark model. Compare your model's performance against GR4J to assess added value of complexity.
