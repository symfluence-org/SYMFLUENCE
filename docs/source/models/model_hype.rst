=========================================
HYPE Model Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

HYPE (HYdrological Predictions for the Environment) is a semi-distributed, process-based hydrological model developed by SMHI (Swedish Meteorological and Hydrological Institute). Originally designed for integrated water quality and quantity modeling, HYPE combines robust hydrological process representation with nutrient and contaminant transport capabilities.

**Key Capabilities:**

- Semi-distributed structure (subcatchments + soil/land classes)
- Coupled water and water quality modeling
- Snow accumulation and melt
- Soil moisture dynamics
- Groundwater processes
- Lake and river routing
- Nutrient cycling (N, P, C)
- Optional sediment and contaminant transport
- Applicable from local to continental scales

**Typical Applications:**

- Watershed hydrology
- Water quality modeling (nutrients, sediments)
- Climate change impact assessment
- Land-use change scenarios
- Flood forecasting
- Drought analysis
- Regional to continental-scale assessments

**Spatial Scales:** Sub-basin (10 km²) to continental (millions of km²)

**Temporal Resolution:** Daily (standard) or sub-daily

Model Structure
===============

Spatial Discretization
----------------------

HYPE uses a unique two-tier spatial structure:

**1. Subcatchments (Sub-basins):**

- Topographically delineated areas
- Connected through river network
- Each has elevation, slope, and geographic attributes

**2. Soil-Land Classes (SLCs):**

- Within each subcatchment
- Defined by combinations of:
  - Soil type (e.g., clay, sand, loam)
  - Land use (e.g., forest, agriculture, urban)
  - Elevation zone (optional for snow modeling)

**Example Spatial Structure:**

.. code-block:: text

   Subcatchment 1 (500 km²)
   ├─ Forest-Sandy_Soil (40%)
   ├─ Agriculture-Clay_Soil (35%)
   ├─ Urban-Impervious (15%)
   └─ Wetland-Organic_Soil (10%)

   Subcatchment 2 (300 km²)
   ├─ Forest-Loam_Soil (60%)
   └─ Agriculture-Clay_Soil (40%)

Process Representation
----------------------

**Snow and Ice:**

- Degree-day snowmelt
- Snow distribution by elevation
- Refreezing and cold content
- Glacier mass balance (optional)

**Soil Water:**

- Up to 3 soil layers
- Infiltration and percolation
- Macropore flow
- Tile drainage
- Saturation excess and infiltration excess runoff

**Evapotranspiration:**

- Temperature-based PET (Priestley-Taylor or Thornthwaite)
- Actual ET from soil layers
- Interception evaporation
- Lake evaporation

**Groundwater:**

- Two aquifer levels (shallow, deep)
- Baseflow generation
- Aquifer recharge from soil layers

**Routing:**

- River routing (kinematic wave or rating curve)
- Lake retention and routing
- Delays in subcatchments
- Flood plains and wetlands

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: HYPE

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
   * - HYPE_INSTALL_PATH
     - default
     - Path to HYPE executable
   * - HYPE_EXE
     - hype
     - HYPE executable name
   * - SETTINGS_HYPE_PATH
     - default
     - HYPE working directory
   * - SETTINGS_HYPE_INFO
     - info.txt
     - Main HYPE configuration file

Calibration Parameters
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HYPE_PARAMS_TO_CALIBRATE
     - null
     - Comma-separated list of parameters
   * - HYPE_SPINUP_DAYS
     - 365
     - Spinup period (days)

Input File Specifications
=========================

HYPE uses text-based input files. SYMFLUENCE automatically generates these from your configuration.

Core Configuration Files
------------------------

**1. info.txt** (Main Configuration)

Specifies simulation settings, output variables, and model options:

.. code-block:: text

   !!----------------------
   !! Simulation settings
   !!----------------------
   bdate         2010-01-01
   cdate         2010-01-01
   edate         2020-12-31
   tstep         1     ! Daily timestep

   !!----------------------
   !! Model structure
   !!----------------------
   substance     0     ! 0 = hydrology only, 1 = include water quality
   simN          0     ! Nitrogen simulation (0/1)
   simP          0     ! Phosphorus simulation (0/1)
   simC          0     ! Carbon simulation (0/1)

   !!----------------------
   !! Output settings
   !!----------------------
   outregion     1
   outbasin      1
   basinoutput   4   ! Daily basin output

   !!----------------------
   !! Performance criteria
   !!----------------------
   criterion     1   ! NSE
   criterion     2   ! KGE

**2. par.txt** (Parameters)

Contains model parameters (general + class-specific):

.. code-block:: text

   !!----------------------
   !! General parameters
   !!----------------------
   !! parameter     value
   gratk            0.100    ! Recession coefficient for runoff (1/day)
   rrcs1            0.200    ! Runoff coefficient for soil layer 1
   rrcs2            0.100    ! Runoff coefficient for soil layer 2
   rrcs3            0.050    ! Runoff coefficient for soil layer 3
   ttmp             0.000    ! Threshold temperature for snowmelt (°C)
   cmlt             3.500    ! Degree-day factor for snowmelt (mm/°C/day)

   !!----------------------
   !! Soil class parameters
   !!----------------------
   !! parameter   soilclass1   soilclass2   soilclass3
   wcfc           0.300        0.250        0.200    ! Field capacity
   wcwp           0.150        0.120        0.100    ! Wilting point
   wcep           0.400        0.350        0.300    ! Effective porosity

**3. GeoData.txt** (Subcatchment Information)

Attributes for each subcatchment:

.. code-block:: text

   SUBID    AREA_km2    ELEV_mean    SLOPE    LAKE_frac    SLC1    SLC1_frac    SLC2    SLC2_frac
   1        250.5       450          0.05     0.02         101     0.60         102     0.40
   2        180.3       520          0.08     0.00         101     0.45         103     0.55
   3        320.1       380          0.03     0.10         102     0.70         104     0.30

**Columns:**

- ``SUBID``: Subcatchment ID
- ``AREA_km2``: Subcatchment area [km²]
- ``ELEV_mean``: Mean elevation [m]
- ``SLOPE``: Mean slope [-]
- ``LAKE_frac``: Fraction covered by lakes
- ``SLC#``: Soil-land class IDs
- ``SLC#_frac``: Fraction of subcatchment in each class

**4. GeoClass.txt** (Soil-Land Class Definitions)

Defines what each SLC represents:

.. code-block:: text

   SLCID    SOILTYPE    LANDUSE    CROP    VEGETATION
   101      1           1          0       1          ! Sandy soil + Forest
   102      2           2          1       2          ! Clay soil + Agriculture + Crop type 1
   103      1           3          0       3          ! Sandy soil + Urban
   104      3           4          0       4          ! Organic soil + Wetland

**5. ForcKey.txt** (Forcing Data Mapping)

Maps subcatchments to forcing files:

.. code-block:: text

   SUBID    PSFILE    TSFILE    TMAXFILE    TMINFILE
   1        P1.txt    T1.txt    TMAX1.txt   TMIN1.txt
   2        P2.txt    T2.txt    TMAX2.txt   TMIN2.txt
   3        P3.txt    T3.txt    TMAX3.txt   TMIN3.txt

Forcing Data Files
------------------

**Precipitation (Pnnn.txt):**

.. code-block:: text

   DATE            VALUE
   2010-01-01      5.2
   2010-01-02      0.0
   2010-01-03      12.4
   ...

**Temperature (Tnnn.txt):**

.. code-block:: text

   DATE            VALUE
   2010-01-01      2.3
   2010-01-02      3.1
   2010-01-03      1.8
   ...

Output File Specifications
==========================

HYPE produces multiple text output files.

Basin Output (basinoutput.txt)
------------------------------

Time series for each subcatchment:

.. code-block:: text

   DATE        SUBID    COUT    ROUT    SNOW    SOIL    EVAP
   2010-01-01  1        45.3    42.1    150.2   234.5   0.8
   2010-01-01  2        32.1    29.8    180.4   198.2   0.6
   2010-01-02  1        42.8    40.2    145.8   232.1   0.9
   ...

**Variables:**

- ``COUT``: Computed outflow [m³/s]
- ``ROUT``: Routed outflow [m³/s]
- ``SNOW``: Snow water equivalent [mm]
- ``SOIL``: Total soil moisture [mm]
- ``EVAP``: Actual evapotranspiration [mm/day]

Map Output (mapoutput.txt)
--------------------------

Spatial maps for specific dates:

.. code-block:: text

   SUBID    COUT    SNOW    SOIL    SM1    SM2    SM3
   1        45.3    150.2   234.5   89.2   98.3   47.0
   2        32.1    180.4   198.2   75.1   85.3   37.8
   3        52.8    95.3    267.1   102.4  115.2  49.5
   ...

Time Output (timeoutput.txt)
----------------------------

Basin-averaged time series:

.. code-block:: text

   DATE        PREC    TEMP    EVAP    COUT    SNOW
   2010-01-01  5.2     2.3     0.8     45.3    150.2
   2010-01-02  0.0     3.1     0.9     42.8    145.8
   ...

Model-Specific Workflows
========================

Basic HYPE Workflow
-------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_watershed
   HYDROLOGICAL_MODEL: HYPE

   # Domain definition
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [15.5, 59.2]  # Sweden

   # Spatial discretization
   DISCRETIZATION:
     landclass:
       sources: ['MODIS']
     soilclass:
       sources: ['SoilGrids']

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # HYPE configuration
   HYPE_INSTALL_PATH: /path/to/hype
   HYPE_SPINUP_DAYS: 365

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

Multi-Land-Use HYPE Application
-------------------------------

For agricultural watersheds:

.. code-block:: yaml

   # config.yaml
   HYDROLOGICAL_MODEL: HYPE

   # Detailed land and soil discretization
   DISCRETIZATION:
     landclass:
       sources: ['MODIS', 'CroplandDataLayer']
       n_classes: 10  # Forest, crops, urban, etc.
     soilclass:
       sources: ['SoilGrids']
       n_classes: 5   # Sand, clay, loam, organic, rocky

   # Creates soil-land combinations (up to 50 SLCs)

   # Calibrate key parameters
   HYPE_PARAMS_TO_CALIBRATE: "gratk,rrcs1,rrcs2,ttmp,cmlt,wcfc,wcwp"

Snow-Focused HYPE
-----------------

For snow-dominated regions:

.. code-block:: yaml

   # Add elevation discretization
   DISCRETIZATION:
     elevation_bands:
       n_bands: 5
     landclass:
       sources: ['MODIS']
     soilclass:
       sources: ['SoilGrids']

   # Snow parameters to calibrate
   HYPE_PARAMS_TO_CALIBRATE: "ttmp,cmlt,sfrost,cmrad"

   # Parameters:
   # ttmp:   Threshold temp for snow/rain (°C)
   # cmlt:   Degree-day melt factor (mm/°C/day)
   # sfrost: Frost depth parameter
   # cmrad:  Radiation melt factor

Glacierised HYPE with Radiation-Driven Processes
------------------------------------------------

For an elevation-banded glacier catchment forced by sub-daily reanalysis that
includes shortwave radiation, humidity and wind (e.g. CARRA):

.. code-block:: yaml

   model:
     hype:
       pet_model: 5                  # FAO Penman-Monteith; writes SWobs, RHobs, Uobs
       snow_melt_model: 2            # temperature + radiation melt (snow and glacier ice)
       snowfall_fraction: wet_bulb   # SFobs from phase on every forcing step; snowfallmodel 1
       wet_bulb_snow_range: [0.0, 2.0]
       wind_measurement_height: 10.0 # forcing wind height; Uobs is reduced to 2 m (FAO-56 eq. 47)
       river_length_overrides: {1: 54485, 2: 26801, 3: 12486, 4: 26569, 5: 19023}
       extra_parameters:
         pcusnow: 0.0                # add a par.txt entry so it can be calibrated
       params_to_calibrate: ttmp,cmlt,cmrad,glaccmlt,glaccmrad,pcusnow,rivvel

- **SWobs** (MJ/m²/day) integrates downward shortwave flux over complete days.
- **RHobs** is the daily mean relative humidity as a fraction. It is read
  directly when the forcing has ``relative_humidity``; otherwise it is derived
  from specific humidity, pressure and temperature.
- **Uobs** is the daily mean wind speed at 2 m.
- **SFobs** is the daily snowfall fraction. Phase is decided on each forcing
  step, linear in wet-bulb temperature (Stull 2011) between the two thresholds,
  and precipitation-weighted. Dry days get the unweighted mean. With a single
  lumped forcing column, band temperatures are lapse-corrected before phase is
  decided.
- ``extra_parameters`` adds or replaces ``par.txt`` entries after generation. A
  scalar for a land-use or soil parameter is replicated to every class.
- A calibrated parameter that is missing from ``par.txt`` is an error, not a
  silent no-op.

Large-Scale HYPE (Continental)
------------------------------

HYPE scales well to large domains:

.. code-block:: yaml

   # E-HYPE style application (European scale)
   DOMAIN_DEFINITION_METHOD: merit_basins
   MERIT_BASIN_IDS: [1000001, 1000002, ..., 1005000]  # 5000 subcatchments

   HYDROLOGICAL_MODEL: HYPE

   # Simplified classes for computational efficiency
   DISCRETIZATION:
     landclass:
       n_classes: 5  # Major land uses only
     soilclass:
       n_classes: 3  # Coarse, medium, fine

   # Parallel calibration
   OPTIMIZATION_ALGORITHM: NSGA2
   OPTIMIZATION_POPULATION: 100
   OPTIMIZATION_MAX_ITERATIONS: 500

Calibration Strategies
======================

Recommended Parameters
----------------------

**Core hydrological parameters:**

.. code-block:: yaml

   HYPE_PARAMS_TO_CALIBRATE: "gratk,rrcs1,rrcs2,rrcs3,wcfc,wcwp"

**Parameter descriptions:**

- ``gratk``: Recession coefficient for runoff [1/day] (0.01-0.5)
- ``rrcs1``: Runoff coefficient soil layer 1 [-] (0.05-0.5)
- ``rrcs2``: Runoff coefficient soil layer 2 [-] (0.01-0.3)
- ``rrcs3``: Runoff coefficient soil layer 3 [-] (0.001-0.1)
- ``wcfc``: Soil water content at field capacity [-] (0.1-0.45)
- ``wcwp``: Soil water content at wilting point [-] (0.05-0.25)

**Snow-dominated basins:**

.. code-block:: yaml

   HYPE_PARAMS_TO_CALIBRATE: "gratk,rrcs1,rrcs2,ttmp,cmlt,sfrost"

- ``ttmp``: Threshold temperature for snowmelt [°C] (-2 to +2)
- ``cmlt``: Degree-day factor [mm/°C/day] (1-8)
- ``sfrost``: Frost depth parameter [cm] (5-50)

**Agricultural/nutrient modeling:**

.. code-block:: yaml

   # Hydrology + nutrients
   HYPE_PARAMS_TO_CALIBRATE: "gratk,rrcs1,rrcs2,wcfc,denitrlu,minerfn,dissolfN"

- ``denitrlu``: Denitrification parameter
- ``minerfn``: Mineralization parameter
- ``dissolfN``: Dissolved nitrogen parameter

Parameter Bounds
----------------

Typical ranges:

.. code-block:: python

   # Runoff and routing
   gratk:     [0.01, 0.5]      # Recession coefficient [1/day]
   rrcs1:     [0.05, 0.5]      # Runoff coeff layer 1
   rrcs2:     [0.01, 0.3]      # Runoff coeff layer 2
   rrcs3:     [0.001, 0.1]     # Runoff coeff layer 3

   # Soil properties
   wcfc:      [0.1, 0.45]      # Field capacity [-]
   wcwp:      [0.05, 0.25]     # Wilting point [-]
   wcep:      [0.35, 0.55]     # Effective porosity [-]

   # Snow parameters
   ttmp:      [-2, 2]          # Snow/rain threshold [°C]
   cmlt:      [1, 8]           # Melt factor [mm/°C/day]
   sfrost:    [5, 50]          # Frost depth [cm]
   cmrad:     [0.05, 0.15]     # Radiation factor [-]

   # Evapotranspiration
   lp:        [0.5, 1.0]       # Limit for PET [-]
   cevp:      [0.1, 0.3]       # Correction factor for PET

Calibration Tips
----------------

1. **Spinup is critical:**

   .. code-block:: yaml

      HYPE_SPINUP_DAYS: 365  # Minimum 1 year

2. **Soil parameters per class:**

   - Can calibrate wcfc, wcwp per soil type
   - Creates many parameters but improves realism

3. **Multi-site calibration:**

   - HYPE excels at multi-site (multiple gauges)
   - Helps constrain parameters spatially

4. **Use NSE and KGE together:**

   .. code-block:: yaml

      OBJECTIVE_METRICS: [NSE, KGE, RMSE_log]

5. **Start simple:**

   - Begin with 6-8 core parameters
   - Add complexity (class-specific) if needed

Known Limitations
=================

1. **Complexity:**

   - Many input files required
   - Soil-land class setup can be tedious
   - Steep learning curve

2. **Data Requirements:**

   - Needs detailed land use and soil maps
   - Lake data required for lake-dominated regions
   - Multiple forcing variables needed

3. **Computational Cost:**

   - Daily timestep somewhat slow for large domains
   - Thousands of subcatchments + many SLCs = slow calibration
   - Consider HPC for regional/continental applications

4. **Simplified Processes:**

   - Degree-day snowmelt (no energy balance)
   - Simplified groundwater (2 layers)
   - No detailed urban hydrology

5. **Water Quality:**

   - Nutrient modules require extensive parameterization
   - Difficult to calibrate hydrology + nutrients simultaneously

Troubleshooting
===============

Common Issues
-------------

**Error: "HYPE executable not found"**

.. code-block:: yaml

   # Solution: Verify installation
   HYPE_INSTALL_PATH: /absolute/path/to/hype/bin
   HYPE_EXE: hype

**Error: "GeoData.txt format error"**

Check that SYMFLUENCE generated valid GeoData:

.. code-block:: bash

   # Inspect GeoData.txt
   cat <project_dir>/settings/HYPE/GeoData.txt

   # Verify:
   # - SUBID column is integer
   # - AREA_km2 > 0
   # - SLC fractions sum to 1.0 per subcatchment

**Error: "Missing forcing files"**

.. code-block:: bash

   # Check forcing files exist
   ls <project_dir>/forcing/HYPE_input/P*.txt
   ls <project_dir>/forcing/HYPE_input/T*.txt

**Error: "Simulation crashed - water balance error"**

1. **Check parameter values:**

   - Ensure wcfc > wcwp
   - Ensure wcep > wcfc

2. **Increase spinup:**

   .. code-block:: yaml

      HYPE_SPINUP_DAYS: 730  # 2 years

3. **Check forcing data for gaps or extreme values**

**Poor performance in snow-dominated regions**

.. code-block:: yaml

   # Add elevation discretization
   DISCRETIZATION:
     elevation_bands:
       n_bands: 5

   # Calibrate snow parameters
   HYPE_PARAMS_TO_CALIBRATE: "gratk,rrcs1,rrcs2,ttmp,cmlt,sfrost,cmrad"

**Slow execution**

Solutions:

1. Reduce number of SLCs (combine similar classes)
2. Aggregate small subcatchments
3. Use coarser time steps if appropriate
4. Enable parallel HYPE (if compiled with OpenMP/MPI)

Performance Optimization
------------------------

**Speed up calibration:**

1. Shorter calibration period (5-7 years)
2. Fewer SLCs (<=10 per subcatchment)
3. Use DDS or DE algorithms (faster than NSGA2 for HYPE)

**Improve accuracy:**

1. Detailed SLC definition (land × soil × elevation)
2. Multi-site calibration with multiple gauges
3. Include snow cover data if available
4. Longer calibration period (10+ years)

**Debug HYPE run:**

.. code-block:: bash

   # Run HYPE manually to see detailed output
   cd <project_dir>/settings/HYPE/
   ../../installs/hype/hype

   # Check output for warnings/errors

Additional Resources
====================

**HYPE Documentation:**

- Official site: https://www.smhi.se/en/research/research-departments/hydrology/hype
- User guide: https://hypeweb.smhi.se/
- GitHub: https://github.com/HYPECODE-team/HYPE

**Publications:**

- Lindström et al. (2010): "Development and test of the distributed HBV-96 hydrological model"
  https://doi.org/10.1016/j.jhydrol.2010.03.007

- Arheimer et al. (2012): "European continental-scale hydrological modelling using the HYPE model"
  https://doi.org/10.5194/hess-16-2777-2012

- Donnelly et al. (2016): "Regional overview of nutrient load in Europe"
  https://doi.org/10.1016/j.scitotenv.2016.04.152

**SYMFLUENCE-specific:**

- :doc:`../configuration`: HYPE parameter reference
- :doc:`../calibration`: Calibration best practices
- :doc:`model_summa`: Comparison with SUMMA
- :doc:`../troubleshooting`: General troubleshooting

**Example Configurations:**

.. code-block:: bash

   # View HYPE examples
   symfluence examples list | grep HYPE

**Large-Scale Applications:**

- E-HYPE (European HYPE): https://hypeweb.smhi.se/explore-water/
- S-HYPE (Swedish HYPE): https://vattenwebb.smhi.se/

**Training Materials:**

- HYPE wiki: https://hypeweb.smhi.se/explore-water/documentation/
- Tutorials: https://hypeweb.smhi.se/explore-water/tutorials/
