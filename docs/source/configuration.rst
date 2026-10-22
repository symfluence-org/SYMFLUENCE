Configuration
=============

Overview
--------
All SYMFLUENCE workflows are driven by a single YAML configuration file.
This file defines domain setup, model selection, data sources, optimization strategy, and output behavior.
Configurations are modular, validated at runtime, and fully reproducible.

---

Structure
---------
Configurations are organized into logical blocks:

1. **Global and Logging Settings** — Experiment metadata and runtime controls
2. **Geospatial Definition** — Domain delineation and discretization
3. **Data and Forcings** — Input datasets and acquisition options
4. **Model Configuration** — Model selection and parameters
5. **Routing** — Flow routing with mizuRoute
6. **Calibration and Optimization** — Parameter estimation and metrics
7. **Emulation** — Differentiable emulators for large-domain workflows
8. **Paths and Resources** — Custom paths, file structure, and parallelism

---

Global and Logging Settings
---------------------------
Define experiment identifiers, paths, and computational options.

.. code-block:: yaml

   SYMFLUENCE_DATA_DIR: "/path/to/data"
   SYMFLUENCE_CODE_DIR: "/path/to/code"
   DOMAIN_NAME: "bow_river"
   EXPERIMENT_ID: "baseline_01"
   EXPERIMENT_TIME_START: "2018-01-01"
   EXPERIMENT_TIME_END: "2019-12-31"
   NUM_PROCESSES: 40
   LOG_LEVEL: INFO
   LOG_TO_FILE: True
   LOG_FORMAT: detailed

---

Geospatial Definition
---------------------
Configure watershed delineation, thresholds, and HRU discretization.

.. code-block:: yaml

   POUR_POINT_COORDS: 51.17/-115.57
   DOMAIN_DEFINITION_METHOD: semidistributed  # point | lumped | semidistributed | distributed
   SUBSET_FROM_GEOFABRIC: false               # Extract from existing geofabric
   GEOFABRIC_TYPE: TDX                        # TDX | MERIT | NWS (required if subset)
   GRID_SOURCE: generate                      # generate | native (for distributed)
   STREAM_THRESHOLD: 7500
   MULTI_SCALE_THRESHOLDS: [2500, 7500, 15000]
   USE_DROP_ANALYSIS: True
   DROP_ANALYSIS_NUM_THRESHOLDS: 5
   DELINEATE_COASTAL_WATERSHEDS: False
   SUB_GRID_DISCRETIZATION: elevation
   ELEVATION_BAND_SIZE: 400
   MIN_HRU_SIZE: 5
   RADIATION_CLASS_NUMBER: 8
   ASPECT_CLASS_NUMBER: 4

These parameters control how the domain is delineated and discretized before model setup.

**Definition Methods:**

- ``point``: Single point/pixel for FLUXNET sites
- ``lumped``: Single catchment, no spatial discretization
- ``semidistributed``: Subcatchment delineation from DEM via TauDEM
- ``distributed``: Grid-based representation with D8 routing

Set ``SUBSET_FROM_GEOFABRIC: true`` to extract from existing geofabric instead of delineating.

---

Data and Forcings
-----------------
Forcing data and meteorological drivers are defined here.

.. code-block:: yaml

   FORCING_DATASET: ERA5
   FORCING_VARIABLES: default            # comma-separated names, or "default"
   FORCING_TIME_STEP_SIZE: 3600
   APPLY_LAPSE_RATE: True
   LAPSE_RATE: -6.5

Optional extensions:
- **EM-Earth** integration for high-resolution precipitation/temperature downscaling
- **Supplemental forcing** and derived variables (e.g., PET via Priestley–Taylor)

---

Model Configuration
-------------------
Select hydrologic and routing models, and configure per-model parameters.

.. code-block:: yaml

   HYDROLOGICAL_MODEL: SUMMA            # SUMMA | FUSE | GR | LSTM | NextGen
   ROUTING_MODEL: mizuRoute

### SUMMA
.. code-block:: yaml

   SETTINGS_SUMMA_CONNECT_HRUS: yes
   SETTINGS_SUMMA_TRIALPARAM_N: 1
   SETTINGS_SUMMA_USE_PARALLEL_SUMMA: True
   SETTINGS_SUMMA_CPUS_PER_TASK: 32
   SETTINGS_SUMMA_GRU_PER_JOB: 10

### FUSE
.. code-block:: yaml

   FUSE_SPATIAL_MODE: distributed
   FUSE_DECISION_OPTIONS: default
   SETTINGS_FUSE_PARAMS_TO_CALIBRATE: [alpha, beta, k_storage]

### NextGen
.. code-block:: yaml

   NGEN_MODULES_SELECTED: "SLOTH,PET,CFE"
   NGEN_NOAH_PARAMS_TO_CALIBRATE: [bexp, dksat, psisat, refkdt]
   NGEN_ACTIVE_CATCHMENT_ID: 1002

### GR4J and LSTM
.. code-block:: yaml

   GR_SPATIAL_MODE: lumped
   LSTM_HIDDEN_SIZE: 256
   LSTM_EPOCHS: 100
   LSTM_USE_ATTENTION: True

---

Routing
-------
mizuRoute parameters for flow routing and network operations.

.. code-block:: yaml

   SETTINGS_MIZU_ROUTING_DT: 3600
   SETTINGS_MIZU_ROUTING_UNITS: seconds
   SETTINGS_MIZU_WITHIN_BASIN: 0
   SETTINGS_MIZU_OUTPUT_FREQ: daily
   SETTINGS_MIZU_OUTPUT_VARS: [Qout, Qsim, storage]

---

Calibration and Optimization
-----------------------------
Define calibration period, optimization algorithms, and objective metrics.

.. code-block:: yaml

   CALIBRATION_PERIOD: "2018-01-01,2018-06-30"
   EVALUATION_PERIOD: "2018-07-01,2018-12-31"
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, newSnowDenMin]
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30

Supported algorithms:
- **DE** – Differential Evolution
- **DDS** – Dynamically Dimensioned Search
- **PSO** – Particle Swarm Optimization
- **NSGA-II** – Multi-objective optimization

---


Paths and Resources
-------------------
Set paths for custom installations and parallel runtime behavior.

.. code-block:: yaml

   SUMMA_INSTALL_PATH: /opt/summa
   MIZUROUTE_INSTALL_PATH: /opt/mizuroute
   OUTPUT_DIR: /scratch/symfluence/output
   SETTINGS_SUMMA_TIME_LIMIT: 02:00:00
   SETTINGS_SUMMA_MEM: 12G

---

Parameter Reference
--------------------

This section provides a comprehensive reference of all configuration parameters, organized by category.

Global Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - SYMFLUENCE_DATA_DIR
     - string
     - required
     - Root directory for all project data and outputs
   * - SYMFLUENCE_CODE_DIR
     - string
     - required
     - Root directory for SYMFLUENCE source code
   * - DOMAIN_NAME
     - string
     - required
     - Unique identifier for the modeling domain
   * - EXPERIMENT_ID
     - string
     - "default"
     - Identifier for this simulation experiment
   * - EXPERIMENT_TIME_START
     - date
     - required
     - Start date for simulation (YYYY-MM-DD)
   * - EXPERIMENT_TIME_END
     - date
     - required
     - End date for simulation (YYYY-MM-DD)
   * - SPINUP_PERIOD
     - string
     - optional
     - Spinup period as "YYYY-MM-DD,YYYY-MM-DD"
   * - NUM_PROCESSES
     - integer
     - 1
     - Number of MPI processes for parallel execution
   * - FORCE_RUN_ALL_STEPS
     - boolean
     - False
     - Force re-execution of all workflow steps
   * - STOP_ON_ERROR
     - boolean
     - True
     - Stop workflow execution on first error

Logging Parameters
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - LOG_LEVEL
     - string
     - "INFO"
     - Logging verbosity (DEBUG, INFO, WARNING, ERROR)
   * - LOG_TO_FILE
     - boolean
     - True
     - Write logs to files in _workLog_*/
   * - LOG_FORMAT
     - string
     - "detailed"
     - Log message format (simple, detailed, json)

Domain Definition Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - POUR_POINT_COORDS
     - string
     - optional
     - Pour point as "latitude/longitude" (e.g., "51.17/-115.57")
   * - BOUNDING_BOX_COORDS
     - string
     - optional
     - Bounding box as "lat_max/lon_min/lat_min/lon_max"
   * - DOMAIN_DEFINITION_METHOD
     - string
     - "semidistributed"
     - Method for domain definition (point, lumped, semidistributed, distributed)
   * - SUBSET_FROM_GEOFABRIC
     - boolean
     - False
     - Extract domain from existing geofabric instead of delineating
   * - GRID_SOURCE
     - string
     - "generate"
     - Grid creation method for distributed (generate, native)
   * - NATIVE_GRID_DATASET
     - string
     - "era5"
     - Dataset identifier for native grid source
   * - GEOFABRIC_TYPE
     - string
     - "TDX"
     - Geofabric data source (TDX, MERIT, NWS)
   * - STREAM_THRESHOLD
     - integer
     - 7500
     - Flow accumulation threshold for stream network (cells)
   * - MULTI_SCALE_THRESHOLDS
     - list
     - optional
     - List of thresholds for multi-scale delineation
   * - USE_DROP_ANALYSIS
     - boolean
     - False
     - Use drop analysis to optimize stream threshold
   * - DROP_ANALYSIS_NUM_THRESHOLDS
     - integer
     - 5
     - Number of thresholds to test in drop analysis
   * - DELINEATE_COASTAL_WATERSHEDS
     - boolean
     - False
     - Allow delineation of coastal watersheds
   * - SUB_GRID_DISCRETIZATION
     - string
     - "lumped"
     - Discretization method (lumped, elevation, radiation, combined)
   * - ELEVATION_BAND_SIZE
     - integer
     - 400
     - Elevation band size in meters (for elevation discretization)
   * - MIN_HRU_SIZE
     - float
     - 5.0
     - Minimum HRU size in km² (smaller HRUs merged)
   * - RADIATION_CLASS_NUMBER
     - integer
     - 8
     - Number of radiation classes (for radiation discretization)
   * - ASPECT_CLASS_NUMBER
     - integer
     - 4
     - Number of aspect classes (for combined discretization)

Forcing Data Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - FORCING_DATASET
     - string
     - required
     - Forcing dataset source (ERA5, RDRS, CARRA, CARRA2, CERRA, CONUS404, etc.)
   * - FORCING_VARIABLES
     - string
     - "default"
     - Comma-separated meteorological variables to acquire, or "default"
   * - FORCING_TIME_STEP_SIZE
     - integer
     - 3600
     - Forcing data timestep in seconds
   * - FORCING_MEASUREMENT_HEIGHT
     - float
     - 2.0
     - Measurement height for meteorological variables (m)
   * - APPLY_LAPSE_RATE
     - boolean
     - False
     - Apply temperature lapse rate correction
   * - LAPSE_RATE
     - float
     - -6.5
     - Temperature lapse rate (°C/km)

Model Selection Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - HYDROLOGICAL_MODEL
     - string
     - required
     - Primary hydrological model (SUMMA, FUSE, GR, HYPE, MESH, NGEN, LSTM)
   * - ROUTING_MODEL
     - string
     - optional
     - Routing model (mizuRoute, auto)

SUMMA Model Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - SETTINGS_SUMMA_CONNECT_HRUS
     - string
     - "yes"
     - Connect HRUs for lateral flow ("yes", "no")
   * - SETTINGS_SUMMA_TRIALPARAM_N
     - integer
     - 1
     - Number of trial parameter sets
   * - SETTINGS_SUMMA_USE_PARALLEL_SUMMA
     - boolean
     - False
     - Use parallelized SUMMA execution
   * - SETTINGS_SUMMA_PARALLEL_BACKEND
     - string
     - "slurm"
     - SUMMA parallel backend ("slurm", "local")
   * - SETTINGS_SUMMA_CPUS_PER_TASK
     - integer
     - 1
     - CPUs per SUMMA task (for parallel execution)
   * - SETTINGS_SUMMA_GRU_PER_JOB
     - integer
     - 1
     - GRUs per parallel job
   * - SUMMA_INSTALL_PATH
     - string
     - auto
     - Path to SUMMA executable

FUSE Model Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - FUSE_SPATIAL_MODE
     - string
     - "lumped"
     - Spatial configuration (lumped, distributed)
   * - FUSE_DECISION_OPTIONS
     - string
     - "default"
     - Model structure decision options
   * - SETTINGS_FUSE_PARAMS_TO_CALIBRATE
     - list
     - []
     - List of FUSE parameters to calibrate

GR Model Parameters
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - GR_SPATIAL_MODE
     - string
     - "auto"
     - Spatial configuration (lumped, distributed, auto)
   * - GR_MODEL_TYPE
     - string
     - "GR4J"
     - GR model variant (GR4J, GR5J, GR6J)
   * - GR_USE_SNOW_MODULE
     - boolean
     - False
     - Enable CemaNeige snow module

HYPE Model Parameters
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - HYPE_SPINUP_DAYS
     - integer
     - 365
     - Number of spinup days

mizuRoute Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - MIZU_FROM_MODEL
     - string
     - auto
     - Source model for routing (auto-detected if not specified)
   * - SETTINGS_MIZU_ROUTING_DT
     - integer
     - 3600
     - Routing timestep in seconds
   * - SETTINGS_MIZU_ROUTING_UNITS
     - string
     - "seconds"
     - Time units for routing (seconds, hours, days)
   * - SETTINGS_MIZU_WITHIN_BASIN
     - integer
     - 0
     - Within-basin routing flag
   * - SETTINGS_MIZU_OUTPUT_FREQ
     - string
     - "daily"
     - Output frequency (daily, hourly, timestep)
   * - SETTINGS_MIZU_OUTPUT_VARS
     - list
     - ["Qout"]
     - List of output variables

Calibration Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - CALIBRATION_PERIOD
     - string
     - required
     - Calibration period as "YYYY-MM-DD,YYYY-MM-DD"
   * - EVALUATION_PERIOD
     - string
     - required
     - Evaluation period as "YYYY-MM-DD,YYYY-MM-DD"
   * - PARAMS_TO_CALIBRATE
     - list
     - required
     - List of parameters to calibrate
   * - OPTIMIZATION_ALGORITHM
     - string
     - "DDS"
     - Optimization algorithm (DDS, DE, PSO, NSGA2, ASYNC_DDS, POPULATION_DDS)
   * - OPTIMIZATION_METRIC
     - string
     - "KGE"
     - Objective metric (KGE, KGE_PRIME, NSE, RMSE, MAE)
   * - POPULATION_SIZE
     - integer
     - 20
     - Population size (for DE, PSO, NSGA2)
   * - NUMBER_OF_ITERATIONS
     - integer
     - 100
     - Maximum optimization iterations
   * - CALIBRATION_TIMESTEP
     - string
     - "daily"
     - Timestep for calibration evaluation (daily, hourly)

Output Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - OUTPUT_DIR
     - string
     - auto
     - Custom output directory (default: project_dir)

Path Parameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - CATCHMENT_PATH
     - string
     - auto
     - Path to catchment shapefile
   * - RIVER_NETWORK_SHP_PATH
     - string
     - auto
     - Path to river network shapefile
   * - DEM_PATH
     - string
     - auto
     - Path to DEM file
   * - LAND_CLASS_PATH
     - string
     - auto
     - Path to land classification file
   * - SOIL_CLASS_PATH
     - string
     - auto
     - Path to soil classification file

HPC Parameters
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 45

   * - Parameter
     - Type
     - Default
     - Description
   * - SETTINGS_SUMMA_TIME_LIMIT
     - string
     - "01:00:00"
     - SLURM time limit (HH:MM:SS)
   * - SETTINGS_SUMMA_MEM
     - string
     - "8G"
     - Memory allocation per job

---

Validation and Best Practices
-----------------------------
1. Validate before execution:

   .. code-block:: bash

      symfluence config validate --config my_project.yaml

2. Comment custom values and rationale within YAML.
3. Version-control all configuration files (except `config_active.yaml`).
4. Use template as baseline and document deviations clearly.
5. Test with small domains and short time periods first.
6. Use appropriate discretization for your modeling objectives:

   - Lumped: Fast, for calibration and large-scale studies
   - Elevation bands: Capture orographic effects
   - Radiation classes: Important for energy balance models
   - Combined: Maximum spatial detail (slower execution)

7. Match forcing timestep to model requirements:

   - SUMMA: Any timestep (typically 3600s)
   - FUSE: Daily recommended
   - GR: Daily only
   - HYPE: Daily only

---

Experimental Model Parameters
-----------------------------

WM-Fire Parameters (RHESSys Fire Module)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - RHESSYS_USE_WMFIRE
     - boolean
     - false
     - Enable WM-Fire module
   * - WMFIRE_GRID_RESOLUTION
     - integer
     - 30
     - Fire grid resolution in meters (10-200)
   * - WMFIRE_TIMESTEP_HOURS
     - integer
     - 24
     - Simulation timestep (1-24 hours)
   * - WMFIRE_NDAYS_AVERAGE
     - float
     - 30.0
     - Fuel moisture averaging window (days)
   * - WMFIRE_FUEL_SOURCE
     - string
     - "static"
     - Fuel source (static, rhessys_litter)
   * - WMFIRE_MOISTURE_SOURCE
     - string
     - "static"
     - Moisture source (static, rhessys_soil)
   * - WMFIRE_IGNITION_POINT
     - string
     - null
     - Ignition coordinates as "lat/lon"
   * - WMFIRE_IGNITION_DATE
     - string
     - null
     - Ignition date (YYYY-MM-DD)
   * - WMFIRE_WRITE_GEOTIFF
     - boolean
     - true
     - Write GeoTIFF outputs

Advanced Optimization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Parameter
     - Type
     - Default
     - Description
   * - GRADIENT_MODE
     - string
     - "auto"
     - Gradient mode (auto, native, finite_difference)
   * - GRADIENT_EPSILON
     - float
     - 1e-4
     - Finite difference perturbation size
   * - GRADIENT_CLIP_VALUE
     - float
     - 1.0
     - Maximum gradient L2 norm
   * - DDS_R
     - float
     - 0.2
     - DDS neighborhood size parameter
   * - PSO_COGNITIVE_PARAM
     - float
     - 1.5
     - PSO cognitive parameter (c1)
   * - PSO_SOCIAL_PARAM
     - float
     - 1.5
     - PSO social parameter (c2)
   * - PSO_INERTIA_WEIGHT
     - float
     - 0.7
     - PSO inertia weight
   * - DE_SCALING_FACTOR
     - float
     - 0.5
     - Differential Evolution F parameter
   * - DE_CROSSOVER_RATE
     - float
     - 0.9
     - Differential Evolution CR parameter

---
