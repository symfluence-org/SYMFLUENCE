=========================================
HBV Model Guide
=========================================

.. warning::
   **EXPERIMENTAL MODULE** - The HBV model is in active development and should be
   used at your own risk. The API may change without notice in future releases.
   Please report any issues at https://github.com/symfluence-org/SYMFLUENCE/issues

   **Known Limitations:**

   - Distributed routing integration is still in development
   - Some parameter combinations may produce numerical instabilities
   - GPU acceleration requires JAX with CUDA/ROCm support

   To disable this module at import time, set: ``SYMFLUENCE_DISABLE_EXPERIMENTAL=1``

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The HBV-96 (Hydrologiska Byråns Vattenbalansavdelning) model is a conceptual rainfall-runoff model originally developed by the Swedish Meteorological and Hydrological Institute (SMHI). SYMFLUENCE implements a modern JAX-based version enabling automatic differentiation for gradient-based calibration.

**Key Capabilities:**

- Pure JAX implementation for autodiff and JIT compilation
- 14 physically-meaningful parameters
- Four-routine structure (snow, soil, response, routing)
- GPU acceleration when available
- NumPy fallback for environments without JAX
- Lumped and distributed spatial modes
- Optional mizuRoute integration for channel routing
- Differentiable loss functions (NSE, KGE)

**Typical Applications:**

- Operational flood forecasting
- Climate change impact assessment
- Snow-dominated catchments
- Gradient-based parameter optimization
- Ensemble simulations via vectorization (vmap)
- Benchmark comparisons

**Spatial Scales:** Small catchments (1 km²) to large basins (50,000 km²)

**Temporal Resolution:** Daily to hourly (sub-daily supported via automatic parameter scaling)

Model Structure
===============

The HBV-96 model consists of four main routines executed in sequence:

.. code-block:: text

   Precipitation, Temperature, PET
              ↓
   ┌─────────────────────────────┐
   │      1. SNOW ROUTINE        │  Degree-day accumulation/melt
   │   Snow pack ↔ Liquid water  │  with refreezing
   └─────────────────────────────┘
              ↓ Rainfall + Melt
   ┌─────────────────────────────┐
   │      2. SOIL ROUTINE        │  Beta-function recharge
   │   Soil moisture → Recharge  │  ET reduction below LP
   └─────────────────────────────┘
              ↓ Recharge
   ┌─────────────────────────────┐
   │    3. RESPONSE ROUTINE      │  Two-box groundwater model
   │   Upper zone → Lower zone   │  Fast/slow flow generation
   └─────────────────────────────┘
              ↓ Total runoff
   ┌─────────────────────────────┐
   │    4. ROUTING ROUTINE       │  Triangular transfer function
   │   Convolution smoothing     │  (MAXBAS parameter)
   └─────────────────────────────┘
              ↓
         Streamflow

Snow Routine
------------

Partitions precipitation into rain and snow based on threshold temperature (TT).
Uses degree-day method for snowmelt with refreezing capability.

**Processes:**

- Rain/snow partitioning at threshold temperature
- Degree-day melt: ``melt = CFMAX × max(T - TT, 0)``
- Snowfall correction factor (SFCF)
- Refreezing of liquid water in snowpack
- Water holding capacity (CWH)

Soil Routine
------------

Controls soil moisture dynamics and actual evapotranspiration.

**Processes:**

- Non-linear recharge: ``recharge = input × (SM/FC)^BETA``
- ET reduction below LP threshold
- Soil moisture budget

Response Routine
----------------

Two-box groundwater model generating runoff components.

**Processes:**

- Upper zone: Surface runoff (Q0) and interflow (Q1)
- Lower zone: Baseflow (Q2)
- Percolation from upper to lower zone
- Threshold-based fast flow (UZL)

Routing Routine
---------------

Triangular transfer function for hydrograph smoothing.

**Processes:**

- Convolution with triangular weights
- Response time controlled by MAXBAS

Parameters
==========

HBV-96 has 14 parameters organized by routine:

Snow Routine Parameters
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 50

   * - Parameter
     - Unit
     - Bounds
     - Description
   * - TT
     - °C
     - [-3, 3]
     - Threshold temperature for snow/rain partitioning
   * - CFMAX
     - mm/°C/day
     - [1, 10]
     - Degree-day factor for snowmelt
   * - SFCF
     - -
     - [0.5, 1.5]
     - Snowfall correction factor (undercatch adjustment)
   * - CFR
     - -
     - [0, 0.1]
     - Refreezing coefficient
   * - CWH
     - -
     - [0, 0.2]
     - Water holding capacity of snow (fraction)

Soil Routine Parameters
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 50

   * - Parameter
     - Unit
     - Bounds
     - Description
   * - FC
     - mm
     - [50, 700]
     - Maximum soil moisture storage / field capacity
   * - LP
     - -
     - [0.3, 1.0]
     - Soil moisture threshold for ET reduction (fraction of FC)
   * - BETA
     - -
     - [1, 6]
     - Shape coefficient for recharge function

Response Routine Parameters
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 50

   * - Parameter
     - Unit
     - Bounds
     - Description
   * - K0
     - 1/day
     - [0.05, 0.99]
     - Recession coefficient for surface runoff (fast)
   * - K1
     - 1/day
     - [0.01, 0.5]
     - Recession coefficient for interflow (medium)
   * - K2
     - 1/day
     - [0.0001, 0.1]
     - Recession coefficient for baseflow (slow)
   * - UZL
     - mm
     - [0, 100]
     - Threshold for surface runoff generation
   * - PERC
     - mm/day
     - [0, 10]
     - Maximum percolation rate to lower zone

Routing Parameter
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 50

   * - Parameter
     - Unit
     - Bounds
     - Description
   * - MAXBAS
     - days
     - [1, 7]
     - Length of triangular routing function

Sub-Daily Simulation
====================

SYMFLUENCE's HBV implementation supports sub-daily timesteps (e.g., hourly) through
automatic parameter scaling. Parameters are always specified in **daily units** per
HBV-96 convention; the model internally scales them for the chosen timestep.

Enabling Sub-Daily Simulation
-----------------------------

.. code-block:: yaml

   # Set timestep in hours (1-24)
   HBV_TIMESTEP_HOURS: 1  # hourly simulation

   # Forcing files must match the timestep
   # e.g., <domain>_hbv_forcing_1h.nc for hourly

Parameter Scaling
-----------------

Different parameter types require different scaling approaches:

**Flux Rate Parameters (linear scaling):**

Parameters with units of mm/day are scaled linearly:

.. math::

   P_{sub} = P_{daily} \times \frac{\Delta t}{24}

Applies to: ``CFMAX``, ``PERC``

**Recession Coefficients (exact exponential scaling):**

Recession coefficients represent exponential reservoir decay. Linear scaling introduces
errors up to ~13% for typical parameter values. SYMFLUENCE uses the mathematically
exact transformation:

.. math::

   k_{sub} = 1 - (1 - k_{daily})^{\frac{\Delta t}{24}}

Applies to: ``K0``, ``K1``, ``K2``

This ensures that running 24 hourly timesteps produces identical results to running
1 daily timestep (to numerical precision).

**Why Exact Scaling Matters:**

Consider K0 = 0.3 (30% drains per day):

- Linear approximation: k_hourly = 0.3/24 = 0.0125 → Daily drain = 26% (13% error)
- Exact formula: k_hourly = 0.0147 → Daily drain = 30% (exact)

The error compounds during high-flow events, causing peak flow discrepancies.

**Unchanged Parameters:**

- Dimensionless: ``SFCF``, ``CFR``, ``CWH``, ``LP``, ``BETA``
- Thresholds: ``TT``, ``FC``, ``UZL``
- Duration: ``MAXBAS`` (remains in days; routing buffer adjusted internally)

Sub-Daily Forcing Requirements
------------------------------

Sub-daily simulation requires forcing data at the specified timestep:

.. code-block:: text

   # Hourly forcing file
   <domain>_hbv_forcing_1h.nc

   Dimensions:
     time: N_HOURS

   Variables:
     pr(time)    - Precipitation [mm/hour]
     temp(time)  - Temperature [°C]
     pet(time)   - PET [mm/hour]

.. note::

   Ensure forcing units match the timestep (mm/hour for hourly, not mm/day).
   The model expects fluxes per timestep, not per day.

Validation Approach
-------------------

When validating sub-daily implementation against daily observations:

1. **Calibrate at daily timestep** (where real observations exist)
2. **Run sub-daily with same parameters** (tests scaling correctness)
3. **Aggregate sub-daily outputs to daily** for comparison
4. **Check consistency** - results should match closely (r > 0.999)

This approach avoids circular logic when sub-daily observations are not available.

Spatial Modes
=============

Lumped Mode
-----------

Single catchment simulation with basin-averaged forcing:

.. code-block:: yaml

   HBV_SPATIAL_MODE: lumped

   # Basin-averaged forcing
   # 14 parameters
   # Fastest execution

**Use case:** Simple basins, rapid assessment, benchmarking

Distributed Mode
----------------

Per-HRU simulation with optional mizuRoute routing:

.. code-block:: yaml

   HBV_SPATIAL_MODE: distributed

   # Requires HRU discretization
   DOMAIN_DEFINITION_METHOD: semidistributed

   # Optional routing
   ROUTING_MODEL: mizuRoute
   HBV_ROUTING_INTEGRATION: mizuRoute

**Use case:** Large basins, heterogeneous catchments, routing required

Auto Mode (Default)
-------------------

Automatically selects mode based on domain definition:

.. code-block:: yaml

   HBV_SPATIAL_MODE: auto  # Default

   # delineate → distributed
   # polygon → lumped

Configuration in SYMFLUENCE
===========================

Model Selection
---------------

.. code-block:: yaml

   HYDROLOGICAL_MODEL: HBV

Key Configuration Parameters
----------------------------

Runtime Configuration
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HBV_SPATIAL_MODE
     - auto
     - Spatial mode (auto, lumped, distributed)
   * - HBV_TIMESTEP_HOURS
     - 24
     - Model timestep in hours (1-24). Use 1 for hourly, 24 for daily.
   * - HBV_BACKEND
     - jax
     - Computation backend (jax, numpy)
   * - HBV_USE_GPU
     - False
     - Enable GPU acceleration (requires JAX+CUDA)
   * - HBV_JIT_COMPILE
     - True
     - JIT compile model (JAX only)
   * - HBV_WARMUP_DAYS
     - 365
     - Spinup period (days)

Initial State Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HBV_INITIAL_SNOW
     - 0.0
     - Initial snow storage (mm SWE)
   * - HBV_INITIAL_SM
     - 150.0
     - Initial soil moisture (mm)
   * - HBV_INITIAL_SUZ
     - 10.0
     - Initial upper zone storage (mm)
   * - HBV_INITIAL_SLZ
     - 10.0
     - Initial lower zone storage (mm)

Calibration Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HBV_PARAMS_TO_CALIBRATE
     - tt,cfmax,fc,lp,beta,k0,k1,k2,uzl,perc,maxbas
     - Parameters to calibrate (comma-separated)
   * - HBV_USE_GRADIENT_CALIBRATION
     - True
     - Use gradient-based optimization (requires JAX)
   * - HBV_CALIBRATION_METRIC
     - KGE
     - Objective function (KGE, NSE)

PET Configuration
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HBV_PET_METHOD
     - input
     - PET method (input, hamon, thornthwaite)
   * - HBV_LATITUDE
     - (auto)
     - Latitude for PET calculation (if not from shapefile)

Output Configuration
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - HBV_SAVE_STATES
     - False
     - Save internal state variables to output
   * - HBV_OUTPUT_FREQUENCY
     - daily
     - Output frequency (daily, timestep)

Input File Specifications
=========================

HBV requires preprocessed forcing data with precipitation, temperature, and PET.

Lumped Forcing (CSV)
--------------------

**File:** ``<domain>_hbv_forcing.csv``

.. code-block:: text

   time,pr,temp,pet
   2015-01-01,5.2,2.3,1.1
   2015-01-02,0.0,3.1,1.3
   2015-01-03,12.4,1.8,0.9
   ...

**Columns:**

- ``time``: Date (YYYY-MM-DD)
- ``pr``: Precipitation (mm/day)
- ``temp``: Mean temperature (°C)
- ``pet``: Potential evapotranspiration (mm/day)

Lumped Forcing (NetCDF)
-----------------------

**File:** ``<domain>_hbv_forcing.nc``

.. code-block:: text

   Dimensions:
     time: N_DAYS

   Variables:
     pr(time)     - Precipitation [mm/day]
     temp(time)   - Temperature [°C]
     pet(time)    - PET [mm/day]

Distributed Forcing (NetCDF)
----------------------------

**File:** ``<domain>_hbv_forcing_distributed.nc``

.. code-block:: text

   Dimensions:
     time: N_DAYS
     hru: N_HRUS

   Variables:
     pr(time, hru)    - Precipitation [mm/day]
     temp(time, hru)  - Temperature [°C]
     pet(time, hru)   - PET [mm/day]
     hru_id(hru)      - HRU identifiers

Output File Specifications
==========================

Lumped Output
-------------

**CSV File:** ``<domain>_hbv_output.csv``

.. code-block:: text

   datetime,streamflow_mm_day,streamflow_cms
   2015-01-01,1.23,45.6
   2015-01-02,1.18,43.8
   ...

**NetCDF File:** ``<domain>_hbv_output.nc``

.. code-block:: text

   Dimensions:
     time: N_DAYS

   Variables:
     streamflow(time)  - Discharge [m³/s]
     runoff(time)      - Runoff depth [mm/day]

   Attributes:
     model: HBV-96
     spatial_mode: lumped
     catchment_area_m2: AREA

Distributed Output (for mizuRoute)
----------------------------------

**File:** ``<domain>_<experiment>_runs_def.nc``

.. code-block:: text

   Dimensions:
     time: N_DAYS
     gru: N_HRUS

   Variables:
     gruId(gru)        - HRU identifiers
     q_routed(time, gru) - Runoff [m/s] for routing

Model-Specific Workflows
========================

Basic Lumped Workflow
---------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: test_basin
   HYDROLOGICAL_MODEL: HBV

   # Lumped mode
   HBV_SPATIAL_MODE: lumped
   DOMAIN_DEFINITION_METHOD: polygon
   CATCHMENT_SHP_PATH: ./basin.shp

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # Calibration with gradients
   OPTIMIZATION_ALGORITHM: ADAM
   HBV_USE_GRADIENT_CALIBRATION: True
   HBV_CALIBRATION_METRIC: KGE

Run:

.. code-block:: bash

   symfluence workflow run --config config.yaml

Distributed with Routing
------------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: large_basin
   HYDROLOGICAL_MODEL: HBV

   # Distributed mode
   HBV_SPATIAL_MODE: distributed
   DOMAIN_DEFINITION_METHOD: semidistributed
   POUR_POINT_COORDS: [-118.5, 49.2]

   # Enable routing
   ROUTING_MODEL: mizuRoute
   HBV_ROUTING_INTEGRATION: mizuRoute

   # Forcing
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2005
   FORCING_END_YEAR: 2020

Snow-Dominated Catchment
------------------------

.. code-block:: yaml

   # Optimize snow parameters
   DOMAIN_NAME: alpine_basin
   HYDROLOGICAL_MODEL: HBV

   # Ensure snow parameters are calibrated
   HBV_PARAMS_TO_CALIBRATE: tt,cfmax,sfcf,cfr,cwh,fc,lp,beta,k0,k1,k2,uzl,perc,maxbas

   # Appropriate initial snow for winter start
   HBV_INITIAL_SNOW: 100.0

   # Longer warmup for snow equilibration
   HBV_WARMUP_DAYS: 730

Calibration Strategies
======================

Gradient-Based Calibration (Recommended)
----------------------------------------

HBV's JAX implementation enables efficient gradient-based optimization:

.. code-block:: yaml

   # Use ADAM optimizer with gradients
   OPTIMIZATION_ALGORITHM: ADAM
   HBV_USE_GRADIENT_CALIBRATION: True

   # Or L-BFGS for faster convergence
   OPTIMIZATION_ALGORITHM: LBFGS

**Advantages:**

- Faster convergence than evolutionary algorithms
- Efficient for 14-parameter problems
- Exact gradients via autodiff

Evolutionary Calibration
------------------------

Fall back to evolutionary methods when JAX unavailable:

.. code-block:: yaml

   HBV_USE_GRADIENT_CALIBRATION: False
   OPTIMIZATION_ALGORITHM: DDS
   OPTIMIZATION_MAX_ITERATIONS: 2000

Parameter Sensitivity
---------------------

**High sensitivity (always calibrate):**

- FC - Field capacity
- BETA - Recharge shape
- K1, K2 - Recession coefficients

**Moderate sensitivity:**

- TT, CFMAX - Snow parameters (snow-dominated only)
- LP - ET threshold
- MAXBAS - Routing time

**Lower sensitivity (can fix):**

- SFCF, CFR, CWH - Snow corrections
- K0, UZL - Fast flow (flashy catchments only)
- PERC - Percolation

Reduced Parameter Set
---------------------

For faster calibration, use a reduced parameter set:

.. code-block:: yaml

   # 7-parameter version (good performance, faster)
   HBV_PARAMS_TO_CALIBRATE: fc,lp,beta,k1,k2,perc,maxbas

Sub-Daily Calibration
---------------------

When calibrating at sub-daily timesteps:

1. **Ensure observations match timestep** - Calibrating hourly models against
   interpolated daily observations produces poor results.

2. **Parameters remain in daily units** - The calibration optimizes parameters in
   their standard daily units; scaling is handled internally.

3. **Computational cost** - Hourly calibration is ~24× more expensive than daily.
   Consider calibrating at daily timestep first, then validating at hourly.

.. code-block:: yaml

   # Hourly calibration (requires hourly observations)
   HBV_TIMESTEP_HOURS: 1
   OPTIMIZATION_MAX_ITERATIONS: 1000  # May need fewer iterations

JAX Features
============

JIT Compilation
---------------

HBV automatically JIT-compiles for faster execution:

.. code-block:: yaml

   HBV_JIT_COMPILE: True  # Default

First run compiles the model; subsequent runs are much faster.

GPU Acceleration
----------------

Enable GPU for large ensembles:

.. code-block:: yaml

   HBV_USE_GPU: True
   HBV_BACKEND: jax

Requires JAX with CUDA support:

.. code-block:: bash

   pip install jax[cuda12_pip]

Ensemble Simulations
--------------------

Use ``simulate_ensemble`` for efficient parallel runs:

.. code-block:: python

   from symfluence.models.hbv.model import simulate_ensemble

   # Run 100 parameter sets in parallel
   results = simulate_ensemble(
       precip, temp, pet,
       params_batch={
           'fc': np.random.uniform(100, 500, 100),
           'beta': np.random.uniform(1, 5, 100),
           # ... other parameters
       }
   )

NumPy Fallback
--------------

When JAX is unavailable, HBV automatically uses NumPy:

.. code-block:: yaml

   HBV_BACKEND: numpy

Features available:

- Full model simulation
- Parameter optimization (no gradients)
- Single-threaded execution

Features NOT available without JAX:

- Gradient computation
- JIT compilation
- GPU acceleration
- Vectorized ensemble runs

Known Limitations
=================

1. **Sub-Daily Observations:**

   - Sub-daily simulation is supported, but calibration requires observations at the
     same timestep for meaningful comparison
   - When only daily observations are available, validate by comparing daily vs
     aggregated-hourly simulations using the same parameters

2. **Conceptual Model:**

   - Parameters lack direct physical measurement
   - Equifinality in parameter space

3. **Snow Module:**

   - Simple degree-day approach
   - No energy balance
   - May struggle with rain-on-snow events

4. **No Explicit Vegetation:**

   - ET controlled by soil moisture only
   - No interception or transpiration partitioning

5. **JAX Dependency:**

   - Full functionality requires JAX
   - NumPy fallback has reduced features

Troubleshooting
===============

Common Issues
-------------

**Error: "JAX not available"**

.. code-block:: bash

   # Install JAX
   pip install jax jaxlib

   # For GPU support
   pip install jax[cuda12_pip]

**Error: "Forcing file not found"**

Ensure preprocessing ran successfully:

.. code-block:: bash

   # Check forcing directory
   ls <project>/forcing/HBV_input/

   # Re-run preprocessing
   symfluence preprocess --model HBV

**Poor calibration performance**

1. Check forcing data quality:

   .. code-block:: python

      import pandas as pd
      df = pd.read_csv('basin_hbv_forcing.csv')
      print(df.describe())
      print(df.isnull().sum())

2. Extend warmup period:

   .. code-block:: yaml

      HBV_WARMUP_DAYS: 730  # 2 years

3. Try different optimizer:

   .. code-block:: yaml

      OPTIMIZATION_ALGORITHM: DDS
      HBV_USE_GRADIENT_CALIBRATION: False

**NaN in simulation output**

Usually caused by invalid parameters:

.. code-block:: yaml

   # Ensure K0 > K1 > K2 for physical consistency
   # Check parameter bounds

**Memory issues with large ensembles**

Reduce batch size or use CPU:

.. code-block:: yaml

   HBV_USE_GPU: False

Performance Tips
----------------

**Speed up simulation:**

1. Enable JIT compilation (default)
2. Use GPU for ensembles > 100 members
3. Reduce warmup if initializing from known state

**Improve calibration:**

1. Use gradient-based methods (ADAM, L-BFGS)
2. Start from reasonable initial parameters
3. Calibrate sensitive parameters first
4. Use KGE for balanced performance

Additional Resources
====================

**HBV Model Documentation:**

- Lindström et al. (1997): "Development and test of the distributed HBV-96 model"
  https://doi.org/10.1016/S0022-1694(97)00041-3

- SMHI HBV Documentation: https://www.smhi.se/en/research/research-departments/hydrology/hbv-1.90007

**JAX Resources:**

- JAX Documentation: https://jax.readthedocs.io/
- JAX Installation: https://github.com/google/jax#installation

**SYMFLUENCE-specific:**

- :doc:`../configuration`: Full parameter reference
- :doc:`../calibration`: Calibration best practices
- :doc:`model_gr`: Comparison with GR models
- :doc:`model_summa`: Comparison with SUMMA
- :doc:`../troubleshooting`: General troubleshooting

**Example Configurations:**

.. code-block:: bash

   # List HBV examples
   symfluence examples list | grep HBV

   # Copy example
   symfluence examples copy bow_river_hbv ./my_hbv_project
