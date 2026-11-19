.. _calibration:

Calibration and Optimization
============================

SYMFLUENCE provides comprehensive calibration capabilities for hydrological models through automated parameter estimation, multi-objective optimization, and robust evaluation frameworks.

Overview
--------

Model calibration in SYMFLUENCE follows a systematic approach:

1. **Parameter Selection** — Define which parameters to calibrate
2. **Objective Functions** — Choose appropriate metrics for optimization
3. **Algorithm Configuration** — Select optimization strategy
4. **Evaluation** — Assess calibrated model performance

Basic Calibration Setup
-----------------------

Configure calibration in your project YAML:

.. code-block:: yaml

   # Calibration periods
   CALIBRATION_PERIOD: "2018-01-01,2018-06-30"
   EVALUATION_PERIOD: "2018-07-01,2018-12-31"

   # Parameters to calibrate
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, newSnowDenMin]

   # Optimization settings
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30

Parameter Selection
-------------------

**SUMMA Parameters**

Common parameters for calibration:

.. code-block:: yaml

   PARAMS_TO_CALIBRATE:
     - k_snow          # snow thermal conductivity
     - fcapil          # capillary fringe thickness
     - newSnowDenMin   # minimum new snow density
     - theta_sat       # soil porosity
     - theta_res       # residual soil moisture
     - vGn_alpha       # van Genuchten alpha parameter
     - vGn_n           # van Genuchten n parameter

**FUSE Parameters**

For FUSE model calibration:

.. code-block:: yaml

   SETTINGS_FUSE_PARAMS_TO_CALIBRATE:
     - alpha           # baseflow recession parameter
     - beta            # percolation parameter
     - k_storage       # storage coefficient
     - qbrate_2c       # baseflow rate
     - percfrac        # percolation fraction

**NextGen Parameters**

Noah-OWP parameters:

.. code-block:: yaml

   NGEN_NOAH_PARAMS_TO_CALIBRATE:
     - bexp            # pore size distribution
     - dksat           # saturated hydraulic conductivity
     - psisat          # saturated soil potential
     - refkdt          # reference infiltration parameter

Optimization Algorithms
-----------------------

**Differential Evolution (DE)**

Robust global optimizer. Recommended for most applications.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: DE
   POPULATION_SIZE: 48
   NUMBER_OF_ITERATIONS: 30
   DE_SCALING_FACTOR: 0.5
   DE_CROSSOVER_RATE: 0.9

**Dynamically Dimensioned Search (DDS)**

Efficient for high-dimensional problems.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: DDS
   NUMBER_OF_ITERATIONS: 1000
   DDS_R: 0.2

**Particle Swarm Optimization (PSO)**

Good for continuous optimization problems.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: PSO
   POPULATION_SIZE: 30
   NUMBER_OF_ITERATIONS: 50
   PSO_INERTIA_WEIGHT: 0.7
   PSO_COGNITIVE_PARAM: 1.5
   PSO_SOCIAL_PARAM: 1.5

**Multi-Objective (NSGA-II)**

For multiple competing objectives.

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA-II
   NSGA2_PRIMARY_METRIC: KGE
   NSGA2_SECONDARY_METRIC: NSE
   POPULATION_SIZE: 100
   NUMBER_OF_ITERATIONS: 50

Objective Functions
-------------------

**Single Objective**

.. code-block:: yaml

   OPTIMIZATION_METRIC: KGE

Available metrics:
- **KGE** — Kling-Gupta Efficiency (recommended)
- **NSE** — Nash-Sutcliffe Efficiency
- **RMSE** — Root Mean Square Error
- **PBIAS** — Percent Bias
- **R2** — Coefficient of Determination

**Multi-Objective (NSGA-II / MOEA/D)**

Multi-objective algorithms optimize two metrics simultaneously:

.. code-block:: yaml

   OPTIMIZATION_ALGORITHM: NSGA-II
   NSGA2_PRIMARY_METRIC: KGE
   NSGA2_SECONDARY_METRIC: NSE

**Multivariate Objective**

Combine multiple target variables with per-variable weights and metrics:

.. code-block:: yaml

   OBJECTIVE_FUNCTION: MULTIVARIATE
   OBJECTIVE_METRICS:
     streamflow: KGE
     swe: NSE
   OBJECTIVE_WEIGHTS:
     streamflow: 0.7
     swe: 0.3

Unset variables use bias-sensitive default metrics (KGE; NSE for TWS;
``GMB_SCORE`` for ``GLACIER_MB``). Glacier mass balance, basin SWE, SCA and ET
can be evaluated for any model with an output adapter, and glacier mass
balance can be a COMPOSITE component or an NSGA-II objective; see
:doc:`multivariable_evaluation`.

Calibration Execution
---------------------

**Command Line**

.. code-block:: bash

   # Run calibration step
   symfluence workflow step calibrate_model --config my_project.yaml

   # Run full workflow including calibration
   symfluence workflow run --config my_project.yaml

   # Check workflow status
   symfluence workflow status --config my_project.yaml

**Python API**

.. code-block:: python

   from symfluence import SYMFLUENCE

   # Initialize SYMFLUENCE with configuration
   sf = SYMFLUENCE('my_project.yaml')

   # Run calibration step
   sf.run_individual_steps(['calibrate_model'])

   # Or run the optimization manager directly
   from symfluence.optimization.optimization_manager import OptimizationManager

   opt_manager = OptimizationManager(config, logger)
   results = opt_manager.run_optimization_workflow()

   # Get best parameters from results
   best_params = results.get('best_parameters', {})
   best_score = results.get('best_score', None)

Results and Evaluation
----------------------

**Output Files**

Calibration produces:

- ``calibration_results.csv`` — Parameter evolution
- ``best_parameters.yaml`` — Optimal parameter set
- ``objective_history.png`` — Convergence plot
- ``parameter_sensitivity.csv`` — Sensitivity analysis

**Validation**

Performance on the independent ``EVALUATION_PERIOD`` is reported automatically
alongside the calibration-period score using ``OPTIMIZATION_METRIC``.

Best Practices
--------------

1. **Parameter Bounds**

   Override default parameter ranges where needed:

   .. code-block:: yaml

      PARAMETER_BOUNDS:
        k_snow: [0.01, 1.0]
        theta_sat: [0.3, 0.6]

2. **Computational Efficiency**

   Run trials in parallel across worker processes:

   .. code-block:: yaml

      NUM_PROCESSES: 16

Troubleshooting
---------------

**Common Issues**

- **Slow convergence**: Increase population size or iterations
- **Parameter bounds**: Check realistic ranges for your domain
- **Memory issues**: Reduce the number of parallel processes
- **Poor performance**: Verify observation data quality

Example Workflows
-----------------

**Basic Single-Objective**

.. code-block:: yaml

   CALIBRATION_PERIOD: "2015-01-01,2017-12-31"
   EVALUATION_PERIOD: "2018-01-01,2020-12-31"
   PARAMS_TO_CALIBRATE: [k_snow, fcapil, theta_sat]
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_METRIC: KGE
   POPULATION_SIZE: 30
   NUMBER_OF_ITERATIONS: 50

**Multi-Objective with Validation**

.. code-block:: yaml

   CALIBRATION_PERIOD: "2010-01-01,2015-12-31"
   EVALUATION_PERIOD: "2016-01-01,2018-12-31"
   PARAMS_TO_CALIBRATE: [alpha, beta, k_storage, qbrate_2c]
   OPTIMIZATION_ALGORITHM: NSGA-II
   NSGA2_PRIMARY_METRIC: KGE
   NSGA2_SECONDARY_METRIC: NSE
   POPULATION_SIZE: 100
   NUMBER_OF_ITERATIONS: 100

---

**See Also**

- :doc:`multivariable_evaluation` — glacier mass balance, SWE, SCA and ET for any model
- :doc:`configuration` — Complete parameter reference
- :doc:`troubleshooting` — Calibration troubleshooting and diagnostics
- :doc:`api` — Programmatic calibration control
