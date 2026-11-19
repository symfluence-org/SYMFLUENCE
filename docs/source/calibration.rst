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

Split-sample design
-------------------

Calibration is scored on ``CALIBRATION_PERIOD`` and the calibrated model is
judged on periods it never saw. Up to four non-overlapping windows inside
``EXPERIMENT_TIME_START`` .. ``EXPERIMENT_TIME_END`` are recognised:

.. code-block:: yaml

   SPINUP_PERIOD:      "2000-01-01, 2001-12-31"   # warm-up: never scored
   CALIBRATION_PERIOD: "2002-01-01, 2009-12-31"   # the optimisation objective
   EVALUATION_PERIOD:  "2010-01-01, 2014-12-31"   # held out, reported (Eval_*)
   TEST_PERIOD:        "2015-01-01, 2018-12-31"   # optional, independent (Test_*)

- **Spin-up** is one notion. Output before the end of ``SPINUP_PERIOD`` is
  dropped from every score. The deprecated ``EVALUATION_SPINUP_YEARS`` (N years
  from the simulation start) still works and maps onto the same window; when
  both are set, the later end wins.
- **Evaluation and test** periods are scored only in the final evaluation
  (``calibration_only=False``), never inside the optimisation loop. Use
  ``EVALUATION_PERIOD`` to compare runs and choose between them; keep
  ``TEST_PERIOD`` for a final, independent check that nothing is selected on.
- **Benchmarks.** The final evaluation also scores three reference predictors
  on exactly the same periods: the calibration-period mean flow (KGE
  :math:`= 1-\sqrt{2} \approx -0.41` on the calibration period), the
  calibration-period day-of-year climatology, and persistence (the previous
  observation). They are fitted on the calibration period only. Keys are
  ``<Period>_KGE_benchmark_<name>`` and the skill relative to each,
  ``<Period>_KGE_skill_<name>`` :math:`= (KGE - KGE_{bm}) / (1 - KGE_{bm})`
  (1 is perfect, 0 no better than the benchmark). The hydrobm benchmarking
  analysis uses the same configured periods.

Problems are reported as warnings, not errors, so existing configurations keep
running. At config load, SYMFLUENCE warns when calibration, evaluation and
test periods overlap, when spin-up does not end before a scored period, and
when ``CALIBRATION_PERIOD`` is missing while an evaluation/test period is set or
an optimisation method is configured. Without ``CALIBRATION_PERIOD``, or when it
has no overlap with the data, metrics fall back to the full record (which
contains the held-out years): a ``SPLIT-SAMPLE WARNING`` is logged and the
metrics dictionary carries a ``period_warning`` entry explaining why.

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

Forcing corrections (precipitation multipliers such as GR ``PCORR`` or SUMMA
``precip_mult``) can be shared by every model of an ensemble, fixed or
calibrated with common bounds and a prior: see :ref:`forcing_correction`.

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

Performance on the independent ``EVALUATION_PERIOD`` (and ``TEST_PERIOD``, if
set) is reported automatically alongside the calibration-period score, with
skill relative to the benchmarks; see `Split-sample design`_.

Checkpoint and Resume
---------------------

Long calibrations (thousands of evaluations of SUMMA, ParFlow, ...) can be
continued after the process dies, instead of starting over. Every
``CHECKPOINT_INTERVAL`` iterations (default 10; an iteration is one
evaluation for DDS and one generation for population algorithms) the
optimizer writes, next to the iteration-results CSV and best-parameter JSON,

``<EXPERIMENT_ID>_<algorithm>_resume_state.json``

atomically (temporary file + rename, so a crash mid-write leaves the previous
checkpoint intact). It holds the ordered log of every evaluation request and
its result, the iteration and evaluation counters, the random seed (or, for an
unseeded run, the seed it drew, which a resume reuses) and the random-generator state, the best
solution, crash counters, the iteration history, and a fingerprint of the
calibration setup.

To continue an interrupted run, rerun the same configuration with
``--resume``:

.. code-block:: bash

   symfluence workflow step calibrate_model --config my_config.yaml --resume
   # or: symfluence workflow run --config my_config.yaml --resume

or set ``CALIBRATION_RESUME: true`` in the configuration. The resumed run
replays the algorithm against the logged evaluations, which takes seconds and
runs no model, to restore its exact internal state (DDS's current solution and
position in its perturbation schedule, DE/PSO/SCE-UA/NSGA-II populations and
fitnesses, DREAM chains, GLUE samples, the simulated-annealing temperature,
CMA-ES covariance, L-BFGS history, and the random-number generator), then
continues with real model runs. A seeded resumed run ends with the same result
as an uninterrupted one. The results CSV and best-parameter JSON are
continued: earlier rows keep their timestamps, and nothing is moved aside.
Evaluations made after the last checkpoint are run again, so a smaller
``CHECKPOINT_INTERVAL`` loses less work (``CHECKPOINT_INTERVAL: 1`` checkpoints
after every iteration).

Resume refuses, with ``CalibrationResumeError``, rather than silently starting
over when:

- the calibrated parameters, their bounds, the algorithm, the model,
  ``OPTIMIZATION_METRIC``, ``CALIBRATION_PERIOD``, ``RANDOM_SEED``,
  ``POPULATION_SIZE``, the calibration budget (``CALIBRATION_BUDGET_MODE``
  and its settings) or the shared forcing correction (``FORCING_CORRECTION_*``)
  changed since the checkpoint (the error names the
  setting);
- the replay diverges from the log: the algorithm asks for different
  parameter sets than were recorded, or its random-number generator is not in
  the checkpointed state afterwards. This happens when other settings that
  steer the search changed (``NUMBER_OF_ITERATIONS`` for DDS, algorithm
  options, ``INITIAL_GUESS``, ``NUM_PROCESSES`` for Async-DDS batch sizes) or
  after a SYMFLUENCE upgrade that changed the algorithm;
- the algorithm does not support resume (an externally registered algorithm
  class can declare ``supports_resume = False``; all built-in algorithms
  support it).

If there is no checkpoint, ``--resume`` logs a warning and starts a new run.
Without ``--resume``, a new run moves the previous run's results and resume
state aside as ``*.prev-<timestamp>`` files, as before.

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

- :doc:`configuration` — Complete parameter reference
- :doc:`troubleshooting` — Calibration troubleshooting and diagnostics
- :doc:`api` — Programmatic calibration control
