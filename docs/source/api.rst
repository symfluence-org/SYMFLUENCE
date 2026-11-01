=========================================
API Reference
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The SYMFLUENCE Python API provides programmatic access to the full workflow, from project setup through calibration and analysis. The primary entry point is the ``SYMFLUENCE`` class, which coordinates manager components through the ``WorkflowOrchestrator``.

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from symfluence import SYMFLUENCE

   # Initialize from configuration file
   conf = SYMFLUENCE("my_project.yaml")

   # Run complete workflow
   conf.run_workflow()

   # Or run a selected subset of steps (by canonical step name)
   conf.run_individual_steps([
       "setup_project",
       "define_domain",
       "acquire_forcings",
       "run_model",
   ])

Step-by-Step Execution
----------------------

The ``SYMFLUENCE`` facade does not expose one method per step. Drive the workflow
through ``run_workflow()`` (everything) or ``run_individual_steps([...])`` (a
selected, ordered subset). Pass canonical step names — run
``symfluence workflow list-steps`` to see the authoritative list.

.. code-block:: python

   from symfluence import SYMFLUENCE

   # Initialize
   conf = SYMFLUENCE("config.yaml")

   # Run a curated subset, in order
   conf.run_individual_steps([
       # Project setup
       "setup_project",
       "create_pour_point",
       # Domain definition
       "acquire_attributes",
       "define_domain",
       "discretize_domain",
       # Data acquisition + preprocessing
       "process_observed_data",
       "acquire_forcings",
       "model_agnostic_preprocessing",
       "build_model_ready_store",
       # Model execution
       "model_specific_preprocessing",
       "run_model",
       "postprocess_results",
       # Calibration + analysis
       "calibrate_model",
       "run_benchmarking",
       "run_sensitivity_analysis",
   ])

   # Inspect status / diagnostics
   status = conf.get_workflow_status()
   issues = conf.run_diagnostics_for_step("calibrate_model")

Configuration Access
--------------------

.. code-block:: python

   from symfluence import SYMFLUENCE
   from symfluence.core.config import SymfluenceConfig

   # Load typed configuration directly
   config = SymfluenceConfig.from_file("config.yaml")

   # Access typed attributes
   print(f"Domain: {config.domain.name}")
   print(f"Model: {config.model.hydrological_model}")
   print(f"Start: {config.experiment.time_start}")

   # Initialize SYMFLUENCE with typed config
   conf = SYMFLUENCE(config)

Core API
========

SYMFLUENCE (Main Class)
-----------------------

The primary interface for all SYMFLUENCE operations.

.. autoclass:: symfluence.project.system.SYMFLUENCE
   :members:
   :undoc-members:
   :show-inheritance:

**Initialization:**

The constructor accepts a config file path (``str``/``Path``) or a
``SymfluenceConfig`` instance. To start from a dictionary, build a
``SymfluenceConfig`` first.

.. code-block:: python

   from symfluence import SYMFLUENCE

   # From YAML file path
   conf = SYMFLUENCE("path/to/config.yaml")

   # From SymfluenceConfig object
   from symfluence.core.config import SymfluenceConfig
   config = SymfluenceConfig.from_file("config.yaml")
   conf = SYMFLUENCE(config)

   # From dictionary (wrap it in SymfluenceConfig first)
   config_dict = {"DOMAIN_NAME": "test", ...}
   conf = SYMFLUENCE(SymfluenceConfig(**config_dict))

**Core Methods:**

.. code-block:: python

   # Full workflow execution
   conf.run_workflow(force_run=False)

   # Run a selected, ordered subset of steps
   conf.run_individual_steps(["setup_project", "define_domain"])

   # Workflow status
   status = conf.get_workflow_status()

   # Diagnostics for a single step (returns a list of issue strings)
   issues = conf.run_diagnostics_for_step("calibrate_model")

Manager Classes
===============

SYMFLUENCE uses a manager-based architecture where each major subsystem has a dedicated manager class.

Project Manager
---------------

Handles project initialization and structure.

.. automodule:: symfluence.project.project_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.project.project_manager import ProjectManager

   pm = ProjectManager(config, logger)

   # Setup project directory structure
   pm.setup_project()

   # Create pour point from coordinates
   pm.create_pour_point()

   # Get project information
   info = pm.get_project_info()

Domain Manager
--------------

Manages domain definition and discretization.

.. automodule:: symfluence.geospatial.domain_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.geospatial.domain_manager import DomainManager

   dm = DomainManager(config, logger)

   # Define domain boundaries
   dm.define_domain()

   # Discretize into HRUs/GRUs
   dm.discretize_domain()

   # Get domain statistics
   stats = dm.get_domain_statistics()

Data Manager
------------

Coordinates data acquisition and preprocessing.

.. automodule:: symfluence.data.data_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.data.data_manager import DataManager

   data_mgr = DataManager(config, logger)

   # Acquire geospatial attributes
   data_mgr.acquire_attributes()

   # Process observed streamflow data
   data_mgr.process_observed_data()

   # Acquire forcing data (ERA5, RDRS, etc.)
   data_mgr.acquire_forcings()

   # Run model-agnostic preprocessing
   data_mgr.run_model_agnostic_preprocessing()

Model Manager
-------------

Coordinates model preprocessing, execution, and postprocessing.

.. automodule:: symfluence.project.model_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.project.model_manager import ModelManager

   mm = ModelManager(config, logger)

   # Preprocess for all configured models
   mm.preprocess_models()

   # Run model simulations
   mm.run_models()

   # Extract and format results
   mm.postprocess_results()

   # Get available models
   models = mm.get_available_models()

Optimization Manager
--------------------

Handles calibration and optimization.

.. automodule:: symfluence.optimization.optimization_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.optimization.optimization_manager import OptimizationManager

   opt = OptimizationManager(config, logger)

   # Run full optimization workflow
   results = opt.run_optimization_workflow()

   # Or run calibration directly
   results_path = opt.calibrate_model()

   # Check optimization status
   status = opt.get_optimization_status()

   # Validate configuration
   validation = opt.validate_optimization_configuration()

   # Get available optimizers
   optimizers = opt.get_available_optimizers()

Analysis Manager
----------------

Performs model evaluation and analysis.

.. automodule:: symfluence.evaluation.analysis_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Key Methods:**

.. code-block:: python

   from symfluence.evaluation.analysis_manager import AnalysisManager

   am = AnalysisManager(config, logger)

   # Run benchmarking analysis
   am.run_benchmarking()

   # Run sensitivity analysis
   am.run_sensitivity_analysis()

   # Run decision analysis
   am.run_decision_analysis()

Workflow Orchestrator
---------------------

Manages workflow step execution and dependencies.

.. automodule:: symfluence.project.workflow_orchestrator
   :members:
   :undoc-members:
   :show-inheritance:

**Usage:**

.. code-block:: python

   from symfluence.project.workflow_orchestrator import WorkflowOrchestrator

   orchestrator = WorkflowOrchestrator(config, logger, managers)

   # Run full workflow
   orchestrator.run_workflow()

   # Run specific step
   orchestrator.run_step("calibrate_model")

   # Get workflow status
   status = orchestrator.get_workflow_status()

Registry
========

SYMFLUENCE uses a registry-based plugin system. Each kind of component
(runners, preprocessors, postprocessors, optimizers, acquisition handlers, ...)
has its own registry, exposed through the unified facade ``R``
(``from symfluence.core.registries import R``). Models register all of their
components in one place via ``model_manifest()``; one-off components use
``R.<registry>.add()`` / ``add_lazy()``.

.. note::

   The old ``ModelRegistry`` / ``OptimizerRegistry`` decorator API and the
   ``symfluence.models.registry`` module were removed before 1.0. Use
   ``model_manifest()`` and the ``R`` facade instead.

Registering a Model
-------------------

Declare all components for a model in a single ``model_manifest()`` call (this is
what each model's ``__init__.py`` does at import time):

.. code-block:: python

   from symfluence.core.registry import model_manifest

   class MyPreProcessor:
       def __init__(self, config, logger):
           self.config = config
           self.logger = logger

       def run_preprocessing(self):
           ...

   class MyRunner:
       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger

       def run_my_model(self):
           ...

   class MyPostProcessor:
       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger

       def extract_streamflow(self):
           ...

   model_manifest(
       "MY_MODEL",
       preprocessor=MyPreProcessor,
       runner=MyRunner,
       runner_method="run_my_model",
       postprocessor=MyPostProcessor,
   )

For a one-off component (without a full manifest), add it to the relevant
registry directly:

.. code-block:: python

   from symfluence.core.registries import R

   R.runners.add("MY_MODEL", MyRunner, runner_method="run_my_model")
   # or register lazily by import path
   R.preprocessors.add_lazy("MY_MODEL", "my_package.preprocessor:MyPreProcessor")

Querying the Registry
---------------------

.. code-block:: python

   from symfluence.core.registries import R

   # List registered model runners
   models = R.runners.keys()        # ['SUMMA', 'FUSE', 'GR', 'HYPE', 'NGEN', ...]

   # Get specific components
   runner_cls = R.runners["SUMMA"]
   preprocessor_cls = R.preprocessors.get("SUMMA")    # None if absent
   postprocessor_cls = R.postprocessors.get("SUMMA")

   # Check if a model runner is registered
   is_registered = "MY_MODEL" in R.runners

The same pattern applies to every component kind: ``R.optimizers``,
``R.acquisition_handlers``, ``R.observation_handlers``, ``R.metrics``,
``R.calibration_targets``, and so on. The live catalog is also available from
the CLI via ``symfluence list``.

Optimization API
================

Base Optimizer
--------------

.. automodule:: symfluence.core.calibration.optimizers.base_model_optimizer
   :members:
   :undoc-members:
   :show-inheritance:

DDS Algorithm
-------------

Dynamically Dimensioned Search (DDS) is accessed through the BaseModelOptimizer interface.
Model-specific optimizers inherit from BaseModelOptimizer and use the DDS algorithm
via the ``run_dds()`` method.

**Usage:**

.. code-block:: python

   # DDS is invoked through model-specific optimizers
   from symfluence.optimization.optimization_manager import OptimizationManager

   opt_manager = OptimizationManager(config, logger)
   results = opt_manager.calibrate_model()  # Uses algorithm from config

   # Or directly via model optimizer
   # optimizer.run_dds()  # Runs DDS optimization

Algorithm Selection
-------------------

.. code-block:: python

   # Configure algorithm in YAML
   # OPTIMIZATION_ALGORITHM: DDS  # or DE, PSO, SCE-UA, NSGA-II

   # Programmatic algorithm selection
   from symfluence.optimization.optimization_manager import OptimizationManager

   opt = OptimizationManager(config, logger)

   # Available algorithms
   algorithms = ['DDS', 'DE', 'PSO', 'SCE-UA', 'NSGA-II', 'ADAM', 'LBFGS']

Data Acquisition
================

Acquisition Service
-------------------

.. automodule:: symfluence.data.acquisition.acquisition_service
   :members:
   :undoc-members:
   :show-inheritance:

**Available Data Sources:**

.. code-block:: python

   # Forcing datasets
   forcing_sources = [
       'ERA5',        # ECMWF reanalysis
       'ERA5-Land',   # High-resolution land reanalysis
       'RDRS',        # Regional Deterministic Reforecast System
       'CARRA',       # Copernicus Arctic Regional Reanalysis
       'AORC',        # Analysis of Record for Calibration
       'CONUS404',    # CONUS 404 dataset
       'HRRR',        # High-Resolution Rapid Refresh
       'EM-Earth',    # EM-Earth reanalysis
       'NEX-GDDP',    # NASA climate projections
   ]

   # Observation datasets
   obs_sources = [
       'USGS',        # US Geological Survey streamflow
       'WSC',         # Water Survey of Canada
       'GRDC',        # Global Runoff Data Centre
       'MODIS',       # Remote sensing products
       'GRACE',       # Gravity recovery data
   ]

Acquisition Handlers
--------------------

.. code-block:: python

   from symfluence.data.acquisition import AcquisitionRegistry

   # Get available handlers
   handlers = AcquisitionRegistry.list_handlers()

   # Get specific handler
   era5_handler = AcquisitionRegistry.get_handler('ERA5')

Geospatial Operations
=====================

Domain Discretization
---------------------

.. automodule:: symfluence.geospatial.discretization.core
   :members:
   :undoc-members:
   :show-inheritance:

**Discretization Methods:**

.. code-block:: python

   # Available discretization approaches
   methods = [
       'lumped',           # Single unit
       'GRUs',             # Grouped Response Units
       'elevation',        # Elevation bands
       'radiation',        # Radiation-based
       'combined',         # Multiple criteria
   ]

Evaluation
==========

Evaluators
----------

.. automodule:: symfluence.evaluation.evaluators.base
   :members:
   :undoc-members:
   :show-inheritance:

**Available Evaluators:**

.. code-block:: python

   from symfluence.evaluation.evaluators import (
       StreamflowEvaluator,
       ETEvaluator,
       SnowEvaluator,
       SoilMoistureEvaluator,
       GroundwaterEvaluator,
       TWSEvaluator,
   )

   # Initialize evaluator
   evaluator = StreamflowEvaluator(config, project_dir, logger)

   # Evaluate simulation
   metrics = evaluator.evaluate(sim_dir)
   # Returns: {'KGE': 0.85, 'NSE': 0.82, 'RMSE': 12.5, ...}

Metrics
-------

.. code-block:: python

   # Available metrics
   metrics = [
       'KGE',      # Kling-Gupta Efficiency
       'KGEnp',    # Non-parametric KGE
       'NSE',      # Nash-Sutcliffe Efficiency
       'RMSE',     # Root Mean Square Error
       'MAE',      # Mean Absolute Error
       'PBIAS',    # Percent Bias
       'R2',       # Coefficient of Determination
   ]

Reporting
=========

Reporting Manager
-----------------

.. automodule:: symfluence.reporting.reporting_manager
   :members:
   :undoc-members:
   :show-inheritance:

**Visualization Methods:**

.. code-block:: python

   from symfluence.reporting.reporting_manager import ReportingManager

   rm = ReportingManager(config, logger)

   # Visualize the domain
   rm.visualize_domain()

   # Visualize the discretized domain
   rm.visualize_discretized_domain(discretization_method="elevation")

   # Visualize optimization progress / convergence
   rm.visualize_optimization_progress(history, output_dir, calibration_variable, metric)

   # Visualize sensitivity analysis results
   rm.visualize_sensitivity_analysis(sensitivity_data, output_file)

Configuration
=============

SymfluenceConfig
----------------

.. automodule:: symfluence.core.config
   :members:
   :undoc-members:
   :show-inheritance:

**Loading and Using Configuration:**

.. code-block:: python

   from symfluence.core.config import SymfluenceConfig, ensure_typed_config

   # Load from file
   config = SymfluenceConfig.from_file("config.yaml")

   # From dictionary
   config = SymfluenceConfig(**config_dict)

   # Ensure typed config (for mixed dict/config inputs)
   config = ensure_typed_config(maybe_dict_or_config)

   # Access configuration values
   domain = config.domain.name
   model = config.model.hydrological_model

   # Convert to dictionary
   flat_dict = config.to_dict(flatten=True)

Utilities
=========

Path Management
---------------

.. code-block:: python

   from symfluence.data.path_manager import PathManager

   pm = PathManager(config)

   # Access standard paths
   project_dir = pm.project_dir
   forcing_dir = pm.forcing_dir
   simulations_dir = pm.simulations_dir
   observations_dir = pm.observations_dir

Logging
-------

.. code-block:: python

   from symfluence.project.logging_manager import LoggingManager

   # Initialize logging
   log_mgr = LoggingManager(config)
   logger = log_mgr.get_logger("my_module")

   # Log messages
   logger.info("Processing started")
   logger.warning("Optional data not found")
   logger.error("Critical failure")

Error Handling
--------------

.. code-block:: python

   from symfluence.core.exceptions import (
       SYMFLUENCEError,           # Base exception
       ConfigurationError,        # Config issues
       DataAcquisitionError,      # Data download failures
       ModelExecutionError,       # Model run failures
       ValidationError,           # Validation failures
   )

   try:
       conf.run_workflow()
   except ConfigurationError as e:
       print(f"Configuration problem: {e}")
   except ModelExecutionError as e:
       print(f"Model failed: {e}")
   except SYMFLUENCEError as e:
       print(f"General error: {e}")

Advanced Usage
==============

Custom Workflow
---------------

.. code-block:: python

   from symfluence import SYMFLUENCE

   conf = SYMFLUENCE("config.yaml")

   # Run a subset of steps in order
   conf.run_individual_steps(["setup_project", "define_domain"])

   # Skip to model execution (assumes data already exists)
   conf.run_individual_steps([
       "model_specific_preprocessing",
       "run_model",
       "postprocess_results",
   ])

   # Access internal managers (LazyManagerDict)
   # keys: 'project', 'domain', 'data', 'model', 'analysis', 'optimization', 'reporting'
   model_mgr = conf.managers['model']
   data_mgr = conf.managers['data']

Parallel Execution
------------------

.. code-block:: python

   # Configure in YAML
   # NUM_PROCESSES: 8
   # PARALLEL_CALIBRATION: true

   # Or programmatically
   from symfluence.core.config import SymfluenceConfig

   config_dict['NUM_PROCESSES'] = 8
   config_dict['PARALLEL_CALIBRATION'] = True

   conf = SYMFLUENCE(SymfluenceConfig(**config_dict))
   conf.run_individual_steps(["calibrate_model"])  # Uses parallel execution

Batch Processing
----------------

.. code-block:: python

   from symfluence import SYMFLUENCE
   from pathlib import Path

   # Process multiple domains
   config_files = Path("configs/").glob("*.yaml")

   for config_file in config_files:
       print(f"Processing {config_file.name}")
       conf = SYMFLUENCE(str(config_file))
       conf.run_workflow()

References
==========

- :doc:`getting_started` — High-level workflow tutorial
- :doc:`configuration` — Configuration parameter reference
- :doc:`configuration` — Configuration system usage
- :doc:`developer_guide` — Extending SYMFLUENCE
- :doc:`examples` — Example workflows and use cases
