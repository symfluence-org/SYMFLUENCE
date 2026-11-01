=========================================
Architecture Guide
=========================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

SYMFLUENCE is built on a modular, manager-based architecture with clear separation of concerns. The framework follows established design patterns to enable extensibility, maintainability, and loose coupling between components.

**Core Principles:**

- **Manager Pattern**: Subsystems coordinated through dedicated manager classes
- **Registry Pattern**: Models self-register via ``model_manifest()`` for plugin extensibility
- **Mixin Pattern**: Shared functionality distributed through composable mixins
- **Typed Configuration**: Pydantic models for validation and type safety
- **Lazy Loading**: Components instantiated on-demand for efficiency

System Architecture Diagram
===========================

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                           SYMFLUENCE Class                              │
   │                    (Primary Entry Point / Facade)                       │
   └───────────────────────────────┬─────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                      WorkflowOrchestrator                               │
   │              (Step Coordination & Execution Control)                    │
   └───────────────────────────────┬─────────────────────────────────────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
           ▼                       ▼                       ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │  DataManager  │      │ ModelManager  │      │ Optimization  │
   │               │      │               │      │   Manager     │
   └───────┬───────┘      └───────┬───────┘      └───────┬───────┘
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐      ┌───────────────┐      ┌───────────────┐
   │  Acquisition  │      │  R registry   │      │  Optimizers   │
   │   Services    │      │   (Plugin)    │      │  (DE/DDS/     │
   │               │      │               │      │   ADAM/PSO)   │
   └───────────────┘      └───────────────┘      └───────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 │               │               │
                 ▼               ▼               ▼
          ┌──────────┐    ┌──────────┐    ┌──────────┐
          │Preproc   │    │ Runner   │    │Postproc  │
          │(per      │    │(per      │    │(per      │
          │ model)   │    │ model)   │    │ model)   │
          └──────────┘    └──────────┘    └──────────┘

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                     Cross-Cutting Concerns                              │
   ├─────────────────────┬─────────────────────┬─────────────────────────────┤
   │   DomainManager     │  ReportingManager   │    AnalysisManager          │
   │   (Geospatial)      │  (Visualization)    │    (Evaluation)             │
   └─────────────────────┴─────────────────────┴─────────────────────────────┘

Directory Structure
===================

.. code-block:: text

   symfluence/
   ├── core/                      # Foundation + every contract surface
   │   ├── base_manager.py       # Abstract base for all managers
   │   ├── registry.py           # Registry + model_manifest()
   │   ├── registries.py         # Unified registry facade (R)
   │   ├── contracts.py          # Per-family contract versions (ADR-0009)
   │   ├── config/               # Configuration system
   │   │   ├── models/           # SymfluenceConfig (Pydantic) package
   │   │   ├── transformers.py   # Flat ↔ nested conversion
   │   │   └── legacy_aliases.py # Framework-level aliases (+ schema-declared)
   │   ├── modeling/             # Model-adapter contract tier
   │   │   ├── base/             # Runner/pre/post/extractor bases
   │   │   ├── adapters/         # ForcingAdapter base + registry
   │   │   ├── mixins/           # Adapter mixins (subprocess, SLURM, ...)
   │   │   ├── execution/        # ModelExecutor, spatial orchestration
   │   │   ├── state/            # Model state management
   │   │   ├── templates/        # New-model scaffolding
   │   │   ├── utilities/        # Shared adapter utilities
   │   │   └── config_schema.py  # ConfigKey schema machinery
   │   ├── calibration/          # Generic calibration engine
   │   │   ├── optimizers/       # BaseModelOptimizer + algorithms/
   │   │   ├── workers/          # BaseWorker, InMemoryModelWorker
   │   │   └── parameters/       # BaseParameterManager, bounds registry
   │   ├── metrics/              # KGE, NSE, ... + metric registry
   │   ├── build/                # Build-environment snippet helpers
   │   ├── geometry_utils.py     # Catchment geometry utilities
   │   ├── exceptions.py         # Custom exception hierarchy
   │   ├── constants.py          # Unit conversions, defaults
   │   └── mixins/               # Shared functionality mixins
   │
   ├── project/                   # Project & workflow management
   │   ├── system.py             # SYMFLUENCE main class (entry point)
   │   ├── project_manager.py    # Directory structure setup
   │   ├── workflow_orchestrator.py  # Step execution engine
   │   ├── logging_manager.py    # Logging configuration
   │   └── manager_factory.py    # Lazy manager instantiation
   │
   ├── data/                      # Data acquisition & preprocessing
   │   ├── data_manager.py       # Data operations facade
   │   ├── acquisition/          # Cloud data acquisition
   │   │   ├── services/         # CDS, GEE, S3 handlers
   │   │   └── handlers/         # Dataset-specific logic
   │   ├── preprocessing/        # Model-agnostic preprocessing
   │   └── utils/                # Spatial, NetCDF, archive utils
   │
   ├── geospatial/               # Domain definition & discretization
   │   ├── domain_manager.py     # Spatial operations facade
   │   ├── geofabric/            # River network extraction
   │   └── discretization/       # HRU generation methods
   │
   ├── models/                    # Model adapters + their data (extraction-
   │   │                          # ready: nothing imports it at module level)
   │   └── {model}/              # One adapter package per model
   │       ├── preprocessor.py   #   (subclasses core.modeling bases,
   │       ├── runner.py         #    registers via model_manifest /
   │       ├── postprocessor.py  #    R.* — the same path external
   │       ├── config_schema.py  #    plugin packages use)
   │       └── base_settings/    # Template settings (package data)
   │
   ├── optimization/             # Calibration orchestration
   │   ├── optimization_manager.py
   │   ├── calibration_targets/  # Target factory (registry-first)
   │   ├── objectives/           # Objective functions
   │   └── regionalization/      # Parameter regionalization
   │
   ├── evaluation/               # Output evaluation & analysis
   │   ├── analysis_manager.py
   │   └── evaluators/           # Streamflow, snow, ET, ...
   │
   └── reporting/                # Visualization & output
       ├── reporting_manager.py
       └── plotters/             # Specialized plot generators

A few historical import paths for code promoted into ``core`` (e.g.
``symfluence.models.base``, ``symfluence.optimization.optimizers.base_model_optimizer``,
``symfluence.evaluation.metrics``) remain as back-compat shims because external
model plugins still import them. Each emits a ``DeprecationWarning`` naming its
canonical module and will be removed before 1.0; in-tree code imports the
canonical ``core`` paths only (enforced by a guard test). See
``docs/adr/0011-retire-unused-back-compat-shims.md`` for the list.

Contract families
=================

External packages build against four independently versioned contract
families declared in ``symfluence.core.contracts`` (ADR-0009): ``models``
(the adapter contract: ``model_manifest()`` + the registry namespaces),
``calibration`` (the engine bases and seams), ``metrics``, and
``geospatial-utils``. Versioning follows the acquisition-backend contract's
semantics: same major required; pre-1.0 minors are additive-only, and a
package targeting a newer minor than the framework provides is declined at
registration (``assert_compatible``).

Core Design Patterns
====================

Manager Pattern
---------------

Each major subsystem has a dedicated manager class that:

- Coordinates multiple services/components
- Provides a high-level API to the orchestrator
- Inherits from ``BaseManager`` for consistent behavior
- Lazy-loads dependencies for efficiency

**BaseManager provides:**

.. code-block:: python

   class BaseManager(ConfigurableMixin, ABC):
       """Abstract base for all SYMFLUENCE managers."""

       def __init__(self, config, logger, reporting_manager=None):
           # Auto-convert dict to typed config
           self._config = SymfluenceConfig(**config) if isinstance(config, dict) else config
           self.logger = logger
           self.reporting_manager = reporting_manager
           self._initialize_services()  # Subclass hook

       def _execute_workflow(self, items, handler, operation_name):
           """Standardized batch processing with error handling."""

       def _safe_visualize(self, viz_func, *args, **kwargs):
           """Safe visualization with error handling."""

**Key Managers:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Manager
     - Responsibility
   * - ``DataManager``
     - Data acquisition, merging, preprocessing
   * - ``DomainManager``
     - Catchment delineation, HRU discretization
   * - ``ModelManager``
     - Model preprocessing, execution, postprocessing
   * - ``OptimizationManager``
     - Parameter calibration, algorithm orchestration
   * - ``AnalysisManager``
     - Performance evaluation, metrics calculation
   * - ``ReportingManager``
     - Visualization, plot generation

Registry Pattern (Plugin System)
--------------------------------

SYMFLUENCE uses a unified ``Registry[T]`` generic class as the single
source of truth for all component registrations.  All registries are
accessible through the ``Registries`` facade (aliased as ``R``):

.. code-block:: python

   from symfluence.core.registries import R

**Declarative model registration (``model_manifest``):**

.. code-block:: python

   # In a model's __init__.py
   from symfluence.core.registry import model_manifest

   model_manifest(
       "MYMODEL",
       preprocessor=MyModelPreprocessor,
       runner=MyModelRunner,
       runner_method="run_mymodel",
       postprocessor=MyModelPostProcessor,
       config_adapter=MyModelConfigAdapter,
   )

**Direct registration for individual components:**

.. code-block:: python

   R.observation_handlers.add("grace", GraceHandler)
   R.objectives.add("NSE", NseObjective)
   R.metrics.add("MyMetric", my_metric_fn)

**Component discovery:**

.. code-block:: python

   runner_cls = R.runners["SUMMA"]
   meta = R.runners.meta("SUMMA")        # e.g. {"runner_method": "run_summa"}
   everything = R.for_model("SUMMA")     # all registries for one model
   R.validate_model("SUMMA")             # completeness check

**Registry features:**

- UPPERCASE key normalization by default (lowercase for data registries)
- Lazy imports via ``add_lazy()`` — class resolved on first access
- Aliases via ``alias()`` — e.g. ``"SAC-SMA"`` → ``"SACSMA"``
- Advisory protocol validation — warns on interface mismatch
- Freeze/clear lifecycle for post-bootstrap safety
- Decorator support: ``@R.runners.add("SUMMA")``

**External plugin discovery via pip:**

External packages can register components by declaring a
``symfluence.plugins`` entry point.  SYMFLUENCE discovers these
automatically at startup via ``importlib.metadata``:

.. code-block:: toml

   # In an external package's pyproject.toml
   [project.entry-points."symfluence.plugins"]
   my_model = "my_package:register"

See :ref:`plugins` in the Developer Guide for full details.

**Benefits:**

- **Uniform API**: Every component type uses the same ``R.*.add()`` / ``R.*["KEY"]`` interface
- **Loose coupling**: Framework discovers components; doesn't import them directly
- **pip-installable plugins**: ``pip install symfluence-mymodel`` and it's registered
- **Cross-domain queries**: ``R.for_model()``, ``R.registered_models()``, ``R.summary()``
- **Testing**: Mock components via ``R.*.add()`` / ``R.*.remove()`` / ``R.*.clear()``

Mixin Pattern
-------------

Shared functionality is distributed through composable mixins:

**ConfigurableMixin:**

.. code-block:: python

   class ConfigurableMixin:
       """Provides config access and common properties."""

       @property
       def config(self) -> SymfluenceConfig:
           return self._config

       @property
       def project_dir(self) -> Path:
           return Path(self.config.root.data_dir) / "domain" / self.config.domain.name

       @property
       def experiment_id(self) -> str:
           return self.config.domain.experiment_id

       def _get_config_value(self, accessor, default):
           """Safe config access with fallback."""
           try:
               value = accessor()
               return value if value is not None else default
           except (AttributeError, KeyError):
               return default

**MizuRouteConfigMixin:**

.. code-block:: python

   class MizuRouteConfigMixin:
       """Adds mizuRoute routing capabilities to runners."""

       @property
       def mizu_settings_path(self):
           return self._get_config_value(
               lambda: self.config.path.mizu_settings,
               self.project_dir / 'settings' / 'mizuroute'
           )

       def _run_mizuroute(self, spatial_config, model_name):
           """Execute mizuRoute routing."""
           # Shared routing logic
           pass

**Usage:**

.. code-block:: python

   class HBVRunner(BaseModelRunner, UnifiedModelExecutor, MizuRouteConfigMixin):
       """HBV runner with routing support via mixin."""
       pass

Typed Configuration (Pydantic)
------------------------------

Configuration uses hierarchical Pydantic models for validation:

**Structure:**

.. code-block:: python

   class SymfluenceConfig(BaseModel):
       """Root configuration model."""
       root: RootConfig
       domain: DomainConfig
       forcing: ForcingConfig
       model: ModelConfig
       optimization: OptimizationConfig
       paths: PathConfig
       logging: LoggingConfig

   class DomainConfig(BaseModel):
       name: str
       experiment_id: str = "default"
       definition_method: Literal['polygon', 'delineate', 'merit_basins']
       discretization: str = "GRUs"

   class ModelConfig(BaseModel):
       hydrological_model: str
       summa: Optional[SummaConfig] = None
       fuse: Optional[FuseConfig] = None
       hbv: Optional[HBVConfig] = None
       # ... other models

**Loading & Validation:**

.. code-block:: python

   # From YAML file
   config = SymfluenceConfig.from_file("config.yaml")

   # With CLI overrides
   config = SymfluenceConfig.from_file(
       "config.yaml",
       overrides={'FORCING_DATASET': 'ERA5'},
       use_env=True,  # Allow environment variables
       validate=True  # Run validation rules
   )

   # Convert to flat dict (backward compatibility)
   flat_config = config.to_dict(flatten=True)

**Validation Rules:**

.. code-block:: python

   @field_validator('forcing_dataset')
   @classmethod
   def validate_dataset(cls, v):
       valid = ['ERA5', 'RDRS', 'CARRA', 'CERRA', 'Daymet']
       if v not in valid:
           raise ValueError(f"Invalid dataset: {v}")
       return v

Data Flow
=========

Complete Workflow
-----------------

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    1. PROJECT INITIALIZATION                            │
   │                                                                         │
   │  ProjectManager.setup()                                                 │
   │  ├── Create directory structure                                        │
   │  ├── Validate configuration                                            │
   │  └── Initialize logging                                                 │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    2. DOMAIN DEFINITION                                 │
   │                                                                         │
   │  DomainManager.define_domain()                                          │
   │  ├── Polygon: Load shapefile → extract pour point                       │
   │  ├── Delineate: DEM + pour point → watershed delineation                │
   │  └── MERIT: Basin ID → pre-computed catchment                           │
   │                                                                         │
   │  DomainManager.discretize()                                             │
   │  ├── Elevation bands: Split by elevation                                │
   │  ├── Radiation: Split by aspect/slope                                   │
   │  └── Combined: Multi-attribute discretization                           │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    3. DATA ACQUISITION                                  │
   │                                                                         │
   │  DataManager.acquire_forcing()                                          │
   │  ├── ERA5: CDS API → download → subset to domain                        │
   │  ├── Observations: GRDC/USGS → align to time window                     │
   │  └── Attributes: Shapefiles → compute basin properties                  │
   │                                                                         │
   │  DataManager.merge_forcing()                                            │
   │  ├── Temporal alignment across datasets                                 │
   │  ├── Spatial interpolation to HRUs                                      │
   │  └── Unit conversion and standardization                                │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    4. MODEL PREPROCESSING                               │
   │                                                                         │
   │  ModelManager.preprocess(model_name)                                    │
   │  ├── Registry lookup: R.preprocessors[model_name]                       │
   │  ├── Instantiate: preprocessor = PreprocessorClass(config, logger)      │
   │  └── Execute: preprocessor.run_preprocessing()                          │
   │                                                                         │
   │  Model-specific operations:                                             │
   │  ├── SUMMA: Generate forcingFileList.txt, attributes.nc                 │
   │  ├── HBV: Create forcing CSV/NetCDF with PET calculation                │
   │  └── GR: Create R-compatible forcing files                              │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    5. MODEL EXECUTION                                   │
   │                                                                         │
   │  ModelManager.run_models()                                              │
   │  ├── Registry lookup: R.runners[model_name]                             │
   │  ├── Get method: R.runners.meta(model_name)["runner_method"]            │
   │  ├── Instantiate: runner = RunnerClass(config, logger)                  │
   │  └── Execute: getattr(runner, method_name)()                            │
   │                                                                         │
   │  Execution patterns:                                                    │
   │  ├── Process-based: SUMMA, FUSE (subprocess call)                       │
   │  ├── In-memory: HBV, LSTM (Python/JAX execution)                        │
   │  └── R interface: GR (rpy2 bridge)                                      │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    6. POSTPROCESSING & EVALUATION                       │
   │                                                                         │
   │  ModelManager.postprocess(model_name)                                   │
   │  ├── Registry lookup: R.postprocessors[model_name]                      │
   │  ├── Extract streamflow from model outputs                              │
   │  └── Standardize to common format (CSV, NetCDF)                         │
   │                                                                         │
   │  AnalysisManager.evaluate()                                             │
   │  ├── Load simulations and observations                                  │
   │  ├── Calculate metrics (KGE, NSE, RMSE, etc.)                           │
   │  └── Generate evaluation report                                         │
   └───────────────────────────────────┬─────────────────────────────────────┘
                                       ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │                    7. CALIBRATION (Optional)                            │
   │                                                                         │
   │  OptimizationManager.calibrate_model()                                  │
   │  ├── Initialize optimizer (DE, DDS, ADAM, PSO, NSGA-II)                 │
   │  ├── Create parameter manager for model                                 │
   │  ├── Spawn workers for parallel evaluation                              │
   │  └── Iterate until convergence                                          │
   │                                                                         │
   │  Per iteration:                                                         │
   │  ├── Optimizer proposes parameters                                      │
   │  ├── Worker applies parameters to model                                 │
   │  ├── Worker runs model                                                  │
   │  ├── Worker extracts metrics                                            │
   │  └── Optimizer updates based on fitness                                 │
   └─────────────────────────────────────────────────────────────────────────┘

Model Component Lifecycle
-------------------------

.. code-block:: text

   1. Registration (import time or plugin discovery):
      ┌─────────────────────────────────────────────────────┐
      │  model_manifest("SUMMA",                            │
      │      runner=SUMMARunner, runner_method="run_summa")  │
      │                                                      │
      │  → R.runners["SUMMA"] = SUMMARunner                 │
      │  → R.runners.meta("SUMMA") = {runner_method: ...}   │
      └─────────────────────────────────────────────────────┘

   2. Discovery (workflow time):
      ┌─────────────────────────────────────────────────────┐
      │  runner_cls = R.runners["SUMMA"]                    │
      │  method = R.runners.meta("SUMMA")["runner_method"]  │
      │                                                      │
      │  → runner_cls = SUMMARunner                         │
      │  → method = "run_summa"                             │
      └─────────────────────────────────────────────────────┘

   3. Instantiation (per-execution):
      ┌─────────────────────────────────────────────────────┐
      │  runner = runner_cls(config, logger, reporting_mgr) │
      │                                                      │
      │  → runner.project_dir = /data/domain/my_basin       │
      │  → runner.experiment_id = "calibration_001"         │
      └─────────────────────────────────────────────────────┘

   4. Execution:
      ┌─────────────────────────────────────────────────────┐
      │  result = getattr(runner, method)()                 │
      │                                                      │
      │  → runner.run_summa()                               │
      │  → subprocess.run(["summa.exe", config_file])       │
      │  → return output_directory                          │
      └─────────────────────────────────────────────────────┘

Extending SYMFLUENCE
====================

All extensions use the unified registry.  See :doc:`developer_guide` for
full walkthroughs and :ref:`plugins` for external pip-installable plugins.

Adding a New Model
------------------

.. code-block:: python

   # src/symfluence/models/mymodel/__init__.py
   from symfluence.core.registry import model_manifest
   from .preprocessor import MyModelPreprocessor
   from .runner import MyModelRunner
   from .postprocessor import MyModelPostProcessor

   model_manifest(
       "MYMODEL",
       preprocessor=MyModelPreprocessor,
       runner=MyModelRunner,
       runner_method="run_mymodel",
       postprocessor=MyModelPostProcessor,
   )

Adding a New Optimization Algorithm
-----------------------------------

.. code-block:: python

   from symfluence.core.registries import R

   R.optimizers.add("MYMODEL", MyOptimizer)

Adding a New Data Handler
-------------------------

.. code-block:: python

   from symfluence.core.registries import R

   R.acquisition_handlers.add("mydata", MyDataHandler)
   R.observation_handlers.add("my_sensor", MySensorHandler)

Configuration System Details
============================

Hierarchical Structure
----------------------

.. code-block:: yaml

   # config.yaml
   DOMAIN_NAME: my_basin
   EXPERIMENT_ID: calibration_001

   # Nested under 'forcing'
   FORCING_DATASET: ERA5
   FORCING_START_YEAR: 2010
   FORCING_END_YEAR: 2020

   # Nested under 'model.summa'
   SUMMA_SPATIAL_MODE: distributed
   SUMMA_ROUTING_INTEGRATION: mizuRoute

   # Nested under 'optimization'
   OPTIMIZATION_ALGORITHM: DE
   OPTIMIZATION_MAX_ITERATIONS: 5000

**Transformation to typed config:**

.. code-block:: python

   # Flat dict (legacy format)
   flat = {'DOMAIN_NAME': 'my_basin', 'FORCING_DATASET': 'ERA5', ...}

   # Converted to hierarchical (flat keys are accepted by the constructor)
   config = SymfluenceConfig(**flat)
   config.domain.name  # 'my_basin'
   config.forcing.dataset  # 'ERA5'
   config.model.summa.spatial_mode  # 'distributed'

Environment Variable Support
----------------------------

.. code-block:: bash

   export SYMFLUENCE_DATA_DIR=/data/symfluence
   export ERA5_CDS_API_KEY=xxxxx
   export SUMMA_EXE=/opt/summa/bin/summa.exe

.. code-block:: python

   config = SymfluenceConfig.from_file("config.yaml", use_env=True)
   # Environment variables override YAML values

Error Handling
==============

Exception Hierarchy
-------------------

.. code-block:: text

   SYMFLUENCEError                    # Base exception
   ├── ConfigurationError             # Invalid configuration
   │   └── ConfigValidationError      # Pydantic/config validation failure
   ├── DataAcquisitionError           # Failed to acquire/process data
   ├── ModelExecutionError            # Model runtime failure
   ├── GeospatialError                # Geospatial operation failure
   │   ├── DiscretizationError        # HRU/GRU discretization failure
   │   ├── ShapefileError             # Shapefile read/write failure
   │   └── RasterProcessingError      # Raster processing failure
   ├── OptimizationError              # Calibration / optimization issues
   │   ├── WorkerExecutionError       # Worker subprocess failure
   │   └── RetryExhaustedError        # Retries exhausted
   ├── ValidationError                # Input validation failure
   ├── FileOperationError             # File I/O failure
   ├── CodeAnalysisError              # Static/code analysis failure
   ├── EvaluationError                # Metric/evaluation failure
   └── ReportingError                 # Reporting/visualization failure

Context Manager Pattern
-----------------------

.. code-block:: python

   from symfluence.core.exceptions import symfluence_error_handler

   with symfluence_error_handler(
       "Model execution",
       self.logger,
       error_type=ModelExecutionError
   ):
       # Protected code block
       result = subprocess.run(cmd)
       if result.returncode != 0:
           raise RuntimeError(f"Model failed: {result.stderr}")

Testing Architecture
====================

Test Organization
-----------------

.. code-block:: text

   tests/
   ├── unit/                    # Fast, isolated tests
   │   ├── core/               # Config, exceptions, utils
   │   ├── models/             # Model components
   │   │   ├── test_hbv_model.py
   │   │   └── test_summa_preprocessor.py
   │   └── optimization/       # Optimizers, workers
   ├── integration/            # Component interaction tests
   │   ├── calibration/        # End-to-end calibration
   │   ├── domain/             # Domain definition
   │   └── preprocessing/      # Data pipeline
   └── e2e/                    # Full workflow tests

Test Fixtures
-------------

.. code-block:: python

   @pytest.fixture
   def mock_config():
       """Provide test configuration."""
       return SymfluenceConfig(
           root=RootConfig(data_dir='/tmp/test'),
           domain=DomainConfig(name='test_basin'),
           ...
       )

   @pytest.fixture
   def sample_forcing():
       """Provide sample forcing data."""
       return xr.Dataset({
           'pr': (['time'], np.random.rand(365)),
           'temp': (['time'], np.random.rand(365) * 20),
       })

Additional Resources
====================

**Internal Documentation:**

- :doc:`developer_guide`: Adding new models, testing
- :doc:`configuration`: Full configuration reference
- :doc:`api`: API reference

**Design Pattern References:**

- Registry Pattern: Gang of Four
- Factory Pattern: Creational design patterns
- Mixin Pattern: Python composition patterns
- Pydantic: https://docs.pydantic.dev/

**Contributing:**

- `Contribution Guidelines <https://github.com/symfluence-org/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_
- GitHub: https://github.com/symfluence-org/SYMFLUENCE
