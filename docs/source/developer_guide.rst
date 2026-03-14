Developer Guide
===============

This guide provides technical documentation for developers who want to extend SYMFLUENCE, add new models, or contribute to the codebase.

---

Architecture Overview
---------------------

SYMFLUENCE follows a modular, manager-based architecture with clear separation of concerns:

**Core Components**

.. code-block:: text

   symfluence/
   ├── core/                    # Core system and configuration
   │   ├── system.py           # Main SYMFLUENCE class
   │   ├── base_manager.py     # Base class for all managers
   │   ├── config/             # Typed configuration models
   │   └── exceptions.py       # Custom exception hierarchy
   ├── project/                 # Project and workflow management
   │   ├── project_manager.py  # Project initialization
   │   └── workflow_orchestrator.py  # Step orchestration
   ├── data/                    # Data acquisition and preprocessing
   │   ├── data_manager.py     # Data operations facade
   │   ├── acquisition/        # Cloud data acquisition
   │   └── preprocessing/      # Model-agnostic preprocessing
   ├── geospatial/             # Domain definition and discretization
   │   ├── domain_manager.py   # Domain operations
   │   └── discretization/     # HRU generation
   ├── models/                  # Model integrations
   │   ├── model_manager.py    # Model execution coordination
   │   ├── registry.py         # Plugin registration system
   │   └── {model}/            # Model-specific implementations
   ├── optimization/           # Calibration and optimization
   │   ├── optimization_manager.py
   │   └── optimizers/         # Algorithm implementations
   ├── evaluation/             # Performance metrics and analysis
   │   └── analysis_manager.py
   └── reporting/              # Visualization and output
       └── reporting_manager.py

**Design Patterns**

1. **Manager Pattern**: Each major subsystem has a manager class that coordinates operations
2. **Registry Pattern**: Models self-register using decorators (see :doc:`api`)
3. **Mixin Pattern**: Common functionality shared through mixins
4. **Typed Configuration**: Pydantic models for configuration validation

---

Adding a New Hydrological Model
--------------------------------

SYMFLUENCE uses a unified registry system (``Registry[T]``) and a declarative
``model_manifest()`` helper that makes adding new models straightforward.
Models can be added **inside** the SYMFLUENCE source tree or as **external
pip-installable plugins** — the registration API is the same.

Step 1: Create Model Directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a new directory under ``src/symfluence/models/``:

.. code-block:: bash

   mkdir src/symfluence/models/mymodel
   touch src/symfluence/models/mymodel/__init__.py
   touch src/symfluence/models/mymodel/preprocessor.py
   touch src/symfluence/models/mymodel/runner.py
   touch src/symfluence/models/mymodel/postprocessor.py

Step 2: Implement Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a preprocessor, runner, and postprocessor.  Each should inherit from
the appropriate base class:

.. code-block:: python

   # src/symfluence/models/mymodel/preprocessor.py
   from symfluence.models.base import BaseModelPreProcessor

   class MyModelPreProcessor(BaseModelPreProcessor):

       def _get_model_name(self) -> str:
           return "MYMODEL"

       def __init__(self, config, logger):
           super().__init__(config, logger)
           self.model_input_dir = self.project_dir / 'forcing' / 'MYMODEL_input'
           self.model_input_dir.mkdir(parents=True, exist_ok=True)

       def run_preprocessing(self):
           self.logger.info("Starting MyModel preprocessing")
           forcing_data = self.load_forcing_data()
           model_input = self.transform_forcing(forcing_data)
           self.write_config_files()
           self.write_model_inputs(model_input)

.. code-block:: python

   # src/symfluence/models/mymodel/runner.py
   from pathlib import Path

   class MyModelRunner:

       def __init__(self, config, logger, reporting_manager=None):
           self.config = config
           self.logger = logger
           self.project_dir = Path(config.root.data_dir) / "domain" / config.domain.name

       def run_mymodel(self):
           import subprocess
           executable = self.get_executable_path()
           config_file = self.project_dir / 'settings' / 'MYMODEL' / 'config.txt'
           result = subprocess.run(
               [str(executable), str(config_file)],
               capture_output=True, text=True,
           )
           if result.returncode != 0:
               raise RuntimeError(f"MyModel failed: {result.stderr}")

Step 3: Register with ``model_manifest()``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In your model's ``__init__.py``, use the declarative ``model_manifest()`` to
register all components in a single call:

.. code-block:: python

   # src/symfluence/models/mymodel/__init__.py
   from symfluence.core.registry import model_manifest
   from .preprocessor import MyModelPreProcessor
   from .runner import MyModelRunner
   from .postprocessor import MyModelPostProcessor

   model_manifest(
       "MYMODEL",
       preprocessor=MyModelPreProcessor,
       runner=MyModelRunner,
       runner_method="run_mymodel",
       postprocessor=MyModelPostProcessor,
   )

This single call registers the preprocessor, runner (with its method name),
and postprocessor into ``R.preprocessors``, ``R.runners``, and
``R.postprocessors`` respectively.  You can also pass ``config_adapter``,
``result_extractor``, ``plotter``, ``optimizer``, and many more — see the
``model_manifest()`` signature for the full list.

Alternatively, you can register components individually:

.. code-block:: python

   from symfluence.core.registries import R

   R.runners.add("MYMODEL", MyModelRunner, runner_method="run_mymodel")

Step 4: Test Your Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # tests/unit/models/test_mymodel.py
   from symfluence.core.registries import R

   def test_mymodel_registered():
       assert "MYMODEL" in R.runners
       assert "MYMODEL" in R.preprocessors

   def test_mymodel_validation():
       result = R.validate_model("MYMODEL")
       assert result["valid"] is True

Step 5: Documentation
~~~~~~~~~~~~~~~~~~~~~~

1. Update ``docs/source/configuration.rst`` with model-specific parameters
2. Add example configuration in ``src/symfluence/resources/config_templates/``
3. Document in ``docs/source/api.rst`` if API changes

---

Extending Functionality
-----------------------

All extensions use the same ``R.*.add()`` API from the unified registry.

Adding a New Optimization Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from symfluence.core.registries import R
   from .base_model_optimizer import BaseModelOptimizer

   class MyOptimizer(BaseModelOptimizer):
       def optimize(self):
           ...

   R.optimizers.add("MYMODEL", MyOptimizer)

Adding a New Data Source
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from symfluence.core.registries import R

   R.acquisition_handlers.add("mydata", MyDataHandler)

Adding a New Observation Handler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from symfluence.core.registries import R

   R.observation_handlers.add("my_sensor", MySensorHandler)

Adding New Discretization Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create class in ``src/symfluence/geospatial/discretization/attributes/``
2. Follow existing patterns (elevation.py, radiation.py)
3. Register in discretization core
4. Add configuration parameters


.. _plugins:

External Plugins (pip-installable)
-----------------------------------

SYMFLUENCE supports **external plugins** that are discovered automatically
via Python `entry points <https://packaging.python.org/en/latest/specifications/entry-points/>`_.
This means anyone can publish a pip package that registers models, data
handlers, calibration algorithms, or any other component — and it becomes
available the moment the package is installed.  No changes to SYMFLUENCE
itself are required.

How it works
~~~~~~~~~~~~

At startup, SYMFLUENCE scans the ``symfluence.plugins`` entry-point group
using ``importlib.metadata.entry_points()``.  Each entry point is expected
to be a **zero-argument callable** (typically a function) that performs
registrations using the standard ``R.*.add()`` or ``model_manifest()`` API.

If a plugin raises an exception during loading, it is logged as a warning
and skipped — it never crashes the framework.

Writing a plugin
~~~~~~~~~~~~~~~~

1. **Create your package** with the component implementations:

.. code-block:: text

   symfluence-mymodel/
   ├── pyproject.toml
   └── symfluence_mymodel/
       ├── __init__.py
       ├── preprocessor.py
       ├── runner.py
       └── postprocessor.py

2. **Write a ``register()`` function** that wires everything into the
   registry:

.. code-block:: python

   # symfluence_mymodel/__init__.py
   def register():
       \"\"\"Called automatically by SYMFLUENCE on startup.\"\"\"
       from symfluence.core.registry import model_manifest
       from .preprocessor import MyModelPreProcessor
       from .runner import MyModelRunner
       from .postprocessor import MyModelPostProcessor

       model_manifest(
           "MYMODEL",
           preprocessor=MyModelPreProcessor,
           runner=MyModelRunner,
           runner_method="run_mymodel",
           postprocessor=MyModelPostProcessor,
       )

3. **Declare the entry point** in your ``pyproject.toml``:

.. code-block:: toml

   [project]
   name = "symfluence-mymodel"
   version = "0.1.0"
   dependencies = ["symfluence"]

   [project.entry-points."symfluence.plugins"]
   mymodel = "symfluence_mymodel:register"

4. **Install** the package:

.. code-block:: bash

   pip install symfluence-mymodel
   # or during development:
   pip install -e ./symfluence-mymodel

That's it.  The next time SYMFLUENCE is imported, your model will be
discovered and registered automatically:

.. code-block:: python

   from symfluence.core.registries import R

   R.runners["MYMODEL"]       # your runner class
   R.for_model("MYMODEL")     # all registered components
   R.validate_model("MYMODEL")  # completeness check

Plugin scope
~~~~~~~~~~~~

Plugins are not limited to models.  You can register **any** component type:

.. code-block:: python

   def register():
       from symfluence.core.registries import R
       from .handler import MyGRACEHandler
       from .metric import my_custom_metric

       # A new observation data handler
       R.observation_handlers.add("grace_v2", MyGRACEHandler)

       # A new evaluation metric
       R.metrics.add("MyKGE", my_custom_metric)

       # A new calibration objective
       R.objectives.add("MULTI_OBJ", MyMultiObjective)

Verifying a plugin
~~~~~~~~~~~~~~~~~~

After installation, verify that your components are discoverable:

.. code-block:: python

   from symfluence.core.registries import R

   # List everything in a specific registry
   print(R.runners.keys())

   # Full model validation
   print(R.validate_model("MYMODEL"))

   # Summary of all registries
   print(R.summary())

---

Testing Guidelines
------------------

SYMFLUENCE uses pytest with multiple test levels:

**Test Organization**

.. code-block:: text

   tests/
   ├── unit/                # Fast, isolated tests
   │   ├── core/
   │   ├── models/
   │   └── optimization/
   ├── integration/         # Component interaction tests
   │   ├── calibration/
   │   ├── domain/
   │   └── preprocessing/
   └── e2e/                # End-to-end workflow tests

**Running Tests**

.. code-block:: bash

   # All tests
   pytest

   # Unit tests only
   pytest tests/unit/

   # Specific module
   pytest tests/unit/models/test_summa_preprocessor.py

   # With coverage
   coverage erase
   pytest --cov=symfluence --cov-report=html

   # Specific markers
   pytest -m "not slow"
   pytest -m "requires_data"

**Test Markers**

.. code-block:: python

   @pytest.mark.slow  # Long-running tests
   @pytest.mark.requires_data  # Needs external data
   @pytest.mark.requires_binaries  # Needs model executables
   @pytest.mark.integration  # Integration test
   @pytest.mark.e2e  # End-to-end test

**Writing Good Tests**

.. code-block:: python

   import pytest
   from symfluence.models.summa.preprocessor import SummaPreProcessor

   @pytest.fixture
   def sample_config():
       \"\"\"Provide test configuration.\"\"\"
       return {
           'DOMAIN_NAME': 'test_domain',
           'FORCING_DATASET': 'ERA5',
           # ... other required parameters
       }

   def test_preprocessor_creates_output_directory(sample_config, tmp_path):
       \"\"\"Test that preprocessor creates required directories.\"\"\"
       # Arrange
       config = sample_config.copy()
       config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

       # Act
       preprocessor = SummaPreProcessor(config, logger)
       preprocessor.run_preprocessing()

       # Assert
       assert (tmp_path / 'forcing' / 'SUMMA_input').exists()

---

Code Style and Standards
-------------------------

**Python Style**

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use Black for formatting (configuration in ``pyproject.toml``)

**Docstring Format**

Use Google-style docstrings:

.. code-block:: python

   def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
       \"\"\"
       Calculate performance metrics for model evaluation.

       Args:
           observed: Array of observed values
           simulated: Array of simulated values

       Returns:
           Dictionary containing metric names and values

       Raises:
           ValueError: If arrays have different lengths

       Example:
           >>> obs = np.array([1, 2, 3])
           >>> sim = np.array([1.1, 2.1, 2.9])
           >>> metrics = calculate_metrics(obs, sim)
           >>> print(metrics['nse'])
           0.95
       \"\"\"

**Import Order**

.. code-block:: python

   # Standard library
   from pathlib import Path
   from typing import Dict, Any

   # Third-party
   import numpy as np
   import pandas as pd

   # Local imports
   from symfluence.core.base_manager import BaseManager
   from symfluence.models.registry import ModelRegistry

---

Configuration System
--------------------

For detailed configuration patterns, see :doc:`configuration`.

Key points:

- Uses Pydantic for type validation
- Immutable configuration objects
- Typed configuration models in ``src/symfluence/core/config/models.py``
- Legacy dict support for backward compatibility

---

Contribution Workflow
---------------------

See the `Contribution Guidelines <https://github.com/symfluence-org/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_ for complete information.

**Quick Start**

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make changes and add tests
4. Run tests: ``pytest``
5. Format code: ``black src/``
6. Commit with descriptive message
7. Push and create pull request to ``develop``

**Pull Request Checklist**

- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] Type hints included
- [ ] CHANGELOG.md updated
- [ ] Code formatted with Black

---

Release Process
---------------

SYMFLUENCE follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

**Creating a Release**

1. Update version in ``pyproject.toml``
2. Update CHANGELOG.md
3. Create release branch: ``git checkout -b release/v0.X.0``
4. Run full test suite
5. Merge to ``main``
6. Tag release: ``git tag -a v0.X.0 -m "Release v0.X.0"``
7. Push tags: ``git push --tags``
8. GitHub Actions handles PyPI deployment

---

Additional Resources
--------------------

**Internal Documentation**

- :doc:`api` — API reference with autodoc
- :doc:`configuration` — Configuration system usage
- TESTING.md — Comprehensive testing guide (in tests/ directory)

**External Resources**

- `SUMMA Documentation <https://summa.readthedocs.io/>`_
- `FUSE Documentation <https://naddor.github.io/fuse/>`_
- `Pydantic Documentation <https://docs.pydantic.dev/>`_
- `Pytest Documentation <https://docs.pytest.org/>`_

**Community**

- GitHub Issues: https://github.com/symfluence-org/SYMFLUENCE/issues
- GitHub Discussions: https://github.com/symfluence-org/SYMFLUENCE/discussions
- `Contributing Guide <https://github.com/symfluence-org/SYMFLUENCE/blob/main/CONTRIBUTING.md>`_

---

Getting Help
------------

**For Development Questions:**

1. Check existing documentation and examples
2. Search GitHub issues for similar questions
3. Ask in GitHub Discussions
4. Open an issue with ``[dev]`` tag

**For Bug Reports:**

Include:
- SYMFLUENCE version
- Python version
- Operating system
- Minimal reproducible example
- Full error traceback
- Steps to reproduce
