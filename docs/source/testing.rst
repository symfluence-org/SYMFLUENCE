.. _testing:

=========================================
Testing Guide for Contributors
=========================================

This guide covers SYMFLUENCE's testing infrastructure, including how to write,
run, and debug tests. The project uses a three-tier testing strategy with
comprehensive marker-based organization.

.. contents:: Table of Contents
   :local:
   :depth: 2

Test Organization
=================

Directory Structure
-------------------

Tests are organized into three tiers:

.. code-block:: text

   tests/
   ├── unit/               # Fast, isolated tests (~5 sec each)
   │   ├── agent/
   │   ├── cli/
   │   ├── config/
   │   ├── data/
   │   ├── evaluation/
   │   ├── geospatial/
   │   ├── models/         # 24 model-specific subdirectories
   │   ├── optimization/
   │   ├── preprocessing/
   │   └── reporting/
   │
   ├── integration/        # Module interaction tests (30s - 5 min)
   │   ├── calibration/
   │   ├── cli/
   │   ├── data/
   │   ├── domain/
   │   ├── models/
   │   └── preprocessing/
   │
   ├── e2e/                # Full workflow tests (30+ min)
   │   └── test_install_validate.py
   │
   ├── fixtures/           # Shared test fixtures
   ├── test_helpers/       # Test utilities
   ├── configs/            # Test configuration files
   └── data/               # Real test data (~5 MB)

Test Tiers
----------

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 40

   * - Tier
     - Duration
     - Purpose
     - Example
   * - Unit
     - < 5 sec
     - Isolated function tests
     - Testing a single utility function
   * - Integration
     - 30s - 5 min
     - Module interactions
     - Testing preprocessor with real data
   * - E2E
     - 30+ min
     - Complete workflows
     - Full calibration workflow

Running Tests
=============

Basic Commands
--------------

.. code-block:: bash

   # Run all unit tests (recommended for development)
   pytest -v -m "unit"

   # Run quick tests (like CI quick mode)
   pytest -v -m "ci_quick"

   # Run integration tests
   pytest -v -m "integration"

   # Run smoke tests (minimal validation)
   pytest -v -m "smoke"

Running Specific Tests
----------------------

.. code-block:: bash

   # Run tests for a specific component
   pytest -v -m "models and summa"
   pytest -v -m "calibration"
   pytest -v -m "domain"

   # Run a specific test file
   pytest -v tests/unit/config/test_config_loading.py

   # Run a specific test function
   pytest -v tests/unit/config/test_config_loading.py::test_load_config

   # Run tests matching a pattern
   pytest -v -k "test_summa"

Test Flags
----------

.. code-block:: bash

   # Run with coverage reporting
   coverage erase
   pytest -v --cov=src/symfluence --cov-report=html

   # Run tests in parallel (faster)
   pytest -v -n auto           # Auto-detect CPU count
   pytest -v -n 4              # Use 4 processes

   # Show print statements
   pytest -v -s tests/path/to/test.py

   # Full traceback on failures
   pytest -v --tb=long tests/path/to/test.py

   # Run with debugger on failure
   pytest -v --pdb tests/path/to/test.py

Custom CLI Options
------------------

.. code-block:: bash

   # Include full test matrix (multi-year workflows)
   pytest -v --run-full

   # Include cloud API tests (requires credentials)
   pytest -v --run-cloud

   # Include multi-year optimization examples
   pytest -v --run-full-examples

   # Clear cached data before running
   pytest -v --clear-cache

   # Run tests requiring external data
   pytest -v --run-data -m "requires_data"

Test Markers
============

SYMFLUENCE uses pytest markers to organize tests. Always use appropriate markers
when writing tests.

Test Type Markers
-----------------

.. code-block:: python

   @pytest.mark.unit          # Fast, isolated tests
   @pytest.mark.integration   # Module interaction tests
   @pytest.mark.e2e           # End-to-end workflow tests

Speed Markers
-------------

.. code-block:: python

   @pytest.mark.quick         # Tests under 5 seconds
   @pytest.mark.slow          # Tests over 30 seconds

Requirement Markers
-------------------

.. code-block:: python

   @pytest.mark.requires_data      # Needs external data bundles
   @pytest.mark.requires_cloud     # Requires cloud API credentials
   @pytest.mark.requires_binaries  # Requires external binaries (SUMMA, etc.)
   @pytest.mark.requires_acquisition  # Requires data acquisition

Component Markers
-----------------

.. code-block:: python

   @pytest.mark.domain        # Domain workflow tests
   @pytest.mark.data          # Data acquisition/processing
   @pytest.mark.models        # Model execution tests
   @pytest.mark.calibration   # Calibration/optimization tests
   @pytest.mark.cli           # CLI component tests

Model-Specific Markers
----------------------

.. code-block:: python

   @pytest.mark.summa         # SUMMA tests
   @pytest.mark.fuse          # FUSE tests
   @pytest.mark.ngen          # NGEN tests
   @pytest.mark.gr            # GR model tests
   @pytest.mark.hype          # HYPE tests
   @pytest.mark.mesh          # MESH tests
   @pytest.mark.lstm          # LSTM tests
   @pytest.mark.hbv           # HBV tests

CI Markers
----------

.. code-block:: python

   @pytest.mark.smoke         # Minimal smoke tests (~5 min)
   @pytest.mark.ci_quick      # Quick CI validation (~20 min)
   @pytest.mark.ci_full       # Full CI validation (~90 min)
   @pytest.mark.full          # Full test matrix (requires --run-full)

Writing Tests
=============

Unit Test Example
-----------------

.. code-block:: python

   import pytest
   from symfluence.utils import some_function

   pytestmark = [pytest.mark.unit, pytest.mark.quick]


   def test_function_returns_expected_value():
       """Test that function returns expected output for valid input."""
       result = some_function(input_data=42)
       assert result == expected_output


   def test_function_raises_on_invalid_input():
       """Test that function raises ValueError for invalid input."""
       with pytest.raises(ValueError, match="must be positive"):
           some_function(input_data=-1)

Integration Test Example
------------------------

.. code-block:: python

   import pytest
   from symfluence.models.summa import SummaPreProcessor

   pytestmark = [
       pytest.mark.integration,
       pytest.mark.models,
       pytest.mark.summa,
       pytest.mark.requires_data,
       pytest.mark.slow,
   ]


   def test_summa_preprocessing_creates_output(bow_test_data, tmp_path):
       """Test SUMMA preprocessing creates expected output files."""
       config = bow_test_data['config']
       config['EXPERIMENT_OUTPUT_SUMMA'] = str(tmp_path)

       preprocessor = SummaPreProcessor(config, logger=None)
       preprocessor.run_preprocessing()

       assert (tmp_path / 'forcing').exists()
       assert (tmp_path / 'attributes.nc').exists()

E2E Test Example
----------------

.. code-block:: python

   import pytest
   from symfluence import SYMFLUENCE

   pytestmark = [
       pytest.mark.e2e,
       pytest.mark.requires_binaries,
       pytest.mark.ci_full,
   ]


   def test_complete_workflow(tmp_path, bow_domain):
       """Test complete workflow from setup to results."""
       config = bow_domain['config']
       config['SYMFLUENCE_DATA_DIR'] = str(tmp_path)

       sf = SYMFLUENCE(config)

       # Run workflow steps
       sf.setup_project()
       sf.preprocess()
       sf.run_model()

       # Verify outputs
       assert (tmp_path / 'simulations').exists()

Using Fixtures
==============

Available Fixtures
------------------

**Session Fixtures (conftest.py):**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fixture
     - Description
   * - ``symfluence_code_dir``
     - Path to SYMFLUENCE source code
   * - ``tests_dir``
     - Path to tests directory
   * - ``config_template``
     - Loaded configuration template

**Data Fixtures (fixtures/data_fixtures.py):**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fixture
     - Description
   * - ``bow_domain``
     - Bow at Banff domain configuration
   * - ``iceland_domain``
     - Iceland regional domain
   * - ``paradise_domain``
     - Paradise point-scale domain
   * - ``ellioaar_domain``
     - Elliðaár Iceland (CARRA)
   * - ``fyris_domain``
     - Fyris Uppsala (CERRA)

**Real Data Fixtures (fixtures/real_data_fixtures.py):**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Fixture
     - Description
   * - ``real_forcing_nc``
     - Real ERA5 NetCDF file
   * - ``real_dem_tif``
     - Real DEM GeoTIFF
   * - ``real_landclass_tif``
     - Real land class GeoTIFF
   * - ``real_soilclass_tif``
     - Real soil class GeoTIFF
   * - ``real_streamflow_csv``
     - Real streamflow observations

**Mock Fixtures (unit/conftest.py):**

.. code-block:: python

   @pytest.fixture
   def mock_config():
       """Create a basic mock configuration for unit tests."""
       return {
           'SYMFLUENCE_DATA_DIR': '/tmp/test',
           'DOMAIN_NAME': 'test_domain',
           'EXPERIMENT_ID': 'test_exp'
       }

   @pytest.fixture
   def mock_logger():
       """Create a mock logger for unit tests."""
       return MagicMock()

Using Fixtures in Tests
-----------------------

.. code-block:: python

   def test_with_real_data(real_forcing_nc):
       """Test using real ERA5 forcing data."""
       import xarray as xr
       ds = xr.open_dataset(real_forcing_nc)
       assert 'airtemp' in ds.variables


   def test_with_domain(bow_domain, tmp_path):
       """Test using Bow domain configuration."""
       config = bow_domain['config']
       config['EXPERIMENT_OUTPUT'] = str(tmp_path)
       # ... run test


   def test_isolated(mock_config, mock_logger):
       """Test with mocked dependencies."""
       processor = Processor(mock_config, mock_logger)
       result = processor.process()
       mock_logger.info.assert_called()

Test Helpers
============

Location: ``tests/test_helpers/``

Configuration Helpers
---------------------

.. code-block:: python

   from test_helpers.helpers import (
       load_config_template,
       write_config,
       has_cds_credentials,
   )

   # Load test configuration template
   config = load_config_template()

   # Write configuration to file
   write_config(config, path='/tmp/config.yaml')

   # Check for cloud credentials
   if has_cds_credentials():
       # Run cloud tests
       pass

Assertion Helpers
-----------------

.. code-block:: python

   from test_helpers.assertions import (
       assert_netcdf_has_variables,
       assert_netcdf_dimensions,
       assert_simulation_outputs_exist,
   )

   # Verify NetCDF structure
   assert_netcdf_has_variables(path, ['airtemp', 'pptrate'])
   assert_netcdf_dimensions(path, {'time': 24, 'hru': 10})

   # Verify simulation outputs
   assert_simulation_outputs_exist(output_dir, model='SUMMA')

CI/CD Integration
=================

CI Workflows
------------

The project has several GitHub Actions workflows:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Workflow
     - Duration
     - Trigger
   * - ci.yml (lint, unit tests, contracts)
     - ~10 min
     - Every PR and push to main/develop (stale PR runs are cancelled)
   * - cross-platform.yml
     - ~45 min
     - Push to main/develop, nightly, on demand
   * - install-validate.yml (x86_64 source build + e2e)
     - 20-90 min
     - Push to main/develop, weekly, on demand
   * - install-methods.yml / install-validate-arm.yml
     - 20-120 min
     - Nightly, on demand, and pushes to main/develop that touch
       install-relevant paths (dependency manifests, ``scripts/``, ``npm/``,
       build instructions/patches, the CLI install services)
   * - docs.yml
     - ~5 min
     - PRs and pushes that touch ``docs/``, ``src/`` or dependency files

Test Modes in CI
----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Mode
     - Duration
     - Coverage
   * - Smoke
     - ~5 min
     - Binary validation, imports, 3-hour workflow
   * - Quick
     - ~20 min
     - Unit tests, basic integration
   * - Full
     - ~90 min
     - All tests including 1-month workflows

Running Tests Like CI
---------------------

.. code-block:: bash

   # Smoke mode
   pytest -v -m "smoke"

   # Quick mode (develop branch standard)
   pytest -v -m "unit"

   # Full mode (main branch, weekly)
   pytest -v --run-full -m "not full_examples"

Debugging Failed Tests
======================

Common Debug Commands
---------------------

.. code-block:: bash

   # Short traceback (default)
   pytest -v --tb=short tests/path/to/test.py

   # Full traceback
   pytest -v --tb=long tests/path/to/test.py

   # Show print statements
   pytest -v -s tests/path/to/test.py::test_name

   # Drop into debugger on failure
   pytest -v --pdb tests/path/to/test.py::test_name

   # Run only last failed tests
   pytest -v --lf

   # Run failed tests first
   pytest -v --ff

Common Issues
-------------

**HDF5/netCDF4 Segmentation Faults**

Already handled in conftest.py:

.. code-block:: python

   os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
   import tqdm
   tqdm.tqdm.monitor_interval = 0

**Missing Test Data**

Use ``pytest.skip()`` gracefully:

.. code-block:: python

   def test_requires_data(bow_test_data):
       if bow_test_data is None:
           pytest.skip("Test data not available")
       # ... run test

**Tests Pass Locally but Fail in CI**

- Check for missing binaries in CI
- Verify environment differences
- Check floating-point precision issues
- Download CI artifacts for debugging

**Slow Test Execution**

Use parallel execution:

.. code-block:: bash

   pytest -v -n auto  # Auto-detect CPUs

Best Practices
==============

Writing Tests
-------------

1. **Use appropriate markers** - Apply all relevant markers from pytest.ini
2. **Use fixtures** - Share setup code via conftest.py
3. **Test one thing** - Each test validates one specific behavior
4. **Use descriptive names** - ``test_<component>_<behavior>_<condition>``
5. **Document purpose** - Include docstrings explaining intent

Test Data
---------

1. **Prefer real data** - Use files from ``tests/data/`` for I/O tests
2. **Mock external APIs** - Use mocks only for cloud services
3. **Use tmp_path** - For generated files, use pytest's ``tmp_path`` fixture
4. **Keep data small** - Test data should be minimal but representative

Code Quality
------------

1. **Clean up resources** - Use fixtures and cleanup hooks
2. **Handle platform differences** - Account for OS-specific behavior
3. **Skip gracefully** - Use ``pytest.skip()`` for unavailable dependencies
4. **Avoid flaky tests** - Tests should be deterministic

Quick Reference
===============

**Minimal Unit Test:**

.. code-block:: python

   import pytest

   pytestmark = [pytest.mark.unit, pytest.mark.quick]

   def test_behavior():
       assert function(input) == expected

**Run Unit Tests:**

.. code-block:: bash

   pytest -v -m "unit"

**Run with Coverage:**

.. code-block:: bash

   pytest -v --cov=src/symfluence --cov-report=html -m "unit"

**Debug Failing Test:**

.. code-block:: bash

   pytest -v --tb=long -s tests/path/to/test.py::test_name

Contributing via the Agent
==========================

SYMFLUENCE does not ship its own AI agent. Instead, ``symfluence agent launch``
hands off to an installed coding-agent CLI (Claude Code, Codex, Gemini, ...),
primed with the packaged SYMFLUENCE *skills*. That agent brings its own editing,
search, test-running, and git tooling — SYMFLUENCE just makes sure it knows how to
work with this codebase. See :doc:`agent_guide` for the full walkthrough.

Launching the Agent
-------------------

.. code-block:: bash

   # Interactive session (run from your project / repo directory)
   symfluence agent launch

   # One-shot prompt
   symfluence agent launch "Help me fix the bug in the config loader"

Install one supported CLI and set the matching API key (e.g. ``ANTHROPIC_API_KEY``
for Claude Code); SYMFLUENCE detects the CLI on your ``PATH`` and exposes the skills
to it. Override detection with ``SYMFLUENCE_AGENT_CLI`` and skip skill
materialization with ``SYMFLUENCE_NO_SKILLS``.

Skills for Contributors
-----------------------

The packaged skills cover both running and extending SYMFLUENCE — for example,
``add-data-handler``, ``add-model-handler``, ``add-optimizer``, and
``debug-calibration``. The agent consults the relevant skill when you ask it to
work on a task, so it follows the framework's conventions (registration via
``model_manifest()``, license headers, test layout, etc.).

Best Practices with the Agent
-----------------------------

1. **Be Specific** - Describe the exact change you want
2. **Review Diffs** - Always review the agent's changes before committing
3. **Run Tests** - Ask the agent to run the relevant test markers after changes
4. **Iterate** - Refine changes through conversation
5. **Human Review** - All changes require human review before merging

See Also
========

- :doc:`developer_guide` - Developer documentation
- :doc:`architecture` - System architecture
- :doc:`cli_reference` - CLI reference for running tests
- :doc:`agent_guide` - Full agent guide
