.. _cli_reference:

=========================================
CLI Reference
=========================================

SYMFLUENCE provides a comprehensive command-line interface for managing hydrological
modeling workflows. This reference documents all available commands, options, and usage patterns.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

The CLI follows a two-level hierarchical structure:

- **Level 1**: Command categories (workflow, project, binary, config, job, example, agent,
  gui, tui, data, fews, list) plus the top-level ``doctor`` command
- **Level 2**: Specific actions within each category

Basic usage:

.. code-block:: bash

   symfluence <category> <command> [options]

Global Options
==============

These options may be written before the category or after the action. Options
that affect only supported operations are explicitly identified below.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Option
     - Description
   * - ``--config PATH``
     - Path to configuration file (default: ./config.yaml;
       override with SYMFLUENCE_DEFAULT_CONFIG)
   * - ``--debug``
     - Enable debug output and stack traces (console shows DEBUG-level detail)
   * - ``--quiet, -q``
     - Suppress console INFO output; warnings and errors are still shown, and
       the file log is unaffected. Console verbosity is three-state:
       quiet (WARNING+), normal (INFO+), debug (DEBUG+)
   * - ``--visualise / --visualize``
     - Enable visualization during workflow execution
   * - ``--diagnostic``
     - Enable diagnostic plots during workflow execution
   * - ``--dry-run``
     - Preview without making changes. Supported by ``workflow run/step/steps/resume/status/clean``
       (lists the steps and their completion state), ``binary install`` and
       ``binary install-sysdeps``, ``data download`` and ``job submit``; read-only
       commands run as usual. Every other command refuses to run under ``--dry-run``
       (exit code 2) instead of executing for real.
   * - ``--profile``
     - Enable I/O profiling for workflow execution
   * - ``--profile-output PATH``
     - Path for workflow profiling report (default: profile_report.json)
   * - ``--profile-stacks``
     - Capture stack traces in workflow profiling
   * - ``--version``
     - Display SYMFLUENCE version

For bundled-binary pass-through, global options may precede ``binary``. Every
argument after the tool name is forwarded unchanged; for example,
``symfluence --debug binary summa --version`` forwards ``--version`` to SUMMA.

Workflow Commands
=================

Manage and execute SYMFLUENCE workflows.

workflow run
------------

Execute the complete workflow from start to finish.

.. code-block:: bash

   symfluence workflow run [--config CONFIG] [--force-rerun] [--continue-on-error]

**Options:**

- ``--force-rerun``: Force rerun of all steps (skip caching)
- ``--continue-on-error``: Continue executing steps on errors

**Example:**

.. code-block:: bash

   symfluence workflow run --config my_config.yaml --force-rerun

workflow step
-------------

Execute a single workflow step.

.. code-block:: bash

   symfluence workflow step STEP_NAME [--config CONFIG] [--force-rerun]

**Available Steps:**

The canonical step names, in execution order, are:

- ``setup_project`` - Initialize project structure and shapefiles
- ``create_pour_point`` - Create pour point shapefile from coordinates
- ``acquire_attributes`` - Download and process geospatial attributes
- ``define_domain`` - Define hydrological domain boundaries
- ``discretize_domain`` - Discretize domain into HRUs or other units
- ``process_observed_data`` - Process observational data (streamflow, etc.)
- ``acquire_forcings`` - Acquire meteorological forcing data
- ``model_agnostic_preprocessing`` - Model-agnostic preprocessing
- ``build_model_ready_store`` - Build model-ready forcing/attributes/observations store
- ``model_specific_preprocessing`` - Model-specific input setup
- ``run_model`` - Execute the hydrological model
- ``postprocess_results`` - Postprocess and finalize results
- ``calibrate_model`` - Run calibration / parameter optimization
- ``run_benchmarking`` - Run benchmarking analysis
- ``run_decision_analysis`` - Run decision analysis for model comparison
- ``run_sensitivity_analysis`` - Run sensitivity analysis

This list (and the supported short aliases) can drift as the framework evolves;
run ``symfluence workflow list-steps`` for the authoritative, up-to-date set.

**Example:**

.. code-block:: bash

   symfluence workflow step calibrate_model --config config.yaml

workflow steps
--------------

Execute multiple workflow steps in sequence.

.. code-block:: bash

   symfluence workflow steps STEP1 STEP2 ... [--config CONFIG] [--force-rerun]

**Example:**

.. code-block:: bash

   symfluence workflow steps acquire_forcings model_agnostic_preprocessing run_model

workflow status
---------------

Show workflow execution status.

.. code-block:: bash

   symfluence workflow status [--config CONFIG]

workflow validate
-----------------

Validate configuration file syntax.

.. code-block:: bash

   symfluence workflow validate [--config CONFIG]

workflow list-steps
-------------------

List all available workflow steps.

.. code-block:: bash

   symfluence workflow list-steps

workflow resume
---------------

Resume workflow from a specific step.

.. code-block:: bash

   symfluence workflow resume STEP_NAME [--config CONFIG] [--force-rerun]

**Example:**

.. code-block:: bash

   symfluence workflow resume calibrate_model --config config.yaml

workflow clean
--------------

Clean intermediate or output files.

.. code-block:: bash

   symfluence workflow clean [--config CONFIG] [--level LEVEL] [--dry-run]

**Options:**

- ``--level``: Cleaning level (intermediate, outputs, all; default: intermediate)
- ``--dry-run``: Preview what would be cleaned

**Example:**

.. code-block:: bash

   symfluence workflow clean --level all --dry-run

workflow diagnose
-----------------

Run diagnostic plots on existing workflow outputs.

.. code-block:: bash

   symfluence workflow diagnose [--config CONFIG]

Project Commands
================

Initialize projects and configure pour points.

project init
------------

Initialize a new SYMFLUENCE project.

.. code-block:: bash

   symfluence project init [PRESET] [options]

**Options:**

- ``--domain TEXT``: Domain name
- ``--model {SUMMA,FUSE,GR,HYPE,MESH,RHESSys,NGEN,LSTM}``: Model selection
- ``--start-date YYYY-MM-DD``: Simulation start date
- ``--end-date YYYY-MM-DD``: Simulation end date
- ``--forcing TEXT``: Forcing dataset
- ``--discretization TEXT``: Discretization method
- ``--definition-method TEXT``: Domain definition method
- ``--output-dir PATH``: Output directory (default: ./)
- ``--scaffold``: Create full directory structure
- ``--minimal``: Create minimal configuration (10 required fields)
- ``--comprehensive``: Create comprehensive configuration (400+ options)
- ``-i, --interactive``: Run interactive configuration wizard

**Examples:**

.. code-block:: bash

   # Interactive setup with scaffold
   symfluence project init --interactive --scaffold

   # With preset
   symfluence project init fuse-provo --model FUSE --start-date 2020-01-01

   # Minimal config
   symfluence project init --domain MyDomain --minimal

project pour-point
------------------

Set up pour point workflow.

.. code-block:: bash

   symfluence project pour-point LAT/LON --domain-name NAME --definition METHOD [options]

**Arguments:**

- ``coordinates``: Pour point as "lat/lon" (e.g., 51.1722/-115.5717)

**Required Options:**

- ``--domain-name NAME``: Domain/watershed name
- ``--definition METHOD``: Definition method (lumped, point, subset, delineate)

**Optional:**

- ``--bounding-box``: Custom bounding box (LAT_MAX/LON_MIN/LAT_MIN/LON_MAX)
- ``--experiment-id``: Override experiment ID
- ``--output-dir``: Output directory

**Example:**

.. code-block:: bash

   symfluence project pour-point 51.1722/-115.5717 --domain-name Bow --definition delineate

project list-presets
--------------------

List available initialization presets.

.. code-block:: bash

   symfluence project list-presets

project show-preset
-------------------

Show details of a specific preset.

.. code-block:: bash

   symfluence project show-preset PRESET_NAME

Binary Commands
===============

Install and manage external tools.

binary install
--------------

Install external tools.

.. code-block:: bash

   symfluence binary install [TOOL1 TOOL2 ...] [--force]

Without tool names the *default* tools are installed (those registered without
``optional``); optional tools are installed only when named. ``symfluence binary
install --help`` prints both lists for the installed model packages. Tools required
by a named tool (e.g. ``sundials`` for ``summa``) are added automatically;
``--force``, ``--branch`` and ``--git-hash`` apply to the named tools only. An
unknown tool name, or any tool that fails, gives a non-zero exit code.

**Commonly used tools:**

- ``summa`` - SUMMA hydrological model
- ``mizuroute`` - mizuRoute routing model
- ``fuse`` - FUSE hydrological model
- ``hype`` - HYPE hydrological model
- ``mesh`` - MESH model
- ``taudem`` - TauDEM terrain analysis
- ``gistool`` - GIS analysis tool
- ``datatool`` - Data processing tool
- ``rhessys`` - RHESSys ecosystem model
- ``ngen`` - NextGen framework
- ``ngiab`` - NextGen in a Box
- ``sundials`` - SUNDIALS ODE solver

**Example:**

.. code-block:: bash

   symfluence binary install summa mizuroute taudem --force

binary validate
---------------

Validate installed binaries.

.. code-block:: bash

   symfluence binary validate [--verbose]

binary doctor
-------------

Run comprehensive system diagnostics.

.. code-block:: bash

   symfluence binary doctor

binary install-sysdeps
----------------------

Install system dependencies (compilers, libraries) for the current platform.

.. code-block:: bash

   symfluence binary install-sysdeps [--tool TOOL] [--dry-run]

**Options:**

- ``--tool``: Install deps for a specific tool only (e.g. summa, fuse)
- ``--dry-run``: Show install commands without executing them

binary info
-----------

Display information about installed tools.

.. code-block:: bash

   symfluence binary info

Config Commands
===============

Manage and validate configuration files.

config list-templates
---------------------

List available configuration templates.

.. code-block:: bash

   symfluence config list-templates

config validate
---------------

Validate configuration file.

.. code-block:: bash

   symfluence config validate [--config CONFIG]

config validate-env
-------------------

Validate system environment.

.. code-block:: bash

   symfluence config validate-env

config update
-------------

Update / migrate a configuration file to the current schema.

.. code-block:: bash

   symfluence config update CONFIG_FILE [--interactive]

config resolve
--------------

Resolve a configuration through the full 5-layer hierarchy and print the
effective values (useful for debugging defaults, env overrides, and aliases).

.. code-block:: bash

   symfluence config resolve [--config CONFIG] [--flat] [--json] [--diff] [--section SECTION]

**Options:**

- ``--flat``: Print the flat (legacy) key form instead of nested
- ``--json``: Emit JSON
- ``--diff``: Show only values that differ from the defaults
- ``--section SECTION``: Restrict output to a single config section

Job Commands
============

Submit workflows to HPC clusters via SLURM.

job submit
----------

Submit workflow as SLURM job.

.. code-block:: bash

   symfluence job submit [options] [WORKFLOW_COMMAND ...]

**Options:**

- ``--name NAME``: SLURM job name
- ``--time TIME``: Time limit (HH:MM:SS; default: 48:00:00)
- ``--nodes N``: Number of nodes (default: 1)
- ``--tasks N``: Number of tasks (default: 1)
- ``--memory MEM``: Memory requirement (default: 50G)
- ``--account ACCOUNT``: Account to charge
- ``--partition PARTITION``: Partition/queue name
- ``--modules MODULES``: Saved module collection to ``module restore`` in the job
  (default: none; the job inherits the submitting environment)
- ``--conda-env ENV``: Conda environment to activate in the job (default: none)
- ``--wait``: Monitor the job until it ends; exits non-zero unless the job finishes
  in state ``COMPLETED`` (checked with ``sacct``)
- ``--template PATH``: Batch script used instead of the generated ``#SBATCH`` header
  and environment setup; the SYMFLUENCE command block replaces the placeholder
  ``{{SYMFLUENCE_COMMAND}}`` or is appended when the template has none

**Examples:**

.. code-block:: bash

   # Simple submission
   symfluence job submit workflow run --config config.yaml

   # With resources
   symfluence job submit --time 72:00:00 --nodes 4 --tasks 16 \
     --account myaccount workflow run --config config.yaml

   # With monitoring
   symfluence job submit --wait workflow run --config config.yaml

Example Commands
================

Launch and manage example notebooks.

example launch
--------------

Launch an example notebook.

.. code-block:: bash

   symfluence example launch EXAMPLE_ID [--lab] [--notebook]

**Options:**

- ``--lab``: Launch in JupyterLab (default)
- ``--notebook``: Launch in classic Jupyter Notebook

example list
------------

List available example notebooks.

.. code-block:: bash

   symfluence example list

Agent Commands
==============

Launch an installed coding-agent CLI (Claude Code, Codex, Gemini, ...) primed as
the *SYMFLUENCE agent*, in one of two modes: ``agent model`` (drive experiments
conversationally) or ``agent code`` (extend the platform). SYMFLUENCE does not
ship its own language model — it detects an installed agent CLI, primes it with
that mode's profile (skills, identity with live project context, the SYMFLUENCE
MCP server, subagents), and replaces itself with that agent. Bare
``symfluence agent`` opens the TUI agent screen to pick a mode. See
:doc:`agent_guide` for the full walkthrough.

agent model
-----------

Start a **modelling session**: configs, runs, calibrations, and results, driven
conversationally through the workflow tools. Primed with the operational skills
(``explore-platform``, ``run-workflow-locally``, ``debug-calibration``) and the
MCP tools; its house rules forbid editing platform source code. Inside the TUI
this opens the native chat screen when Claude Code is the runtime (headless
stream-JSON driving); other runtimes get the suspend round-trip with modelling
priming.

.. code-block:: bash

   # Interactive modelling session (run from your project directory)
   symfluence agent model

   # One-shot prompt (runs once and exits — handy in scripts)
   symfluence agent model "validate my config and run the next step"

agent code
----------

Start a **coding session** in the host coding-agent CLI, primed with every
packaged skill and subagent. Interactive sessions open the TUI agent screen
first; the handoff always happens after the TUI exits and restores the
terminal. A one-shot ``PROMPT``, ``--direct``, a missing TTY, or a missing TUI
extra (``textual``) hands off immediately.

.. code-block:: bash

   # Interactive coding session (opens the TUI agent screen first)
   symfluence agent code

   # Hand off to the agent CLI immediately
   symfluence agent code --direct

   # One-shot prompt
   symfluence agent code "add an MSWEP forcing data handler"

   # Pick a specific CLI, or launch it bare (no SYMFLUENCE priming)
   symfluence agent code --cli codex
   symfluence agent code --no-skills

   # Forward extra flags to the underlying CLI after --
   symfluence agent code -- --model claude-sonnet-4-6

CLI selection and priming can also be controlled by environment variables
(``SYMFLUENCE_AGENT_CLI``, ``SYMFLUENCE_NO_SKILLS``); see :doc:`agent_guide`.
The Agent screen is also available inside ``symfluence tui launch`` (key ``7``).

agent doctor
------------

Diagnose the agent setup: detected runtimes (and which is active), API keys,
per-mode priming, packaged skills and subagents, cache directory, MCP server,
and detected project context. ``--json`` emits the diagnosis machine-readably.

.. code-block:: bash

   symfluence agent doctor
   symfluence agent doctor --json

agent mcp
---------

Serve the SYMFLUENCE MCP server on stdio. The session verbs wire this into the
host CLI automatically where supported; it can also be registered manually in
any MCP-capable tool. ``--mode`` restricts the tool set to one agent mode's
profile.

.. code-block:: bash

   symfluence agent mcp
   symfluence agent mcp --mode model

Deprecated aliases
~~~~~~~~~~~~~~~~~~~

``symfluence agent launch`` is a deprecated alias for ``agent code`` and will be
removed in a future release. The former ``start``, ``run``, ``list``, and
``skills`` verbs have been removed (``doctor`` covers the last two).

GUI Commands
============

Launch the Panel-based web GUI for interactive workflow management.

gui launch
----------

Start the SYMFLUENCE web interface.

.. code-block:: bash

   symfluence gui launch [--port PORT] [--address HOST] [--allow-origin HOST:PORT] [--no-browser] [--demo NAME] [--config CONFIG]

**Options:**

- ``--port PORT``: Server port (default: 5006)
- ``--address HOST``: Interface to bind (default: 127.0.0.1, local only). The GUI has
  no authentication; a non-loopback bind exposes it to the network.
- ``--allow-origin HOST:PORT``: Origin browsers use to reach the GUI (repeatable);
  required when ``--address`` is not loopback
- ``--no-browser``: Do not auto-open a browser tab
- ``--demo NAME``: Load a built-in demo (e.g., "bow" for Bow at Banff)

**Example:**

.. code-block:: bash

   # Launch with default settings
   symfluence gui launch

   # Launch on custom port without opening browser
   symfluence gui launch --port 8080 --no-browser

   # Launch with a demo configuration
   symfluence gui launch --demo bow

TUI Commands
============

Launch the interactive terminal user interface.

tui launch
----------

Start the SYMFLUENCE interactive terminal UI.

.. code-block:: bash

   symfluence tui launch [--demo NAME] [--config CONFIG]

**Options:**

- ``--demo NAME``: Load a built-in demo (e.g., "bow" for Bow at Banff)

Data Commands
=============

Download datasets independently without a full project setup.

data download
-------------

Download a specific dataset.

.. code-block:: bash

   symfluence data download DATASET [options]

**Arguments:**

- ``DATASET``: Dataset name (e.g., modis_lai, era5, grace)

**Options:**

- ``--bbox LAT_MAX/LON_MIN/LAT_MIN/LON_MAX``: Bounding box (required unless ``--config`` is provided)
- ``--shapefile PATH``: Shapefile to derive bounding box from (alternative to ``--bbox``)
- ``--start YYYY-MM-DD``: Start date (required unless ``--config`` is provided)
- ``--end YYYY-MM-DD``: End date (required unless ``--config`` is provided)
- ``--output PATH``: Output directory (default: ``./data/<dataset>``)
- ``--domain NAME``: Domain name for file naming (default: standalone)
- ``--force``: Force re-download of existing data
- ``--extra KEY=VALUE``: Extra configuration keys (repeatable)

**Examples:**

.. code-block:: bash

   # Download with explicit bounding box and dates
   symfluence data download era5 --bbox 52/-116/50/-114 --start 2020-01-01 --end 2020-12-31

   # Download using a shapefile for the bounding box
   symfluence data download modis_lai --shapefile my_basin.shp --start 2020-01-01 --end 2020-12-31

   # Download using settings from a config file
   symfluence data download era5 --config my_config.yaml

data list
---------

List all available datasets.

.. code-block:: bash

   symfluence data list

data info
---------

Show details about a specific dataset.

.. code-block:: bash

   symfluence data info DATASET

**Example:**

.. code-block:: bash

   symfluence data info era5

List Commands
=============

Introspect the live registry and config schema to discover what is available
right now (models, datasets, optimizers, config keys, etc.). Because the catalog
is read from the registry, it never goes stale.

list
----

.. code-block:: bash

   # Show every catalog and its count
   symfluence list

   # List the entries for one kind
   symfluence list KIND

**Kinds:** ``models``, ``forcings``, ``observations``, ``optimizers``,
``targets``, ``metrics``, ``presets``, ``templates``, ``steps``, ``config-keys``.

**Examples:**

.. code-block:: bash

   symfluence list models
   symfluence list forcings
   symfluence list config-keys

Doctor Command
==============

Run comprehensive system diagnostics (environment, path resolution, and binary
checks). This is a top-level command (distinct from ``binary doctor``, which is
scoped to installed binaries).

.. code-block:: bash

   symfluence doctor

FEWS Commands
=============

Delft-FEWS General Adapter operations — run pre/post processing and launch an
existing Delft-FEWS installation. Delft-FEWS itself is obtained from Deltares;
``symfluence binary install openfews`` only generates the General Adapter
configuration template.

fews pre
--------

Run the FEWS pre-adapter (import forcing, generate config).

.. code-block:: bash

   symfluence fews pre --run-info run_info.xml [--format {pi-xml,netcdf-cf}] [--id-map PATH] [--config CONFIG]

fews post
---------

Run the FEWS post-adapter (export results, write diagnostics).

.. code-block:: bash

   symfluence fews post --run-info run_info.xml [--format {pi-xml,netcdf-cf}] [--id-map PATH] [--config CONFIG]

fews run
--------

Run the full FEWS adapter cycle (pre -> model -> post). ``--config`` is required:
the workflow is run with the run file's period (and ``DOMAIN_NAME`` /
``EXPERIMENT_ID`` / ``HYDROLOGICAL_MODEL`` properties) applied on top of it, and the
post-adapter exports the workflow's results (``results/<experiment_id>_results.csv``).
Exported variables need an explicit ID-map entry (for example ``fews_id: Q.sim``,
``symfluence_id: discharge``, ``direction: export``). The forcing FEWS supplies is
translated but not injected into the model run, which uses the forcing of the
SYMFLUENCE configuration.

.. code-block:: bash

   symfluence fews run --run-info run_info.xml --config CONFIG [--format {pi-xml,netcdf-cf}] [--id-map PATH]

fews launch
-----------

Launch an existing Delft-FEWS installation (``FEWS_HOME``) with SYMFLUENCE on its
environment. Fails when no Delft-FEWS installation is found.

.. code-block:: bash

   symfluence fews launch [--config CONFIG]

**Options (pre/post/run):**

- ``--run-info PATH``: Path to ``run_info.xml`` (required)
- ``--format {pi-xml,netcdf-cf}``: Data exchange format (default: netcdf-cf)
- ``--id-map PATH``: Variable ID mapping YAML file

Exit Codes
==========

All commands return standardized exit codes:

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - Code
     - Name
     - Meaning
   * - 0
     - SUCCESS
     - Command completed successfully
   * - 1
     - GENERAL_ERROR
     - General error
   * - 2
     - USAGE_ERROR
     - Invalid arguments/usage
   * - 3
     - CONFIG_ERROR
     - Configuration file issues
   * - 4
     - VALIDATION_ERROR
     - Input validation failed
   * - 5
     - FILE_NOT_FOUND
     - Required file missing
   * - 6
     - DIRECTORY_NOT_FOUND
     - Required directory missing
   * - 7
     - BINARY_ERROR
     - External binary not found
   * - 8
     - BINARY_BUILD_ERROR
     - Failed to build binary
   * - 9
     - NETWORK_ERROR
     - Network/download failure
   * - 10
     - PERMISSION_ERROR
     - Permission denied
   * - 11
     - TIMEOUT_ERROR
     - Operation timed out
   * - 12
     - DEPENDENCY_ERROR
     - Missing dependency
   * - 13
     - MODEL_ERROR
     - Model execution error
   * - 14
     - WORKFLOW_ERROR
     - Workflow execution error
   * - 15
     - DATA_ERROR
     - Data processing error
   * - 20
     - JOB_SUBMIT_ERROR
     - SLURM submission failed
   * - 21
     - JOB_EXECUTION_ERROR
     - Job execution failed
   * - 130
     - USER_INTERRUPT
     - User pressed Ctrl+C
   * - 143
     - SIGTERM
     - Process terminated

Common Usage Patterns
=====================

Quick Project Setup
-------------------

.. code-block:: bash

   # Interactive setup
   symfluence project init --interactive --scaffold

   # Or with preset
   symfluence project init fuse-provo --scaffold

Complete Workflow
-----------------

.. code-block:: bash

   # Validate first
   symfluence workflow validate --config config.yaml

   # Run everything
   symfluence workflow run --config config.yaml

Selective Steps
---------------

.. code-block:: bash

   # Single step
   symfluence workflow step calibrate_model --config config.yaml

   # Multiple steps
   symfluence workflow steps acquire_forcings run_model

   # Resume from step
   symfluence workflow resume calibrate_model --config config.yaml

HPC Submission
--------------

.. code-block:: bash

   # Save environment
   module save symfluence_modules

   # Submit with monitoring
   symfluence job submit --time 72:00:00 --nodes 4 --wait \
     workflow run --config config.yaml

Debugging
---------

.. code-block:: bash

   # Enable profiling
   symfluence workflow run --config config.yaml --profile --profile-stacks

   # Debug mode
   symfluence workflow run --config config.yaml --debug

Getting Help
============

.. code-block:: bash

   # General help
   symfluence --help

   # Category help
   symfluence workflow --help

   # Command help
   symfluence workflow run --help
