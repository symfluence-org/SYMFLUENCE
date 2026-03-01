# Changelog

All notable changes to SYMFLUENCE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.7.1] - 2026-03-01

### Fixed
- **HPC file-locking workaround**: Added tempfile fallback (attempt 4) to ERA5
  NetCDF writer for parallel filesystems (Lustre/GPFS/BeeGFS) where HDF5
  `fcntl()` locking fails even with `HDF5_USE_FILE_LOCKING=FALSE`. Writes to
  `$TMPDIR` first, then moves the file to the target path.
- **mizuRoute Makefile build**: Collapse multi-line `LIBNETCDF` before appending
  RPATH flags to avoid breaking backslash continuation (caused "missing separator"
  errors on CI).
- **ngen linker (expat)**: Detect system UDUNITS2 via `pkg-config`/`ldconfig`
  (not just explicit env var) so `-lexpat` is added for XML symbol resolution.
- **Wflow calibration**: Fix snow parameters, routing, and unit conversion issues.
  Add Oudin PET, log-transform bounds, and sub-daily resampling support.
- **FUSE preprocessing**: Make streamflow observations optional.
- **MPI launcher**: Add fallback when preferred launcher fails; prefer `mpirun`
  over `srun` for launcher detection.

### Added
- SPDX license headers on all source files.
- Coverage tracking in cross-platform CI workflow.
- Unit tests for NGEN, JFUSE, HYPE model runners and configs.
- Unit tests for core modules (file_utils, validation, path_resolver).
- Unit tests for `_safe_to_netcdf` fallback chain.
- WATFLOOD: Expand calibration to 16 parameters with tests.
- Bow River preset and default `DEM_SOURCE` to Copernicus.
- Config-hash stage invalidation for workflow orchestrator.

---

## [0.7.0] - 2026-02-10

> **Note**: This release jumps from the last tagged release v0.5.2 directly to v0.7.0.
> Versions 0.5.3–0.6.0 were development milestones on the `develop` branch and were
> never tagged or published to `main`. Their changes are included in this release and
> documented below in the [0.6.0] and [0.5.x] sections for historical reference.

> **Breaking Change**: This release refactors the CLI to a subcommand architecture.
> All existing CLI commands will need to be updated.

### Added
- **New CLI Commands**
  - `symfluence gui launch` - Panel-based web GUI for workflow management
  - `symfluence data download|list|info` - Standalone dataset acquisition without a full project
  - `--shapefile` option for `symfluence data download` to derive bounding box from a shapefile

- **Documentation Improvements**
  - New CFuse model guide (experimental differentiable PyTorch-based FUSE)
  - New JFuse model guide (experimental JAX-based FUSE with gradient support)
  - New WM-Fire model guide (wildfire spread simulation with RHESSys)
  - Comprehensive CLI reference documentation with all commands and options
  - Expanded configuration parameter reference
  - Fixed Python API examples in calibration documentation
  - New testing guide for contributors with pytest patterns and CI integration
  - Added agent-assisted contribution workflow documentation

- **Agent Improvements**
  - Enhanced `show_staged_changes` with diff statistics and file summary
  - Improved `generate_pr_description` with auto-detection of modified files
  - Better PR descriptions with context-specific sections and testing checklists
  - Added documentation for fuzzy matching threshold parameter

- **HBV Module Refactoring**
  - New `losses.py` module with differentiable NSE/KGE loss functions and gradient utilities
  - New `parameters.py` module with parameter bounds, defaults, and scaling utilities
  - Lazy imports in `__init__.py` for optional JAX dependency
  - Reduced `model.py` by ~650 lines through modularization

- **HBV Parameter Regionalization**
  - Neural transfer functions for spatially-varying parameter estimation
  - Support for catchment attribute-based parameter prediction

- **HBV Calibration Improvements**
  - Hydrograph signature metrics for multi-objective calibration
  - Enhanced optimizer with adaptive learning rates

- New modular command structure in `src/symfluence/cli/`:
  - `argument_parser.py` - Main parser with subcommand structure
  - `validators.py` - Validation utilities
  - `commands/` directory with category-specific handlers

### Changed
- **Complete CLI Refactor**
  - Replaced flat flag-based interface with modern two-level subcommand architecture
  - New structure: `symfluence <category> <action>` instead of `symfluence --flag`
  - 9 command categories: workflow, project, binary, config, job, example, agent, gui, data
  - Eliminated complex mode detection logic
  - Archived old `cli_argument_manager.py` for reference

- **New CLI Structure**
  ```bash
  # Workflow commands
  symfluence workflow run [--config CONFIG]
  symfluence workflow step STEP_NAME
  symfluence workflow list-steps
  symfluence workflow status

  # Project commands
  symfluence project init [PRESET]
  symfluence project pour-point LAT/LON --domain-name NAME --definition METHOD
  symfluence project list-presets

  # Binary/tool commands
  symfluence binary install [TOOL...]
  symfluence binary validate
  symfluence binary doctor

  # Configuration commands
  symfluence config validate
  symfluence config validate-env
  symfluence config list-templates

  # Job commands
  symfluence job submit [WORKFLOW_CMD] [SLURM_OPTIONS]

  # Example commands
  symfluence example launch EXAMPLE_ID
  symfluence example list

  # Agent commands
  symfluence agent start
  symfluence agent run PROMPT

  # GUI commands
  symfluence gui launch

  # Data commands
  symfluence data download DATASET --bbox W S E N
  symfluence data list
  symfluence data info DATASET
  ```

- **HBV Sub-Daily Parameter Scaling**
  - Implemented exact exponential scaling for recession coefficients (K0, K1, K2)
  - Formula: `k_sub = 1 - (1 - k_daily)^(dt/24)` replaces linear approximation
  - Eliminates ~5-13% error in recession behavior at sub-daily timesteps
  - Flux rate parameters (CFMAX, PERC) continue to use linear scaling
  - Added `FLUX_RATE_PARAMS` and `RECESSION_PARAMS` constants

- **MESH Preprocessing Consolidation**
  - Moved configuration defaults to dedicated `config_defaults.py`
  - Streamlined `config_generator.py` and `meshflow_manager.py`
  - Expanded `parameter_fixer.py` with robust parameter handling
  - Updated NALCMS to CLASS land cover mapping (wetland, snow/ice corrections)
  - Added unit conversion utilities for forcing data
  - Improved logging (replaced debug prints with proper logger calls)

- **FLUXCOM ET Acquisition Rework**
  - Replaced stub ICOS downloader with fully functional `_download_from_icos()` using ICOS Carbon Portal metadata API
  - Smart variable matching (exact name, then substring, then fallback)
  - Automatic unit conversion (W/m2, mm/hr, mm/day)
  - Nearest-neighbor fallback for basins smaller than the grid cell

- **MODIS ET Acquisition Improvements**
  - Spatial subsetting via sinusoidal projection for large tiles with small domains
  - Improved special-value masking (all MODIS QC codes, not just fill value)
  - SDS name matching now tries exact match before substring search

- **Reporting Module Cleanup**
  - Simplified plotter implementations
  - Improved shapefile handling

- **Example Notebooks — Typed Config Migration**
  - Migrated notebooks 02a–03b from legacy `yaml.safe_load` + flat dict pattern to `SymfluenceConfig.from_minimal()` typed config API
  - All 9 example notebooks (01a–04b) now use the same typed, validated, frozen config pattern
  - Removed MAF-specific language from all example notebooks; acquisition comments now reference `DATA_ACCESS: 'cloud'` or `'maf'` config setting
  - Fixed pre-existing bugs: 02a `NameError` on config access, 03b `cf.managers` wrong variable name, 03b config overwrite bug, 03b outdated CLI syntax

### Fixed
- MESH lumped mode routing: switched from `run_def` to `noroute` mode to correctly preserve lower-zone baseflow (`wf_lzs`); `run_def` in MESH 1.5.6 bypasses `STGGW` storage, causing zero baseflow
- Traceback handling in MESH postprocessor
- Added missing `pydantic` runtime dependency to `pyproject.toml`
- Fixed invalid escape sequences in `__init__.py` warning filters for Python 3.12+ compatibility
- CI pipeline now correctly fails on unit/integration test regressions (removed `|| true` from pytest invocations)

### Documentation
- Added comprehensive sub-daily simulation section to HBV model guide
- Documented parameter scaling approaches (exact vs linear)
- Added validation methodology for sub-daily implementations

### Migration Guide

| Old Command (v0.6.x) | New Command (v0.7.0) |
|----------------------|----------------------|
| `symfluence --calibrate_model` | `symfluence workflow step calibrate_model` |
| `symfluence --setup_project --create_pour_point` | `symfluence workflow steps setup_project create_pour_point` |
| `symfluence --get_executables summa` | `symfluence binary install summa` |
| `symfluence --validate_binaries` | `symfluence binary validate` |
| `symfluence --doctor` | `symfluence binary doctor` |
| `symfluence --init fuse-provo --scaffold` | `symfluence project init fuse-provo --scaffold` |
| `symfluence --list_presets` | `symfluence project list-presets` |
| `symfluence --workflow_status` | `symfluence workflow status` |
| `symfluence --list_steps` | `symfluence workflow list-steps` |
| `symfluence --pour_point 51/-115 --domain_def delineate --domain_name Test` | `symfluence project pour-point 51/-115 --domain-name Test --definition delineate` |
| `symfluence --agent` | `symfluence agent start` |
| `symfluence --example_notebook 1a` | `symfluence example launch 1a` |

**Benefits of new CLI:**
- Clearer command organization and discoverability
- Better help messages (`symfluence workflow --help` shows workflow-specific options)
- Easier to extend with new commands
- Industry-standard pattern (like git, docker, kubectl)

---

## [0.6.0] - 2025-12-29

### Added
- **Calibration Observation Data Utilities**
  - `download_smhi_discharge.py` - Download discharge data from Swedish Meteorological Institute
  - `prepare_streamflow_for_calibration.py` - Convert discharge CSV to calibration format
  - `setup_calibration.py` - Automated calibration setup with parameter bounds
  - Calibration demo tests for Elliðaár (Iceland) and Fyris (Sweden) basins

- **CARRA/CERRA Data Processing Improvements**
  - Fixed CARRA longitude normalization in spatial subsetting
  - `FORCING_TIME_STEP_SIZE` configuration support (10800s for CERRA)
  - `FORCING_SHAPE_ID_NAME` configuration support with default 'ID'

### Changed
- CLI orchestrator integration completed with full workflow execution support
- Version bump to 0.6.0 across all version references

### Removed
- **Deprecated CONFLUENCE backward compatibility**
  - Removed `CONFLUENCE.py` wrapper file
  - Removed `./confluence` shell script
  - Removed `CONFLUENCE_DATA_DIR` and `CONFLUENCE_CODE_DIR` configuration support
  - All documentation now uses SYMFLUENCE exclusively

### Fixed
- CARRA spatial subsetting for small basin extents
- EASYMORE remapping failures for CARRA datasets

---

## [0.5.11] - 2025-12-15

### Changed
- Enhanced ngen outlet detection
- Cleaned up technical debt

### Fixed
- Mypy type errors in config property inheritance
- Completed typed config migration for base classes

### Improved
- Centralized evaluation metric logic
- Improved linting and added ruff tests to pyproject.toml

---

## [0.5.3] - 2025-11-15

### Added
- Support for cloud acquisition of:
  - Copernicus DEM
  - MODIS land cover
  - Global USDA-NRCS soil texture class map
  - Forcing datasets: ERA5, NEX-GDDP, CONUS404, AORC
- Agnostic pipeline for cloud-based ERA5 matched workflows
- Full cloud-integrated workflow tested with ERA5
- Made MPI worker log generation optional
- Initial t-route support (in progress)

---

## [0.5.2] - 2025-11-12

### Major: Formal Initial Release with End-to-End CI Validation

**This release marks the first fully reproducible SYMFLUENCE workflow with continuous integration.**

### Added
- **End-to-End CI Pipeline (Example Notebook 2a Equivalent)**
  Integrated a comprehensive GitHub Actions workflow that builds, validates, and runs SYMFLUENCE automatically on every commit to `main`.
  - Compiles all hydrologic model dependencies (TauDEM, mizuRoute, FUSE, NGEN).
  - Validates MPI, NetCDF, GDAL, and HDF5 environments.
  - Executes key steps (`setup_project`, `create_pour_point`, `define_domain`, `discretize_domain`, `model_agnostic_preprocessing`, `run_model`, `calibrate_model`, `run_benchmarking`).
  - Confirms reproducible outputs under `SYMFLUENCE_DATA_DIR/domain_Bow_at_Banff`.
  - Runs both wrapper (`./symfluence`) and direct Python entrypoints equivalently.

### Changed
- Updated `external_tools_config.py` to include automatic path resolution for TauDEM binaries (e.g., `moveoutletstostrms → moveoutletstostreams`).
- Expanded logging and run summaries for CI visibility.
- Protected `main` branch to require successful CI validation before merge.

### Notes
This release formalizes SYMFLUENCE's **reproducibility framework**, guaranteeing that all supported workflows can be rebuilt and validated automatically on clean systems.

---

## [0.5.0] - 2025-01-09

### Major: CONFLUENCE → SYMFLUENCE Rebranding

**This is the rebranding release.** The project is now SYMFLUENCE (SYnergistic Modelling Framework for Linking and Unifying Earth-system Nexii for Computational Exploration).

### Added
- Complete rebranding to SYMFLUENCE
- New domain: [symfluence.org](https://symfluence.org)
- PyPI package: `pip install symfluence`
- Backward compatibility for all CONFLUENCE names (with deprecation warnings)

### Changed
- Main script: `CONFLUENCE.py` → `symfluence.py`
- Shell command: `./confluence` → `./symfluence`
- Config parameters: `CONFLUENCE_*` → `SYMFLUENCE_*`
- Repository: `DarriEy/CONFLUENCE` → `DarriEy/SYMFLUENCE`

### Deprecated
- All CONFLUENCE naming (removed in v0.6.0)

---

## Links

- **PyPI**: [pypi.org/project/symfluence](https://pypi.org/project/symfluence)
- **Documentation**: [symfluence.readthedocs.io](https://symfluence.readthedocs.io)
- **GitHub**: [github.com/DarriEy/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)
- **Issues**: [github.com/DarriEy/SYMFLUENCE/issues](https://github.com/DarriEy/SYMFLUENCE/issues)
