Troubleshooting
===============

Overview
--------
This guide outlines how to diagnose and resolve common issues in SYMFLUENCE related to configuration, installation, runtime execution, and HPC workflows.

---

Configuration Issues
--------------------
**Validation errors**
- Review reported line numbers and missing keys.
- Ensure YAML syntax uses spaces, not tabs.

**Common causes**
- Incorrect indentation or duplicated keys.
- Invalid parameter names or deprecated fields.
- Missing domain coordinates or file paths.

**Tip:** Always start from `config_template.yaml` and comment custom changes.

---

Installation and Environment
----------------------------
**Environment not found**
- Ensure you installed using the built-in installer:

  .. code-block:: bash

     ./scripts/symfluence-bootstrap --install

- Activate environment:

  .. code-block:: bash

     source .venv/bin/activate

**Missing libraries**
- Verify GDAL, NetCDF, and HDF5 are installed and accessible.
- Check ``which gdalinfo`` and ``ldd <libnetcdf.so>`` on Linux.

**Version conflicts**
- Use Python 3.11. Other versions are not guaranteed to be supported.
- Reinstall dependencies cleanly:

  .. code-block:: bash

     rm -rf .venv
     ./scripts/symfluence-bootstrap --install

---

Runtime and Workflow
--------------------
**Model execution stops prematurely**

- Inspect logs in ``_workLog_<domain_name>/``.
- Common issues:

  - Missing forcing data files.
  - Invalid HRU discretization thresholds.
  - Incorrect relative paths in model setup.

**Parallel job hangs**
- Check ``NUM_PROCESSES`` matches available cores.
- Reduce thread counts if memory limits are reached.
- Ensure the cluster scheduler allows requested resources.

**Routing or calibration failure**
- Confirm mizuRoute input files are generated.
- Verify all calibration parameters exist in the model setup.
- Run model once without calibration to isolate the issue.

---

Optimization and Emulation
--------------------------
**Slow convergence or stagnation**
- Reduce parameter bounds or start with fewer variables.
- Use smaller population size for initial testing.
- Consider switching algorithms (DE → DDS).

---

HPC and Cluster Use
-------------------
**Job submission fails**
- Check SLURM script formatting and environment variables.
- Ensure ``--slurm_job`` flag is used for automated submission.

**Missing output**
- Check for out-of-memory (OOM) events in SLURM logs.
- Confirm output directory write permissions.

---

Logging and Debugging
---------------------
All major steps produce detailed logs stored in ``_workLog_<domain_name>/``.

**Log files**
- ``system.log`` — Overall workflow progress and manager operations
- ``model_run.log`` — Model execution output and errors
- ``calibration.log`` — Optimization progress and parameter trials
- ``data_acquisition.log`` — Forcing and attribute data downloads

**Increasing verbosity**

Set log level in your configuration:

.. code-block:: yaml

   LOG_LEVEL: DEBUG  # Options: DEBUG, INFO, WARNING, ERROR

Or use environment variable:

.. code-block:: bash

   export SYMFLUENCE_LOG_LEVEL=DEBUG
   symfluence workflow run --config my_config.yaml

**Common debugging steps**

1. Check the most recent log file for error messages
2. Verify all input file paths exist and are accessible
3. Confirm configuration parameters match expected types
4. Run individual workflow steps to isolate the problem:

   .. code-block:: bash

      symfluence workflow step setup_project --config my_config.yaml
      symfluence workflow step acquire_forcings --config my_config.yaml

5. Enable traceback output for detailed error information

---

Common Error Messages
---------------------

**"Configuration validation failed"**
- Cause: Invalid YAML syntax or missing required fields
- Solution: Use ``symfluence config validate --config my_config.yaml`` to identify specific issues
- Check indentation (use spaces, not tabs)

**"Domain delineation failed"**
- Cause: Invalid pour point coordinates or DEM data issues
- Solution: Verify coordinates are in decimal degrees (latitude/longitude)
- Ensure DEM data covers the specified domain
- Check that TauDEM is properly installed: ``symfluence binary validate``

**"Forcing data download failed"**
- Cause: Network issues, invalid credentials, or unsupported date range
- Solution: Check internet connectivity
- For CARRA/CERRA: Verify CDS API credentials in ``~/.cdsapirc``
- For ERA5: Ensure dates are within available range (1940-present)

**"Model executable not found"**
- Cause: Model binaries not installed or not in PATH
- Solution: Install binaries: ``symfluence binary install summa fuse``
- Validate installation: ``symfluence binary validate``
- Check PATH includes binary locations

**"Calibration iteration failed"**
- Cause: Model run failed during parameter evaluation
- Solution: Test model run manually before calibration
- Reduce parameter search space
- Check observation data format and date alignment

**"Memory error" or "Killed"**
- Cause: Insufficient RAM for data processing or model execution
- Solution: Reduce domain size or discretization resolution
- Increase HPC job memory allocation
- Use chunked processing for large datasets

**"Permission denied"**
- Cause: Missing write permissions for output directories
- Solution: Check directory permissions: ``ls -la``
- Create output directory manually: ``mkdir -p <path>``
- Verify user has write access to project directory

**"NetCDF library not found"**
- Cause: Missing or incompatible NetCDF installation
- Solution: Install system dependencies (see :doc:`installation`)
- On HPC: Load appropriate modules: ``module load netcdf hdf5``
- Verify installation: ``python -c "import netCDF4; print(netCDF4.__version__)"``

---

Data Issues
-----------

**Missing forcing data**
- Verify data acquisition completed successfully
- Check ``forcing/raw_data/`` directory for downloaded files
- Confirm date range in configuration matches data availability
- Re-run acquisition: ``symfluence workflow step acquire_forcings``

**Spatial mismatch errors**
- Ensure all spatial data uses consistent coordinate reference system (CRS)
- Verify domain shapefile covers the study area
- Check DEM resolution is appropriate for domain size

**Observation data not found**
- Place observation files in ``observations/streamflow/preprocessed/``
- Ensure filename matches pattern: ``{domain_name}_streamflow*.csv``
- Verify CSV has datetime and discharge columns

---

Performance Issues
------------------

**Slow execution**
- Reduce domain size for initial testing
- Lower discretization resolution (fewer HRUs)
- Use coarser forcing data timestep if appropriate
- Enable parallel processing:

  .. code-block:: yaml

     NUM_PROCESSES: 4  # Match available cores

**Out of memory**
- Process forcing data in smaller time chunks
- Reduce number of HRUs in discretization
- Use distributed processing on HPC
- Monitor memory usage: ``top`` or ``htop``

**Long calibration time**
- Reduce number of iterations: ``MAX_ITERATIONS: 100``
- Use fewer calibration parameters
- Start with faster algorithm (DDS instead of DE)
- Use parallel calibration on HPC

---

Platform-Specific Issues
------------------------

**macOS**
- Install Xcode Command Line Tools: ``xcode-select --install``
- Use Homebrew for dependencies: ``brew install gdal netcdf hdf5``
- For Apple Silicon (M1/M2/M3), ensure arm64 compatible packages

**Linux**
- Install build essentials: ``sudo apt-get install build-essential gfortran``
- Verify shared library paths: ``ldconfig -p | grep netcdf``
- Check for conflicting conda environments

**HPC Clusters**
- Load required modules before running (see :doc:`installation` for cluster-specific recipes)
- Request appropriate resources in SLURM script
- Use cluster-optimized Python and library builds
- Check filesystem quotas and permissions

---

Getting Help
------------

If you encounter an issue not covered here:

1. **Check existing documentation**

   - :doc:`installation` — Setup and dependencies
   - :doc:`configuration` — Configuration parameters
   - :doc:`api` — Programmatic usage

2. **Search GitHub Issues**

   - https://github.com/symfluence-org/SYMFLUENCE/issues
   - Look for similar problems and solutions

3. **Report a new issue**

   When reporting issues, include:

   - SYMFLUENCE version: ``symfluence --version``
   - Operating system and version
   - Python version: ``python --version``
   - Complete error message and traceback
   - Configuration file (redact sensitive information)
   - Steps to reproduce the issue

4. **Community discussions**

   - https://github.com/symfluence-org/SYMFLUENCE/discussions
   - Ask questions and share experiences

---

Diagnostic Commands
-------------------

**Check installation**

.. code-block:: bash

   # Verify SYMFLUENCE is installed
   symfluence --version

   # Check system compatibility
   symfluence binary doctor

   # Validate binary installations
   symfluence binary validate

**Validate configuration**

.. code-block:: bash

   # Validate YAML syntax and required fields
   symfluence config validate --config my_config.yaml

   # Validate environment setup
   symfluence config validate-env

**Check workflow status**

.. code-block:: bash

   # Show workflow progress
   symfluence workflow status --config my_config.yaml

   # List available workflow steps
   symfluence workflow list-steps

**Test components individually**

.. code-block:: bash

   # Test project initialization
   symfluence workflow step setup_project --config my_config.yaml

   # Test data acquisition
   symfluence workflow step acquire_forcings --config my_config.yaml

   # Test model preprocessing
   symfluence workflow step preprocess_models --config my_config.yaml

---

Tips for Successful Runs
-------------------------

1. **Start small**: Test with a small domain and short time period first
2. **Validate early**: Use ``config validate`` before running workflows
3. **Check logs frequently**: Monitor ``_workLog_*/`` for progress and errors
4. **Test incrementally**: Run individual workflow steps before full workflow
5. **Keep backups**: Save working configurations for reference
6. **Use version control**: Track configuration changes with git
7. **Document changes**: Comment custom modifications in configuration files
8. **Test on HPC**: Start with small jobs to verify cluster setup
