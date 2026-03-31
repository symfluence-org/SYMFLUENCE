# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NGen Model Runner.

Manages the execution of the NOAA NextGen Framework (ngen).
Refactored to use the Unified Model Execution Framework.
"""

import logging
import os
import subprocess  # nosec B404 - Required for running Docker containers and model executables
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.models.base import BaseModelRunner
from symfluence.models.registry import ModelRegistry


@ModelRegistry.register_runner('NGEN', method_name='run_ngen')
class NgenRunner(BaseModelRunner):  # type: ignore[misc]
    """
    Runner for NextGen Framework simulations.

    Handles execution of ngen with proper paths and error handling.
    Uses the Unified Model Execution Framework for subprocess execution.
    """

    MODEL_NAME = "NGEN"

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        reporting_manager: Optional[Any] = None,
        ngen_settings_dir: Optional[Path] = None,
        ngen_output_dir: Optional[Path] = None,
    ):
        # Store overrides before super().__init__ so _setup_model_specific_paths can use them
        self._ngen_settings_dir_override = Path(ngen_settings_dir) if ngen_settings_dir else None
        self._ngen_output_dir_override = Path(ngen_output_dir) if ngen_output_dir else None
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

    def _setup_model_specific_paths(self) -> None:
        """Set up NGEN-specific paths."""
        # Check if parallel worker has provided isolated settings directory
        if getattr(self, '_ngen_settings_dir_override', None):
            self.ngen_setup_dir = self._ngen_settings_dir_override
        else:
            self.ngen_setup_dir = self.project_dir / "settings" / "NGEN"

        # Use enhanced get_model_executable with candidates for multi-location search
        # Search order: root, cmake_build, bin subdirectories
        self.ngen_exe = self.get_model_executable(
            install_path_key='NGEN_INSTALL_PATH',
            default_install_subpath='installs/ngen',
            default_exe_name='ngen',
            candidates=['', 'cmake_build', 'bin'],
            must_exist=True
        )

    def _should_create_output_dir(self) -> bool:
        """NGEN creates directories on-demand in run_ngen."""
        return False

    def run_ngen(self, experiment_id: str = None):
        """Execute NextGen model simulation."""
        use_ngiab = self._get_config_value(lambda: None, default=False, dict_key='USE_NGIAB')
        if use_ngiab:
            self.logger.info("Using NGIAB Docker for NextGen execution")
            return self._run_ngen_docker(experiment_id)

        self.logger.debug("Starting NextGen model run")

        with symfluence_error_handler(
            "NextGen model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            if experiment_id is None:
                experiment_id = self.experiment_id

            if getattr(self, '_ngen_output_dir_override', None):
                output_dir = self._ngen_output_dir_override
            else:
                output_dir = self.get_experiment_output_dir(experiment_id)
            output_dir.mkdir(parents=True, exist_ok=True)

            catchment_file, fallback_catchment_file, nexus_file, realization_file, use_geojson = (
                self._resolve_ngen_paths(output_dir)
            )

            # Build ngen command with patched realization
            import shutil
            patched_realization = output_dir / "realization_config_patched.json"
            shutil.copy(realization_file, patched_realization)
            self._patch_realization_libraries(patched_realization)

            ngen_cmd = [
                str(self.ngen_exe), str(catchment_file), "all",
                str(nexus_file), "all", str(patched_realization),
            ]
            self.logger.debug(f"Running command: {' '.join(ngen_cmd)}")

            log_file = output_dir / "ngen_log.txt"
            env = self._setup_ngen_environment()

            # Retry logic for SIGSEGV (-11) crashes during initialization.
            # NGEN has a non-deterministic null-pointer bug in CsvPerFeatureForcingProvider::read_csv()
            # on macOS ARM64 that crashes before model simulation starts. Retrying recovers.
            max_retries = self._get_config_value(lambda: None, default=4, dict_key='NGEN_SIGSEGV_RETRIES')
            last_error = None

            for attempt in range(1 + max_retries):
                try:
                    self.execute_subprocess(
                        ngen_cmd, log_file,
                        cwd=self.ngen_exe.parent, env=env,
                        success_message="NextGen model run completed successfully",
                    )
                    self._move_ngen_outputs(self.ngen_exe.parent, output_dir)

                    if self._get_config_value(lambda: self.config.model.ngen.run_troute, default=True):
                        self.logger.debug("Starting t-route routing...")
                        troute_success = self._run_troute_routing(output_dir)
                        if not troute_success:
                            self.logger.debug("T-Route routing not available. Using NGEN nexus outputs directly.")
                    else:
                        self.logger.debug("T-Route routing disabled in config")

                    return True

                except subprocess.CalledProcessError as e:
                    last_error = e
                    # SIGSEGV (-11) during init: retry with exponential backoff
                    if e.returncode == -11 and attempt < max_retries:
                        import time
                        delay = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s, 4s
                        self.logger.warning(
                            f"NGEN crashed with SIGSEGV (attempt {attempt + 1}/{1 + max_retries}). "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        continue
                    break

            return self._handle_gpkg_fallback(
                last_error, ngen_cmd, log_file, env, output_dir,
                use_geojson, fallback_catchment_file,
            )

    def _resolve_ngen_paths(self, output_dir):
        """Resolve catchment, nexus, and realization paths; clean stale outputs.

        Returns (catchment_file, fallback_catchment_file, nexus_file,
        realization_file, use_geojson).
        """
        domain_name = self.domain_name

        import platform
        use_geojson = getattr(self, "_use_geojson_catchments", False)
        if platform.system() == "Darwin":
            self.logger.debug("Forcing GeoJSON catchments on macOS for stability")
            use_geojson = True

        if use_geojson:
            catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
        else:
            catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.gpkg"
        fallback_catchment_file = self.ngen_setup_dir / f"{domain_name}_catchments.geojson"
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        realization_file = self.ngen_setup_dir / "realization_config.json"

        self.verify_required_files(
            [catchment_file, nexus_file, realization_file],
            context="NextGen model execution",
        )

        for stale_pattern in ['nex-*.csv', 'cat-*.csv']:
            for stale_file in output_dir.glob(stale_pattern):
                stale_file.unlink()

        return catchment_file, fallback_catchment_file, nexus_file, realization_file, use_geojson

    def _setup_ngen_environment(self):
        """Build subprocess environment dict with venv and library paths."""
        env = os.environ.copy()

        install_path = self._get_config_value(
            lambda: self.config.model.ngen.install_path,
            default='default'
        )

        if install_path == 'default':
            ngen_base = self.data_dir / 'installs' / 'ngen'
        else:
            p = Path(install_path)
            ngen_base = p.parent if p.name == 'cmake_build' else p

        # Virtual-environment handling
        ngen_venv = ngen_base / "ngen_venv"
        if ngen_venv.exists():
            self.logger.debug(f"Using NGEN venv: {ngen_venv}")
            env['VIRTUAL_ENV'] = str(ngen_venv)
            env.pop('PYTHONHOME', None)
            env.pop('PYTHONPATH', None)
            ngen_venv_bin = str(ngen_venv / "bin")
            if 'PATH' in env:
                path_parts = env['PATH'].split(os.pathsep)
                filtered_parts = [p for p in path_parts if '/venv' not in p.lower()
                                  and '/.venv' not in p.lower()
                                  and '/envs/' not in p.lower()]
                env['PATH'] = ngen_venv_bin + os.pathsep + os.pathsep.join(filtered_parts)
            else:
                env['PATH'] = ngen_venv_bin
        else:
            env.pop('PYTHONPATH', None)
            env.pop('PYTHONHOME', None)
            env.pop('VIRTUAL_ENV', None)
            if 'PATH' in env:
                path_parts = env['PATH'].split(os.pathsep)
                filtered_parts = [p for p in path_parts if '/venv' not in p.lower()
                                  and '/.venv' not in p.lower()
                                  and '/envs/' not in p.lower()]
                env['PATH'] = os.pathsep.join(filtered_parts)

        # UDUNITS2 XML database path — required for unit conversions between BMI modules.
        # When UDUNITS2 is statically linked, dladdr resolves the XML path relative to the
        # ngen binary (e.g. <exe_dir>/../share/udunits/udunits2.xml). If the build installed
        # UDUNITS2 into a subdirectory (e.g. ngen/udunits2/), the computed path is wrong.
        # Setting the env var is the most reliable way to ensure UDUNITS2 finds its database.
        udunits_xml = ngen_base / "udunits2" / "share" / "udunits" / "udunits2.xml"
        if not udunits_xml.exists():
            # Fallback: check the path UDUNITS2 computes from the binary location
            udunits_xml = ngen_base / "share" / "udunits" / "udunits2.xml"
        if udunits_xml.exists():
            env['UDUNITS2_XML_PATH'] = str(udunits_xml)
        else:
            self.logger.warning(
                "UDUNITS2 XML database not found — unit conversions between BMI modules "
                "will fail silently, causing incorrect results (e.g. zero precipitation in CFE)."
            )

        # BMI library paths
        lib_paths = []
        for sub in ["extern/sloth/cmake_build", "extern/cfe/cmake_build",
                     "extern/evapotranspiration/evapotranspiration/cmake_build",
                     "extern/noah-owp-modular/cmake_build",
                     "extern/topmodel/cmake_build",
                     "extern/sac-sma/cmake_build",
                     "extern/snow17/cmake_build"]:
            p1 = ngen_base / sub
            p2 = ngen_base / "cmake_build" / sub
            if p1.exists():
                lib_paths.append(str(p1.resolve()))
            elif p2.exists():
                lib_paths.append(str(p2.resolve()))

        lib_paths.append(str(self.ngen_exe.parent.resolve()))
        lib_paths.append("/opt/homebrew/lib")
        lib_path_str = os.pathsep.join(lib_paths)

        for var in ['DYLD_LIBRARY_PATH', 'LD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
            existing_path = env.get(var, '')
            env[var] = f"{lib_path_str}{os.pathsep}{existing_path}" if existing_path else lib_path_str

        self.logger.debug(f"Executing ngen with DYLD_LIBRARY_PATH={env.get('DYLD_LIBRARY_PATH')}")
        return env

    def _handle_gpkg_fallback(self, error, ngen_cmd, log_file, env,
                              output_dir, use_geojson, fallback_catchment_file):
        """Retry with GeoJSON catchments on SQLite/GPKG failure."""
        is_likely_sqlite_issue = (error.returncode == -6)

        if not use_geojson and fallback_catchment_file.exists():
            try:
                log_text = log_file.read_text(encoding='utf-8', errors='ignore')
            except (FileNotFoundError, OSError, PermissionError):
                log_text = ""
            sqlite_error = "SQLite3 support required to read GeoPackage files"

            if is_likely_sqlite_issue or sqlite_error in log_text:
                self.logger.warning(
                    f"NGEN failed (code {error.returncode}); retrying with GeoJSON catchments"
                )
                ngen_cmd[1] = str(fallback_catchment_file)
                try:
                    self.execute_subprocess(
                        ngen_cmd, log_file,
                        cwd=self.ngen_exe.parent, env=env,
                        success_message="NextGen model run completed successfully (GeoJSON fallback)",
                    )
                    self._use_geojson_catchments = True
                    self._move_ngen_outputs(self.ngen_exe.parent, output_dir)
                    return True
                except subprocess.CalledProcessError as retry_error:
                    self.logger.error(f"NextGen model run failed with error code {retry_error.returncode}")
                    self.logger.error(f"Check log file: {log_file}")
                    return False

        self.logger.error(f"NextGen model run failed with error code {error.returncode}")
        self.logger.error(f"Check log file: {log_file}")
        return False

    def _patch_realization_libraries(self, realization_file: Path):
        """Patch realization config to use absolute paths for libraries and init_configs."""
        import json
        try:
            content = realization_file.read_text(encoding='utf-8')
            if len(content.strip()) < 10:
                # File is empty/corrupted — re-copy from source
                source_realization = self.ngen_setup_dir / "realization_config.json"
                if source_realization.exists() and source_realization.resolve() != realization_file.resolve():
                    import shutil
                    shutil.copy2(source_realization, realization_file)
                    content = realization_file.read_text(encoding='utf-8')
                    self.logger.warning("Recovered empty realization config from source")
                else:
                    self.logger.warning("Realization config is empty and no source available")
                    return

            data = json.loads(content)

            changed = False
            # Determine absolute ngen base directory
            install_path = self._get_config_value(
                lambda: self.config.model.ngen.install_path,
                default='default'
            )

            if install_path == 'default':
                ngen_base = self.data_dir / 'installs' / 'ngen'
            else:
                p = Path(install_path)
                if p.name == 'cmake_build': ngen_base = p.parent
                else: ngen_base = p

            if 'global' in data and 'formulations' in data['global']:
                for formulation in data['global']['formulations']:
                    if 'params' in formulation and 'modules' in formulation['params']:
                        for module in formulation['params']['modules']:
                            mod_params = module.get('params', {})

                            # 1. Patch library_file
                            if 'library_file' in mod_params:
                                lib_path = mod_params['library_file']
                                target_subpath = None
                                if 'pet' in lib_path.lower():
                                    target_subpath = "extern/evapotranspiration/evapotranspiration/cmake_build"
                                elif 'cfe' in lib_path.lower():
                                    target_subpath = "extern/cfe/cmake_build"
                                elif 'sloth' in lib_path.lower():
                                    target_subpath = "extern/sloth/cmake_build"
                                elif 'surface' in lib_path.lower() or 'noah' in lib_path.lower():
                                    target_subpath = "extern/noah-owp-modular/cmake_build"
                                elif 'topmodel' in lib_path.lower():
                                    target_subpath = "extern/topmodel/cmake_build"
                                elif 'sac' in lib_path.lower():
                                    target_subpath = "extern/sac-sma/cmake_build"
                                elif 'snow17' in lib_path.lower():
                                    target_subpath = "extern/snow17/cmake_build"

                                if target_subpath:
                                    filename = Path(lib_path).name
                                    # Try both ngen_base/extern and ngen_base/cmake_build/extern
                                    p1 = ngen_base / target_subpath
                                    p2 = ngen_base / "cmake_build" / target_subpath
                                    lib_dir = p1 if p1.exists() else p2

                                    actual_lib = None
                                    if (lib_dir / filename).exists():
                                        actual_lib = lib_dir / filename
                                    else:
                                        stem = filename.split('.')[0]
                                        candidates = []
                                        for ext in ("dylib", "so"):
                                            candidates.extend(lib_dir.glob(f"{stem}*.{ext}"))
                                        if candidates:
                                            actual_lib = candidates[0]

                                    if actual_lib:
                                        abs_lib_path = str(actual_lib.resolve())
                                        if mod_params['library_file'] != abs_lib_path:
                                            mod_params['library_file'] = abs_lib_path
                                            changed = True
                                            self.logger.debug(f"Patched library {lib_path} -> {abs_lib_path}")

                            # 2. Patch init_config
                            if 'init_config' in mod_params:
                                old_path = mod_params['init_config']
                                if old_path and old_path != "/dev/null":
                                    mod_type_name = str(mod_params.get('model_type_name', '')).upper()
                                    target_mod = None
                                    if 'PET' in mod_type_name or 'pet' in old_path.lower(): target_mod = 'PET'
                                    elif 'CFE' in mod_type_name or 'cfe' in old_path.lower(): target_mod = 'CFE'
                                    elif 'NOAH' in mod_type_name or 'noah' in old_path.lower() or '.input' in old_path.lower(): target_mod = 'NOAH'
                                    elif 'TOPMODEL' in mod_type_name or 'topmodel' in old_path.lower(): target_mod = 'TOPMODEL'
                                    elif 'SACSMA' in mod_type_name or 'sacsma' in old_path.lower(): target_mod = 'SACSMA'
                                    elif 'SNOW17' in mod_type_name or 'snow17' in old_path.lower(): target_mod = 'SNOW17'

                                    if target_mod:
                                        filename = Path(old_path).name
                                        # Preserve {{id}} template in filenames; only make path absolute.
                                        new_path = str((self.ngen_setup_dir / target_mod / filename).resolve())
                                        if old_path != new_path:
                                            mod_params['init_config'] = new_path
                                            changed = True
                                            self.logger.debug(f"Patched {target_mod} init_config to {new_path}")

            # 3. Patch forcing file pattern
            if 'global' in data and 'forcing' in data['global']:
                forcing = data['global']['forcing']
                if 'file_pattern' in forcing and '{{id}}' in forcing['file_pattern']:
                    # Preserve {{id}} template; avoid collapsing to a single catchment.
                    self.logger.debug("Preserving {{id}} in forcing file pattern for multi-catchment runs")

            # 4. Patch output_root for isolated calibration directories
            if getattr(self, '_ngen_output_dir_override', None):
                isolated_output_dir = str(self._ngen_output_dir_override.resolve())
                if data.get('output_root') != isolated_output_dir:
                    data['output_root'] = isolated_output_dir
                    changed = True
                    self.logger.debug(f"Patched output_root to {isolated_output_dir}")

            if changed:
                with open(realization_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                self.logger.debug("Patched absolute paths in realization config copy")
        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.warning(f"Failed to patch realization libraries: {e}")

    def _move_ngen_outputs(self, build_dir: Path, output_dir: Path):
        """
        Move ngen output files from build directory to output directory.

        ngen writes outputs to its working directory, so we need to move them
        to the proper experiment output directory.

        Args:
            build_dir: ngen build directory where outputs are written
            output_dir: Target output directory for this experiment
        """
        import shutil

        # Common ngen output patterns
        output_patterns = [
            'cat-*.csv',      # Catchment outputs
            'nex-*.csv',      # Nexus outputs
            '*.parquet',      # Parquet outputs
            'cfe_output_*.txt',  # CFE specific outputs
            'noah_output_*.txt', # Noah specific outputs
        ]

        moved_files = []
        for pattern in output_patterns:
            for file in build_dir.glob(pattern):
                dest = output_dir / file.name
                shutil.move(str(file), str(dest))
                moved_files.append(file.name)

        if moved_files:
            self.logger.debug(f"Moved {len(moved_files)} output files to {output_dir}")
            for f in moved_files[:10]:  # Log first 10
                self.logger.debug(f"  - {f}")
            if len(moved_files) > 10:
                self.logger.debug(f"  ... and {len(moved_files) - 10} more")
        else:
            existing_outputs: list[Any] = []
            for pattern in output_patterns:
                existing_outputs.extend(output_dir.glob(pattern))
            if not existing_outputs:
                self.logger.warning(
                    f"No output files found in {build_dir} or {output_dir}. Check if model ran correctly."
                )

    def _run_troute_routing(self, output_dir: Path) -> bool:
        """
        Run t-route routing on NGEN nexus outputs.

        Args:
            output_dir: Directory containing NGEN outputs (nex-*.csv files)

        Returns:
            True if routing succeeded, False otherwise
        """
        import json

        try:
            # Check if ngen_routing is available
            from ngen_routing.ngen_main import ngen_main
        except ImportError:
            self.logger.debug("T-Route (ngen_routing) not installed. Skipping routing.")
            return False

        # Find t-route config
        troute_config = self.ngen_setup_dir / "troute_config.yaml"
        if not troute_config.exists():
            self.logger.warning(f"T-Route config not found: {troute_config}. Skipping routing.")
            self.logger.info("Run NGEN preprocessing to generate troute_config.yaml")
            return False

        # Check if this is a lumped domain (single nexus) - routing is optional
        nexus_file = self.ngen_setup_dir / "nexus.geojson"
        is_lumped = False
        if nexus_file.exists():
            try:
                with open(nexus_file, 'r', encoding='utf-8') as f:
                    nexus_data = json.load(f)
                num_nexuses = len(nexus_data.get('features', []))
                if num_nexuses == 1:
                    is_lumped = True
                    self.logger.info("Lumped domain detected (single nexus). Nexus output is equivalent to routed flow.")
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.debug(f"Could not parse nexus file for lumped detection: {e}")

        # For lumped domains, we can skip routing and use nexus output directly
        if is_lumped:
            self.logger.info("Skipping t-route for lumped domain - nex-*.csv output is already at the outlet.")
            return True  # Return True since flow is already at outlet

        # Create troute output directory
        troute_output_dir = output_dir / "troute_output"
        troute_output_dir.mkdir(parents=True, exist_ok=True)

        # Build t-route command arguments
        troute_args = [
            "-f", str(troute_config),  # Config file
            "-V", str(output_dir),      # Nexus output folder (NGEN outputs)
        ]

        self.logger.info(f"Running t-route with config: {troute_config}")
        self.logger.debug(f"T-Route args: {troute_args}")

        # Run t-route
        troute_log = output_dir / "troute_log.txt"
        try:
            from contextlib import redirect_stderr, redirect_stdout

            # Capture t-route output to log file
            with open(troute_log, 'w', encoding='utf-8') as log_f:
                with redirect_stdout(log_f), redirect_stderr(log_f):
                    ngen_main(troute_args)

            self.logger.info("T-Route routing completed successfully")
            self.logger.info(f"T-Route log: {troute_log}")
            return True

        except Exception as e:  # noqa: BLE001 — model execution resilience
            self.logger.error(f"T-Route routing failed: {e}")
            if troute_log.exists():
                self.logger.error(f"Check t-route log: {troute_log}")
            return False

    def _run_ngen_docker(self, experiment_id: str = None) -> bool:
        """Wrapper for NGIAB Docker execution implementation."""
        return self._run_ngen_docker_impl(experiment_id)

    def _run_ngen_docker_impl(self, experiment_id: str = None) -> bool:
        """Execute NGIAB Docker workflow via the detailed pipeline helper."""
        return self._run_ngen_docker_pipeline(experiment_id)

    def _run_ngen_docker_pipeline(self, experiment_id: str = None) -> bool:
        """
        Execute NextGen model using NGIAB Docker container.

        This provides T-Route routing out of the box since the container
        includes all required components pre-built.

        Args:
            experiment_id: Optional experiment identifier.

        Returns:
            True if execution succeeded, False otherwise.
        """
        import shutil

        with symfluence_error_handler(
            "NGIAB Docker execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            (
                experiment_id,
                output_dir,
                is_calibration_mode,
                original_ngen_setup_dir,
                base_setup_dir,
                bmi_config_source,
                domain_name,
            ) = self._resolve_ngiab_context(experiment_id)

            ngiab_image = self._get_config_value(
                lambda: None,
                default='awiciroh/ciroh-ngen-image:latest',
                dict_key='NGIAB_IMAGE'
            )
            if not self._ensure_ngiab_image(ngiab_image):
                return False

            ngiab_run_dir, ngiab_config_dir, ngiab_forcings_dir, ngiab_outputs_dir = (
                self._prepare_ngiab_run_directories(output_dir)
            )
            self.logger.info("Preparing NGIAB-compatible directory structure...")

            setup_files = self._prepare_ngiab_inputs(
                base_setup_dir=base_setup_dir,
                bmi_config_source=bmi_config_source,
                original_ngen_setup_dir=original_ngen_setup_dir,
                is_calibration_mode=is_calibration_mode,
                domain_name=domain_name,
                ngiab_config_dir=ngiab_config_dir,
                ngiab_forcings_dir=ngiab_forcings_dir,
            )
            if setup_files is None:
                return False
            gpkg_file, catchment_file = setup_files

            # Copy and configure T-Route files for routing
            # Try NetCDF first (NHDNetwork - most reliable), then GeoPackage/GeoJSON (HYFeaturesNetwork)
            domain_name = self.domain_name
            troute_gpkg_src = base_setup_dir / f"{domain_name}_hydrofabric_troute.gpkg"
            troute_flowlines_src = base_setup_dir / "troute_flowlines.geojson"
            troute_topology_src = base_setup_dir / "troute_topology.nc"
            troute_config_src = base_setup_dir / "troute_config.yaml"

            # Check if topology has required channel geometry columns for T-Route
            topology_complete = False
            use_gpkg = False
            use_geojson = False
            use_netcdf = False

            # First try NetCDF topology with NHDNetwork (most reliable)
            if troute_topology_src.exists() and troute_config_src.exists():
                try:
                    import xarray as xr
                    topo_ds = xr.open_dataset(troute_topology_src, decode_timedelta=False)
                    topo_vars = set(topo_ds.variables.keys())
                    topo_ds.close()

                    required_cols = {'comid', 'to_node', 'length', 'slope', 'n'}
                    extended_cols = {'bw', 'tw', 'twcc', 'ncc', 'musk', 'musx', 'cs'}

                    if not required_cols.issubset(topo_vars):
                        missing = required_cols - topo_vars
                        self.logger.debug(f"NetCDF topology missing required columns: {missing}. Trying GeoPackage.")
                    elif not extended_cols.issubset(topo_vars):
                        self.logger.debug(f"NetCDF topology missing channel geometry columns: {extended_cols - topo_vars}. Trying GeoPackage.")
                    else:
                        topology_complete = True
                        use_netcdf = True
                        self.logger.info("Using NetCDF topology for T-Route NHDNetwork")
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.debug(f"NetCDF topology check failed: {e}. Trying GeoPackage.")

            # Fall back to hydrofabric GeoPackage with HYFeaturesNetwork
            if not topology_complete and troute_gpkg_src.exists() and troute_config_src.exists():
                try:
                    import fiona
                    layers = fiona.listlayers(troute_gpkg_src)
                    # T-Route HYFeaturesNetwork expects 'flowpaths' and 'flowpath_attributes' layers
                    if 'flowpaths' in layers and 'flowpath_attributes' in layers:
                        import geopandas as gpd
                        import pandas as pd
                        flowpaths_gdf = gpd.read_file(troute_gpkg_src, layer='flowpaths')
                        # Read attributes from sqlite table
                        import sqlite3
                        conn = sqlite3.connect(troute_gpkg_src)
                        attrs_df = pd.read_sql("SELECT * FROM flowpath_attributes", conn)
                        conn.close()
                        # Check required columns across both tables
                        all_cols = set(flowpaths_gdf.columns) | set(attrs_df.columns)
                        required_cols = {'id', 'toid', 'lengthkm', 'So', 'n'}
                        if required_cols.issubset(all_cols):
                            topology_complete = True
                            use_gpkg = True
                            self.logger.debug(f"Using hydrofabric GeoPackage for T-Route HYFeaturesNetwork (layers: {layers})")
                    elif 'flowlines' in layers:
                        # Legacy format - single flowlines layer
                        import geopandas as gpd
                        flowlines_gdf = gpd.read_file(troute_gpkg_src, layer='flowlines')
                        required_cols = {'id', 'toid', 'lengthkm', 'So', 'n'}
                        if required_cols.issubset(set(flowlines_gdf.columns)):
                            topology_complete = True
                            use_gpkg = True
                            self.logger.debug("Using hydrofabric GeoPackage (legacy flowlines layer)")
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.debug(f"GeoPackage check failed: {e}. Trying GeoJSON.")

            # Fall back to GeoJSON flowlines
            if not topology_complete and troute_flowlines_src.exists() and troute_config_src.exists():
                try:
                    import geopandas as gpd
                    flowlines_gdf = gpd.read_file(troute_flowlines_src)
                    required_cols = {'id', 'toid', 'lengthkm', 'So', 'n'}
                    available_cols = set(flowlines_gdf.columns)

                    if required_cols.issubset(available_cols):
                        topology_complete = True
                        use_geojson = True
                        self.logger.debug("Using GeoJSON flowlines for T-Route HYFeaturesNetwork")
                    else:
                        missing = required_cols - available_cols
                        self.logger.warning(f"T-Route flowlines missing required columns: {missing}. Skipping routing.")
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Failed to read T-Route flowlines: {e}. Skipping routing.")

            if not topology_complete and not troute_flowlines_src.exists() and not troute_topology_src.exists():
                self.logger.warning(f"T-Route files not found in {base_setup_dir}. Routing will be disabled.")

            if topology_complete:
                self.logger.info("Configuring T-Route routing for NGIAB...")

                # Copy topology file (GeoPackage, GeoJSON, or NetCDF)
                import yaml
                container_base = "/ngen/ngen/data"

                if use_gpkg:
                    # Copy hydrofabric GeoPackage
                    topology_dest = ngiab_config_dir / f"{domain_name}_hydrofabric_troute.gpkg"
                    shutil.copy(troute_gpkg_src, topology_dest)
                    geo_file_path = f"{container_base}/config/{domain_name}_hydrofabric_troute.gpkg"
                    network_type = 'HYFeaturesNetwork'
                    # HYFeaturesNetwork column mappings for GeoPackage
                    columns = {
                        'key': 'id',              # Segment ID (wb-XX format)
                        'downstream': 'toid',      # Downstream segment ID
                        'dx': 'lengthkm',          # Segment length [km]
                        'n': 'n',                  # Manning's n
                        's0': 'So',                # Slope [m/m]
                        'bw': 'BtmWdth',           # Bottom width [m]
                        'tw': 'TopWdth',           # Top width [m]
                        'twcc': 'TopWdthCC',       # Top width compound channel [m]
                        'ncc': 'nCC',              # Manning's n compound channel
                        'musk': 'MusK',            # Muskingum K
                        'musx': 'MusX',            # Muskingum X
                        'cs': 'Cs',                # Cross-section area [m²]
                        'alt': 'alt',              # Altitude
                    }
                elif use_geojson:
                    # Copy GeoJSON flowlines
                    topology_dest = ngiab_config_dir / "troute_flowlines.geojson"
                    shutil.copy(troute_flowlines_src, topology_dest)
                    geo_file_path = f"{container_base}/config/troute_flowlines.geojson"
                    network_type = 'HYFeaturesNetwork'
                    # HYFeaturesNetwork column mappings
                    columns = {
                        'key': 'id',              # Segment ID (wb-XX format)
                        'downstream': 'toid',      # Downstream segment ID
                        'dx': 'lengthkm',          # Segment length [km]
                        'n': 'n',                  # Manning's n
                        's0': 'So',                # Slope [m/m]
                        'bw': 'BtmWdth',           # Bottom width [m]
                        'tw': 'TopWdth',           # Top width [m]
                        'twcc': 'TopWdthCC',       # Top width compound channel [m]
                        'ncc': 'nCC',              # Manning's n compound channel
                        'musk': 'MusK',            # Muskingum K
                        'musx': 'MusX',            # Muskingum X
                        'cs': 'Cs',                # Cross-section area [m²]
                        'alt': 'alt',              # Altitude
                    }
                elif use_netcdf:
                    # Copy NetCDF topology
                    topology_dest = ngiab_config_dir / "troute_topology.nc"
                    shutil.copy(troute_topology_src, topology_dest)
                    geo_file_path = f"{container_base}/config/troute_topology.nc"
                    network_type = 'NHDNetwork'
                    # NHDNetwork column mappings (must include ALL columns to override defaults)
                    columns = {
                        'key': 'comid',           # Segment ID
                        'downstream': 'to_node',   # Downstream segment ID
                        'dx': 'length',            # Segment length [m]
                        'n': 'n',                  # Manning's n
                        's0': 'slope',             # Slope [m/m]
                        'bw': 'bw',                # Bottom width [m]
                        'tw': 'tw',                # Top width [m]
                        'twcc': 'twcc',            # Top width compound channel [m]
                        'ncc': 'ncc',              # Manning's n compound channel
                        'musk': 'musk',            # Muskingum K
                        'musx': 'musx',            # Muskingum X
                        'cs': 'ChSlp',             # Channel side slope (not cross-section)
                        'alt': 'alt',              # Altitude
                        'waterbody': 'waterbody',  # Waterbody ID (-9999 for none)
                        'gages': 'gages',          # Gage IDs (empty string for none)
                    }
                else:
                    self.logger.warning("No valid topology format selected. Skipping T-Route.")
                    topology_complete = False

            if topology_complete:
                # Load and modify T-Route config
                with open(troute_config_src, 'r', encoding='utf-8') as f:
                    troute_config = yaml.safe_load(f)

                # Update paths for container
                if 'network_topology_parameters' not in troute_config:
                    troute_config['network_topology_parameters'] = {}
                ntp = troute_config['network_topology_parameters']

                if 'supernetwork_parameters' not in ntp:
                    ntp['supernetwork_parameters'] = {}
                snp = ntp['supernetwork_parameters']

                # Set geo_file_path and columns
                snp['geo_file_path'] = geo_file_path
                snp['columns'] = columns

                # Add default values
                snp['waterbody_null_code'] = -9999
                snp['terminal_code'] = 0
                # Set both network_type (for class selection) and geo_file_type (for file format)
                snp['network_type'] = network_type
                snp['geo_file_type'] = network_type

                # Default values for missing columns (will be used if not in topology)
                if 'synthetic_wb_segments' not in snp:
                    snp['synthetic_wb_segments'] = []

                # Add waterbody_parameters (required by HYFeaturesNetwork even if no waterbodies)
                if 'network_topology_parameters' in troute_config:
                    if 'waterbody_parameters' not in troute_config['network_topology_parameters']:
                        troute_config['network_topology_parameters']['waterbody_parameters'] = {
                            'break_network_at_waterbodies': False,
                            'level_pool': {
                                'level_pool_waterbody_parameter_file_path': None,
                            },
                        }

                # Add preprocessing_parameters (required even if not using preprocessed data)
                if 'network_topology_parameters' in troute_config:
                    if 'preprocessing_parameters' not in troute_config['network_topology_parameters']:
                        troute_config['network_topology_parameters']['preprocessing_parameters'] = {
                            'use_preprocessed_data': False,
                        }

                # Use Muskingum-Cunge routing (simpler, doesn't need channel geometry)
                if 'compute_parameters' not in troute_config:
                    troute_config['compute_parameters'] = {}
                cp = troute_config['compute_parameters']

                # Set routing method to Muskingum-Cunge which is more forgiving
                cp['compute_kernel'] = 'V02-structured'
                cp['assume_short_ts'] = True
                cp['subnetwork_target_size'] = 10000
                cp['parallel_compute_method'] = 'serial'
                cp['cpu_pool'] = 1

                if 'forcing_parameters' not in cp:
                    cp['forcing_parameters'] = {}
                cp['forcing_parameters']['qlat_input_folder'] = f"{container_base}/outputs/"
                cp['forcing_parameters']['qlat_file_pattern_filter'] = 'nex-*_output.csv'

                # Add hybrid_parameters (required even if not using hybrid routing)
                if 'hybrid_parameters' not in cp:
                    cp['hybrid_parameters'] = {
                        'run_hybrid_routing': False,
                    }

                # Fix output parameters
                if 'output_parameters' not in troute_config:
                    troute_config['output_parameters'] = {}
                op = troute_config['output_parameters']

                if 'stream_output' not in op:
                    op['stream_output'] = {}
                so = op['stream_output']
                so['stream_output_directory'] = f"{container_base}/outputs/"
                so['stream_output_time'] = 60  # Minutes (must be >= internal_frequency)
                so['stream_output_type'] = '.csv'
                so['stream_output_internal_frequency'] = 60  # Minutes

                # Write container-compatible config
                ngiab_troute_config = ngiab_config_dir / "troute_config.yaml"
                with open(ngiab_troute_config, 'w', encoding='utf-8') as f:
                    yaml.dump(troute_config, f, default_flow_style=False, sort_keys=False)  # type: ignore[call-overload]

                self.logger.debug(f"Created NGIAB T-Route config: {ngiab_troute_config}")

                # Update realization to include routing section
                self._add_routing_to_realization(ngiab_config_dir / "realization.json")
                self.logger.info("T-Route routing enabled in realization")

            # Run NGIAB Docker container
            # We run ngen-serial directly since SYMFLUENCE uses separate catchment/nexus files
            # The auto mode expects a combined hydrofabric GeoPackage
            self.logger.info("Running NGIAB Docker container...")

            # Determine hydrofabric files in container paths
            container_base = "/ngen/ngen/data"
            if gpkg_file.exists():
                container_catchment = f"{container_base}/config/{gpkg_file.name}"
            else:
                container_catchment = f"{container_base}/config/{catchment_file.name}"
            container_nexus = f"{container_base}/config/nexus.geojson"
            container_realization = f"{container_base}/config/realization.json"

            # Run ngen-serial directly (bypasses interactive menu)
            # Override entrypoint since HelloNGEN.sh expects different arguments
            docker_cmd = [
                'docker', 'run', '--rm',
                '-v', f'{ngiab_run_dir}:/ngen/ngen/data',
                '-w', '/ngen/ngen/data',  # Set working directory
                '--entrypoint', '/dmod/bin/ngen-serial',  # Override entrypoint
                ngiab_image,
                container_catchment,
                'all',
                container_nexus,
                'all',
                container_realization
            ]

            self.logger.debug(f"Docker command: {' '.join(docker_cmd)}")

            docker_log = output_dir / "ngiab_docker_log.txt"
            try:
                with open(docker_log, 'w', encoding='utf-8') as log_f:
                    result = subprocess.run(  # nosec B603
                        docker_cmd,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                        timeout=3600  # 1 hour timeout
                    )

                if result.returncode != 0:
                    self.logger.error(f"NGIAB Docker execution failed. Check log: {docker_log}")
                    return False

                self.logger.info("NGIAB Docker execution completed successfully")

                # Copy outputs from ngiab_outputs_dir to output_dir
                for output_file in ngiab_outputs_dir.glob('*'):
                    if output_file.is_file():
                        shutil.copy(output_file, output_dir / output_file.name)

                # Count outputs
                output_count = len(list(output_dir.glob('*.csv'))) + len(list(output_dir.glob('*.parquet')))
                self.logger.info(f"Generated {output_count} output files")

                return True

            except subprocess.TimeoutExpired:
                self.logger.error("NGIAB Docker execution timed out")
                return False
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.error(f"NGIAB Docker execution failed: {e}")
                return False

    def _resolve_ngiab_context(
        self,
        experiment_id: Optional[str],
    ) -> Tuple[str, Path, bool, Path, Path, Path, str]:
        if experiment_id is None:
            experiment_id = self.experiment_id

        if self._ngen_output_dir_override is not None:
            output_dir = self._ngen_output_dir_override
        else:
            output_dir = self.get_experiment_output_dir(experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_calibration_mode = self._ngen_settings_dir_override is not None
        if is_calibration_mode:
            original_ngen_setup_dir = self.project_dir / "settings" / "NGEN"
            isolated_settings_dir = self._ngen_settings_dir_override
            if isolated_settings_dir is None:
                isolated_settings_dir = original_ngen_setup_dir
            self.logger.debug(
                f"Calibration mode: isolated={isolated_settings_dir}, original={original_ngen_setup_dir}"
            )
        else:
            original_ngen_setup_dir = cast(Path, self.ngen_setup_dir)
            isolated_settings_dir = None

        base_setup_dir = original_ngen_setup_dir
        bmi_config_source: Path = isolated_settings_dir if is_calibration_mode else original_ngen_setup_dir
        domain_name = str(self.domain_name)
        return (
            experiment_id,
            output_dir,
            is_calibration_mode,
            original_ngen_setup_dir,
            base_setup_dir,
            bmi_config_source,
            domain_name,
        )

    def _ensure_ngiab_image(self, ngiab_image: str) -> bool:
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)  # nosec B603 B607
            if result.returncode != 0:
                self.logger.error("Docker is not available")
                return False
        except FileNotFoundError:
            self.logger.error("Docker is not installed")
            return False

        self.logger.info(f"Checking for NGIAB Docker image: {ngiab_image}")
        pull_result = subprocess.run(  # nosec B603 B607
            ['docker', 'image', 'inspect', ngiab_image],
            capture_output=True
        )
        if pull_result.returncode == 0:
            return True

        self.logger.info(f"Pulling NGIAB Docker image: {ngiab_image}")
        pull_result = subprocess.run(  # nosec B603 B607
            ['docker', 'pull', ngiab_image],
            capture_output=True,
            text=True
        )
        if pull_result.returncode != 0:
            self.logger.error(f"Failed to pull Docker image: {pull_result.stderr}")  # type: ignore[str-bytes-safe]
            return False
        return True

    def _prepare_ngiab_run_directories(
        self,
        output_dir: Path,
    ) -> Tuple[Path, Path, Path, Path]:
        import shutil

        ngiab_run_dir = output_dir / "ngiab_run"
        ngiab_config_dir = ngiab_run_dir / "config"
        ngiab_forcings_dir = ngiab_run_dir / "forcings"
        ngiab_outputs_dir = ngiab_run_dir / "outputs"

        if ngiab_run_dir.exists():
            shutil.rmtree(ngiab_run_dir)
        ngiab_config_dir.mkdir(parents=True, exist_ok=True)
        ngiab_forcings_dir.mkdir(parents=True, exist_ok=True)
        ngiab_outputs_dir.mkdir(parents=True, exist_ok=True)
        return ngiab_run_dir, ngiab_config_dir, ngiab_forcings_dir, ngiab_outputs_dir

    def _prepare_ngiab_inputs(
        self,
        base_setup_dir: Path,
        bmi_config_source: Path,
        original_ngen_setup_dir: Path,
        is_calibration_mode: bool,
        domain_name: str,
        ngiab_config_dir: Path,
        ngiab_forcings_dir: Path,
    ) -> Optional[Tuple[Path, Path]]:
        import shutil

        realization_file = base_setup_dir / "realization_config.json"
        if not realization_file.exists():
            self.logger.error(f"Realization file not found: {realization_file}")
            return None
        self._create_ngiab_realization(realization_file, ngiab_config_dir, domain_name)

        catchment_file = base_setup_dir / f"{domain_name}_catchments.geojson"
        nexus_file = base_setup_dir / "nexus.geojson"
        gpkg_file = base_setup_dir / f"{domain_name}_catchments.gpkg"

        if gpkg_file.exists():
            shutil.copy(gpkg_file, ngiab_config_dir / gpkg_file.name)
            self.logger.debug(f"Copied hydrofabric: {gpkg_file.name}")
        elif catchment_file.exists():
            shutil.copy(catchment_file, ngiab_config_dir / catchment_file.name)
            self.logger.debug(f"Copied catchments: {catchment_file.name}")

        if nexus_file.exists():
            shutil.copy(nexus_file, ngiab_config_dir / nexus_file.name)
            self.logger.debug(f"Copied nexus: {nexus_file.name}")

        cat_config_dir = ngiab_config_dir / "cat-config"
        cat_config_dir.mkdir(exist_ok=True)
        module_mapping = {
            'CFE': 'CFE',
            'PET': 'PET',
            'NOAH': 'NOAH-OWP-M',
            'SLOTH': 'SLOTH'
        }
        for src_name, dst_name in module_mapping.items():
            src_dir = bmi_config_source / src_name
            if src_dir.exists():
                dst_dir = cat_config_dir / dst_name
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                self.logger.debug(f"Copied {src_name} configs to {dst_name}")

        if is_calibration_mode:
            noah_dst_dir = cat_config_dir / 'NOAH-OWP-M'
            noah_params_src = original_ngen_setup_dir / 'NOAH' / 'parameters'
            noah_params_dst = noah_dst_dir / 'parameters'
            if noah_params_src.exists() and not noah_params_dst.exists():
                shutil.copytree(noah_params_src, noah_params_dst)
                self.logger.debug("Copied NOAH parameters directory from original setup")

        self._patch_ngiab_noah_configs(cat_config_dir / 'NOAH-OWP-M')

        forcing_src = self.project_forcing_dir / 'NGEN_input' / 'csv'
        if forcing_src.exists():
            for forcing_file in forcing_src.glob('*.csv'):
                shutil.copy(forcing_file, ngiab_forcings_dir / forcing_file.name)
            self.logger.debug(f"Copied forcing files to {ngiab_forcings_dir}")
        else:
            self.logger.warning(f"Forcing directory not found: {forcing_src}")

        return gpkg_file, catchment_file

    def _patch_ngiab_noah_configs(self, noah_config_dir: Path) -> None:
        if not noah_config_dir.exists():
            return

        import re
        container_param_dir = "/ngen/ngen/data/config/cat-config/NOAH-OWP-M/parameters/"
        for config_file in noah_config_dir.glob('*.input'):
            try:
                content = config_file.read_text(encoding='utf-8')
                content = re.sub(
                    r'parameter_dir\s*=\s*"[^"]*"',
                    f'parameter_dir      = "{container_param_dir}"',
                    content
                )
                config_file.write_text(content, encoding='utf-8')
            except Exception as e:  # noqa: BLE001 — model execution resilience
                self.logger.warning(f"Failed to patch NOAH config {config_file.name}: {e}")
        self.logger.debug("Patched NOAH config files with container parameter paths")

    def _create_ngiab_realization(self, src_realization: Path, config_dir: Path, domain_name: str):
        """
        Create NGIAB-compatible realization file with container paths.

        NGIAB expects paths relative to /ngen/ngen/data/ inside the container.
        """
        import json

        with open(src_realization, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # NGIAB container paths
        container_base = "/ngen/ngen/data"

        # Patch global forcing path
        if 'global' in data and 'forcing' in data['global']:
            forcing = data['global']['forcing']
            if 'path' in forcing:
                forcing['path'] = f"{container_base}/forcings/"
            if 'file_pattern' in forcing:
                # Keep the pattern but ensure it doesn't have absolute paths
                pattern = Path(forcing['file_pattern']).name
                forcing['file_pattern'] = pattern

        # Patch formulation library and init_config paths
        if 'global' in data and 'formulations' in data['global']:
            for formulation in data['global']['formulations']:
                if 'params' in formulation and 'modules' in formulation['params']:
                    for module in formulation['params']['modules']:
                        mod_params = module.get('params', {})

                        # NGIAB has libraries at standard locations
                        if 'library_file' in mod_params:
                            lib_name = Path(mod_params['library_file']).name
                            # NGIAB container library paths
                            if 'pet' in lib_name.lower():
                                mod_params['library_file'] = '/dmod/shared_libs/libpetbmi.so'
                            elif 'cfe' in lib_name.lower():
                                mod_params['library_file'] = '/dmod/shared_libs/libcfebmi.so'
                            elif 'sloth' in lib_name.lower():
                                mod_params['library_file'] = '/dmod/shared_libs/libslothmodel.so'
                            elif 'surface' in lib_name.lower() or 'noah' in lib_name.lower():
                                mod_params['library_file'] = '/dmod/shared_libs/libsurfacebmi.so'

                        # Patch init_config paths
                        if 'init_config' in mod_params:
                            old_path = mod_params['init_config']
                            if old_path and old_path != "/dev/null":
                                # Extract module type and filename
                                mod_type = None
                                if 'PET' in str(mod_params.get('model_type_name', '')).upper() or 'pet' in old_path.lower():
                                    mod_type = 'PET'
                                elif 'CFE' in str(mod_params.get('model_type_name', '')).upper() or 'cfe' in old_path.lower():
                                    mod_type = 'CFE'
                                elif 'NOAH' in str(mod_params.get('model_type_name', '')).upper() or 'noah' in old_path.lower():
                                    mod_type = 'NOAH-OWP-M'
                                elif 'SLOTH' in str(mod_params.get('model_type_name', '')).upper() or 'sloth' in old_path.lower():
                                    mod_type = 'SLOTH'

                                if mod_type:
                                    filename = Path(old_path).name
                                    mod_params['init_config'] = f"{container_base}/config/cat-config/{mod_type}/{filename}"

        # Set output root
        data['output_root'] = f"{container_base}/outputs/"

        # Write patched realization
        dst_realization = config_dir / "realization.json"
        with open(dst_realization, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        self.logger.debug(f"Created NGIAB realization: {dst_realization}")

    def _add_routing_to_realization(self, realization_path: Path):
        """
        Add T-Route routing configuration to an existing NGIAB realization file.

        Args:
            realization_path: Path to the realization.json file to update.
        """
        import json

        container_base = "/ngen/ngen/data"

        with open(realization_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Add routing section pointing to T-Route config
        data['routing'] = {
            "t_route_config_file_with_path": f"{container_base}/config/troute_config.yaml"
        }

        with open(realization_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        self.logger.debug(f"Added routing section to realization: {realization_path}")
