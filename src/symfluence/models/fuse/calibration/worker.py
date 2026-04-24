# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FUSE Worker

Worker implementation for FUSE model optimization.
Thin orchestrator delegating to focused modules for parameter application,
model execution, metrics calculation, and file manager updates.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.evaluation.utilities import StreamflowMetrics
from symfluence.models.fuse.calibration.file_manager import (
    update_fuse_file_manager,
)
from symfluence.models.fuse.calibration.metrics_calculation import (
    align_and_filter,
    compute_metrics,
    find_simulation_output,
    load_observations,
    read_fuse_streamflow,
    read_routed_streamflow,
)
from symfluence.models.fuse.calibration.model_execution import (
    detect_fuse_run_mode,
    execute_fuse,
    handle_fuse_output,
    log_execution_directory,
    prepare_input_files,
    resolve_fuse_paths,
    validate_fuse_inputs,
)
from symfluence.models.fuse.calibration.multi_gauge_metrics import MultiGaugeMetrics
from symfluence.models.fuse.calibration.parameter_application import (
    apply_regionalization,
    update_constraints_file,
    update_para_def_nc,
)
from symfluence.models.fuse.utilities import FuseToMizurouteConverter
from symfluence.models.utilities.routing_decider import RoutingDecider
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask

# Suppress xarray FutureWarning about timedelta64 decoding
warnings.filterwarnings('ignore',
                       message='.*decode_timedelta.*',
                       category=FutureWarning,
                       module='xarray.*')

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('FUSE')
class FUSEWorker(BaseWorker):
    """
    Worker for FUSE model calibration.

    Handles parameter application to NetCDF files, FUSE execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(config, logger)
        self._consecutive_output_failures = 0

    # Shared utilities
    _routing_decider = RoutingDecider()
    _streamflow_metrics = StreamflowMetrics()
    _format_converter = FuseToMizurouteConverter()

    def needs_routing(self, config: Dict[str, Any], settings_dir: Optional[Path] = None) -> bool:
        """Determine if routing (mizuRoute) is needed for FUSE."""
        return self._routing_decider.needs_routing(config, 'FUSE', settings_dir)

    # =========================================================================
    # Parameter Application
    # =========================================================================

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to FUSE constraints file AND para_def.nc.

        In run_pre mode (the default for calibration), FUSE reads parameters
        from para_def.nc. We update BOTH the constraints file (for consistency)
        AND the para_def.nc file (which run_pre reads directly).
        """
        try:
            config = kwargs.get('config', self.config)

            self.logger.debug(f"APPLY_PARAMS: Applying {len(params)} parameters to {settings_dir}")
            for p, v in list(params.items())[:5]:
                self.logger.debug(f"  PARAM: {p} = {v:.4f}")

            # Resolve FUSE settings directory
            if settings_dir.name == 'FUSE':
                fuse_settings_dir = settings_dir
            elif (settings_dir / 'FUSE').exists():
                fuse_settings_dir = settings_dir / 'FUSE'
            else:
                fuse_settings_dir = settings_dir

            # Determine regionalization mode
            regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped') if config else 'lumped'
            if config and config.get('USE_TRANSFER_FUNCTIONS', False):
                regionalization_method = 'transfer_function'

            # Update constraints file (skip in regionalization mode)
            constraints_file = fuse_settings_dir / 'fuse_zConstraints_snow.txt'
            if regionalization_method != 'lumped':
                self.logger.debug(
                    f"Regionalization mode ({regionalization_method}): "
                    f"skipping constraints file (using para_def.nc)"
                )
            elif constraints_file.exists():
                params_updated_txt = update_constraints_file(constraints_file, params, self.logger)
                if params_updated_txt:
                    sample_params = list(params.items())[:3]
                    self.logger.debug(f"APPLY: Updated {len(params_updated_txt)} params in {constraints_file.name}, sample: {sample_params}")
                else:
                    self.logger.warning("APPLY_PARAMS: No params updated in constraints file")
            else:
                self.logger.error(
                    f"APPLY_PARAMS: Constraints file not found at {constraints_file}. "
                    f"FUSE calibration will not work!"
                )
                return False

            # Update para_def.nc
            if config:
                self._update_para_def(fuse_settings_dir, params, config, regionalization_method)

            return True

        except (FileNotFoundError, OSError) as e:
            self.logger.error(f"File error applying FUSE parameters: {e}")
            return False
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error applying FUSE parameters: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def _update_para_def(
        self,
        fuse_settings_dir: Path,
        params: Dict[str, float],
        config: Dict[str, Any],
        regionalization_method: str
    ) -> None:
        """Update para_def.nc file using appropriate strategy."""
        domain_name = config.get('DOMAIN_NAME', '')
        experiment_id = config.get('EXPERIMENT_ID', 'run_1')
        fuse_id = config.get('FUSE_FILE_ID', experiment_id)
        para_def_path = fuse_settings_dir / f"{domain_name}_{fuse_id}_para_def.nc"

        if regionalization_method != 'lumped' and para_def_path.exists():
            params_updated_nc = apply_regionalization(
                para_def_path, params, config, self.logger
            )
            if params_updated_nc:
                self.logger.debug(
                    f"APPLY: Updated {len(params_updated_nc)} distributed params "
                    f"via {regionalization_method} regionalization"
                )
            else:
                self.logger.warning(f"APPLY_PARAMS: {regionalization_method} update failed")
        elif para_def_path.exists():
            params_updated_nc = update_para_def_nc(para_def_path, params, self.logger)
            if params_updated_nc:
                self.logger.debug(f"APPLY: Updated {len(params_updated_nc)} params in {para_def_path.name}")
            else:
                self.logger.warning("APPLY_PARAMS: No params updated in para_def.nc")
        else:
            self.logger.warning(f"APPLY_PARAMS: para_def.nc not found at {para_def_path}")

    # =========================================================================
    # Model Execution
    # =========================================================================

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """Run FUSE model."""
        try:
            import subprocess

            mode = detect_fuse_run_mode(config, kwargs, self.logger)
            fuse_exe, filemanager_path, execution_cwd = resolve_fuse_paths(
                config, settings_dir, self.logger
            )

            if not fuse_exe.exists():
                self.logger.error(f"FUSE executable not found: {fuse_exe}")
                return False
            if not filemanager_path.exists():
                self.logger.error(f"FUSE file manager not found: {filemanager_path}")
                return False

            # Setup output directory
            fuse_output_dir = kwargs.get('sim_dir', output_dir)
            if fuse_output_dir:
                Path(fuse_output_dir).mkdir(parents=True, exist_ok=True)

            # Prepare input files (symlinks, copies)
            result = prepare_input_files(config, execution_cwd, self.logger)
            if result is None:
                return False
            fuse_run_id, actual_decisions_file = result

            # Pre-flight validation: catch broken symlinks/missing files
            # BEFORE running FUSE (prevents silent Fortran crashes)
            if not validate_fuse_inputs(
                execution_cwd, fuse_run_id, config, self.logger
            ):
                return False

            # Update file manager
            if not update_fuse_file_manager(
                filemanager_path, execution_cwd, fuse_output_dir,
                config=config, log=self.logger,
                use_local_input=True, decisions_file=actual_decisions_file
            ):
                return False

            log_execution_directory(execution_cwd, self.logger)

            # Execute FUSE
            result = execute_fuse(
                fuse_exe, filemanager_path, execution_cwd,
                fuse_run_id, mode, config, self.logger
            )
            if result is None:
                return False

            # Handle output (move, validate)
            final_output_path = handle_fuse_output(
                execution_cwd, fuse_output_dir, fuse_run_id,
                mode, config, result, self.logger
            )
            if final_output_path is None:
                return False

            # Run routing if needed
            return self._handle_routing(
                config, settings_dir, fuse_output_dir, execution_cwd,
                mode, kwargs
            )

        except subprocess.TimeoutExpired:
            self.logger.error("FUSE execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for FUSE: {e}")
            return False
        except (OSError, IOError) as e:
            self.logger.error(f"I/O error running FUSE: {e}")
            return False
        except (subprocess.SubprocessError, RuntimeError) as e:
            self.logger.error(f"Error running FUSE: {e}")
            return False

    def _handle_routing(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        fuse_output_dir: Path,
        execution_cwd: Path,
        mode: str,
        kwargs: Dict[str, Any]
    ) -> bool:
        """Handle post-FUSE routing if needed."""
        needs_routing_check = self.needs_routing(config, settings_dir=settings_dir)
        self.logger.debug(f"Routing check: needs_routing={needs_routing_check}, settings_dir={settings_dir}")

        if not needs_routing_check:
            return True

        self.logger.debug("Running mizuRoute for FUSE output")
        proc_id = kwargs.get('proc_id', 0)

        sim_dir = kwargs.get('sim_dir')
        if sim_dir:
            mizuroute_dir = Path(sim_dir).parent / 'mizuRoute'
        else:
            mizuroute_dir = Path(fuse_output_dir).parent / 'mizuRoute'
        mizuroute_dir.mkdir(parents=True, exist_ok=True)

        # Clean stale output
        stale_files = list(mizuroute_dir.glob("proc_*.nc")) + list(mizuroute_dir.glob("*.h.*.nc"))
        if stale_files:
            self.logger.debug(f"Cleaning {len(stale_files)} stale mizuRoute output file(s)")
            for stale_f in stale_files:
                try:
                    stale_f.unlink()
                except OSError:
                    pass

        # Convert FUSE output to mizuRoute format
        if not self._convert_fuse_to_mizuroute_format(
            fuse_output_dir, config, execution_cwd, proc_id=proc_id
        ):
            self.logger.error("Failed to convert FUSE output to mizuRoute format")
            return False

        # Run mizuRoute
        keys_to_remove = {'proc_id', 'mizuroute_dir', 'settings_dir'}
        kwargs_filtered = {k: v for k, v in kwargs.items() if k not in keys_to_remove}
        if not self._run_mizuroute_for_fuse(
            config, fuse_output_dir, mizuroute_dir,
            settings_dir=settings_dir, proc_id=proc_id, **kwargs_filtered
        ):
            if mode == 'run_pre':
                self.logger.error("Routing failed during calibration — returning failure")
                return False
            else:
                self.logger.warning("Routing failed, but FUSE succeeded (non-calibration mode)")

        return True

    # =========================================================================
    # Metrics Calculation
    # =========================================================================

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate metrics from FUSE output."""
        try:
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Load observations
            observed = load_observations(config, project_dir, self.logger)
            if observed is None:
                return {'kge': self.penalty_score}

            # Find simulation output
            mizuroute_dir = kwargs.get('mizuroute_dir')
            proc_id = kwargs.get('proc_id', 0)
            sim_dir = kwargs.get('sim_dir')

            sim_file_path, use_routed = find_simulation_output(
                config, output_dir, self.logger,
                mizuroute_dir=mizuroute_dir, proc_id=proc_id, sim_dir=sim_dir
            )
            if sim_file_path is None:
                return {'kge': self.penalty_score}

            # Check multi-gauge mode
            multi_gauge_enabled = config.get('MULTI_GAUGE_CALIBRATION', False)
            if multi_gauge_enabled and use_routed:
                kwargs_clean = {k: v for k, v in kwargs.items() if k != 'project_dir'}
                return self._calculate_multi_gauge_metrics(
                    config=config,
                    mizuroute_output_path=sim_file_path,
                    project_dir=project_dir,
                    **kwargs_clean
                )

            # Read simulated streamflow
            if use_routed:
                simulated = read_routed_streamflow(sim_file_path, config, self.logger)
            else:
                simulated = read_fuse_streamflow(
                    sim_file_path, config, project_dir,
                    self._streamflow_metrics, self.logger
                )

            if simulated is None:
                self._consecutive_output_failures += 1
                if self._consecutive_output_failures >= 5:
                    self.logger.error(
                        f"FUSE has failed to produce readable output for "
                        f"{self._consecutive_output_failures} consecutive iterations. "
                        f"This usually means FUSE is crashing silently on every parameter set. "
                        f"Check: (1) FUSE executable is compatible with your system, "
                        f"(2) input files are valid, (3) run FUSE manually to see Fortran errors."
                    )
                return {'kge': self.penalty_score}

            # Reset consecutive failure counter on success
            self._consecutive_output_failures = 0

            # Align and filter
            obs_values, sim_values = align_and_filter(observed, simulated, config, self.logger)
            if len(obs_values) == 0:
                return {'kge': self.penalty_score}

            # Compute metrics
            return compute_metrics(obs_values, sim_values, config,
                                   self._streamflow_metrics, self.logger)

        except FileNotFoundError as e:
            self.logger.error(f"Output or observation file not found: {e}")
            return {'kge': self.penalty_score}
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"Data error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}
        except (ImportError, OSError) as e:
            self.logger.error(f"Error calculating FUSE metrics: {e}")
            return {'kge': self.penalty_score}

    def _get_catchment_area(self, config: Dict[str, Any], project_dir: Path) -> float:
        """Get catchment area for FUSE unit conversion."""
        domain_name = config.get('DOMAIN_NAME')
        return self._streamflow_metrics.get_catchment_area(config, project_dir, domain_name)

    # =========================================================================
    # Multi-Gauge Metrics (delegates to MultiGaugeMetrics)
    # =========================================================================

    def _calculate_multi_gauge_metrics(
        self,
        config: Dict[str, Any],
        mizuroute_output_path: Path,
        project_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate performance metrics across multiple stream gauges."""
        try:
            gauge_mapping_path = config.get('GAUGE_SEGMENT_MAPPING')
            obs_dir = config.get('MULTI_GAUGE_OBS_DIR')
            gauge_ids = config.get('MULTI_GAUGE_IDS')
            exclude_ids = config.get('MULTI_GAUGE_EXCLUDE_IDS', [])
            aggregation = config.get('MULTI_GAUGE_AGGREGATION', 'mean')
            min_gauges = config.get('MULTI_GAUGE_MIN_GAUGES', 5)

            if not gauge_mapping_path:
                self.logger.error("GAUGE_SEGMENT_MAPPING not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}
            if not obs_dir:
                self.logger.error("MULTI_GAUGE_OBS_DIR not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}

            gauge_mapping_path = Path(gauge_mapping_path)
            obs_dir = Path(obs_dir)

            # Auto-download LaMAH-Ice daily streamflow if the user pointed
            # at a LaMAH-Ice D_gauges directory that isn't there yet.
            if 'D_gauges' in obs_dir.parts and not obs_dir.exists():
                try:
                    from symfluence.data.observation.handlers.lamah_ice import (
                        ensure_lamah_ice_streamflow,
                    )
                    # Trim to the dataset root (parent of D_gauges/...)
                    lamah_root = obs_dir
                    while lamah_root.name != 'D_gauges' and lamah_root.parent != lamah_root:
                        lamah_root = lamah_root.parent
                    lamah_root = lamah_root.parent
                    ensure_lamah_ice_streamflow(lamah_root, self.logger)
                except Exception as exc:  # noqa: BLE001 — let the existence check below surface the real failure
                    self.logger.warning(
                        f"LaMAH-Ice auto-download skipped: {exc}"
                    )

            # Get topology file
            topology_path = self._find_topology_path(kwargs.get('settings_dir'))

            # Get calibration period
            start_date, end_date = self._parse_calibration_period(config)

            multi_gauge = MultiGaugeMetrics(
                gauge_segment_mapping_path=gauge_mapping_path,
                obs_data_dir=obs_dir,
                logger=self.logger
            )

            # Apply exclusion list
            if gauge_ids is None and exclude_ids:
                all_gauge_ids = multi_gauge.gauge_mapping['id'].tolist()
                gauge_ids = [gid for gid in all_gauge_ids if gid not in exclude_ids]
                self.logger.debug(f"Excluded {len(exclude_ids)} gauges, using {len(gauge_ids)} gauges")

            # Build quality filter config
            filter_config = {}
            for key, cfg_key in [('max_distance', 'MULTI_GAUGE_MAX_DISTANCE'),
                                  ('min_obs_cv', 'MULTI_GAUGE_MIN_OBS_CV'),
                                  ('min_specific_q', 'MULTI_GAUGE_MIN_SPECIFIC_Q')]:
                val = config.get(cfg_key)
                if val is not None:
                    filter_config[key] = float(val)

            min_overlap = int(config.get('MULTI_GAUGE_MIN_OVERLAP_DAYS', 10))
            kge_floor = config.get('MULTI_GAUGE_KGE_FLOOR')
            if kge_floor is not None:
                kge_floor = float(kge_floor)

            results = multi_gauge.calculate_multi_gauge_metrics(
                mizuroute_output_path=mizuroute_output_path,
                gauge_ids=gauge_ids,
                start_date=start_date,
                end_date=end_date,
                topology_path=topology_path,
                min_gauges=min_gauges,
                aggregation=aggregation,
                filter_config=filter_config if filter_config else None,
                min_overlap_days=min_overlap,
                kge_floor=kge_floor
            )

            self.logger.debug(
                f"Multi-gauge calibration: KGE={results['kge']:.4f} "
                f"({results['n_valid_gauges']}/{results['n_total_gauges']} valid gauges)"
            )

            return {
                'kge': results['kge'],
                'kge_std': results.get('kge_std', 0.0),
                'kge_min': results.get('kge_min', results['kge']),
                'kge_max': results.get('kge_max', results['kge']),
                'n_gauges': results['n_valid_gauges'],
                'multi_gauge_details': results.get('per_gauge', {})
            }

        except FileNotFoundError as e:
            self.logger.error(f"Multi-gauge file not found: {e}")
            return {'kge': self.penalty_score}
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in multi-gauge metrics calculation: {e}")
            return {'kge': self.penalty_score}

    def _find_topology_path(self, settings_dir: Any) -> Optional[Path]:
        """Find mizuRoute topology file."""
        if not settings_dir:
            return None
        settings_dir = Path(settings_dir)
        mizu_settings = settings_dir / 'mizuRoute'
        if mizu_settings.exists():
            for name in ('topology.nc', 'network_topology.nc'):
                path = mizu_settings / name
                if path.exists():
                    return path
        return None

    def _parse_calibration_period(self, config: Dict[str, Any]) -> tuple:
        """Parse calibration period from config."""
        calib_period = config.get('CALIBRATION_PERIOD', '')
        if calib_period and ',' in str(calib_period):
            try:
                return tuple(s.strip() for s in str(calib_period).split(','))
            except (ValueError, AttributeError):
                pass
        return None, None

    # =========================================================================
    # Routing (kept in worker — small, delegates to converters/subprocesses)
    # =========================================================================

    def _convert_fuse_to_mizuroute_format(
        self,
        fuse_output_dir: Path,
        config: Dict[str, Any],
        settings_dir: Path,
        proc_id: int = 0
    ) -> bool:
        """Convert FUSE distributed output to mizuRoute-compatible format."""
        converter = FuseToMizurouteConverter(logger=self.logger)
        return converter.convert(fuse_output_dir, config, proc_id)

    def _run_mizuroute_for_fuse(
        self,
        config: Dict[str, Any],
        fuse_output_dir: Path,
        mizuroute_dir: Path,
        **kwargs
    ) -> bool:
        """Execute mizuRoute for FUSE output."""
        try:
            import subprocess

            # Get mizuRoute executable
            mizuroute_install = config.get('MIZUROUTE_INSTALL_PATH', 'default')
            if mizuroute_install == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                mizuroute_exe = data_dir / 'installs' / 'mizuRoute' / 'route' / 'bin' / 'mizuRoute.exe'
            else:
                mizuroute_exe = Path(mizuroute_install) / 'mizuRoute.exe'

            if not mizuroute_exe.exists():
                self.logger.error(f"mizuRoute executable not found: {mizuroute_exe}")
                return False

            # Resolve control file
            control_file = self._find_mizuroute_control(config, kwargs)
            if not control_file or not control_file.exists():
                self.logger.error(f"mizuRoute control file not found: {control_file}")
                return False

            self.logger.debug(f"Using mizuRoute control file: {control_file}")

            cmd = [str(mizuroute_exe), str(control_file)]
            self.logger.debug(f"Executing mizuRoute: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=config.get('MIZUROUTE_TIMEOUT', 3600)
            )

            if result.returncode != 0:
                self.logger.error(f"mizuRoute failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False

            if result.stdout:
                stdout_lines = result.stdout.strip().split('\n')
                self.logger.debug(f"mizuRoute completed, stdout lines: {len(stdout_lines)}")
                display_lines = stdout_lines[:5] + stdout_lines[-5:] if len(stdout_lines) > 20 else stdout_lines
                for line in display_lines:
                    if line.strip():
                        self.logger.debug(f"  mizuRoute: {line}")
            else:
                self.logger.debug("mizuRoute completed successfully (no stdout)")
            return True

        except subprocess.TimeoutExpired:
            self.logger.error("mizuRoute execution timed out")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found for mizuRoute: {e}")
            return False
        except (OSError, subprocess.SubprocessError) as e:
            self.logger.error(f"Error running mizuRoute: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _find_mizuroute_control(self, config: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[Path]:
        """Find mizuRoute control file."""
        mizuroute_settings_dir = kwargs.get('mizuroute_settings_dir')
        if mizuroute_settings_dir:
            return Path(mizuroute_settings_dir) / 'mizuroute.control'

        settings_dir_path = Path(kwargs.get('settings_dir', Path('.')))
        control_file = settings_dir_path / 'mizuRoute' / 'mizuroute.control'
        if not control_file.exists() and settings_dir_path.name == 'FUSE':
            control_file = settings_dir_path.parent / 'mizuRoute' / 'mizuroute.control'

        if control_file.exists():
            return control_file

        # Fallback to main control file
        domain_name = config.get('DOMAIN_NAME')
        data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
        project_dir = data_dir / f"domain_{domain_name}"
        return project_dir / 'settings' / 'mizuRoute' / 'mizuroute.control'

    # =========================================================================
    # Static worker function
    # =========================================================================

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Static worker function for process pool execution."""
        return _evaluate_fuse_parameters_worker(task_data)


def _evaluate_fuse_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level worker function for MPI/ProcessPool execution."""
    worker = FUSEWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
