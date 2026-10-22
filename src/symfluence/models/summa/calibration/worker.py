# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA Worker

Worker implementation for SUMMA model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.logging_utils import get_worker_logger
from symfluence.core.metrics import StreamflowMetrics
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R

logger = logging.getLogger(__name__)


@R.workers.add('SUMMA')
class SUMMAWorker(BaseWorker):
    """
    Worker for SUMMA model calibration.

    Handles parameter application, SUMMA execution, optional mizuRoute routing,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize SUMMA worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self._routing_needed = None

    def needs_routing(self, config: Dict[str, Any]) -> bool:
        """
        Determine if routing (mizuRoute) is needed.

        Args:
            config: Configuration dictionary

        Returns:
            True if routing is needed
        """
        calibration_var = config.get('CALIBRATION_VARIABLE', 'streamflow')

        if calibration_var != 'streamflow':
            return False

        domain_method = config.get('DOMAIN_DEFINITION_METHOD', 'lumped')
        routing_delineation = config.get('ROUTING_DELINEATION', 'lumped')

        if domain_method not in ['point', 'lumped']:
            return True
        if domain_method == 'lumped' and routing_delineation == 'river_network':
            return True

        return False

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to SUMMA configuration files.

        When regionalization is active, coefficient names (k_soil_a, k_soil_b)
        are expanded to per-HRU arrays with the original parameter names
        (k_soil) before writing to trialParams.nc.
        """
        try:
            # Expand regionalization coefficients to per-HRU arrays
            if any(k.endswith('_a') or k.endswith('_b') for k in params):
                config = kwargs.get('config', self.config)
                if not hasattr(self, '_cached_pm') or self._cached_pm is None:  # type: ignore[has-type]
                    from symfluence.models.summa.calibration.parameter_manager import SUMMAParameterManager
                    self._cached_pm = SUMMAParameterManager(config, self.logger, settings_dir)
                return self._cached_pm.update_model_files(params)

            # No coefficients — use standard worker path
            from .worker_impl import _apply_parameters_worker

            task_data = kwargs.get('task_data', {}).copy() if kwargs.get('task_data') else kwargs.copy()
            if 'config' not in task_data:
                task_data['config'] = self.config
            if 'params' not in task_data:
                task_data['params'] = params
            # Final evaluation does not go through TaskBuilder. Resolve the
            # same sibling routing directory used by run_model in that path.
            task_data.setdefault('mizuroute_settings_dir', str(Path(settings_dir).parent / 'mizuRoute'))

            # Worker logger at the root's effective level (no hard-coded
            # WARNING gate — that used to hide parameter-application detail)
            internal_logger = get_worker_logger(int(task_data.get('proc_id', 0) or 0))

            debug_info = {'stage': 'apply_parameters', 'files_checked': [], 'errors': []}

            return _apply_parameters_worker(
                params, task_data, settings_dir, internal_logger, debug_info
            )

        except ImportError:
            return self._apply_parameters_direct(params, settings_dir, kwargs.get('config', {}))

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying SUMMA parameters: {e}", exc_info=True)
            return False

    def _apply_parameters_direct(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        config: Dict[str, Any]
    ) -> bool:
        """
        Apply parameters directly to SUMMA trial parameter file.

        Args:
            params: Parameter values
            settings_dir: Settings directory
            config: Configuration dictionary

        Returns:
            True if successful
        """
        try:
            import netCDF4 as nc

            # Find trial parameter file
            trial_param_file = settings_dir / 'trialParams.nc'
            if not trial_param_file.exists():
                # Try alternate name
                experiment_id = config.get('EXPERIMENT_ID', 'default')
                trial_param_file = settings_dir / f'{experiment_id}_trialParams.nc'

            if not trial_param_file.exists():
                self.logger.error(f"Trial parameter file not found in {settings_dir}")
                return False

            # Update parameters
            with nc.Dataset(trial_param_file, 'r+') as ds:
                for param_name, value in params.items():
                    if param_name in ds.variables:
                        ds.variables[param_name][:] = value
                    else:
                        self.logger.warning(f"Parameter {param_name} not found in file")

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in direct parameter application: {e}", exc_info=True)
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run SUMMA model (and mizuRoute if needed).

        Args:
            config: Configuration dictionary
            settings_dir: SUMMA settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:
            # Import existing functions
            from .worker_impl import _run_mizuroute_worker, _run_summa_worker

            summa_install_path = config.get('SUMMA_INSTALL_PATH', 'default')
            summa_exe = config.get("SUMMA_EXE", "summa_sundials.exe")
            if summa_install_path == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                summa_exe = data_dir / 'installs' / 'summa' / 'bin' / summa_exe
            else:
                summa_install_path = Path(summa_install_path)
                summa_exe = summa_install_path / summa_exe

            file_manager = settings_dir / 'fileManager.txt'
            if not file_manager.exists():
                # Try alternate names
                experiment_id = config.get('EXPERIMENT_ID', 'default')
                file_manager = settings_dir / f'{experiment_id}_fileManager.txt'

            summa_dir = output_dir
            sim_dir = kwargs.get('sim_dir', output_dir)

            # Worker logger at the root's effective level (no hard-coded
            # WARNING gate — that used to hide model-run detail)
            internal_logger = get_worker_logger(int(kwargs.get('proc_id', 0) or 0))

            debug_info = {
                'stage': 'model_run',
                'commands_run': [],
                'files_checked': [],
                'errors': []
            }

            # Run SUMMA
            summa_timeout = int(config.get('SUMMA_TIMEOUT', 7200))
            success = _run_summa_worker(
                summa_exe, file_manager, summa_dir, internal_logger, debug_info, settings_dir,
                timeout=summa_timeout, config=config
            )

            if not success:
                return False

            # Coupled land-only mode: when SUMMA is the upstream model in a coupled chain whose
            # routing is owned by a downstream model (e.g. dRoute), emit runoff only and let the
            # downstream model route. The coupled worker sets ``skip_routing`` for non-objective
            # models. This path is never taken in standalone SUMMA calibration (the flag is absent),
            # so the existing SUMMA -> mizuRoute coupling is unchanged.
            if kwargs.get('skip_routing'):
                self.logger.info(
                    "SUMMA coupled land-only mode: skipping mizuRoute (downstream model routes runoff)")
                return True

            # Run routing if needed
            if self.needs_routing(config):
                # Respect explicit routing output directories from callers
                mizuroute_dir = kwargs.get('mizuroute_dir')
                if mizuroute_dir is not None:
                    mizuroute_dir = Path(mizuroute_dir)
                else:
                    # mizuRoute output should be alongside SUMMA, not inside it
                    # e.g., /run_1/mizuRoute not /run_1/SUMMA/mizuRoute
                    mizuroute_dir = (
                        sim_dir.parent / 'mizuRoute' if sim_dir
                        else output_dir.parent / 'mizuRoute'
                    )
                mizuroute_dir.mkdir(parents=True, exist_ok=True)

                # Propagate all kwargs into task_data for legacy compatibility
                task_data = kwargs.copy()
                mizuroute_settings_dir = settings_dir.parent / 'mizuRoute'
                task_data.update({
                    'config': config,
                    'summa_dir': str(summa_dir),
                    'mizuroute_dir': str(mizuroute_dir),
                    'mizuroute_settings_dir': str(mizuroute_settings_dir),
                })

                routing_success = _run_mizuroute_worker(
                    task_data, mizuroute_dir, internal_logger, debug_info, summa_dir
                )

                if not routing_success:
                    # Routing is required for non-lumped domains - this is a failure
                    self.logger.error("Routing failed for semi-distributed/distributed domain - cannot calculate streamflow metrics")
                    return False

            return True

        except ImportError:
            # Fallback: Run SUMMA directly
            return self._run_summa_direct(config, settings_dir, output_dir)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running SUMMA: {e}", exc_info=True)
            return False

    def _run_summa_direct(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path
    ) -> bool:
        """
        Run SUMMA directly using subprocess.

        Args:
            config: Configuration dictionary
            settings_dir: Settings directory
            output_dir: Output directory

        Returns:
            True if successful
        """
        try:

            summa_install_path = config.get('SUMMA_INSTALL_PATH', 'default')
            summa_exe_name = config.get('SUMMA_EXE', 'summa_sundials.exe')
            if summa_install_path == 'default':
                data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                summa_exe = data_dir / 'installs' / 'summa' / 'bin' / summa_exe_name
            else:
                summa_exe = Path(summa_install_path) / summa_exe_name

            file_manager = settings_dir / 'fileManager.txt'

            if not summa_exe.exists() or not file_manager.exists():
                return False

            # Run SUMMA
            cmd = [str(summa_exe), '-m', str(file_manager)]
            result = run_subprocess(
                cmd,
                capture_output=True,
                text=True,
                timeout=config.get('SUMMA_TIMEOUT', 600)
            )

            return result.returncode == 0

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in direct SUMMA execution: {e}", exc_info=True)
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from SUMMA output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            multi_gauge_enabled = config.get('MULTI_GAUGE_CALIBRATION', False)
            if multi_gauge_enabled and self.needs_routing(config):
                mizuroute_dir = kwargs.get('mizuroute_dir')
                if not mizuroute_dir:
                    sim_dir = kwargs.get('sim_dir', output_dir)
                    mizuroute_dir = sim_dir / 'mizuRoute'
                mizuroute_output = self._find_mizuroute_output(Path(mizuroute_dir))
                if mizuroute_output is not None:
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    domain_name = config.get('DOMAIN_NAME', '')
                    project_dir = data_dir / f'domain_{domain_name}'
                    return self._calculate_multi_gauge_metrics(
                        config=config,
                        mizuroute_output_path=mizuroute_output,
                        project_dir=project_dir,
                        settings_dir=kwargs.get('settings_dir'),
                    )

            # Import existing function
            from .worker_impl import _calculate_metrics_with_target

            # Resolve mizuroute_dir if needed
            mizuroute_dir = kwargs.get('mizuroute_dir')
            if not mizuroute_dir and self.needs_routing(config):
                sim_dir = kwargs.get('sim_dir', output_dir)
                mizuroute_dir = sim_dir / 'mizuRoute'
                # Fall back to the standard simulations path when the
                # optimization-dir path has no output (e.g. final evaluation
                # redirects SUMMA output but mizuRoute still writes to simulations/)
                if not list(mizuroute_dir.glob('*.nc')):
                    experiment_id = config.get('EXPERIMENT_ID', '')
                    data_dir = config.get('SYMFLUENCE_DATA_DIR', '')
                    domain_name = config.get('DOMAIN_NAME', '')
                    if experiment_id and data_dir and domain_name:
                        fallback = Path(data_dir) / f"domain_{domain_name}" / 'simulations' / experiment_id / 'mizuRoute'
                        if list(fallback.glob('*.nc')):
                            mizuroute_dir = fallback

            metrics = _calculate_metrics_with_target(
                output_dir, mizuroute_dir, config, self.logger
            )

            # Ensure return is a dict
            if isinstance(metrics, (int, float)):
                metric_name = config.get('CALIBRATION_METRIC', 'KGE').lower()
                return {metric_name: float(metrics)}

            return metrics or {'kge': self.penalty_score}

        except ImportError:
            # Fallback: Calculate metrics directly
            return self._calculate_metrics_direct(output_dir, config)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating SUMMA metrics: {e}", exc_info=True)
            return {'kge': self.penalty_score}

    # =========================================================================
    # Multi-Gauge Metrics (delegates to shared MultiGaugeMetrics)
    # =========================================================================

    def _calculate_multi_gauge_metrics(
        self,
        config: Dict[str, Any],
        mizuroute_output_path: Path,
        project_dir: Path,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate performance metrics across multiple stream gauges.

        Uses the shared :class:`MultiGaugeMetrics` to read per-segment
        routed discharge from mizuRoute output and compare against
        LaMAH-Ice gauge observations.
        """
        try:
            from symfluence.core.calibration.multi_gauge.metrics import MultiGaugeMetrics

            gauge_mapping_path = config.get('GAUGE_SEGMENT_MAPPING')
            obs_dir = config.get('MULTI_GAUGE_OBS_DIR')
            gauge_ids = config.get('MULTI_GAUGE_IDS')
            exclude_ids = config.get('MULTI_GAUGE_EXCLUDE_IDS', [])
            aggregation = config.get('MULTI_GAUGE_AGGREGATION', 'mean')
            min_gauges = config.get('MULTI_GAUGE_MIN_GAUGES', 5)

            # Auto-generate gauge mapping if not configured
            if not gauge_mapping_path:
                gauge_mapping_path = self._ensure_gauge_mapping(config, project_dir)
            if not gauge_mapping_path:
                self.logger.error("GAUGE_SEGMENT_MAPPING not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}
            if not obs_dir:
                self.logger.error("MULTI_GAUGE_OBS_DIR not configured for multi-gauge calibration")
                return {'kge': self.penalty_score}

            gauge_mapping_path = Path(gauge_mapping_path)
            obs_dir = Path(obs_dir)

            if 'D_gauges' in obs_dir.parts and not obs_dir.exists():
                try:
                    from symfluence.core.registries import R
                    lamah_root = obs_dir
                    while lamah_root.name != 'D_gauges' and lamah_root.parent != lamah_root:
                        lamah_root = lamah_root.parent
                    lamah_root = lamah_root.parent
                    R.observation_handlers.get('lamah_ice_streamflow').ensure_streamflow(lamah_root, self.logger)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"LaMAH-Ice auto-download skipped: {exc}")

            topology_path = self._find_topology_path(kwargs.get('settings_dir'))
            start_date, end_date = self._parse_calibration_period(config)

            multi_gauge = MultiGaugeMetrics(
                gauge_segment_mapping_path=gauge_mapping_path,
                obs_data_dir=obs_dir,
                logger=self.logger,
            )

            if gauge_ids is None and exclude_ids:
                all_gauge_ids = multi_gauge.gauge_mapping['id'].tolist()
                gauge_ids = [gid for gid in all_gauge_ids if gid not in exclude_ids]

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
                kge_floor=kge_floor,
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
                'multi_gauge_details': results.get('per_gauge', {}),
            }

        except FileNotFoundError as e:
            self.logger.error(f"Multi-gauge file not found: {e}")
            return {'kge': self.penalty_score}
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in multi-gauge metrics calculation: {e}", exc_info=True)
            return {'kge': self.penalty_score}

    def _ensure_gauge_mapping(
        self,
        config: Dict[str, Any],
        project_dir: Path,
    ) -> Optional[Path]:
        """Auto-generate gauge-segment mapping if it doesn't exist."""
        lamah_path = config.get('LAMAH_ICE_PATH')
        if not lamah_path:
            data_dir = config.get('SYMFLUENCE_DATA_DIR')
            if data_dir:
                lamah_path = Path(data_dir) / 'lamah_ice'
        if not lamah_path:
            return None
        lamah_path = Path(lamah_path)
        if lamah_path.name == 'default':
            data_dir = config.get('SYMFLUENCE_DATA_DIR')
            if data_dir:
                lamah_path = Path(data_dir) / 'lamah_ice'

        domain_name = config.get('DOMAIN_NAME', '')
        try:
            from symfluence.core.calibration.multi_gauge.gauge_mapping import ensure_gauge_mapping
            return ensure_gauge_mapping(
                project_dir, lamah_path, domain_name,
                output_subdir='mizuRoute',
                output_filename='gauge_segment_mapping.csv',
                logger=self.logger,
            )
        except Exception as e:  # noqa: BLE001
            self.logger.warning(f"Could not auto-generate gauge mapping: {e}")
            return None

    def _find_mizuroute_output(self, mizuroute_dir: Path) -> Optional[Path]:
        """Find the mizuRoute output NetCDF file."""
        if not mizuroute_dir.exists():
            return None
        for pattern in ('*.nc', '*output*.nc'):
            files = sorted(mizuroute_dir.glob(pattern))
            if files:
                return files[-1]
        return None

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
        cal_period = config.get('CALIBRATION_PERIOD', '')
        if isinstance(cal_period, str) and ',' in cal_period:
            parts = [p.strip() for p in cal_period.split(',')]
            return parts[0], parts[1]
        start = config.get('EXPERIMENT_TIME_START', '')
        end = config.get('EXPERIMENT_TIME_END', '')
        return str(start).strip(), str(end).strip()

    def _calculate_metrics_direct(
        self,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate metrics directly from output files.

        Args:
            output_dir: Output directory
            config: Configuration dictionary

        Returns:
            Dictionary of metrics
        """
        try:
            import xarray as xr

            from symfluence.core.metrics import kge, nse

            # Find output file
            output_files = list(output_dir.glob('*_day.nc')) + list(output_dir.glob('*_output.nc'))
            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files found'}

            # Read simulation
            with xr.open_dataset(output_files[0]) as ds:
                if 'scalarTotalRunoff' in ds:
                    sim = ds['scalarTotalRunoff'].values.flatten()  # m/s (runoff depth)
                elif 'averageRoutedRunoff' in ds:
                    sim = ds['averageRoutedRunoff'].values.flatten()  # m/s (runoff depth)
                else:
                    return {'kge': self.penalty_score, 'error': 'No runoff variable found'}

            # Convert runoff depth (m/s) to discharge (m³/s) using catchment area
            domain_name = config.get('DOMAIN_NAME')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))

            # Try to get area from SUMMA attributes file first (more reliable)
            attrs_file = data_dir / f'domain_{domain_name}' / 'settings' / 'SUMMA' / 'attributes.nc'
            if attrs_file.exists():
                with xr.open_dataset(attrs_file) as attrs:
                    if 'HRUarea' in attrs.data_vars:
                        catchment_area_m2 = float(attrs['HRUarea'].values.sum())  # m²
                    else:
                        self.logger.warning("HRUarea not found in attributes, cannot convert units")
                        return {'kge': self.penalty_score, 'error': 'Cannot get catchment area'}
            else:
                self.logger.warning(f"Attributes file not found: {attrs_file}")
                return {'kge': self.penalty_score, 'error': 'Attributes file not found'}

            # Convert: runoff (m/s) * area (m²) = discharge (m³/s)
            sim = sim * catchment_area_m2
            self.logger.debug(f"Converted runoff to discharge using area={catchment_area_m2:.2e} m²")

            # Load observations — model-ready store first, CSV fallback.
            project_dir = data_dir / f'domain_{domain_name}'
            obs_values, obs_index = StreamflowMetrics().load_observations(
                config, project_dir, domain_name, resample_freq=None,
            )
            if obs_values is None or obs_index is None:
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs = obs_values

            # Align lengths
            min_len = min(len(sim), len(obs))
            sim = sim[:min_len]
            obs = obs[:min_len]

            # Calculate metrics
            kge_val = kge(obs, sim, transfo=1)
            nse_val = nse(obs, sim, transfo=1)

            return {
                'kge': float(kge_val),
                'nse': float(nse_val),
            }

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in direct metrics calculation: {e}", exc_info=True)
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Maintains backward compatibility with existing parallel processing.

        Args:
            task_data: Task dictionary in legacy format

        Returns:
            Result dictionary in legacy format
        """
        try:
            # Use existing safe wrapper for full functionality
            from .worker_impl import _evaluate_parameters_worker_safe
            return _evaluate_parameters_worker_safe(task_data)

        except ImportError:
            # Fallback: Use new worker infrastructure
            worker = SUMMAWorker()
            task = WorkerTask.from_legacy_dict(task_data)
            result = worker.evaluate(task)
            return result.to_legacy_dict()


def _evaluate_summa_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.
    Delegates to the standard evaluate_worker_function.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    return SUMMAWorker.evaluate_worker_function(task_data)
