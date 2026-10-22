# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
HYPE Worker

Worker implementation for HYPE model optimization.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from symfluence.core.calibration.workers.base_worker import BaseWorker, WorkerTask
from symfluence.core.logging_utils import log_once
from symfluence.core.metrics import StreamflowMetrics, kge, nse
from symfluence.core.process_exec import run as run_subprocess
from symfluence.core.registries import R
from symfluence.models.hype.preprocessor import HYPEPreProcessor


@R.workers.add('HYPE')
class HYPEWorker(BaseWorker):
    """
    Worker for HYPE model calibration.

    Handles parameter application, HYPE execution, and metric calculation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize HYPE worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)
        self._hype_exe = self._resolve_hype_exe()

    _regionalization_cache = None

    def _get_regionalization(self, config, settings_dir):
        """Get or create the cached regionalization adapter."""
        if self._regionalization_cache is not None:
            return self._regionalization_cache

        from symfluence.core.calibration.parameters.parameter_bounds_registry import get_hype_bounds
        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )

        geoclass_path = settings_dir / 'GeoClass.txt'
        if not geoclass_path.exists():
            return None

        bounds = get_hype_bounds()
        tuple_bounds = {k: (v['min'], v['max']) for k, v in bounds.items()}

        method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
        geodata_path = settings_dir / 'GeoData.txt'
        project_dir = config.get('project_dir', '')
        climate_stats_path = None
        soil_csv_path = None
        if project_dir:
            p = Path(project_dir)
            cs = p / 'data' / 'attributes' / 'climate' / 'climate_statistics.csv'
            if cs.exists():
                climate_stats_path = cs
            sc = p / 'data' / 'attributes' / 'soilclass' / 'iceland_soil_attributes.csv'
            if sc.exists():
                soil_csv_path = sc

        self._regionalization_cache = create_hype_regionalization(
            method=method, param_bounds=tuple_bounds,
            geoclass_path=geoclass_path,
            geodata_path=geodata_path if geodata_path.exists() else None,
            climate_stats_path=climate_stats_path,
            soil_attributes_csv=soil_csv_path,
            logger=self.logger,
        )
        return self._regionalization_cache

    def _expand_hype_coefficients(self, params, config, settings_dir):
        """Expand TF coefficients to par.txt values using the regionalization adapter."""
        try:
            reg = self._get_regionalization(config, settings_dir)
            if reg is None:
                log_once(self.logger, logging.WARNING, key='hype-geoclass-missing',
                         message="GeoClass.txt not found for coefficient expansion")
                return None
            return reg.expand_to_par_values(params)
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to expand HYPE coefficients: {e}")
            return None

    def _resolve_hype_exe(self) -> str:
        """Resolve the HYPE executable path from the worker's full config."""
        import shutil

        # Extract flat dict from whatever config type we have
        cfg: dict = {}
        if isinstance(self.config, dict):
            cfg = self.config
        elif hasattr(self.config, 'config_dict'):
            cfg = self.config.config_dict
        elif hasattr(self.config, 'model_dump'):
            cfg = self.config.model_dump()

        exe_name = cfg.get('HYPE_EXE', 'hype')

        if Path(exe_name).is_absolute() and Path(exe_name).exists():
            return exe_name

        # Honor the explicitly configured install path (as runner.py does).
        # Without this the worker ignores HYPE_INSTALL_PATH and falls through to
        # the bare exe name -> FileNotFoundError -> 100% calibration crashes.
        install_path = cfg.get('HYPE_INSTALL_PATH', '')
        if install_path and install_path != 'default':
            ip = Path(install_path)
            for candidate in (ip / exe_name, ip):
                if candidate.is_file():
                    return str(candidate)

        # Try SYMFLUENCE_DATA_DIR install location
        data_dir = cfg.get('SYMFLUENCE_DATA_DIR', '')
        if data_dir and data_dir != 'default':
            candidate = Path(data_dir) / 'installs' / 'hype' / 'bin' / exe_name
            if candidate.exists():
                return str(candidate)

        # Try typed config accessors
        if hasattr(self.config, 'system') and hasattr(self.config.system, 'data_dir'):
            data_dir = str(self.config.system.data_dir or '')
            if data_dir and data_dir != 'default':
                candidate = Path(data_dir) / 'installs' / 'hype' / 'bin' / exe_name
                if candidate.exists():
                    return str(candidate)

        # Try PATH
        resolved: Optional[str] = shutil.which(exe_name)
        if resolved:
            return resolved

        return exe_name

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to HYPE configuration files.

        When regionalization is active, coefficients are expanded to
        per-SLC values via the parameter manager before writing par.txt.
        """
        try:
            config = kwargs.get('config', self.config)

            # Expand regionalization coefficients to per-SLC par.txt values
            if any(k.endswith('_a') or k.endswith('_b') for k in params):
                cfg = config if isinstance(config, dict) else getattr(config, 'config_dict', {})
                settings_dir_path = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir
                expanded = self._expand_hype_coefficients(params, cfg, settings_dir_path)
                if expanded is not None:
                    params = expanded

            # Use HYPEPreProcessor to regenerate configs with new params
            preprocessor = HYPEPreProcessor(config, self.logger, params=params)

            # Set model-specific paths to point to the worker's settings dir
            # Ensure settings_dir is a Path object
            settings_dir = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir

            preprocessor.output_path = settings_dir
            preprocessor.hype_setup_dir = settings_dir
            # IMPORTANT: forcing_data_dir must point to where forcing files are located
            # Forcing files live in data/forcing/HYPE_input/ and don't change during calibration
            # Keep the original forcing_data_dir from the preprocessor (already set correctly)

            # CRITICAL: Also update the manager output paths so par.txt and config
            # files are written to the worker's isolated directory, not the shared default
            preprocessor.config_manager.output_path = settings_dir
            preprocessor.geodata_manager.output_path = settings_dir

            # Use isolated output directory for the worker
            output_dir = kwargs.get('proc_output_dir') or kwargs.get('output_dir')
            if output_dir:
                preprocessor.hype_results_dir = Path(output_dir)
                preprocessor.hype_results_dir_str = str(Path(output_dir)).rstrip('/') + '/'

            # Only regenerate par.txt and info.txt/filedir.txt
            # GeoData and forcing files should already be copied to worker directory
            # and don't change during calibration
            self.logger.debug("Regenerating par.txt and info.txt for calibration")
            self.logger.debug(f"  output_path: {preprocessor.output_path}")
            self.logger.debug(f"  forcing_data_dir: {preprocessor.forcing_data_dir}")

            # Get land uses from existing GeoClass.txt (should already be copied)
            geoclass_file = settings_dir / 'GeoClass.txt'
            if not geoclass_file.exists():
                self.logger.error(f"GeoClass.txt not found at {geoclass_file}")
                self.logger.error("GeoData files must be copied to worker directory before calibration")
                return False

            # Read land uses from GeoClass.txt
            import pandas as pd
            try:
                geoclass_df = pd.read_csv(geoclass_file, sep='\t', skiprows=1, header=None)
                land_uses = geoclass_df.iloc[:, 1].unique()
                self.logger.debug(f"Read {len(land_uses)} land use types from GeoClass.txt")
            except Exception as e:  # noqa: BLE001 — calibration resilience
                self.logger.error(f"Failed to read GeoClass.txt: {e}", exc_info=True)
                return False

            # Write parameter file with calibration parameters
            preprocessor.config_manager.write_par_file(
                params=params,
                land_uses=land_uses
            )

            # Get experiment dates from config
            experiment_start = preprocessor._get_config_value(
                lambda: preprocessor.config.domain.time_start,
                default=None,
            )
            experiment_end = preprocessor._get_config_value(
                lambda: preprocessor.config.domain.time_end,
                default=None,
            )

            # Write info and file directory files — resultdir must point to
            # the per-iteration output_dir so the runner finds HYPE output
            calib_results_dir = str(output_dir).rstrip('/') + '/'
            preprocessor.config_manager.write_info_filedir(
                spinup_days=preprocessor.spinup_days,
                results_dir=calib_results_dir,
                experiment_start=experiment_start,
                experiment_end=experiment_end,
                forcing_data_dir=preprocessor.forcing_data_dir
            )

            self.logger.debug("par.txt and info.txt regeneration completed")

            # Debug: Check if forcing files exist
            forcing_files = ['Pobs.txt', 'TMAXobs.txt', 'TMINobs.txt', 'Tobs.txt']
            for f in forcing_files:
                fpath = settings_dir / f
                if not fpath.exists():
                    self.logger.error(f"Missing forcing file during calibration: {fpath}")
                else:
                    self.logger.debug(f"Forcing file exists: {fpath}")

            # Debug: Check if filedir.txt points to correct location
            filedir_path = settings_dir / 'filedir.txt'
            if filedir_path.exists():
                with open(filedir_path, 'r', encoding='utf-8') as f:
                    forcing_path_in_filedir = f.read().strip()
                self.logger.debug(f"filedir.txt points to: {forcing_path_in_filedir}")
                if not Path(forcing_path_in_filedir).exists():
                    self.logger.error(f"Forcing path in filedir.txt does not exist: {forcing_path_in_filedir}")
            else:
                self.logger.error(f"filedir.txt not found at {filedir_path}")

            # Debug: Check if par.txt was updated with calibration parameters
            par_file = settings_dir / 'par.txt'
            if par_file.exists() and params:
                with open(par_file, 'r', encoding='utf-8') as f:
                    par_content = f.read()
                for param_name in list(params.keys())[:3]:  # Check first 3 params
                    if param_name in par_content:
                        self.logger.debug(f"Parameter {param_name} found in par.txt")
                    else:
                        log_once(self.logger, logging.WARNING, key=f'hype-param-missing-{param_name}',
                                 message=f"Parameter {param_name} NOT found in par.txt")

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying HYPE parameters: {e}", exc_info=True)
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def run_model(
        self,
        config: Dict[str, Any],
        settings_dir: Path,
        output_dir: Path,
        **kwargs
    ) -> bool:
        """
        Run HYPE model.

        Args:
            config: Configuration dictionary
            settings_dir: HYPE settings directory
            output_dir: Output directory
            **kwargs: Additional arguments

        Returns:
            True if model ran successfully
        """
        try:

            settings_dir = Path(settings_dir) if not isinstance(settings_dir, Path) else settings_dir
            output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            hype_exe = self._hype_exe

            cmd = [str(hype_exe), str(settings_dir).rstrip('/') + '/']
            result = run_subprocess(
                cmd, capture_output=True, text=True, timeout=7200
            )

            if result.returncode != 0:
                self.logger.warning(
                    f"HYPE exited with rc={result.returncode}: "
                    f"{result.stderr[:200] if result.stderr else 'no stderr'}"
                )
                return False

            # Verify output was written
            cout_file = output_dir / 'timeCOUT.txt'
            if not cout_file.exists():
                self.logger.warning(
                    f"HYPE exited 0 but timeCOUT.txt not found in {output_dir}"
                )
                return False

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Error running HYPE: {e}", exc_info=True)
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from HYPE output.

        Args:
            output_dir: Directory containing model outputs
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        output_dir = Path(output_dir)

        try:
            # Check for multi-gauge calibration
            multi_gauge = self._get_multi_gauge_flag(config)
            if multi_gauge:
                return self._calculate_multi_gauge_metrics(output_dir, config)

            # HYPE output file for computed discharge
            sim_file = output_dir / 'timeCOUT.txt'
            if not sim_file.exists():
                self.logger.error(f"timeCOUT.txt not found at {sim_file}")
                return {'kge': self.penalty_score, 'error': 'timeCOUT.txt not found'}

            # Read simulation (HYPE output is tab-separated, first row is comment)
            sim_df = pd.read_csv(sim_file, sep='\t', skiprows=1)

            # Parse DATE column
            if 'DATE' in sim_df.columns:
                sim_df['DATE'] = pd.to_datetime(sim_df['DATE'])
                sim_df = sim_df.set_index('DATE')
            elif 'time' in sim_df.columns:
                sim_df['time'] = pd.to_datetime(sim_df['time'])
                sim_df = sim_df.set_index('time')

            # Get subbasin columns (exclude date columns)
            subbasin_cols = [col for col in sim_df.columns if col not in ['DATE', 'time']]
            if len(subbasin_cols) == 0:
                return {'kge': self.penalty_score, 'error': 'No subbasin columns in output'}

            # For lumped domains or auto-select outlet (highest mean flow)
            if len(subbasin_cols) > 1:
                # Convert to numeric first
                for col in subbasin_cols:
                    sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
                subbasin_means = sim_df[subbasin_cols].mean()
                outlet_col = subbasin_means.idxmax()
                sim_series = sim_df[outlet_col]
                self.logger.debug(
                    f"Auto-selected outlet '{outlet_col}' from {len(subbasin_cols)} subbasins "
                    f"(highest mean discharge: {subbasin_means[outlet_col]:.3f} m³/s)"
                )
            else:
                outlet_col = subbasin_cols[0]
                sim_series = pd.to_numeric(sim_df[outlet_col], errors='coerce')
                self.logger.debug(f"Using single outlet column: {outlet_col}")

            # Load observations — resolve domain_name and data_dir from
            # whichever config carries them (task config or worker config).
            domain_name = None
            data_dir = Path('.')
            for cfg in (config, self.config):
                if domain_name:
                    break
                try:
                    domain_name = cfg.domain.name
                    data_dir = Path(str(cfg.system.data_dir))
                except (AttributeError, TypeError):
                    pass
                if not domain_name and isinstance(cfg, dict):
                    domain_name = cfg.get('DOMAIN_NAME')
                    dd = cfg.get('SYMFLUENCE_DATA_DIR', '')
                    if dd and dd != 'default':
                        data_dir = Path(dd)
                if not domain_name and hasattr(cfg, 'config_dict'):
                    domain_name = cfg.config_dict.get('DOMAIN_NAME')
                    dd = cfg.config_dict.get('SYMFLUENCE_DATA_DIR', '')
                    if dd and dd != 'default':
                        data_dir = Path(dd)

            # Model-ready store first, legacy preprocessed CSV fallback.
            project_dir = data_dir / f'domain_{domain_name}'
            obs_values, obs_index = StreamflowMetrics().load_observations(
                config, project_dir, domain_name, resample_freq=None,
            )
            if obs_values is None or obs_index is None:
                self.logger.error(f"Observations not found for domain_name={domain_name}")
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.DataFrame({'discharge_cms': obs_values}, index=obs_index)

            # HYPE outputs daily data, observations may be hourly
            # Resample observations to daily mean if they are sub-daily
            obs_freq = pd.infer_freq(obs_df.index[:10])
            if obs_freq and obs_freq in ['H', 'h', 'T', 'min', 'S', 's']:
                # Hourly or sub-hourly observations - resample to daily mean
                obs_daily = obs_df.resample('D').mean()
            else:
                obs_daily = obs_df

            # Normalize both indices to date-only for alignment
            sim_series.index = sim_series.index.normalize()
            obs_daily.index = obs_daily.index.normalize()

            # Restrict scoring to the calibration period.
            #
            # This used to read config.optimization.calibration_period, which
            # does not exist — the field lives on the domain section
            # (CALIBRATION_PERIOD). The AttributeError was swallowed silently,
            # so every HYPE calibration scored the whole overlapping record,
            # evaluation period included: the optimizer trained on held-out
            # data and its reported score covered a different window than the
            # Calib_* metrics written at final evaluation. Resolve it through
            # the canonical key, exactly as _calculate_multi_gauge_metrics does.
            calib_period = self._resolve_config_value('CALIBRATION_PERIOD', config)
            if not calib_period:
                try:
                    calib_period = config.domain.calibration_period
                except (AttributeError, TypeError):
                    calib_period = None
            if calib_period:
                try:
                    start_str, end_str = [s.strip() for s in str(calib_period).split(',')]
                    calib_start = pd.to_datetime(start_str)
                    calib_end = pd.to_datetime(end_str)
                    sim_series = sim_series[(sim_series.index >= calib_start) & (sim_series.index <= calib_end)]
                    obs_daily = obs_daily[(obs_daily.index >= calib_start) & (obs_daily.index <= calib_end)]
                except Exception as e:  # noqa: BLE001 — calibration resilience
                    self.logger.warning(f"Could not apply calibration period: {e}", exc_info=True)
            else:
                # Silence here is what hid the original bug; say so loudly.
                self.logger.warning(
                    "No CALIBRATION_PERIOD resolved for HYPE — scoring the whole "
                    "overlapping record, which includes any evaluation period."
                )

            # Find common dates
            common_idx = sim_series.index.intersection(obs_daily.index)
            if len(common_idx) == 0:
                self.logger.error(
                    f"No common dates between simulation ({sim_series.index.min()} to {sim_series.index.max()}) "
                    f"and observations ({obs_daily.index.min()} to {obs_daily.index.max()})"
                )
                return {'kge': self.penalty_score, 'error': 'No common dates between sim and obs'}

            self.logger.debug(f"Calculating metrics using {len(common_idx)} common timesteps")

            # Get the discharge column from observations
            obs_col = obs_daily.columns[0] if len(obs_daily.columns) > 0 else 'discharge_cms'
            obs_aligned = obs_daily.loc[common_idx, obs_col].values.flatten() if obs_col in obs_daily.columns else obs_daily.loc[common_idx].values.flatten()
            sim_aligned = sim_series.loc[common_idx].values.flatten()

            # Log diagnostic statistics for debugging bias issues
            mean_obs = float(obs_aligned[~pd.isna(obs_aligned)].mean()) if len(obs_aligned) > 0 else 0.0
            mean_sim = float(sim_aligned[~pd.isna(sim_aligned)].mean()) if len(sim_aligned) > 0 else 0.0
            self.logger.debug(
                f"Calibration diagnostics | mean_obs: {mean_obs:.3f} m³/s | mean_sim: {mean_sim:.3f} m³/s | "
                f"bias_ratio: {mean_sim/mean_obs:.3f}" if mean_obs != 0 else
                f"Calibration diagnostics | mean_obs: {mean_obs:.3f} m³/s | mean_sim: {mean_sim:.3f} m³/s | bias_ratio: N/A"
            )

            # Check for NaN values in aligned data
            obs_nan_count = pd.isna(obs_aligned).sum()
            sim_nan_count = pd.isna(sim_aligned).sum()
            if obs_nan_count > 0 or sim_nan_count > 0:
                log_once(
                    self.logger, logging.WARNING,
                    key='hype-nan-pairs',
                    message=(
                        f"NaN values detected: {obs_nan_count} in observations, {sim_nan_count} in simulation. "
                        f"Removing NaN pairs for metric calculation."
                    ),
                )
                # Remove NaN pairs
                valid_mask = ~(pd.isna(obs_aligned) | pd.isna(sim_aligned))
                obs_aligned = obs_aligned[valid_mask]
                sim_aligned = sim_aligned[valid_mask]

                if len(obs_aligned) == 0:
                    return {'kge': self.penalty_score, 'error': 'All data pairs contain NaN'}

            # Check for all-zero simulations (model didn't produce discharge)
            if sim_aligned.sum() == 0:
                log_once(self.logger, logging.WARNING, key='hype-zero-discharge',
                     message="HYPE simulation produced zero discharge - check model parameters")
                return {'kge': self.penalty_score, 'error': 'Zero discharge from model'}

            kge_val = kge(obs_aligned, sim_aligned, transfo=1)
            nse_val = nse(obs_aligned, sim_aligned, transfo=1)

            # Handle NaN values
            if pd.isna(kge_val):
                kge_val = self.penalty_score
            if pd.isna(nse_val):
                nse_val = self.penalty_score

            metrics = {'kge': float(kge_val), 'nse': float(nse_val)}
            if self._resolve_config_value('HYPE_CRYOSPHERE_CONSTRAINTS', config, False):
                from ..cryosphere import calculate_constraints
                metrics.update(calculate_constraints(
                    output_dir, project_dir, calib_period,
                    self._resolve_config_value('HYPE_SNOW_COVER_ERROR_SCALE', config, 0.2),
                    self._resolve_config_value('HYPE_GLACIER_MB_ERROR_SCALE', config, 1.0)))
            return metrics

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating HYPE metrics: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return {'kge': self.penalty_score}

    def _get_multi_gauge_flag(self, config: Any) -> bool:
        """Check whether multi-gauge calibration is enabled."""
        for cfg in (config, self.config):
            if isinstance(cfg, dict) and cfg.get('MULTI_GAUGE_CALIBRATION'):
                return True
            if hasattr(cfg, 'config_dict') and cfg.config_dict.get('MULTI_GAUGE_CALIBRATION'):
                return True
            try:
                if cfg.calibration.multi_gauge.enabled:
                    return True
            except (AttributeError, TypeError):
                pass
        return False

    def _resolve_config_value(self, key: str, config: Any, default: Any = None) -> Any:
        """Resolve a config key from task config or worker config."""
        for cfg in (config, self.config):
            if isinstance(cfg, dict) and key in cfg:
                return cfg[key]
            if hasattr(cfg, 'config_dict') and key in cfg.config_dict:
                return cfg.config_dict[key]
        return default

    def _calculate_multi_gauge_metrics(
        self,
        output_dir: Path,
        config: Any,
    ) -> Dict[str, Any]:
        """Calculate KGE across multiple LamaH-ICE gauges.

        Uses :class:`HYPEMultiGaugeMetrics` to read per-subbasin
        discharge from ``timeCOUT.txt`` and compare against LamaH-ICE
        observations at each gauge.
        """
        from symfluence.core.registries import R

        from .multi_gauge_metrics import HYPEMultiGaugeMetrics, ensure_hype_gauge_mapping

        # Resolve paths
        domain_name = self._resolve_config_value('DOMAIN_NAME', config)
        data_dir = self._resolve_config_value('SYMFLUENCE_DATA_DIR', config, '.')
        if data_dir == 'default' or not data_dir:
            data_dir = '.'
        project_dir = Path(data_dir) / f'domain_{domain_name}'

        # LamaH-ICE observation data
        obs_dir_cfg = self._resolve_config_value('MULTI_GAUGE_OBS_DIR', config)
        if obs_dir_cfg:
            lamah_root = Path(obs_dir_cfg)
        else:
            lamah_root = Path(data_dir) / 'lamah_ice'

        lamah_root = R.observation_handlers.get('lamah_ice_streamflow').ensure_streamflow(lamah_root, self.logger)
        obs_daily_dir = lamah_root / 'D_gauges' / '2_timeseries' / 'daily'

        # Gauge→subbasin mapping
        mapping_path_cfg = self._resolve_config_value('GAUGE_SEGMENT_MAPPING', config)
        if mapping_path_cfg and Path(mapping_path_cfg).exists():
            mapping_path = Path(mapping_path_cfg)
        else:
            mapping_path = ensure_hype_gauge_mapping(
                project_dir, lamah_root, domain_name or '', self.logger
            )
        if mapping_path is None:
            self.logger.error("Cannot generate gauge→subbasin mapping")
            return {'kge': self.penalty_score, 'error': 'No gauge mapping'}

        # Instantiate multi-gauge evaluator
        mgm = HYPEMultiGaugeMetrics(mapping_path, obs_daily_dir, self.logger)

        # Parse gauge IDs and calibration period
        gauge_ids_cfg = self._resolve_config_value('MULTI_GAUGE_IDS', config)
        gauge_ids = [int(g) for g in gauge_ids_cfg] if gauge_ids_cfg else None

        exclude_ids = self._resolve_config_value('MULTI_GAUGE_EXCLUDE_IDS', config) or []
        if exclude_ids and gauge_ids is None:
            all_ids = mgm.get_available_gauges()
            gauge_ids = [g for g in all_ids if g not in exclude_ids]

        start_date = self._resolve_config_value('CALIBRATION_PERIOD', config)
        if isinstance(start_date, str) and ',' in start_date:
            start_date, end_date = [s.strip() for s in start_date.split(',', 1)]
        else:
            end_date = None

        aggregation = self._resolve_config_value('MULTI_GAUGE_AGGREGATION', config, 'mean')
        min_gauges = int(self._resolve_config_value('MULTI_GAUGE_MIN_GAUGES', config, 3))
        kge_floor = self._resolve_config_value('MULTI_GAUGE_KGE_FLOOR', config)
        if kge_floor is not None:
            kge_floor = float(kge_floor)

        filter_config: Dict[str, Any] = {}
        max_dist = self._resolve_config_value('MULTI_GAUGE_MAX_DISTANCE', config)
        if max_dist is not None:
            filter_config['max_distance'] = float(max_dist)
        min_cv = self._resolve_config_value('MULTI_GAUGE_MIN_OBS_CV', config)
        if min_cv is not None:
            filter_config['min_obs_cv'] = float(min_cv)

        results = mgm.calculate_multi_gauge_metrics(
            output_path=output_dir,
            gauge_ids=gauge_ids,
            start_date=start_date,
            end_date=end_date,
            min_gauges=min_gauges,
            aggregation=aggregation,
            filter_config=filter_config or None,
            kge_floor=kge_floor,
        )

        score = results.get('kge', self.penalty_score)
        return {
            'kge': float(score) if score is not None else self.penalty_score,
            'kge_std': results.get('kge_std', 0.0),
            'kge_min': results.get('kge_min', 0.0),
            'kge_max': results.get('kge_max', 0.0),
            'n_gauges': results.get('n_valid_gauges', 0),
        }

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_hype_parameters_worker(task_data)


def _evaluate_hype_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    import os
    import random
    import signal
    import sys
    import time
    import traceback

    from symfluence.core.constants import ModelDefaults

    # Set up signal handler for clean termination
    def signal_handler(signum, frame):
        sys.exit(1)

    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        pass  # Signal handling not available in this context

    # Force single-threaded execution for parallel workers
    os.environ.update({
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    })

    # Add small random delay to prevent file system contention
    initial_delay = random.uniform(0.1, 0.5)
    time.sleep(initial_delay)

    try:
        worker = HYPEWorker(config=task_data.get('config'))
        task = WorkerTask.from_legacy_dict(task_data)
        result = worker.evaluate(task)
        return result.to_legacy_dict()
    except Exception as e:  # noqa: BLE001 — calibration resilience
        return {
            'individual_id': task_data.get('individual_id', -1),
            'params': task_data.get('params', {}),
            'score': ModelDefaults.PENALTY_SCORE,
            'error': f'Critical HYPE worker exception: {str(e)}\n{traceback.format_exc()}',
            'proc_id': task_data.get('proc_id', -1)
        }
