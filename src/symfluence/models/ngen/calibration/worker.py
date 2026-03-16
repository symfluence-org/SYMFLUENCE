# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
NextGen (NGEN) Worker

Worker implementation for NextGen model optimization.
Delegates to existing worker functions while providing BaseWorker interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel as PydanticBaseModel

from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.optimization.registry import OptimizerRegistry
from symfluence.optimization.workers.base_worker import BaseWorker, WorkerTask

logger = logging.getLogger(__name__)


@OptimizerRegistry.register_worker('NGEN')
class NgenWorker(BaseWorker):
    """
    Worker for NextGen (ngen) model calibration.

    Handles parameter application to JSON config files, ngen execution,
    and metric calculation for streamflow calibration.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ngen worker.

        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        super().__init__(config, logger)

    def _check_cfe_parameter_feasibility(self, params: Dict[str, float]) -> bool:
        """
        Check if CFE parameter combinations are numerically feasible.

        Certain parameter combinations cause segfaults (SIGSEGV, exit code -11)
        in CFE's Fortran solver. This pre-flight check rejects known crash-prone
        combinations, avoiding wasted model execution time.

        Args:
            params: Parameter values (MODULE.param format, e.g., 'CFE.satdk')

        Returns:
            True if parameters are feasible, False if they would likely crash
        """
        # Extract CFE parameters (strip module prefix)
        cfe = {}
        for k, v in params.items():
            if k.startswith('CFE.'):
                cfe[k[4:]] = v

        if not cfe:
            return True  # No CFE params to check

        satdk = cfe.get('satdk')
        bb = cfe.get('bb')
        slop = cfe.get('slop')
        k_nash = cfe.get('K_nash', cfe.get('Kn'))
        k_lf = cfe.get('K_lf', cfe.get('Klf'))
        cgw = cfe.get('Cgw')
        max_gw = cfe.get('max_gw_storage')

        # Check 1: Low satdk + high bb → Brooks-Corey water retention singularity
        if satdk is not None and bb is not None:
            if satdk < 5e-7 and bb > 6.0:
                self.logger.warning(
                    f"Infeasible CFE params: satdk={satdk:.2e} + bb={bb:.1f} "
                    f"(Brooks-Corey singularity risk). Skipping trial."
                )
                return False

        # Check 2: Low satdk + high slope → infiltration instability
        if satdk is not None and slop is not None:
            if satdk < 5e-7 and slop > 0.3:
                self.logger.warning(
                    f"Infeasible CFE params: satdk={satdk:.2e} + slop={slop:.2f} "
                    f"(infiltration instability). Skipping trial."
                )
                return False

        # Check 3: Routing coefficients sum too high → CFL violation
        if k_nash is not None and k_lf is not None:
            if k_nash + k_lf > 0.9:
                self.logger.warning(
                    f"Infeasible CFE params: K_nash={k_nash:.3f} + K_lf={k_lf:.3f} "
                    f"= {k_nash + k_lf:.3f} > 0.9 (routing instability). Skipping trial."
                )
                return False

        # Check 4: High groundwater storage + high Cgw → mass balance violation
        if max_gw is not None and cgw is not None:
            if max_gw > 1.5 and cgw > 0.005:
                self.logger.warning(
                    f"Infeasible CFE params: max_gw_storage={max_gw:.2f} + Cgw={cgw:.4f} "
                    f"(groundwater mass balance risk). Skipping trial."
                )
                return False

        return True

    def _check_sacsma_parameter_feasibility(self, params: Dict[str, float]) -> bool:
        """
        Check if SAC-SMA parameter combinations are physically feasible.

        Args:
            params: Parameter values (MODULE.param format, e.g., 'SACSMA.UZTWM')

        Returns:
            True if parameters are feasible, False if infeasible
        """
        sac = {}
        for k, v in params.items():
            if k.startswith('SACSMA.'):
                sac[k[7:]] = v

        if not sac:
            return True

        # Check: UZTWM < LZTWM (upper zone tension water < lower zone)
        uztwm = sac.get('UZTWM')
        lztwm = sac.get('LZTWM')
        if uztwm is not None and lztwm is not None:
            if uztwm >= lztwm:
                self.logger.warning(
                    f"Infeasible SAC-SMA params: UZTWM={uztwm:.1f} >= LZTWM={lztwm:.1f}. "
                    f"Skipping trial."
                )
                return False

        # Check: PCTIM + ADIMP < 1.0 (total impervious fraction)
        pctim = sac.get('PCTIM')
        adimp = sac.get('ADIMP')
        if pctim is not None and adimp is not None:
            if pctim + adimp >= 1.0:
                self.logger.warning(
                    f"Infeasible SAC-SMA params: PCTIM={pctim:.3f} + ADIMP={adimp:.3f} "
                    f"= {pctim + adimp:.3f} >= 1.0. Skipping trial."
                )
                return False

        return True

    def apply_parameters(
        self,
        params: Dict[str, float],
        settings_dir: Path,
        **kwargs
    ) -> bool:
        """
        Apply parameters to ngen configuration files (JSON, BMI text, or TBL).

        Parameters use MODULE.param naming convention (e.g., CFE.Kn).

        Args:
            params: Parameter values to apply (MODULE.param format)
            settings_dir: Ngen settings directory (isolated for parallel workers)
            **kwargs: Additional arguments including 'config'

        Returns:
            True if successful
        """
        try:
            # Pre-flight feasibility checks: reject parameter combinations
            # known to cause crashes or physical inconsistencies
            if not self._check_cfe_parameter_feasibility(params):
                return False
            if not self._check_sacsma_parameter_feasibility(params):
                return False
            # Import NgenParameterManager
            from .parameter_manager import NgenParameterManager

            # Ensure settings_dir is a Path
            if isinstance(settings_dir, str):
                settings_dir = Path(settings_dir)

            # settings_dir may already be the NGEN directory (e.g., .../settings/NGEN)
            # or it may be the parent (e.g., .../settings/)
            if settings_dir.name == 'NGEN':
                ngen_dir = settings_dir
            else:
                ngen_dir = settings_dir / 'NGEN'

            if not ngen_dir.exists():
                # Attempt to recover by re-copying from source settings
                config = kwargs.get('config', self.config)
                if config:
                    import shutil
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    domain_name = config.get('DOMAIN_NAME', '')
                    source_ngen = data_dir / f"domain_{domain_name}" / 'settings' / 'NGEN'
                    if source_ngen.exists():
                        self.logger.warning(
                            f"NGEN settings directory missing at {ngen_dir}, "
                            f"recovering from {source_ngen}"
                        )
                        ngen_dir.mkdir(parents=True, exist_ok=True)
                        for item in source_ngen.iterdir():
                            dest = ngen_dir / item.name
                            if item.is_file():
                                shutil.copy2(item, dest)
                            elif item.is_dir():
                                if dest.exists():
                                    shutil.rmtree(dest)
                                shutil.copytree(item, dest)
                    else:
                        self.logger.error(f"NGEN settings directory not found: {ngen_dir}")
                        return False
                else:
                    self.logger.error(f"NGEN settings directory not found: {ngen_dir}")
                    return False
            else:
                # Sync critical config files if source has updates
                config = kwargs.get('config', self.config)
                if config:
                    data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
                    domain_name = config.get('DOMAIN_NAME', '')
                    source_ngen = data_dir / f"domain_{domain_name}" / 'settings' / 'NGEN'

                    # Sync realization_config.json if source has updated forcing mappings
                    src_realization = source_ngen / "realization_config.json"
                    dst_realization = ngen_dir / "realization_config.json"
                    if src_realization.exists():
                        try:
                            src_text = src_realization.read_text(encoding='utf-8')
                            dst_text = dst_realization.read_text(encoding='utf-8') if dst_realization.exists() else ""
                            if "atmosphere_air_water~vapor__specific_humidity" in src_text and \
                               "atmosphere_air_water~vapor__relative_saturation" in dst_text:
                                import shutil
                                shutil.copy2(src_realization, dst_realization)
                                self.logger.debug(
                                    "Synced realization_config.json (specific humidity mapping)"
                                )
                        except OSError as e:
                            self.logger.debug(f"Could not sync realization_config.json: {e}")

                    # Sync CFE configs if source has catchment_area_km2 and destination differs
                    # This is critical for CFE to output m³/s instead of depth units
                    src_cfe_dir = source_ngen / "CFE"
                    dst_cfe_dir = ngen_dir / "CFE"
                    if src_cfe_dir.exists() and dst_cfe_dir.exists():
                        try:
                            import re
                            import shutil
                            for src_cfe_file in src_cfe_dir.glob("*_bmi_config_cfe*.txt"):
                                dst_cfe_file = dst_cfe_dir / src_cfe_file.name
                                if dst_cfe_file.exists():
                                    src_content = src_cfe_file.read_text(encoding='utf-8')
                                    dst_content = dst_cfe_file.read_text(encoding='utf-8')
                                    # Extract catchment_area_km2 values to compare
                                    area_re = re.compile(r'catchment_area_km2=([0-9.eE+-]+)')
                                    src_match = area_re.search(src_content)
                                    dst_match = area_re.search(dst_content)
                                    # Sync if source has area but destination doesn't,
                                    # or if the values differ significantly
                                    should_sync = False
                                    if src_match and not dst_match:
                                        should_sync = True
                                    elif src_match and dst_match:
                                        try:
                                            src_area = float(src_match.group(1))
                                            dst_area = float(dst_match.group(1))
                                            # Sync if values differ by more than 0.1%
                                            if abs(src_area - dst_area) / max(src_area, 1e-10) > 0.001:
                                                should_sync = True
                                                self.logger.debug(
                                                    f"CFE area mismatch: source={src_area:.2f}, dest={dst_area:.2f}"
                                                )
                                        except ValueError:
                                            should_sync = True  # Can't parse, sync to be safe
                                    if should_sync:
                                        shutil.copy2(src_cfe_file, dst_cfe_file)
                                        self.logger.debug(
                                            f"Synced CFE config with catchment_area_km2: {src_cfe_file.name}"
                                        )
                        except OSError as e:
                            self.logger.debug(f"Could not sync CFE configs: {e}")

            # Use NgenParameterManager to update files
            # It handles CFE, NOAH (namelists and TBLs), and PET
            config = kwargs.get('config', self.config)
            param_manager = NgenParameterManager(config, self.logger, ngen_dir)

            success = param_manager.update_model_files(params)

            if success:
                self.logger.debug(f"Applied {len(params)} parameter updates via NgenParameterManager in {ngen_dir}")
            else:
                self.logger.error(f"NgenParameterManager failed to update model files in {ngen_dir}")

            return success

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error applying ngen parameters: {e}")
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
        Run ngen model.

        Supports both serial and parallel execution modes.

        Args:
            config: Configuration dictionary
            settings_dir: Ngen settings directory
            output_dir: Output directory
            **kwargs: Additional arguments including parallel config keys

        Returns:
            True if model ran successfully
        """
        try:
            # Import NgenRunner
            from symfluence.models.ngen import NgenRunner

            if isinstance(config, PydanticBaseModel):
                experiment_id = getattr(getattr(config, 'domain', None), 'experiment_id', None)
            else:
                experiment_id = config.get('EXPERIMENT_ID')

            # Initialize runner with isolated directories as constructor kwargs
            runner = NgenRunner(
                config, self.logger,
                ngen_settings_dir=settings_dir,
                ngen_output_dir=output_dir,
            )
            success = runner.run_ngen(experiment_id)

            return success

        except FileNotFoundError as e:
            self.logger.error(f"Required ngen input file not found: {e}")
            return False
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error running ngen: {e}")
            return False

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate metrics from ngen output.

        Args:
            output_dir: Directory containing model outputs (isolated)
            config: Configuration dictionary
            **kwargs: Additional arguments

        Returns:
            Dictionary of metric names to values
        """
        try:
            # Try to use calibration target
            from symfluence.optimization.calibration_targets import NgenStreamflowTarget

            domain_name = config.get('DOMAIN_NAME')
            experiment_id = config.get('EXPERIMENT_ID')
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"

            # Create calibration target
            target = NgenStreamflowTarget(config, project_dir, self.logger)

            # Calculate metrics using isolated output_dir
            # NgenStreamflowTarget needs to be aware of the isolated directory
            metrics = target.calculate_metrics(experiment_id=experiment_id, output_dir=output_dir)  # type: ignore[call-arg]

            # Normalize metric keys to lowercase
            return {k.lower(): float(v) for k, v in metrics.items()}

        except ImportError:
            # Fallback: Calculate metrics directly
            return self._calculate_metrics_direct(output_dir, config)

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating ngen metrics: {e}")
            return {'kge': self.penalty_score}

    def _calculate_metrics_direct(
        self,
        output_dir: Path,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate metrics directly from ngen output files.

        Args:
            output_dir: Output directory (isolated)
            config: Configuration dictionary

        Returns:
            Dictionary of metrics
        """
        try:
            import pandas as pd

            from symfluence.evaluation.metrics import kge, nse

            domain_name = config.get('DOMAIN_NAME')

            # Find ngen output in isolated output_dir
            output_files = list(output_dir.glob('*.csv')) + list(output_dir.glob('*.nc'))

            if not output_files:
                return {'kge': self.penalty_score, 'error': 'No output files found'}

            # Read simulation
            if output_files[0].suffix == '.csv':
                sim_df = pd.read_csv(output_files[0], index_col=0, parse_dates=True)
                if 'q_cms' in sim_df.columns:
                    sim = sim_df['q_cms'].values
                else:
                    sim = sim_df.iloc[:, 0].values
            else:
                import xarray as xr
                with xr.open_dataset(output_files[0]) as ds:
                    # Generic extraction - pick first data variable
                    var = next(iter(ds.data_vars))
                    sim = ds[var].values.flatten()

            # Load observations
            data_dir = Path(config.get('SYMFLUENCE_DATA_DIR', '.'))
            project_dir = data_dir / f"domain_{domain_name}"
            obs_file = (resolve_data_subdir(project_dir, 'observations') / 'streamflow' / 'preprocessed' /
                       f'{domain_name}_streamflow_processed.csv')

            if not obs_file.exists():
                return {'kge': self.penalty_score, 'error': 'Observations not found'}

            obs_df = pd.read_csv(obs_file, index_col='datetime', parse_dates=True)

            # Robust alignment (CSV with index) or simple length match (NetCDF/fallback)
            if 'sim_df' in locals() and sim_df is not None:
                # Handle potential index mismatch (timezone, etc)
                if hasattr(sim_df.index, 'tz_localize'):
                     # Ensure sim is timezone-naive or matches obs
                     if sim_df.index.tz is not None:
                         sim_df.index = sim_df.index.tz_convert(None)

                sim_series = pd.Series(sim, index=sim_df.index)
                common_idx = sim_series.index.intersection(obs_df.index)

                if common_idx.empty:
                    self.logger.warning("No overlapping dates between simulation and observation")
                    return {'kge': self.penalty_score, 'error': 'No overlapping dates'}

                sim_vals = sim_series.loc[common_idx].values
                obs_vals = obs_df.loc[common_idx, 'discharge_cms'].values
            else:
                # Fallback: Simple length truncation
                min_len = min(len(sim), len(obs_df))
                sim_vals = sim[:min_len]
                obs_vals = obs_df['discharge_cms'].values[:min_len]

            kge_val = kge(obs_vals, sim_vals, transfo=1)
            nse_val = nse(obs_vals, sim_vals, transfo=1)

            return {'kge': float(kge_val), 'nse': float(nse_val)}

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error in direct ngen metrics calculation: {e}")
            return {'kge': self.penalty_score}

    @staticmethod
    def evaluate_worker_function(task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Static worker function for process pool execution.

        Args:
            task_data: Task dictionary

        Returns:
            Result dictionary
        """
        return _evaluate_ngen_parameters_worker(task_data)


def _evaluate_ngen_parameters_worker(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Module-level worker function for MPI/ProcessPool execution.

    Args:
        task_data: Task dictionary

    Returns:
        Result dictionary
    """
    worker = NgenWorker(config=task_data.get('config'))
    task = WorkerTask.from_legacy_dict(task_data)
    result = worker.evaluate(task)
    return result.to_legacy_dict()
