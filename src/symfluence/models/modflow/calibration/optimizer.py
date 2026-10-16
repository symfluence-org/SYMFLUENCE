# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Coupled Groundwater Model Optimizer

Generic optimizer for any land surface model coupled with MODFLOW
groundwater. The land surface model is determined by LAND_SURFACE_MODEL
config key. Copies both land surface and MODFLOW settings to parallel
directories and delegates execution to CoupledGWWorker.

When dCoupler is available, uses CouplingGraphBuilder for graph-based
coupling. Otherwise falls back to sequential file-based coupling.

No external routing is needed because MODFLOW drain discharge provides
the baseflow component directly.

Config keys:
    HYDROLOGICAL_MODEL: COUPLED_GW
    LAND_SURFACE_MODEL: SUMMA, CLM, MESH, etc.
    GROUNDWATER_MODEL: MODFLOW
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from symfluence.core.calibration.optimizers.base_model_optimizer import BaseModelOptimizer
from symfluence.core.file_utils import copy_file
from symfluence.core.registries import R

from .targets import CoupledGWStreamflowTarget  # noqa: F401 - triggers target registration
from .worker import CoupledGWWorker  # noqa: F401 - triggers worker registration


@R.optimizers.add('COUPLED_GW')
class CoupledGWModelOptimizer(BaseModelOptimizer):
    """Optimizer for coupled land-surface + MODFLOW calibration.

    Dynamically resolves the land surface model from config and sets up
    parallel directories with both land surface and MODFLOW settings.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: logging.Logger,
        optimization_settings_dir: Optional[Path] = None,
        reporting_manager: Optional[Any] = None,
    ):
        self.config = config
        self.land_model_name = self._resolve_land_model_name(config)
        super().__init__(
            config, logger, optimization_settings_dir,
            reporting_manager=reporting_manager,
        )
        self.logger.debug(
            f"CoupledGWModelOptimizer initialized with "
            f"land_surface={self.land_model_name}"
        )

    @staticmethod
    def _resolve_land_model_name(config: Dict[str, Any]) -> str:
        """Resolve land surface model name from config."""
        from symfluence.models.modflow.land_adapters import resolve_land_model
        return resolve_land_model(config)

    def _get_model_name(self) -> str:
        return 'COUPLED_GW'

    def _get_final_file_manager_path(self) -> Path:
        """Get path to the land surface model's file manager for final evaluation."""
        # SUMMA uses fileManager.txt; other models may differ
        if self.land_model_name == 'SUMMA':
            summa_fm = self._get_config_value(
                lambda: self.config.model.summa.filemanager,
                default='fileManager.txt',
                dict_key='SETTINGS_SUMMA_FILEMANAGER',
            )
            if summa_fm == 'default':
                summa_fm = 'fileManager.txt'
            return self.project_dir / 'settings' / 'SUMMA' / summa_fm

        # For other models, return a generic path
        return self.project_dir / 'settings' / self.land_model_name

    def _create_parameter_manager(self):
        """Create joint land-surface + MODFLOW parameter manager."""
        from .parameter_manager import CoupledGWParameterManager
        settings_dir = self.project_dir / 'settings'
        return CoupledGWParameterManager(
            self.config, self.logger, settings_dir,
        )

    def _check_routing_needed(self) -> bool:
        """No external routing needed; MODFLOW provides baseflow directly."""
        return False

    def _ensure_modflow_settings(self) -> None:
        """Generate MODFLOW settings files if they don't exist."""
        source_modflow = self.project_dir / 'settings' / 'MODFLOW'
        if source_modflow.exists() and any(source_modflow.iterdir()):
            return

        self.logger.info("MODFLOW settings not found, generating from config...")
        from symfluence.models.modflow.preprocessor import MODFLOWPreProcessor
        preprocessor = MODFLOWPreProcessor(self.config, self.logger)
        # Ensure config_dict is populated for dict-based configs
        if isinstance(self.config, dict) and not preprocessor.config_dict:
            preprocessor.config_dict = self.config
        preprocessor.project_dir = self.project_dir
        preprocessor.run_preprocessing()

    def _extend_parallel_sim_end(
        self,
        land_dirs: Dict[int, Dict[str, Path]],
        fm_name: str,
    ) -> None:
        """Overwrite simEndTime in parallel fileManagers with full experiment end.

        The base ``update_file_managers`` sets simEndTime to the calibration
        period end, which is correct for uncoupled models (faster runs, same
        calibration-window results).  For COUPLED_GW the MODFLOW transient
        simulation is path-dependent: groundwater heads — and therefore drain
        discharge — for the calibration window change when driven by a longer
        recharge series.  To keep DDS iteration scores consistent with the
        final evaluation (which always uses the full period), we extend
        simEndTime here.
        """
        sim_end = self._get_config_value(
            lambda: self.config.domain.time_end,
            dict_key='EXPERIMENT_TIME_END',
        )
        if not sim_end:
            return

        sim_end = self._adjust_end_time_for_forcing(sim_end)

        for proc_id, dirs in land_dirs.items():
            fm_path = dirs['settings_dir'] / fm_name
            if not fm_path.exists():
                continue
            try:
                with open(fm_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                updated = []
                for line in lines:
                    if 'simEndTime' in line and not line.strip().startswith('!'):
                        updated.append(f"simEndTime           '{sim_end}'\n")
                    else:
                        updated.append(line)

                with open(fm_path, 'w', encoding='utf-8') as f:
                    f.writelines(updated)

                self.logger.debug(
                    f"Extended simEndTime to {sim_end} for coupled process {proc_id}"
                )
            except (IOError, OSError) as e:
                self.logger.error(
                    f"Failed to extend simEndTime for process {proc_id}: {e}"
                )

    def _setup_parallel_dirs(self) -> None:
        """Setup parallel directories with both land surface and MODFLOW settings."""
        self._ensure_modflow_settings()

        algorithm = self._get_config_value(
            lambda: self.config.optimization.algorithm,
            default='optimization',
            dict_key='ITERATIVE_OPTIMIZATION_ALGORITHM',
        ).lower()

        base_dir = self._resolve_sim_base_dir(algorithm)

        self.parallel_dirs = self.setup_parallel_processing(
            base_dir, self.land_model_name, self.experiment_id,
        )

        # setup_parallel_processing created dirs for the land model:
        #   sim_dir     = process_0/simulations/run_1/SUMMA
        #   settings_dir = process_0/settings/SUMMA
        # Save those as land aliases, then lift sim_dir/settings_dir to the
        # experiment level so the worker (which creates SUMMA/ and MODFLOW/
        # subdirs) gets the right base paths.
        for proc_id, dirs in self.parallel_dirs.items():
            dirs['land_dir'] = dirs['sim_dir']
            dirs['land_settings_dir'] = dirs['settings_dir']

            dirs['sim_dir'] = dirs['sim_dir'].parent
            dirs['settings_dir'] = dirs['root'] / 'settings'

            dirs['modflow_dir'] = dirs['sim_dir'] / 'MODFLOW'
            dirs['modflow_settings_dir'] = dirs['root'] / 'settings' / 'MODFLOW'

            dirs['modflow_dir'].mkdir(parents=True, exist_ok=True)
            dirs['modflow_settings_dir'].mkdir(parents=True, exist_ok=True)

        # Copy land surface model settings
        source_land = self.project_dir / 'settings' / self.land_model_name
        if source_land.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['land_settings_dir']
                for item in source_land.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

            # Update file managers for SUMMA (other models may not need this)
            if self.land_model_name == 'SUMMA':
                fm_name = self._get_config_value(
                    lambda: self.config.model.summa.filemanager,
                    default='fileManager.txt',
                    dict_key='SETTINGS_SUMMA_FILEMANAGER',
                )
                if fm_name == 'default':
                    fm_name = 'fileManager.txt'
                # Build dirs with settings_dir and sim_dir pointing to SUMMA
                # subdirectories so update_file_managers finds the fileManager
                # and sets correct output paths
                land_dirs = {}
                for proc_id, dirs in self.parallel_dirs.items():
                    land_dirs[proc_id] = dict(dirs)
                    land_dirs[proc_id]['settings_dir'] = dirs['land_settings_dir']
                    land_dirs[proc_id]['sim_dir'] = dirs['land_dir']
                self.update_file_managers(
                    land_dirs,
                    'SUMMA',
                    self.experiment_id,
                    fm_name,
                )

                # COUPLED_GW fix: extend simEndTime to full experiment period.
                # The base update_file_managers truncates simEndTime to the
                # calibration end, but MODFLOW is a transient GW solver whose
                # drain discharge for the calibration window depends on the
                # full recharge history.  Running only through the calibration
                # end during iterations but the full period during final
                # evaluation produces different GW heads and thus different
                # KGE scores for the same calibration window.
                self._extend_parallel_sim_end(land_dirs, fm_name)

        # Copy MODFLOW settings
        source_modflow = self.project_dir / 'settings' / 'MODFLOW'
        if source_modflow.exists():
            for proc_id, dirs in self.parallel_dirs.items():
                dest = dirs['modflow_settings_dir']
                for item in source_modflow.iterdir():
                    if item.is_file():
                        copy_file(item, dest / item.name)

    def _apply_best_parameters_for_final(self, best_params: Dict[str, float]) -> bool:
        """Apply best parameters to project-level settings for final evaluation.

        Instead of re-applying parameters to the project-level settings (which
        can silently fail when the project-level trialParams.nc is missing
        variables that were added during iteration setup), we copy the
        already-correct settings files from the parallel directory.  The
        parallel dir's SUMMA trialParams.nc and MODFLOW files are guaranteed
        to have the right values because they were successfully used in the
        best DDS iteration.

        Falls back to direct parameter application if parallel dirs are not
        available (e.g. single-process mode).
        """
        settings_dir = self.project_dir / 'settings'

        # Try copying from the parallel directory first (reliable path)
        if hasattr(self, 'parallel_dirs') and self.parallel_dirs:
            proc_dirs = next(iter(self.parallel_dirs.values()))
            try:
                # First apply params to the parallel dir (which has the right
                # trialParams.nc structure)
                parallel_settings = proc_dirs['settings_dir']
                self.worker.apply_parameters(
                    best_params, parallel_settings, config=self.config
                )

                # Copy calibrated SUMMA settings to project level
                parallel_land = proc_dirs.get('land_settings_dir')
                project_land = settings_dir / self.land_model_name
                if parallel_land and parallel_land.exists():
                    for item in parallel_land.iterdir():
                        if item.is_file():
                            copy_file(item, project_land / item.name)
                    self.logger.info(
                        f"Copied calibrated {self.land_model_name} settings "
                        f"from parallel dir to project settings"
                    )

                # Copy calibrated MODFLOW settings to project level
                parallel_mf = proc_dirs.get('modflow_settings_dir')
                project_mf = settings_dir / 'MODFLOW'
                if parallel_mf and parallel_mf.exists():
                    for item in parallel_mf.iterdir():
                        if item.is_file():
                            copy_file(item, project_mf / item.name)
                    self.logger.info(
                        "Copied calibrated MODFLOW settings "
                        "from parallel dir to project settings"
                    )

                return True

            except (ValueError, IOError, RuntimeError) as e:
                self.logger.warning(
                    f"Parallel dir copy failed ({e}), falling back to "
                    f"direct parameter application"
                )

        # Fallback: apply parameters directly to project-level settings
        try:
            return self.worker.apply_parameters(
                best_params, settings_dir, config=self.config
            )
        except (ValueError, IOError, RuntimeError) as e:
            self.logger.error(f"Error applying parameters for final evaluation: {e}")
            return False

    def _run_model_for_final_evaluation(self, output_dir: Path) -> bool:
        """Run coupled model for final evaluation."""
        settings_dir = self.project_dir / 'settings'
        return self.worker.run_model(
            self.config, settings_dir, output_dir,
        )

    def _update_file_manager_output_path(self, output_dir: Path) -> None:
        """Update SUMMA file manager to write output into the land-model subdir.

        The base class sets outputPath to ``output_dir/`` but the coupled
        target expects SUMMA output in ``output_dir/SUMMA/`` and MODFLOW
        output in ``output_dir/MODFLOW/``.  We also restore settingsPath
        to the project-level SUMMA settings so SUMMA reads the trialParams
        we just wrote with _apply_best_parameters_for_final.
        """
        file_manager_path = self._get_final_file_manager_path()
        if not file_manager_path.exists() or not file_manager_path.is_file():
            return

        try:
            with open(file_manager_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            land_output = str(output_dir / self.land_model_name)
            if not land_output.endswith('/'):
                land_output += '/'

            land_settings = str(self.project_dir / 'settings' / self.land_model_name)
            if not land_settings.endswith('/'):
                land_settings += '/'

            updated_lines = []
            for line in lines:
                if 'outputPath' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"outputPath '{land_output}' \n")
                elif 'settingsPath' in line and not line.strip().startswith('!'):
                    updated_lines.append(f"settingsPath '{land_settings}' \n")
                else:
                    updated_lines.append(line)

            with open(file_manager_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

            self.logger.debug(
                f"Updated file manager: outputPath={land_output}, "
                f"settingsPath={land_settings}"
            )

        except (FileNotFoundError, IOError, ValueError) as e:
            self.logger.error(f"Failed to update file manager output path: {e}")

    def _update_file_manager_for_final_run(self) -> None:
        """Update land surface model file manager for full experiment period."""
        super()._update_file_manager_for_final_run()

    def _restore_file_manager_for_optimization(self) -> None:
        """Restore land surface model file manager to calibration period."""
        super()._restore_file_manager_for_optimization()
