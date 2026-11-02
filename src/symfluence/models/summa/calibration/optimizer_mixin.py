# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA-Specific Optimizer Mixin

Contains SUMMA-specific functionality extracted from the legacy BaseOptimizer.
This mixin provides backward compatibility for algorithm optimizers that need
SUMMA-specific behavior while enabling cleaner separation of concerns.

For new implementations, prefer using SUMMAModelOptimizer which extends
BaseModelOptimizer with proper abstract method patterns.
"""
from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from symfluence.core.calibration.optimizers.lifecycle import adjust_end_time_for_forcing
from symfluence.core.exceptions import FileOperationError
from symfluence.core.modeling.utilities.periods import split_period

if TYPE_CHECKING:
    pass


class SUMMAOptimizerMixin:
    """Mixin providing SUMMA-specific functionality for optimizers.

    This mixin extracts SUMMA-specific methods from the legacy BaseOptimizer
    to enable cleaner separation of concerns. Classes using this mixin should
    have the following attributes available:

    Required attributes:
        - logger: Python logger instance
        - project_dir: Path to project directory
        - optimization_dir: Path to optimization directory
        - optimization_settings_dir: Path to optimization settings directory
        - config_dict: Dictionary with configuration values
        - _config: Optional typed SymfluenceConfig

    The mixin provides:
        - SUMMA file manager updates
        - mizuRoute control file updates
        - SUMMA settings file copying
        - Glacier mode detection and handling
    """

    # =========================================================================
    # SUMMA Directory Properties
    # =========================================================================

    @property
    def summa_sim_dir(self) -> Path:
        """Get SUMMA simulation output directory."""
        if not hasattr(self, '_summa_sim_dir'):
            self._summa_sim_dir = self.optimization_dir / "SUMMA"
        return self._summa_sim_dir

    @summa_sim_dir.setter
    def summa_sim_dir(self, value: Path) -> None:
        self._summa_sim_dir = value

    @property
    def mizuroute_sim_dir(self) -> Path:
        """Get mizuRoute simulation output directory."""
        if not hasattr(self, '_mizuroute_sim_dir'):
            self._mizuroute_sim_dir = self.optimization_dir / "mizuRoute"
        return self._mizuroute_sim_dir

    @mizuroute_sim_dir.setter
    def mizuroute_sim_dir(self, value: Path) -> None:
        self._mizuroute_sim_dir = value

    # =========================================================================
    # SUMMA Configuration Helpers
    # =========================================================================

    def _get_summa_file_manager_name(self) -> str:
        """Get SUMMA file manager filename from config."""
        if hasattr(self, '_config') and self._config is not None:
            try:
                if self._config.model.summa:
                    return self._config.model.summa.filemanager or 'fileManager.txt'
            except (AttributeError, TypeError):
                pass
        return self._get_config_value_safe('SETTINGS_SUMMA_FILEMANAGER', 'fileManager.txt')

    def _is_glacier_mode(self) -> bool:
        """Determine if glacier mode is enabled based on file manager name."""
        summa_fm = self._get_summa_file_manager_name()
        return 'glac' in summa_fm.lower()

    # =========================================================================
    # SUMMA Settings File Management
    # =========================================================================

    def _copy_summa_settings_files(self) -> None:
        """Copy necessary SUMMA settings files from project to optimization directory.

        Copies SUMMA configuration and control files needed for model runs.
        Handles two modes:
        1. **Regular mode**: Uses standard attributes.nc and coldState.nc
        2. **Glacier mode**: Tries glacier-specific files first, falls back to regular

        Files Copied:
            Required (all must exist):
            - modelDecisions.txt: SUMMA process choices
            - outputControl.txt: SUMMA output configuration
            - localParamInfo.txt: Local parameter definitions
            - basinParamInfo.txt: Basin parameter definitions
            - fileManager.txt (or custom): SUMMA file manager

            Conditional (glacier mode):
            - attributes.nc or attributes_glac.nc: Domain attributes
            - coldState.nc or coldState_glac.nc: Initial conditions

            Optional (copy if exists):
            - *.TBL files: Parameter tables (GENPARM, MPTABLE, SOILPARM, VEGPARM)
            - trialParams.nc: Trial parameter file
            - forcingFileList.txt: Forcing file list

            mizuRoute settings (if exists):
            - All files in settings/mizuRoute subdirectory

        Raises:
            FileNotFoundError: If required file or attributes file not found
        """
        source_settings_dir = self.project_dir / "settings" / "SUMMA"

        if not source_settings_dir.exists():
            raise FileOperationError(f"Source SUMMA settings directory not found: {source_settings_dir}")

        glacier_mode = self._is_glacier_mode()
        summa_fm = self._get_summa_file_manager_name()

        required_files = [
            'modelDecisions.txt', 'outputControl.txt',
            'localParamInfo.txt', 'basinParamInfo.txt',
        ]

        # Add appropriate file manager
        if summa_fm and summa_fm != 'default':
            required_files.append(summa_fm)
        else:
            required_files.append('fileManager.txt')

        # Add attributes and coldState files based on glacier mode
        if glacier_mode:
            attr_files = ['attributes_glac.nc', 'attributes.nc']
            cold_files = ['coldState_glac.nc', 'coldState.nc']
        else:
            attr_files = ['attributes.nc']
            cold_files = ['coldState.nc']

        optional_files = [
            'TBL_GENPARM.TBL', 'TBL_MPTABLE.TBL', 'TBL_SOILPARM.TBL', 'TBL_VEGPARM.TBL',
            'trialParams.nc', 'forcingFileList.txt',
        ]

        # Copy required files
        for file_name in required_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied required file: {file_name}")
            else:
                raise FileOperationError(f"Required SUMMA settings file not found: {source_path}")

        # Copy attributes files (glacier mode tries glacier files first)
        attr_copied = False
        for attr_file in attr_files:
            source_path = source_settings_dir / attr_file
            if source_path.exists():
                dest_path = self.optimization_settings_dir / attr_file
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied attributes file: {attr_file}")
                attr_copied = True
                break
        if not attr_copied:
            raise FileOperationError(f"No attributes file found in {source_settings_dir}")

        # Copy coldState files (glacier mode tries glacier files first)
        cold_copied = False
        for cold_file in cold_files:
            source_path = source_settings_dir / cold_file
            if source_path.exists():
                dest_path = self.optimization_settings_dir / cold_file
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied coldState file: {cold_file}")
                cold_copied = True
                break
        if not cold_copied:
            self.logger.warning("coldState.nc not found, depth calibration may fail if enabled.")

        # Copy optional files
        for file_name in optional_files:
            source_path = source_settings_dir / file_name
            dest_path = self.optimization_settings_dir / file_name

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                self.logger.debug(f"Copied optional file: {file_name}")
            else:
                self.logger.debug(f"Optional SUMMA settings file not found: {source_path}")

        # Copy mizuRoute settings if they exist
        self._copy_mizuroute_settings()

    def _copy_mizuroute_settings(self) -> None:
        """Copy mizuRoute settings files if they exist."""
        source_mizu_dir = self.project_dir / "settings" / "mizuRoute"
        dest_mizu_dir = self.optimization_dir / "settings" / "mizuRoute"

        if source_mizu_dir.exists():
            dest_mizu_dir.mkdir(parents=True, exist_ok=True)
            for mizu_file in source_mizu_dir.glob("*"):
                if mizu_file.is_file():
                    shutil.copy2(mizu_file, dest_mizu_dir / mizu_file.name)
            self.logger.debug("Copied mizuRoute settings")

    # =========================================================================
    # SUMMA File Manager Updates
    # =========================================================================

    def _update_summa_file_managers(self) -> None:
        """Update SUMMA and mizuRoute file managers for optimization runs."""
        summa_fm_name = self._get_summa_file_manager_name()
        file_manager_path = self.optimization_settings_dir / summa_fm_name
        if not file_manager_path.exists():
            file_manager_path = self.optimization_settings_dir / 'fileManager.txt'
        if file_manager_path.exists():
            self._update_summa_file_manager(file_manager_path)

        mizu_control_path = self.optimization_dir / "settings" / "mizuRoute" / "mizuroute.control"
        if mizu_control_path.exists():
            self._update_mizuroute_control_file(mizu_control_path)

    def _update_summa_file_manager(self, file_manager_path: Path, use_calibration_period: bool = True) -> None:
        """Update SUMMA file manager with spinup + calibration period.

        Args:
            file_manager_path: Path to file manager file
            use_calibration_period: Whether to use calibration period (vs full experiment period)
        """
        with open(file_manager_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        sim_start, sim_end = self._get_simulation_period(use_calibration_period)

        updated_lines = []
        for line in lines:
            if 'simStartTime' in line:
                updated_lines.append(f"simStartTime         '{sim_start}'\n")
            elif 'simEndTime' in line:
                updated_lines.append(f"simEndTime           '{sim_end}'\n")
            elif 'outputPath' in line:
                output_path = str(self.summa_sim_dir).replace('\\', '/')
                updated_lines.append(f"outputPath           '{output_path}/'\n")
            elif 'settingsPath' in line:
                settings_path = str(self.optimization_settings_dir).replace('\\', '/')
                updated_lines.append(f"settingsPath         '{settings_path}/'\n")
            else:
                updated_lines.append(line)

        with open(file_manager_path, 'w', encoding='utf-8') as f:
            f.writelines(updated_lines)

    def _get_simulation_period(self, use_calibration_period: bool = True) -> tuple:
        """Get simulation start and end times from config.

        Args:
            use_calibration_period: Whether to use calibration/spinup period

        Returns:
            Tuple of (sim_start, sim_end) strings
        """
        if use_calibration_period:
            calibration_period_str = self._get_config_value_safe('CALIBRATION_PERIOD', '')
            spinup_period_str = self._get_config_value_safe('SPINUP_PERIOD', '')

            if calibration_period_str and spinup_period_str:
                try:
                    spinup_dates = split_period(spinup_period_str, allow_extra=True)
                    cal_dates = split_period(calibration_period_str, allow_extra=True)

                    if spinup_dates is not None and cal_dates is not None:
                        spinup_start = datetime.strptime(spinup_dates[0], '%Y-%m-%d').replace(hour=1, minute=0)

                        forcing_timestep_seconds = self._get_config_value_safe('FORCING_TIME_STEP_SIZE', 3600)
                        cal_end_day = datetime.strptime(cal_dates[1], '%Y-%m-%d').strftime('%Y-%m-%d')

                        sim_start = spinup_start.strftime('%Y-%m-%d %H:%M')
                        sim_end = adjust_end_time_for_forcing(f'{cal_end_day} 23:00', forcing_timestep_seconds)

                        self.logger.info(f"Using spinup + calibration period: {sim_start} to {sim_end}")
                        return sim_start, sim_end

                except Exception as e:  # noqa: BLE001 — calibration resilience
                    self.logger.warning(f"Could not parse spinup+calibration periods: {str(e)}", exc_info=True)

        # Fall back to experiment time period
        sim_start = self._get_config_value_safe('EXPERIMENT_TIME_START', '1980-01-01 01:00')
        sim_end = self._adjust_end_time_for_forcing_internal(
            self._get_config_value_safe('EXPERIMENT_TIME_END', '2018-12-31 23:00')
        )
        return sim_start, sim_end

    # Map of legacy config dict keys to typed config accessor lambdas
    _TYPED_CONFIG_MAP = {
        'CALIBRATION_PERIOD': lambda c: c.domain.calibration_period,
        'SPINUP_PERIOD': lambda c: c.domain.spinup_period,
        'FORCING_TIME_STEP_SIZE': lambda c: c.forcing.time_step_size,
        'EXPERIMENT_TIME_START': lambda c: str(c.domain.time_start) if c.domain.time_start else None,
        'EXPERIMENT_TIME_END': lambda c: str(c.domain.time_end) if c.domain.time_end else None,
        'SETTINGS_SUMMA_FILEMANAGER': lambda c: c.model.summa.filemanager,
    }

    def _get_config_value_safe(self, key: str, default: Any) -> Any:
        """Safely get config value with fallback."""
        accessor_fn = self._TYPED_CONFIG_MAP.get(key)
        typed_accessor = (lambda: accessor_fn(self._config)) if accessor_fn and hasattr(self, '_config') and self._config is not None else (lambda: None)
        if hasattr(self, '_get_config_value'):
            return self._get_config_value(typed_accessor, default=default, dict_key=key)
        return self.config_dict.get(key, default)

    def _adjust_end_time_for_forcing_internal(self, end_time_str: str) -> str:
        """Adjust end time to align with forcing data timestep."""
        try:
            forcing_timestep_seconds = self._get_config_value_safe('FORCING_TIME_STEP_SIZE', 3600)
            adjusted = adjust_end_time_for_forcing(end_time_str, forcing_timestep_seconds)
            if adjusted != end_time_str:
                self.logger.info(
                    "Adjusted end time from %s to %s for %sh forcing",
                    end_time_str,
                    adjusted,
                    forcing_timestep_seconds / 3600,
                )
            return adjusted

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.warning(f"Could not adjust end time: {e}", exc_info=True)
            return end_time_str

    # =========================================================================
    # mizuRoute Control File Updates
    # =========================================================================

    def _update_mizuroute_control_file(self, control_path: Path) -> None:
        """Update mizuRoute control file with appropriate paths."""
        def _normalize_path(path):
            return str(path).replace("\\", "/").rstrip("/") + "/"

        with open(control_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.strip().startswith('<input_dir>'):
                if hasattr(self, 'use_parallel') and not self.use_parallel:
                    new_path = _normalize_path(self.output_dir)
                else:
                    new_path = _normalize_path(self.summa_sim_dir)
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    lines[i] = f"<input_dir>             {new_path}    {comment}"
                else:
                    lines[i] = f"<input_dir>             {new_path}    ! Folder that contains runoff data from SUMMA\n"

            elif line.strip().startswith('<output_dir>'):
                if hasattr(self, 'use_parallel') and not self.use_parallel:
                    new_path = _normalize_path(self.output_dir / "mizuRoute")
                else:
                    new_path = _normalize_path(self.mizuroute_sim_dir)
                if '!' in line:
                    comment = '!' + '!'.join(line.split('!')[1:])
                    lines[i] = f"<output_dir>            {new_path}    {comment}"
                else:
                    lines[i] = f"<output_dir>            {new_path}    ! Folder that will contain mizuRoute simulations\n"

        with open(control_path, "w", encoding="ascii", newline="\n") as f:
            f.writelines(lines)

    # =========================================================================
    # SUMMA Parallel Processing
    # =========================================================================

    def _setup_summa_parallel_processing(self) -> None:
        """Setup SUMMA-specific parallel processing directories and files."""
        self.logger.info(f"Setting up parallel processing with {self.num_processes} processes")

        for proc_id in range(self.num_processes):
            proc_base_dir = self.optimization_dir / f"parallel_proc_{proc_id:02d}"
            proc_summa_dir = proc_base_dir / "SUMMA"
            proc_mizuroute_dir = proc_base_dir / "mizuRoute"
            proc_summa_settings_dir = proc_base_dir / "settings" / "SUMMA"
            proc_mizu_settings_dir = proc_base_dir / "settings" / "mizuRoute"

            for directory in [proc_base_dir, proc_summa_dir, proc_mizuroute_dir,
                            proc_summa_settings_dir, proc_mizu_settings_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            (proc_summa_dir / "logs").mkdir(parents=True, exist_ok=True)
            (proc_mizuroute_dir / "logs").mkdir(parents=True, exist_ok=True)

            self._copy_settings_to_process_dir(proc_summa_settings_dir, proc_mizu_settings_dir)
            self._update_process_file_managers(proc_id, proc_summa_dir, proc_mizuroute_dir,
                                            proc_summa_settings_dir, proc_mizu_settings_dir)

            self.parallel_dirs.append({
                'proc_id': proc_id,
                'base_dir': proc_base_dir,
                'summa_dir': proc_summa_dir,
                'mizuroute_dir': proc_mizuroute_dir,
                'summa_settings_dir': proc_summa_settings_dir,
                'mizuroute_settings_dir': proc_mizu_settings_dir
            })

    def _copy_settings_to_process_dir(self, proc_summa_settings_dir: Path, proc_mizu_settings_dir: Path) -> None:
        """Copy settings files to process-specific directory."""
        if self.optimization_settings_dir.exists():
            for settings_file in self.optimization_settings_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_summa_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)

        mizu_source_dir = self.optimization_dir / "settings" / "mizuRoute"
        if mizu_source_dir.exists():
            for settings_file in mizu_source_dir.glob("*"):
                if settings_file.is_file():
                    dest_file = proc_mizu_settings_dir / settings_file.name
                    shutil.copy2(settings_file, dest_file)

    def _update_process_file_managers(self, proc_id: int, summa_dir: Path, mizuroute_dir: Path,
                                    summa_settings_dir: Path, mizu_settings_dir: Path) -> None:
        """Update file managers for a specific process."""
        summa_fm_name = self._get_summa_file_manager_name()
        file_manager = summa_settings_dir / summa_fm_name
        if not file_manager.exists():
            file_manager = summa_settings_dir / 'fileManager.txt'
        if file_manager.exists():
            with open(file_manager, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if 'outFilePrefix' in line:
                    updated_lines.append(f"outFilePrefix        'proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}'\n")
                elif 'outputPath' in line:
                    output_path = str(summa_dir).replace('\\', '/')
                    updated_lines.append(f"outputPath           '{output_path}/'\n")
                elif 'settingsPath' in line:
                    settings_path = str(summa_settings_dir).replace('\\', '/')
                    updated_lines.append(f"settingsPath         '{settings_path}/'\n")
                else:
                    updated_lines.append(line)

            with open(file_manager, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

        control_file = mizu_settings_dir / 'mizuroute.control'
        def _normalize_path(path):
            return str(path).replace("\\", "/").rstrip("/") + "/"

        if control_file.exists():
            with open(control_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            updated_lines = []
            for line in lines:
                if '<input_dir>' in line:
                    input_path = _normalize_path(summa_dir)
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<input_dir>             {input_path}    {comment}")
                    else:
                        updated_lines.append(f"<input_dir>             {input_path}    ! Folder that contains runoff data from SUMMA\n")
                elif '<output_dir>' in line:
                    output_path = _normalize_path(mizuroute_dir)
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<output_dir>            {output_path}    {comment}")
                    else:
                        updated_lines.append(f"<output_dir>            {output_path}    ! Folder that will contain mizuRoute simulations\n")
                elif '<case_name>' in line:
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<case_name>             proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}    {comment}")
                    else:
                        updated_lines.append(f"<case_name>             proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}    ! Simulation case name\n")
                elif '<fname_qsim>' in line:
                    if '!' in line:
                        comment = '!' + '!'.join(line.split('!')[1:])
                        updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc    {comment}")
                    else:
                        updated_lines.append(f"<fname_qsim>            proc_{proc_id:02d}_{self.algorithm_name}_opt_{self.experiment_id}_timestep.nc    ! netCDF name for HM_HRU runoff\n")
                else:
                    updated_lines.append(line)

            with open(control_file, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)

    # =========================================================================
    # SUMMA Model Decisions Management
    # =========================================================================

    def _update_model_decisions_for_final_run(self) -> None:
        """Update modelDecisions.txt to use direct solver for final evaluation."""
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'
        if not model_decisions_path.exists():
            return
        try:
            with open(model_decisions_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            updated_lines = []
            for line in lines:
                if line.strip().startswith('num_method') and not line.strip().startswith('!'):
                    updated_lines.append(re.sub(r'(num_method\s+)\w+(\s+.*)', r'\1ida\2', line))
                else:
                    updated_lines.append(line)
            with open(model_decisions_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating modelDecisions.txt: {str(e)}", exc_info=True)

    def _restore_model_decisions_for_optimization(self) -> None:
        """Restore modelDecisions.txt to use iterative solver for optimization."""
        model_decisions_path = self.optimization_settings_dir / 'modelDecisions.txt'
        if not model_decisions_path.exists():
            return
        try:
            with open(model_decisions_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            updated_lines = []
            for line in lines:
                if line.strip().startswith('num_method') and not line.strip().startswith('!'):
                    updated_lines.append(re.sub(r'(num_method\s+)\w+(\s+.*)', r'\1itertive\2', line))
                else:
                    updated_lines.append(line)
            with open(model_decisions_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error restoring modelDecisions.txt: {str(e)}", exc_info=True)

    # =========================================================================
    # SUMMA Executable Path
    # =========================================================================

    def _get_summa_exe_path(self) -> Path:
        """Get SUMMA executable path."""
        summa_path = self._get_config_value_safe('SUMMA_INSTALL_PATH', None)
        if summa_path == 'default' or summa_path is None:
            data_dir = self._get_config_value_safe('SYMFLUENCE_DATA_DIR', '.')
            summa_path = Path(data_dir) / 'installs' / 'summa' / 'bin'
        else:
            summa_path = Path(summa_path)

        summa_exe = self._get_config_value_safe('SUMMA_EXE', 'summa.exe')
        return summa_path / summa_exe
