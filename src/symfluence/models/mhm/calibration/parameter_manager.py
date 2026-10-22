# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
mHM Parameter Manager

Handles mHM parameter bounds, normalization, and Fortran namelist file updates.
Parameters are updated using regex-based parsing of .nml files.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R


@R.parameter_managers.add('MHM')
class MHMParameterManager(BaseParameterManager):
    """Handles mHM parameter bounds, normalization, and namelist updates.

    mHM uses Fortran namelists (.nml files) for configuration. Parameters
    are stored as name-value pairs in the format: paramName = value

    This manager uses regex-based parsing to find and replace parameter
    values in the mhm.nml file.
    """

    def __init__(self, config: Dict, logger: logging.Logger, mhm_settings_dir: Path):
        """
        Initialize mHM parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            mhm_settings_dir: Path to mHM settings directory containing .nml files
        """
        super().__init__(config, logger, mhm_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse mHM parameters to calibrate from config
        mhm_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'mhm'):
                mhm_params_str = config.model.mhm.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if mhm_params_str is None:
            mhm_params_str = self._get_config_value(lambda: self.config.model.mhm.params_to_calibrate, default=None, dict_key='MHM_PARAMS_TO_CALIBRATE')

        if mhm_params_str is None:
            mhm_params_str = (
                'canopyInterceptionFactor,snowTreshholdTemperature,'
                'degreeDayFactor_forest,degreeDayFactor_pervious,'
                'PTF_Ks_constant,interflowRecession_slope,rechargeCoefficient,'
                'GeoParam(1,:)'
            )
            logger.warning(
                f"MHM_PARAMS_TO_CALIBRATE missing; using fallback: {mhm_params_str}"
            )

        self.mhm_params = self._split_param_names(str(mhm_params_str))

        # Path to namelist files
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_dir = mhm_settings_dir

        # Get namelist file names -- prefer mhm_parameter.nml (the dedicated
        # parameter file) over mhm.nml for reading/writing parameter values.
        self.namelist_file = 'mhm.nml'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'mhm'):
                self.namelist_file = config.model.mhm.namelist_file or self.namelist_file
        except (AttributeError, TypeError):
            pass

        # Use mhm_parameter.nml when it exists (mHM v5.12+ convention)
        param_nml = self.settings_dir / 'mhm_parameter.nml'
        if param_nml.exists():
            self.namelist_path = param_nml
            self._is_param_nml = True
        else:
            self.namelist_path = self.settings_dir / self.namelist_file
            self._is_param_nml = False

    @staticmethod
    def _split_param_names(param_str: str) -> List[str]:
        """Split comma-separated parameter names, respecting parentheses.

        ``GeoParam(1,:)`` contains a comma but must not be split.
        """
        params: List[str] = []
        current: List[str] = []
        depth = 0
        for char in param_str:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                token = ''.join(current).strip().strip('"').strip("'")
                if token:
                    params.append(token)
                current = []
            else:
                current.append(char)
        token = ''.join(current).strip().strip('"').strip("'")
        if token:
            params.append(token)
        return params

    def _get_parameter_names(self) -> List[str]:
        """Return mHM parameter names from config."""
        return self.mhm_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return mHM parameter bounds.

        mHM parameters with typical calibration ranges from the MPR framework.
        """
        from ..parameters import PARAM_BOUNDS

        mhm_bounds = dict(PARAM_BOUNDS)

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(lambda: None, default={}, dict_key='MHM_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(mhm_bounds, config_bounds)

        # Log bounds for calibrated parameters
        for param_name in self.mhm_params:
            if param_name in mhm_bounds:
                b = mhm_bounds[param_name]
                self.logger.info(f"mHM param {param_name}: bounds=[{b['min']:.6f}, {b['max']:.6f}]")

        return mhm_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update mHM namelist files with new parameter values."""
        return self.update_namelist(params)

    def update_namelist(self, params: Dict[str, float]) -> bool:
        """
        Update mHM Fortran namelist (.nml) file with new parameter values.

        Supports two formats:

        1. **mhm_parameter.nml** (5-element tuple)::

               paramName = lower, upper, value, FLAG, SCALING

           Only the *value* (3rd element) is replaced.

        2. **Legacy / mhm.nml** (single value)::

               paramName = value

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            if not self.namelist_path.exists():
                self.logger.error(f"mHM namelist file not found: {self.namelist_path}")
                return False

            content = self.namelist_path.read_text(encoding='utf-8')

            for param_name, value in params.items():
                formatted_value = self._format_namelist_value(value)

                if self._is_param_nml:
                    # 5-column format: paramName = lower, upper, VALUE, flag, scaling
                    pattern = re.compile(
                        r'(^[ \t]*' + re.escape(param_name)
                        + r'\s*=\s*'
                        + r'[^,]+,'        # lower,
                        + r'\s*[^,]+,'     # upper,
                        + r'\s*)'          # whitespace before value
                        + r'([^,]+)'       # VALUE  <-- group to replace
                        + r'(,\s*\d+,\s*\d+)',  # , flag, scaling
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value + r'\g<3>',
                            content
                        )
                        self.logger.debug(
                            f"Updated {param_name} value = {formatted_value} "
                            f"(5-column format)"
                        )
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {self.namelist_path}"
                        )
                else:
                    # Legacy single-value format
                    pattern = re.compile(
                        r'(^[ \t]*' + re.escape(param_name) + r'\s*=\s*)([^\s,!/]+)',
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        content = pattern.sub(
                            r'\g<1>' + formatted_value,
                            content
                        )
                        self.logger.debug(f"Updated {param_name} = {formatted_value}")
                    else:
                        self.logger.warning(
                            f"Parameter {param_name} not found in {self.namelist_path}"
                        )

            # Write updated content
            self.namelist_path.write_text(content, encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating mHM namelist: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _format_namelist_value(self, value: float) -> str:
        """
        Format a parameter value for Fortran namelist syntax.

        Uses scientific notation for very small or very large values,
        and fixed-point notation otherwise.

        Args:
            value: Parameter value to format

        Returns:
            Formatted string suitable for Fortran namelist
        """
        abs_val = abs(value)
        if abs_val == 0.0:
            return '0.0'
        elif abs_val < 0.001 or abs_val >= 1e6:
            return f'{value:.6e}'
        elif abs_val == int(abs_val) and abs_val < 1e6:
            return f'{value:.1f}'
        else:
            return f'{value:.6f}'

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from namelist file or defaults.

        Supports both the 5-column ``mhm_parameter.nml`` format
        (``lower, upper, value, flag, scaling``) and the legacy single-value
        format in ``mhm.nml``.
        """
        from ..parameters import DEFAULT_PARAMS

        try:
            if not self.namelist_path.exists():
                return self._get_default_initial_values()

            content = self.namelist_path.read_text(encoding='utf-8')
            params = {}

            for param_name in self.mhm_params:
                extracted = None

                if self._is_param_nml:
                    # 5-column: paramName = lower, upper, VALUE, flag, scaling
                    pattern = re.compile(
                        r'^[ \t]*' + re.escape(param_name)
                        + r'\s*=\s*'
                        + r'[^,]+,'        # lower,
                        + r'\s*[^,]+,'     # upper,
                        + r'\s*([^,]+)'    # VALUE  <-- capture group
                        + r',\s*\d+,\s*\d+',
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        try:
                            value_str = match.group(1).strip().replace('d', 'e').replace('D', 'E')
                            extracted = float(value_str)
                        except ValueError:
                            pass

                if extracted is None:
                    # Fall back to single-value pattern
                    pattern = re.compile(
                        r'^[ \t]*' + re.escape(param_name) + r'\s*=\s*([^\s,!/]+)',
                        re.MULTILINE
                    )
                    match = pattern.search(content)
                    if match:
                        try:
                            value_str = match.group(1).replace('d', 'e').replace('D', 'E')
                            extracted = float(value_str)
                        except ValueError:
                            pass

                if extracted is not None:
                    params[param_name] = extracted
                elif param_name in DEFAULT_PARAMS:
                    # Forcing pseudo-parameters are not in the namelist.
                    params[param_name] = DEFAULT_PARAMS[param_name]
                else:
                    # Use midpoint of bounds
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {e}", exc_info=True)
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values from DEFAULT_PARAMS."""
        from ..parameters import DEFAULT_PARAMS

        params = {}
        for param_name in self.mhm_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_settings_dir: Path) -> bool:
        """
        Copy namelist files to a worker-specific directory for parallel calibration.

        Args:
            worker_settings_dir: Target directory for worker's namelist files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_settings_dir.mkdir(parents=True, exist_ok=True)

            # Copy all namelist files
            for nml_file in self.settings_dir.glob('*.nml'):
                shutil.copy2(nml_file, worker_settings_dir / nml_file.name)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error copying namelists to {worker_settings_dir}: {e}", exc_info=True)
            return False
