# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PRMS Parameter Manager

Handles PRMS parameter bounds, normalization, and parameter file updates.
Parameters are updated by rewriting sections of the PRMS params.dat file.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R


@R.parameter_managers.add('PRMS')
class PRMSParameterManager(BaseParameterManager):
    """Handles PRMS parameter bounds, normalization, and file updates.

    PRMS uses a custom text-based parameter file (params.dat) where each
    parameter block has the format::

        parameter_name
        ndim
        dimension_name
        dimension_size
        type_code
        value(s)
        ####

    This manager parses and replaces parameter values in-place.
    """

    def __init__(self, config: Dict, logger: logging.Logger, prms_settings_dir: Path):
        """
        Initialize PRMS parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            prms_settings_dir: Path to PRMS settings directory containing params.dat
        """
        super().__init__(config, logger, prms_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse PRMS parameters to calibrate from config
        prms_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'prms'):
                prms_params_str = config.model.prms.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if prms_params_str is None:
            prms_params_str = self._get_config_value(lambda: self.config.model.prms.params_to_calibrate, default=None, dict_key='PRMS_PARAMS_TO_CALIBRATE')

        if prms_params_str is None:
            prms_params_str = (
                'soil_moist_max,soil_rechr_max,tmax_allrain,tmax_allsnow,'
                'hru_percent_imperv,carea_max,smidx_coef,slowcoef_lin,'
                'gwflow_coef,ssr2gw_rate'
            )
            logger.warning(
                f"PRMS_PARAMS_TO_CALIBRATE missing; using fallback: {prms_params_str}"
            )

        self.prms_params = [p.strip() for p in str(prms_params_str).split(',') if p.strip()]

        # Paths
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_dir = prms_settings_dir

        # Parameter file name
        self.param_file = self._get_config_value(lambda: None, default='params.dat', dict_key='PRMS_PARAMETER_FILE')

    def _get_parameter_names(self) -> List[str]:
        """Return PRMS parameter names from config."""
        return self.prms_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return PRMS parameter bounds.

        PRMS parameters with typical calibration ranges from the literature.
        """
        from ..parameters import PARAM_BOUNDS

        prms_bounds = dict(PARAM_BOUNDS)

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(lambda: self.config.model.prms.param_bounds, default={}, dict_key='PRMS_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(prms_bounds, config_bounds)

        for param_name in self.prms_params:
            if param_name in prms_bounds:
                b = prms_bounds[param_name]
                self.logger.info(f"PRMS param {param_name}: bounds=[{b['min']:.6f}, {b['max']:.6f}]")

        return prms_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update PRMS parameter file with new parameter values."""
        return self.update_parameter_file(params)

    def update_parameter_file(self, params: Dict[str, float]) -> bool:
        """
        Update PRMS params.dat file with new parameter values.

        The PRMS parameter file uses blocks separated by ``####``.
        Each block starts with the parameter name, followed by metadata
        lines, then value lines. We locate each parameter block and
        replace its value lines.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            param_path = self.settings_dir / self.param_file
            if not param_path.exists():
                self.logger.error(f"PRMS parameter file not found: {param_path}")
                return False

            from ..derived_parameters import apply_derived_parameters
            content = param_path.read_text(encoding='utf-8')
            content, params = apply_derived_parameters(content, params)
            blocks = content.split('####\n')

            updated_blocks = []
            for block in blocks:
                updated_block = block
                for param_name, value in params.items():
                    if self._block_contains_param(block, param_name):
                        updated_block = self._update_block_values(
                            block, param_name, value
                        )
                        self.logger.debug(f"Updated {param_name} = {value:.6f}")
                updated_blocks.append(updated_block)

            updated_content = '####\n'.join(updated_blocks)
            param_path.write_text(updated_content, encoding='utf-8')
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating PRMS parameter file: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def _block_contains_param(self, block: str, param_name: str) -> bool:
        """Check if a parameter block contains the named parameter."""
        lines = block.strip().split('\n')
        for line in lines:
            if line.strip() == param_name:
                return True
        return False

    def _update_block_values(self, block: str, param_name: str, value: float) -> str:
        """
        Replace value lines in a PRMS parameter block.

        A block looks like::

            param_name
            ndim
            dimension_name
            dimension_size
            type_code
            value1
            value2
            ...

        We replace all value lines (after the type_code line) with the new value.
        For monthly parameters (nmonths dimension), all 12 values are set the same.

        Preserves leading/trailing whitespace so #### delimiters remain
        properly separated when blocks are rejoined.
        """
        stripped = block.strip()
        lines = stripped.split('\n')
        if not lines:
            return block

        # Find the parameter name line
        param_idx = None
        for i, line in enumerate(lines):
            if line.strip() == param_name:
                param_idx = i
                break

        if param_idx is None:
            return block

        # The structure after param_name is:
        # ndim (1 line), dimension_name (1 line), dimension_size (1 line), type_code (1 line)
        # Then value lines follow
        metadata_lines = int(lines[param_idx + 1]) + 3  # ndim, dimension names, count, type
        value_start = param_idx + 1 + metadata_lines

        if value_start > len(lines):
            return block

        # Get dimension size to know how many value lines to replace
        try:
            dim_size_idx = param_idx + int(lines[param_idx + 1]) + 2
            dim_size = int(lines[dim_size_idx].strip())
        except (IndexError, ValueError):
            dim_size = 1

        # Rebuild block with updated values
        new_lines = lines[:value_start]
        formatted_value = f"{value:.6f}" if isinstance(value, float) else str(value)
        for _ in range(dim_size):
            new_lines.append(formatted_value)

        # Preserve any trailing content (unlikely in standard format)
        remaining_start = value_start + dim_size
        if remaining_start < len(lines):
            new_lines.extend(lines[remaining_start:])

        result = '\n'.join(new_lines)
        # Preserve the trailing newline so #### delimiters stay on their own line
        if block.endswith('\n'):
            result += '\n'
        return result

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from parameter file or defaults."""
        try:
            param_path = self.settings_dir / self.param_file
            if not param_path.exists():
                return self._get_default_initial_values()

            content = param_path.read_text(encoding='utf-8')
            blocks = content.split('####')

            from ..derived_parameters import DERIVED_PARAMETERS, read_derived_parameter
            params = {}
            for param_name in self.prms_params:
                if param_name in DERIVED_PARAMETERS:
                    value = read_derived_parameter(content, param_name)
                    if value is not None:
                        params[param_name] = value
                for block in blocks:
                    if param_name in params:
                        break
                    if self._block_contains_param(block, param_name):
                        value = self._extract_first_value(block, param_name)
                        if value is not None:
                            params[param_name] = value
                            break

                if param_name not in params:
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {e}", exc_info=True)
            return self._get_default_initial_values()

    def _extract_first_value(self, block: str, param_name: str) -> Optional[float]:
        """Extract the first value from a parameter block."""
        lines = block.strip().split('\n')
        param_idx = None
        for i, line in enumerate(lines):
            if line.strip() == param_name:
                param_idx = i
                break

        if param_idx is None:
            return None

        value_start = param_idx + int(lines[param_idx + 1]) + 4
        if value_start < len(lines):
            try:
                return float(lines[value_start].strip())
            except ValueError:
                return None
        return None

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values from DEFAULT_PARAMS."""
        from ..parameters import DEFAULT_PARAMS

        params = {}
        for param_name in self.prms_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_settings_dir: Path) -> bool:
        """
        Copy parameter and control files to a worker-specific directory.

        Args:
            worker_settings_dir: Target directory for worker's files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_settings_dir.mkdir(parents=True, exist_ok=True)

            # Copy all .dat files (params.dat, control.dat, data.dat)
            for dat_file in self.settings_dir.glob('*.dat'):
                shutil.copy2(dat_file, worker_settings_dir / dat_file.name)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error copying PRMS files to {worker_settings_dir}: {e}", exc_info=True)
            return False
