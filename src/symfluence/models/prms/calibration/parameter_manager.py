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
from symfluence.core.exceptions import ConfigurationError
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

        from ..parameters import (
            DEFAULT_CALIBRATION_PARAMS,
            PRMS5_PARAMETER_NAMES,
            default_calibration_parameters,
            is_prms5_mode,
        )
        model_mode = self._get_config_value(
            lambda: self.config.model.prms.model_mode, default='DAILY', dict_key='PRMS_MODEL_MODE')
        default_names = default_calibration_parameters('DAILY')

        if prms_params_str is None:
            prms_params_str = DEFAULT_CALIBRATION_PARAMS
            logger.warning(
                f"PRMS_PARAMS_TO_CALIBRATE missing; using default list for model mode {model_mode}"
            )

        self.prms_params = [p.strip() for p in str(prms_params_str).split(',') if p.strip()]

        if is_prms5_mode(model_mode):
            if self.prms_params == default_names:
                # The schema default names the PRMS4-style parameters; under
                # PRMS5 the file holds (and PRMS reads) their replacements.
                self.prms_params = default_calibration_parameters(model_mode)
                logger.info(
                    f"PRMS5 mode: default calibration list uses {sorted(PRMS5_PARAMETER_NAMES.values())} "
                    f"in place of {sorted(PRMS5_PARAMETER_NAMES)}"
                )
            else:
                ignored = sorted(set(self.prms_params) & PRMS5_PARAMETER_NAMES.keys())
                if ignored:
                    raise ConfigurationError(
                        f"PRMS5 does not read {ignored}; calibrate "
                        f"{[PRMS5_PARAMETER_NAMES[name] for name in ignored]} instead "
                        "(PRMS_PARAMS_TO_CALIBRATE)."
                    )

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

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful

        Raises:
            ConfigurationError: a parameter has no block in the parameter file.
        """
        try:
            param_path = self.settings_dir / self.param_file
            if not param_path.exists():
                self.logger.error(f"PRMS parameter file not found: {param_path}")
                return False

            from ..parameter_file import update_parameter_file
            update_parameter_file(param_path, params)
            return True

        except ConfigurationError:
            raise
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating PRMS parameter file: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from parameter file or defaults."""
        try:
            param_path = self.settings_dir / self.param_file
            if not param_path.exists():
                return self._get_default_initial_values()

            content = param_path.read_text(encoding='utf-8')
            blocks = content.split('####')

            from ..derived_parameters import DERIVED_PARAMETERS, read_derived_parameter
            from ..parameter_file import first_block_value
            params = {}
            for param_name in self.prms_params:
                if param_name in DERIVED_PARAMETERS:
                    value = read_derived_parameter(content, param_name)
                    if value is not None:
                        params[param_name] = value
                for block in blocks:
                    if param_name in params:
                        break
                    value = first_block_value(block, param_name)
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
