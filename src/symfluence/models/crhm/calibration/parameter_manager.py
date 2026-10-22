# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CRHM Parameter Manager

Handles CRHM parameter bounds, normalization, and parameter file updates.
CRHM uses a text-based .prj (project) file with key-value parameter format.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R


@R.parameter_managers.add('CRHM')
class CRHMParameterManager(BaseParameterManager):
    """Handles CRHM parameter bounds, normalization, and file updates.

    CRHM parameters are stored in .prj text files using a key-value format.
    This manager reads and writes parameters by parsing the project file.
    """

    def __init__(self, config: Dict, logger: logging.Logger, crhm_settings_dir: Path):
        """
        Initialize CRHM parameter manager.

        Args:
            config: Configuration dictionary
            logger: Logger instance
            crhm_settings_dir: Path to CRHM settings directory
        """
        super().__init__(config, logger, crhm_settings_dir)

        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key='DOMAIN_NAME')
        self.experiment_id = self._get_config_value(lambda: self.config.domain.experiment_id, default=None, dict_key='EXPERIMENT_ID')

        # Parse CRHM parameters to calibrate from config
        crhm_params_str = None
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'crhm'):
                crhm_params_str = config.model.crhm.params_to_calibrate
        except (AttributeError, TypeError):
            pass

        if crhm_params_str is None:
            crhm_params_str = self._get_config_value(lambda: self.config.model.crhm.params_to_calibrate, default=None, dict_key='CRHM_PARAMS_TO_CALIBRATE')

        if crhm_params_str is None:
            # Includes blowing-snow / snow-sublimation controls (inhibit_subl,
            # Qe_subl_from_SWE, N_S). Without them DDS cannot reduce CRHM's
            # PBSM blowing-snow sublimation, which in maritime/high-wind regimes
            # (e.g. Iceland) defaults far too high (~58% of precip) and caps the
            # achievable runoff at beta~0.4 regardless of the other parameters.
            crhm_params_str = ('Ht,soil_rechr_max,soil_moist_max,soil_gw_K,Sdmax,fetch,'
                               'gw_K,gw_max,Kstorage,Lag,lapse_rate,'
                               'inhibit_subl,Qe_subl_from_SWE,N_S')
            logger.warning(
                f"CRHM_PARAMS_TO_CALIBRATE missing; using fallback: {crhm_params_str}"
            )

        self.crhm_params = [p.strip() for p in str(crhm_params_str).split(',') if p.strip()]

        # Path to project file
        self.data_dir = Path(self._get_config_value(lambda: self.config.system.data_dir, dict_key='SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.settings_dir = self.project_dir / 'settings' / 'CRHM'

        # Get project file name
        prj_file = 'model.prj'
        try:
            if hasattr(config, 'model') and hasattr(config.model, 'crhm'):
                prj_file = config.model.crhm.project_file or prj_file
        except (AttributeError, TypeError):
            pass
        self.prj_file = self.settings_dir / prj_file

    def _get_parameter_names(self) -> List[str]:
        """Return CRHM parameter names from config."""
        return self.crhm_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """
        Return CRHM parameter bounds.

        CRHM parameters with typical calibration ranges covering
        cold-region hydrology processes: blowing snow, energy-balance
        snowmelt, frozen soil infiltration, and prairie hydrology.
        """
        from symfluence.models.crhm.parameters import PARAM_BOUNDS

        crhm_bounds = dict(PARAM_BOUNDS)

        # Check for config overrides (preserves transform metadata from registry)
        config_bounds = self._get_config_value(
            lambda: self.config.model.crhm.param_bounds,
            default={}, dict_key='CRHM_PARAM_BOUNDS')
        if config_bounds:
            self._apply_config_bounds_override(crhm_bounds, config_bounds)

        # Log bounds for calibrated parameters
        for param_name in self.crhm_params:
            if param_name in crhm_bounds:
                b = crhm_bounds[param_name]
                self.logger.info(f"CRHM param {param_name}: bounds=[{b['min']:.6f}, {b['max']:.6f}]")

        return crhm_bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update CRHM project file with new parameter values."""
        return self.update_prj_file(params)

    def update_prj_file(self, params: Dict[str, float]) -> bool:
        """
        Update CRHM project file (.prj) with new parameter values.

        In the native CRHM .prj format each parameter is stored as a
        two-line block inside the ``Parameters:`` section::

            <module> <param_name> [<min to max>]
            <value(s)>

        This method scans for header lines that reference a calibrated
        parameter and replaces the *following* value line.

        Args:
            params: Dictionary of parameter names to values

        Returns:
            True if successful
        """
        try:
            if not self.prj_file.exists():
                self.logger.error(f"CRHM project file not found: {self.prj_file}")
                return False

            content = self.prj_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            new_lines = []

            param_set = set(params.keys())
            skip_next = False
            pending_param: Optional[str] = None

            for line in lines:
                if skip_next and pending_param is not None:
                    # This line holds the old value(s) -- replace it
                    value = params[pending_param]
                    if abs(value) < 0.001 and value != 0:
                        formatted = f"{value:.8e}"
                    elif abs(value) > 9999:
                        formatted = f"{value:.2f}"
                    else:
                        formatted = f"{value:.6f}"
                    # Broadcast the (scalar) calibrated value across however many
                    # per-HRU tokens the original value line held. With elevation
                    # bands the line carries one value per band; writing a single
                    # token would silently collapse the parameter to 1 HRU and
                    # corrupt the .prj dimensions.
                    n_tokens = max(1, len(line.split()))
                    new_lines.append(' '.join([formatted] * n_tokens))
                    self.logger.debug(
                        f"Updated {pending_param} = {formatted} x{n_tokens}")
                    skip_next = False
                    pending_param = None
                    continue

                # Check if this is a parameter header line:
                # ``<module> <param_name> [<min to max>]``
                stripped = line.strip()
                parts = stripped.split()
                if len(parts) >= 2 and parts[1] in param_set:
                    pending_param = parts[1]
                    skip_next = True

                new_lines.append(line)

            # Write modified content back
            temp_file = self.prj_file.with_suffix('.prj.tmp')
            temp_file.write_text('\n'.join(new_lines), encoding='utf-8')
            temp_file.replace(self.prj_file)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating CRHM project file: {e}", exc_info=True)
            import traceback
            self.logger.debug(traceback.format_exc())
            return False

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from project file or defaults.

        In the native .prj format, parameters are stored as two-line blocks::

            <module> <param_name> [<min to max>]
            <value(s)>

        This method finds the parameter header line and reads the value
        from the immediately following line.
        """
        try:
            if not self.prj_file.exists():
                return self._get_default_initial_values()

            content = self.prj_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            params = {}

            for param_name in self.crhm_params:
                found = False
                for i, line in enumerate(lines):
                    parts = line.strip().split()
                    # Header format: <module> <param_name> [<min to max>]
                    if len(parts) >= 2 and parts[1] == param_name:
                        # Next line holds the value(s)
                        if i + 1 < len(lines):
                            val_line = lines[i + 1].strip()
                            # Take the first token (handles multi-HRU files)
                            val_tokens = val_line.split()
                            if val_tokens:
                                try:
                                    params[param_name] = float(val_tokens[0])
                                    found = True
                                except ValueError:
                                    pass
                        break

                if not found:
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {e}", exc_info=True)
            return self._get_default_initial_values()

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values from DEFAULT_PARAMS or midpoint of bounds."""
        from symfluence.models.crhm.parameters import DEFAULT_PARAMS

        params = {}
        for param_name in self.crhm_params:
            if param_name in DEFAULT_PARAMS:
                params[param_name] = DEFAULT_PARAMS[param_name]
            else:
                bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                params[param_name] = (bounds['min'] + bounds['max']) / 2
        return params

    def copy_params_to_worker_dir(self, worker_params_dir: Path) -> bool:
        """
        Copy project file to a worker-specific directory for parallel calibration.

        Args:
            worker_params_dir: Target directory for worker's parameter files

        Returns:
            True if successful
        """
        import shutil

        try:
            worker_params_dir.mkdir(parents=True, exist_ok=True)

            # Copy project file
            if self.prj_file.exists():
                shutil.copy2(self.prj_file, worker_params_dir / self.prj_file.name)

            # Also copy observation file if it exists
            obs_file = self.settings_dir / 'forcing.obs'
            if obs_file.exists():
                shutil.copy2(obs_file, worker_params_dir / obs_file.name)

            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error copying params to {worker_params_dir}: {e}", exc_info=True)
            return False
