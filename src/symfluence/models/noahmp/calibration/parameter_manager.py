# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Noah-MP Parameter Manager.

Manages parameter bounds, normalization, and file updates for Noah-MP
calibration.  Parameters are split between namelist.input (scalar knobs)
and SOILPARM.TBL (soil hydraulic properties).
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.parameters.parameter_bounds_registry import get_noahmp_bounds
from symfluence.core.registries import R


@R.parameter_managers.add('NOAHMP')
class NoahMPParameterManager(BaseParameterManager):
    """Handles Noah-MP parameter bounds, normalization, and file updates."""

    def __init__(
        self,
        config: Any,
        logger: logging.Logger,
        settings_dir: Path,
    ):
        super().__init__(config, logger, settings_dir)

        params_str = self._get_config_value(
            lambda: self.config.model.noahmp.params_to_calibrate,
            dict_key='NOAHMP_PARAMS_TO_CALIBRATE',
        )
        self.noahmp_params = [
            p.strip() for p in str(params_str).split(',') if p.strip()
        ]

    def _get_parameter_names(self) -> List[str]:
        """Get ordered list of parameter names to calibrate."""
        return list(self.noahmp_params)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """Load parameter bounds from the centralized registry."""
        bounds = get_noahmp_bounds()

        # Config overrides
        config_bounds = self._get_config_value(
            lambda: None, default={}, dict_key='NOAHMP_PARAM_BOUNDS'
        )
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)

        return bounds

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update Noah-MP model files via the worker.

        Delegates to worker for namelist.input and SOILPARM.TBL updates.
        """
        from .worker import NoahMPWorker

        worker = NoahMPWorker(config=self.config, logger=self.logger)
        return worker.apply_parameters(params, self.settings_dir)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values (mid-range of bounds)."""
        bounds = self._load_parameter_bounds()
        initial = {}
        for name in self.noahmp_params:
            if name not in bounds:
                continue
            b = bounds[name]
            transform = b.get('transform', 'linear')
            if transform == 'log' and b['min'] > 0:
                initial[name] = math.sqrt(b['min'] * b['max'])
            else:
                initial[name] = (b['min'] + b['max']) / 2.0
        return initial if initial else None
