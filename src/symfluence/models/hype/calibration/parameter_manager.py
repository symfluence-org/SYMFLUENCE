#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
HYPE Parameter Manager

Handles HYPE parameter bounds, normalization, and par.txt file updates.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.calibration.parameters.parameter_bounds_registry import get_hype_bounds
from symfluence.core.registries import R


@R.parameter_managers.add('HYPE')
class HYPEParameterManager(BaseParameterManager):
    """Handles HYPE parameter bounds, normalization, and file updates"""

    def __init__(self, config: Dict, logger: logging.Logger, hype_settings_dir: Path):
        super().__init__(config, logger, hype_settings_dir)

        # HYPE-specific setup
        self.domain_name = config.get('DOMAIN_NAME')
        self.experiment_id = config.get('EXPERIMENT_ID')

        # Parse HYPE parameters to calibrate from config
        # Default includes critical baseflow/groundwater parameter (rcgrw)
        # This is essential for snow-dominated and cold-region basins to generate winter baseflow
        hype_params_str = config.get('HYPE_PARAMS_TO_CALIBRATE')
        if hype_params_str is None:
            hype_params_str = 'ttmp,cmlt,cevp,lp,epotdist,rrcs1,rrcs2,rcgrw,rivvel,damp,wcwp,wcfc,wcep,srrcs'

        self.hype_params = [p.strip() for p in str(hype_params_str).split(',') if p.strip()]

        # Path to par.txt file
        self.data_dir = Path(config.get('SYMFLUENCE_DATA_DIR'))
        self.project_dir = self.data_dir / f"domain_{self.domain_name}"
        self.hype_setup_dir = self.project_dir / 'settings' / 'HYPE'
        self.par_file_path = self.hype_setup_dir / 'par.txt'

        # Regionalization setup
        self._regionalization_method = config.get('PARAMETER_REGIONALIZATION', 'lumped')
        self._regionalization = None
        self._regionalization_initialized = False

    # ========================================================================
    # REGIONALIZATION
    # ========================================================================

    def _init_regionalization(self):
        """Lazily initialize the regionalization strategy."""
        if self._regionalization_initialized:
            return
        self._regionalization_initialized = True

        if self._regionalization_method == 'lumped':
            return

        from symfluence.models.hype.calibration.hype_regionalization import (
            create_hype_regionalization,
        )

        geoclass_path = self.hype_setup_dir / 'GeoClass.txt'
        if not geoclass_path.exists():
            self.logger.warning(
                f"GeoClass.txt not found at {geoclass_path} — falling back to lumped"
            )
            self._regionalization_method = 'lumped'
            return

        tuple_bounds: Dict[str, Tuple[float, float]] = {}
        raw_bounds = self._load_raw_bounds()
        for p in self.hype_params:
            if p in raw_bounds:
                tuple_bounds[p] = (raw_bounds[p]['min'], raw_bounds[p]['max'])

        soil_csv = self.config.get('HYPE_SOIL_ATTRIBUTES_CSV')
        soil_csv_path = Path(soil_csv) if soil_csv else None
        if soil_csv_path is None:
            default_soil_csv = self.project_dir / 'data' / 'attributes' / 'soilclass' / 'iceland_soil_attributes.csv'
            if default_soil_csv.exists():
                soil_csv_path = default_soil_csv

        geodata_path = self.hype_setup_dir / 'GeoData.txt'
        climate_stats_path = self.project_dir / 'data' / 'attributes' / 'climate' / 'climate_statistics.csv'
        self._regionalization = create_hype_regionalization(
            method=self._regionalization_method,
            param_bounds=tuple_bounds,
            geoclass_path=geoclass_path,
            geodata_path=geodata_path if geodata_path.exists() else None,
            climate_stats_path=climate_stats_path if climate_stats_path.exists() else None,
            soil_attributes_csv=soil_csv_path,
            lu_param_config=self.config.get('HYPE_LU_PARAM_CONFIG'),
            soil_param_config=self.config.get('HYPE_SOIL_PARAM_CONFIG'),
            global_params=self.config.get('HYPE_GLOBAL_PARAMS'),
            logger=self.logger,
        )
        self.logger.info(
            f"HYPE regionalization: {self._regionalization_method} — "
            f"{len(self._regionalization.get_calibration_parameters())} coefficients"
        )

    def _load_raw_bounds(self) -> Dict[str, Dict[str, Any]]:
        """Load raw bounds without regionalization override."""
        bounds = get_hype_bounds()
        config_bounds = self.config.get('HYPE_PARAM_BOUNDS', {})
        if config_bounds:
            self._apply_config_bounds_override(bounds, config_bounds)
        return bounds

    @property
    def _use_regionalization(self) -> bool:
        return self._regionalization_method != 'lumped'

    # ========================================================================
    # IMPLEMENT ABSTRACT METHODS
    # ========================================================================

    def _get_parameter_names(self) -> List[str]:
        """Return HYPE parameter names or coefficient names."""
        if self._use_regionalization:
            self._init_regionalization()
            if self._regionalization:
                return list(self._regionalization.get_calibration_parameters().keys())
        return self.hype_params

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, float]]:
        """Return HYPE parameter bounds or coefficient bounds."""
        if self._use_regionalization:
            self._init_regionalization()
            if self._regionalization:
                bounds: Dict[str, Dict[str, Any]] = {}
                transforms = self._regionalization.get_coefficient_transforms()
                for name, (lo, hi) in self._regionalization.get_calibration_parameters().items():
                    entry: Dict[str, Any] = {'min': lo, 'max': hi}
                    t = transforms.get(name, 'linear')
                    if t != 'linear':
                        entry['transform'] = t
                    bounds[name] = entry
                return bounds
        return self._load_raw_bounds()

    def update_model_files(self, params: Dict[str, float]) -> bool:
        """Update par.txt. Expands coefficients when regionalization active."""
        if self._use_regionalization and self._regionalization:
            par_values = self._regionalization.expand_to_par_values(params)
            return self.update_par_file(par_values)
        return self.update_par_file(params)

    def get_initial_parameters(self) -> Optional[Dict[str, float]]:
        """Get initial parameter values from par.txt or defaults."""
        if self._use_regionalization:
            return self._get_default_initial_values()
        try:
            if not self.par_file_path.exists():
                self.logger.warning(f"par.txt not found: {self.par_file_path}")
                return self._get_default_initial_values()

            # Parse existing par.txt
            params = {}
            with open(self.par_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for param_name in self.hype_params:
                # Match: param_name followed by value(s), ignoring comments
                pattern = rf'^{param_name}\s+([\d\.\-\+eE]+)'
                match = re.search(pattern, content, re.MULTILINE)
                if match:
                    params[param_name] = float(match.group(1))
                else:
                    # Use midpoint of bounds
                    bounds = self.param_bounds.get(param_name, {'min': 0.1, 'max': 10.0})
                    params[param_name] = (bounds['min'] + bounds['max']) / 2

            # Validate and fix any problematic initial values
            params = self._validate_and_fix_initial_parameters(params)
            return params

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error reading initial parameters: {e}", exc_info=True)
            return self._get_default_initial_values()

    # ========================================================================
    # HYPE-SPECIFIC METHODS
    # ========================================================================

    def update_par_file(self, params: Dict[str, Any]) -> bool:
        """
        Update par.txt file with new parameter values.

        Uses regex to find and replace parameter values while preserving
        file structure and comments.
        """
        try:
            if not self.par_file_path.exists():
                self.logger.error(f"par.txt not found: {self.par_file_path}")
                return False

            with open(self.par_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            updated = 0
            missing: List[str] = []
            for param_name, value in params.items():
                if isinstance(value, (list, tuple)):
                    val_str = '\t'.join(f'{v:.6f}' for v in value)
                    new_content = re.sub(
                        rf'^({param_name}\s+)[^\!\n]*',
                        rf'\g<1>{val_str}\t',
                        content,
                        flags=re.MULTILINE,
                    )
                    if new_content != content:
                        updated += 1
                    content = new_content
                    continue

                # A scalar applies to every class column on the line (land-use
                # and soil parameters have one column per class); replacing only
                # the first column would calibrate class 1 alone.
                pattern = rf'^({param_name}[ \t]+)([\d\.\-\+eE]+(?:[ \t]+[\d\.\-\+eE]+)*)'

                def replacer(match, _value=value):
                    count = len(match.group(2).split())
                    return match.group(1) + '\t'.join([f'{_value:.6f}'] * count)

                content, n = re.subn(pattern, replacer, content, count=1, flags=re.MULTILINE)
                if n == 0:
                    missing.append(param_name)
                updated += n

            if missing:
                # A calibrated parameter absent from par.txt would be silently inert.
                self.logger.error(f"Parameters not found in par.txt: {missing} "
                                  "(add them with HYPE_EXTRA_PARAMETERS)")
                return False

            # Write updated content
            with open(self.par_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.debug(f"Updated {updated} parameters in par.txt")
            return True

        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error updating par.txt: {e}", exc_info=True)
            return False

    def _get_default_initial_values(self) -> Dict[str, float]:
        """Get default initial parameter values (midpoint of bounds).

        If INITIAL_PARAMETERS is provided in config, use those values
        instead of midpoints for a warm-start.
        """
        config_initial = self.config.get('INITIAL_PARAMETERS') if isinstance(self.config, dict) else None
        if config_initial and isinstance(config_initial, dict):
            params = {}
            for name in self.all_param_names:
                if name in config_initial:
                    params[name] = float(config_initial[name])
                else:
                    b = self.param_bounds.get(name, {'min': 0, 'max': 1})
                    params[name] = (b['min'] + b['max']) / 2
            self.logger.info(f"Using INITIAL_PARAMETERS from config ({sum(1 for n in params if n in config_initial)}/{len(params)} specified)")
            return params
        params = {}
        for param_name, param_bounds_dict in self.param_bounds.items():
            params[param_name] = (param_bounds_dict['min'] + param_bounds_dict['max']) / 2
        return params

    def _validate_and_fix_initial_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        Validate initial parameters and fix common issues.

        Addresses problems where initial par.txt may have problematic values
        that prevent the optimizer from finding good solutions.
        """
        fixed_params = params.copy()

        # Critical fix: cevp=0.0 disables evapotranspiration entirely
        # This breaks lumped models by preventing water from leaving the system
        if 'cevp' in fixed_params and fixed_params['cevp'] <= 0.01:
            self.param_bounds.get('cevp', {'min': 0.0, 'max': 1.0})
            # Set to a reasonable initial value (60% of PET)
            fixed_params['cevp'] = 0.6
            self.logger.warning(
                f"Initial cevp value was {params.get('cevp', 'unknown'):.4f} (disables ET). "
                f"Reset to {fixed_params['cevp']:.4f} to enable evapotranspiration."
            )

        return fixed_params
