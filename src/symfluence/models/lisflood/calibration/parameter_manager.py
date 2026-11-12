# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""LISFLOOD Parameter Manager."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr

from symfluence.core.calibration.parameters.base_parameter_manager import BaseParameterManager
from symfluence.core.registries import R


@R.parameter_managers.add("LISFLOOD")
class LisfloodParameterManager(BaseParameterManager):
    """Handles LISFLOOD parameter bounds, normalization, and file updates.

    LISFLOOD stores parameters as individual NetCDF map files in a maps/
    directory. Each parameter is a separate file (e.g., ksat1.nc).
    """

    # Parameters stored as NetCDF map files
    PARAM_FILE_MAP = {
        "ksat1": "ksat1",
        "ksat2": "ksat2",
        "ksat3": "ksat3",
        "lambda1": "lambda1",
        "lambda2": "lambda2",
        "lambda3": "lambda3",
        "thetas1": "thetas1",
        "thetas2": "thetas2",
        "thetas3": "thetas3",
        "thetar1": "thetar1",
        "thetar2": "thetar2",
        "thetar3": "thetar3",
        "genuinvalf1": "genuinvalf1",
        "genuinvalf2": "genuinvalf2",
        "genuinvalf3": "genuinvalf3",
        "chanman": "chanman",
        "mannings": "mannings",
        "cropcoef": "cropcoef",
        "forest_fraction": "forest_fraction",
        "snow_melt_coef": "snow_melt_coef",
        "temp_melt": "temp_melt",
        "temp_snow": "temp_snow",
    }

    # Parameters stored as scalar values in the settings XML lfuser section
    XML_PARAMS = {
        "UpperZoneTimeConstant",
        "LowerZoneTimeConstant",
        "GwPercValue",
        "GwLoss",
        "LZThreshold",
        "b_Xinanjiang",
        "PowerPrefFlow",
        "CalChanMan",
        "SnowMeltCoef",
        "CalEvaporation",
        "SnowSeasonAdj",
    }

    DEFAULT_BOUNDS = {
        # Map-based parameters
        "ksat1": {"min": 5.0, "max": 100.0},
        "ksat2": {"min": 2.0, "max": 60.0},
        "ksat3": {"min": 0.5, "max": 30.0},
        "lambda1": {"min": 0.05, "max": 0.6},
        "lambda2": {"min": 0.05, "max": 0.5},
        "lambda3": {"min": 0.03, "max": 0.4},
        "thetas1": {"min": 0.3, "max": 0.6},
        "thetas2": {"min": 0.28, "max": 0.55},
        "thetas3": {"min": 0.25, "max": 0.50},
        "thetar1": {"min": 0.01, "max": 0.15},
        "thetar2": {"min": 0.01, "max": 0.15},
        "thetar3": {"min": 0.01, "max": 0.15},
        "genuinvalf1": {"min": 0.01, "max": 0.1},
        "genuinvalf2": {"min": 0.01, "max": 0.08},
        "genuinvalf3": {"min": 0.005, "max": 0.06},
        "chanman": {"min": 0.01, "max": 0.1},
        "mannings": {"min": 0.02, "max": 0.15},
        "cropcoef": {"min": 0.5, "max": 1.5},
        "forest_fraction": {"min": 0.0, "max": 1.0},
        "snow_melt_coef": {"min": 1.0, "max": 10.0},
        "temp_melt": {"min": -2.0, "max": 4.0},
        "temp_snow": {"min": -3.0, "max": 2.0},
        # XML scalar parameters
        "UpperZoneTimeConstant": {"min": 2.0, "max": 100.0},
        "LowerZoneTimeConstant": {"min": 20.0, "max": 1000.0},
        "GwPercValue": {"min": 0.01, "max": 5.0},
        "GwLoss": {"min": 0.0, "max": 0.5},
        "LZThreshold": {"min": 0.0, "max": 50.0},
        "b_Xinanjiang": {"min": 0.01, "max": 1.5},
        "PowerPrefFlow": {"min": 1.0, "max": 6.0},
        "CalChanMan": {"min": 0.1, "max": 10.0},
        "SnowMeltCoef": {"min": 1.0, "max": 10.0},
        "CalEvaporation": {"min": 0.5, "max": 2.0},
        "SnowSeasonAdj": {"min": 0.0, "max": 2.0},
    }

    PREPROCESSOR_DEFAULTS = {
        # Map-based
        "ksat1": 25.0,
        "ksat2": 15.0,
        "ksat3": 5.0,
        "lambda1": 0.25,
        "lambda2": 0.22,
        "lambda3": 0.18,
        "thetas1": 0.45,
        "thetas2": 0.43,
        "thetas3": 0.40,
        "thetar1": 0.05,
        "thetar2": 0.05,
        "thetar3": 0.05,
        "genuinvalf1": 0.037,
        "genuinvalf2": 0.035,
        "genuinvalf3": 0.030,
        "chanman": 0.04,
        "mannings": 0.07,
        "cropcoef": 1.0,
        "forest_fraction": 0.5,
        "snow_melt_coef": 4.5,
        "temp_melt": 1.0,
        "temp_snow": 0.0,
        # XML scalar
        "UpperZoneTimeConstant": 10.0,
        "LowerZoneTimeConstant": 100.0,
        "GwPercValue": 0.5,
        "GwLoss": 0.0,
        "LZThreshold": 5.0,
        "b_Xinanjiang": 0.7,
        "PowerPrefFlow": 3.5,
        "CalChanMan": 2.0,
        "SnowMeltCoef": 4.5,
        "CalEvaporation": 1.0,
        "SnowSeasonAdj": 1.0,
    }

    def __init__(self, config: Dict, logger: logging.Logger, lisflood_settings_dir: Path):
        super().__init__(config, logger, lisflood_settings_dir)
        self.domain_name = self._get_config_value(lambda: self.config.domain.name, default=None, dict_key="DOMAIN_NAME")
        params_str = self._get_config_value(
            lambda: self.config.model.lisflood.params_to_calibrate,
            default=None,
            dict_key="LISFLOOD_PARAMS_TO_CALIBRATE",
        )
        if params_str is None:
            params_str = ",".join(self.PARAM_FILE_MAP.keys())
        self.params_to_calibrate = [p.strip() for p in params_str.split(",") if p.strip()]

    def _get_parameter_names(self) -> List[str]:
        return list(self.params_to_calibrate)

    def _load_parameter_bounds(self) -> Dict[str, Dict[str, Any]]:
        return {p: self.DEFAULT_BOUNDS[p] for p in self.params_to_calibrate if p in self.DEFAULT_BOUNDS}

    def get_initial_parameters(self) -> Optional[Dict[str, Any]]:
        bounds = self._load_parameter_bounds()
        initial = {}
        for p, b in bounds.items():
            default = self.PREPROCESSOR_DEFAULTS.get(p)
            if default is not None and b["min"] <= default <= b["max"]:
                initial[p] = default
            else:
                initial[p] = (b["min"] + b["max"]) / 2.0
        return initial

    def validate_parameters(self, params: Dict[str, float]) -> bool:
        for layer in ["1", "2", "3"]:
            r_key, s_key = f"thetar{layer}", f"thetas{layer}"
            if r_key in params and s_key in params:
                if params[r_key] >= params[s_key]:
                    params[r_key] = params[s_key] * 0.15
        bounds = self._load_parameter_bounds()
        for param, value in params.items():
            if param in bounds:
                params[param] = float(np.clip(value, bounds[param]["min"], bounds[param]["max"]))
        return True

    def update_model_files(self, params: Dict[str, float], settings_dir: Optional[Path] = None, **kwargs) -> bool:
        try:
            from xml.etree import ElementTree as ET  # nosec B405

            self.validate_parameters(params)
            target_dir = settings_dir or self.settings_dir

            map_params = {k: v for k, v in params.items() if k in self.PARAM_FILE_MAP}
            xml_params = {k: v for k, v in params.items() if k in self.XML_PARAMS}

            # Update NetCDF map files
            maps_dir = target_dir / "maps"
            if maps_dir.exists():
                for param_name, value in map_params.items():
                    nc_file = self.PARAM_FILE_MAP[param_name]
                    nc_path = maps_dir / f"{nc_file}.nc"
                    if not nc_path.exists():
                        continue
                    ds = xr.open_dataset(nc_path)
                    ds_new = ds.load()
                    ds.close()
                    if nc_file in ds_new:
                        mask = ~np.isnan(ds_new[nc_file].values)
                        ds_new[nc_file].values[mask] = value
                    tmp_path = nc_path.parent / f"{nc_path.stem}_tmp{nc_path.suffix}"
                    ds_new.to_netcdf(tmp_path)
                    shutil.move(str(tmp_path), str(nc_path))

            # Update XML scalar parameters in settings file
            if xml_params:
                settings_file = self._get_config_value(
                    lambda: self.config.model.lisflood.settings_file,
                    dict_key='LISFLOOD_SETTINGS_FILE',
                )
                xml_path = target_dir / settings_file
                if xml_path.exists():
                    tree = ET.parse(xml_path)  # nosec B314
                    root = tree.getroot()
                    for textvar in root.iter("textvar"):
                        name = textvar.get("name", "")
                        if name in xml_params:
                            textvar.set("value", str(xml_params[name]))
                    tree.write(xml_path, encoding="unicode", xml_declaration=True)

            return True
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to update LISFLOOD parameters: {e}")
            return False
