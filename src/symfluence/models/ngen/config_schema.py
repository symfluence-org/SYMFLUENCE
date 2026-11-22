# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Typed configuration schema for NGEN.

Moved from ``symfluence.core.config.models`` (service-decomposition
prep): the schema ships with the model package and reaches the core
config system through ``R.config_schemas`` (via the model manifest /
``register()``), the same path an external plugin uses.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError

from symfluence.core.config.models.base import FROZEN_CONFIG


class NGENConfig(BaseModel):
    """NGEN (Next Generation Water Resources Modeling Framework) configuration"""
    model_config = FROZEN_CONFIG

    install_path: str = Field(default='default', alias='NGEN_INSTALL_PATH')
    exe: str = Field(default='ngen', alias='NGEN_EXE')
    modules_to_calibrate: str = Field(default='CFE', alias='NGEN_MODULES_TO_CALIBRATE')
    cfe_params_to_calibrate: str = Field(
        default='maxsmc,satdk,bb,slop',
        alias='NGEN_CFE_PARAMS_TO_CALIBRATE'
    )
    noah_params_to_calibrate: str = Field(
        default='refkdt,slope,smcmax,dksat',
        alias='NGEN_NOAH_PARAMS_TO_CALIBRATE'
    )
    pet_params_to_calibrate: str = Field(
        default='wind_speed_measurement_height_m',
        alias='NGEN_PET_PARAMS_TO_CALIBRATE'
    )
    sacsma_params_to_calibrate: str = Field(
        default='UZTWM,UZFWM,UZK,LZTWM,LZFPM,LZFSM,LZPK,LZSK,ZPERC,REXP,PFREE',
        alias='NGEN_SACSMA_PARAMS_TO_CALIBRATE'
    )
    snow17_params_to_calibrate: str = Field(
        default='SCF,MFMAX,MFMIN,TIPM,PLWHC',
        # Canonical alias is hyphen-free so it can be documented as a YAML key;
        # the hyphenated NGEN_SNOW-17_* form is kept as a backward-compat input alias.
        alias='NGEN_SNOW17_PARAMS_TO_CALIBRATE',
        validation_alias=AliasChoices(
            'snow-17_params_to_calibrate',
            'NGEN_SNOW-17_PARAMS_TO_CALIBRATE',
            'NGEN_SNOW17_PARAMS_TO_CALIBRATE',
        ),
    )
    active_catchment_id: Optional[str] = Field(default=None, alias='NGEN_ACTIVE_CATCHMENT_ID')
    realization: Optional[str] = Field(
        default='realization_config.json', alias='SETTINGS_NGEN_REALIZATION',
        description='Realization config filename (default realization_config.json)'
    )
    calibration_nexus_id: Optional[str] = Field(
        default=None, alias='CALIBRATION_NEXUS_ID',
        description='Nexus ID whose output is used for calibration'
    )
    calibration_warmup_days: Optional[int] = Field(
        default=None, alias='CALIBRATION_WARMUP_DAYS',
        description='Warmup days excluded from calibration metrics'
    )
    experiment_output: str = Field(default='default', alias='EXPERIMENT_OUTPUT_NGEN')
    hydrofabric_version: Optional[str] = Field(
        default=None, alias='NWS_HYDROFABRIC_VERSION',
        description='NWS hydrofabric version to acquire'
    )
    # Parameter bounds overrides (per-module)
    cfe_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_CFE_PARAM_BOUNDS')
    noah_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_NOAH_PARAM_BOUNDS')
    pet_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_PET_PARAM_BOUNDS')
    snow17_param_bounds: Optional[Dict[str, Any]] = Field(default=None, alias='NGEN_SNOW17_PARAM_BOUNDS')
    # Module selection (replaces individual ENABLE_* flags)
    modules_selected: str = Field(default='SLOTH,PET,CFE', alias='NGEN_MODULES_SELECTED')
    noah_et_fallback: str = Field(default='ETRAN', alias='NGEN_NOAH_ET_FALLBACK')
    csv_output_is_flow: bool = Field(default=False, alias='NGEN_CSV_OUTPUT_IS_FLOW')
    # ngen forcing provider; None -> CsvPerFeature. 'NetCDF' needs AORC-format
    # NetCDF (epoch_start attr, CSDMS names) which the forcing pipeline doesn't
    # produce yet.
    forcing_provider: Optional[str] = Field(default=None, alias='NGEN_FORCING_PROVIDER')

    @model_validator(mode='before')
    @classmethod
    def _migrate_enable_flags(cls, values: Any) -> Any:
        """Auto-migrate deprecated ENABLE_* flags to NGEN_MODULES_SELECTED."""
        if not isinstance(values, dict):
            return values

        # Check for any legacy ENABLE_* keys
        enable_keys = {
            'ENABLE_SLOTH': ('SLOTH', True),
            'ENABLE_PET': ('PET', True),
            'ENABLE_NOAH': ('NOAH', False),
            'ENABLE_CFE': ('CFE', True),
        }
        found_legacy = {k: v for k, v in enable_keys.items() if k in values}

        if not found_legacy:
            return values

        # Only migrate if NGEN_MODULES_SELECTED is not already explicitly set
        if 'NGEN_MODULES_SELECTED' in values or 'modules_selected' in values:
            # Remove stale legacy keys so they don't cause Pydantic errors
            for k in found_legacy:
                values.pop(k, None)
            return values

        # Build modules list from legacy flags
        modules = []
        for key, (mod_name, default_on) in enable_keys.items():
            raw = values.get(key, default_on)
            # Handle string booleans from YAML
            if isinstance(raw, str):
                enabled = raw.lower() in ('true', '1', 'yes')
            else:
                enabled = bool(raw)
            if enabled:
                modules.append(mod_name)

        values['NGEN_MODULES_SELECTED'] = ','.join(modules)

        # Remove legacy keys
        for k in found_legacy:
            values.pop(k, None)

        warnings.warn(
            "ENABLE_SLOTH/ENABLE_PET/ENABLE_NOAH/ENABLE_CFE are deprecated. "
            f"Use NGEN_MODULES_SELECTED: '{values['NGEN_MODULES_SELECTED']}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return values

    @model_validator(mode='after')
    def _validate_calibrate_subset(self) -> 'NGENConfig':
        """Ensure modules_to_calibrate is a subset of modules_selected."""
        def canonical_module_name(module: str) -> str:
            raw = module.strip().upper()
            collapsed = raw.replace('-', '').replace('_', '')
            if collapsed == 'SACSMA':
                return 'SACSMA'
            return raw

        selected = {canonical_module_name(m) for m in self.modules_selected.split(',') if m.strip()}
        calibrate = {canonical_module_name(m) for m in self.modules_to_calibrate.split(',') if m.strip()}
        not_selected = calibrate - selected
        if not_selected:
            # PydanticCustomError (not a bare ValueError) keeps the models/
            # generic-raise guard clean while still surfacing through pydantic
            # as a ValidationError.
            raise PydanticCustomError(
                'ngen_calibrate_subset',
                'NGEN_MODULES_TO_CALIBRATE contains modules not in '
                'NGEN_MODULES_SELECTED: {not_selected}. Either add them to '
                'NGEN_MODULES_SELECTED or remove them from NGEN_MODULES_TO_CALIBRATE.',
                {'not_selected': str(not_selected)},
            )
        return self
    run_troute: bool = Field(default=True, alias='NGEN_RUN_TROUTE')
