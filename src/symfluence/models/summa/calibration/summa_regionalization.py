# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA-specific adapter for parameter regionalization.

Wraps the model-agnostic regionalization framework from
``models/fuse/calibration/parameter_regionalization`` with SUMMA defaults
and attribute loading from SUMMA ``attributes.nc`` files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.calibration.regionalization.strategies import (
    ParameterRegionalization,
    RegionalizationFactory,
)

# Default mapping of SUMMA parameters to catchment attributes.
# ``calibrate_b=True`` means the slope coefficient is calibrated (spatial
# variation driven by the attribute); ``False`` means only the intercept
# is calibrated (spatially uniform).
SUMMA_DEFAULT_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    # Snow albedo — vary with elevation (snow persistence)
    'albedoMax':          {'attribute': 'elev_m',                       'calibrate_b': True},
    'albedoMinWinter':    {'attribute': 'elev_m',                       'calibrate_b': True},
    'albedoMinSpring':    {'attribute': 'elev_m',                       'calibrate_b': True},
    'albedoMaxVisible':   {'attribute': 'elev_m',                       'calibrate_b': False},
    'albedoDecayRate':    {'attribute': 'climate.srad_annual_mean',     'calibrate_b': True,  'fallback': 'elev_m'},
    # Snow density — vary with snow fraction (independent of elevation)
    'newSnowDenMin':      {'attribute': 'snow_frac',                    'calibrate_b': True},
    'newSnowDenMult':     {'attribute': 'snow_frac',                    'calibrate_b': False},
    'newSnowDenScal':     {'attribute': 'snow_frac',                    'calibrate_b': False},
    # Rain/snow threshold — vary with actual temperature when available
    'tempCritRain':       {'attribute': 'climate.tavg_annual_mean',     'calibrate_b': True,  'fallback': 'elev_m'},
    'tempRangeTimestep':  {'attribute': 'elev_m',                       'calibrate_b': False},
    # Soil hydraulics — vary with measured Ksat when available
    'k_soil':             {'attribute': 'soil.ksat',                    'calibrate_b': True,  'fallback': 'aridity'},
    'k_macropore':        {'attribute': 'soil.ksat',                    'calibrate_b': True,  'fallback': 'aridity'},
    # Aquifer — vary with regolith thickness when available
    'aquiferBaseflowExp': {'attribute': 'soil.regolith_thickness_mean', 'calibrate_b': True,  'fallback': 'aridity'},
    'aquiferBaseflowRate':{'attribute': 'soil.regolith_thickness_mean', 'calibrate_b': True,  'fallback': 'aridity'},
    # Precipitation correction — vary with snow fraction
    'frozenPrecipMultip': {'attribute': 'snow_frac',                    'calibrate_b': True},
    # Soil storage — vary with measured porosity when available
    'theta_sat':          {'attribute': 'soil.porosity',                'calibrate_b': True,  'fallback': 'precip_mm_yr'},
    'vGn_n':              {'attribute': 'aridity',                      'calibrate_b': False},
    'snowfrz_scale':      {'attribute': 'elev_m',                       'calibrate_b': True},
    'routingGammaScale':  {'attribute': 'precip_mm_yr',                 'calibrate_b': False},
}


def load_hru_attributes(
    attributes_nc_path: Path,
    csv_path: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load HRU-level catchment attributes for transfer-function regionalization.

    Reads from the SUMMA ``attributes.nc`` file and, optionally, from a
    supplementary CSV (e.g. CAMELS-style catchment descriptors).

    The returned DataFrame is indexed by HRU order (0-based) and contains
    columns such as ``elev_m``, ``precip_mm_yr``, ``aridity``, etc.

    Supplementary tables are joined to the HRUs on their ``hru_id`` column
    (matched to ``hruId`` in ``attributes.nc``), never on row position.

    Args:
        attributes_nc_path: Path to SUMMA ``attributes.nc``.
        csv_path: Optional path to a CSV with additional per-HRU attributes.
                  Joined on its ``hru_id`` column when present; without
                  one its rows must be in ``attributes.nc`` HRU order.
        logger: Logger instance.

    Returns:
        DataFrame with one row per HRU and attribute columns.

    Raises:
        ValidationError: if an auto-discovered attribute CSV predates the
            ``hru_id`` column and has too many rows to be matched safely.
    """
    from symfluence.data.model_ready.hru_tables import align_table_to_hru_ids, read_hru_table

    logger = logger or logging.getLogger(__name__)

    # --- Read from attributes.nc -------------------------------------------
    with xr.open_dataset(attributes_nc_path) as ds:
        data: Dict[str, np.ndarray] = {}
        hru_ids = ds['hruId'].values.tolist() if 'hruId' in ds.variables else None

        # Standard SUMMA attribute → regionalization column mapping
        attr_map = {
            'elevation': 'elev_m',
            'HRUarea':   'area_km2',
            'latitude':  'latitude',
            'longitude': 'longitude',
        }
        for nc_var, col_name in attr_map.items():
            if nc_var in ds.variables:
                data[col_name] = ds[nc_var].values.copy()

    df = pd.DataFrame(data)

    # --- Merge supplementary CSV if provided -------------------------------
    if csv_path is not None:
        csv_path = Path(csv_path)
        if csv_path.exists():
            csv_raw = read_hru_table(csv_path)
            if hru_ids is None and 'hru_id' in csv_raw.columns:
                # No hruId in attributes.nc to join on
                csv_df = None
            else:
                csv_df = align_table_to_hru_ids(
                    csv_raw,
                    hru_ids if hru_ids is not None else list(range(len(df))),
                    csv_path,
                    legacy_order='positional',
                    logger=logger,
                )
            if csv_df is not None:
                for col in csv_df.columns:
                    if col not in df.columns:
                        df[col] = csv_df[col].values
            else:
                logger.warning(
                    f"CSV attribute file {csv_path} could not be matched to the "
                    f"{len(df)} HRUs of attributes.nc — skipping CSV merge"
                )
        else:
            logger.warning(f"Transfer function attributes CSV not found: {csv_path}")

    # --- Auto-discover attribute profile CSVs -----------------------------
    # Profile processors write per-HRU CSVs to standard locations under
    # data/attributes/.  Discover and merge them so transfer functions can
    # reference richer attributes (soil.ksat, climate.tavg_annual_mean, etc.)
    _profile_csv_dirs = {
        'worldclim':   ('climate',   'worldclim_attributes.csv'),
        'soil':        ('soilclass', 'soil_extended_attributes.csv'),
        'landcover':   ('landclass', 'landcover_extended_attributes.csv'),
        'geology':     ('geology',   'glhymps_attributes.csv'),
    }
    attr_base = csv_path.parent.parent if csv_path is not None else None
    if attr_base is not None and attr_base.is_dir():
        for profile_name, (subdir, filename) in _profile_csv_dirs.items():
            profile_csv = attr_base / subdir / filename
            if not profile_csv.exists():
                continue
            try:
                profile_raw = read_hru_table(profile_csv)
            except (OSError, ValueError) as exc:
                logger.warning(f"Could not read {profile_name} attributes {profile_csv}: {exc}")
                continue
            if hru_ids is None:
                logger.warning(
                    f"attributes.nc has no hruId variable; cannot join {profile_name} "
                    f"attributes ({profile_csv}) to HRUs — skipping"
                )
                continue
            # Raises for a pre-hru_id table that cannot be matched safely.
            profile_df = align_table_to_hru_ids(
                profile_raw, hru_ids, profile_csv, logger=logger
            )
            if profile_df is None:
                continue
            added = []
            for col in profile_df.columns:
                if col not in df.columns:
                    df[col] = profile_df[col].values
                    added.append(col)
            if added:
                logger.info(
                    f"Merged {len(added)} {profile_name} profile attributes"
                )

    # --- Derive synthetic attributes if missing ----------------------------
    if 'precip_mm_yr' not in df.columns:
        # Placeholder – should be provided via CSV in production
        logger.debug("precip_mm_yr not available; using uniform placeholder (1000)")
        df['precip_mm_yr'] = 1000.0

    if 'aridity' not in df.columns:
        logger.debug("aridity not available; using uniform placeholder (1.0)")
        df['aridity'] = 1.0

    if 'snow_frac' not in df.columns:
        logger.debug("snow_frac not available; using uniform placeholder (0.3)")
        df['snow_frac'] = 0.3

    if 'temp_C' not in df.columns and 'elev_m' in df.columns:
        # Rough lapse-rate estimate: 15°C at sea level, -6.5°C/km
        df['temp_C'] = 15.0 - 0.0065 * df['elev_m']

    logger.info(
        f"Loaded HRU attributes: {len(df)} HRUs, columns={list(df.columns)}"
    )
    return df


def create_summa_regionalization(
    method: str,
    param_bounds: Dict[str, Tuple[float, float]],
    n_hrus: int,
    attributes_nc_path: Path,
    csv_path: Optional[Path] = None,
    param_config: Optional[Dict[str, Dict]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> ParameterRegionalization:
    """
    Create a regionalization strategy configured for SUMMA.

    Convenience wrapper around :class:`RegionalizationFactory` that loads
    HRU attributes and applies SUMMA-specific defaults.

    Args:
        method: ``'lumped'``, ``'transfer_function'``, ``'zones'``, or ``'distributed'``.
        param_bounds: ``{param_name: (min, max)}`` for the parameters to regionalize.
        n_hrus: Number of HRUs.
        attributes_nc_path: Path to SUMMA ``attributes.nc``.
        csv_path: Optional supplementary CSV with per-HRU attributes.
        param_config: Per-parameter config override (attribute mapping, calibrate_b).
                      Defaults to :data:`SUMMA_DEFAULT_PARAM_CONFIG`.
        extra_config: Additional factory config options (b_bounds, etc.).
        logger: Logger instance.

    Returns:
        A :class:`ParameterRegionalization` instance ready for use.
    """
    logger = logger or logging.getLogger(__name__)
    config: Dict[str, Any] = dict(extra_config or {})

    # Load attributes if the method needs them
    attributes = None
    if method in ('transfer_function',):
        attributes = load_hru_attributes(attributes_nc_path, csv_path, logger)

    # Apply SUMMA default param config unless overridden
    if 'TRANSFER_FUNCTION_PARAM_CONFIG' not in config:
        config['TRANSFER_FUNCTION_PARAM_CONFIG'] = param_config or SUMMA_DEFAULT_PARAM_CONFIG

    return RegionalizationFactory.create(
        method=method,
        param_bounds=param_bounds,
        n_units=n_hrus,
        config=config,
        attributes=attributes,
        logger=logger,
    )
