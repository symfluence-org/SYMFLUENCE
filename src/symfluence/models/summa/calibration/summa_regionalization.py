# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SUMMA-specific adapter for parameter regionalization.

Wraps the model-agnostic regionalization framework from
``models/fuse/calibration/parameter_regionalization`` with SUMMA defaults
and attribute loading from SUMMA ``attributes.nc`` files.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.models.fuse.calibration.parameter_regionalization import (
    ParameterRegionalization,
    RegionalizationFactory,
)

# Default mapping of SUMMA parameters to catchment attributes.
# ``calibrate_b=True`` means the slope coefficient is calibrated (spatial
# variation driven by the attribute); ``False`` means only the intercept
# is calibrated (spatially uniform).
SUMMA_DEFAULT_PARAM_CONFIG: Dict[str, Dict[str, Any]] = {
    'frozenPrecipMultip': {'attribute': 'precip_mm_yr', 'calibrate_b': True},
    'tempCritRain':       {'attribute': 'elev_m',       'calibrate_b': True},
    'k_soil':             {'attribute': 'aridity',      'calibrate_b': True},
    'theta_sat':          {'attribute': 'precip_mm_yr', 'calibrate_b': True},
    'vGn_n':              {'attribute': 'aridity',      'calibrate_b': False},
    'snowfrz_scale':      {'attribute': 'elev_m',       'calibrate_b': True},
    'albedoMax':          {'attribute': 'elev_m',       'calibrate_b': False},
    'routingGammaScale':  {'attribute': 'precip_mm_yr', 'calibrate_b': False},
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

    Args:
        attributes_nc_path: Path to SUMMA ``attributes.nc``.
        csv_path: Optional path to a CSV with additional per-HRU attributes.
                  Must contain a column that can be matched to HRU order or ID.
        logger: Logger instance.

    Returns:
        DataFrame with one row per HRU and attribute columns.
    """
    logger = logger or logging.getLogger(__name__)

    # --- Read from attributes.nc -------------------------------------------
    with xr.open_dataset(attributes_nc_path) as ds:
        data: Dict[str, np.ndarray] = {}

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
            csv_df = pd.read_csv(csv_path)
            if len(csv_df) == len(df):
                # Same HRU ordering – just concat columns
                for col in csv_df.columns:
                    if col not in df.columns:
                        df[col] = csv_df[col].values
            else:
                logger.warning(
                    f"CSV attribute file has {len(csv_df)} rows but "
                    f"attributes.nc has {len(df)} HRUs — skipping CSV merge"
                )
        else:
            logger.warning(f"Transfer function attributes CSV not found: {csv_path}")

    # --- Derive synthetic attributes if missing ----------------------------
    if 'precip_mm_yr' not in df.columns:
        # Placeholder – should be provided via CSV in production
        logger.info("precip_mm_yr not available; using uniform placeholder (1000)")
        df['precip_mm_yr'] = 1000.0

    if 'aridity' not in df.columns:
        logger.info("aridity not available; using uniform placeholder (1.0)")
        df['aridity'] = 1.0

    if 'snow_frac' not in df.columns:
        logger.info("snow_frac not available; using uniform placeholder (0.3)")
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
        n_subcatchments=n_hrus,
        config=config,
        attributes=attributes,
        logger=logger,
    )
