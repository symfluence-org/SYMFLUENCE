# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Basin-mode settings from the typed ``CLMPARFLOW_BASIN_*`` configuration.

Paths left at ``default`` resolve inside the project directory: the river-basin outline
(``shapefiles/river_basins``), the catchment HRUs (``shapefiles/catchment``, unit field ``HRU_ID``), the domain DEM
(``attributes/elevation/dem``), the pour point (``shapefiles/pour_point``) and the basin-averaged forcing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from symfluence.core.exceptions import ConfigurationError, FileOperationError

from .case import BasinParameters
from .chain import ChainSettings

GRID_SUBDIR = Path('settings') / 'CLMPARFLOW' / 'basin'


@dataclass
class BasinInputs:
    outline: Path
    units: Path
    unit_field: str
    dem: Path
    pour_point: Path
    crs: str
    dx: float
    min_slope: float
    ice_field: Optional[str]
    ice_value: str
    forcing: Path


def _first(directory: Path, patterns: Iterable[str], what: str) -> Path:
    for pattern in patterns:
        found = sorted(directory.rglob(pattern)) if directory.exists() else []
        if found:
            return found[0]
    raise FileOperationError(f'CLMParFlow basin mode: no {what} found under {directory}; set it explicitly')


def _path(value: str, default: Path) -> Path:
    return default if value in (None, '', 'default') else Path(value)


def basin_inputs(cfg, project_dir: Path, domain_name: str) -> BasinInputs:
    """Resolve the grid and forcing inputs (``cfg`` is ``config.model.clmparflow``)."""
    if not cfg.basin_crs:
        raise ConfigurationError('CLMParFlow basin mode needs CLMPARFLOW_BASIN_CRS (a projected CRS in metres, e.g. EPSG:3057)')
    from symfluence.core.mixins.project import resolve_forcing_basin_path

    shp = ('*.shp', '*.gpkg')
    outline = (_path(cfg.basin_outline, Path()) if cfg.basin_outline not in (None, '', 'default')
               else _first(project_dir / 'shapefiles' / 'river_basins', shp, 'river-basin outline'))
    units = (_path(cfg.basin_units, Path()) if cfg.basin_units not in (None, '', 'default')
             else _first(project_dir / 'shapefiles' / 'catchment', shp, 'catchment HRU file'))
    pour = (_path(cfg.basin_pour_point, Path()) if cfg.basin_pour_point not in (None, '', 'default')
            else _first(project_dir / 'shapefiles' / 'pour_point', shp, 'pour point'))
    dem = _path(cfg.basin_dem, project_dir / 'attributes' / 'elevation' / 'dem' / f'domain_{domain_name}_elv.tif')
    forcing = _path(cfg.basin_forcing, resolve_forcing_basin_path(project_dir))
    return BasinInputs(outline=outline, units=units, unit_field=cfg.basin_unit_field, dem=dem, pour_point=pour,
                       crs=str(cfg.basin_crs), dx=float(cfg.basin_dx), min_slope=float(cfg.basin_min_slope),
                       ice_field=cfg.basin_ice_field, ice_value=cfg.basin_ice_value, forcing=forcing)


def basin_parameters(cfg) -> BasinParameters:
    albedo = cfg.basin_ice_albedo
    if albedo is not None and len(albedo) != 2:
        raise ConfigurationError('CLMPARFLOW_BASIN_ICE_ALBEDO needs two values: visible and near-infrared')
    return BasinParameters(soil_k=cfg.basin_soil_k, aquifer_k=cfg.basin_aquifer_k, manning=cfg.basin_manning,
                           glacier_bed_k=cfg.basin_glacier_bed_k, precip_scale=cfg.basin_precip_scale,
                           ice_albedo=tuple(albedo) if albedo is not None else None,
                           initial_water_table_depth=cfg.basin_initial_water_table_depth)


def water_years(start: str, end: str) -> List[int]:
    """Complete October-September water years inside [start, end]."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    first = s.year + 1 if s <= pd.Timestamp(s.year, 10, 1) else s.year + 2  # first 1 October on or after start
    last = e.year if e >= pd.Timestamp(e.year, 9, 30) else e.year - 1  # last 30 September on or before end
    return list(range(first, last + 1))


def chain_settings(cfg, time_start: str, time_end: str, num_procs: int) -> ChainSettings:
    """Production water years default to the complete water years of the experiment period; spin-up to its first
    three (rewound and repeated until the storage drift test passes)."""
    years = water_years(time_start, time_end)
    first = cfg.basin_first_water_year or (years[0] if years else None)
    last = cfg.basin_last_water_year or (years[-1] if years else None)
    if first is None or last is None or last < first:
        raise ConfigurationError(f'CLMParFlow basin mode: no complete water year in {time_start}..{time_end}')
    production = list(range(first, last + 1))
    spinup = list(cfg.basin_spinup_years or production[:3])
    initial = cfg.basin_initial_state
    return ChainSettings(spinup_years=spinup, production_years=production, drift_tol=cfg.basin_drift_tol,
                         max_cycles=cfg.basin_max_spinup_cycles,
                         initial_state=_path(initial, Path()) if initial else None, P=max(1, num_procs), Q=1,
                         forcing_unit_variable=cfg.basin_forcing_unit_variable,
                         forcing_names=dict(cfg.basin_forcing_variables or {}))
