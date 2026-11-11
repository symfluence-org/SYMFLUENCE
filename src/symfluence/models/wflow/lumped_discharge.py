# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Lumped Wflow discharge: one conversion and routing path for every reader.

Lumped Wflow is one active SBM cell with a pit. Three fluxes take water out
of that cell (Wflow.jl 1.0.4, ``mass_balance.jl``; confirmed with a one-cell
run in which ``P - ET - dS`` equals their sum to 1e-12 mm per step):

- ``soil_surface_water__net_runoff_volume_flux`` (mm per step, the ``Q``
  column): runoff of the land fraction, the inflow of overland flow;
- the runoff of the river fraction (``net_runoff_river``), which has no
  standard name of its own but is the whole of
  ``river_water__lateral_inflow_volume_flow_rate`` (m3/s) when no upstream
  cell drains into the river;
- lateral subsurface flow ``subsurface_water__volume_flow_rate`` (m3/d). It
  only reaches a river through the *downstream* cell
  (``subsurface_water__to_river_volume_flow_rate`` sums over upstream cells),
  so with one cell it leaves the model across the cell boundary.

The two volume rates refer to Wflow's own cell area (from the grid
coordinates), not to the basin area, and are turned into mm per step with it.
The sum is the runoff of the basin; turning it into the discharge that is
scored and saved then takes the conversion mm/step -> m3/s with the basin
area and the model step, and the optional post-hoc two-store routing
(``ROUTE_*`` calibration parameters). The calibration worker and the
postprocessor both go through :func:`lumped_discharge_cms`, so the saved
series is the series that was scored.
"""
from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

#: Standard name of the lumped ``Q`` column (mm per step).
NET_RUNOFF_PARAMETER = 'soil_surface_water__net_runoff_volume_flux'
#: Extra CSV columns of a lumped run: header -> (standard name, seconds per
#: unit of time of the volume rate). Summed with ``Q`` into the cell outflow.
LUMPED_OUTFLOW_COLUMNS: Dict[str, Tuple[str, float]] = {
    'Q_river_inflow': ('river_water__lateral_inflow_volume_flow_rate', 1.0),   # m3 s-1
    'Q_subsurface': ('subsurface_water__volume_flow_rate', 86400.0),           # m3 d-1
}

_warned_legacy_tomls: Set[str] = set()

#: Sidecar next to the staticmaps holding the ``ROUTE_*`` values that belong
#: to the parameter set written into those staticmaps.
ROUTING_PARAMS_FILE = 'wflow_lumped_routing.json'
ROUTING_PARAM_NAMES = ('ROUTE_ALPHA', 'ROUTE_BETA', 'ROUTE_SPLIT', 'ROUTE_BASEFLOW')


def is_runoff_flux_output(toml_path: Path) -> bool:
    """True when the TOML's discharge column is a runoff flux (mm per step).

    Units follow the selected discharge column only, never unrelated
    diagnostic columns elsewhere in the configuration.
    """
    native = tomllib.loads(Path(toml_path).read_text())
    columns = native.get('output', {}).get('csv', {}).get('column', [])
    for header in ('Q', 'Q_av', 'q_av'):
        selected = next((c for c in columns if c.get('header') == header), None)
        if selected is not None:
            return 'net_runoff_volume_flux' in selected.get('parameter', '')
    if len(columns) == 1:
        return 'net_runoff_volume_flux' in columns[0].get('parameter', '')
    return False


def model_timestep_seconds(toml_path: Path) -> int:
    """Model step (s) from ``[time] timestepsecs`` of the run's TOML.

    Raises rather than assuming hourly: a wrong step scales the whole
    discharge series (3x for 3-hourly forcing, 24x for daily).
    """
    native = tomllib.loads(Path(toml_path).read_text())
    step = native.get('time', {}).get('timestepsecs')
    if isinstance(step, bool) or not isinstance(step, (int, float)) or step <= 0:
        raise ConfigurationError(
            f"{toml_path} has no positive [time] timestepsecs; cannot convert "
            f"lumped Wflow runoff (mm per step) to m3/s")
    return int(step)


def basin_area_m2(staticmaps_path: Path) -> float:
    """Basin area (m2): sum of ``wflow_cellarea`` in the staticmaps."""
    staticmaps_path = Path(staticmaps_path)
    if not staticmaps_path.exists():
        raise ConfigurationError(
            f"Wflow staticmaps not found at {staticmaps_path}; cannot convert "
            f"lumped Wflow runoff (mm per step) to m3/s")
    with xr.open_dataset(staticmaps_path) as ds:
        if 'wflow_cellarea' not in ds:
            raise ConfigurationError(f"{staticmaps_path} has no wflow_cellarea")
        area = float(np.nansum(ds['wflow_cellarea'].values))
    if not area > 0:
        raise ConfigurationError(f"{staticmaps_path}: wflow_cellarea sums to {area}")
    return area


def _csv_columns(toml_path: Path) -> Dict[str, str]:
    """``header -> parameter`` of the TOML's ``[[output.csv.column]]`` entries."""
    native = tomllib.loads(Path(toml_path).read_text())
    columns = native.get('output', {}).get('csv', {}).get('column', [])
    return {c.get('header', ''): c.get('parameter', '') for c in columns}


def wflow_cell_area_m2(staticmaps_path: Path, toml_path: Path) -> float:
    """Area (m2) Wflow itself assigns to the single active cell.

    Wflow does not read ``wflow_cellarea``; it derives the cell size from the
    grid coordinates (``cell_lengths`` / ``lattometres`` in Wflow.jl
    ``utils.jl``, reproduced here), or takes the grid spacing as metres when
    ``[model] cell_length_in_meter__flag`` is set. Its volume rates (m3/s,
    m3/d) refer to this area.
    """
    staticmaps_path = Path(staticmaps_path)
    if not staticmaps_path.exists():
        raise ConfigurationError(
            f"Wflow staticmaps not found at {staticmaps_path}; cannot convert "
            f"lumped Wflow volume rates to mm per step")
    with xr.open_dataset(staticmaps_path) as ds:
        missing = [k for k in ('wflow_subcatch', 'x', 'y') if k not in ds.variables]
        if missing:
            raise ConfigurationError(
                f"{staticmaps_path} has no {', '.join(missing)}; cannot derive the Wflow cell area")
        active = np.argwhere(np.isfinite(ds['wflow_subcatch'].transpose('y', 'x').values))
        x = np.asarray(ds['x'].values, dtype=float)
        y = np.asarray(ds['y'].values, dtype=float)
    if len(active) != 1:
        raise ConfigurationError(
            f"{staticmaps_path}: lumped Wflow discharge needs exactly one active cell, "
            f"found {len(active)}")
    if x.size < 2:
        raise ConfigurationError(f"{staticmaps_path}: x axis has fewer than two points")
    cell_length = abs(float(np.mean(np.diff(x))))
    native = tomllib.loads(Path(toml_path).read_text())
    if native.get('model', {}).get('cell_length_in_meter__flag', False):
        area = cell_length * cell_length
    else:
        lat = np.deg2rad(float(y[active[0][0]]))
        lat_len = 111132.92 - 559.82 * np.cos(2 * lat) + 1.175 * np.cos(4 * lat) - 0.0023 * np.cos(6 * lat)
        lon_len = 111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat) + 0.118 * np.cos(5 * lat)
        area = float(lon_len * cell_length * lat_len * cell_length)
    if not area > 0:
        raise ConfigurationError(f"{staticmaps_path}: Wflow cell area is {area}")
    return area


def lumped_outflow_mm_per_step(
    output: Union[pd.Series, pd.DataFrame],
    toml_path: Path,
    staticmaps_path: Path,
) -> pd.Series:
    """Total water leaving the lumped cell, mm per model step.

    ``Q`` (net runoff of the land fraction) plus the river-fraction runoff and
    the lateral subsurface flow, when the TOML writes them (see the module
    docstring). A TOML generated before these columns existed has only ``Q``:
    the series is then net runoff alone, as it always was for that setup, and
    a warning says what is missing. A TOML that declares the columns while the
    output lacks them is an error.
    """
    declared = _csv_columns(toml_path)
    frame = output.to_frame('Q') if isinstance(output, pd.Series) else output
    q_header = next((h for h in ('Q', 'Q_av', 'q_av') if h in frame.columns), frame.columns[0])
    total = frame[q_header].astype(float)

    present = [h for h, (parameter, _) in LUMPED_OUTFLOW_COLUMNS.items() if declared.get(h) == parameter]
    if not present:
        key = str(Path(toml_path).resolve())
        if key not in _warned_legacy_tomls:
            _warned_legacy_tomls.add(key)
            logger.warning(
                "%s writes only net runoff for the lumped cell: lateral subsurface flow and "
                "river-fraction runoff leave the model but are not in the discharge "
                "(P - ET - dS > Q). Regenerate the Wflow settings to add the %s columns.",
                toml_path, ' and '.join(LUMPED_OUTFLOW_COLUMNS))
        return total
    if len(present) != len(LUMPED_OUTFLOW_COLUMNS):
        raise ConfigurationError(
            f"{toml_path} declares {present} but not all of {list(LUMPED_OUTFLOW_COLUMNS)}; "
            f"the lumped outflow needs both")
    absent = [h for h in present if h not in frame.columns]
    if absent:
        raise ConfigurationError(
            f"{toml_path} declares the lumped outflow columns {absent} but the Wflow "
            f"output does not contain them; rerun the model with this TOML")

    step = model_timestep_seconds(toml_path)
    cell_area = wflow_cell_area_m2(staticmaps_path, toml_path)
    for header in present:
        seconds_per_unit = LUMPED_OUTFLOW_COLUMNS[header][1]
        # m3 per unit time over the Wflow cell -> mm per model step
        total = total + frame[header].astype(float) * (step / seconds_per_unit) / cell_area * 1000.0
    return total.rename(q_header)


def runoff_mm_per_step_to_cms(q: pd.Series, area_m2: float, timestep_seconds: float) -> pd.Series:
    """mm per model step over ``area_m2`` -> m3/s."""
    return q * area_m2 / (timestep_seconds * 1000.0)


def apply_routing(q: pd.Series, routing_params: Dict[str, float]) -> pd.Series:
    """Two-store linear reservoir routing (fast + slow) with baseflow offset.

    Splits instantaneous runoff into fast (surface) and slow (baseflow)
    components, each routed through an exponential reservoir, plus a
    constant baseflow term for irreducible glacial/deep-GW contribution::

        S_fast(t) = alpha * S_fast(t-1) + split * Q_in(t)
        S_slow(t) = beta  * S_slow(t-1) + (1-split) * Q_in(t)
        Q_out(t)  = (1-alpha) * S_fast(t) + (1-beta) * S_slow(t) + baseflow

    Parameters (from routing_params dict):
        ROUTE_ALPHA:    fast reservoir retention [0, 0.95]
        ROUTE_BETA:     slow reservoir retention [0.9, 0.9999]
        ROUTE_SPLIT:    fraction to fast store [0.1, 0.9]
        ROUTE_BASEFLOW: constant baseflow offset in m3/s [0, 15]
    """
    alpha = routing_params.get('ROUTE_ALPHA', 0.5)
    beta = routing_params.get('ROUTE_BETA', 0.98)
    split = routing_params.get('ROUTE_SPLIT', 0.5)
    baseflow = routing_params.get('ROUTE_BASEFLOW', 0.0)
    vals = q.values.astype(float)
    n = len(vals)
    s_fast = 0.0
    s_slow = 0.0
    out = np.empty(n)
    for i in range(n):
        q_in = max(vals[i], 0.0)
        s_fast = alpha * s_fast + split * q_in
        s_slow = beta * s_slow + (1.0 - split) * q_in
        out[i] = (1.0 - alpha) * s_fast + (1.0 - beta) * s_slow + baseflow
    return pd.Series(out, index=q.index, name=q.name)


def lumped_discharge_cms(
    output: Union[pd.Series, pd.DataFrame],
    toml_path: Path,
    staticmaps_path: Path,
    routing_params: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Lumped Wflow output columns -> routed discharge (m3/s).

    The single path from the lumped CSV to discharge: the outflow of the cell
    (:func:`lumped_outflow_mm_per_step`; pass the whole output table so the
    subsurface and river-fraction columns are included), unit conversion with
    the basin area and the TOML's own step, then the post-hoc routing when
    ``ROUTE_*`` parameters are given.
    """
    q = lumped_outflow_mm_per_step(output, toml_path, staticmaps_path)
    q = runoff_mm_per_step_to_cms(q, basin_area_m2(staticmaps_path), model_timestep_seconds(toml_path))
    if routing_params:
        q = apply_routing(q, routing_params)
    return q


def save_routing_params(settings_dir: Path, params: Dict[str, float]) -> None:
    """Record the ``ROUTE_*`` values of a parameter set next to its staticmaps.

    A parameter set without routing parameters removes the sidecar, so a
    stale file from an earlier calibration is never applied to it.
    """
    path = Path(settings_dir) / ROUTING_PARAMS_FILE
    routing = {k: float(v) for k, v in params.items() if k in ROUTING_PARAM_NAMES}
    if routing:
        path.write_text(json.dumps(routing, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    else:
        path.unlink(missing_ok=True)


def load_routing_params(settings_dir: Path) -> Optional[Dict[str, float]]:
    """``ROUTE_*`` values recorded by :func:`save_routing_params`, or None."""
    path = Path(settings_dir) / ROUTING_PARAMS_FILE
    if not path.exists():
        return None
    routing = json.loads(path.read_text(encoding='utf-8'))
    unknown = set(routing) - set(ROUTING_PARAM_NAMES)
    if unknown:
        raise ConfigurationError(f"{path}: unknown routing parameters {sorted(unknown)}")
    return {k: float(v) for k, v in routing.items()}
