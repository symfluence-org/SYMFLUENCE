# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Lumped Wflow discharge: one conversion and routing path for every reader.

Lumped Wflow writes ``soil_surface_water__net_runoff_volume_flux`` (mm per
model step) as its ``Q`` column. Turning that into the discharge that is
scored and saved takes three steps -- deciding whether the column is a runoff
flux, converting mm/step to m3/s with the basin area and the model step, and
the optional post-hoc two-store routing (``ROUTE_*`` calibration parameters).
The calibration worker and the postprocessor both go through the functions
here, so the saved series is the series that was scored.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.exceptions import ConfigurationError

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
    q: pd.Series,
    toml_path: Path,
    staticmaps_path: Path,
    routing_params: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """Runoff-flux ``Q`` column -> routed discharge (m3/s).

    The single path from the lumped CSV column to discharge: unit conversion
    with the TOML's own step, then the post-hoc routing when ``ROUTE_*``
    parameters are given.
    """
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
