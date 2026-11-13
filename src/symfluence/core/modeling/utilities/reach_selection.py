# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""One rule for picking the evaluation reach from routed (mizuRoute) output.

Calibration workers, extractors, postprocessors and the streamflow evaluator
used to disagree -- highest-mean segment, last segment (``seg=-1``) or first
segment (``seg=0``) -- so calibration and the final ``results`` CSV could
score different reaches. Everything now goes through
:func:`select_outlet_index`:

1. ``SIM_REACH_ID`` when set: the reach whose ``reachID``/``segId`` equals
   it. An explicitly requested reach that is absent (or duplicated) raises
   :class:`EvaluationError` -- silently scoring another reach is worse.
2. Otherwise the network outlet from the mizuRoute topology file
   (``downSegId`` <= 0 or pointing outside the network). With several
   outlets (independent networks) the one with the highest mean flow wins.
3. Otherwise -- no topology or no reach ids in the output -- the reach with
   the highest mean flow, with a warning (once per process).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from symfluence.core.exceptions import EvaluationError

logger = logging.getLogger(__name__)

#: Spatial dimension names used by mizuRoute output.
REACH_DIMS: Sequence[str] = ('seg', 'reachID')
#: Per-reach id variables in mizuRoute output / topology.
REACH_ID_VARS: Sequence[str] = ('reachID', 'segId')

_warned_highest_mean = False


def resolve_sim_reach_id(value: Any) -> Optional[int]:
    """Normalise a configured SIM_REACH_ID to an int, or None when unset."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    if not isinstance(value, str):
        return None  # not a configured value (e.g. an unset typed field)
    text = value.strip()
    if not text or text.lower() in ('default', 'none', 'null'):
        return None
    try:
        return int(float(text))
    except ValueError as e:
        raise EvaluationError(f"SIM_REACH_ID must be an integer reach id, got {value!r}") from e


def topology_candidates(
    project_dir: Optional[Path] = None,
    topology_name: Optional[str] = None,
    near: Optional[Path] = None,
    max_levels: int = 6,
) -> Iterable[Path]:
    """Plausible mizuRoute topology files: the project's, then any found
    in ``settings/mizuRoute`` above ``near`` (an output file)."""
    name = topology_name or 'topology.nc'
    if project_dir is not None:
        yield Path(project_dir) / 'settings' / 'mizuRoute' / name
    if near is not None:
        for parent in list(Path(near).resolve().parents)[:max_levels]:
            yield parent / 'settings' / 'mizuRoute' / name


def topology_candidates_from_config(config: Any, near: Optional[Path] = None) -> Iterable[Path]:
    """:func:`topology_candidates` for a typed or flat-dict config."""
    from symfluence.core.mixins.config import resolve_config_value

    project_dir = None
    if config is not None:
        domain = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        data_dir = resolve_config_value(config, lambda: config.system.data_dir, None, 'SYMFLUENCE_DATA_DIR')
        if domain and data_dir:
            project_dir = Path(data_dir) / f"domain_{domain}"
    name = resolve_config_value(config, lambda: config.model.mizuroute.topology, dict_key='SETTINGS_MIZU_TOPOLOGY') if config is not None else 'topology.nc'
    if not name or name == 'default':
        name = 'topology.nc'
    return topology_candidates(project_dir, name, near=near)


def sim_reach_id_from_config(config: Any) -> Any:
    """Raw SIM_REACH_ID from a typed or flat-dict config."""
    from symfluence.core.mixins.config import resolve_config_value

    if config is None:
        return None
    return resolve_config_value(config, lambda: config.evaluation.sim_reach_id, None, 'SIM_REACH_ID')


def outlet_reach_ids(topology_file: Path) -> Optional[np.ndarray]:
    """Segment ids of the network outlets in a mizuRoute topology file."""
    import xarray as xr

    try:
        with xr.open_dataset(topology_file) as topo:
            if 'segId' not in topo or 'downSegId' not in topo:
                return None
            seg = np.asarray(topo['segId'].values).ravel()
            down = np.asarray(topo['downSegId'].values).ravel()
    except (OSError, ValueError, KeyError) as e:
        logger.debug(f"Cannot read mizuRoute topology {topology_file}: {e}")
        return None
    outlets = seg[(down <= 0) | ~np.isin(down, seg)]
    return outlets if outlets.size else None


def _reach_dim(var: Any) -> Optional[str]:
    return next((d for d in REACH_DIMS if d in var.dims), None)


def _reach_ids(ds: Any, dim: str) -> Optional[np.ndarray]:
    for name in REACH_ID_VARS:
        if name in ds.variables and ds[name].ndim == 1 and ds[name].dims[0] == dim:
            return np.asarray(ds[name].values).ravel()
    return None


def select_outlet_index(
    ds: Any,
    var: Any,
    sim_reach_id: Any = None,
    topology_files: Iterable[Path] = (),
    log: Optional[logging.Logger] = None,
) -> Optional[int]:
    """Index along the reach dimension of ``var`` to evaluate.

    Returns None when ``var`` has no reach dimension (already a series).
    See the module docstring for the rule.
    """
    global _warned_highest_mean
    log = log or logger
    dim = _reach_dim(var)
    if dim is None:
        return None
    ids = _reach_ids(ds, dim)

    requested = resolve_sim_reach_id(sim_reach_id)
    if requested is not None:
        if ids is None:
            raise EvaluationError(
                f"SIM_REACH_ID={requested} requires a reachID or segId variable in the routed output"
            )
        matches = np.flatnonzero(ids == requested)
        if len(matches) != 1:
            raise EvaluationError(
                f"Requested routing reach SIM_REACH_ID={requested} is missing or duplicated "
                f"in the routed output"
            )
        return int(matches[0])

    if var.sizes[dim] == 1:
        return 0

    means = None
    if ids is not None:
        for topology_file in topology_files:
            if not Path(topology_file).exists():
                continue
            outlets = outlet_reach_ids(Path(topology_file))
            if outlets is None:
                continue
            idx = np.flatnonzero(np.isin(ids, outlets))
            if idx.size == 1:
                return int(idx[0])
            if idx.size > 1:
                means = np.asarray(var.mean(dim='time').values, dtype=float)
                return int(idx[np.nanargmax(means[idx])])
            break

    if means is None:
        means = np.asarray(var.mean(dim='time').values, dtype=float)
    if not _warned_highest_mean:
        log.warning(
            "No SIM_REACH_ID and no usable mizuRoute topology: evaluating the reach "
            "with the highest mean flow as the outlet"
        )
        _warned_highest_mean = True
    return int(np.nanargmax(means))


def select_outlet_series(
    ds: Any,
    var: Any,
    sim_reach_id: Any = None,
    topology_files: Iterable[Path] = (),
    log: Optional[logging.Logger] = None,
) -> pd.Series:
    """``var`` at the evaluation reach (see :func:`select_outlet_index`)."""
    idx = select_outlet_index(ds, var, sim_reach_id, topology_files, log)
    if idx is None:
        return var.to_pandas()
    return var.isel({_reach_dim(var): idx}).to_pandas()
