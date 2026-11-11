# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Soil-depth pseudo-parameters: one implementation of the layer rescaling.

``total_soil_depth_multiplier`` (legacy name ``total_mult``) and
``shape_factor`` are calibrated like parameters but are applied by rewriting
the soil layer thicknesses in a SUMMA cold-state file. The arithmetic and the
file update used to exist three times (SUMMA parameter manager, SUMMA worker,
``optimization.core.transformers.SoilDepthTransformer``). They live here, below
both the models layer and the optimization package, so all three share them:
``optimization`` may not import ``models`` (see scripts/check_core_layering.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np

from symfluence.core.profiling import get_profiler

# Soil-depth pseudo-parameters: they rescale coldState layers, SUMMA has no
# parameter of that name.
DEPTH_MULTIPLIER_PARAM = 'total_soil_depth_multiplier'
LEGACY_DEPTH_MULTIPLIER_PARAM = 'total_mult'
DEPTH_SHAPE_PARAM = 'shape_factor'
DEPTH_PSEUDO_PARAMS: Tuple[str, ...] = (
    DEPTH_MULTIPLIER_PARAM, LEGACY_DEPTH_MULTIPLIER_PARAM, DEPTH_SHAPE_PARAM,
)


def record_error(message: str, logger, debug_info: Optional[Dict]) -> None:
    """Log *message* as an error and append it to ``debug_info['errors']``."""
    logger.error(message)
    if debug_info is not None:
        debug_info.setdefault('errors', []).append(message)


def record_file(entry: str, debug_info: Optional[Dict]) -> None:
    """Append *entry* to ``debug_info['files_checked']``."""
    if debug_info is not None:
        debug_info.setdefault('files_checked', []).append(entry)


def has_depth_params(params: Dict[str, Any]) -> bool:
    """Whether ``params`` carries any soil-depth pseudo-parameter."""
    return any(name in params for name in DEPTH_PSEUDO_PARAMS)


def _first(value: Any) -> Any:
    return value[0] if isinstance(value, np.ndarray) else value


def depth_multipliers(params: Dict[str, Any]) -> Tuple[float, float]:
    """Return ``(total_mult, shape_factor)`` from a trial parameter set.

    ``total_soil_depth_multiplier`` is the current name of the total-depth
    multiplier and ``total_mult`` the legacy one; either is accepted, and a
    missing multiplier or shape factor is neutral (1.0).
    """
    total_mult = params.get(DEPTH_MULTIPLIER_PARAM)
    if total_mult is None:
        total_mult = params.get(LEGACY_DEPTH_MULTIPLIER_PARAM, 1.0)
    shape_factor = params.get(DEPTH_SHAPE_PARAM, 1.0)
    return _first(total_mult), _first(shape_factor)


def calculate_soil_depths(original_depths: np.ndarray, total_mult: float, shape_factor: float) -> np.ndarray:
    """Rescale soil layer depths.

    ``shape_factor`` redistributes thickness exponentially over the layers
    (>1 thickens the deep layers, <1 the shallow ones) with weights that
    average 1; ``total_mult`` then scales every layer.
    """
    arr = np.asarray(original_depths).copy()
    n = len(arr)
    idx = np.arange(n)

    if shape_factor > 1:
        w = np.exp(idx / (n - 1) * np.log(shape_factor))
    elif shape_factor < 1:
        w = np.exp((n - 1 - idx) / (n - 1) * np.log(1 / shape_factor))
    else:
        w = np.ones(n)

    w /= w.mean()
    return arr * w * total_mult


def update_soil_depths(
    coldstate_path: Path,
    original_depths: Sequence[float],
    params: Dict[str, Any],
    logger,
    debug_info: Optional[Dict] = None,
    component: str = 'summa_worker',
) -> bool:
    """Write rescaled ``mLayerDepth`` / ``iLayerHeight`` into the coldState file."""
    try:
        total_mult, shape_factor = depth_multipliers(params)
        logger.debug(f"Updating soil depths: total_mult={total_mult:.3f}, shape_factor={shape_factor:.3f}")

        new_depths = calculate_soil_depths(np.array(original_depths), total_mult, shape_factor)

        heights = np.zeros(len(new_depths) + 1)
        for i in range(len(new_depths)):
            heights[i + 1] = heights[i] + new_depths[i]

        coldstate_path = Path(coldstate_path)
        if not coldstate_path.exists():
            record_error(f"{coldstate_path.name} not found: {coldstate_path}", logger, debug_info)
            return False

        record_file(f"{coldstate_path.name}: {coldstate_path}", debug_info)

        # Track coldState write (secondary IOPS bottleneck)
        with get_profiler().track_netcdf_write(str(coldstate_path), component=component):
            with nc.Dataset(coldstate_path, 'r+') as ds:
                if 'mLayerDepth' not in ds.variables or 'iLayerHeight' not in ds.variables:
                    record_error(
                        f"Required depth variables not found in {coldstate_path.name}", logger, debug_info)
                    return False

                num_hrus = ds.dimensions['hru'].size
                for h in range(num_hrus):
                    ds.variables['mLayerDepth'][:, h] = new_depths
                    ds.variables['iLayerHeight'][:, h] = heights

        logger.debug("Soil depths updated successfully")
        return True

    except (FileNotFoundError, IOError, ValueError) as e:
        record_error(f"Error updating soil depths: {str(e)}", logger, debug_info)
        return False
