# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model-facing access to host-provided calibration targets."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

_TARGET_EXPORTS: Mapping[str, str] = {
    "streamflow": "StreamflowTarget", "flow": "StreamflowTarget", "discharge": "StreamflowTarget",
    "swe": "SnowTarget", "sca": "SnowTarget", "snow_depth": "SnowTarget", "snow": "SnowTarget",
    "gw_depth": "GroundwaterTarget", "gw_grace": "GroundwaterTarget",
    "groundwater": "GroundwaterTarget", "gw": "GroundwaterTarget",
    "et": "ETTarget", "latent_heat": "ETTarget", "evapotranspiration": "ETTarget",
    "sm_point": "SoilMoistureTarget", "sm_smap": "SoilMoistureTarget",
    "sm_esa": "SoilMoistureTarget", "sm_ismn": "SoilMoistureTarget",
    "soil_moisture": "SoilMoistureTarget", "sm": "SoilMoistureTarget",
    "tws": "TWSTarget", "grace": "TWSTarget", "grace_tws": "TWSTarget",
    "total_storage": "TWSTarget", "stor_grace": "TWSTarget", "stor_mb": "TWSTarget",
    "multivariate": "MultivariateTarget",
    "glacier_mb": "GlacierMBTarget", "glacier_mass_balance": "GlacierMBTarget",
    "basin_swe": "BasinSWETarget", "basin_sca": "BasinSCATarget", "basin_et": "BasinETTarget",
}


def resolve_calibration_target(target_type: str, *, default: str = "streamflow") -> type:
    """Resolve a generic host calibration target class at call time."""
    from symfluence.optimization import calibration_targets

    export_name = _TARGET_EXPORTS.get(target_type.lower(), _TARGET_EXPORTS[default.lower()])
    return getattr(calibration_targets, export_name)


def create_calibration_target(
    target_type: str,
    config: Any,
    project_dir: Path,
    logger: logging.Logger,
    *,
    default: str = "streamflow",
) -> Any:
    """Create a generic host calibration target for a model adapter."""
    target_class = resolve_calibration_target(target_type, default=default)
    return target_class(config, project_dir, logger)


__all__ = ["create_calibration_target", "resolve_calibration_target"]
