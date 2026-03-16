# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE Model Evaluators Package

This package contains base evaluators for different hydrological variables including:
- Streamflow (routed and non-routed)
- Snow (SWE, SCA, depth)
- Groundwater (depth, GRACE TWS)
- Evapotranspiration (ET, latent heat)
- Soil moisture (point, SMAP, ESA)

Model-specific streamflow evaluators (GR, HYPE, RHESSys) have been moved to
symfluence.optimization.calibration_targets for consistency with the calibration
target pattern. Use those modules for model-specific calibration.
"""

from .base import ModelEvaluator
from .et import ETEvaluator
from .groundwater import GroundwaterEvaluator
from .snow import SnowEvaluator
from .soil_moisture import SoilMoistureEvaluator
from .streamflow import StreamflowEvaluator
from .tws import TWSEvaluator

__all__ = [
    "ModelEvaluator",
    "ETEvaluator",
    "StreamflowEvaluator",
    "SoilMoistureEvaluator",
    "SnowEvaluator",
    "GroundwaterEvaluator",
    "TWSEvaluator",
]

# ---------------------------------------------------------------------------
# Evaluator aliases — kept separate from canonical registrations so that
# provenance records and config diffs always use the canonical name.
# ---------------------------------------------------------------------------
from symfluence.core.registries import R  # noqa: E402

# ET aliases (canonical: ET)
R.evaluators.alias('MODIS_ET', 'ET')
R.evaluators.alias('MOD16', 'ET')
R.evaluators.alias('FLUXNET', 'ET')
R.evaluators.alias('FLUXNET_ET', 'ET')

# Snow aliases (canonical: SNOW)
R.evaluators.alias('SCA', 'SNOW')
R.evaluators.alias('SWE', 'SNOW')

# Soil moisture aliases (canonical: SOIL_MOISTURE)
R.evaluators.alias('SM', 'SOIL_MOISTURE')
R.evaluators.alias('SM_POINT', 'SOIL_MOISTURE')
R.evaluators.alias('SM_SMAP', 'SOIL_MOISTURE')
R.evaluators.alias('SM_ISMN', 'SOIL_MOISTURE')
R.evaluators.alias('SM_ESA', 'SOIL_MOISTURE')
