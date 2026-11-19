# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION and MODFLOW.

MODFLOW reads no meteorological forcing: its recharge comes from the coupled
land model (HYPE, SUMMA, ...), which applies the correction itself (in a
forward run through its own ``forcing_correction.apply_forward``; in a
``COUPLED_GW`` calibration through the land model's adapter, the coupled
parameter manager routing the land parameters to the land model). There is
nothing to write here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """No-op: the land model feeding MODFLOW's recharge carries the correction."""
    return {'applied_by': 'land model (recharge)'}
