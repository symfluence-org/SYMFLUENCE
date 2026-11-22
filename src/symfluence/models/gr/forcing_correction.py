# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Shared FORCING_CORRECTION in GR forward runs.

GR applies PCORR/TCORR to BasinObs inside the R run, so there is
nothing to write: GRRunner._forcing_corrections resolves the same values
itself when no calibration parameter set is passed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def apply_forward(config: Any, logger: Any, project_dir: Path, info: Dict[str, Any]) -> Dict[str, Any]:
    """Nothing to write for GR; the runner applies the values (see module docstring)."""
    return {'applied_by': 'GRRunner._forcing_corrections'}
