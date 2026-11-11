# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""FEWS adapter configuration models.

Defines the Pydantic models for the optional FEWS General Adapter section of
:class:`SymfluenceConfig`:

- ``FEWSConfig``: Top-level FEWS configuration (an optional section of the
  root config model).
- ``IDMapEntry``: Single variable mapping entry (FEWS name <-> SYMFLUENCE name
  + unit conversion).

These live in ``core.config`` because the root config model composes them; the
``symfluence.fews`` package re-exports them so the FEWS adapters can keep
importing from ``symfluence.fews.config``. ``fews`` -> ``core`` is the correct
dependency direction.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from symfluence.core.config.models.base import FROZEN_CONFIG


class IDMapEntry(BaseModel):
    """Single variable ID mapping between FEWS and SYMFLUENCE."""
    model_config = FROZEN_CONFIG

    fews_id: str = Field(..., description='Variable ID in FEWS (e.g., "P.obs")')
    symfluence_id: str = Field(..., description='Variable name in SYMFLUENCE (e.g., "pptrate")')
    fews_unit: Optional[str] = Field(default=None, description='Unit in FEWS (e.g., "mm/h")')
    symfluence_unit: Optional[str] = Field(default=None, description='Unit in SYMFLUENCE (e.g., "kg m-2 s-1")')
    conversion_factor: float = Field(default=1.0, description='Multiplicative factor: symfluence = fews * factor')
    conversion_offset: float = Field(default=0.0, description='Additive offset: symfluence = fews * factor + offset')
    direction: Literal['import', 'export', 'both'] = Field(
        default='both',
        description=(
            'Which way the entry applies: "import" (FEWS -> SYMFLUENCE, e.g. P.obs -> pptrate), '
            '"export" (SYMFLUENCE -> FEWS, e.g. discharge -> Q.sim) or "both". '
            'Several import entries may share a symfluence_id; an exported variable '
            'needs exactly one export entry.'
        ),
    )


class FEWSConfig(BaseModel):
    """Delft-FEWS General Adapter configuration."""
    model_config = FROZEN_CONFIG

    work_dir: str = Field(
        default='.',
        alias='FEWS_WORK_DIR',
        description='FEWS module working directory (contains toModel/, toFews/)'
    )
    data_format: Literal['pi-xml', 'netcdf-cf'] = Field(
        default='netcdf-cf',
        alias='FEWS_DATA_FORMAT',
        description='Data exchange format between FEWS and SYMFLUENCE'
    )
    id_map_file: Optional[str] = Field(
        default=None,
        alias='FEWS_ID_MAP_FILE',
        description='Path to YAML file with variable ID mappings'
    )
    id_map: List[IDMapEntry] = Field(
        default_factory=list,
        alias='FEWS_ID_MAP',
        description='Inline variable ID mappings'
    )
    state_dir: Optional[str] = Field(
        default=None,
        alias='FEWS_STATE_DIR',
        description='Directory for model warm-start state files'
    )
    diagnostics_file: str = Field(
        default='diag.xml',
        alias='FEWS_DIAGNOSTICS_FILE',
        description='Name of the PI diagnostics output file'
    )
    missing_value: float = Field(
        default=-999.0,
        alias='FEWS_MISSING_VALUE',
        description='Missing value sentinel used in PI-XML/NetCDF exchange'
    )
    auto_id_map: bool = Field(
        default=True,
        alias='FEWS_AUTO_ID_MAP',
        description='Auto-detect common forcing variable mappings via VariableStandardizer'
    )
