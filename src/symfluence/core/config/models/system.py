# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
System configuration model.

Contains SystemConfig for system-level settings: paths, logging, parallelism, debugging.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator

from .base import FROZEN_CONFIG


class SystemConfig(BaseModel):
    """System-level configuration: paths, logging, parallelism, debugging"""
    model_config = FROZEN_CONFIG

    # Required paths
    data_dir: Path = Field(alias='SYMFLUENCE_DATA_DIR')
    code_dir: Path = Field(alias='SYMFLUENCE_CODE_DIR')

    # System settings
    num_processes: int = Field(default=1, validation_alias=AliasChoices('NUM_PROCESSES', 'MPI_PROCESSES'))
    debug_mode: bool = Field(default=False, alias='DEBUG_MODE')
    log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(default='INFO', alias='LOG_LEVEL')
    log_to_file: bool = Field(default=True, alias='LOG_TO_FILE')
    log_format: str = Field(default='detailed', alias='LOG_FORMAT')
    force_run_all_steps: bool = Field(default=False, alias='FORCE_RUN_ALL_STEPS')
    # Note: force_download removed (unused field)
    use_local_scratch: bool = Field(default=False, alias='USE_LOCAL_SCRATCH')
    random_seed: Optional[int] = Field(default=None, alias='RANDOM_SEED')
    stop_on_error: bool = Field(default=True, alias='STOP_ON_ERROR')
    record_provenance: bool = Field(default=True, alias='RECORD_PROVENANCE')

    @field_validator('data_dir', 'code_dir')
    @classmethod
    def validate_paths(cls, v):
        """Expand and resolve paths, stripping Jupyter checkpoint dirs."""
        p = Path(v).expanduser().resolve()
        # Guard against Jupyter notebook checkpoint directories
        parts = p.parts
        clean_parts = [part for part in parts if part != '.ipynb_checkpoints']
        if len(clean_parts) != len(parts):
            import warnings
            p = Path(*clean_parts)
            warnings.warn(
                f"Path contained '.ipynb_checkpoints' — auto-corrected to: {p}",
                UserWarning, stacklevel=2
            )
        return p

    @field_validator('num_processes')
    @classmethod
    def validate_positive_integers(cls, v, info):
        """Ensure positive integers"""
        if v < 1:
            raise ValueError(f"{info.field_name} must be at least 1, got {v}")
        return v
