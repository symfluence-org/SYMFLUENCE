# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
CLI Services for SYMFLUENCE.

This package provides modular services for CLI operations:

Binary Management:
- ToolInstaller: External tool installation (clone, build)
- ToolValidator: Binary validation and testing
- SystemDiagnostics: System health checks and diagnostics

Project Management:
- InitializationService: Project initialization and scaffolding
- JobSchedulerService: SLURM/HPC job submission
- NotebookService: Jupyter notebook launching

Build Configuration:
- BuildInstructionsRegistry: Tool build configuration registry
- build_snippets: Shared shell script helpers
- build_schema: Build instruction schema and validation
"""
from __future__ import annotations

from symfluence.core.build.build_snippets import (
    get_all_snippets,
    get_bison_detection_and_build,
    get_common_build_environment,
    get_flex_detection_and_build,
    get_geos_proj_detection,
    get_hdf5_detection,
    get_netcdf_detection,
    get_netcdf_lib_detection,
    get_safe_build_path,
    get_udunits2_detection_and_build,
)

from .base import BaseService

# Build configuration
from .build_registry import BuildInstructionsRegistry
from .build_schema import (
    BuildInstructionSchema,
    VerifyInstallSchema,
    validate_all_instructions,
    validate_build_instructions,
)

# Project management services
from .initialization import InitializationManager, InitializationService
from .job_scheduler import JobScheduler, JobSchedulerService
from .notebook import NotebookService
from .system_deps import SystemDepsRegistry
from .system_diagnostics import SystemDiagnostics

# Binary management services
from .tool_installer import ToolInstaller
from .tool_validator import ToolValidator

__all__ = [
    # Base
    'BaseService',
    # Binary management services
    'ToolInstaller',
    'ToolValidator',
    'SystemDiagnostics',
    'SystemDepsRegistry',
    # Project management services
    'InitializationService',
    'InitializationManager',  # Backward compatibility alias
    'JobSchedulerService',
    'JobScheduler',  # Backward compatibility alias
    'NotebookService',
    # Build registry
    'BuildInstructionsRegistry',
    # Build snippets
    'get_common_build_environment',
    'get_netcdf_detection',
    'get_hdf5_detection',
    'get_netcdf_lib_detection',
    'get_safe_build_path',
    'get_geos_proj_detection',
    'get_udunits2_detection_and_build',
    'get_bison_detection_and_build',
    'get_flex_detection_and_build',
    'get_all_snippets',
    # Build schema
    'BuildInstructionSchema',
    'VerifyInstallSchema',
    'validate_build_instructions',
    'validate_all_instructions',
]
