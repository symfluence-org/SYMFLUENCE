# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
SYMFLUENCE CLI Command Handlers

This package contains command handlers for the SYMFLUENCE CLI subcommand architecture.
Each module implements handlers for a specific command category.
"""
from __future__ import annotations

from .agent_commands import AgentCommands
from .binary_commands import BinaryCommands
from .config_commands import ConfigCommands
from .data_commands import DataCommands
from .doctor_commands import DoctorCommands
from .ensemble_commands import EnsembleCommands
from .example_commands import ExampleCommands
from .fews_commands import FEWSCommands
from .gui_commands import GUICommands
from .job_commands import JobCommands
from .list_commands import ListCommands
from .project_commands import ProjectCommands
from .tui_commands import TUICommands
from .workflow_commands import WorkflowCommands

__all__ = [
    'WorkflowCommands',
    'ProjectCommands',
    'BinaryCommands',
    'ConfigCommands',
    'JobCommands',
    'ExampleCommands',
    'AgentCommands',
    'GUICommands',
    'TUICommands',
    'DataCommands',
    'DoctorCommands',
    'EnsembleCommands',
    'FEWSCommands',
    'ListCommands',
]
