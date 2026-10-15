"""MESH errors in the core exception hierarchy.

Each class also derives from the built-in type it replaced (``ValueError`` or
``FileNotFoundError``), so existing handlers keep working. Example: the
calibration worker scores ``ValueError`` from cryosphere checks as a penalty
rather than a crash.
"""
from __future__ import annotations

from symfluence.core.exceptions import (
    ConfigurationError,
    FileOperationError,
    ModelExecutionError,
    ValidationError,
)


class MESHConfigurationError(ConfigurationError, ValueError):
    """Invalid or inconsistent MESH setup, bounds or options."""


class MESHInputError(ValidationError, ValueError):
    """Invalid MESH input data: parameter files, geometry, forcing or observations."""


class MESHOutputError(ModelExecutionError, ValueError):
    """Missing, incomplete or physically inconsistent native MESH output."""


class MESHFileNotFoundError(FileOperationError, FileNotFoundError):
    """A required MESH file is missing."""
