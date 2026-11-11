# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Base class for model-specific forcing processors.

Provides common initialization, file operations, and logging patterns
used across FUSE, SUMMA, HYPE, and other model forcing processors.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from symfluence.core.mixins import ConfigurableMixin

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class BaseForcingProcessor(ABC, ConfigurableMixin):
    """
    Abstract base class for model-specific forcing data processors.

    Provides:
    - Common initialization patterns for config, logger, and paths
    - Standard utility methods for file operations
    - Consistent logging interface
    - Template for model-specific forcing data preparation

    Subclasses must implement:
    - model_name property: For identification in logs
    - Main processing method (signature varies by model)

    Attributes:
        config: Configuration dictionary
        logger: Logger instance
        input_path: Path to input forcing data
        output_path: Path to output processed forcing data
    """

    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        input_path: Path,
        output_path: Path,
        **kwargs
    ):
        """
        Initialize base forcing processor attributes.

        Args:
            config: Configuration (typed SymfluenceConfig or legacy dict)
            logger: Logger instance
            input_path: Path to input forcing data (e.g., basin-averaged)
            output_path: Path to output processed forcing data
            **kwargs: Additional model-specific attributes
        """
        # Set up typed config via ConfigurableMixin
        from symfluence.core.config.coercion import coerce_config
        self._config = coerce_config(config, warn=False)
        # Backward compatibility alias
        self.config = self._config

        self.logger = logger
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        # Store additional kwargs as attributes
        for key, value in kwargs.items():
            if isinstance(value, (str, Path)) and key.endswith('_path'):
                setattr(self, key, Path(value))
            else:
                setattr(self, key, value)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for logging (e.g., 'FUSE', 'SUMMA', 'HYPE')."""
        pass

    def _ensure_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured output directory exists: {self.output_path}")

    def _load_forcing_files(
        self,
        pattern: str = '*.nc',
        path: Optional[Path] = None
    ) -> List[Path]:
        """
        Load forcing files matching pattern from input path.

        Args:
            pattern: Glob pattern for file matching (default: '*.nc')
            path: Optional path override (default: self.input_path)

        Returns:
            Sorted list of matching file paths

        Raises:
            FileNotFoundError: If no files match the pattern
        """
        search_path = path or self.input_path
        files = sorted(search_path.glob(pattern))

        if not files:
            raise FileNotFoundError(
                f"No forcing files matching '{pattern}' found in {search_path}"
            )

        self.logger.debug(f"Found {len(files)} forcing files in {search_path}")
        return files

    def _select_run_files(self, files: List[Path]) -> List[Path]:
        """Narrow *files* to the forcing remapped onto this run's HRUs (issue #339).

        Uses the ``forcing_selector`` the owning preprocessor passed in (its
        ``_select_run_forcing``); without one, *files* are returned unchanged.
        """
        selector = getattr(self, 'forcing_selector', None)
        return list(selector(list(files))) if selector is not None else list(files)

    def _cleanup_stale_files(
        self,
        pattern: str,
        path: Optional[Path] = None,
        prefix: Optional[str] = None
    ) -> int:
        """
        Remove stale files matching pattern from output directory.

        Args:
            pattern: Glob pattern for files to remove
            path: Optional path override (default: self.output_path)
            prefix: Optional prefix filter for files

        Returns:
            Number of files removed
        """
        cleanup_path = path or self.output_path
        removed_count = 0

        for file in cleanup_path.glob(pattern):
            if prefix is None or file.name.startswith(prefix):
                file.unlink()
                removed_count += 1

        if removed_count > 0:
            self.logger.debug(
                f"Removed {removed_count} stale files matching '{pattern}' from {cleanup_path}"
            )

        return removed_count

    def _log_operation_start(self, operation: str) -> None:
        """Log the start of a processing operation."""
        self.logger.info(f"[{self.model_name}] Starting: {operation}")

    def _log_operation_complete(
        self,
        operation: str,
        file_count: Optional[int] = None,
        output_path: Optional[Path] = None
    ) -> None:
        """
        Log the completion of a processing operation.

        Args:
            operation: Name of the operation
            file_count: Optional number of files processed
            output_path: Optional output path to include in log
        """
        msg = f"[{self.model_name}] Completed: {operation}"
        if file_count is not None:
            msg += f" ({file_count} files)"
        if output_path is not None:
            msg += f" -> {output_path}"
        self.logger.info(msg)
