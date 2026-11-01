# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Core mixins for SYMFLUENCE modules.

Provides base mixins for logging, configuration, project context, file operations,
validation, and timing that other utilities and mixins can build upon.

Quick Start
-----------
For most use cases, use ``ConfigurableMixin`` which combines all core mixins:

>>> from symfluence.core.mixins import ConfigurableMixin
>>>
>>> class MyProcessor(ConfigurableMixin):
...     def __init__(self, config):
...         self.config = config  # Required
...
...     def process(self):
...         self.logger.info(f"Processing {self.domain_name}")
...         self.ensure_dir(self.project_dir / "output")
...         with self.time_limit("processing"):
...             # do work
...             pass

Mixin Hierarchy
---------------
::

    ConfigurableMixin (recommended - combines all core mixins)
    ├── LoggingMixin          # self.logger property
    ├── ProjectContextMixin   # project paths (data_dir, project_dir, domain_name)
    │   └── ConfigMixin       # self.config + convenience properties
    ├── FileUtilsMixin        # ensure_dir, copy_file, copy_tree, safe_delete
    ├── ValidationMixin       # validate_config, validate_file, validate_dir
    └── TimingMixin           # time_limit context manager

    ShapefileAccessMixin      # Shapefile column name accessors (standalone)
    └── ConfigMixin

Individual Mixins
-----------------
LoggingMixin
    Provides ``self.logger`` - a module-specific logger instance.

ConfigMixin
    Provides ``self.config`` property and convenience accessors for common
    configuration values (experiment_id, time_start, time_end, etc.).

ProjectContextMixin (extends ConfigMixin)
    Provides path properties: ``data_dir``, ``domain_name``, ``project_dir``,
    and standard subdirectories (project_forcing_dir, project_simulations_dir, etc.).

ShapefileAccessMixin (extends ConfigMixin)
    Provides shapefile column name accessors for catchment, river, and basin IDs.

FileUtilsMixin
    Provides file operations: ``ensure_dir``, ``copy_file``, ``copy_tree``, ``safe_delete``.

ValidationMixin
    Provides validation utilities: ``validate_config``, ``validate_file``, ``validate_dir``.

TimingMixin
    Provides ``time_limit`` context manager for measuring execution time.

ConfigurableMixin
    **Recommended for most use cases.** Combines all core mixins above
    (except ShapefileAccessMixin which is domain-specific).

See Also
--------
- core.calibration.mixins : ParallelExecutionMixin, ResultsTrackingMixin, etc.
- core.modeling.mixins : PETCalculatorMixin, ObservationLoaderMixin, etc.
- data.acquisition.mixins : RetryMixin, ChunkedDownloadMixin, SpatialSubsetMixin
"""
from __future__ import annotations

from .config import ConfigMixin
from .configurable import ConfigurableMixin
from .file_utils import FileUtilsMixin
from .logging import LoggingMixin
from .project import ProjectContextMixin
from .shapefile import ShapefileAccessMixin, ShapefileColumnProperty, shapefile_column
from .timing import TimingMixin
from .validation import ValidationMixin

__all__ = [
    # Base mixins (alphabetical)
    "ConfigMixin",
    "FileUtilsMixin",
    "LoggingMixin",
    "ProjectContextMixin",
    "ShapefileAccessMixin",
    "ShapefileColumnProperty",
    "shapefile_column",
    "TimingMixin",
    "ValidationMixin",
    # Combined mixin (recommended for most use cases)
    "ConfigurableMixin",
]
