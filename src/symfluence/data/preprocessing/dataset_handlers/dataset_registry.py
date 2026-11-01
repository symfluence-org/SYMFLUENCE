# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Dataset Registry for SYMFLUENCE

Provides a central registry for dataset preprocessing handlers.
Uses standardized BaseRegistry pattern with lowercase key normalization.
"""
from __future__ import annotations

from typing import List

from symfluence.core.mixins.config import resolve_config_value
from symfluence.data.base_registry import BaseRegistry


class DatasetRegistry(BaseRegistry):
    """
    Registry for dataset preprocessing handlers.

    Handlers register via the unified registry (``@R.dataset_handlers.add('x')``)
    and are retrieved using get_handler(). All keys are normalized to lowercase.
    """

    _r_registry_name = "dataset_handlers"

    @classmethod
    def get_handler(
        cls,
        name: str,
        *args,
        **kwargs
    ):
        """
        Get an instance of the appropriate dataset handler.

        Args:
            name: Name of the dataset (case-insensitive)
            *args: Positional arguments (config, logger, project_dir)
            **kwargs: Additional handler arguments

        Returns:
            Handler instance

        Raises:
            ValueError: If handler not found
        """
        handler_class = cls._get_handler_class(name)

        # Dataset handlers typically expect (config, logger, project_dir)
        # and extra kwargs for forcing_timestep_seconds etc.

        # If config is provided in args or kwargs, try to inject defaults
        config = kwargs.get('config')
        if not config and len(args) > 0:
            config = args[0]

        if config:
            kwargs.setdefault("forcing_timestep_seconds", resolve_config_value(config, lambda: config.forcing.time_step_size, 3600, 'FORCING_TIME_STEP_SIZE'))

        return handler_class(*args, **kwargs)

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names (alias for list_handlers)."""
        return cls.list_handlers()
