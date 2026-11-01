# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Configuration mixin for SYMFLUENCE modules.

Provides standardized access to typed SymfluenceConfig configuration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, cast

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig



def resolve_config_value(
    config: Any,
    typed_accessor: Optional[Callable[[], Any]] = None,
    default: Any = None,
    dict_key: Optional[str] = None,
) -> Any:
    """The single implementation of typed-first config lookup.

    Config reaches code both as a typed ``SymfluenceConfig`` and as a flat
    dict (e.g. inside calibration worker processes). This tries
    ``typed_accessor()`` first; if it raises ``AttributeError``/``KeyError``/
    ``TypeError`` (a dict config, or an unset optional section) or returns
    ``None``, it falls back to ``config.get(dict_key)``, then ``default``.

    ``tests/unit/core/test_config_paths_resolve.py`` statically checks the
    typed paths passed through ``_get_config_value``, since a mistyped path
    would otherwise silently fall through to the default.
    """
    if callable(typed_accessor):
        try:
            value = typed_accessor()
            if value is not None:
                return value
        except (AttributeError, KeyError, TypeError):
            pass
    if dict_key is not None and config is not None and hasattr(config, 'get'):
        value = config.get(dict_key)
        if value is not None:
            return value
    return default

class ConfigMixin:
    """
    Mixin for classes that use typed SymfluenceConfig configuration.

    Provides standardized access to configuration via the typed config object.
    The config_dict property provides a flattened dictionary view for legacy
    code that iterates over keys.
    """

    @property
    def config(self) -> 'SymfluenceConfig':  # type: ignore[misc]
        """
        Get the typed configuration object.

        Returns:
            SymfluenceConfig instance
        """
        return cast('SymfluenceConfig', getattr(self, '_config', None))

    @config.setter
    def config(self, value: 'SymfluenceConfig') -> None:
        """
        Set the typed configuration object.

        Args:
            value: SymfluenceConfig instance
        """
        self._config = value

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        Get configuration as a flattened dictionary.

        This property provides a flattened dict view of the typed config
        for code that needs to iterate over keys or use string-based access.
        The dict is cached internally by SymfluenceConfig for performance.

        Returns:
            Flattened configuration dictionary with uppercase keys
        """
        # Check for override dict first
        if hasattr(self, '_config_dict_override') and self._config_dict_override is not None:
            return self._config_dict_override
        cfg = self.config
        if cfg is not None:
            # Handle both SymfluenceConfig and plain dict
            if isinstance(cfg, dict):
                return cfg
            return cfg.to_dict(flatten=True)
        return {}

    @config_dict.setter
    def config_dict(self, value: Dict[str, Any]) -> None:
        """
        Set the configuration dictionary override.

        This allows direct setting of config values for testing or
        overriding specific configuration entries.

        Args:
            value: Dictionary of configuration values
        """
        self._config_dict_override = value

    def _get_config_value(
        self,
        typed_accessor: Callable[[], Any],
        default: Any = None,
        dict_key: Optional[str] = None
    ) -> Any:
        """
        Get a configuration value from typed config with default fallback.

        Args:
            typed_accessor: Callable that accesses the typed config value,
                           e.g., lambda: self.config.domain.name
            default: Default value if accessor fails or returns None
            dict_key: Optional legacy dict key for backward compatibility
                     with dict-based configs (e.g., 'DOMAIN_NAME')

        Returns:
            Configuration value or default

        Example:
            name = self._get_config_value(
                lambda: self.config.domain.name,
                default='unnamed'
            )

            # With backward compatibility for dict configs:
            data_dir = self._get_config_value(
                lambda: self.config.system.data_dir,
                default='.',
                dict_key='SYMFLUENCE_DATA_DIR'
            )
        """
        # Check override dict first (for testing and explicit overrides)
        if dict_key is not None and hasattr(self, '_config_dict_override') and self._config_dict_override is not None:
            if dict_key in self._config_dict_override:
                return self._config_dict_override[dict_key]

        return resolve_config_value(self.config, typed_accessor, default, dict_key)

    @property
    def forcing_discretization(self) -> Optional[str]:
        """The run's spatial discretization (``SUB_GRID_DISCRETIZATION``).

        Identifies which catchment discretization this run uses (``lumped``,
        ``elevation``, ``grus``, ...) — i.e. the HRU set its forcing was remapped
        onto. Pass it to
        :func:`symfluence.core.modeling.model_ready.forcing_reader.open_canonical_forcing`
        so a model reads only the forcing matching ITS discretization when a
        domain's store holds more than one (issue #339).
        """
        return self._get_config_value(
            lambda: self.config.domain.discretization,
            dict_key='SUB_GRID_DISCRETIZATION',
        )

    def _resolve_point_bbox(self) -> Optional[str]:
        """Derive a square bounding box from the pour point for point domains.

        Single source of truth shared by the point delineator and the data
        acquisition layer: a ``point`` domain may be configured with only
        ``POUR_POINT_COORDS``, in which case the spatial extent is a small square
        centred on that point with half-width ``POINT_BUFFER_DISTANCE`` degrees
        (default 0.01°, ~1 km).

        Returns:
            A ``"lat_max/lon_min/lat_min/lon_max"`` (north/west/south/east) string,
            or ``None`` when the domain is not a point domain or no parseable
            pour point is configured (callers decide whether that is an error).
        """
        method = str(self._get_config_value(
            lambda: self.config.domain.definition_method, default='',
            dict_key='DOMAIN_DEFINITION_METHOD') or '').lower()
        if method != 'point':
            return None

        pour_point = self._get_config_value(
            lambda: self.config.domain.pour_point_coords, default=None,
            dict_key='POUR_POINT_COORDS')
        if not pour_point:
            return None
        try:
            lat, lon = (float(part) for part in str(pour_point).split('/'))
        except (ValueError, TypeError):
            return None

        buffer = self._get_config_value(
            lambda: self.config.domain.delineation.point_buffer_distance, default=None,
            dict_key='POINT_BUFFER_DISTANCE')
        buffer = 0.01 if buffer is None else float(buffer)
        return f"{lat + buffer}/{lon - buffer}/{lat - buffer}/{lon + buffer}"

    # =========================================================================
    # Convenience Properties for Common Config Values
    # =========================================================================

    @property
    def experiment_id(self) -> str:
        """Experiment identifier from config.domain.experiment_id."""
        _experiment_id = getattr(self, '_experiment_id', None)
        if _experiment_id is not None:
            return _experiment_id
        return self._get_config_value(
            lambda: self.config.domain.experiment_id,
            default='run_1',
            dict_key='EXPERIMENT_ID'
        )

    @experiment_id.setter
    def experiment_id(self, value: str) -> None:
        """Set the experiment identifier."""
        self._experiment_id = value

    @property
    def domain_definition_method(self) -> str:
        """Domain definition method from config.domain.definition_method."""
        return self._get_config_value(
            lambda: self.config.domain.definition_method,
            default='lumped',
            dict_key='DOMAIN_DEFINITION_METHOD'
        )

    @property
    def time_start(self) -> Optional[str]:
        """Experiment start time from config.domain.time_start."""
        return self._get_config_value(
            lambda: self.config.domain.time_start,
            default=None
        )

    @property
    def time_end(self) -> Optional[str]:
        """Experiment end time from config.domain.time_end."""
        return self._get_config_value(
            lambda: self.config.domain.time_end,
            default=None
        )

    @property
    def sub_grid_discretization(self) -> str:
        """Sub-grid discretization method from config.domain.discretization."""
        return self._get_config_value(
            lambda: self.config.domain.discretization,
            default='lumped'
        )

    @property
    def domain_discretization(self) -> str:
        """Alias for sub_grid_discretization (backward compatibility)."""
        return self.sub_grid_discretization

    @property
    def calibration_period(self) -> Optional[str]:
        """Calibration period string from config.domain.calibration_period."""
        _calibration_period = getattr(self, '_calibration_period', None)
        if _calibration_period is not None:
            return _calibration_period
        return self._get_config_value(
            lambda: self.config.domain.calibration_period,
            default=None
        )

    @calibration_period.setter
    def calibration_period(self, value) -> None:
        """Set the calibration period (can be string or tuple)."""
        self._calibration_period = value

    @property
    def spinup_period(self) -> Optional[str]:
        """Spinup period string from config.domain.spinup_period."""
        return self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default=None
        )

    @property
    def evaluation_period(self) -> Optional[str]:
        """Evaluation period string from config.domain.evaluation_period."""
        _evaluation_period = getattr(self, '_evaluation_period', None)
        if _evaluation_period is not None:
            return _evaluation_period
        return self._get_config_value(
            lambda: self.config.domain.evaluation_period,
            default=None
        )

    @evaluation_period.setter
    def evaluation_period(self, value) -> None:
        """Set the evaluation period (can be string or tuple)."""
        self._evaluation_period = value

    @property
    def forcing_dataset(self) -> str:
        """Forcing dataset name from config.forcing.dataset."""
        # Allow override via _forcing_dataset_override attribute
        if hasattr(self, '_forcing_dataset_override') and self._forcing_dataset_override:
            return self._forcing_dataset_override.lower()
        return self._get_config_value(
            lambda: self.config.forcing.dataset,
            default=''
        ).lower()

    @forcing_dataset.setter
    def forcing_dataset(self, value: str) -> None:
        """Set forcing dataset override (for backward compatibility)."""
        self._forcing_dataset_override = value

    @property
    def forcing_time_step_size(self) -> int:
        """Forcing time step size in seconds from config.forcing.time_step_size."""
        return int(self._get_config_value(
            lambda: self.config.forcing.time_step_size,
            default=3600
        ))

    @property
    def hydrological_model(self) -> str:
        """Hydrological model name from config.model.hydrological_model."""
        model = self._get_config_value(
            lambda: self.config.model.hydrological_model,
            default=''
        )
        # Handle list case (multi-model)
        if isinstance(model, list):
            return model[0] if model else ''
        return model

    @property
    def routing_model(self) -> str:
        """Routing model name from config.model.routing_model."""
        return self._get_config_value(
            lambda: self.config.model.routing_model,
            default='none'
        )

    @property
    def optimization_metric(self) -> str:
        """Optimization metric from config.optimization.metric."""
        return self._get_config_value(
            lambda: self.config.optimization.metric,
            default='KGE'
        )
