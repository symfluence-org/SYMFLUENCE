# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Root configuration model for SYMFLUENCE.

Contains SymfluenceConfig - the main configuration class that orchestrates
all other config models and provides validation, factory methods, and
backward compatibility.
"""
from __future__ import annotations

import logging
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

from .data import DataConfig
from .domain import DomainConfig
from .evaluation import EvaluationConfig
from .fews import FEWSConfig
from .forcing import ForcingConfig
from .model_configs import ModelConfig
from .optimization import OptimizationConfig
from .paths import PathsConfig
from .state_config import DataAssimilationConfig, StateConfig
from .system import SystemConfig


def _sanitize_cwd() -> Path:
    """Return CWD with any .ipynb_checkpoints segments stripped."""
    cwd = Path.cwd().resolve()
    parts = cwd.parts
    clean = [p for p in parts if p != '.ipynb_checkpoints']
    return Path(*clean) if len(clean) != len(parts) else cwd


# Legacy spatial-mode tokens that older configs placed in the discretization key
# (before DOMAIN_DEFINITION_METHOD existed), mapped to the canonical method.
# Deliberately excludes 'lumped' and 'point', which are valid discretization values
# in their own right ('lumped' = a single HRU per GRU).
_LEGACY_SPATIAL_MODE_TOKENS = {
    'semi_distributed': 'semidistributed',
    'semi-distributed': 'semidistributed',
    'semidistributed': 'semidistributed',
    'distributed': 'distributed',
}


class SymfluenceConfig(BaseModel):
    """
    Hierarchical root configuration model for SYMFLUENCE.

    Organizes 346+ configuration parameters into logical nested sections:
    - system: System settings (paths, logging, MPI)
    - domain: Domain definition (timing, spatial extent, discretization)
    - forcing: Meteorological forcing data
    - model: Hydrological model configurations
    - optimization: Calibration and optimization settings
    - evaluation: Evaluation data and analysis
    - paths: File paths and directories

    Features:
    - Type-safe hierarchical access: config.domain.name vs config['DOMAIN_NAME']
    - Factory methods: from_preset(), from_minimal(), from_file()
    - Backward compatibility: to_dict(), get(), __getitem__()
    - Immutable after creation (frozen=True) to prevent mutation bugs
    - All validation logic preserved from original flat model
    """

    # Immutable config to prevent mutation bugs
    model_config = ConfigDict(
        extra='allow',  # Allow extra keys for extensibility
        frozen=True,   # Immutable configs for thread safety and caching
        populate_by_name=True,  # Allow both field names and aliases for backward compat
    )

    # ========================================
    # NESTED CONFIGURATION SECTIONS
    # ========================================

    system: SystemConfig = Field(default_factory=lambda: SystemConfig(
        SYMFLUENCE_DATA_DIR=_sanitize_cwd() / 'data',
        SYMFLUENCE_CODE_DIR=_sanitize_cwd()
    ))
    domain: DomainConfig = Field(default_factory=lambda: DomainConfig(
        DOMAIN_NAME='unnamed_domain',
        EXPERIMENT_ID='run_1',
        EXPERIMENT_TIME_START='2010-01-01 00:00',
        EXPERIMENT_TIME_END='2020-12-31 23:00',
        DOMAIN_DEFINITION_METHOD='lumped',
        SUB_GRID_DISCRETIZATION='grus'
    ))
    data: DataConfig = Field(default_factory=DataConfig)
    forcing: ForcingConfig = Field(default_factory=lambda: ForcingConfig(FORCING_DATASET='ERA5'))
    model: ModelConfig = Field(default_factory=lambda: ModelConfig(HYDROLOGICAL_MODEL='SUMMA'))
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    data_assimilation: Optional[DataAssimilationConfig] = Field(default=None)
    fews: Optional[FEWSConfig] = Field(default=None)

    # ========================================
    # CROSS-FIELD VALIDATORS (from original model)
    # ========================================

    @classmethod
    def _migrate_legacy_spatial_mode(cls, values: dict) -> None:
        """Back-compat: migrate spatial mode expressed in the discretization key.

        Older flat configs (pre-DOMAIN_DEFINITION_METHOD) set the spatial mode via
        ``SUB_GRID_DISCRETIZATION``/``DOMAIN_DISCRETIZATION: semi_distributed`` (etc.).
        When ``DOMAIN_DEFINITION_METHOD`` is absent and the discretization key holds
        such a spatial-mode token, move it to ``DOMAIN_DEFINITION_METHOD`` and reset
        the discretization to ``GRUs`` (one HRU per GRU), with a deprecation warning.
        Mutates ``values`` in place; no-op for current configs.
        """
        if values.get('DOMAIN_DEFINITION_METHOD'):
            return

        def _spatial_token(key):
            v = values.get(key)
            if isinstance(v, str):
                return _LEGACY_SPATIAL_MODE_TOKENS.get(v.strip().lower())
            return None

        # Find the discretization key that holds a spatial-mode token (canonical
        # SUB_GRID_DISCRETIZATION wins over its legacy DOMAIN_DISCRETIZATION alias).
        disc_key = next(
            (k for k in ('SUB_GRID_DISCRETIZATION', 'DOMAIN_DISCRETIZATION') if _spatial_token(k)),
            None,
        )
        if disc_key is None:
            return
        canonical = _spatial_token(disc_key)

        warnings.warn(
            f"{disc_key}: '{values[disc_key]}' is a legacy spatial-mode value. Set "
            f"DOMAIN_DEFINITION_METHOD: {canonical} and use SUB_GRID_DISCRETIZATION "
            f"for the sub-grid method (e.g. 'GRUs', 'elevation'). Auto-migrated for now.",
            DeprecationWarning,
            stacklevel=2,
        )
        values['DOMAIN_DEFINITION_METHOD'] = canonical

        # Preserve a genuine sub-grid method if the *other* discretization key
        # carries one (e.g. SUB_GRID='semi_distributed' + DOMAIN_DISCRETIZATION='elevation');
        # otherwise default to one HRU per GRU. Either way SUB_GRID_DISCRETIZATION is
        # the single key that survives, so the required-field check still passes.
        other_key = 'DOMAIN_DISCRETIZATION' if disc_key == 'SUB_GRID_DISCRETIZATION' else 'SUB_GRID_DISCRETIZATION'
        other_val = values.get(other_key)
        if isinstance(other_val, str) and _spatial_token(other_key) is None:
            values['SUB_GRID_DISCRETIZATION'] = other_val
        else:
            values['SUB_GRID_DISCRETIZATION'] = 'GRUs'
        values.pop('DOMAIN_DISCRETIZATION', None)

    @model_validator(mode='before')
    @classmethod
    def normalize_flat_config(cls, values):
        """Normalize flat config inputs into nested sections with required checks."""
        if isinstance(values, cls):
            return values
        if not isinstance(values, dict):
            return values

        cls._migrate_legacy_spatial_mode(values)

        from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP
        from symfluence.core.config.transformers import transform_flat_to_nested

        # Build combined mapping: base + model-specific transformers
        combined_map = dict(FLAT_TO_NESTED_MAP)
        hydrological_model = values.get('HYDROLOGICAL_MODEL')
        if hydrological_model:
            try:
                from symfluence.core.config.config_resolution import get_config_transformers
                model_transformers = get_config_transformers(hydrological_model)
                if model_transformers:
                    combined_map.update(model_transformers)
            except (ImportError, KeyError, AttributeError):
                pass

        flat_keys = {key for key in values if key in combined_map}
        if not flat_keys:
            return values

        required_flat = {
            'SYMFLUENCE_DATA_DIR',
            'SYMFLUENCE_CODE_DIR',
            'DOMAIN_NAME',
            'EXPERIMENT_ID',
            'EXPERIMENT_TIME_START',
            'EXPERIMENT_TIME_END',
            'DOMAIN_DEFINITION_METHOD',
            'SUB_GRID_DISCRETIZATION',
            'HYDROLOGICAL_MODEL',
            'FORCING_DATASET',
        }

        if 'DOMAIN_DEFINITION_METHOD' in values:
            from symfluence.core.config.constants import VALID_DOMAIN_METHODS_WITH_LEGACY
            allowed_definition_methods = VALID_DOMAIN_METHODS_WITH_LEGACY
            if values.get('DOMAIN_DEFINITION_METHOD') not in allowed_definition_methods:
                raise ValueError(
                    "DOMAIN_DEFINITION_METHOD must be one of "
                    + ", ".join(sorted(allowed_definition_methods))
                )

        def has_nested_value(data, path):
            current = data
            for part in path:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            return current is not None and current != ''

        missing = []
        for key in required_flat:
            if key in values and values[key] not in (None, ''):
                continue
            path = FLAT_TO_NESTED_MAP.get(key)
            if path and has_nested_value(values, path):
                continue
            missing.append(key)

        if missing:
            raise ValueError(
                "Missing required configuration keys: " + ", ".join(sorted(missing))
            )

        nested = transform_flat_to_nested({key: values[key] for key in flat_keys})

        sections = ('system', 'domain', 'data', 'forcing', 'model', 'optimization', 'evaluation', 'paths', 'state', 'data_assimilation', 'fews')
        for section in sections:
            if section in values and isinstance(values[section], dict):
                nested.setdefault(section, {}).update(values[section])
            if not nested.get(section):
                nested.pop(section, None)

        for key, value in values.items():
            if key in flat_keys or key in sections:
                continue
            nested[key] = value

        # Route top-level 'lapse' into forcing section for LapseRateConfig
        if 'lapse' in nested and isinstance(nested['lapse'], dict):
            nested.setdefault('forcing', {})['lapse'] = nested.pop('lapse')

        return nested

    @model_validator(mode='after')
    def validate_time_periods(self):
        """Validate that time periods make logical sense"""
        from symfluence.core.exceptions import ConfigurationError, ValidationError
        from symfluence.core.validation import validate_date_range

        try:
            start, end = validate_date_range(
                self.domain.time_start,
                self.domain.time_end,
                context="experiment time period",
                logger=logger,
            )

            # Validate calibration period is within experiment period
            if self.domain.calibration_period:
                cal_start, cal_end = self._parse_period(self.domain.calibration_period)
                if cal_start < start or cal_end > end:
                    raise ConfigurationError(
                        f"CALIBRATION_PERIOD ({self.domain.calibration_period}) must be within "
                        f"EXPERIMENT_TIME_START ({start}) and EXPERIMENT_TIME_END ({end})"
                    )

            # Validate evaluation period is within experiment period
            if self.domain.evaluation_period:
                eval_start, eval_end = self._parse_period(self.domain.evaluation_period)
                if eval_start < start or eval_end > end:
                    raise ConfigurationError(
                        f"EVALUATION_PERIOD ({self.domain.evaluation_period}) must be within "
                        f"EXPERIMENT_TIME_START ({start}) and EXPERIMENT_TIME_END ({end})"
                    )

        except ConfigurationError:
            raise
        except ValidationError as e:
            raise ConfigurationError(str(e)) from e
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid date format: {e}") from e

        return self

    @model_validator(mode='after')
    def validate_coordinates(self):
        """Validate coordinate formats and bounds"""
        from symfluence.core.exceptions import ConfigurationError, ValidationError
        from symfluence.core.validation import parse_pour_point_coords, validate_bounding_box

        # Validate pour point coordinates
        if self.domain.pour_point_coords:
            try:
                lat, lon = self.domain.pour_point_coords.split('/')
                lat_f, lon_f = float(lat), float(lon)

                if not (-90 <= lat_f <= 90):
                    raise ConfigurationError(
                        f"POUR_POINT_COORDS latitude {lat_f} out of range [-90, 90]"
                    )
                if not (-180 <= lon_f <= 180):
                    raise ConfigurationError(
                        f"POUR_POINT_COORDS longitude {lon_f} out of range [-180, 180]"
                    )
            except ValueError:
                raise ConfigurationError(
                    f"POUR_POINT_COORDS must be 'lat/lon' format, got '{self.domain.pour_point_coords}'"
                ) from None

        # Validate additional interior outlet coordinates (one or many 'lat/lon').
        # parse_pour_point_coords raises ConfigurationError on any malformed/out-of-range pair.
        if self.domain.pour_point_additional_coords:
            parse_pour_point_coords(
                self.domain.pour_point_additional_coords,
                context='POUR_POINT_ADDITIONAL_COORDS',
            )

        # Validate bounding box coordinates
        if self.domain.bounding_box_coords:
            try:
                north, west, south, east = self.domain.bounding_box_coords.split('/')
                north_f, west_f, south_f, east_f = float(north), float(west), float(south), float(east)

                validate_bounding_box(
                    {
                        'lat_min': south_f,
                        'lat_max': north_f,
                        'lon_min': west_f,
                        'lon_max': east_f,
                    },
                    context="BOUNDING_BOX_COORDS",
                    logger=logger,
                )
            except ValueError:
                raise ConfigurationError(
                    f"BOUNDING_BOX_COORDS must be 'north/west/south/east' format, got '{self.domain.bounding_box_coords}'"
                ) from None
            except ConfigurationError:
                raise
            except (ValidationError, TypeError) as e:
                # Translate lat_min/lat_max to south/north for user-facing messages
                msg = str(e).replace('lat_min', 'south latitude').replace('lat_max', 'north latitude')
                msg = msg.replace('lon_min', 'west longitude').replace('lon_max', 'east longitude')
                raise ConfigurationError(msg) from e

        return self

    @model_validator(mode='after')
    def validate_model_requirements(self):
        """
        Validate model-specific required fields based on HYDROLOGICAL_MODEL.

        Delegates to the model config-resolution helpers for all
        model-specific validation.
        """
        from symfluence.core.config.config_resolution import validate_model_config
        from symfluence.core.config.flattening import flatten_nested_config
        from symfluence.core.exceptions import ConfigurationError

        models = self._parse_models()
        flat_config = flatten_nested_config(self)
        all_errors = []

        for model_name in models:
            try:
                validate_model_config(model_name, flat_config)
            except Exception as e:  # noqa: BLE001 — configuration resilience
                all_errors.append(f"{model_name}: {str(e)}")

        if all_errors:
            raise ConfigurationError(
                "Model configuration validation failed:\n"
                + "\n".join(f"  - {error}" for error in all_errors)
            )

        return self

    @model_validator(mode='after')
    def validate_spatial_mode_consistency(self):
        """Validate and auto-align spatial modes with domain definition"""

        models = self._parse_models()
        issues = []

        # Auto-align FUSE spatial mode with domain definition
        if 'FUSE' in models and self.model.fuse:
            # Map domain definition to appropriate FUSE spatial mode
            domain_to_fuse_mode = {
                'lumped': 'lumped',
                'point': 'lumped',
                'semidistributed': 'semi_distributed',
                'semi_distributed': 'semi_distributed',
                'distributed': 'distributed',
                'discretized': 'distributed',  # Treat discretized as distributed
                'subset': 'semi_distributed',
                'delineate': 'semi_distributed',
            }
            expected_fuse_mode = domain_to_fuse_mode.get(self.domain.definition_method)

            if expected_fuse_mode and self.model.fuse.spatial_mode != expected_fuse_mode:
                # Warn about mismatch; downstream code infers correct mode from
                # DOMAIN_DEFINITION_METHOD when FUSE_SPATIAL_MODE is not explicit.
                issues.append(
                    f"FUSE_SPATIAL_MODE is '{self.model.fuse.spatial_mode}' but "
                    f"DOMAIN_DEFINITION_METHOD is '{self.domain.definition_method}' "
                    f"(expected '{expected_fuse_mode}'). Downstream processing will "
                    f"infer the correct spatial mode automatically."
                )

        # Check GR spatial mode
        if 'GR' in models and self.model.gr:
            if self.domain.definition_method in ['distributed', 'discretized'] and self.model.gr.spatial_mode == 'lumped':
                issues.append(
                    f"GR_SPATIAL_MODE is 'lumped' but DOMAIN_DEFINITION_METHOD is '{self.domain.definition_method}'. "
                    f"Consider setting GR_SPATIAL_MODE to 'distributed'."
                )

        # Log alignment actions as info, other issues as warnings
        if issues:
            for issue in issues:
                if 'Auto-aligned' in issue or 'infer the correct spatial mode' in issue:
                    logger.info(f"Spatial mode configuration: {issue}")
                else:
                    warnings.warn(f"Spatial mode configuration: {issue}", UserWarning)

        return self

    @model_validator(mode='after')
    def validate_optimization_configuration(self):
        """Validate optimization algorithm and parameter settings"""
        from symfluence.core.exceptions import ConfigurationError

        # Check if optimization is enabled
        optimization_methods = self.optimization.methods
        if isinstance(optimization_methods, str):
            optimization_methods = [m.strip() for m in optimization_methods.split(',') if m.strip()]

        if not optimization_methods or len(optimization_methods) == 0:
            # No optimization configured, skip validation
            return self

        errors = []

        # Validate optimization algorithm
        from symfluence.core.config.constants import VALID_OPTIMIZATION_ALGORITHMS
        if self.optimization.algorithm not in VALID_OPTIMIZATION_ALGORITHMS:
            errors.append(
                f"ITERATIVE_OPTIMIZATION_ALGORITHM '{self.optimization.algorithm}' not recognized. "
                f"Valid algorithms: {', '.join(sorted(VALID_OPTIMIZATION_ALGORITHMS))}"
            )

        # Validate optimization metric (uppercase for case-insensitive matching)
        from symfluence.core.config.constants import VALID_OPTIMIZATION_METRICS
        if self.optimization.metric not in VALID_OPTIMIZATION_METRICS:
            errors.append(
                f"OPTIMIZATION_METRIC '{self.optimization.metric}' not recognized. "
                f"Valid metrics: {', '.join(sorted(VALID_OPTIMIZATION_METRICS))}"
            )

        # Validate iterations and population size
        if self.optimization.iterations < 1:
            errors.append(f"NUMBER_OF_ITERATIONS must be >= 1, got {self.optimization.iterations}")

        if self.optimization.population_size < 1:
            errors.append(f"POPULATION_SIZE must be >= 1, got {self.optimization.population_size}")

        # Algorithm-specific validation
        if self.optimization.algorithm in ['DE', 'Differential Evolution'] and self.optimization.de:
            if not (0 <= self.optimization.de.scaling_factor <= 2):
                errors.append(f"DE_SCALING_FACTOR should be in [0, 2], got {self.optimization.de.scaling_factor}")
            if not (0 <= self.optimization.de.crossover_rate <= 1):
                errors.append(f"DE_CROSSOVER_RATE should be in [0, 1], got {self.optimization.de.crossover_rate}")

        if self.optimization.algorithm == 'DDS' and self.optimization.dds:
            if not (0 < self.optimization.dds.r <= 1):
                errors.append(f"DDS_R should be in (0, 1], got {self.optimization.dds.r}")

        if self.optimization.algorithm == 'PSO' and self.optimization.pso:
            if self.optimization.pso.cognitive_param < 0:
                errors.append(f"PSO_COGNITIVE_PARAM should be >= 0, got {self.optimization.pso.cognitive_param}")
            if self.optimization.pso.social_param < 0:
                errors.append(f"PSO_SOCIAL_PARAM should be >= 0, got {self.optimization.pso.social_param}")
            if not (0 <= self.optimization.pso.inertia_weight <= 1):
                errors.append(f"PSO_INERTIA_WEIGHT should be in [0, 1], got {self.optimization.pso.inertia_weight}")

        if self.optimization.algorithm in ['NSGA-II', 'NSGA2'] and self.optimization.nsga2:
            if not (0 <= self.optimization.nsga2.crossover_rate <= 1):
                errors.append(f"NSGA2_CROSSOVER_RATE should be in [0, 1], got {self.optimization.nsga2.crossover_rate}")
            if not (0 <= self.optimization.nsga2.mutation_rate <= 1):
                errors.append(f"NSGA2_MUTATION_RATE should be in [0, 1], got {self.optimization.nsga2.mutation_rate}")

        if errors:
            raise ConfigurationError(
                "Optimization configuration invalid:\n"
                + "\n".join(f"  • {error}" for error in errors)
            )

        return self

    # ========================================
    # HELPER METHODS (from original model)
    # ========================================

    def _parse_period(self, period_str: str):
        """Parse period string 'YYYY-MM-DD, YYYY-MM-DD' into start/end dates"""
        from symfluence.core.modeling.utilities.periods import parse_period
        return parse_period(period_str, strict=True)

    def _parse_models(self):
        """Parse HYDROLOGICAL_MODEL into list of model names"""
        if isinstance(self.model.hydrological_model, str):
            return [m.strip().upper() for m in self.model.hydrological_model.split(',')]
        return [str(m).upper() for m in self.model.hydrological_model]

    # ========================================
    # BACKWARD COMPATIBILITY LAYER
    # ========================================

    @cached_property
    def _flattened_dict_cache(self) -> Dict[str, Any]:
        """
        Cached flattened dictionary for performance optimization.

        Since the config is frozen (immutable), we can safely cache this.
        This significantly improves performance of get() and __getitem__() methods.
        """
        from symfluence.core.config.flattening import flatten_nested_config
        return flatten_nested_config(self)

    def to_dict(self, flatten: bool = True) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Args:
            flatten: If True, returns flat dict with uppercase keys (legacy format)
                    If False, returns nested dict structure

        Returns:
            Configuration as dictionary

        Example:
            >>> config = SymfluenceConfig.from_preset('fuse-basic')
            >>> flat_dict = config.to_dict(flatten=True)
            >>> flat_dict['DOMAIN_NAME']
            'my_basin'
        """
        if not flatten:
            # Return nested structure
            return self.model_dump(by_alias=False)

        # Use cached flattened dict for performance (config is immutable)
        return self._flattened_dict_cache

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dict-like get method for backward compatibility.

        Supports both flat keys ('DOMAIN_NAME') and dotted paths ('domain.name').

        Args:
            key: Configuration key (uppercase) or dotted path
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('DOMAIN_NAME')
            'my_basin'
            >>> config.get('NONEXISTENT', 'fallback')
            'fallback'
        """
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def __getitem__(self, key: str) -> Any:
        """
        Dict-like bracket access for backward compatibility.

        Args:
            key: Configuration key (uppercase)

        Returns:
            Configuration value

        Raises:
            KeyError: If key not found

        Example:
            >>> config['DOMAIN_NAME']
            'my_basin'
        """
        # Use cached flat dict for performance (config is immutable)
        if key in self._flattened_dict_cache:
            return self._flattened_dict_cache[key]

        raise KeyError(f"Configuration key not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __getattr__(self, name: str) -> Any:
        """Provide attribute-style access for legacy flat keys."""
        from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP

        path = FLAT_TO_NESTED_MAP.get(name)
        if path:
            value = self
            for part in path:
                value = getattr(value, part)
            return value

        raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    # ========================================
    # FACTORY METHODS
    # ========================================

    @classmethod
    def from_file(
        cls,
        path: Path,
        overrides: Optional[Dict[str, Any]] = None,
        *,
        use_env: bool = True,
        validate: bool = True
    ) -> 'SymfluenceConfig':
        """
        Load configuration from YAML file with full 5-layer hierarchy.

        Loading precedence (highest to lowest):
        1. CLI overrides (programmatic)
        2. Environment variables (SYMFLUENCE_*)
        3. Config file (YAML)
        4. Defaults from nested models

        Args:
            path: Path to configuration YAML file
            overrides: Dictionary of CLI/programmatic overrides
            use_env: Whether to load environment variables (default: True)
            validate: Whether to validate using Pydantic (default: True)

        Returns:
            Validated SymfluenceConfig instance

        Raises:
            ConfigurationError: If configuration is invalid
            FileNotFoundError: If config file is missing

        Example:
            >>> config = SymfluenceConfig.from_file(
            ...     'config.yaml',
            ...     overrides={'DEBUG_MODE': True}
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_file_factory
        return from_file_factory(cls, path, overrides, use_env=use_env, validate=validate)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'SymfluenceConfig':
        """
        Create configuration from a named preset.

        Args:
            preset_name: Name of preset ('fuse-provo', 'summa-basic', etc.)
            **overrides: Additional overrides to apply on top of preset

        Returns:
            Fully validated SymfluenceConfig instance

        Example:
            >>> config = SymfluenceConfig.from_preset(
            ...     'fuse-provo',
            ...     DOMAIN_NAME='my_basin',
            ...     EXPERIMENT_TIME_START='2020-01-01 00:00'
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_preset_factory
        return from_preset_factory(cls, preset_name, **overrides)

    @classmethod
    def from_minimal(
        cls,
        domain_name: str,
        model: str,
        forcing_dataset: str = 'ERA5',
        **overrides
    ) -> 'SymfluenceConfig':
        """
        Create minimal viable configuration for quick setup.

        Automatically applies sensible defaults based on model choice.

        Args:
            domain_name: Name for the domain/basin
            model: Hydrological model ('SUMMA', 'FUSE', 'GR', etc.)
            forcing_dataset: Forcing data source (default: 'ERA5')
            **overrides: Additional configuration overrides

        Returns:
            Validated SymfluenceConfig with minimal required fields

        Example:
            >>> config = SymfluenceConfig.from_minimal(
            ...     domain_name='test_basin',
            ...     model='SUMMA',
            ...     POUR_POINT_COORDS='51.17/-115.57',
            ...     EXPERIMENT_TIME_START='2020-01-01 00:00',
            ...     EXPERIMENT_TIME_END='2020-12-31 23:00'
            ... )
        """
        # Import here to avoid circular dependency
        from symfluence.core.config.factories import from_minimal_factory
        return from_minimal_factory(cls, domain_name, model, forcing_dataset, **overrides)
