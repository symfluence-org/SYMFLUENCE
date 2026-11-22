# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Factory methods for creating SYMFLUENCE configurations.

This module provides factory functions for creating SymfluenceConfig instances
from various sources:
- from_file_factory: Load from YAML file with 5-layer hierarchy
- from_preset_factory: Load from named preset
- from_minimal_factory: Create minimal configuration with smart defaults

Each factory handles the complexity of merging defaults, loading from sources,
and transforming to the hierarchical structure required by Pydantic models.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import yaml

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


def _resolve_default_code_dir() -> str:
    """Resolve the default SYMFLUENCE_CODE_DIR.

    Priority:
    1. SYMFLUENCE_CODE_DIR environment variable
    2. Repository root (detected from package location)
    3. Current working directory
    """
    env_val = os.getenv('SYMFLUENCE_CODE_DIR')
    if env_val:
        return env_val

    # Try to find repo root from package location
    try:
        import symfluence
        pkg_dir = Path(symfluence.__file__).parent  # src/symfluence/
        # Walk up to find repo root (contains pyproject.toml or .git)
        candidate = pkg_dir
        for _ in range(5):
            candidate = candidate.parent
            if (candidate / 'pyproject.toml').exists() or (candidate / '.git').exists():
                return str(candidate)
    except (ImportError, AttributeError, OSError, TypeError):
        pass

    cwd = Path.cwd()
    # Guard against running from Jupyter checkpoint directory
    if cwd.name == '.ipynb_checkpoints':
        cwd = cwd.parent
    return str(cwd)


def _resolve_default_data_dir(code_dir: Optional[str] = None) -> str:
    """Resolve the default SYMFLUENCE_DATA_DIR.

    Priority:
    1. SYMFLUENCE_DATA_DIR environment variable
    2. SYMFLUENCE_DATA environment variable (CI/workflow compat)
    3. Sibling directory to CODE_DIR named SYMFLUENCE_data
    4. Current working directory / data
    """
    env_val = os.getenv('SYMFLUENCE_DATA_DIR') or os.getenv('SYMFLUENCE_DATA')
    if env_val:
        return env_val

    # Use sibling directory to code_dir
    if code_dir is None:
        code_dir = _resolve_default_code_dir()
    # Resolve relative code_dir values (e.g., ".") so parent lookup does not
    # accidentally collapse to cwd and create ./SYMFLUENCE_data.
    code_path = Path(code_dir).expanduser().resolve()
    # Guard against Jupyter checkpoint directories in code_dir
    parts = code_path.parts
    clean = [p for p in parts if p != '.ipynb_checkpoints']
    if len(clean) != len(parts):
        code_path = Path(*clean)
    return str(code_path.parent / 'SYMFLUENCE_data')


def _is_nested_config(config: Dict[str, Any]) -> bool:
    """
    Detect if a configuration dictionary is in nested format.

    Nested format has lowercase section keys like 'system', 'domain', 'forcing', 'model'.
    Flat format has uppercase keys like 'DOMAIN_NAME', 'FORCING_DATASET'.

    Args:
        config: Configuration dictionary loaded from YAML

    Returns:
        True if config appears to be in nested format
    """
    nested_section_keys = {'system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths', 'data'}
    config_keys_lower = {k.lower() for k in config.keys()}
    # If any nested section keys are present, treat as nested config
    return bool(nested_section_keys & config_keys_lower)


def _normalize_nested_config(nested_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a nested configuration to ensure section keys are lowercase.

    This handles cases where section keys might be uppercase (SYSTEM vs system).

    Args:
        nested_config: Nested configuration dictionary

    Returns:
        Normalized nested configuration with lowercase section keys
    """
    section_keys = {'system', 'domain', 'forcing', 'model', 'optimization', 'evaluation', 'paths', 'data'}
    normalized = {}

    for key, value in nested_config.items():
        key_lower = key.lower()
        if key_lower in section_keys:
            # Ensure section key is lowercase
            normalized[key_lower] = value
        else:
            # Keep other keys as-is
            normalized[key] = value

    # A top-level ``lapse`` section is shorthand for ``forcing.lapse`` (the
    # elevation corrections of remapped forcing). Explicit ``forcing.lapse``
    # entries win over the shorthand.
    lapse = normalized.get('lapse')
    if isinstance(lapse, dict):
        del normalized['lapse']
        forcing = dict(normalized.get('forcing') or {})
        forcing['lapse'] = {**lapse, **(forcing.get('lapse') or {})}
        normalized['forcing'] = forcing

    return normalized


def _resolve_nested_path_sentinels(nested_config: Dict[str, Any]) -> None:
    """Resolve ``default`` sentinels (and missing values) for the required
    system paths in a nested config, mutating ``nested_config`` in place.

    The flat-config branch treats a literal ``default`` for the data/code dir as
    "use the computed default" and drops it so Pydantic falls back. The nested
    branch has no such handling, so ``system.data_dir: default`` reaches Pydantic
    verbatim and resolves as the relative path ``./default``. Mirror the flat
    behavior here. Code dir is resolved first because the data-dir default is
    derived from it. Both the field name (``data_dir``) and the flat alias
    (``SYMFLUENCE_DATA_DIR``) spellings are honored.
    """
    system = nested_config.get('system')
    if not isinstance(system, dict):
        system = {}
        nested_config['system'] = system

    def _resolve(name_key: str, alias_key: str, compute) -> str:
        key = alias_key if alias_key in system else name_key
        value = system.get(key)
        if value is None or value == 'default':
            resolved = compute()
            # Prefer the field-name spelling so the value is unambiguous.
            system.pop(alias_key, None)
            system[name_key] = resolved
            return resolved
        return value

    code_dir = _resolve('code_dir', 'SYMFLUENCE_CODE_DIR', _resolve_default_code_dir)
    _resolve('data_dir', 'SYMFLUENCE_DATA_DIR', lambda: _resolve_default_data_dir(str(code_dir)))


def from_file_factory(
    cls: type,
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
    4. Defaults from nested Pydantic models

    Supports both flat format (uppercase keys like DOMAIN_NAME) and
    nested format (hierarchical structure like domain.name).

    Args:
        cls: SymfluenceConfig class
        path: Path to configuration YAML file
        overrides: Dictionary of CLI/programmatic overrides
        use_env: Whether to load environment variables (default: True)
        validate: Whether to validate using Pydantic (default: True)

    Returns:
        Validated SymfluenceConfig instance

    Raises:
        ConfigurationError: If configuration is invalid
        FileNotFoundError: If config file is missing
    """
    from pydantic import ValidationError

    from symfluence.core.config.config_loader import _format_validation_error, _load_env_overrides, _normalize_key
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError

    # 1. Load from file first to detect format
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        file_config = yaml.safe_load(f) or {}

    # Archived configs (``_workLog_*/config_*.yaml``) carry a ``_metadata``
    # block written by LoggingManager; it is provenance, not configuration.
    if isinstance(file_config, dict):
        file_config = {
            k: v for k, v in file_config.items()
            if not (isinstance(k, str) and k.lower() == '_metadata')
        }

    # 2. Detect if config is in nested or flat format
    is_nested = _is_nested_config(file_config)

    if is_nested:
        # Handle nested config format - don't transform, just normalize section keys
        nested_config = _normalize_nested_config(file_config)

        # Resolve `default`/missing system paths at the file+defaults layer, so
        # env/CLI overrides applied below still take precedence.
        _resolve_nested_path_sentinels(nested_config)

        # Apply environment variable overrides (converted to nested format)
        if use_env:
            model_section = nested_config.get('model') if isinstance(nested_config.get('model'), dict) else {}
            env_overrides = _load_env_overrides(model_section.get('hydrological_model'))
            if env_overrides:
                # Transform env overrides to nested and merge
                env_nested = transform_flat_to_nested(env_overrides)
                nested_config = _deep_merge(nested_config, env_nested)

        # Apply CLI overrides (can be flat or nested)
        if overrides:
            if _is_nested_config(overrides):
                nested_config = _deep_merge(nested_config, _normalize_nested_config(overrides))
            else:
                # Flat overrides - transform and merge
                normalized_overrides = {_normalize_key(k): v for k, v in overrides.items()}
                override_nested = transform_flat_to_nested(normalized_overrides)
                nested_config = _deep_merge(nested_config, override_nested)

        # Unknown-key validation for nested configs (warn; raise in strict
        # mode) — previously only the flat branch was checked, so misplaced
        # nested keys were silently ignored.
        from symfluence.core.config.key_validation import validate_known_nested_keys
        validate_known_nested_keys(nested_config, source=str(path))
    else:
        # Handle flat config format — Pydantic Field defaults are the single
        # source of truth; no pre-seeding with a separate defaults dict.
        config_dict = {}

        # Add sensible defaults for required system paths
        if 'SYMFLUENCE_CODE_DIR' not in config_dict:
            config_dict['SYMFLUENCE_CODE_DIR'] = _resolve_default_code_dir()
        if 'SYMFLUENCE_DATA_DIR' not in config_dict:
            config_dict['SYMFLUENCE_DATA_DIR'] = _resolve_default_data_dir(config_dict.get('SYMFLUENCE_CODE_DIR'))

        # Normalize keys from file config
        file_config = {_normalize_key(k): v for k, v in file_config.items()}

        # Treat 'default' as sentinel: use computed defaults instead
        for path_key in ('SYMFLUENCE_DATA_DIR', 'SYMFLUENCE_CODE_DIR'):
            if file_config.get(path_key) == 'default':
                del file_config[path_key]

        config_dict.update(file_config)

        # Override with environment variables
        if use_env:
            hydrological_model = config_dict.get('HYDROLOGICAL_MODEL')
            if isinstance(hydrological_model, list):
                hydrological_model = ','.join(str(m) for m in hydrological_model)
            env_overrides = _load_env_overrides(hydrological_model)
            config_dict.update(env_overrides)

        # Apply CLI overrides (highest priority)
        if overrides:
            normalized_overrides = {_normalize_key(k): v for k, v in overrides.items()}
            config_dict.update(normalized_overrides)

        # The ``lapse`` section (elevation corrections of remapped forcing)
        # is a mapping with no flat spelling; carry it past the flat-key
        # machinery into ``forcing.lapse``.
        lapse_section = config_dict.pop('LAPSE', None)

        # Tiered validation of unrecognized keys (RTI review item 21):
        # warn by default, raise under STRICT_CONFIG / SYMFLUENCE_STRICT_CONFIG.
        # Runs on the fully-resolved flat user config before transformation, so
        # the allowlist already includes any plugin-registered schema keys.
        from symfluence.core.config.key_validation import validate_known_flat_keys
        from symfluence.core.config.legacy_aliases import RECOGNIZED_FLAT_KEYS
        from symfluence.core.config.transformers import build_combined_flat_to_nested_map
        # The allowlist is the flat->nested transform map (core + plugin schemas +
        # legacy aliases for the selected model) unioned with the recognized flat
        # keys that are read directly from _extra rather than a nested field.
        known_keys = set(build_combined_flat_to_nested_map(config_dict.get('HYDROLOGICAL_MODEL')))
        known_keys |= RECOGNIZED_FLAT_KEYS
        validate_known_flat_keys(config_dict, known_keys, source=str(path))

        # Transform flat dict to nested structure
        nested_config = transform_flat_to_nested(config_dict)
        if isinstance(lapse_section, dict):
            nested_config.setdefault('forcing', {})['lapse'] = lapse_section
        elif lapse_section is not None:
            raise ConfigurationError("'lapse' must be a mapping of elevation-correction settings")

    # Filter out None values so Pydantic can use field defaults
    nested_config = _filter_none_values(nested_config)

    # Validate and create
    if validate:
        try:
            instance = cls(**nested_config)
            # Store source file path as internal attribute
            object.__setattr__(instance, '_source_file', path)
            return instance
        except ValidationError as e:
            error_msg = _format_validation_error(e, file_config if is_nested else config_dict)
            raise ConfigurationError(error_msg) from e
    else:
        instance = cls.model_construct(**nested_config)
        object.__setattr__(instance, '_source_file', path)
        return instance


def _filter_none_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively filter out None values from a nested dictionary.

    This allows Pydantic to use field defaults when config explicitly sets null.

    Args:
        d: Dictionary potentially containing None values

    Returns:
        New dictionary with None values removed at all levels
    """
    result = {}
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, dict):
            filtered = _filter_none_values(value)
            if filtered:  # Only include non-empty dicts
                result[key] = filtered
        else:
            result[key] = value
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    For nested dictionaries, recursively merge. For other values, override wins.

    Args:
        base: Base dictionary
        override: Override dictionary (values take precedence)

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def from_preset_factory(
    cls: type,
    preset_name: str,
    **overrides
) -> 'SymfluenceConfig':
    """
    Create configuration from a named preset.

    Args:
        cls: SymfluenceConfig class
        preset_name: Name of preset ('fuse-provo', 'summa-basic', etc.)
        **overrides: Additional overrides to apply on top of preset

    Returns:
        Fully validated SymfluenceConfig instance

    Raises:
        ConfigurationError: If preset not found or configuration invalid
    """
    from pydantic import ValidationError

    from symfluence.cli.init_presets import get_preset
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError

    # 1. Load preset definition
    try:
        preset = get_preset(preset_name)
    except (KeyError, ValueError):
        raise ConfigurationError(
            f"Preset '{preset_name}' not found. "
            f"Use 'symfluence project list-presets' to see available presets."
        ) from None

    preset_settings = preset['settings'].copy()

    # 2. Apply model-specific decisions
    if 'fuse_decisions' in preset:
        preset_settings['FUSE_DECISION_OPTIONS'] = preset['fuse_decisions']
    if 'summa_decisions' in preset:
        preset_settings['SUMMA_DECISION_OPTIONS'] = preset['summa_decisions']

    # 2.5. Add sensible defaults for required system paths if not provided
    if 'SYMFLUENCE_CODE_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_CODE_DIR'] = _resolve_default_code_dir()
    if 'SYMFLUENCE_DATA_DIR' not in preset_settings:
        preset_settings['SYMFLUENCE_DATA_DIR'] = _resolve_default_data_dir(preset_settings.get('SYMFLUENCE_CODE_DIR'))

    # 3. Apply user overrides (highest priority)
    preset_settings.update(overrides)

    # 4. Transform flat dict to nested structure
    nested_config = transform_flat_to_nested(preset_settings)

    # 5. Create and validate
    try:
        return cls(**nested_config)
    except ValidationError as e:
        from symfluence.core.config.config_loader import _format_validation_error
        error_msg = _format_validation_error(e, preset_settings)
        raise ConfigurationError(
            f"Failed to create config from preset '{preset_name}':\n{error_msg}"
        ) from e


def _normalize_config_key(key: str) -> str:
    """
    Normalize configuration key to uppercase format.

    Accepts both modern lowercase/snake_case and legacy UPPERCASE formats.

    Args:
        key: Configuration key in any format (time_start, EXPERIMENT_TIME_START, etc.)

    Returns:
        Normalized uppercase key (EXPERIMENT_TIME_START)

    Examples:
        >>> _normalize_config_key('time_start')
        'EXPERIMENT_TIME_START'
        >>> _normalize_config_key('EXPERIMENT_TIME_START')
        'EXPERIMENT_TIME_START'
        >>> _normalize_config_key('routing_model')
        'ROUTING_MODEL'
    """
    from symfluence.core.config.transformers import get_flat_to_nested_map

    # If already uppercase, return as-is (legacy format)
    if key.isupper() or key == key.upper():
        return key

    # Build reverse mapping: nested path -> uppercase key
    # This maps ('domain', 'time_start') -> 'EXPERIMENT_TIME_START'
    flat_to_nested = get_flat_to_nested_map()
    nested_to_flat = {path: flat_key for flat_key, path in flat_to_nested.items()}

    # Common patterns for new lowercase keys
    # Try to find matching nested path based on the key name
    lowercase_key = key.lower()

    # Direct field name matches (e.g., 'time_start' in ('domain', 'time_start'))
    for nested_path, flat_key in nested_to_flat.items():
        if nested_path[-1] == lowercase_key:
            return flat_key

    # Reverse lookup: build field_name → flat_key from the auto-generated map
    # This handles abbreviated forms like 'iterations' → 'NUMBER_OF_ITERATIONS'
    field_to_flat: Dict[str, str] = {}
    for flat_key, nested_path in flat_to_nested.items():
        last = nested_path[-1]
        # First match wins — canonical keys are iterated before legacy aliases
        if last not in field_to_flat:
            field_to_flat[last] = flat_key

    if lowercase_key in field_to_flat:
        return field_to_flat[lowercase_key]

    # Fallback: return uppercase version of the key
    # This handles simple cases like 'DEBUG_MODE' or 'debug_mode' -> 'DEBUG_MODE'
    return key.upper()


def from_minimal_factory(
    cls: type,
    domain_name: str,
    model: str,
    forcing_dataset: str = 'ERA5',
    **overrides
) -> 'SymfluenceConfig':
    """
    Create minimal viable configuration for quick setup.

    Automatically applies sensible defaults based on model choice.

    Supports both modern lowercase/snake_case keys and legacy UPPERCASE keys:
    - Modern: time_start='2020-01-01 00:00', routing_model='mizuRoute'
    - Legacy: EXPERIMENT_TIME_START='2020-01-01 00:00', ROUTING_MODEL='mizuRoute'

    Args:
        cls: SymfluenceConfig class
        domain_name: Name for the domain/basin
        model: Hydrological model ('SUMMA', 'FUSE', 'GR', etc.)
        forcing_dataset: Forcing data source (default: 'ERA5')
        **overrides: Additional configuration overrides (accepts both formats)

    Returns:
        Validated SymfluenceConfig with minimal required fields

    Raises:
        ConfigurationError: If required fields missing or configuration invalid

    Examples:
        >>> # Modern syntax
        >>> config = SymfluenceConfig.from_minimal(
        ...     domain_name='test',
        ...     model='SUMMA',
        ...     time_start='2020-01-01 00:00',
        ...     time_end='2020-12-31 23:00',
        ...     routing_model='mizuRoute'
        ... )
        >>>
        >>> # Legacy syntax (still supported)
        >>> config = SymfluenceConfig.from_minimal(
        ...     domain_name='test',
        ...     model='SUMMA',
        ...     EXPERIMENT_TIME_START='2020-01-01 00:00',
        ...     EXPERIMENT_TIME_END='2020-12-31 23:00',
        ...     ROUTING_MODEL='mizuRoute'
        ... )
    """
    from pydantic import ValidationError

    from symfluence.core.config.defaults import ForcingDefaults, ModelDefaults
    from symfluence.core.config.transformers import transform_flat_to_nested
    from symfluence.core.exceptions import ConfigurationError

    # 0. Normalize all override keys to uppercase format
    normalized_overrides = {_normalize_config_key(k): v for k, v in overrides.items()}

    # 1. Start with absolute minimal required fields
    minimal = {
        'DOMAIN_NAME': domain_name,
        'EXPERIMENT_ID': 'run_1',
        'HYDROLOGICAL_MODEL': model,
        'FORCING_DATASET': forcing_dataset,

        # Required paths (from environment or defaults)
        'SYMFLUENCE_CODE_DIR': normalized_overrides.get(
            'SYMFLUENCE_CODE_DIR',
            _resolve_default_code_dir()
        ),
        'SYMFLUENCE_DATA_DIR': normalized_overrides.get(
            'SYMFLUENCE_DATA_DIR',
            _resolve_default_data_dir(normalized_overrides.get('SYMFLUENCE_CODE_DIR'))
        ),

        # Required domain settings (user should override, but provide safe defaults)
        'DOMAIN_DEFINITION_METHOD': normalized_overrides.get('DOMAIN_DEFINITION_METHOD', 'lumped'),
        'SUB_GRID_DISCRETIZATION': normalized_overrides.get('SUB_GRID_DISCRETIZATION', 'grus'),

        # Required time settings (user MUST override these)
        'EXPERIMENT_TIME_START': normalized_overrides.get('EXPERIMENT_TIME_START', '2010-01-01 00:00'),
        'EXPERIMENT_TIME_END': normalized_overrides.get('EXPERIMENT_TIME_END', '2020-12-31 23:00'),
    }

    # 2. Apply model-specific defaults
    model_defaults = ModelDefaults.get_defaults_for_model(model.upper())
    if model_defaults:
        minimal.update(model_defaults)

    # 3. Apply forcing-specific defaults
    forcing_defaults = ForcingDefaults.get_defaults_for_forcing(forcing_dataset.upper())
    if forcing_defaults:
        minimal.update(forcing_defaults)

    # 4. Apply user overrides (highest priority)
    minimal.update(normalized_overrides)

    # 5. Validate required overrides (check both original and normalized keys)
    required_overrides = ['EXPERIMENT_TIME_START', 'EXPERIMENT_TIME_END']
    missing = []
    for field in required_overrides:
        if field not in normalized_overrides:
            # Check if they provided placeholder values
            if minimal[field] in ['2010-01-01 00:00', '2020-12-31 23:00']:
                missing.append(field)

    if missing:
        raise ConfigurationError(
            f"Missing required fields for minimal config: {', '.join(missing)}\n\n"
            f"Example (modern syntax):\n"
            f"  config = SymfluenceConfig.from_minimal(\n"
            f"      domain_name='{domain_name}',\n"
            f"      model='{model}',\n"
            f"      time_start='2020-01-01 00:00',\n"
            f"      time_end='2020-12-31 23:00',\n"
            f"      pour_point_coords='51.17/-115.57'  # Optional but recommended\n"
            f"  )\n\n"
            f"Example (legacy syntax):\n"
            f"  config = SymfluenceConfig.from_minimal(\n"
            f"      domain_name='{domain_name}',\n"
            f"      model='{model}',\n"
            f"      EXPERIMENT_TIME_START='2020-01-01 00:00',\n"
            f"      EXPERIMENT_TIME_END='2020-12-31 23:00',\n"
            f"      POUR_POINT_COORDS='51.17/-115.57'  # Optional but recommended\n"
            f"  )"
        )

    # 6. Transform and create
    nested_config = transform_flat_to_nested(minimal)

    try:
        return cls(**nested_config)
    except ValidationError as e:
        from symfluence.core.config.config_loader import _format_validation_error
        error_msg = _format_validation_error(e, minimal)
        raise ConfigurationError(
            f"Failed to create minimal config:\n{error_msg}"
        ) from e
