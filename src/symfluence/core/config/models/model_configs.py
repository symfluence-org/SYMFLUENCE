# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Model configuration entrypoint and compatibility exports.

This module preserves the historical import path
``symfluence.core.config.models.model_configs`` while delegating concrete
model-specific class definitions to focused modules.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

from symfluence.core.registries import R

from .base import FROZEN_CONFIG
from .model_config_types import SpatialModeType

# Per-model config schemas live with their model packages
# (``symfluence.models.<model>.config_schema``) and register into
# ``R.config_schemas`` through each package's manifest/``register()`` — the
# same path an external plugin uses (``model_manifest(config_schema=...)``,
# e.g. the droute package). Legacy class-name imports from this module keep
# working via the registry-backed ``__getattr__`` below.
_LEGACY_SCHEMA_EXPORTS: Dict[str, str] = {
    'SUMMAConfig': 'SUMMA', 'FUSEConfig': 'FUSE', 'GRConfig': 'GR',
    'HYPEConfig': 'HYPE', 'NGENConfig': 'NGEN', 'MESHConfig': 'MESH',
    'RHESSysConfig': 'RHESSYS', 'VICConfig': 'VIC', 'CLMConfig': 'CLM',
    'SWATConfig': 'SWAT', 'MHMConfig': 'MHM', 'CRHMConfig': 'CRHM',
    'WRFHydroConfig': 'WRFHYDRO', 'PRMSConfig': 'PRMS',
    'GSFLOWConfig': 'GSFLOW', 'WATFLOODConfig': 'WATFLOOD',
    'WflowConfig': 'WFLOW', 'LisfloodConfig': 'LISFLOOD',
    'PCRGLOBWBConfig': 'PCRGLOBWB', 'CWatMConfig': 'CWATM',
    'NoahMPConfig': 'NOAHMP', 'MODFLOWConfig': 'MODFLOW',
    'ParFlowConfig': 'PARFLOW', 'CLMParFlowConfig': 'CLMPARFLOW',
    'PIHMConfig': 'PIHM', 'MizuRouteConfig': 'MIZUROUTE',
    'TRouteConfig': 'TROUTE', 'LSTMConfig': 'LSTM', 'GNNConfig': 'GNN',
    'IGNACIOConfig': 'IGNACIO', 'WMFireConfig': 'WMFIRE',
}


def __getattr__(name: str):
    key = _LEGACY_SCHEMA_EXPORTS.get(name)
    if key is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    schema = R.config_schemas.get(key)
    if schema is None:
        raise AttributeError(
            f"{name} is unavailable: no config schema registered for model "
            f"'{key}' (is the model package installed and registered?)"
        )
    globals()[name] = schema
    return schema

# Selection keys (alias, field-name) whose value names a single model whose
# typed config should be auto-created, in addition to the hydrological list.
_SINGLE_SELECT_KEYS = (
    ('ROUTING_MODEL', 'routing_model'),
    ('GROUNDWATER_MODEL', 'groundwater_model'),
    ('FIRE_MODEL', 'fire_model'),
)

# Deprecated NGEN flags that may arrive at the ModelConfig level in flat configs.
_NGEN_ENABLE_KEYS = ('ENABLE_SLOTH', 'ENABLE_PET', 'ENABLE_NOAH', 'ENABLE_CFE')


class ModelConfig(BaseModel):
    """Hydrological model configuration.

    Model-specific typed configs are no longer declared as ~32 hardcoded
    ``Optional[*Config]`` fields. They live in :attr:`model_specific`, a dict
    keyed by the lower-case canonical model name, populated from the
    ``R.config_schemas`` registry at validation time. This lets an external
    plugin ship its own typed config (via ``model_manifest(config_schema=...)``)
    and have it validated into the config tree without editing this core module
    (RTI review item 18).

    Backward compatibility:

    * ``config.model.summa`` etc. keep working via :meth:`__getattr__`, which
      resolves against ``model_specific`` and returns ``None`` for any
      registered-but-unselected model (preserving the old ``Optional = None``
      access semantics).
    * Serialization hoists ``model_specific`` back to the top level via a
      ``@model_serializer`` so the config flattener and stage-marker hashing
      keep seeing the historical ``{'summa': {...}}`` shape.
    """
    model_config = FROZEN_CONFIG

    # Required model selection
    hydrological_model: Union[str, List[str]] = Field(alias='HYDROLOGICAL_MODEL')
    routing_model: Optional[str] = Field(default=None, alias='ROUTING_MODEL')
    groundwater_model: Optional[str] = Field(default=None, alias='GROUNDWATER_MODEL')
    fire_model: Optional[str] = Field(default=None, alias='FIRE_MODEL')
    # Coupling settings
    coupling_mode: str = Field(default='sequential', alias='COUPLING_MODE')
    conservation_mode: Optional[str] = Field(default=None, alias='CONSERVATION_MODE')
    topology_file: Optional[str] = Field(default=None, alias='TOPOLOGY_FILE')
    snow_module: str = Field(default='', alias='SNOW_MODULE')

    # Model-specific typed configs, keyed by lower-case canonical model name.
    # Typed ``Any`` (not ``BaseModel``) so already-validated subclass instances
    # are stored and serialized verbatim rather than coerced to the base class.
    model_specific: Dict[str, Any] = Field(default_factory=dict)

    def __getattr__(self, item: str) -> Any:
        # Only consulted when normal attribute lookup fails. Resolve
        # ``config.model.<model>`` against ``model_specific``, falling back to
        # ``None`` for any registered model schema that was simply not selected
        # (mirrors the historical ``Optional[*Config] = None`` fields).
        if not item.startswith('_'):
            model_specific = self.__dict__.get('model_specific')
            if isinstance(model_specific, dict) and item in model_specific:
                return model_specific[item]
            if item.upper() in R.config_schemas:
                return None
        # Defer to pydantic's BaseModel.__getattr__ (handles extras / private
        # attrs and raises AttributeError for genuinely unknown names).
        return super().__getattr__(item)  # type: ignore[misc]

    @field_validator('hydrological_model')
    @classmethod
    def validate_hydrological_model(cls, v):
        """Normalize to an upper-case comma-separated string and validate.

        Model names are registry keys (upper case): a lower-case ``summa``
        used to bypass case-sensitive checks downstream (e.g. it silently
        skipped mizuRoute). Every name must be registered in ``R.runners``
        or ``R.config_schemas`` (core or plugin, resolved lazily — no model
        module is imported); unknown names fail with a did-you-mean suggestion.
        """
        from .choices import MODEL_NAME_HINTS, did_you_mean

        items = v if isinstance(v, list) else str(v).split(',')
        names = [str(i).strip().upper() for i in items if str(i).strip()]
        if not names:
            raise ValueError("HYDROLOGICAL_MODEL must name at least one model")

        known = R.runners.keys()
        if known:  # an unseeded/empty registry (stripped test env) can't judge
            for name in names:
                if name in MODEL_NAME_HINTS:
                    raise ValueError(
                        f"HYDROLOGICAL_MODEL: '{name}' is not a model; {MODEL_NAME_HINTS[name]}"
                    )
                if name not in R.runners and name not in R.config_schemas:
                    raise ValueError(
                        f"HYDROLOGICAL_MODEL: unknown model '{name}'{did_you_mean(name, known)}. "
                        f"Registered models: {', '.join(known)} "
                        "(plugin models must be installed to be recognized)"
                    )
        return ",".join(names)

    @field_validator('routing_model')
    @classmethod
    def validate_routing_model(cls, v):
        """ROUTING_MODEL must be 'none' or a registered (routing) runner."""
        if v is None:
            return v
        from .choices import ROUTING_NONE_VALUES, ROUTING_SPELLINGS, did_you_mean

        known = R.runners.keys()
        if not known:
            return v
        for item in str(v).split(','):
            name = item.strip().upper()
            if name in ROUTING_NONE_VALUES or name in ROUTING_SPELLINGS or name in R.runners:
                continue
            raise ValueError(
                f"ROUTING_MODEL: unknown routing model '{item.strip()}'"
                f"{did_you_mean(name, list(known) + ['NONE'])}. "
                "Use 'none' or a registered model (e.g. mizuRoute, dRoute, troute)"
            )
        return v

    @field_validator('coupling_mode')
    @classmethod
    def validate_coupling_mode(cls, v):
        from .choices import COUPLING_MODES, unknown_value_message

        mode = str(v).strip().lower()
        if mode not in COUPLING_MODES:
            raise ValueError(unknown_value_message('COUPLING_MODE', str(v), COUPLING_MODES))
        return mode

    @model_validator(mode='before')
    @classmethod
    def auto_populate_model_configs(cls, values):
        """Resolve and validate model-specific configs from ``R.config_schemas``.

        Runs before field validation. Moves any incoming model-config payload
        (e.g. a ``summa`` sub-dict from the flat->nested transform, or a
        pre-built config instance) into ``model_specific``, and auto-creates a
        default config for every selected model that has a registered schema.
        """
        if not isinstance(values, dict):
            return values

        # ---- Determine selected model names (canonical UPPER) -------------
        selected: list[str] = []
        hydrological_model = values.get('HYDROLOGICAL_MODEL') or values.get('hydrological_model')
        if hydrological_model:
            if isinstance(hydrological_model, list):
                selected.extend(str(m).strip().upper() for m in hydrological_model)
            else:
                selected.extend(m.strip().upper() for m in str(hydrological_model).split(','))
        for alias_key, field_key in _SINGLE_SELECT_KEYS:
            sel = values.get(alias_key) or values.get(field_key)
            if sel:
                selected.append(str(sel).strip().upper())

        # ---- Forward deprecated ENABLE_* keys into the ngen payload -------
        found = {k: values[k] for k in _NGEN_ENABLE_KEYS if k in values}
        if found:
            ngen_payload = values.get('ngen')
            if ngen_payload is None:
                ngen_payload = {}
                values['ngen'] = ngen_payload
            if isinstance(ngen_payload, dict):
                for k, v in found.items():
                    ngen_payload.setdefault(k, v)
            for k in found:
                values.pop(k, None)

        # ---- Build model_specific ----------------------------------------
        def _coerce(name: str, payload: Any) -> Optional[Any]:
            """Validate *payload* into the registered schema for *name*."""
            if payload is None:
                return None
            if isinstance(payload, BaseModel):
                return payload
            schema_cls = R.config_schemas.get(name.upper())
            if schema_cls is None:
                return None
            if isinstance(payload, dict):
                return schema_cls.model_validate(payload)
            return payload

        model_specific: Dict[str, Any] = {}

        # Carry over any pre-populated ``model_specific`` (e.g. round-trip dump).
        existing = values.get('model_specific')
        if isinstance(existing, dict):
            for key, payload in existing.items():
                coerced = _coerce(key, payload)
                if coerced is not None:
                    model_specific[key.lower()] = coerced

        # Move any top-level model-config payloads into ``model_specific``.
        for key in list(values.keys()):
            if not isinstance(key, str) or key == 'model_specific':
                continue
            if key.upper() in R.config_schemas:
                coerced = _coerce(key, values.pop(key))
                if coerced is not None:
                    model_specific[key.lower()] = coerced

        # Auto-create defaults for selected models with a registered schema.
        for name in selected:
            schema_cls = R.config_schemas.get(name)
            if schema_cls is None:
                continue
            attr = name.lower()
            if attr not in model_specific:
                model_specific[attr] = schema_cls()

        values['model_specific'] = model_specific
        return values

    @model_serializer(mode='wrap')
    def _serialize_model(self, handler, _info):
        """Hoist ``model_specific`` back to the top level on dump.

        Preserves the historical flat shape ``{'summa': {...}, ...}`` so the
        config flattener and stage-marker hashing keep producing correct flat
        keys without needing to know about ``model_specific``.
        """
        data = handler(self)
        if isinstance(data, dict):
            nested = data.pop('model_specific', None)
            if isinstance(nested, dict):
                for key, value in nested.items():
                    data.setdefault(key, value)
        return data

__all__ = ['SpatialModeType', 'ModelConfig', *sorted(_LEGACY_SCHEMA_EXPORTS)]
