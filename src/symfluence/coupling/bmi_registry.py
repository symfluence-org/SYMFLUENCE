# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Registry mapping SYMFLUENCE model names to dCoupler component adapters.

The BMIRegistry provides a single lookup table to instantiate the correct
dCoupler component adapter for any SYMFLUENCE model.

.. deprecated::
    This registry is a thin delegation shim around
    :pydata:`symfluence.core.registries.R`.  Prefer ``R.bmi_adapters``
    directly.
"""

from __future__ import annotations

import logging
import warnings
from typing import Type

from symfluence.core.registries import R

logger = logging.getLogger(__name__)


class BMIRegistry:
    """Maps SYMFLUENCE model identifiers to dCoupler component classes.

    Usage::

        registry = BMIRegistry()
        component_cls = registry.get("SUMMA")
        component = component_cls(name="summa", config=config_dict)

    .. deprecated::
        Use ``R.bmi_adapters`` from :mod:`symfluence.core.registries` instead.
    """

    # Classification metadata -- kept for is_jax_model / is_process_model queries.
    _PROCESS_MODELS = {
        "SUMMA": "symfluence.coupling.adapters.process_adapters.SUMMAProcessComponent",
        "MIZUROUTE": "symfluence.coupling.adapters.process_adapters.MizuRouteProcessComponent",
        "TROUTE": "symfluence.coupling.adapters.process_adapters.TRouteProcessComponent",
        "PARFLOW": "symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent",
        "MODFLOW": "symfluence.coupling.adapters.process_adapters.MODFLOWProcessComponent",
        "MESH": "symfluence.coupling.adapters.process_adapters.MESHProcessComponent",
        "CLM": "symfluence.coupling.adapters.process_adapters.CLMProcessComponent",
    }

    _JAX_MODELS = {
        "SNOW17": "symfluence.coupling.adapters.jax_adapters.Snow17JAXComponent",
        "XAJ": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "XINANJIANG": "symfluence.coupling.adapters.jax_adapters.XAJJAXComponent",
        "SACSMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
        "SAC-SMA": "symfluence.coupling.adapters.jax_adapters.SacSmaJAXComponent",
        "HBV": "symfluence.coupling.adapters.jax_adapters.HBVJAXComponent",
        "HECHMS": "symfluence.coupling.adapters.jax_adapters.HecHmsJAXComponent",
        "HEC-HMS": "symfluence.coupling.adapters.jax_adapters.HecHmsJAXComponent",
        "TOPMODEL": "symfluence.coupling.adapters.jax_adapters.TopmodelJAXComponent",
    }

    def __init__(self):
        # Seed R.bmi_adapters with the static classification metadata if not
        # already populated (first instantiation path).
        for name, path in {**self._PROCESS_MODELS, **self._JAX_MODELS}.items():
            if name not in R.bmi_adapters:
                R.bmi_adapters.add_lazy(name, path)

    def get(self, model_name: str) -> Type:
        """Resolve a model name to its component class.

        Args:
            model_name: Model identifier (case-insensitive), e.g. "SUMMA", "XAJ"

        Returns:
            The component class (not an instance).

        Raises:
            KeyError: If the model is not registered.
            ImportError: If the component class cannot be imported.
        """
        # R.bmi_adapters normalizes to uppercase via its default normalize.
        # Try the name as given first, then strip hyphens/spaces for compat.
        key = model_name.upper()
        result = R.bmi_adapters.get(key)
        if result is None:
            alt_key = key.replace(" ", "").replace("-", "")
            result = R.bmi_adapters.get(alt_key)
        if result is None:
            available = sorted(R.bmi_adapters.keys())
            raise KeyError(
                f"Unknown model '{model_name}'. Available: {available}"
            )
        return result

    def register(self, model_name: str, class_path: str) -> None:
        """Register a custom model adapter.

        Args:
            model_name: Model identifier (will be uppercased)
            class_path: Fully qualified class path, e.g.
                "my_package.adapters.MyComponent"

        .. deprecated::
            Use ``R.bmi_adapters.add_lazy()`` instead.
        """
        warnings.warn(
            "BMIRegistry.register() is deprecated; "
            "use R.bmi_adapters.add_lazy() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R.bmi_adapters.add_lazy(model_name, class_path)

    def is_jax_model(self, model_name: str) -> bool:
        """Check if a model uses JAX (differentiable) backend."""
        return model_name.upper() in self._JAX_MODELS

    def is_process_model(self, model_name: str) -> bool:
        """Check if a model is an external process."""
        return model_name.upper() in self._PROCESS_MODELS

    def available_models(self) -> list:
        """Return sorted list of all registered model names."""
        return sorted(R.bmi_adapters.keys())
