# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Canonical flat-to-nested configuration key mappings.

The primary mapping is now auto-generated from Pydantic model aliases via
``introspection.generate_flat_to_nested_map``.  This module provides a
lazy-loaded fallback that uses the same introspection machinery plus legacy
aliases from ``legacy_aliases.py``.

The exported names (``CANONICAL_FLAT_TO_NESTED_MAP`` and ``FLAT_TO_NESTED_MAP``)
are kept for backward compatibility with any code that imports them directly.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Tuple

from symfluence.core.config.legacy_aliases import iter_legacy_flat_to_nested_aliases

logger = logging.getLogger(__name__)

_CANONICAL_MAP: Dict[str, Tuple[str, ...]] | None = None
_CANONICAL_LOCK = threading.Lock()


def _generate_canonical_map() -> Dict[str, Tuple[str, ...]]:
    """Lazy-generate the canonical mapping from Pydantic model aliases."""
    global _CANONICAL_MAP

    if _CANONICAL_MAP is not None:
        return _CANONICAL_MAP

    with _CANONICAL_LOCK:
        if _CANONICAL_MAP is not None:
            return _CANONICAL_MAP

        try:
            from symfluence.core.config.introspection import generate_flat_to_nested_map
            from symfluence.core.config.models import SymfluenceConfig

            _CANONICAL_MAP = generate_flat_to_nested_map(
                SymfluenceConfig,
                include_model_overrides=False,
            )
            logger.debug(
                "Auto-generated %d canonical mappings from Pydantic aliases",
                len(_CANONICAL_MAP),
            )
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            logger.warning("Canonical map auto-generation failed: %s", e)
            _CANONICAL_MAP = {}

    return _CANONICAL_MAP


class _LazyMap(dict):
    """Dict subclass that lazily populates from the auto-generated mapping.

    This lets ``CANONICAL_FLAT_TO_NESTED_MAP`` and ``FLAT_TO_NESTED_MAP``
    remain importable module-level dicts without triggering Pydantic model
    import at module load time (which would cause circular imports).
    """

    def __init__(self, include_legacy: bool = False):
        super().__init__()
        self._include_legacy = include_legacy
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Populate the dict from auto-generated mappings on first access."""
        if self._loaded:
            return
        self._loaded = True
        self.update(_generate_canonical_map())
        if self._include_legacy:
            # Legacy aliases have lower priority — only add missing keys
            # (framework entries + per-schema declarations from R.config_schemas)
            for key, path in iter_legacy_flat_to_nested_aliases().items():
                self.setdefault(key, path)

    # Override all read methods to ensure data is loaded before access.
    def __getitem__(self, key):
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self):
        self._ensure_loaded()
        return super().__len__()

    def keys(self):
        """Return mapping keys, triggering lazy load if needed."""
        self._ensure_loaded()
        return super().keys()

    def values(self):
        """Return mapping values, triggering lazy load if needed."""
        self._ensure_loaded()
        return super().values()

    def items(self):
        """Return mapping items, triggering lazy load if needed."""
        self._ensure_loaded()
        return super().items()

    def get(self, key, default=None):
        """Return value for *key*, triggering lazy load if needed."""
        self._ensure_loaded()
        return super().get(key, default)

    def copy(self):
        """Return a plain dict copy of the fully loaded mapping."""
        self._ensure_loaded()
        return dict(self)

    def __repr__(self):
        self._ensure_loaded()
        return super().__repr__()


# Canonical mapping (no legacy aliases) — importable as before.
CANONICAL_FLAT_TO_NESTED_MAP: Dict[str, Tuple[str, ...]] = _LazyMap(include_legacy=False)

# Full mapping (canonical + legacy aliases) — importable as before.
FLAT_TO_NESTED_MAP: Dict[str, Tuple[str, ...]] = _LazyMap(include_legacy=True)
