# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Model-ready data store — CF-1.8 compliant, self-describing data packages.

Materializes forcings, observations, and attributes into a structured
``data/model_ready/`` directory with full provenance metadata.
"""
from __future__ import annotations

from symfluence.core.modeling.model_ready.attributes_reader import AttributesReader, open_canonical_attributes
from symfluence.core.modeling.model_ready.cf_conventions import CF_STANDARD_NAMES, build_global_attrs
from symfluence.core.modeling.model_ready.path_resolver import resolve_model_ready_path

from .source_metadata import SourceMetadata
from .store_builder import ModelReadyStoreBuilder

__all__ = [
    'SourceMetadata',
    'CF_STANDARD_NAMES',
    'build_global_attrs',
    'resolve_model_ready_path',
    'ModelReadyStoreBuilder',
    'AttributesReader',
    'open_canonical_attributes',
]
