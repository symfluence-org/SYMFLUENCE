# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Resource loading utilities for SYMFLUENCE package data.

This package holds bundled *data* (config templates, base-settings fixtures,
``system_deps.yml``, packaged agent/skill markdown) and the accessors that
locate its own files. It must not import ``symfluence.core``: the model-name to
settings-directory resolution that used to live here reached into
``core.registries`` while core's preprocessor base imported it straight back —
a genuine dependency cycle between the two packages.

``get_base_settings_dir`` / ``copy_base_settings_to_project`` therefore now live
in :mod:`symfluence.core.modeling.base_settings`. Both names stay importable
here as deprecated shims for external packages (removal at 2.0). Unlike the
other names in this module they are resolved *only* inside ``__getattr__``, and
through a module-path string rather than an ``import`` statement — a
module-level import would re-create the cycle this move removed, and keeping
the package free of ``symfluence.core`` imports entirely makes "no cycle" a
mechanically checkable property (see
``tests/unit/core/test_resources_boundary.py``).
"""
from __future__ import annotations

from .manager import (
    agent_cache_root,
    copy_config_template_to_project,
    get_agents_dir,
    get_bundled_base_settings_dir,
    get_config_template,
    get_skills_dir,
    get_system_deps_registry_path,
    list_config_templates,
    parse_frontmatter,
    prepare_agent_context,
)

#: Names promoted to the core contract tier, kept importable here as shims.
_PROMOTED_TO_CORE = ('get_base_settings_dir', 'copy_base_settings_to_project')

__all__ = [
    'agent_cache_root',
    'get_agents_dir',
    'parse_frontmatter',
    'get_base_settings_dir',
    'get_bundled_base_settings_dir',
    'get_config_template',
    'get_skills_dir',
    'get_system_deps_registry_path',
    'list_config_templates',
    'prepare_agent_context',
    'copy_base_settings_to_project',
    'copy_config_template_to_project',
]


def __getattr__(name: str):
    if name in _PROMOTED_TO_CORE:
        # Resolved at CALL time, never at module level. That distinction is the
        # whole invariant: a module-level import here would re-create the
        # import-time cycle this promotion broke, whereas a deprecated shim
        # reaching its target when someone actually touches it cannot. This is
        # deliberately a plain import rather than importlib-on-a-string — the
        # string form was only ever hiding this edge from the grep instead of
        # stating it.
        from symfluence.core.modeling import base_settings

        return getattr(base_settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
