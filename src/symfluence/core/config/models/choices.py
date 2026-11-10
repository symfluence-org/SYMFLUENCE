# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Known values for enum-like config fields, with did-you-mean helpers.

Closed sets (the consuming code dispatches on a fixed table) raise on an
unknown value; extensible ones (plugin registries, analyses) are checked
against the live registry, or only warn.
"""
from __future__ import annotations

from difflib import get_close_matches
from typing import Iterable

# OptimizationManager.run_optimization_workflow dispatches on these.
OPTIMIZATION_METHODS = frozenset({'iteration', 'data_assimilation', 'emulation'})

# DomainDiscretizer.discretize method_map (single) / combined attributes.
DISCRETIZATION_METHODS = frozenset({
    'grus', 'elevation', 'aspect', 'soilclass', 'landclass', 'radiation',
})
# Also accepted (not listed in suggestions): 'lumped'/'gru'/'point'/'none'
# used by presets and point/lumped domains, and the legacy spatial-mode tokens
# SymfluenceConfig._migrate_legacy_spatial_mode understands.
DISCRETIZATION_ACCEPTED = DISCRETIZATION_METHODS | frozenset({
    'lumped', 'gru', 'point', 'none',
    'semi_distributed', 'semi-distributed', 'semidistributed', 'distributed',
})

# ModelManager.run_models: 'dcoupler' opts into graph execution; the others
# are documented spellings that run sequentially.
COUPLING_MODES = frozenset({'sequential', 'dcoupler', 'auto', 'per_timestep'})

# Workflow analysis steps keyed on ANALYSES, plus the documented 'attributes'.
KNOWN_ANALYSES = frozenset({'benchmarking', 'decision', 'sensitivity', 'attributes', 'koopman'})

# ROUTING_MODEL values that mean "no routing" or are routing-code spellings
# not registered as runner keys.
ROUTING_NONE_VALUES = frozenset({'', 'NONE', 'DEFAULT', 'N/A', 'NULL', 'FALSE'})
ROUTING_SPELLINGS = frozenset({'MIZU_ROUTE', 'MIZU', 'T-ROUTE', 'T_ROUTE'})

# Spellings that are not models; point the user at the real configuration.
MODEL_NAME_HINTS = {
    'SUMMA-MODFLOW': "set HYDROLOGICAL_MODEL: SUMMA and GROUNDWATER_MODEL: MODFLOW",
    'COUPLED_GW': "set HYDROLOGICAL_MODEL to the land model and GROUNDWATER_MODEL: MODFLOW",
}


def did_you_mean(value: str, choices: Iterable[str]) -> str:
    """`` (did you mean 'x'?)`` suffix, or empty string."""
    matches = get_close_matches(str(value), [str(c) for c in choices], n=1, cutoff=0.6)
    return f" (did you mean '{matches[0]}'?)" if matches else ""


def unknown_value_message(field: str, value: str, choices: Iterable[str]) -> str:
    choices = sorted(set(choices))
    return (
        f"{field}: unknown value '{value}'{did_you_mean(value, choices)}. "
        f"Valid values: {', '.join(choices)}"
    )
