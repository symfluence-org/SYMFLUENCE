# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regression test for the `symfluence binary install` default tier.

Co-author PC (02_model_ensemble) reported "Some binaries are
missing" when running the paper's multi-model benchmark — the
default `symfluence binary install` compiled only a subset of the
engines used in the paper's Fig 4/Fig 8 ensemble, so several models
never produced output.

Pin the fix: every *process-based* model that appears in the paper
ensemble (Fig 4 row, Fig 8 table) must be in DEFAULT_TOOLS so a
fresh install reproduces the paper. JAX re-implementations come via
pyproject.toml deps and are NOT binaries; LSTM is PyTorch-only;
GR4J uses airGR via rpy2. Those are called out in the help text but
not in this list.
"""

import pytest

from symfluence.cli.argument_parser import DEFAULT_TOOLS, EXPERIMENTAL_TOOLS

pytestmark = [pytest.mark.unit, pytest.mark.quick]


# The binary-installable models that appear in Fig 4 / Fig 8 of
# the paper. Order doesn't matter; membership does.
PAPER_ENSEMBLE_BINARIES = {
    'summa', 'fuse', 'hype', 'mesh', 'ngen', 'ngiab',   # merged into default in earlier PRs
    'clm', 'clmparflow', 'crhm', 'gsflow', 'mhm', 'parflow',  # promoted
    'pihm', 'prms', 'rhessys', 'swat', 'vic', 'watflood',
    'wflow', 'wrfhydro',
}


def test_default_tools_cover_paper_ensemble():
    """Every process-based engine from the paper's multi-model
    ensemble must be in the default install set so a reviewer
    reproducing the paper doesn't need to discover and enable
    models one-by-one."""
    default = set(DEFAULT_TOOLS)
    missing = PAPER_ENSEMBLE_BINARIES - default
    assert not missing, (
        f"paper-ensemble binaries missing from DEFAULT_TOOLS: "
        f"{sorted(missing)}. Add them or justify removal."
    )


def test_default_includes_framework_dependencies():
    """The glue binaries SYMFLUENCE relies on to operate any of the
    above (routing, DEM analysis, data tooling) must also be in
    default."""
    required = {'sundials', 'mizuroute', 'troute', 'taudem',
                'gistool', 'datatool'}
    assert required.issubset(set(DEFAULT_TOOLS))


def test_default_and_experimental_are_disjoint():
    """Prevent silent double-counting between tiers."""
    assert not (set(DEFAULT_TOOLS) & set(EXPERIMENTAL_TOOLS))


def test_help_text_names_install_path_and_jax_models():
    """The -h output must tell the user where binaries land and
    which paper models don't need a binary build (JAX re-impls,
    LSTM, GR4J) — co-authors spent time trying to 'install' LSTM
    because the help didn't mention it's Python-only."""
    from symfluence.cli.argument_parser import CLIParser

    parser = CLIParser().parser
    help_text = parser.format_help()

    # Drill into binary → install sub-parser for its own help.
    for action in parser._actions:
        if getattr(action, 'choices', None) and 'binary' in action.choices:
            bin_p = action.choices['binary']
            for sub in bin_p._actions:
                if getattr(sub, 'choices', None) and 'install' in sub.choices:
                    help_text = sub.choices['install'].format_help()
                    break

    assert 'SYMFLUENCE_DATA_DIR/installs' in help_text, \
        "help must name the install path"
    assert 'JAX' in help_text or 'jax' in help_text, \
        "help must name JAX re-implementations"
    assert 'LSTM' in help_text, "help must clarify LSTM is not a binary"
    assert 'GR4J' in help_text, "help must clarify GR4J uses airGR/rpy2"
