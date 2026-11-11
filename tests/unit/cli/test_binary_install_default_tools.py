# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""`symfluence binary install`: what the help says is what gets installed.

The help used to advertise 26 "default" tools from a hand-written list while
the installer built the 12 registered without ``optional`` -- and the list
named a tool that was not registered at all and omitted registered ones. The
lists are now derived from the build-instruction registry the installer uses,
so they cannot drift. Which tools are registered depends on the installed
model packages, so these tests compare against the registry rather than a
fixed set of names.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from symfluence.cli import argument_parser
from symfluence.cli.argument_parser import (
    CLIParser,
    get_default_tools,
    get_optional_tools,
)
from symfluence.cli.external_tools_config import get_external_tools_definitions

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _install_help() -> str:
    parser = CLIParser().parser
    for action in parser._actions:
        if getattr(action, 'choices', None) and 'binary' in action.choices:
            for sub in action.choices['binary']._actions:
                if getattr(sub, 'choices', None) and 'install' in sub.choices:
                    return sub.choices['install'].format_help()
    raise AssertionError("binary install parser not found")


def test_lists_partition_the_registry():
    registry = get_external_tools_definitions()
    default, optional = get_default_tools(), get_optional_tools()
    assert set(default) | set(optional) == set(registry)
    assert not set(default) & set(optional)
    assert all(not registry[t].get('optional', False) for t in default)
    assert all(registry[t].get('optional', False) for t in optional)


def test_bare_install_installs_exactly_the_advertised_default(tmp_path):
    """The installer's own selection for `specific_tools=None` equals the
    list the help prints."""
    from symfluence.cli.services.tool_installer import ToolInstaller

    installer = ToolInstaller()
    installer._load_config = MagicMock(return_value={"SYMFLUENCE_DATA_DIR": str(tmp_path)})
    result = installer.install(specific_tools=None, dry_run=True)

    planned = {name.removesuffix(" (dry run)") for name in result["successful"]}
    planned |= set(result["skipped"])
    assert planned == set(get_default_tools())
    assert not (tmp_path / "installs").exists()


def test_help_lists_every_registered_tool_in_its_tier():
    help_text = " ".join(_install_help().split())
    default, optional = get_default_tools(), get_optional_tools()
    assert f"Default ({len(default)} tools)" in help_text
    assert ", ".join(default) in help_text
    assert f"Optional ({len(optional)} tools, installed only when named)" in help_text
    assert ", ".join(optional) in help_text


def test_legacy_constants_are_registry_derived():
    assert argument_parser.DEFAULT_TOOLS == get_default_tools()
    assert argument_parser.EXPERIMENTAL_TOOLS == get_optional_tools()
    assert set(argument_parser.EXTERNAL_TOOLS) == set(get_external_tools_definitions())


def test_default_includes_framework_dependencies():
    """The glue binaries SYMFLUENCE relies on (DEM analysis, data tooling,
    SUMMA's solver) are infrastructure tools registered by the core package,
    so they are always present and always in the default set."""
    assert {'sundials', 'taudem', 'gistool', 'datatool'} <= set(get_default_tools())


def test_help_text_names_install_path_and_jax_models():
    """The -h output must tell the user where binaries land and
    which paper models don't need a binary build (JAX re-impls,
    LSTM, GR4J) — co-authors spent time trying to 'install' LSTM
    because the help didn't mention it's Python-only."""
    install_help = _install_help()
    assert 'SYMFLUENCE_DATA_DIR/installs' in install_help
    assert 'JAX' in install_help
    assert 'LSTM' in install_help
    assert 'paper-repro' in install_help
