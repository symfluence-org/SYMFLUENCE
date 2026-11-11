# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""`symfluence list` — introspect the live registry and config schema.

Discovery is registry-driven: every catalog below is read from `R.*`,
`ALGORITHM_REGISTRY`, or the auto-generated config map, so it never goes stale.
Used by humans and by the `symfluence agent` skills to see what is available.
"""
from __future__ import annotations

from argparse import Namespace

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler, supports_dry_run

# Catalog kinds, in display order. Values are the CLI tokens.
LIST_KINDS = [
    'models', 'forcings', 'observations', 'optimizers', 'targets',
    'metrics', 'presets', 'templates', 'steps', 'config-keys',
]


def _catalog() -> dict[str, list[str]]:
    """Build the live catalog from the registry + config schema (lazy imports)."""
    from symfluence.core.calibration.optimizers.algorithms import list_algorithms
    from symfluence.core.config.canonical_mappings import FLAT_TO_NESTED_MAP
    from symfluence.core.registries import R
    from symfluence.resources import list_config_templates
    from symfluence.workflow_steps import WORKFLOW_STEP_ITEMS

    # Presets register as a side effect of importing each package's
    # ``init_preset`` module, which packages declare rather than the framework
    # discovering on disk. A plain ``R.presets.keys()`` therefore reported zero
    # presets even though ``symfluence init --preset fuse-basic`` worked --
    # listing is exactly the consumer that must drain the declarations first.
    R.presets.load_modules()

    return {
        'models': sorted(R.runners.keys()),
        'forcings': sorted(R.acquisition_handlers.keys()),
        'observations': sorted(R.observation_handlers.keys()),
        'optimizers': list_algorithms(),
        'targets': sorted(R.calibration_targets.keys()),
        'metrics': sorted(R.metrics.keys()),
        'presets': sorted(R.presets.keys()),
        'templates': sorted(p.name for p in list_config_templates()),
        'steps': [name for name, _ in WORKFLOW_STEP_ITEMS],
        'config-keys': sorted(FLAT_TO_NESTED_MAP.keys()),
    }


class ListCommands(BaseCommand):
    """Handler for `symfluence list [KIND]`."""

    @staticmethod
    @supports_dry_run
    @cli_exception_handler
    def list_items(args: Namespace) -> int:
        """
        Execute: symfluence list [KIND]

        With no KIND, print every catalog and its count. With a KIND, print the
        registered entries for it.
        """
        kind = BaseCommand.get_arg(args, 'kind', None)
        catalog = _catalog()

        if not kind:
            BaseCommand._console.info("Available catalogs (run `symfluence list <kind>`):")
            for name in LIST_KINDS:
                BaseCommand._console.indent(f"{name:<13} {len(catalog[name])}")
            return ExitCode.SUCCESS

        if kind not in catalog:
            BaseCommand._console.error(
                f"Unknown kind '{kind}'. Choose from: {', '.join(LIST_KINDS)}"
            )
            return ExitCode.USAGE_ERROR

        items = catalog[kind]
        BaseCommand._console.info(f"{kind} ({len(items)}):")
        for item in items:
            BaseCommand._console.indent(item)
        return ExitCode.SUCCESS
