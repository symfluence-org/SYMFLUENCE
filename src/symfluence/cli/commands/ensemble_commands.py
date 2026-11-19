# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""``symfluence ensemble``: combine and compare multi-model members."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Dict

from ..exit_codes import ExitCode
from .base import BaseCommand, cli_exception_handler


def _parse_members(specs) -> Dict[str, Path]:
    members: Dict[str, Path] = {}
    for spec in specs or []:
        if "=" not in spec:
            raise ValueError(f"--member expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        members[name.strip()] = Path(path.strip())
    return members


class EnsembleCommands(BaseCommand):
    """Handler for the top-level ``ensemble`` command."""

    @staticmethod
    @cli_exception_handler
    def ensemble(args: Namespace) -> int:
        """Combine member discharge series and score members + combinations."""
        from symfluence.evaluation.multimodel_ensemble import read_series, run_ensemble

        console = BaseCommand._console
        output = Path(args.output)
        common = dict(
            output_dir=output,
            weighting=args.weighting,
            median=args.median,
            resample=args.resample,
            config_path=Path(args.period_config) if args.period_config else None,
        )
        if args.manifest:
            result = run_ensemble(manifest=Path(args.manifest), **common)
        else:
            members = _parse_members(args.member)
            if len(members) < 2 or not args.obs:
                console.error("Give --manifest, or at least two --member NAME=PATH and --obs PATH")
                return ExitCode.USAGE_ERROR
            periods = {k: getattr(args, f"{k}_period") for k in ("calibration", "evaluation", "test")}
            result = run_ensemble(
                members={n: read_series(p) for n, p in members.items()},
                observed=read_series(Path(args.obs), column=args.obs_column),
                periods={k: v for k, v in periods.items() if v} or None,
                **common,
            )
        console.info(result.scores.round(3).to_string())
        console.info("Weights: " + ", ".join(
            f"{n}={w:.3f}" for n, w in result.weights["skill_weighted"].items()))
        console.success(f"Ensemble comparison written to {output}")
        return ExitCode.SUCCESS
