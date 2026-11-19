# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Return value of :meth:`SYMFLUENCE.run_workflow` / :meth:`SYMFLUENCE.run_individual_steps`.

Example:
    >>> result = SYMFLUENCE("config.yaml").run()
    >>> result.ok
    True
    >>> result.metrics["models"]["SUMMA"]["metrics"]["evaluation"]["KGE"]
    0.71
    >>> result.outputs["metrics_json"]
    PosixPath('.../domain_X/results/run_1/metrics.json')
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

__all__ = ['RunResult', 'StepResult']


@dataclass(frozen=True)
class StepResult:
    """Outcome of one workflow step in this run."""

    name: str
    cli_name: str
    status: str  # 'completed' | 'skipped' | 'failed'
    duration_s: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> 'StepResult':
        """Build from an orchestrator ``last_step_results`` entry."""
        name = str(record.get('name') or record.get('cli_name') or '')
        return cls(
            name=name,
            cli_name=str(record.get('cli_name') or name),
            status=str(record.get('status') or 'unknown'),
            duration_s=float(record.get('duration_s') or 0.0),
            error=record.get('error') or None,
        )


@dataclass
class RunResult:
    """What a run did and where its outputs are.

    Attributes:
        status: ``'completed'`` (no step failed), ``'partial'`` (some steps
            failed, others completed) or ``'failed'``.
        steps: Per-step outcomes in execution order.
        duration_s: Wall-clock duration of the call.
        domain_name / experiment_id / project_dir: What was run.
        outputs: Named output paths that exist after the run: ``results_csv``,
            ``metrics_json``, ``metrics_csv``, ``summary_figure``,
            ``run_manifest``, ``log_dir``.
        metrics: Contents of ``results/<experiment_id>/metrics.json``
            (see :mod:`symfluence.core.metrics.run_metrics`), or None when no
            run has scored this experiment yet.
    """

    status: str
    steps: List[StepResult] = field(default_factory=list)
    duration_s: float = 0.0
    domain_name: str = ''
    experiment_id: str = ''
    project_dir: Optional[Path] = None
    outputs: Dict[str, Path] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        """True when no step failed."""
        return self.status == 'completed'

    @property
    def failed_steps(self) -> List[str]:
        """CLI names of the failed steps."""
        return [s.cli_name for s in self.steps if s.status == 'failed']
