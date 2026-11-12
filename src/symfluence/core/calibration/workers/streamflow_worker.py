# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Worker base for models scored on outlet streamflow.

Every process-based worker used to carry its own copy of the same scoring
tail: resolve the project directory, load observations, pick the scoring
period, align, compute the requested metrics, and turn any failure into a
penalty. Only the first step, reading the simulated series out of the model's
own output format, differs between models.

``StreamflowWorker`` owns that tail. A model worker implements
:meth:`StreamflowWorker.read_simulated_streamflow` and, when it scores more
than discharge, :meth:`StreamflowWorker.additional_metrics`.
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from symfluence.core.exceptions import ModelExecutionError
from symfluence.core.metrics.streamflow_metrics import StreamflowMetrics
from symfluence.core.mixins.config import resolve_config_value
from symfluence.core.modeling.utilities.periods import split_period

from .base_worker import BaseWorker

__all__ = ['StreamflowWorker']


class StreamflowWorker(BaseWorker):
    """``BaseWorker`` whose objective is outlet discharge against observations.

    Subclass contract:

    - :meth:`read_simulated_streamflow` returns the simulated discharge in
      m3/s on a ``DatetimeIndex`` at the resolution it should be scored at
      (observations are resampled to :attr:`OBSERVATION_FREQUENCY`). Warm-up
      trimming that is specific to the model belongs there too. Return
      ``None`` or an empty series when there is nothing to score, or raise
      :class:`ModelExecutionError` to report a specific reason.
    - :meth:`additional_metrics` (optional) returns further score components,
      e.g. snow-cover or glacier mass-balance constraints.

    A trial that cannot be scored returns
    ``{'kge': penalty_score, 'error': <reason>}``; it never raises.
    """

    _streamflow_metrics = StreamflowMetrics()

    #: Frequency observations are resampled to before alignment.
    OBSERVATION_FREQUENCY: Optional[str] = 'D'

    @abstractmethod
    def read_simulated_streamflow(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs,
    ) -> Optional[pd.Series]:
        """Simulated outlet discharge (m3/s, datetime-indexed) for this trial.

        ``kwargs`` carries what :meth:`BaseWorker.evaluate` passes to
        ``calculate_metrics`` (``sim_dir``, ``settings_dir``, ``proc_id``, …).
        """

    def additional_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        *,
        project_dir: Path,
        **kwargs,
    ) -> Dict[str, Any]:
        """Score components beyond the streamflow metrics (default: none).

        Called only after the streamflow metrics were computed. Raising
        fails the trial with the exception message as the reason.
        """
        return {}

    @property
    def _model_label(self) -> str:
        name = type(self).__name__
        return name[:-len('Worker')] if name.endswith('Worker') else name

    @staticmethod
    def _project_location(config: Dict[str, Any]) -> Tuple[Path, Optional[str]]:
        """``(project_dir, domain_name)`` for a task config."""
        domain_name = resolve_config_value(config, lambda: config.domain.name, None, 'DOMAIN_NAME')
        data_dir = Path(resolve_config_value(config, lambda: config.system.data_dir, '.', 'SYMFLUENCE_DATA_DIR'))
        return data_dir / f'domain_{domain_name}', domain_name

    def _failed(self, reason: str) -> Dict[str, Any]:
        return {'kge': self.penalty_score, 'error': reason}

    def calculate_metrics(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Score the trial's simulated discharge against observations.

        The scoring window is ``kwargs['period']`` when given (a
        ``'start, end'`` string, used to score the evaluation period from a
        full-period run), else ``CALIBRATION_PERIOD``; with neither, the
        whole overlap is scored.
        """
        try:
            sim_series = self.read_simulated_streamflow(output_dir, config, **kwargs)
            if sim_series is None or len(sim_series) == 0:
                return self._failed('No streamflow data')

            project_dir, domain_name = self._project_location(config)
            obs_values, obs_index = self._streamflow_metrics.load_observations(
                config, project_dir, domain_name, resample_freq=self.OBSERVATION_FREQUENCY
            )
            if obs_values is None:
                return self._failed('No observations')
            obs_series = pd.Series(obs_values, index=obs_index)

            period = kwargs.get('period') or resolve_config_value(
                config, lambda: config.domain.calibration_period, '', 'CALIBRATION_PERIOD')
            obs_aligned, sim_aligned = self._streamflow_metrics.align_timeseries(
                sim_series, obs_series,
                calibration_period=split_period(period, allow_extra=True),
            )
            results: Dict[str, Any] = self._streamflow_metrics.calculate_metrics(
                obs_aligned, sim_aligned, metrics=self._requested_metrics(config)
            )
            results.update(self.additional_metrics(
                output_dir, config, project_dir=project_dir, **kwargs))
            return results

        except ModelExecutionError as e:
            # A reader's own verdict on the output (missing, invalid): the
            # message is the diagnosis, a traceback adds nothing.
            self.logger.error(f"{self._model_label}: {e}")
            return self._failed(str(e))
        except Exception as e:  # noqa: BLE001 — calibration resilience
            self.logger.error(f"Error calculating {self._model_label} metrics: {e}", exc_info=True)
            return self._failed(str(e))
