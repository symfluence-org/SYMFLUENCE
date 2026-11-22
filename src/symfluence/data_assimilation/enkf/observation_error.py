# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Streamflow observation-error model for the EnKF.

Discharge errors grow with discharge (rating-curve extrapolation, stage
error times the slope of the rating), so a single scalar R is too confident
in floods and too vague at low flow. The default model is heteroscedastic,

    sigma(Q_obs) = a + b * Q_obs,      R = diag(sigma**2)

with ``b`` a relative error (``ENKF_OBS_ERROR_STD``, default 0.1, i.e. 10 %)
and ``a`` an absolute floor (``ENKF_OBS_ERROR_ABS`` in m3/s; default 1 % of
the mean observed discharge) that keeps zero or near-zero flows from being
trusted absolutely. Because the conversion from discharge to the model's
runoff depth is linear, ``sigma`` is computed in model units by converting
``a`` once. Several gauges give a vector observation with a diagonal R
(errors assumed independent between gauges).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# Default absolute floor ``a`` as a fraction of the mean observed flow.
DEFAULT_ABS_FLOOR_FRACTION = 0.01

# Legacy 'relative' model: floor = obs_error_std * this * mean observed flow.
_RELATIVE_ERROR_FLOOR_FRACTION = 0.1


@dataclass
class StreamflowObservationError:
    """Observation-error standard deviation as a function of the observation.

    Attributes:
        kind: 'heteroscedastic' (a + b*Q), 'relative' (max(b*Q, a)) or
            'absolute' (a constant ``b``).
        a: Absolute term (model units), per gauge or scalar.
        b: Relative term (fraction), or the constant std for 'absolute'.
    """

    kind: str
    a: Union[float, np.ndarray]
    b: float

    def std(self, y_obs: np.ndarray, gauges: Optional[np.ndarray] = None) -> np.ndarray:
        """Error std for each observation in ``y_obs`` (model units).

        Args:
            y_obs: Observations (n_obs,).
            gauges: Gauge index of each observation, to pick per-gauge ``a``.
        """
        y = np.abs(np.atleast_1d(np.asarray(y_obs, dtype=float)))
        a = np.asarray(self.a, dtype=float)
        if a.ndim > 0:
            a = a[gauges] if gauges is not None else a[:len(y)]
        if self.kind == 'heteroscedastic':
            sigma = a + self.b * y
        elif self.kind == 'relative':
            sigma = np.maximum(self.b * y, a)
        elif self.kind == 'absolute':
            sigma = np.full(y.shape, float(self.b))
        else:
            raise ValueError(f"Unknown observation error type '{self.kind}'")
        return np.maximum(sigma, 1e-9)

    def covariance(self, y_obs: np.ndarray, gauges: Optional[np.ndarray] = None) -> np.ndarray:
        """Diagonal observation-error covariance R (n_obs, n_obs)."""
        return np.diag(self.std(y_obs, gauges) ** 2)

    @classmethod
    def from_config(
        cls,
        enkf_config,
        obs_model_units: np.ndarray,
        cms_to_model: Optional[Union[float, np.ndarray]] = None,
    ) -> 'StreamflowObservationError':
        """Build the error model from EnKF settings and the observation record.

        Args:
            enkf_config: :class:`EnKFConfig`.
            obs_model_units: Observations on the model time axis, (n_t,) or
                (n_t, n_gauges), in model units (NaN = missing).
            cms_to_model: Factor converting m3/s into model units (per gauge
                or scalar); needed only when ``ENKF_OBS_ERROR_ABS`` is set.
        """
        kind = enkf_config.obs_error_type
        b = float(enkf_config.obs_error_std)
        obs = np.asarray(obs_model_units, dtype=float)
        if obs.ndim == 1:
            obs = obs[:, None]
        with np.errstate(all='ignore'):
            mean_abs = np.nanmean(np.abs(obs), axis=0) if obs.size else np.zeros(obs.shape[1])
        mean_abs = np.where(np.isfinite(mean_abs), mean_abs, 0.0)

        if kind == 'heteroscedastic':
            abs_cms = getattr(enkf_config, 'obs_error_abs', None)
            if abs_cms is not None:
                if cms_to_model is None:
                    raise ValueError(
                        "ENKF_OBS_ERROR_ABS is in m3/s; a discharge-to-model-units "
                        "factor (catchment area and timestep) is needed to use it.")
                a = float(abs_cms) * np.asarray(cms_to_model, dtype=float) * np.ones(obs.shape[1])
            else:
                a = DEFAULT_ABS_FLOOR_FRACTION * mean_abs
        elif kind == 'relative':
            a = b * _RELATIVE_ERROR_FLOOR_FRACTION * mean_abs
        else:
            a = np.zeros(obs.shape[1])
        a = np.maximum(a, 1e-9)
        return cls(kind=kind, a=a if a.size > 1 else float(a[0]), b=b)
