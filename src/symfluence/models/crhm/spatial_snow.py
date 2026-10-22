"""Conservative quadrature for independent CRHM snow tiles.

Tile factors describe unresolved solid-precipitation deposition, not rainfall or
an empirical conversion from SWE to fractional cover. Each tile must carry its
own native snow mass, temperature, albedo, melt and soil/ice surface states.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri

from symfluence.core.exceptions import ValidationError


def snow_tile_quadrature(coefficient_of_variation: float, count: int = 5):
    """Return equal-area weights and conditional-mean lognormal snow factors.

    The parent-unit mean snowfall is preserved exactly. CV is an unresolved
    deposition assumption requiring spatial evidence; zero gives a homogeneous
    control. Conditional means integrate the distribution rather than sampling
    its midpoint, which would lose mass in its high-accumulation tail.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValidationError('Snow tile count must be a positive integer')
    cv = float(coefficient_of_variation)
    if not np.isfinite(cv) or cv < 0:
        raise ValidationError('Snow deposition CV must be finite and nonnegative')
    weights = np.full(count, 1.0 / count)
    if cv == 0:
        return weights, np.ones(count)
    sigma = np.sqrt(2 * np.log(np.hypot(1, cv)))
    boundaries = ndtri(np.linspace(0, 1, count + 1))
    factors = np.diff(ndtr(boundaries - sigma)) / weights
    factors /= weights @ factors
    return weights, factors


def split_solid_precipitation(snowfall, weights, factors):
    """Reference calculation for the native per-tile snowfall partition.

    Parent rainfall is copied unscaled to every tile. Native audit precipitation
    is rainfall plus partitioned snowfall, preserving the parent water input.
    """
    snow = np.asarray(snowfall, dtype=float)
    weights = np.asarray(weights, dtype=float)
    factors = np.asarray(factors, dtype=float)
    if weights.ndim != 1 or weights.shape != factors.shape or not len(weights):
        raise ValidationError('Tile weights/factors must be matching nonempty vectors')
    if (not np.isfinite(weights).all() or not np.isfinite(factors).all()
            or (weights <= 0).any() or (factors < 0).any()
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
            or not np.isclose(weights @ factors, 1, rtol=0, atol=1e-12)):
        raise ValidationError('Snow tile quadrature must conserve parent snowfall')
    if not np.isfinite(snow).all() or (snow < 0).any():
        raise ValidationError('Snowfall must be finite and nonnegative')
    return snow[..., None] * factors
