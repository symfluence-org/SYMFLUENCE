"""Feasibility-first selection for expensive PIHM calibration trials."""

import math


def search_rank(result):
    """Return an ascending preference key, or None for an unusable trial.

    Conserved but cryosphere-infeasible runs can guide proposals. They can
    never outrank an accepted candidate. Failed or incomplete runs cannot guide
    the search at all.
    """
    if result.get('error'):
        return None
    try:
        snow = float(result['cryosphere']['snow_cover_rmse'])
        glacier = float(result['cryosphere']['glacier_mb_rmse_mwe'])
        water = float(result['checks']['water_residual_fraction'])
        objective = float(result['objective'])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(x) for x in (snow, glacier, water, objective)):
        return None
    if min(snow, glacier) < 0 or abs(water) > .01:
        return None
    violation = max(snow / .35 - 1, 0) + max(glacier / 1.5 - 1, 0)
    return (int(violation > 0), violation, -objective)


def select_incumbents(results):
    """Return (search incumbent, accepted best), keeping acceptance separate."""
    usable = [(search_rank(r), r) for r in results]
    usable = [(key, r) for key, r in usable if key is not None]
    incumbent = min(usable, key=lambda item: item[0])[1] if usable else None
    accepted = [(key, r) for key, r in usable if key[0] == 0]
    best = min(accepted, key=lambda item: item[0])[1] if accepted else None
    return incumbent, best
