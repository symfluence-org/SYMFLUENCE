"""Constraint handling must guide search without weakening acceptance."""
from __future__ import annotations

import pytest

from symfluence.models.pihm.calibration.search import search_rank, select_incumbents


def trial(mb=1.9, snow=.29, objective=0., water=0.):
    return dict(cryosphere=dict(snow_cover_rmse=snow, glacier_mb_rmse_mwe=mb),
                checks=dict(water_residual_fraction=water), objective=objective)


def test_infeasible_trials_supply_search_incumbent_only():
    old, improved = trial(), trial(mb=1.8, objective=-2.)
    assert select_incumbents([old, improved]) == (improved, None)


def test_feasibility_precedes_objective_and_violation():
    rejected, accepted = trial(objective=100), trial(mb=1.5, snow=.35, objective=-100)
    assert select_incumbents([rejected, accepted]) == (accepted, accepted)


def test_objective_ranks_feasible_trials():
    low, high = trial(mb=1., objective=-1), trial(mb=1., objective=1)
    assert select_incumbents([low, high]) == (high, high)


@pytest.mark.parametrize('result', [dict(error='failed'), {}, trial(water=.011),
                                     trial(objective=float('nan')), trial(mb=-1)])
def test_invalid_trials_cannot_guide_search(result):
    assert search_rank(result) is None
    assert select_incumbents([result]) == (None, None)


def test_untrusted_feasible_flag_cannot_bypass_constraints():
    result = trial()
    result['feasible'] = True
    assert select_incumbents([result]) == (result, None)
