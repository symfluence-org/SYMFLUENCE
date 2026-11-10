"""Regression tests: workflow step-completion checks match what the steps write.

Before, discretize/preprocess/run/postprocess/calibrate checked paths no code
writes, so ``workflow run`` re-ran them every time and every run was recorded
as "partial".
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.stage_marker import mark_stage_complete
from symfluence.project import step_completion as sc
from symfluence.project.workflow_orchestrator import WorkflowOrchestrator

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _config(tmp_path, **extra):
    base = dict(
        domain_name='dom',
        model='SUMMA',
        time_start='2010-01-01 00:00',
        time_end='2010-12-31 23:00',
        SYMFLUENCE_DATA_DIR=tmp_path,
        EXPERIMENT_ID='exp1',
        DOMAIN_DEFINITION_METHOD='semidistributed',
        SUB_GRID_DISCRETIZATION='elevation',
    )
    base.update(extra)
    return SymfluenceConfig.from_minimal(**base)


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('x')
    return path


def test_discretization_uses_nested_experiment_layout(tmp_path):
    cfg = _config(tmp_path)
    proj = tmp_path / 'domain_dom'
    assert not sc.discretization_exists(cfg, proj)
    _touch(proj / 'shapefiles/catchment/semidistributed/exp1/dom_HRUs_elevation.shp')
    assert sc.discretization_exists(cfg, proj)


def test_discretization_suffix_case_insensitive_and_legacy_flat(tmp_path):
    cfg = _config(tmp_path, DOMAIN_DEFINITION_METHOD='lumped', SUB_GRID_DISCRETIZATION='GRUs')
    proj = tmp_path / 'domain_dom'
    _touch(proj / 'shapefiles/catchment/dom_HRUs_GRUS.shp')
    assert sc.discretization_exists(cfg, proj)


def test_discretization_other_experiment_does_not_count(tmp_path):
    cfg = _config(tmp_path)
    proj = tmp_path / 'domain_dom'
    _touch(proj / 'shapefiles/catchment/semidistributed/other/dom_HRUs_elevation.shp')
    assert not sc.discretization_exists(cfg, proj)


def test_model_settings_dir(tmp_path):
    cfg = _config(tmp_path)
    proj = tmp_path / 'domain_dom'
    (proj / 'settings/SUMMA').mkdir(parents=True)
    assert not sc.model_settings_exist(cfg, proj)  # empty dir: preprocessing never ran
    _touch(proj / 'settings/SUMMA/fileManager.txt')
    assert sc.model_settings_exist(cfg, proj)


def test_model_outputs_ignore_logs(tmp_path):
    cfg = _config(tmp_path)
    proj = tmp_path / 'domain_dom'
    _touch(proj / 'simulations/exp1/SUMMA/SUMMA_logs/run.log')
    assert not sc.model_outputs_exist(cfg, proj)
    _touch(proj / 'simulations/exp1/SUMMA/exp1_day.nc')
    assert sc.model_outputs_exist(cfg, proj)


def test_postprocess_results_csv(tmp_path):
    cfg = _config(tmp_path)
    proj = tmp_path / 'domain_dom'
    assert not sc.postprocessed_results_exist(cfg, proj)
    _touch(proj / 'results/exp1_results.csv')
    assert sc.postprocessed_results_exist(cfg, proj)


def test_calibration_results_model_subdir(tmp_path):
    cfg = _config(tmp_path, OPTIMIZATION_METHODS=['iteration'], ITERATIVE_OPTIMIZATION_ALGORITHM='DDS')
    proj = tmp_path / 'domain_dom'
    assert not sc.calibration_results_exist(cfg, proj)
    _touch(proj / 'optimization/SUMMA/dds_exp1/exp1_parallel_iteration_results.csv')
    assert sc.calibration_results_exist(cfg, proj)


def test_calibration_disabled_needs_no_output(tmp_path):
    cfg = _config(tmp_path, OPTIMIZATION_METHODS=[])
    assert sc.calibration_results_exist(cfg, tmp_path / 'domain_dom')


def test_analysis_not_requested_needs_no_output(tmp_path):
    cfg = _config(tmp_path, ANALYSES=['benchmarking'])
    out = tmp_path / 'nope.csv'
    assert sc.analysis_output_exists(cfg, 'decision', out)
    assert not sc.analysis_output_exists(cfg, 'benchmarking', out)


def test_completed_project_reports_complete_status(tmp_path):
    """A project whose steps wrote their real outputs + markers is fully complete."""
    cfg = _config(tmp_path, OPTIMIZATION_METHODS=['iteration'],
                  ITERATIVE_OPTIMIZATION_ALGORITHM='DDS', ANALYSES=[])
    managers = {k: MagicMock() for k in
                ('project', 'domain', 'data', 'model', 'analysis', 'optimization')}
    orch = WorkflowOrchestrator(managers, cfg, MagicMock())
    proj = orch.project_dir
    for rel in (
        'shapefiles/catchment/semidistributed/exp1/dom_HRUs_elevation.shp',
        'settings/SUMMA/fileManager.txt',
        'simulations/exp1/SUMMA/exp1_day.nc',
        'results/exp1_results.csv',
        'optimization/SUMMA/dds_exp1/exp1_parallel_iteration_results.csv',
    ):
        _touch(proj / rel)
    by_step = {s.name: s for s in orch.define_workflow_steps()}
    for name in ('discretize_domain', 'preprocess_models', 'run_models',
                 'postprocess_results', 'calibrate_model'):
        mark_stage_complete(cfg, proj, name)
        assert by_step[name].check_func() is True, name

    details = {d['name']: d for d in orch.get_workflow_status()['step_details']}
    for name in ('discretize_domain', 'preprocess_models', 'run_models',
                 'postprocess_results', 'calibrate_model'):
        assert details[name]['complete'], name
