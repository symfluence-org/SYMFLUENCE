# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for SUMMA GRU-parallel execution."""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import xarray as xr

import symfluence.models.summa.runner as summa_runner
from symfluence.core.config.models import SymfluenceConfig
from symfluence.models.summa.runner import SummaRunner


def make_summa_runner(
    tmp_path: Path,
    backend: str = 'local',
    cpus_per_task: int = 4,
) -> SummaRunner:
    """Create a SUMMA runner with local parallel execution enabled."""
    data_dir = tmp_path / 'data'
    code_dir = tmp_path / 'code'
    install_dir = tmp_path / 'summa_bin'
    settings_dir = data_dir / 'domain_test_domain' / 'settings' / 'SUMMA'

    install_dir.mkdir(parents=True)
    settings_dir.mkdir(parents=True)
    code_dir.mkdir(parents=True)

    summa_exe = install_dir / 'summa_sundials.exe'
    summa_exe.write_text('#!/bin/sh\n', encoding='utf-8')
    summa_exe.chmod(0o755)
    (settings_dir / 'fileManager.txt').write_text(
        "simStartTime '2020-01-01 00:00'\n"
        "simEndTime '2020-01-02 00:00'\n"
        "outFilePrefix 'test_run'\n",
        encoding='utf-8',
    )

    config = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(data_dir),
        SYMFLUENCE_CODE_DIR=str(code_dir),
        DOMAIN_NAME='test_domain',
        EXPERIMENT_ID='test_run',
        EXPERIMENT_TIME_START='2020-01-01 00:00',
        EXPERIMENT_TIME_END='2020-01-02 00:00',
        DOMAIN_DEFINITION_METHOD='semidistributed',
        SUB_GRID_DISCRETIZATION='GRUs',
        HYDROLOGICAL_MODEL='SUMMA',
        ROUTING_MODEL='none',
        FORCING_DATASET='ERA5',
        SUMMA_INSTALL_PATH=str(install_dir),
        SETTINGS_SUMMA_PATH=str(settings_dir),
        SETTINGS_SUMMA_USE_PARALLEL_SUMMA=True,
        SETTINGS_SUMMA_PARALLEL_BACKEND=backend,
        SETTINGS_SUMMA_CPUS_PER_TASK=cpus_per_task,
    )

    return SummaRunner(config, logging.getLogger(__name__))


def prepare_local_parallel_runner(runner: SummaRunner, tmp_path: Path) -> None:
    """Patch expensive runner steps for focused local backend tests."""
    runner.output_dir = tmp_path / 'output'
    runner.output_dir.mkdir()
    runner._pre_execution = MagicMock(return_value=True)
    runner._merge_parallel_outputs = MagicMock(return_value=runner.output_dir)


def test_local_parallel_summa_calls_gru_split_helper(monkeypatch, tmp_path):
    """Test local parallel SUMMA delegates to the GRU split helper."""
    runner = make_summa_runner(tmp_path, cpus_per_task=4)
    prepare_local_parallel_runner(runner, tmp_path)
    calls = []

    def fake_run_summa_gru_parallel(**kwargs):
        calls.append(kwargs)
        times = pd.date_range('2020-01-01', '2020-01-02', freq='h')
        xr.Dataset(coords={'time': times}).to_netcdf(
            kwargs['summa_dir'] / 'test_run_timestep.nc'
        )
        return True

    monkeypatch.setattr(
        summa_runner,
        'run_summa_gru_parallel',
        fake_run_summa_gru_parallel,
    )

    result = runner.run_parallel_summa()

    assert result == runner.output_dir
    runner._pre_execution.assert_called_once()
    runner._merge_parallel_outputs.assert_not_called()

    assert len(calls) == 1
    call = calls[0]
    assert call['summa_exe'] == runner.model_exe
    assert call['file_manager'] == runner.file_manager
    assert call['summa_dir'] == runner.output_dir
    assert call['settings_dir'] == runner.settings_path
    assert call['num_parallel'] == 4
    assert call['timeout'] == 7200
    assert call['debug_info'] == {'errors': []}
    assert isinstance(call['env'], dict)


def test_local_parallel_summa_returns_none_after_helper_failure(
    monkeypatch, tmp_path
):
    """Test local parallel SUMMA returns no output after helper failure."""
    runner = make_summa_runner(tmp_path, cpus_per_task=2)
    prepare_local_parallel_runner(runner, tmp_path)

    def fake_run_summa_gru_parallel(**kwargs):
        return False

    monkeypatch.setattr(
        summa_runner,
        'run_summa_gru_parallel',
        fake_run_summa_gru_parallel,
    )

    result = runner.run_parallel_summa()

    assert result is None
    runner._merge_parallel_outputs.assert_not_called()


def test_parallel_summa_slurm_backend_dispatches_to_slurm(tmp_path):
    """Test SUMMA parallel execution dispatches to SLURM when available."""
    runner = make_summa_runner(tmp_path, backend='slurm')
    expected = tmp_path / 'slurm-output'
    runner.is_slurm_available = MagicMock(return_value=True)
    runner._run_parallel_summa_slurm = MagicMock(return_value=expected)
    runner._run_parallel_summa_local = MagicMock()

    result = runner.run_parallel_summa()

    assert result == expected
    runner.is_slurm_available.assert_called_once()
    runner._run_parallel_summa_slurm.assert_called_once()
    runner._run_parallel_summa_local.assert_not_called()


def test_parallel_summa_slurm_backend_falls_back_to_local(tmp_path):
    """Test SUMMA parallel execution uses local splitting without SLURM."""
    runner = make_summa_runner(tmp_path, backend='slurm')
    expected = tmp_path / 'local-output'
    runner.is_slurm_available = MagicMock(return_value=False)
    runner._run_parallel_summa_local = MagicMock(return_value=expected)
    runner._run_parallel_summa_slurm = MagicMock()

    result = runner.run_parallel_summa()

    assert result == expected
    runner.is_slurm_available.assert_called_once()
    runner._run_parallel_summa_local.assert_called_once()
    runner._run_parallel_summa_slurm.assert_not_called()


def test_parallel_summa_unknown_backend_returns_none(tmp_path):
    """Test unknown SUMMA parallel backend returns no output path."""
    runner = make_summa_runner(tmp_path)
    runner._get_config_value = MagicMock(return_value='unknown')
    runner._run_parallel_summa_local = MagicMock()
    runner._run_parallel_summa_slurm = MagicMock()

    result = runner.run_parallel_summa()

    assert result is None
    runner._run_parallel_summa_local.assert_not_called()
    runner._run_parallel_summa_slurm.assert_not_called()
