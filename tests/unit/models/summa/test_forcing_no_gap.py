# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Regenerating SUMMA forcing must never leave a reader without its files.

``{project_dir}/data/forcing/SUMMA_input`` is shared by every experiment on a
domain — the path carries no experiment_id — so a second run reaching
preprocessing rewrites the forcing a first run is calibrating against. The old
code deleted every output up front and wrote them back one at a time, leaving a
window in which the forcing did not exist at all. SUMMA reading it in that
window dies with a bare ``STOP 1`` and no output whatsoever.

Reproduced by timing before the fix: a read at 13:43:54 inside a
13:42:21-13:44:34 rewrite window failed; identical reads at 13:45:18 and later
all succeeded.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from symfluence.models.summa.forcing_processor import SummaForcingProcessor as _SFP


class _Proc:
    """Minimal stand-in exposing only the methods under test."""

    _prepare_forcing_output_dir = _SFP._prepare_forcing_output_dir
    _remove_stale_forcing_files = _SFP._remove_stale_forcing_files
    _stale_forcing_prefix = _SFP._stale_forcing_prefix
    _forcing_outputs_are_current = _SFP._forcing_outputs_are_current
    _select_run_files = _SFP._select_run_files

    def __init__(self, tmp_path: Path, logger):
        self.domain_name = "Dom"
        self.forcing_dataset = "RDRS"
        self.forcing_summa_path = tmp_path / "SUMMA_input"
        self.forcing_basin_path = tmp_path / "basin"
        self.forcing_summa_path.mkdir(parents=True)
        self.forcing_basin_path.mkdir(parents=True)
        self.logger = logger


@pytest.fixture
def proc(tmp_path):
    import logging
    return _Proc(tmp_path, logging.getLogger("test_forcing_no_gap"))


def _write(p: Path, text: str = "x") -> Path:
    p.write_text(text)
    return p


def test_preparing_the_output_dir_does_not_delete_existing_forcing(proc):
    """The regression: preparing must not create a window with no forcing."""
    kept = _write(proc.forcing_summa_path / "Dom_RDRS_remapped_2010-01.nc")
    proc._prepare_forcing_output_dir()
    assert kept.exists(), "existing forcing was deleted before its replacement existed"


def test_stale_files_are_removed_only_after_regeneration(proc):
    """Leftovers still get cleaned up — just at the end, not the start."""
    regenerated = _write(proc.forcing_summa_path / "Dom_RDRS_a.nc")
    stale = _write(proc.forcing_summa_path / "Dom_RDRS_old.nc")
    unrelated = _write(proc.forcing_summa_path / "Other_ERA5_x.nc")

    proc._remove_stale_forcing_files({regenerated.name})

    assert regenerated.exists()
    assert not stale.exists()
    assert unrelated.exists(), "only this domain+dataset's files may be swept"


def test_outputs_are_current_when_newer_than_every_source(proc):
    src = _write(proc.forcing_basin_path / "Dom_2010-01.nc")
    out = _write(proc.forcing_summa_path / "Dom_2010-01.nc")
    os.utime(out, (src.stat().st_mtime + 10, src.stat().st_mtime + 10))
    assert proc._forcing_outputs_are_current() is True


@pytest.mark.parametrize("scenario", ["missing_output", "stale_output", "no_sources"])
def test_regeneration_is_forced_when_anything_is_off(proc, scenario):
    """Conservative: only skip when every source is demonstrably covered."""
    if scenario != "no_sources":
        src = _write(proc.forcing_basin_path / "Dom_2010-01.nc")
    if scenario == "stale_output":
        out = _write(proc.forcing_summa_path / "Dom_2010-01.nc")
        os.utime(out, (src.stat().st_mtime - 10, src.stat().st_mtime - 10))
    # missing_output: source exists, no output written at all
    assert proc._forcing_outputs_are_current() is False


def test_a_second_source_missing_its_output_forces_regeneration(proc):
    """One covered file must not vouch for the rest."""
    a = _write(proc.forcing_basin_path / "Dom_2010-01.nc")
    _write(proc.forcing_basin_path / "Dom_2010-02.nc")
    out_a = _write(proc.forcing_summa_path / "Dom_2010-01.nc")
    os.utime(out_a, (a.stat().st_mtime + 10, a.stat().st_mtime + 10))
    assert proc._forcing_outputs_are_current() is False
