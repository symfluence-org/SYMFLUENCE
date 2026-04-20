# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Tests pinning the canonical fresh-project directory layout.

The DEM acquisition was writing to ``attributes/elevation/dem/`` (legacy
path) while TauDEM in ``define_domain`` was reading from
``data/attributes/elevation/dem/``. Root cause: ``setup_project``
created the legacy ``attributes/`` directory directly, which made the
backward-compat branch in ``resolve_data_subdir`` pick the legacy
path on subsequent reads. ``setup_project`` now creates ``data/...``
up-front so the new layout wins on fresh projects.

Legacy projects continue to work via the existing backward-compat
fallback in ``resolve_data_subdir``.
"""

import logging
from pathlib import Path

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.core.mixins.project import resolve_data_subdir
from symfluence.project.project_manager import ProjectManager

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _make_config(tmp_path: Path) -> SymfluenceConfig:
    return SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        SYMFLUENCE_CODE_DIR=str(tmp_path / "code"),
        DOMAIN_NAME="layout_test",
        DEM_NAME="default",
        DEM_PATH="default",
        DOMAIN_DEFINITION_METHOD="lumped",
        CATCHMENT_PATH="default",
        CATCHMENT_SHP_NAME="default",
        CATCHMENT_SHP_GRUID="GRU_ID",
        CATCHMENT_SHP_HRUID="HRU_ID",
        SUB_GRID_DISCRETIZATION="GRUs",
        EXPERIMENT_ID="test",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        FORCING_DATASET="ERA5",
        HYDROLOGICAL_MODEL="SUMMA",
    )


def test_setup_project_creates_canonical_data_layout(tmp_path):
    """``setup_project`` must pre-create ``data/attributes``,
    ``data/forcing``, ``data/observations``, and ``data/model_ready`` so
    that the canonical layout wins on every subsequent
    ``resolve_data_subdir`` lookup."""
    cfg = _make_config(tmp_path)
    pm = ProjectManager(cfg, logging.getLogger("test_setup_project"))
    project_dir = pm.setup_project()

    # Top-level shapefile structure (unchanged)
    assert (project_dir / "shapefiles" / "pour_point").is_dir()
    assert (project_dir / "shapefiles" / "catchment").is_dir()
    assert (project_dir / "shapefiles" / "river_network").is_dir()
    assert (project_dir / "shapefiles" / "river_basins").is_dir()

    # Canonical data/ subtree
    assert (project_dir / "data" / "attributes").is_dir(), (
        "data/attributes/ must exist after setup_project so the DEM "
        "acquirer writes here and TauDEM reads here."
    )
    assert (project_dir / "data" / "forcing").is_dir()
    assert (project_dir / "data" / "observations").is_dir()
    assert (project_dir / "data" / "model_ready").is_dir()

    # And the legacy paths are NOT pre-created (would cause
    # resolve_data_subdir to pick legacy path)
    assert not (project_dir / "attributes").exists()
    assert not (project_dir / "forcing").exists()
    assert not (project_dir / "observations").exists()


def test_resolve_data_subdir_picks_canonical_after_setup(tmp_path):
    """After ``setup_project``, ``resolve_data_subdir`` for any of the
    data subdirectories must point at the new ``data/...`` location.
    This is what was broken before — the legacy ``attributes/`` was
    created up-front and resolve_data_subdir picked it as a side effect."""
    cfg = _make_config(tmp_path)
    pm = ProjectManager(cfg, logging.getLogger("test_resolve_after_setup"))
    project_dir = pm.setup_project()

    for subdir in ("attributes", "forcing", "observations"):
        resolved = resolve_data_subdir(project_dir, subdir)
        expected = project_dir / "data" / subdir
        assert resolved == expected, (
            f"resolve_data_subdir({project_dir!r}, {subdir!r}) returned "
            f"{resolved} but the canonical layout requires {expected}"
        )


def test_resolve_data_subdir_legacy_layout_still_works(tmp_path):
    """Backward-compat: a project that already has a legacy
    ``attributes/`` directory (from a pre-fix run) must keep resolving
    to that legacy path so existing data is not orphaned."""
    project_dir = tmp_path / "domain_legacy"
    legacy_attrs = project_dir / "attributes"
    legacy_attrs.mkdir(parents=True)

    resolved = resolve_data_subdir(project_dir, "attributes")
    assert resolved == legacy_attrs, (
        "resolve_data_subdir must keep returning the legacy "
        "{project}/attributes/ path when it exists, so existing data "
        "is not orphaned by the canonical-layout fix."
    )
