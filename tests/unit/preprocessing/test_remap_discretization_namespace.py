# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Tests for discretization-namespaced remapped-forcing filenames (issue #339).

The two ``determine_output_filename`` builders — FileProcessor (serial/parallel
resampling path) and RemappingWeightApplier (weight-application path) — must
embed the run's spatial discretization token so a lumped (hru=1) remap and a
12-band elevation (hru=12) remap of the SAME domain get distinct, self-describing
names instead of colliding under a shared ``{domain}_{forcing}_remapped_*``
namespace. Both builders must agree on the name for a given input + discretization.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from symfluence.data.preprocessing.resampling.file_processor import FileProcessor

_LOG = logging.getLogger("test")


def _fp(discretization, out):
    return FileProcessor(
        config={
            "DOMAIN_NAME": "Bow",
            "FORCING_DATASET": "ERA5",
            "SUB_GRID_DISCRETIZATION": discretization,
        },
        output_dir=out,
        logger=_LOG,
    )


def _applier(discretization, out):
    # RemappingWeightApplier lives in a module that imports easymore at import
    # time; skip the applier-side assertions where easymore is unavailable.
    pytest.importorskip("easymore")
    from symfluence.data.preprocessing.remapping_weights import RemappingWeightApplier

    return RemappingWeightApplier(
        config={
            "DOMAIN_NAME": "Bow",
            "FORCING_DATASET": "ERA5",
            "SUB_GRID_DISCRETIZATION": discretization,
        },
        logger=_LOG,
        project_dir=out.parent,
        output_dir=out,
        dataset_handler=None,
    )


def test_file_processor_names_carry_discretization_token(tmp_path):
    src = Path("domain_Bow_ERA5_20020101.nc")
    lumped = _fp("lumped", tmp_path).determine_output_filename(src).name
    elevation = _fp("elevation", tmp_path).determine_output_filename(src).name

    assert "_remapped_lumped_" in lumped
    assert "_remapped_elevation_" in elevation
    # The two discretizations of the SAME input produce DIFFERENT names, so they
    # no longer collide in one domain's store.
    assert lumped != elevation


def test_weight_applier_names_carry_discretization_token(tmp_path):
    src = Path("domain_Bow_ERA5_20020101.nc")
    lumped = _applier("lumped", tmp_path).determine_output_filename(src).name
    elevation = _applier("elevation", tmp_path).determine_output_filename(src).name

    assert "_remapped_lumped_" in lumped
    assert "_remapped_elevation_" in elevation
    assert lumped != elevation


def test_both_builders_agree_on_the_name(tmp_path):
    """Generation must be self-consistent: the two builders name a file the same.

    The parallel resampling path names outputs via FileProcessor while the
    weight-application path names them via RemappingWeightApplier; a divergence
    would resurrect the duplicate-file problem the reader now dedupes.
    """
    src = Path("domain_Bow_ERA5_20020101.nc")
    fp_name = _fp("lumped", tmp_path).determine_output_filename(src).name
    applier_name = _applier("lumped", tmp_path).determine_output_filename(src).name
    assert fp_name == applier_name


def test_names_are_selectable_by_the_reader(tmp_path):
    """The generated names round-trip through the reader's selection helper."""
    from symfluence.core.modeling.model_ready.forcing_reader import select_forcing_files

    src = Path("domain_Bow_ERA5_20020101.nc")
    lumped = _fp("lumped", tmp_path).determine_output_filename(src)
    elevation = _fp("elevation", tmp_path).determine_output_filename(src)
    store = [lumped, elevation]

    assert [p.name for p in select_forcing_files(store, "lumped")] == [lumped.name]
    assert [p.name for p in select_forcing_files(store, "elevation")] == [elevation.name]
