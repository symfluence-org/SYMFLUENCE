# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Regression tests for the DEM_SOURCE error message.

SH reported that paper configs using ``DEM_SOURCE: MERIT`` failed with a
bare "unsupported" message and the fix was discovered by trial and error
(switching to ``copernicus``). MERIT-Hydro is only reachable via the MAF
gistool (``DATA_ACCESS: hpc``) path; with ``DATA_ACCESS: cloud`` the
request cannot succeed. Rather than silently aliasing ``MERIT`` to
something the cloud dispatcher could not serve, we surface a specific
actionable hint in the error so users understand the underlying
constraint.

The four paper configs in ``configs_orig/11_data_pipeline/`` that
previously said ``DEM_SOURCE: MERIT`` were also updated to
``copernicus`` in the same PR; this test only pins the error-message
contract on the code side.
"""

import logging
from unittest.mock import patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.acquisition.acquisition_service import AcquisitionService

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _service(tmp_path, **overrides):
    cfg = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "dem_err_test",
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
        "BOUNDING_BOX_COORDS": "44.5/-87.9/44.2/-87.5",
        "DATA_ACCESS": "cloud",
    }
    cfg.update(overrides)
    return AcquisitionService(
        SymfluenceConfig(**cfg), logging.getLogger("dem_err_test")
    )


def _trigger_dem_dispatch(svc):
    """Drive acquire_attributes far enough to hit the DEM-source branch.

    The cloud path reaches the unsupported-DEM_SOURCE raise inside a
    task appended to a parallel executor. We run the single task inline
    to bubble the raise directly.
    """
    # The raise lives in the cloud branch only; stub out the downloader
    # so we don't actually hit Copernicus etc.
    with patch(
        "symfluence.data.acquisition.acquisition_service.CloudForcingDownloader"
    ), patch.object(
        AcquisitionService, "_run_parallel_tasks",
        side_effect=lambda tasks, desc="": {
            name: (fn() if callable(fn) else None)  # actually execute the closure
            for name, fn in tasks
        },
    ):
        svc.acquire_attributes()


def test_merit_on_cloud_gives_actionable_error(tmp_path):
    """DEM_SOURCE=MERIT with DATA_ACCESS=cloud must name MERIT
    specifically, explain that it needs hpc + gistool, and list cloud
    alternatives."""
    svc = _service(tmp_path, DEM_SOURCE="MERIT")
    with pytest.raises(Exception) as exc:
        _trigger_dem_dispatch(svc)
    msg = str(exc.value)
    # Specific to MERIT → name the cloud/hpc distinction
    assert "MERIT" in msg or "merit" in msg.lower()
    assert "hpc" in msg.lower() or "MAF" in msg or "gistool" in msg
    assert "copernicus" in msg.lower()


def test_generic_unsupported_dem_lists_cloud_alternatives(tmp_path):
    """For any other unsupported DEM_SOURCE, the error lists the cloud
    alternatives so the user can pick one without spelunking."""
    svc = _service(tmp_path, DEM_SOURCE="not_a_real_source")
    with pytest.raises(Exception) as exc:
        _trigger_dem_dispatch(svc)
    msg = str(exc.value)
    assert "not_a_real_source" in msg
    # Must list at least the default and one alternative
    assert "copernicus" in msg.lower()
    assert "fabdem" in msg.lower() or "nasadem" in msg.lower()
