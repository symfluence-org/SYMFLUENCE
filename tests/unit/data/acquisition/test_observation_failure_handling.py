# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Failure-loudness regression tests for AcquisitionService.acquire_observations.

NB and NV reported that downstream calibration / benchmarking failed
because the primary streamflow observation file was missing — even
though ``process_observed_data`` had reported "✓ Complete (Duration:
0.00s)" with exit 0. Root cause: the WSC handler raised an exception
(HYDAT database not found), but acquire_observations only logged a
warning, so the workflow continued past the obs step into calibration
which then crashed with a confusing 'No such file' error.

Fix: the obs configured via ``streamflow_data_provider`` is now PRIMARY
and any acquisition exception there raises out of the obs step. Other
observations (GRACE, SNOTEL, etc.) remain best-effort and continue to
log warnings so a missing optional dataset doesn't break a streamflow-
only calibration.
"""

import logging
from unittest.mock import patch

import pytest

from symfluence.core.config.models import SymfluenceConfig
from symfluence.data.acquisition.acquisition_service import AcquisitionService

pytestmark = [pytest.mark.unit, pytest.mark.quick]


def _service(tmp_path, **overrides):
    config_dict = {
        "SYMFLUENCE_DATA_DIR": str(tmp_path),
        "SYMFLUENCE_CODE_DIR": str(tmp_path / "code"),
        "DOMAIN_NAME": "obs_failure_test",
        "EXPERIMENT_ID": "test",
        "EXPERIMENT_TIME_START": "2020-01-01 00:00",
        "EXPERIMENT_TIME_END": "2020-01-02 00:00",
        "FORCING_DATASET": "ERA5",
        "HYDROLOGICAL_MODEL": "SUMMA",
        "DOMAIN_DEFINITION_METHOD": "lumped",
        "SUB_GRID_DISCRETIZATION": "GRUs",
    }
    config_dict.update(overrides)
    config = SymfluenceConfig(**config_dict)
    return AcquisitionService(
        config, logging.getLogger("test_obs_failure"),
    )


def _stub_run_parallel_with_failures(failures: dict):
    """Build a fake _run_parallel_tasks that returns the given failure map.

    Keys are observation names; values are Exception instances to surface
    as the result for that name.
    """
    def _runner(self, tasks, desc="Acquiring"):
        out = {}
        for name, _func in tasks:
            if name in failures:
                out[name] = failures[name]
            else:
                out[name] = None
        return out
    return _runner


def test_primary_streamflow_failure_raises(tmp_path):
    """When streamflow_data_provider=WSC and the WSC handler raises
    (e.g. HYDAT database missing), acquire_observations must raise
    instead of swallowing the failure as a warning."""
    svc = _service(tmp_path, STREAMFLOW_DATA_PROVIDER="WSC")
    failure = FileNotFoundError(
        "HYDAT database not found at: /opt/data/hydat/Hydat.sqlite3"
    )

    with patch.object(
        AcquisitionService,
        "_run_parallel_tasks",
        _stub_run_parallel_with_failures({"WSC_STREAMFLOW": failure}),
    ), patch(
        "symfluence.core.registries.R"
    ) as mock_R:
        from unittest.mock import MagicMock
        mock_R.observation_handlers.__contains__.return_value = True

        def _make_handler(*a, **k):
            h = MagicMock()
            h.acquire = MagicMock()
            return h

        mock_R.observation_handlers.get.return_value = _make_handler

        with pytest.raises(ValueError) as exc:
            svc.acquire_observations()

    msg = str(exc.value)
    assert "WSC_STREAMFLOW" in msg
    assert "calibration" in msg.lower()  # explains why we stop
    assert "HYDAT" in msg  # actionable hint preserved


def test_primary_usgs_failure_raises(tmp_path):
    """Same loudness contract for USGS provider."""
    svc = _service(tmp_path, STREAMFLOW_DATA_PROVIDER="USGS")
    failure = ConnectionError("USGS API timed out")

    with patch.object(
        AcquisitionService,
        "_run_parallel_tasks",
        _stub_run_parallel_with_failures({"USGS_STREAMFLOW": failure}),
    ), patch(
        "symfluence.core.registries.R"
    ) as mock_R:
        from unittest.mock import MagicMock
        mock_R.observation_handlers.__contains__.return_value = True

        def _make_handler(*a, **k):
            h = MagicMock()
            h.acquire = MagicMock()
            return h

        mock_R.observation_handlers.get.return_value = _make_handler

        with pytest.raises(ValueError) as exc:
            svc.acquire_observations()

    assert "USGS_STREAMFLOW" in str(exc.value)


def test_optional_observation_failure_stays_warning(tmp_path, caplog):
    """A failure on an OPTIONAL obs (e.g. GRACE) must NOT raise — it
    just logs a warning so users with missing optional data can still
    calibrate against streamflow."""
    svc = _service(tmp_path, ADDITIONAL_OBSERVATIONS="GRACE")
    failure = RuntimeError("GRACE service unavailable")

    with patch.object(
        AcquisitionService,
        "_run_parallel_tasks",
        _stub_run_parallel_with_failures({"GRACE": failure}),
    ), patch(
        "symfluence.core.registries.R"
    ) as mock_R:
        from unittest.mock import MagicMock
        mock_R.observation_handlers.__contains__.return_value = True

        def _make_handler(*a, **k):
            h = MagicMock()
            h.acquire = MagicMock()
            return h

        mock_R.observation_handlers.get.return_value = _make_handler

        # Must NOT raise
        svc.acquire_observations()

    # And the warning must be visible in logs
    warnings = [
        rec for rec in caplog.records
        if rec.levelname == "WARNING" and "GRACE" in rec.message
    ]
    assert warnings, "Expected a warning log line for the failed optional GRACE observation"


def test_no_streamflow_provider_no_raise(tmp_path):
    """If no streamflow_data_provider is configured, no observation is
    primary, so even all-failures must NOT raise (best-effort mode)."""
    svc = _service(tmp_path, ADDITIONAL_OBSERVATIONS="GRACE,SNOTEL")
    failures = {
        "GRACE": RuntimeError("oops"),
        "SNOTEL": RuntimeError("nope"),
    }

    with patch.object(
        AcquisitionService,
        "_run_parallel_tasks",
        _stub_run_parallel_with_failures(failures),
    ), patch(
        "symfluence.core.registries.R"
    ) as mock_R:
        from unittest.mock import MagicMock
        mock_R.observation_handlers.__contains__.return_value = True

        def _make_handler(*a, **k):
            h = MagicMock()
            h.acquire = MagicMock()
            return h

        mock_R.observation_handlers.get.return_value = _make_handler

        svc.acquire_observations()  # no exception


# Handler-level opt-out: streamflow_data_provider implies download_*=True
# so users don't need two flags to get the obvious behaviour, and we
# never silently fall through to a local-DB path that requires HYDAT.

def test_wsc_handler_implicit_download_when_provider_set(tmp_path):
    """When streamflow_data_provider=WSC is set, the WSC handler must
    enable cloud GeoMet download by default (not require an additional
    DOWNLOAD_WSC_DATA: True flag). Otherwise the handler falls through
    to the HYDAT path which crashes on machines without the local DB."""
    from unittest.mock import MagicMock

    from symfluence.data.observation.handlers.wsc import WSCStreamflowHandler

    config = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        SYMFLUENCE_CODE_DIR=str(tmp_path / "code"),
        DOMAIN_NAME="wsc_optout_test",
        EXPERIMENT_ID="test",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        FORCING_DATASET="ERA5",
        HYDROLOGICAL_MODEL="SUMMA",
        DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="GRUs",
        STREAMFLOW_DATA_PROVIDER="WSC",
        STATION_ID="05BB001",
        DATA_ACCESS="cloud",
        # Note: DOWNLOAD_WSC_DATA NOT set — must default-on
    )
    handler = WSCStreamflowHandler(config, logging.getLogger("test_wsc_optout"))

    with patch.object(
        WSCStreamflowHandler, "_download_from_geomet", return_value=tmp_path / "fake.csv"
    ) as mock_dl:
        handler.acquire()
    mock_dl.assert_called_once()


def test_usgs_handler_implicit_download_when_provider_set(tmp_path):
    """When streamflow_data_provider=USGS is set, the USGS handler
    must enable NWIS download by default."""
    from unittest.mock import MagicMock

    from symfluence.data.observation.handlers.usgs import USGSStreamflowHandler

    config = SymfluenceConfig(
        SYMFLUENCE_DATA_DIR=str(tmp_path),
        SYMFLUENCE_CODE_DIR=str(tmp_path / "code"),
        DOMAIN_NAME="usgs_optout_test",
        EXPERIMENT_ID="test",
        EXPERIMENT_TIME_START="2020-01-01 00:00",
        EXPERIMENT_TIME_END="2020-01-02 00:00",
        FORCING_DATASET="ERA5",
        HYDROLOGICAL_MODEL="SUMMA",
        DOMAIN_DEFINITION_METHOD="lumped",
        SUB_GRID_DISCRETIZATION="GRUs",
        STREAMFLOW_DATA_PROVIDER="USGS",
        STATION_ID="06892350",
        # Note: DOWNLOAD_USGS_DATA NOT set — must default-on
    )
    handler = USGSStreamflowHandler(config, logging.getLogger("test_usgs_optout"))

    with patch.object(
        USGSStreamflowHandler, "_download_data", return_value=tmp_path / "fake.rdb"
    ) as mock_dl:
        handler.acquire()
    mock_dl.assert_called_once()
