from __future__ import annotations

from pathlib import Path

from symfluence.models.wflow.calibration.parameter_manager import WflowParameterManager


def test_wflow_decay_units_and_no_injected_offset():
    b = WflowParameterManager.DEFAULT_BOUNDS["f"]
    assert b["min"] <= 0.003 <= b["max"] < 0.1
    config = {"WFLOW_PARAMS_TO_CALIBRATE": "KsatVer,f,G_Cfmax", "WFLOW_SPATIAL_MODE": "distributed"}
    import logging

    pm = WflowParameterManager(config, logging.getLogger("test"), Path("/tmp"))
    assert pm._get_parameter_names() == ["KsatVer", "f", "G_Cfmax"]
