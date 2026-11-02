# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""RHESSys calibration must expose libwmfire.so on the dynamic-link path.

``libwmfire.so`` ships inside the RHESSys install tree at
``installs/rhessys/lib/`` (NOT under ``installs/wmfire/lib``). If the calibration
worker omits that directory from ``LD_LIBRARY_PATH``/``DYLD_LIBRARY_PATH``, the
binary fails at dynamic-link time ("libwmfire.so: cannot open shared object
file") on every DDS evaluation, so the whole calibration returns a penalty score
with no genuine evaluation while the workflow still exits rc=0.

Regression guard for that gap: the worker's run environment must include the
``lib`` sibling of the executable's ``bin`` directory, matching
``RHESSysRunner._get_run_environment``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from symfluence.models.rhessys.calibration.worker import RHESSysWorker


def _make_install(data_dir: Path) -> Path:
    """Create a fake RHESSys install with bin/rhessys and lib/libwmfire.so."""
    bin_dir = data_dir / "installs" / "rhessys" / "bin"
    lib_dir = data_dir / "installs" / "rhessys" / "lib"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "rhessys"
    exe.write_text("#!/bin/sh\n")
    (lib_dir / "libwmfire.so").write_bytes(b"\x00")
    return lib_dir


def test_run_env_includes_rhessys_lib_dir(tmp_path):
    """The subprocess env must carry installs/rhessys/lib for libwmfire.so."""
    data_dir = tmp_path
    lib_dir = _make_install(data_dir)

    config = {"DOMAIN_NAME": "test", "SYMFLUENCE_DATA_DIR": str(data_dir)}
    worker = RHESSysWorker(config=config)

    captured = {}

    def _fake_run(cmd, **kwargs):  # test stub
        captured["env"] = kwargs.get("env")
        return mock.Mock(returncode=0, stdout="", stderr="")

    # Avoid needing real worldfiles/tec/routing — the command content is
    # irrelevant to the library-path assertion.
    with mock.patch.object(worker, "_build_command", return_value=["/bin/true"]), \
         mock.patch(
             "symfluence.models.rhessys.calibration.worker.run_subprocess",
             side_effect=_fake_run,
         ):
        worker.run_model(
            config,
            settings_dir=tmp_path / "settings",
            output_dir=tmp_path / "out",
            sim_dir=tmp_path / "out",
        )

    assert captured["env"] is not None, "run_subprocess was not invoked"
    var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    search_path = captured["env"].get(var, "")
    assert str(lib_dir) in search_path, (
        f"{var} must include the RHESSys lib dir holding libwmfire.so; "
        f"got: {search_path!r}"
    )
