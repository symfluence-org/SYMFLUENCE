"""Tests for the install / simulation path helpers and the call sites converted to them.

The equivalence tests pin each converted call site against a verbatim copy of
the hand-built logic it replaced, across the ``<MODEL>_INSTALL_PATH`` override
cases ('default', absolute dir, relative dir, absolute file, missing path), so
the refactor is provably behaviour-preserving.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pytest

from symfluence.core.mixins.project import ProjectContextMixin, simulation_dir
from symfluence.core.path_resolver import resolve_install_dir, resolve_install_executable

# =============================================================================
# resolve_install_dir / resolve_install_executable
# =============================================================================


class TestResolveInstallDir:
    @pytest.mark.parametrize("value", ["default", None, ""])
    def test_default_values_use_data_dir(self, tmp_path, value):
        assert resolve_install_dir(value, tmp_path, "installs/vic/bin") == tmp_path / "installs" / "vic" / "bin"

    def test_absolute_override_used_as_is(self, tmp_path):
        target = tmp_path / "custom"
        assert resolve_install_dir(str(target), tmp_path / "data", "installs/vic") == target

    def test_relative_override_stays_cwd_relative(self, tmp_path):
        # Matches BaseModelRunner.get_install_path: a relative override is NOT
        # re-anchored on data_dir.
        assert resolve_install_dir("rel/vic", tmp_path, "installs/vic") == Path("rel/vic")

    def test_data_dir_accepts_str(self, tmp_path):
        assert resolve_install_dir("default", str(tmp_path), "installs/x") == tmp_path / "installs" / "x"

    def test_data_dir_only_used_for_default(self):
        # An unset data_dir must not break an explicit override.
        assert resolve_install_dir("/opt/x", None, "installs/x") == Path("/opt/x")  # type: ignore[arg-type]


class TestResolveInstallExecutable:
    @pytest.mark.parametrize("value", ["default", None, ""])
    def test_default_joins_exe(self, tmp_path, value):
        result = resolve_install_executable(value, tmp_path, "installs/vic/bin", "vic.exe")
        assert result == tmp_path / "installs" / "vic" / "bin" / "vic.exe"

    def test_existing_dir_override_joins_exe(self, tmp_path):
        d = tmp_path / "bin"
        d.mkdir()
        assert resolve_install_executable(str(d), tmp_path, "installs/vic/bin", "vic.exe") == d / "vic.exe"

    def test_file_override_is_full_path(self, tmp_path):
        exe = tmp_path / "my_vic"
        exe.write_text("")
        assert resolve_install_executable(str(exe), tmp_path, "installs/vic/bin", "vic.exe") == exe

    def test_missing_override_is_full_path(self, tmp_path):
        missing = tmp_path / "nope"
        assert resolve_install_executable(str(missing), tmp_path, "installs/vic/bin", "vic.exe") == missing


# =============================================================================
# simulation_dir / ProjectContextMixin.get_simulation_dir
# =============================================================================


class _Ctx(ProjectContextMixin):
    def __init__(self, config: Dict[str, Any]):
        self._config = config


class TestSimulationDir:
    def test_function(self, tmp_path):
        assert simulation_dir(tmp_path, "run_1", "VIC") == tmp_path / "simulations" / "run_1" / "VIC"

    def test_mixin_defaults_to_experiment_id(self, tmp_path):
        ctx = _Ctx({"SYMFLUENCE_DATA_DIR": str(tmp_path), "DOMAIN_NAME": "d", "EXPERIMENT_ID": "exp7"})
        assert ctx.get_simulation_dir("VIC") == tmp_path / "domain_d" / "simulations" / "exp7" / "VIC"

    def test_mixin_explicit_experiment_id(self, tmp_path):
        ctx = _Ctx({"SYMFLUENCE_DATA_DIR": str(tmp_path), "DOMAIN_NAME": "d", "EXPERIMENT_ID": "exp7"})
        assert ctx.get_simulation_dir("GR", "other") == tmp_path / "domain_d" / "simulations" / "other" / "GR"

    def test_mixin_honours_project_dir_override(self, tmp_path):
        ctx = _Ctx({"EXPERIMENT_ID": "e"})
        ctx.project_dir = tmp_path / "proj"
        assert ctx.get_simulation_dir("SUMMA") == tmp_path / "proj" / "simulations" / "e" / "SUMMA"

    @pytest.mark.parametrize("override", ["default", "/abs/out", "rel/out"])
    def test_experiment_output_override_is_not_consulted(self, tmp_path, override):
        # The hand-built paths this replaced ignored EXPERIMENT_OUTPUT_<MODEL>;
        # the helper must too, or the conversion would change behaviour.
        ctx = _Ctx({
            "SYMFLUENCE_DATA_DIR": str(tmp_path), "DOMAIN_NAME": "d", "EXPERIMENT_ID": "e",
            "EXPERIMENT_OUTPUT_VIC": override,
        })
        hand_built = ctx.project_dir / 'simulations' / ctx.experiment_id / 'VIC'
        assert ctx.get_simulation_dir("VIC") == hand_built


# =============================================================================
# Equivalence of converted call sites with the hand-built logic they replaced
# =============================================================================

def _legacy_auto(install_path: Any, data_dir: Path, model: str, exe_name: str) -> Path:
    """Verbatim pre-refactor logic of the VIC/SWAT/WRF-Hydro/CRHM/PRMS/... workers."""
    if install_path == 'default':
        return data_dir / "installs" / model / "bin" / exe_name
    install_path = Path(install_path)
    if install_path.is_dir():
        return install_path / exe_name
    return install_path


def _legacy_fuse(install_path: Any, data_dir: Path) -> Path:
    if install_path == 'default':
        return data_dir / 'installs' / 'fuse' / 'bin' / 'fuse.exe'
    return Path(install_path) / 'fuse.exe'


def _legacy_dir(install_path: Any, data_dir: Path, sub: str) -> Path:
    if install_path == 'default' or install_path is None:
        return Path(data_dir) / 'installs' / sub
    return Path(install_path)


def _override_cases(tmp_path: Path):
    d = tmp_path / "custom_bin"
    d.mkdir()
    exe = tmp_path / "custom_exe"
    exe.write_text("")
    return {
        "default": "default",
        "absolute_dir": str(d),
        "absolute_file": str(exe),
        "relative_dir": "custom_bin",  # resolved against CWD (chdir'd to tmp_path)
        "missing": str(tmp_path / "does_not_exist"),
    }


OVERRIDE_IDS = ["default", "absolute_dir", "absolute_file", "relative_dir", "missing"]


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.parametrize("case", OVERRIDE_IDS)
@pytest.mark.parametrize(
    "module_path, cls_name, method, key, model, exe_key, exe_default",
    [
        ("symfluence.models.vic.calibration.worker", "VICWorker", "_get_vic_executable",
         "VIC_INSTALL_PATH", "vic", "VIC_EXE", "vic_image.exe"),
        ("symfluence.models.swat.calibration.worker", "SWATWorker", "_get_swat_executable",
         "SWAT_INSTALL_PATH", "swat", "SWAT_EXE", "swat_rel.exe"),
        ("symfluence.models.wrfhydro.calibration.worker", "WRFHydroWorker", "_get_wrfhydro_executable",
         "WRFHYDRO_INSTALL_PATH", "wrfhydro", "WRFHYDRO_EXE", "wrf_hydro.exe"),
        ("symfluence.models.prms.calibration.worker", "PRMSWorker", "_get_prms_executable",
         "PRMS_INSTALL_PATH", "prms", "PRMS_EXE", "prms.exe"),
    ],
)
def test_worker_executable_equivalence(in_tmp, case, module_path, cls_name, method, key, model,
                                       exe_key, exe_default):
    import importlib

    mod = importlib.import_module(module_path)
    worker_cls = getattr(mod, cls_name)
    worker = worker_cls.__new__(worker_cls)
    worker.logger = logging.getLogger("test")
    value = _override_cases(in_tmp)[case]
    data_dir = in_tmp / "data"
    # A '.exe' exe name sidesteps the Windows bare-name fallback some workers add,
    # so the comparison covers exactly the converted path logic.
    config = {key: value, exe_key: exe_default}
    expected = _legacy_auto(value, data_dir, model, exe_default)
    assert getattr(worker, method)(config, data_dir) == expected


@pytest.mark.parametrize("case", OVERRIDE_IDS)
def test_fuse_resolve_paths_equivalence(in_tmp, case):
    from symfluence.models.fuse.calibration.model_execution import resolve_fuse_paths

    value = _override_cases(in_tmp)[case]
    data_dir = in_tmp / "data"
    config = {"FUSE_INSTALL_PATH": value, "SYMFLUENCE_DATA_DIR": str(data_dir)}
    fuse_exe, _, _ = resolve_fuse_paths(config, in_tmp / "settings" / "FUSE")
    assert fuse_exe == _legacy_fuse(value, data_dir)


@pytest.mark.parametrize("case", OVERRIDE_IDS)
@pytest.mark.parametrize(
    "module_path, cls_name, attr, sub, model",
    [
        ("symfluence.models.noahmp.runner", "NoahMPRunner", "noahmp_dir", "noah-owp-modular", "noahmp"),
        ("symfluence.models.cwatm.runner", "CWatMRunner", "cwatm_dir", "cwatm", "cwatm"),
        ("symfluence.models.pcrglobwb.runner", "PCRGLOBWBRunner", "pcrglobwb_dir", "pcrglobwb", "pcrglobwb"),
    ],
)
def test_runner_install_dir_equivalence(in_tmp, case, module_path, cls_name, attr, sub, model):
    import importlib
    from types import SimpleNamespace

    mod = importlib.import_module(module_path)
    runner_cls = getattr(mod, cls_name)
    runner = runner_cls.__new__(runner_cls)
    value = _override_cases(in_tmp)[case]
    data_dir = in_tmp / "data"
    model_ns = SimpleNamespace(install_path=value, exe=None)
    runner._config = SimpleNamespace(
        model=SimpleNamespace(**{model: model_ns}),
        system=SimpleNamespace(data_dir=str(data_dir)),
    )
    runner.project_dir = in_tmp / "domain_x"
    runner._setup_model_specific_paths()
    assert getattr(runner, attr) == _legacy_dir(value, data_dir, sub)
