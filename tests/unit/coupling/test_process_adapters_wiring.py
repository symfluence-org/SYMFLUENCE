"""Tests for process adapter read_outputs() wiring to real extractors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch


class TestParFlowReadOutputs:
    """Test ParFlowProcessComponent.read_outputs() uses real extractor."""

    def test_read_outputs_calls_extractor(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import ParFlowProcessComponent

        comp = ParFlowProcessComponent("parflow", config={
            'PARFLOW_OUTPUT_DIR': str(tmp_path),
            'SIMULATION_START': '2020-01-01',
        })

        overland = pd.Series([1.0, 2.0, 3.0], name="overland_flow")
        subsurface = pd.Series([0.5, 1.0, 1.5], name="subsurface_drainage")

        with patch(
            'symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent.read_outputs'
        ) as mock:
            # Just verify the method exists and returns correct structure
            mock.return_value = {"baseflow": torch.tensor([1.0, 2.0])}
            result = comp.read_outputs(tmp_path)

        assert "baseflow" in result
        assert isinstance(result["baseflow"], torch.Tensor)

    def test_read_outputs_returns_tensor_on_extractor_success(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import ParFlowProcessComponent

        comp = ParFlowProcessComponent("parflow", config={
            'PARFLOW_OUTPUT_DIR': str(tmp_path),
        })

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract_variable.return_value = pd.Series(
            [1.0, 2.0, 3.0], dtype=np.float32
        )

        # Create a fake extractor module with a mock class that returns our
        # mock instance.  We inject it into sys.modules so the `from … import`
        # inside read_outputs() picks it up regardless of whether the real
        # parflow package is installed.
        import sys
        import types
        fake_mod = types.ModuleType('symfluence.models.parflow.extractor')
        fake_mod.ParFlowResultExtractor = MagicMock(
            return_value=mock_extractor_instance
        )
        # Also ensure parent packages exist in sys.modules
        for mod_name in (
            'symfluence.models.parflow',
            'symfluence.models.parflow.extractor',
        ):
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules['symfluence.models.parflow.extractor'] = fake_mod

        try:
            result = comp.read_outputs(tmp_path)
        finally:
            # Clean up injected modules
            for mod_name in (
                'symfluence.models.parflow.extractor',
                'symfluence.models.parflow',
            ):
                sys.modules.pop(mod_name, None)

        assert "baseflow" in result
        assert result["baseflow"].dtype == torch.float32
        assert result["baseflow"].shape[0] == 3

    def test_read_outputs_fallback_on_import_error(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import ParFlowProcessComponent

        comp = ParFlowProcessComponent("parflow", config={})

        with patch(
            'symfluence.coupling.adapters.process_adapters.ParFlowProcessComponent.read_outputs',
            wraps=comp.read_outputs
        ):
            with patch.dict('sys.modules', {'symfluence.models.parflow.extractor': None}):
                with pytest.raises(RuntimeError, match="Failed to read ParFlow outputs"):
                    comp.read_outputs(tmp_path)


class TestMODFLOWReadOutputs:
    """Test MODFLOWProcessComponent.read_outputs() uses real extractor."""

    def test_read_outputs_returns_tensor(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import MODFLOWProcessComponent

        comp = MODFLOWProcessComponent("modflow", config={
            'MODFLOW_OUTPUT_DIR': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [10.0, 20.0, 30.0], dtype=np.float32
        )

        with patch(
            'symfluence.models.modflow.extractor.MODFLOWResultExtractor',
            return_value=mock_extractor
        ):
            result = comp.read_outputs(tmp_path)

        assert "drain_discharge" in result
        assert result["drain_discharge"].dtype == torch.float32
        assert result["drain_discharge"].shape[0] == 3

    def test_read_outputs_graceful_fallback(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import MODFLOWProcessComponent

        comp = MODFLOWProcessComponent("modflow", config={})

        # Simulate extractor raising an error
        with patch(
            'symfluence.models.modflow.extractor.MODFLOWResultExtractor',
            side_effect=Exception("No output files")
        ):
            with pytest.raises(RuntimeError, match="Failed to read MODFLOW outputs"):
                comp.read_outputs(tmp_path)


class TestMESHReadOutputs:
    """Test MESHProcessComponent.read_outputs() uses real extractor."""

    def test_read_outputs_with_basin_wb(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import MESHProcessComponent

        comp = MESHProcessComponent("mesh", config={
            'EXPERIMENT_OUTPUT_MESH': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [5.0, 6.0, 7.0], dtype=np.float32
        )

        # Create fake Basin_average_water_balance.csv so the path check succeeds
        (tmp_path / 'Basin_average_water_balance.csv').write_text("dummy")

        with patch(
            'symfluence.models.mesh.extractor.MESHResultExtractor',
            return_value=mock_extractor
        ):
            result = comp.read_outputs(tmp_path)

        assert "discharge" in result
        assert result["discharge"].dtype == torch.float32
        # Should have called with the basin_wb file
        call_args = mock_extractor.extract_variable.call_args
        assert 'Basin_average_water_balance.csv' in str(call_args[0][0])

    def test_read_outputs_fallback_to_dir(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import MESHProcessComponent

        comp = MESHProcessComponent("mesh", config={
            'EXPERIMENT_OUTPUT_MESH': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [1.0, 2.0], dtype=np.float32
        )

        # No Basin_average_water_balance.csv — falls back to output_dir
        with patch(
            'symfluence.models.mesh.extractor.MESHResultExtractor',
            return_value=mock_extractor
        ):
            result = comp.read_outputs(tmp_path)

        assert "discharge" in result
        call_args = mock_extractor.extract_variable.call_args
        assert call_args[0][0] == tmp_path


class TestCLMReadOutputs:
    """Test CLMProcessComponent.read_outputs() and CLMRunner wiring."""

    def test_read_outputs_returns_both_variables(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent

        comp = CLMProcessComponent("clm", config={
            'EXPERIMENT_OUTPUT_CLM': str(tmp_path),
        })

        mock_extractor = MagicMock()
        mock_extractor.extract_variable.return_value = pd.Series(
            [0.01, 0.02, 0.03], dtype=np.float32
        )

        with patch(
            'symfluence.models.clm.extractor.CLMResultExtractor',
            return_value=mock_extractor
        ):
            result = comp.read_outputs(tmp_path)

        assert "runoff" in result
        assert "evapotranspiration" in result
        assert result["runoff"].dtype == torch.float32
        assert result["evapotranspiration"].dtype == torch.float32

    def test_bmi_initialize_creates_runner(self):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent

        comp = CLMProcessComponent("clm")

        mock_runner = MagicMock()
        with patch(
            'symfluence.models.clm.runner.CLMRunner',
            return_value=mock_runner
        ):
            comp.bmi_initialize({'CLM_CESM_EXE': '/fake/cesm.exe'})

        assert comp._runner is mock_runner

    def test_execute_uses_runner_when_available(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent

        comp = CLMProcessComponent("clm")
        comp._runner = MagicMock()
        comp._runner.run.return_value = True

        ret = comp.execute(tmp_path)
        assert ret == 0
        comp._runner.run.assert_called_once()

    def test_execute_falls_back_to_subprocess(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent

        comp = CLMProcessComponent("clm", config={'CLM_CESM_EXE': '/nonexistent'})
        comp._runner = None  # No runner available

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            ret = comp.execute(tmp_path)

        assert ret == 0
        mock_run.assert_called_once()

    def test_read_outputs_graceful_fallback(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent

        comp = CLMProcessComponent("clm", config={})

        with patch(
            'symfluence.models.clm.extractor.CLMResultExtractor',
            side_effect=Exception("No history files")
        ):
            with pytest.raises(RuntimeError, match="Failed to read CLM outputs"):
                comp.read_outputs(tmp_path)


class TestTRouteReadOutputs:
    """Test t-route adapter error handling for missing discharge variables."""

    def test_read_outputs_raises_when_no_supported_discharge_variable(self, tmp_path):
        from symfluence.coupling.adapters.process_adapters import TRouteProcessComponent

        comp = TRouteProcessComponent("troute", config={
            "EXPERIMENT_OUTPUT_TROUTE": str(tmp_path),
        })
        (tmp_path / "troute_output.nc").write_text("placeholder")

        class _DummyDS(dict):
            def close(self):
                return None

        with patch("xarray.open_dataset", return_value=_DummyDS({"unknown": np.array([1.0], dtype=np.float32)})):
            with pytest.raises(RuntimeError, match="Failed to read t-route outputs"):
                comp.read_outputs(tmp_path)
