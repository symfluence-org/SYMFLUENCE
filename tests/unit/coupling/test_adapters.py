"""Tests for SYMFLUENCE coupling adapters."""
from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")
pytest.importorskip("dcoupler", reason="requires the ml extra (pip install 'symfluence[ml]')")
import torch
from dcoupler.core.component import FluxDirection, GradientMethod


class TestProcessAdapters:
    def test_summa_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import SUMMAProcessComponent
        comp = SUMMAProcessComponent("summa")
        assert comp.name == "summa"
        assert comp.requires_batch is True
        assert comp.gradient_method == GradientMethod.NONE
        assert len(comp.input_fluxes) == 1
        assert len(comp.output_fluxes) == 2
        assert comp.output_fluxes[0].conserved_quantity == "water_mass"

    def test_mizuroute_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import MizuRouteProcessComponent
        comp = MizuRouteProcessComponent("mizuroute")
        assert comp.name == "mizuroute"
        assert comp.input_fluxes[0].name == "lateral_inflow"
        assert comp.output_fluxes[0].name == "discharge"

    def test_parflow_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import ParFlowProcessComponent
        comp = ParFlowProcessComponent("parflow")
        assert comp.input_fluxes[0].units == "m/hr"
        assert comp.KG_M2_S_TO_M_HR == 3.6

    def test_modflow_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import MODFLOWProcessComponent
        comp = MODFLOWProcessComponent("modflow")
        assert comp.input_fluxes[0].units == "m/d"
        assert comp.KG_M2_S_TO_M_D == 86.4

    def test_mesh_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import MESHProcessComponent
        comp = MESHProcessComponent("mesh")
        assert comp.name == "mesh"
        assert comp.output_fluxes[0].name == "discharge"

    def test_clm_adapter_properties(self):
        from symfluence.coupling.adapters.process_adapters import CLMProcessComponent
        comp = CLMProcessComponent("clm")
        assert len(comp.output_fluxes) == 2
        assert comp.output_fluxes[0].name == "runoff"
        assert comp.output_fluxes[1].name == "evapotranspiration"

    def test_summa_bmi_var_names(self):
        from symfluence.coupling.adapters.process_adapters import SUMMAProcessComponent
        comp = SUMMAProcessComponent("summa")
        assert "runoff" in comp.bmi_get_output_var_names()
        assert "forcing" in comp.bmi_get_input_var_names()

    def test_step_raises_on_process(self):
        from symfluence.coupling.adapters.process_adapters import SUMMAProcessComponent
        comp = SUMMAProcessComponent("summa")
        with pytest.raises(RuntimeError, match="batch execution"):
            comp.step({}, torch.empty(0), 1.0)


class TestJAXAdapters:
    @pytest.fixture(autouse=True)
    def check_jax(self):
        try:
            import jax
        except ImportError:
            pytest.skip("JAX not installed")

    def test_snow17_adapter_creation(self):
        try:
            from symfluence.coupling.adapters.jax_adapters import Snow17JAXComponent
            comp = Snow17JAXComponent("snow17")
            assert comp.name == "snow17"
            assert comp.gradient_method == GradientMethod.AUTOGRAD
            assert len(comp.parameters) == 10
            assert comp.state_size == 6
            assert comp.input_fluxes[0].name == "precip"
            assert comp.output_fluxes[0].name == "rain_plus_melt"
        except ImportError:
            pytest.skip("Snow-17 model not available")

    def test_xaj_adapter_creation(self):
        try:
            from symfluence.coupling.adapters.jax_adapters import XAJJAXComponent
            comp = XAJJAXComponent("xaj")
            assert comp.name == "xaj"
            assert len(comp.parameters) == 15
            assert comp.state_size == 7
        except ImportError:
            pytest.skip("XAJ model not available")

    def test_sacsma_adapter_creation(self):
        try:
            from symfluence.coupling.adapters.jax_adapters import SacSmaJAXComponent
            comp = SacSmaJAXComponent("sacsma")
            assert comp.name == "sacsma"
            assert len(comp.parameters) == 16
            assert comp.state_size == 6
        except ImportError:
            pytest.skip("SAC-SMA model not available")

    def test_snow17_physical_params(self):
        try:
            from symfluence.coupling.adapters.jax_adapters import Snow17JAXComponent
            comp = Snow17JAXComponent()
            params = comp.get_physical_parameters()
            assert len(params) == 10
            # All params should be within their bounds
            for pname, lo, hi in comp.SNOW17_PARAMS:
                assert lo <= params[pname].item() <= hi, f"{pname} out of bounds"
        except ImportError:
            pytest.skip("Snow-17 model not available")


def _physical_params(comp) -> dict:
    return {k: float(v.detach().reshape(-1)[0]) for k, v in comp.get_physical_parameters().items()}


def _run_adapter(comp, n: int, precip: float, temp: float, pet: float):
    state = comp.get_initial_state()
    inputs = {
        "precip": torch.tensor(precip),
        "temp": torch.tensor(temp),
        "pet": torch.tensor(pet),
    }
    runoff = []
    for _ in range(n):
        out, state = comp.step(inputs, state, 86400.0)
        runoff.append(float(out["runoff"].reshape(-1)[0]))
    return runoff, state


class TestJAXAdapterStepParity:
    """The HBV and TOPMODEL adapters must step and match the plugins' own simulate().

    Regression: HBV called get_routing_buffer_length(lag_days=...) and raised
    TypeError on construction; TOPMODEL passed ti_mean/n_classes to
    generate_ti_distribution and packed per-TI-class vectors into 5 scalar slots.
    """

    @pytest.fixture(autouse=True)
    def check_jax(self):
        pytest.importorskip("jax")

    def test_hbv_matches_plugin_simulate(self):
        np = pytest.importorskip("numpy")
        jhbv_model = pytest.importorskip("jhbv.model")
        from symfluence.coupling.adapters.jax_adapters import HBVJAXComponent

        comp = HBVJAXComponent()
        n = 40
        runoff, state = _run_adapter(comp, n, precip=5.0, temp=10.0, pet=2.0)
        assert state.shape == (comp.state_size,)

        init = jhbv_model.create_initial_state(
            initial_snow=0.0, initial_sm=0.0, initial_suz=0.0, initial_slz=0.0,
            use_jax=False,
        )
        q, _ = jhbv_model.simulate(
            np.full(n, 5.0), np.full(n, 10.0), np.full(n, 2.0),
            params=_physical_params(comp), initial_state=init,
            warmup_days=0, use_jax=False,
        )
        np.testing.assert_allclose(runoff, np.asarray(q), rtol=1e-4, atol=1e-5)

    def test_topmodel_matches_plugin_simulate(self):
        np = pytest.importorskip("numpy")
        jtop_model = pytest.importorskip("jtopmodel.model")
        from symfluence.coupling.adapters.jax_adapters import TopmodelJAXComponent

        comp = TopmodelJAXComponent()
        n = 40
        runoff, state = _run_adapter(comp, n, precip=5.0, temp=10.0, pet=2.0)
        # State must not grow from step to step (it did with (1,)-shaped params).
        assert state.shape == (comp.state_size,)

        q, _ = jtop_model.simulate(
            np.full(n, 5.0), np.full(n, 10.0), np.full(n, 2.0),
            params=_physical_params(comp), warmup_days=0, use_jax=False,
        )
        np.testing.assert_allclose(runoff, np.asarray(q), rtol=1e-4, atol=1e-4)
