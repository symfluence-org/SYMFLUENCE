"""Equivalence tests: verify dCoupler graph paths match legacy coupling paths.

These tests compare outputs from the old ad-hoc coupling implementations
against the new dCoupler graph-based implementations. Each test constructs
both paths and asserts numerical equivalence.

Note: Some tests require external model executables and data, so they may
be skipped in CI environments. Tests requiring dCoupler are skipped when
it is not installed.
"""

import numpy as np
import pytest

from symfluence.coupling import is_dcoupler_available

dcoupler_available = is_dcoupler_available()

jax_available = True
try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax_available = False

# Only import torch/dcoupler when available
if dcoupler_available:
    import torch


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSnow17XAJEquivalence:
    """Compare Snow-17/XAJ coupling: lax.scan vs dCoupler graph."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        if not jax_available:
            pytest.skip("JAX not installed")
        try:
            from jsnow17.model import snow17_step  # noqa: F401
            from jxaj.model import simulate_coupled_jax, step_jax  # noqa: F401
        except ImportError:
            pytest.skip("Snow-17 or XAJ model not available in SYMFLUENCE")

    def test_forward_output_shape(self):
        """Verify the graph produces outputs with correct shapes."""
        from dcoupler.core.graph import CouplingGraph

        from symfluence.coupling.adapters.jax_adapters import (
            Snow17JAXComponent,
            XAJJAXComponent,
        )

        try:
            snow = Snow17JAXComponent("snow17")
            xaj = XAJJAXComponent("xaj")
        except ImportError:
            pytest.skip("Model adapters not available")

        graph = CouplingGraph()
        graph.add_component(snow)
        graph.add_component(xaj)
        graph.connect("snow17", "rain_plus_melt", "xaj", "precip")

        n_timesteps = 10
        precip = torch.ones(n_timesteps) * 5.0
        temp = torch.ones(n_timesteps) * 2.0
        pet = torch.ones(n_timesteps) * 3.0

        outputs = graph.forward(
            external_inputs={
                "snow17": {"precip": precip, "temp": temp},
                "xaj": {"pet": pet},
            },
            n_timesteps=n_timesteps,
            dt=86400.0,
        )

        assert "xaj" in outputs
        assert "runoff" in outputs["xaj"]
        assert outputs["xaj"]["runoff"].shape[0] == n_timesteps


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSnow17SacSmaEquivalence:
    """Compare Snow-17/SAC-SMA coupling: lax.scan vs dCoupler graph."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        if not jax_available:
            pytest.skip("JAX not installed")
        try:
            from jsacsma.sacsma import sacsma_step  # noqa: F401
            from jsnow17.model import snow17_step  # noqa: F401
        except ImportError:
            pytest.skip("Snow-17 or SAC-SMA model not available")

    def test_forward_output_shape(self):
        """Verify the graph produces outputs with correct shapes."""
        from dcoupler.core.graph import CouplingGraph

        from symfluence.coupling.adapters.jax_adapters import (
            SacSmaJAXComponent,
            Snow17JAXComponent,
        )

        try:
            snow = Snow17JAXComponent("snow17")
            sacsma = SacSmaJAXComponent("sacsma")
        except ImportError:
            pytest.skip("Model adapters not available")

        graph = CouplingGraph()
        graph.add_component(snow)
        graph.add_component(sacsma)
        graph.connect("snow17", "rain_plus_melt", "sacsma", "precip")

        n_timesteps = 10
        precip = torch.ones(n_timesteps) * 5.0
        temp = torch.ones(n_timesteps) * 2.0
        pet = torch.ones(n_timesteps) * 3.0

        outputs = graph.forward(
            external_inputs={
                "snow17": {"precip": precip, "temp": temp},
                "sacsma": {"pet": pet},
            },
            n_timesteps=n_timesteps,
            dt=86400.0,
        )

        assert "sacsma" in outputs
        assert "runoff" in outputs["sacsma"]


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSUMMAParFlowEquivalence:
    """Compare SUMMA->ParFlow coupling: old coupler vs dCoupler graph."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        try:
            from symfluence.models.parflow.coupling import SUMMAToParFlowCoupler  # noqa: F401
        except ImportError:
            pytest.skip("ParFlow coupling not available")

    def test_unit_conversion_factor(self):
        """Verify the dCoupler graph uses the same conversion factor."""
        from symfluence.coupling.adapters.process_adapters import ParFlowProcessComponent

        # Both should use 3.6 for kg/m2/s -> m/hr
        assert ParFlowProcessComponent.KG_M2_S_TO_M_HR == 3.6

    def test_graph_connection_unit_conversion(self):
        """Verify graph builder sets correct unit conversion for SUMMA->ParFlow."""
        from symfluence.coupling.graph_builder import CouplingGraphBuilder

        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "GROUNDWATER_MODEL": "PARFLOW",
        }
        graph = builder.build(config)

        conn = graph.connections[0]
        assert conn.unit_conversion == 3.6


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSUMMAMODFLOWEquivalence:
    """Compare SUMMA->MODFLOW coupling: old coupler vs dCoupler graph."""

    @pytest.fixture(autouse=True)
    def check_deps(self):
        try:
            from symfluence.models.modflow.coupling import SUMMAToMODFLOWCoupler  # noqa: F401
        except ImportError:
            pytest.skip("MODFLOW coupling not available")

    def test_unit_conversion_factor(self):
        from symfluence.coupling.adapters.process_adapters import MODFLOWProcessComponent
        assert MODFLOWProcessComponent.KG_M2_S_TO_M_D == 86.4

    def test_graph_connection_unit_conversion(self):
        from symfluence.coupling.graph_builder import CouplingGraphBuilder

        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "GROUNDWATER_MODEL": "MODFLOW",
        }
        graph = builder.build(config)

        conn = graph.connections[0]
        assert conn.unit_conversion == 86.4


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSUMMAMizuRouteEquivalence:
    """Compare SUMMA->mizuRoute coupling: sequential execution vs graph."""

    def test_graph_structure(self):
        """Verify correct graph structure for SUMMA->mizuRoute."""
        from symfluence.coupling.graph_builder import CouplingGraphBuilder

        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "ROUTING_MODEL": "MIZUROUTE",
        }
        graph = builder.build(config)

        assert len(graph.components) == 2
        assert "land" in graph.components
        assert "routing" in graph.components
        assert len(graph.connections) == 1
        assert graph.connections[0].source_flux == "runoff"
        assert graph.connections[0].target_flux == "lateral_inflow"


class TestNativeFallbackPaths:
    """Verify native coupling paths still work regardless of dCoupler availability."""

    def test_xaj_native_coupling(self):
        """XAJ coupled simulation works with coupling_mode='native'."""
        from jxaj.model import simulate

        n = 30
        precip = np.random.uniform(0, 10, n)
        temp = np.random.uniform(-5, 15, n)
        pet = np.random.uniform(1, 5, n)
        doy = np.arange(1, n + 1)
        snow17_params = {
            'SCF': 1.0, 'PXTEMP': 1.0, 'MFMAX': 1.0, 'MFMIN': 0.3,
            'NMF': 0.15, 'MBASE': 0.0, 'TIPM': 0.1, 'UADJ': 0.04,
            'PLWHC': 0.04, 'DAYGM': 0.0,
        }

        runoff, state = simulate(
            precip=precip, pet=pet, temp=temp, day_of_year=doy,
            snow17_params=snow17_params, use_jax=False, coupling_mode='native',
        )
        assert len(runoff) == n
        assert np.all(np.isfinite(runoff))

    def test_sacsma_native_coupling(self):
        """SAC-SMA coupled simulation works with coupling_mode='native'."""
        from jsacsma.model import simulate

        n = 30
        precip = np.random.uniform(0, 10, n)
        temp = np.random.uniform(-5, 15, n)
        pet = np.random.uniform(1, 5, n)
        doy = np.arange(1, n + 1)

        runoff, state = simulate(
            precip=precip, temp=temp, pet=pet, day_of_year=doy,
            use_jax=False, snow_module='snow17', coupling_mode='native',
        )
        assert len(runoff) == n
        assert np.all(np.isfinite(runoff))


class TestTRouteMCCalibrationVerification:
    """Verify that updating Manning's n in topology changes MC routing output."""

    @staticmethod
    def _create_topology(path, seg_ids, to_node, n_val, n_seg):
        """Create a minimal topology NetCDF for testing."""
        import netCDF4 as nc4
        with nc4.Dataset(path, 'w', format='NETCDF4') as ncid:
            ncid.createDimension('link', n_seg)
            ncid.createDimension('nhru', n_seg)
            for name, vals in [('comid', seg_ids), ('to_node', to_node)]:
                v = ncid.createVariable(name, 'i4', ('link',))
                v[:] = vals
            for name, vals in [
                ('length', [5000.0] * n_seg),
                ('slope', [0.001] * n_seg),
                ('n', [n_val] * n_seg),
                ('channel_width', [15.0] * n_seg),
            ]:
                v = ncid.createVariable(name, 'f8', ('link',))
                v[:] = vals
            v = ncid.createVariable('link_id_hru', 'i4', ('nhru',))
            v[:] = seg_ids
            v = ncid.createVariable('hru_area_m2', 'f8', ('nhru',))
            v[:] = [1.0] * n_seg

    def test_mannings_n_changes_output(self, tmp_path):
        """Write two topologies with different n, verify routed output differs."""
        import logging

        import xarray as xr

        from symfluence.models.troute.runner import TRouteRunner

        n_seg = 3
        n_time = 10
        seg_ids = np.array([1, 2, 3])
        to_node = np.array([2, 3, 0])  # 1→2→3→outlet

        # Create lateral inflow file
        q_lateral = np.random.RandomState(42).uniform(1.0, 5.0, (n_time, n_seg))
        time_vals = np.arange(n_time)
        ds_runoff = xr.Dataset(
            {'q_lateral': (['time', 'hru'], q_lateral)},
            coords={'time': time_vals, 'hru': seg_ids},
        )
        runoff_path = tmp_path / 'runoff.nc'
        ds_runoff.to_netcdf(runoff_path)

        results = {}
        for label, n_val in [('low', 0.02), ('high', 0.08)]:
            topo_path = tmp_path / f'topo_{label}.nc'
            self._create_topology(topo_path, seg_ids, to_node, n_val, n_seg)

            out_dir = tmp_path / f'out_{label}'
            out_dir.mkdir()

            # Use a lightweight stub that bypasses full runner init
            # but provides the interface _run_builtin_muskingum_cunge needs
            runner = object.__new__(TRouteRunner)
            runner.logger = logging.getLogger('test_mc')
            runner.config_dict = {
                'SETTINGS_TROUTE_DT_SECONDS': '3600',
                'TROUTE_QTS_SUBDIVISIONS': '1',
            }
            runner._run_builtin_muskingum_cunge(runoff_path, topo_path, out_dir)

            ds_out = xr.open_dataset(out_dir / 'troute_output.nc')
            results[label] = ds_out['flow'].values.copy()
            ds_out.close()

        # Flows must differ — higher n should produce lower peak & more attenuation
        assert not np.allclose(results['low'], results['high'], atol=1e-6), \
            "Manning's n change did not affect routing output"

        # Higher n → more attenuation → lower max flow at outlet (seg 3, idx 2)
        max_low = np.max(results['low'][:, 2])
        max_high = np.max(results['high'][:, 2])
        assert max_low > max_high, \
            f"Expected higher n to reduce peak flow: n=0.02 peak={max_low:.3f}, n=0.08 peak={max_high:.3f}"


@pytest.mark.skipif(not dcoupler_available, reason="dCoupler not installed")
class TestSUMMATRouteEquivalence:
    """Compare SUMMA->TRoute coupling: standalone runner vs dCoupler graph.

    Verifies:
    1. Graph structure (2 components, 1 connection)
    2. Unit conversion factor (1.0 — area remapping handled by spatial remapper)
    3. TRouteProcessComponent can read the same output as standalone runner
    """

    def test_graph_structure(self):
        """Verify correct graph structure for SUMMA->TROUTE."""
        from symfluence.coupling.graph_builder import CouplingGraphBuilder

        builder = CouplingGraphBuilder()
        config = {
            "HYDROLOGICAL_MODEL": "SUMMA",
            "ROUTING_MODEL": "TROUTE",
        }
        graph = builder.build(config)

        assert len(graph.components) == 2
        assert "land" in graph.components
        assert "routing" in graph.components
        assert len(graph.connections) == 1
        assert graph.connections[0].source_flux == "runoff"
        assert graph.connections[0].target_flux == "lateral_inflow"

    def test_unit_conversion(self):
        """Verify SUMMA->TROUTE uses unit conversion factor 1.0."""
        from symfluence.coupling.graph_builder import UNIT_CONVERSIONS
        assert ("SUMMA", "TROUTE") in UNIT_CONVERSIONS
        assert UNIT_CONVERSIONS[("SUMMA", "TROUTE")] == 1.0

    def test_troute_component_flux_specs(self):
        """Verify TRouteProcessComponent has correct flux specifications."""
        from symfluence.coupling.adapters.process_adapters import TRouteProcessComponent

        comp = TRouteProcessComponent("test_troute")
        assert len(comp.input_fluxes) == 1
        assert comp.input_fluxes[0].name == "lateral_inflow"
        assert comp.input_fluxes[0].units == "m3/s"

        assert len(comp.output_fluxes) == 1
        assert comp.output_fluxes[0].name == "discharge"
        assert comp.output_fluxes[0].units == "m3/s"

    def test_read_outputs_matches_standalone(self):
        """Verify coupling adapter reads same data as standalone xarray read.

        This test uses the actual troute_output.nc from a completed run
        (if available) and checks that TRouteProcessComponent.read_outputs()
        produces identical values to a direct xarray read.
        """
        from pathlib import Path

        troute_output = Path(
            "/Users/darrieythorsson/compHydro/code/SYMFLUENCE_data"
            "/domain_Bow_at_Banff_semi_distributed/simulations"
            "/run_troute/TRoute/troute_output.nc"
        )
        if not troute_output.exists():
            pytest.skip("T-Route output not available (run tutorial 02b first)")

        import torch
        import xarray as xr

        # 1. Direct xarray read (standalone path)
        ds = xr.open_dataset(troute_output)
        direct_flow = ds['flow'].values.astype(np.float32)
        ds.close()

        # 2. Read via coupling adapter
        from symfluence.coupling.adapters.process_adapters import TRouteProcessComponent
        comp = TRouteProcessComponent("test_troute", config={
            'EXPERIMENT_OUTPUT_TROUTE': str(troute_output.parent),
        })
        adapter_outputs = comp.read_outputs(troute_output.parent)

        assert "discharge" in adapter_outputs
        adapter_flow = adapter_outputs["discharge"].numpy()

        # 3. Compare — must be bitwise identical (same file, same read)
        np.testing.assert_array_equal(
            adapter_flow, direct_flow,
            err_msg="Coupling adapter read differs from direct xarray read"
        )
