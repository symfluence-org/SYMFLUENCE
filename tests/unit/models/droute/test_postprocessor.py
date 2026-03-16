"""Tests for dRoute postprocessor."""

import numpy as np
import pytest


class TestDRoutePostProcessorImport:
    """Tests for dRoute postprocessor import and registration."""

    def test_postprocessor_can_be_imported(self):
        from droute.postprocessor import DRoutePostProcessor
        assert DRoutePostProcessor is not None

    def test_postprocessor_registered_with_registry(self):
        from symfluence.models.registry import ModelRegistry
        assert 'DROUTE' in ModelRegistry._postprocessors

    def test_postprocessor_is_correct_class(self):
        from droute.postprocessor import DRoutePostProcessor

        from symfluence.models.registry import ModelRegistry
        assert ModelRegistry._postprocessors.get('DROUTE') == DRoutePostProcessor

    def test_model_name(self):
        from droute.postprocessor import DRoutePostProcessor
        assert DRoutePostProcessor.model_name == "DROUTE"

    def test_streamflow_unit_is_cms(self):
        """dRoute outputs are already in m³/s."""
        from droute.postprocessor import DRoutePostProcessor
        assert DRoutePostProcessor.streamflow_unit == "cms"

    def test_streamflow_variable(self):
        from droute.postprocessor import DRoutePostProcessor
        assert DRoutePostProcessor.streamflow_variable == "outletStreamflow"


class TestDRouteNetCDFExtraction:
    """Tests for dRoute NetCDF extraction logic."""

    def test_outlet_streamflow_selection(self):
        """Test selecting outletStreamflow variable from NetCDF."""
        import pandas as pd
        import xarray as xr

        times = pd.date_range('2020-01-01', periods=10, freq='D')
        outlet_flow = np.random.uniform(10, 100, 10)

        ds = xr.Dataset({
            'outletStreamflow': ('time', outlet_flow),
            'routedRunoff': (['time', 'seg'], np.random.uniform(0, 50, (10, 5))),
        }, coords={'time': times, 'seg': range(5)})

        # outletStreamflow should be preferred
        assert 'outletStreamflow' in ds.data_vars
        series = ds['outletStreamflow'].to_series()
        assert len(series) == 10

    def test_routed_runoff_outlet_selection(self):
        """Test selecting outlet from routedRunoff using max mean discharge."""
        import pandas as pd
        import xarray as xr

        times = pd.date_range('2020-01-01', periods=10, freq='D')
        routed = np.random.uniform(0, 10, (10, 5))
        # Make segment 3 the outlet (highest mean)
        routed[:, 3] = 100.0

        ds = xr.Dataset({
            'routedRunoff': (['time', 'seg'], routed),
        }, coords={'time': times, 'seg': range(5)})

        mean_flow = ds['routedRunoff'].mean(dim='time')
        outlet_idx = int(mean_flow.argmax().values)
        assert outlet_idx == 3
