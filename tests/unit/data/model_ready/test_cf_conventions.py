"""Tests for CF convention helpers."""

import pytest

from symfluence.data.model_ready.cf_conventions import (
    CF_STANDARD_NAMES,
    build_global_attrs,
)


class TestCFStandardNames:
    """Tests for the CF standard name mapping dict."""

    def test_forcing_vars_present(self):
        for var in ('air_temperature', 'precipitation_flux', 'wind_speed', 'specific_humidity', 'surface_downwelling_shortwave_flux', 'surface_downwelling_longwave_flux', 'surface_air_pressure'):
            assert var in CF_STANDARD_NAMES, f"Missing forcing variable: {var}"

    def test_observation_vars_present(self):
        for var in ('discharge_cms', 'swe', 'sca', 'et', 'soil_moisture'):
            assert var in CF_STANDARD_NAMES, f"Missing observation variable: {var}"

    def test_attribute_vars_present(self):
        for var in ('elev_mean', 'hru_area', 'latitude', 'longitude'):
            assert var in CF_STANDARD_NAMES, f"Missing attribute variable: {var}"

    def test_entries_have_required_keys(self):
        for name, attrs in CF_STANDARD_NAMES.items():
            assert 'standard_name' in attrs, f"{name} missing standard_name"
            assert 'units' in attrs, f"{name} missing units"
            assert 'long_name' in attrs, f"{name} missing long_name"


class TestBuildGlobalAttrs:
    """Tests for the global attribute builder."""

    def test_required_keys(self):
        attrs = build_global_attrs('test_domain', 'Test Dataset')
        assert attrs['Conventions'] == 'CF-1.8'
        assert attrs['domain_name'] == 'test_domain'
        assert attrs['title'] == 'Test Dataset'
        assert 'creation_date' in attrs
        assert 'source_software' in attrs

    def test_history_optional(self):
        attrs_no_hist = build_global_attrs('d', 't')
        assert 'history' not in attrs_no_hist

        attrs_hist = build_global_attrs('d', 't', history='created by test')
        assert attrs_hist['history'] == 'created by test'

    def test_creation_date_iso(self):
        attrs = build_global_attrs('d', 't')
        # Should be ISO 8601 with 'Z' suffix
        assert attrs['creation_date'].endswith('Z')
        assert 'T' in attrs['creation_date']
