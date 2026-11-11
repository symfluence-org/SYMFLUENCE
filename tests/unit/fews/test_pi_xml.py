"""Tests for FEWS PI-XML timeseries reader and writer."""
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from symfluence.fews.exceptions import PIXMLError
from symfluence.fews.pi_xml import read_pi_xml_timeseries, write_pi_xml_timeseries


class TestReadPIXML:
    def test_read_basic(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert "P.obs" in ds.data_vars
        assert "T.obs" in ds.data_vars
        assert "time" in ds.coords

    def test_read_timesteps(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert len(ds.time) == 4

    def test_missing_values_become_nan(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml, missing_value=-999.0)
        # Third event is -999.0 -> should be NaN
        assert np.isnan(ds["P.obs"].values[2])

    def test_location_id_in_attrs(self, sample_pi_xml):
        ds = read_pi_xml_timeseries(sample_pi_xml)
        assert ds["P.obs"].attrs["location_id"] == "station_01"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(PIXMLError, match="not found"):
            read_pi_xml_timeseries(tmp_path / "nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        path = tmp_path / "bad.xml"
        path.write_text("<broken>")
        with pytest.raises(PIXMLError, match="Malformed"):
            read_pi_xml_timeseries(path)

    def test_no_series(self, tmp_path):
        path = tmp_path / "empty.xml"
        path.write_text('<?xml version="1.0"?><TimeSeries></TimeSeries>')
        with pytest.raises(PIXMLError, match="No <series>"):
            read_pi_xml_timeseries(path)


class TestWritePIXML:
    def test_write_basic(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-01T04:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"precip": ("time", [1.0, 2.0, 3.0, 4.0])},
            coords={"time": times},
        )
        path = tmp_path / "output.xml"
        write_pi_xml_timeseries(ds, path)
        assert path.exists()

    def test_round_trip(self, tmp_path):
        times = np.arange("2023-06-01", "2023-06-01T03:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {
                "Q": ("time", [10.0, 20.0, 30.0]),
                "H": ("time", [1.5, 2.0, 2.5]),
            },
            coords={"time": times},
        )
        path = tmp_path / "roundtrip.xml"
        write_pi_xml_timeseries(ds, path, location_id="station_99")

        ds2 = read_pi_xml_timeseries(path)
        assert "Q" in ds2.data_vars
        assert "H" in ds2.data_vars
        np.testing.assert_allclose(ds2["Q"].values, [10.0, 20.0, 30.0], rtol=1e-5)

    def test_nan_becomes_missing(self, tmp_path):
        times = np.arange("2023-01-01", "2023-01-01T02:00:00", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"Q": ("time", [1.0, np.nan])},
            coords={"time": times},
        )
        path = tmp_path / "nan.xml"
        write_pi_xml_timeseries(ds, path, missing_value=-999.0)

        ds2 = read_pi_xml_timeseries(path, missing_value=-999.0)
        assert np.isnan(ds2["Q"].values[1])


MULTI_XML = """<?xml version="1.0" encoding="UTF-8"?>
<TimeSeries xmlns="http://www.wldelft.nl/fews/PI" version="1.2">
  <timeZone>1.0</timeZone>
  <series><header><type>accumulative</type><locationId>A</locationId>
    <parameterId>P.obs</parameterId><timeStep unit="second" multiplier="3600"/>
    <startDate date="2023-01-01" time="00:00:00"/><endDate date="2023-01-01" time="01:00:00"/>
    <missVal>NaN</missVal><units>mm</units></header>
    <event date="2023-01-01" time="00:00:00" value="1.0" flag="0"/>
    <event date="2023-01-01" time="01:00:00" value="2.0" flag="0"/></series>
  <series><header><type>accumulative</type><locationId>B</locationId>
    <parameterId>P.obs</parameterId><timeStep unit="second" multiplier="3600"/>
    <startDate date="2023-01-01" time="00:00:00"/><endDate date="2023-01-01" time="01:00:00"/>
    <missVal>-9999</missVal><units>mm</units></header>
    <event date="2023-01-01" time="00:00:00" value="50.0" flag="0"/>
    <event date="2023-01-01" time="01:00:00" value="-9999" flag="9"/></series>
</TimeSeries>
"""


class TestPIXMLConventions:
    def _multi(self, tmp_path):
        path = tmp_path / "in.xml"
        path.write_text(MULTI_XML)
        return read_pi_xml_timeseries(path)

    def test_locations_are_not_collapsed(self, tmp_path):
        """Series are keyed by (parameterId, locationId): location A used to be
        silently replaced by location B."""
        ds = self._multi(tmp_path)
        assert ds["P.obs"].dims == ("time", "location")
        assert list(ds["location"].values) == ["A", "B"]
        np.testing.assert_allclose(ds["P.obs"].sel(location="A").values, [1.0, 2.0])
        assert ds["P.obs"].sel(location="B").values[0] == 50.0

    def test_header_missval_is_honoured(self, tmp_path):
        """-9999 is this series' missVal; it was kept as data because only the
        function's default (-999) was checked."""
        ds = self._multi(tmp_path)
        assert np.isnan(ds["P.obs"].sel(location="B").values[1])

    def test_time_zone_is_applied(self, tmp_path):
        ds = self._multi(tmp_path)
        # 00:00 at GMT+1 -> 23:00 UTC the day before
        assert ds["time"].values[0] == np.datetime64("2022-12-31T23:00:00")

    def test_units_and_type_are_kept(self, tmp_path):
        ds = self._multi(tmp_path)
        assert ds["P.obs"].attrs["units"] == "mm"
        assert ds["P.obs"].attrs["fews_type"] == "accumulative"

    def test_duplicate_series_is_refused(self, tmp_path):
        path = tmp_path / "dup.xml"
        path.write_text(MULTI_XML.replace("<locationId>B</locationId>",
                                          "<locationId>A</locationId>"))
        with pytest.raises(PIXMLError, match="share parameterId"):
            read_pi_xml_timeseries(path)

    def test_writer_header_is_complete(self, tmp_path):
        import xml.etree.ElementTree as ET

        times = np.arange("2023-01-01T00", "2023-01-01T03", dtype="datetime64[h]")
        ds = xr.Dataset(
            {"Q.sim": ("time", [1.0, np.nan, 3.0], {"units": "m3/s"}),
             "nounit": ("time", [1.0, 2.0, 3.0])},
            coords={"time": times},
        )
        out = tmp_path / "out.xml"
        write_pi_xml_timeseries(ds, out, location_id="gauge1", time_zone=1.0)

        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        root = ET.parse(str(out)).getroot()
        assert root.find("pi:timeZone", ns).text == "1.0"
        headers = root.findall("pi:series/pi:header", ns)
        header = headers[0]
        assert [child.tag.split("}")[1] for child in header] == [
            "type", "locationId", "parameterId", "timeStep", "startDate", "endDate",
            "missVal", "units",
        ]
        assert header.find("pi:locationId", ns).text == "gauge1"
        assert header.find("pi:timeStep", ns).attrib == {"unit": "second", "multiplier": "3600"}
        # written in the requested zone: 00:00 UTC is 01:00 at GMT+1
        assert header.find("pi:startDate", ns).attrib == {"date": "2023-01-01", "time": "01:00:00"}
        assert header.find("pi:endDate", ns).attrib == {"date": "2023-01-01", "time": "03:00:00"}
        assert header.find("pi:units", ns).text == "m3/s"
        # no invented "-" unit
        assert headers[1].find("pi:units", ns) is None

    def test_round_trip_with_time_zone_and_locations(self, tmp_path):
        ds = self._multi(tmp_path)
        out = tmp_path / "rt.xml"
        write_pi_xml_timeseries(ds, out, missing_value=-999.0, time_zone=-3.5)
        back = read_pi_xml_timeseries(out)
        np.testing.assert_array_equal(back["time"].values, ds["time"].values)
        np.testing.assert_allclose(back["P.obs"].values, ds["P.obs"].values, equal_nan=True)

