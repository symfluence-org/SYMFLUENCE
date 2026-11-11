"""Tests for FEWS run_info.xml parser."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from symfluence.fews.exceptions import RunInfoParseError
from symfluence.fews.run_info import RunInfo, parse_run_info


class TestParseRunInfo:
    def test_basic_parse(self, sample_run_info_xml, tmp_work_dir):
        ri = parse_run_info(sample_run_info_xml)
        assert isinstance(ri, RunInfo)
        assert ri.work_dir == tmp_work_dir
        assert ri.input_dir == tmp_work_dir / "toModel"
        assert ri.output_dir == tmp_work_dir / "toFews"

    def test_datetime_parsing(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.start_time == datetime(2023, 1, 1, tzinfo=timezone.utc)
        assert ri.end_time == datetime(2023, 1, 10, tzinfo=timezone.utc)

    def test_time_zero(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.time_zero == datetime(2023, 1, 5, tzinfo=timezone.utc)

    def test_time_step(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.time_step_seconds == 3600

    def test_properties(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.properties["DOMAIN_NAME"] == "test_basin"
        assert ri.properties["EXPERIMENT_ID"] == "fews_run_1"
        assert ri.properties["HYDROLOGICAL_MODEL"] == "GR"

    def test_config_overrides(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        overrides = ri.to_config_overrides()
        assert overrides["EXPERIMENT_TIME_START"] == "2023-01-01 00:00"
        assert overrides["EXPERIMENT_TIME_END"] == "2023-01-10 00:00"
        assert overrides["DOMAIN_NAME"] == "test_basin"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(RunInfoParseError, match="not found"):
            parse_run_info(tmp_path / "nonexistent.xml")

    def test_malformed_xml(self, tmp_path):
        bad_file = tmp_path / "bad.xml"
        bad_file.write_text("<broken><xml")
        with pytest.raises(RunInfoParseError, match="Malformed XML"):
            parse_run_info(bad_file)

    def test_missing_start_datetime(self, tmp_path):
        xml = f"<Run><workDir>{tmp_path}</workDir><endDateTime>2023-01-10T00:00:00Z</endDateTime></Run>"
        path = tmp_path / "run_info.xml"
        path.write_text(xml)
        with pytest.raises(RunInfoParseError, match="startDateTime"):
            parse_run_info(path)

    def test_state_dirs(self, sample_run_info_xml, tmp_work_dir):
        ri = parse_run_info(sample_run_info_xml)
        assert ri.state_input_dir == tmp_work_dir / "states_in"
        assert ri.state_output_dir == tmp_work_dir / "states_out"

    def test_frozen_dataclass(self, sample_run_info_xml):
        ri = parse_run_info(sample_run_info_xml)
        with pytest.raises(AttributeError):
            ri.start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)


class TestRealPIRunFile:
    """Run files as Delft-FEWS writes them (pi_run.xsd), which the parser used
    to reject outright: namespaced elements and date/time attributes."""

    def test_parses_namespaced_attribute_form(self, real_run_info_xml, tmp_work_dir):
        from datetime import datetime, timezone

        from symfluence.fews.run_info import parse_run_info

        info = parse_run_info(real_run_info_xml)
        assert info.time_zone_hours == 1.0
        # 00:00 at GMT+1 is 23:00 UTC the day before
        assert info.start_time == datetime(2022, 12, 31, 23, 0, tzinfo=timezone.utc)
        assert info.end_time == datetime(2023, 1, 2, 23, 0, tzinfo=timezone.utc)
        assert info.time_zero == datetime(2023, 1, 1, 23, 0, tzinfo=timezone.utc)
        assert info.work_dir == tmp_work_dir
        # no inputDir/outputDir in the real schema: defaults under workDir
        assert info.input_dir == tmp_work_dir / "toModel"
        assert info.output_dir == tmp_work_dir / "toFews"

    def test_file_lists_and_diagnostic_file(self, real_run_info_xml, tmp_work_dir):
        from symfluence.fews.run_info import parse_run_info

        info = parse_run_info(real_run_info_xml)
        assert info.input_timeseries_files == (tmp_work_dir / "toModel" / "in.xml",)
        assert info.input_netcdf_files == (tmp_work_dir / "toModel" / "met.nc",)
        assert info.output_timeseries_files == (tmp_work_dir / "fromModel" / "q.xml",)
        assert info.output_netcdf_files == (tmp_work_dir / "fromModel" / "q.nc",)
        assert info.output_diagnostic_file == tmp_work_dir / "diagnostics" / "diag.xml"

    def test_typed_properties(self, real_run_info_xml):
        from symfluence.fews.run_info import parse_run_info

        info = parse_run_info(real_run_info_xml)
        assert info.properties == {
            "DOMAIN_NAME": "test_basin", "N_MEMBERS": "5", "ISSUE": "2023-01-02T06:00:00",
        }
        assert info.to_config_overrides()["DOMAIN_NAME"] == "test_basin"
        assert info.to_config_overrides()["EXPERIMENT_TIME_START"] == "2022-12-31 23:00"

    def test_text_form_with_time_zone(self, tmp_path):
        from datetime import datetime, timezone

        from symfluence.fews.run_info import parse_run_info

        path = tmp_path / "run_info.xml"
        path.write_text(
            "<Run><timeZone>-2.0</timeZone><workDir>.</workDir>"
            "<startDateTime>2023-01-01 00:00:00</startDateTime>"
            "<endDateTime>2023-01-02T00:00:00Z</endDateTime></Run>"
        )
        info = parse_run_info(path)
        assert info.start_time == datetime(2023, 1, 1, 2, 0, tzinfo=timezone.utc)
        # an explicit Z is UTC whatever the file's zone
        assert info.end_time == datetime(2023, 1, 2, 0, 0, tzinfo=timezone.utc)

    def test_daylight_saving_zone_is_refused(self, tmp_path):
        from symfluence.fews.exceptions import RunInfoParseError
        from symfluence.fews.run_info import parse_run_info

        path = tmp_path / "run_info.xml"
        path.write_text(
            '<Run xmlns="http://www.wldelft.nl/fews/PI">'
            "<daylightSavingObservingTimeZone>CET</daylightSavingObservingTimeZone>"
            '<startDateTime date="2023-01-01" time="00:00:00"/>'
            '<endDateTime date="2023-01-02" time="00:00:00"/><workDir>.</workDir></Run>'
        )
        with pytest.raises(RunInfoParseError, match="daylightSavingObservingTimeZone"):
            parse_run_info(path)

