"""Tests for FEWS PI Diagnostics collector."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from symfluence.fews.pi_diagnostics import DiagnosticsCollector


class TestDiagnosticsCollector:
    def test_all_levels(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.debug("debug msg")
        diag.info("info msg")
        diag.warning("warn msg")
        diag.error("error msg")
        diag.fatal("fatal msg")
        assert len(diag.messages) == 5
        # Delft-FEWS (pi_diag.xsd): 0=fatal, 1=error, 2=warn, 3=info, >3=debug
        assert [m.level for m in diag.messages] == ["4", "3", "2", "1", "0"]

    def test_fatal_is_written_as_level_zero_attribute(self, tmp_path):
        """A fatal error must reach FEWS as <line level="0" description=".."/>;
        it used to be written as level 4 (debug for FEWS) in child elements."""
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.info("started")
        diag.fatal("model crashed")
        diag.write()
        root = ET.parse(str(tmp_path / "diag.xml")).getroot()
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        lines = root.findall("pi:line", ns)
        assert [(ln.get("level"), ln.get("description")) for ln in lines] == [
            ("3", "started"), ("0", "model crashed"),
        ]
        assert all(len(ln) == 0 for ln in lines)  # no child elements

    def test_has_fatal(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        assert diag.has_fatal is False
        diag.info("ok")
        assert diag.has_fatal is False
        diag.fatal("boom")
        assert diag.has_fatal is True

    def test_has_errors(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        assert diag.has_errors is False
        diag.error("err")
        assert diag.has_errors is True

    def test_write_creates_valid_xml(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.info("test message")
        diag.error("error message")
        diag.write()

        # Parse the output
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        assert "Diag" in root.tag
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        lines = root.findall("pi:line", ns)
        assert len(lines) == 2

    def test_write_never_raises(self, tmp_path):
        # Point to a non-writable path
        diag = DiagnosticsCollector(tmp_path / "nonexistent_dir" / "deep" / "nested" / "diag.xml")
        diag.info("test")
        # write() should not raise even if directory creation works
        diag.write()

    def test_write_creates_parent_dirs(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "subdir" / "diag.xml")
        diag.info("test")
        diag.write()
        assert (tmp_path / "subdir" / "diag.xml").exists()

    def test_empty_diagnostics(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.write()
        assert (tmp_path / "diag.xml").exists()
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        assert len(root.findall("pi:line", ns)) == 0

    def test_message_text_preserved(self, tmp_path):
        diag = DiagnosticsCollector(tmp_path / "diag.xml")
        diag.info("Special chars: <>&\"'")
        diag.write()
        tree = ET.parse(str(tmp_path / "diag.xml"))
        root = tree.getroot()
        ns = {"pi": "http://www.wldelft.nl/fews/PI"}
        line = root.find("pi:line", ns)
        assert line is not None
        assert line.get("description") == "Special chars: <>&\"'"
