# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
FEWS run_info.xml parser.

Parses the run file that FEWS writes into the module working directory
before invoking the General Adapter (``exportRunFileActivity``). Extracts the
simulation time window, paths, and properties needed to drive a SYMFLUENCE
model run.

The published schema is ``pi_run.xsd`` (namespace
``http://www.wldelft.nl/fews/PI``): date-times are ``date``/``time``
*attributes* interpreted in the file's ``<timeZone>`` (decimal hours from
GMT), files are listed individually (``inputTimeSeriesFile``,
``inputNetcdfFile``, ``outputDiagnosticFile``, ...) and properties are
``<string key=".." value=".."/>`` style elements. The parser is deliberately
tolerant: element lookup ignores the namespace, date-times may also be given
as element text, and the non-schema ``inputDir`` / ``outputDir`` /
``stateInputDir`` / ``stateOutputDir`` / ``timeStep`` elements used by
hand-written run files keep working.
"""
from __future__ import annotations

# Security rationale: parsing trusted FEWS run_info.xml
import xml.etree.ElementTree as ET  # nosec B405
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .exceptions import RunInfoParseError

# Text-form datetimes: "2023-01-15T00:00:00Z" or "2023-01-15 00:00:00"
_FEWS_DT_FORMATS = [
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
]


def local_name(tag: str) -> str:
    """Element tag without its ``{namespace}`` prefix."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def parse_time_zone(root: ET.Element, error: type = RunInfoParseError) -> float:
    """Read the PI ``TimeZoneChoice`` of a document root: hours east of GMT.

    ``<timeZone>`` is a decimal hour shift from GMT (``1.0``, ``-3.5``);
    absent means GMT. ``<daylightSavingObservingTimeZone>`` names a zone with
    daylight saving, which cannot be converted with a fixed offset and is
    refused rather than read as GMT.
    """
    for child in root:
        name = local_name(child.tag)
        if name == "daylightSavingObservingTimeZone":
            raise error(
                "daylightSavingObservingTimeZone "
                f"('{(child.text or '').strip()}') is not supported; export from "
                "FEWS with a fixed <timeZone> offset (e.g. GMT)."
            )
        if name == "timeZone":
            text = (child.text or "").strip()
            if not text:
                return 0.0
            try:
                return float(text)
            except ValueError:
                raise error(f"Invalid <timeZone> value: '{text}'") from None
    return 0.0


def _parse_fews_datetime(text: str, time_zone_hours: float = 0.0) -> datetime:
    """Parse a text-form FEWS datetime into a timezone-aware UTC datetime.

    A trailing ``Z`` marks the value as UTC; otherwise it is local time in the
    file's time zone.
    """
    text = text.strip()
    for fmt in _FEWS_DT_FORMATS:
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if not fmt.endswith("Z"):
            dt -= timedelta(hours=time_zone_hours)
        return dt.replace(tzinfo=timezone.utc)
    raise RunInfoParseError(f"Cannot parse FEWS datetime: '{text}'")


def _parse_date_time_attrs(date: str, time: Optional[str], time_zone_hours: float) -> datetime:
    """Parse PI ``date``/``time`` attributes (local to the file's time zone)."""
    stamp = f"{date.strip()}T{(time or '00:00:00').strip()}"
    try:
        dt = datetime.fromisoformat(stamp)
    except ValueError:
        raise RunInfoParseError(f"Cannot parse FEWS date/time: '{stamp}'") from None
    dt -= timedelta(hours=time_zone_hours)
    return dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class RunInfo:
    """Parsed contents of a FEWS run_info.xml file.

    All date-times are timezone-aware UTC; ``time_zone_hours`` records the
    offset the run file used so output can be written back in the same zone.
    """

    work_dir: Path
    input_dir: Path
    output_dir: Path
    state_input_dir: Optional[Path]
    state_output_dir: Optional[Path]
    start_time: datetime
    end_time: datetime
    time_zero: Optional[datetime]
    time_step_seconds: Optional[int]
    properties: Dict[str, str] = field(default_factory=dict)
    time_zone_hours: float = 0.0
    input_timeseries_files: Tuple[Path, ...] = ()
    input_netcdf_files: Tuple[Path, ...] = ()
    output_timeseries_files: Tuple[Path, ...] = ()
    output_netcdf_files: Tuple[Path, ...] = ()
    output_diagnostic_file: Optional[Path] = None

    def to_config_overrides(self) -> Dict[str, Any]:
        """Convert run info into config overrides for SymfluenceConfig.

        Returns flat-key overrides that can be passed to SymfluenceConfig.from_file().
        """
        overrides: Dict[str, Any] = {
            "EXPERIMENT_TIME_START": self.start_time.strftime("%Y-%m-%d %H:%M"),
            "EXPERIMENT_TIME_END": self.end_time.strftime("%Y-%m-%d %H:%M"),
        }
        if "DOMAIN_NAME" in self.properties:
            overrides["DOMAIN_NAME"] = self.properties["DOMAIN_NAME"]
        if "EXPERIMENT_ID" in self.properties:
            overrides["EXPERIMENT_ID"] = self.properties["EXPERIMENT_ID"]
        if "HYDROLOGICAL_MODEL" in self.properties:
            overrides["HYDROLOGICAL_MODEL"] = self.properties["HYDROLOGICAL_MODEL"]
        return overrides


def parse_run_info(path: Path) -> RunInfo:
    """Parse a FEWS run_info.xml file.

    Args:
        path: Path to run_info.xml

    Returns:
        RunInfo dataclass with all extracted fields

    Raises:
        RunInfoParseError: If the file cannot be read or is malformed
    """
    path = Path(path)
    if not path.is_file():
        raise RunInfoParseError(f"run_info.xml not found: {path}")

    try:
        tree = ET.parse(str(path))  # nosec B314
    except ET.ParseError as exc:
        raise RunInfoParseError(f"Malformed XML in {path}: {exc}") from exc

    root = tree.getroot()

    # Namespace-agnostic child lookup (real run files are in the PI namespace)
    children: Dict[str, List[ET.Element]] = {}
    for child in root:
        children.setdefault(local_name(child.tag), []).append(child)

    def _elem(tag: str, required: bool = True) -> Optional[ET.Element]:
        found = children.get(tag)
        if not found:
            if required:
                raise RunInfoParseError(f"Missing required element <{tag}> in {path}")
            return None
        return found[0]

    def _text(tag: str, required: bool = True) -> Optional[str]:
        elem = _elem(tag, required=required)
        if elem is None or elem.text is None or not elem.text.strip():
            if required:
                raise RunInfoParseError(f"Missing required element <{tag}> in {path}")
            return None
        return elem.text.strip()

    def _path_elem(tag: str, required: bool = True) -> Optional[Path]:
        text = _text(tag, required=required)
        if text is None:
            return None
        return Path(text)

    def _paths(tag: str) -> Tuple[Path, ...]:
        return tuple(
            Path(e.text.strip()) for e in children.get(tag, [])
            if e.text and e.text.strip()
        )

    time_zone_hours = parse_time_zone(root)

    def _datetime(tag: str, required: bool = True) -> Optional[datetime]:
        elem = _elem(tag, required=required)
        if elem is None:
            return None
        if elem.get("date"):
            # pi_run.xsd form: <startDateTime date="2023-01-01" time="00:00:00"/>
            return _parse_date_time_attrs(elem.get("date", ""), elem.get("time"), time_zone_hours)
        if elem.text and elem.text.strip():
            return _parse_fews_datetime(elem.text, time_zone_hours)
        if required:
            raise RunInfoParseError(
                f"<{tag}> in {path} has neither date/time attributes nor text"
            )
        return None

    work_dir = _path_elem("workDir") or path.parent
    input_dir = _path_elem("inputDir", required=False)
    if input_dir is None:
        input_dir = work_dir / "toModel"
    output_dir = _path_elem("outputDir", required=False)
    if output_dir is None:
        output_dir = work_dir / "toFews"

    start_time = _datetime("startDateTime", required=True)
    end_time = _datetime("endDateTime", required=True)
    assert start_time is not None and end_time is not None  # required=True
    if end_time < start_time:
        raise RunInfoParseError(
            f"endDateTime ({end_time}) is before startDateTime ({start_time}) in {path}"
        )
    time_zero = _datetime("time0", required=False)

    ts_text = _text("timeStep", required=False)
    try:
        time_step_seconds = int(ts_text) if ts_text else None
    except ValueError:
        raise RunInfoParseError(f"Invalid <timeStep> '{ts_text}' in {path}") from None

    state_input_dir = _path_elem("stateInputDir", required=False)
    state_output_dir = _path_elem("stateOutputDir", required=False)

    # <properties>: pi_sharedtypes.xsd uses <string|int|float|double|bool
    # key=".." value=".."/> and <dateTime key=".." date=".." time=".."/>;
    # hand-written files also use <KEY>value</KEY>.
    properties: Dict[str, str] = {}
    props_elem = _elem("properties", required=False)
    if props_elem is not None:
        for prop in props_elem:
            if "key" in prop.attrib and "value" in prop.attrib:
                properties[prop.attrib["key"]] = prop.attrib["value"]
            elif "key" in prop.attrib and "date" in prop.attrib:
                properties[prop.attrib["key"]] = (
                    f"{prop.attrib['date']}T{prop.attrib.get('time', '00:00:00')}"
                )
            elif prop.text is not None and local_name(prop.tag) != "description":
                properties[local_name(prop.tag)] = prop.text.strip()

    return RunInfo(
        work_dir=work_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        state_input_dir=state_input_dir,
        state_output_dir=state_output_dir,
        start_time=start_time,
        end_time=end_time,
        time_zero=time_zero,
        time_step_seconds=time_step_seconds,
        properties=properties,
        time_zone_hours=time_zone_hours,
        input_timeseries_files=_paths("inputTimeSeriesFile"),
        input_netcdf_files=_paths("inputNetcdfFile"),
        output_timeseries_files=_paths("outputTimeSeriesFile"),
        output_netcdf_files=_paths("outputNetcdfFile"),
        output_diagnostic_file=_path_elem("outputDiagnosticFile", required=False),
    )
