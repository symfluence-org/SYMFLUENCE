# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
PI-XML timeseries reader and writer.

Implements read/write for the Delft-FEWS PI-XML timeseries format
(``pi_timeseries.xsd``) using only ``xml.etree.ElementTree`` (stdlib) and
``xarray``.

PI-XML namespace: ``http://www.wldelft.nl/fews/PI``

Conventions honoured on both sides:

* a series is identified by ``(parameterId, locationId)`` -- two locations of
  one parameter are two series, not one;
* each header's ``<missVal>`` marks that series' missing values;
* event times are local to the document's ``<timeZone>`` (decimal hours from
  GMT); datasets always carry UTC.
"""
from __future__ import annotations

# Security rationale: parsing trusted FEWS PI-XML files
import xml.etree.ElementTree as ET  # nosec B405
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

from .exceptions import PIXMLError
from .run_info import local_name, parse_time_zone

# FEWS PI namespace
_NS = "http://www.wldelft.nl/fews/PI"
_NS_MAP = {"pi": _NS}


def _tag(name: str) -> str:
    """Prepend the PI namespace to a tag name."""
    return f"{{{_NS}}}{name}"


def _strip_ns(tag: str) -> str:
    """Remove namespace prefix from tag."""
    return local_name(tag)


def _child(elem: ET.Element, name: str) -> Optional[ET.Element]:
    """First child with the given local name, whatever its namespace."""
    for child in elem:
        if local_name(child.tag) == name:
            return child
    return None


def _child_text(elem: ET.Element, name: str) -> Optional[str]:
    child = _child(elem, name)
    if child is None or child.text is None or not child.text.strip():
        return None
    return child.text.strip()


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------

def read_pi_xml_timeseries(path: Path, missing_value: float = -999.0) -> xr.Dataset:
    """Read a PI-XML timeseries file into an xarray Dataset.

    Each parameterId becomes a data variable. When the file holds a single
    location the variables are 1-D over ``time`` and carry the location in
    their ``location_id`` attribute; with several locations every variable is
    2-D over ``(time, location)`` with a ``location`` coordinate, so no series
    is dropped. The time axis is the union of all event timestamps, in UTC
    (the document's ``<timeZone>`` offset is removed).

    Args:
        path: Path to PI-XML file
        missing_value: Missing-value sentinel for series whose header has no
            ``<missVal>``

    Returns:
        xr.Dataset with a UTC ``time`` dimension; variables carry ``units``
        (when the header has them) and ``fews_type`` attributes

    Raises:
        PIXMLError: If the file cannot be parsed, or two series share the
            same (parameterId, locationId) -- qualifiers and ensemble members
            are not supported and are refused rather than overwritten
    """
    path = Path(path)
    if not path.is_file():
        raise PIXMLError(f"PI-XML file not found: {path}")

    try:
        tree = ET.parse(str(path))  # nosec B314
    except ET.ParseError as exc:
        raise PIXMLError(f"Malformed PI-XML in {path}: {exc}") from exc

    root = tree.getroot()
    time_zone_hours = parse_time_zone(root, error=PIXMLError)
    tz_shift = timedelta(hours=time_zone_hours)

    series_elements = [e for e in root if local_name(e.tag) == "series"]
    if not series_elements:
        raise PIXMLError(f"No <series> elements found in {path}")

    all_times: set = set()
    series_data: Dict[Tuple[str, str], Dict] = {}

    for series_elem in series_elements:
        header = _child(series_elem, "header")
        if header is None:
            raise PIXMLError("Missing <header> in <series>")

        param_id = _child_text(header, "parameterId")
        if param_id is None:
            raise PIXMLError(f"<series> header without <parameterId> in {path}")
        location_id = _child_text(header, "locationId") or "default"

        key = (param_id, location_id)
        if key in series_data:
            raise PIXMLError(
                f"Two series in {path} share parameterId '{param_id}' and locationId "
                f"'{location_id}' (qualifiers / ensemble members are not supported); "
                "export one series per parameter and location."
            )

        miss_text = _child_text(header, "missVal")
        try:
            series_missing = float(miss_text) if miss_text is not None else float(missing_value)
        except ValueError:
            raise PIXMLError(f"Invalid <missVal> '{miss_text}' in {path}") from None

        events: Dict[datetime, float] = {}
        for event in series_elem:
            if local_name(event.tag) != "event":
                continue
            stamp = f"{event.get('date', '')}T{event.get('time', '00:00:00')}"
            try:
                dt = datetime.fromisoformat(stamp) - tz_shift
            except ValueError:
                raise PIXMLError(
                    f"Invalid event date/time '{stamp}' for {param_id}@{location_id} in {path}"
                ) from None
            try:
                val = float(event.get("value", ""))
            except ValueError:
                val = np.nan
            # NaN missVal (the schema default) never equals anything; NaN
            # values are missing already.
            if not np.isnan(val) and not np.isnan(series_missing) \
                    and np.isclose(val, series_missing):
                val = np.nan
            events[dt] = val

        all_times.update(events)
        series_data[key] = {
            "events": events,
            "units": _child_text(header, "units"),
            "type": _child_text(header, "type"),
        }

    if not all_times:
        raise PIXMLError(f"No timesteps found in {path}")

    unique_times = sorted(all_times)
    position = {t: i for i, t in enumerate(unique_times)}
    time_index = np.array(unique_times, dtype="datetime64[ns]")

    locations = sorted({loc for _, loc in series_data})
    parameters: List[str] = []
    for param_id, _ in series_data:
        if param_id not in parameters:
            parameters.append(param_id)
    multi_location = len(locations) > 1

    data_vars = {}
    for param_id in parameters:
        members = {loc: sd for (p, loc), sd in series_data.items() if p == param_id}
        attrs: Dict[str, str] = {}
        units = {sd["units"] for sd in members.values() if sd["units"]}
        if len(units) > 1:
            raise PIXMLError(
                f"parameterId '{param_id}' has different units at different "
                f"locations in {path}: {sorted(units)}"
            )
        if units:
            attrs["units"] = units.pop()
        types = {sd["type"] for sd in members.values() if sd["type"]}
        if len(types) == 1:
            attrs["fews_type"] = types.pop()

        if multi_location:
            values = np.full((len(unique_times), len(locations)), np.nan)
            for loc, sd in members.items():
                column = locations.index(loc)
                for t, val in sd["events"].items():
                    values[position[t], column] = val
            data_vars[param_id] = xr.DataArray(values, dims=["time", "location"], attrs=attrs)
        else:
            (loc, sd), = members.items()
            values = np.full(len(unique_times), np.nan)
            for t, val in sd["events"].items():
                values[position[t]] = val
            attrs["location_id"] = loc
            data_vars[param_id] = xr.DataArray(values, dims=["time"], attrs=attrs)

    coords: Dict[str, object] = {"time": time_index}
    if multi_location:
        coords["location"] = locations
    return xr.Dataset(data_vars, coords=coords)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _time_step_attrs(times: np.ndarray) -> Dict[str, str]:
    """PI ``<timeStep>`` attributes for a time axis (equidistant or not)."""
    if len(times) >= 2:
        steps = np.diff(times.astype("datetime64[s]")).astype("int64")
        if steps[0] > 0 and np.all(steps == steps[0]):
            return {"unit": "second", "multiplier": str(int(steps[0]))}
    return {"unit": "nonequidistant"}


def _set_date_time(elem: ET.Element, dt: datetime) -> None:
    elem.set("date", dt.strftime("%Y-%m-%d"))
    elem.set("time", dt.strftime("%H:%M:%S"))


def write_pi_xml_timeseries(
    dataset: xr.Dataset,
    path: Path,
    *,
    missing_value: float = -999.0,
    location_id: str = "default",
    time_zone: float = 0.0,
) -> None:
    """Write an xarray Dataset as PI-XML timeseries.

    Each data variable becomes one ``<series>`` per location: variables over
    ``(time, location)`` write a series for every entry of the ``location``
    coordinate; 1-D variables use their ``location_id`` attribute, else
    ``location_id``. Headers are complete per ``pi_timeseries.xsd`` (type,
    locationId, parameterId, timeStep, startDate, endDate, missVal, and units
    when the variable has a ``units`` attribute).

    Args:
        dataset: Dataset with a UTC ``time`` coordinate
        path: Output file path
        missing_value: Value to use for NaN in PI-XML
        location_id: Location ID for 1-D variables without a ``location_id`` attr
        time_zone: Offset in hours from GMT of the *written* times; the UTC
            times of the dataset are shifted by it and ``<timeZone>`` records it

    Raises:
        PIXMLError: If writing fails
    """
    path = Path(path)

    try:
        if "time" not in dataset.coords or dataset.sizes.get("time", 0) == 0:
            raise ValueError("dataset has no time steps")

        root = ET.Element("TimeSeries")
        root.set("xmlns", _NS)
        root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")

        tz_elem = ET.SubElement(root, "timeZone")
        tz_elem.text = str(float(time_zone))

        times_utc = dataset["time"].values.astype("datetime64[s]")
        local = [
            t.astype(datetime) + timedelta(hours=float(time_zone)) for t in times_utc
        ]
        time_step = _time_step_attrs(times_utc)

        for var_name in dataset.data_vars:
            da = dataset[var_name]
            extra_dims = [d for d in da.dims if d != "time"]
            if "time" not in da.dims or extra_dims not in ([], ["location"]):
                raise ValueError(
                    f"variable '{var_name}' has dimensions {da.dims}; PI-XML series "
                    "need (time) or (time, location)"
                )
            if extra_dims:
                columns = [
                    (str(loc), da.sel(location=loc).transpose("time").values)
                    for loc in da["location"].values
                ]
            else:
                columns = [(str(da.attrs.get("location_id", location_id)), da.values)]

            for series_location, values in columns:
                series = ET.SubElement(root, "series")
                header = ET.SubElement(series, "header")

                ET.SubElement(header, "type").text = str(
                    da.attrs.get("fews_type", "instantaneous"))
                ET.SubElement(header, "locationId").text = series_location
                ET.SubElement(header, "parameterId").text = str(var_name)
                ET.SubElement(header, "timeStep", time_step)
                _set_date_time(ET.SubElement(header, "startDate"), local[0])
                _set_date_time(ET.SubElement(header, "endDate"), local[-1])
                ET.SubElement(header, "missVal").text = str(missing_value)
                units = da.attrs.get("units")
                if units:
                    # <units> is optional; an invented "-" is worse than none
                    ET.SubElement(header, "units").text = str(units)

                for dt, raw in zip(local, values):
                    val = float(raw)
                    event = ET.SubElement(series, "event")
                    _set_date_time(event, dt)
                    if np.isnan(val):
                        event.set("value", str(missing_value))
                    else:
                        event.set("value", f"{val:.6g}")

        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(str(path), xml_declaration=True, encoding="UTF-8")

    except Exception as exc:  # noqa: BLE001 — wrap-and-raise to domain error
        raise PIXMLError(f"Failed to write PI-XML to {path}: {exc}") from exc
