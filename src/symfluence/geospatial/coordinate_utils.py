#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

# -*- coding: utf-8 -*-

"""
Coordinate and Bounding Box Utilities

This module provides standardized utilities for handling geographic coordinates including:
- Bounding box parsing and validation
- Longitude normalization (0-360 vs -180 to 180)
- Coordinate transformations
- Meridian wrapping detection
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np


class BoundingBox:
    """
    Bounding box representation with coordinate normalization utilities.

    Attributes
    ----------
    lat_min : float
        Minimum latitude (southern bound)
    lat_max : float
        Maximum latitude (northern bound)
    lon_min : float
        Minimum longitude (western bound)
    lon_max : float
        Maximum longitude (eastern bound)
    """

    def __init__(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float
    ):
        """
        Initialize bounding box.

        Parameters
        ----------
        lat_min : float
            Minimum latitude
        lat_max : float
            Maximum latitude
        lon_min : float
            Minimum longitude
        lon_max : float
            Maximum longitude
        """
        # Validate latitudes
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError(f"Latitudes must be in [-90, 90]. Got: {lat_min}, {lat_max}")

        # Ensure lat_min <= lat_max
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    def to_dict(self) -> Dict[str, float]:
        """Return bounding box as dictionary."""
        return {
            'lat_min': self.lat_min,
            'lat_max': self.lat_max,
            'lon_min': self.lon_min,
            'lon_max': self.lon_max
        }

    def normalize_longitude(
        self,
        target_range: str = '0-360'
    ) -> 'BoundingBox':
        """
        Return new BoundingBox with normalized longitude range.

        Parameters
        ----------
        target_range : str
            Target longitude range: '0-360' or '-180-180'

        Returns
        -------
        BoundingBox
            New bbox with normalized longitudes
        """
        _eps = 1e-10
        # A box spanning the full circle would normalise to lon_min == lon_max
        # (an empty box): it is the whole globe in either convention.
        if self.lon_max - self.lon_min >= 360.0 - _eps:
            if target_range == '0-360':
                return BoundingBox(self.lat_min, self.lat_max, 0.0, 360.0)
            if target_range == '-180-180':
                return BoundingBox(self.lat_min, self.lat_max, -180.0, 180.0)
            raise ValueError(f"Unknown target_range: {target_range}")
        if target_range == '0-360':
            lon_min = self.lon_min % 360
            lon_max = self.lon_max % 360
            # Clamp: modulo of very small negatives can produce 360.0 or
            # values a few ULPs below it (e.g. 359.99999999999994).
            if lon_min >= 360.0 - _eps:
                lon_min = 0.0
            if lon_max >= 360.0 - _eps:
                lon_max = 0.0
        elif target_range == '-180-180':
            lon_min = ((self.lon_min + 180) % 360) - 180
            lon_max = ((self.lon_max + 180) % 360) - 180
            if lon_min >= 180.0 - _eps:
                lon_min = -180.0
            if lon_max >= 180.0 - _eps:
                lon_max = -180.0
        else:
            raise ValueError(f"Unknown target_range: {target_range}")

        return BoundingBox(self.lat_min, self.lat_max, lon_min, lon_max)

    def pad(self, degrees: float) -> 'BoundingBox':
        """
        Expand bounding box by given degrees in all directions.

        Parameters
        ----------
        degrees : float
            Padding in degrees to add in each direction.
            Latitudes are clamped to [-90, 90].

        Returns
        -------
        BoundingBox
            New padded bounding box
        """
        return BoundingBox(
            lat_min=max(-90.0, self.lat_min - degrees),
            lat_max=min(90.0, self.lat_max + degrees),
            lon_min=self.lon_min - degrees,
            lon_max=self.lon_max + degrees,
        )

    def crosses_meridian(self, longitude_range: str = '0-360') -> bool:
        """
        Check if bounding box crosses the prime meridian or antimeridian.

        Parameters
        ----------
        longitude_range : str
            Longitude convention: '0-360' or '-180-180'

        Returns
        -------
        bool
            True if bbox crosses meridian
        """
        bbox_norm = self.normalize_longitude(longitude_range)

        # In normalized coordinates, if min > max, bbox crosses meridian
        return bbox_norm.lon_min > bbox_norm.lon_max

    def get_sorted_coords(self) -> Tuple[float, float, float, float]:
        """
        Get coordinates sorted so min < max.

        Returns
        -------
        tuple
            (lat_min, lat_max, lon_min, lon_max) with sorted values
        """
        lat_min, lat_max = sorted([self.lat_min, self.lat_max])
        lon_min, lon_max = sorted([self.lon_min, self.lon_max])
        return lat_min, lat_max, lon_min, lon_max

    def __str__(self) -> str:
        return f"BBox(lat=[{self.lat_min}, {self.lat_max}], lon=[{self.lon_min}, {self.lon_max}])"

    def __repr__(self) -> str:
        return (f"BoundingBox(lat_min={self.lat_min}, lat_max={self.lat_max}, "
                f"lon_min={self.lon_min}, lon_max={self.lon_max})")


def parse_bbox(bbox_string: str, format: str = 'lat_max/lon_min/lat_min/lon_max') -> Dict[str, float]:
    """
    Parse bounding box from string representation.

    Parameters
    ----------
    bbox_string : str
        Bounding box as string (slash-separated coordinates)
    format : str, optional
        Order of coordinates in string (default: 'lat_max/lon_min/lat_min/lon_max')
        This is the system standard format (North/West/South/East).
        Other supported formats:
        - 'lat_max/lon_max/lat_min/lon_min' (NE corner then SW corner)
        - 'lat_min/lon_min/lat_max/lon_max'
        - 'lon_min/lat_min/lon_max/lat_max'

    Returns
    -------
    dict
        Dictionary with keys: lat_min, lat_max, lon_min, lon_max

    Examples
    --------
    >>> parse_bbox('60.0/-130.0/50.0/-120.0')
    {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -120.0}
    """
    if not bbox_string:
        return {}

    coords = [float(c.strip()) for c in bbox_string.split('/')]

    if len(coords) != 4:
        raise ValueError(f"Expected 4 coordinates, got {len(coords)}: {bbox_string}")

    # Parse based on format
    if format == 'lat_max/lon_min/lat_min/lon_max':  # System standard: N/W/S/E
        return {
            'lat_min': coords[2],
            'lat_max': coords[0],
            'lon_min': coords[1],
            'lon_max': coords[3]
        }
    elif format == 'lat_max/lon_max/lat_min/lon_min':
        return {
            'lat_min': coords[2],
            'lat_max': coords[0],
            'lon_min': coords[3],
            'lon_max': coords[1]
        }
    elif format == 'lat_min/lon_min/lat_max/lon_max':
        return {
            'lat_min': coords[0],
            'lat_max': coords[2],
            'lon_min': coords[1],
            'lon_max': coords[3]
        }
    elif format == 'lon_min/lat_min/lon_max/lat_max':
        return {
            'lat_min': coords[1],
            'lat_max': coords[3],
            'lon_min': coords[0],
            'lon_max': coords[2]
        }
    else:
        raise ValueError(f"Unknown bbox format: {format}")


def normalize_longitude(
    lon: Union[float, np.ndarray],
    target_range: str = '0-360'
) -> Union[float, np.ndarray]:
    """
    Normalize longitude to specified range.

    Parameters
    ----------
    lon : float or np.ndarray
        Longitude value(s) to normalize
    target_range : str
        Target range: '0-360' or '-180-180'

    Returns
    -------
    float or np.ndarray
        Normalized longitude(s)

    Examples
    --------
    >>> normalize_longitude(-120.0, '0-360')
    240.0
    >>> normalize_longitude(240.0, '-180-180')
    -120.0
    """
    _EPS = 1e-10  # tolerance for boundary clamping
    if target_range == '0-360':
        result = lon % 360
        # Clamp: modulo of very small negatives can produce 360.0 or
        # values a few ULPs below it (e.g. 359.99999999999994).
        if isinstance(result, np.ndarray):
            result[result >= 360.0 - _EPS] = 0.0
        elif result >= 360.0 - _EPS:
            result = 0.0
        return result
    elif target_range == '-180-180':
        result = ((lon + 180) % 360) - 180
        if isinstance(result, np.ndarray):
            result[result >= 180.0 - _EPS] = -180.0
        elif result >= 180.0 - _EPS:
            result = -180.0
        return result
    else:
        raise ValueError(f"Unknown target_range: {target_range}")


def create_coordinate_mask(
    lat: np.ndarray,
    lon: np.ndarray,
    bbox: Dict[str, float],
    lon_range: str = '-180-180'
) -> np.ndarray:
    """
    Create boolean mask for coordinates within bounding box.

    Handles meridian crossing correctly for both longitude conventions.

    Parameters
    ----------
    lat : np.ndarray
        Latitude values
    lon : np.ndarray
        Longitude values
    bbox : dict
        Bounding box with keys: lat_min, lat_max, lon_min, lon_max
    lon_range : str
        Longitude convention of input data: '0-360' or '-180-180'

    Returns
    -------
    np.ndarray
        Boolean mask (True for points inside bbox)

    Examples
    --------
    >>> lat = np.array([50.5, 55.0, 60.5])
    >>> lon = np.array([-125.0, -120.0, -115.0])
    >>> bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -110.0}
    >>> create_coordinate_mask(lat, lon, bbox)
    array([True, True, True])
    """
    # Latitude mask (straightforward)
    lat_mask = (lat >= bbox['lat_min']) & (lat <= bbox['lat_max'])

    # Longitude mask (handle meridian crossing)
    bbox_obj = BoundingBox(**bbox)
    bbox_norm = bbox_obj.normalize_longitude(lon_range)

    if bbox_norm.crosses_meridian(lon_range):
        # Bbox crosses meridian - need OR condition
        lon_mask = (lon >= bbox_norm.lon_min) | (lon <= bbox_norm.lon_max)
    else:
        # Normal case - AND condition
        lon_mask = (lon >= bbox_norm.lon_min) & (lon <= bbox_norm.lon_max)

    return lat_mask & lon_mask


def convert_bbox_range(
    bbox: Dict[str, float],
    from_range: str = '-180-180',
    to_range: str = '0-360'
) -> Dict[str, float]:
    """
    Convert bounding box between longitude conventions.

    Parameters
    ----------
    bbox : dict
        Bounding box with lat_min, lat_max, lon_min, lon_max
    from_range : str
        Current longitude convention
    to_range : str
        Target longitude convention

    Returns
    -------
    dict
        Bounding box with converted longitude values

    Examples
    --------
    >>> bbox = {'lat_min': 50, 'lat_max': 60, 'lon_min': -130, 'lon_max': -120}
    >>> convert_bbox_range(bbox, '-180-180', '0-360')
    {'lat_min': 50, 'lat_max': 60, 'lon_min': 230.0, 'lon_max': 240.0}
    """
    if from_range == to_range:
        return bbox.copy()

    return {
        'lat_min': bbox['lat_min'],
        'lat_max': bbox['lat_max'],
        'lon_min': float(normalize_longitude(bbox['lon_min'], to_range)),
        'lon_max': float(normalize_longitude(bbox['lon_max'], to_range))
    }


def get_bbox_extent(bbox: Dict[str, float]) -> Tuple[float, float]:
    """
    Calculate bounding box extent in degrees.

    Parameters
    ----------
    bbox : dict
        Bounding box dictionary

    Returns
    -------
    tuple
        (lat_extent, lon_extent) in degrees
    """
    lat_extent = abs(bbox['lat_max'] - bbox['lat_min'])

    # Handle longitude extent (account for meridian crossing)
    bbox_obj = BoundingBox(**bbox)
    if bbox_obj.crosses_meridian():
        # If crosses meridian, calculate via 360° - gap
        lon_gap = bbox_obj.lon_min - bbox_obj.lon_max
        lon_extent = 360.0 - lon_gap
    else:
        lon_extent = abs(bbox['lon_max'] - bbox['lon_min'])

    return lat_extent, lon_extent


def validate_bbox(bbox: Dict[str, float], raise_error: bool = True) -> bool:
    """
    Validate bounding box coordinates.

    Parameters
    ----------
    bbox : dict
        Bounding box to validate
    raise_error : bool
        If True, raise ValueError on invalid bbox; if False, return False

    Returns
    -------
    bool
        True if valid, False if invalid (when raise_error=False)

    Raises
    ------
    ValueError
        If bbox is invalid and raise_error=True
    """
    required_keys = {'lat_min', 'lat_max', 'lon_min', 'lon_max'}

    if not required_keys.issubset(bbox.keys()):
        msg = f"Missing required keys: {required_keys - bbox.keys()}"
        if raise_error:
            raise ValueError(msg)
        return False

    if not (-90 <= bbox['lat_min'] <= 90 and -90 <= bbox['lat_max'] <= 90):
        msg = f"Latitudes must be in [-90, 90]: {bbox['lat_min']}, {bbox['lat_max']}"
        if raise_error:
            raise ValueError(msg)
        return False

    if bbox['lat_min'] > bbox['lat_max']:
        msg = f"lat_min > lat_max: {bbox['lat_min']} > {bbox['lat_max']}"
        if raise_error:
            raise ValueError(msg)
        return False

    return True


def bbox_to_string(
    bbox: Dict[str, float],
    format: str = 'lat_max/lon_max/lat_min/lon_min'
) -> str:
    """
    Convert bounding box dictionary to string representation.

    Parameters
    ----------
    bbox : dict
        Bounding box dictionary
    format : str
        Output format (default: 'lat_max/lon_max/lat_min/lon_min')

    Returns
    -------
    str
        String representation of bbox

    Examples
    --------
    >>> bbox = {'lat_min': 50.0, 'lat_max': 60.0, 'lon_min': -130.0, 'lon_max': -120.0}
    >>> bbox_to_string(bbox)
    '60.0/-120.0/50.0/-130.0'
    """
    if format == 'lat_max/lon_max/lat_min/lon_min':
        return f"{bbox['lat_max']}/{bbox['lon_max']}/{bbox['lat_min']}/{bbox['lon_min']}"
    elif format == 'lat_min/lon_min/lat_max/lon_max':
        return f"{bbox['lat_min']}/{bbox['lon_min']}/{bbox['lat_max']}/{bbox['lon_max']}"
    elif format == 'lon_min/lat_min/lon_max/lat_max':
        return f"{bbox['lon_min']}/{bbox['lat_min']}/{bbox['lon_max']}/{bbox['lat_max']}"
    else:
        raise ValueError(f"Unknown format: {format}")


from symfluence.core.mixins import LoggingMixin


class CoordinateUtilsMixin(LoggingMixin):
    """
    Mixin providing coordinate and bounding box utilities to classes.
    """

    def _parse_bbox(
        self,
        bbox_string: Optional[str],
        format: str = 'lat_max/lon_min/lat_min/lon_max'
    ) -> Dict[str, float]:
        """
        Parse bounding box from string (instance method).
        """
        if not bbox_string:
            return {}
        return parse_bbox(bbox_string, format=format)

    def _normalize_longitude(
        self,
        lon: Union[float, np.ndarray],
        target_range: str = '0-360'
    ) -> Union[float, np.ndarray]:
        """
        Normalize longitude (instance method).
        """
        return normalize_longitude(lon, target_range=target_range)

    def _validate_bbox(self, bbox: Dict[str, float], raise_error: bool = True) -> bool:
        """
        Validate bounding box (instance method).
        """
        return validate_bbox(bbox, raise_error=raise_error)
