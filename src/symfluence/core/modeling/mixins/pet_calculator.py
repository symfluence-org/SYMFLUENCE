# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Potential Evapotranspiration (PET) Calculator Mixin.

Provides three PET calculation methods for use in hydrological model preprocessors:
- Oudin's formula (simple, temperature-based)
- Hamon's method (temperature and daylight-based)
- Hargreaves method (simplified version)

All methods automatically detect and handle temperature units (Kelvin or Celsius).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from symfluence.core.constants import PhysicalConstants

#: Calibration coefficient of the Lu et al. (2005) form of Hamon PET.
HAMON_LU_COEFFICIENT = 0.1651


def day_length_hours(lat, day_of_year):
    """Day length N (hours, 0-24) from latitude (degrees) and day of year.

    FAO-56 solar declination and sunset hour angle. The arccos argument is
    clipped, so latitudes inside the polar circles give exactly 0 h (polar
    night) or 24 h (polar day) rather than NaN. Arguments broadcast (numpy or
    xarray).
    """
    lat_rad = np.deg2rad(lat)
    decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * day_of_year - 1.39)
    cos_arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
    return 24.0 / np.pi * np.arccos(cos_arg)


def hamon_pet(temp_c, day_length_h, coefficient: float = HAMON_LU_COEFFICIENT):
    """Hamon PET (mm/day) in the form of Lu et al. (2005).

    ``PET = coefficient * (N / 12) * RHOSAT`` with

    - ``N``: day length in hours,
    - ``RHOSAT = 216.7 * es / (T + 273.3)``: saturated vapour density (g/m3),
    - ``es = 6.108 * exp(17.26939 * T / (T + 237.3))``: saturation vapour
      pressure (hPa), ``T`` in degrees Celsius.

    At 15 degC and a 12 h day this gives 2.12 mm/day.

    Sub-zero temperatures are NOT cut off: the formula stays positive and
    decays smoothly (about 0.5 mm/day at -5 degC with a 12 h day), which is
    what every earlier copy in this code base did. Callers that want the
    "PET = 0 for T <= 0 degC" convention apply it themselves. NaN temperature
    gives NaN PET.

    This is the single Hamon implementation; arguments broadcast and may be
    numpy arrays or xarray objects (dask-backed included).

    Reference:
        Lu, J., Sun, G., McNulty, S. G., Amatya, D. M. (2005). A comparison of
        six potential evapotranspiration methods for regional use in the
        southeastern United States. JAWRA 41(3), 621-633.
        Hamon, W. R. (1961). Estimating potential evapotranspiration.
        J. Hydraul. Div. ASCE 87(HY3), 107-120.
    """
    es_hpa = 6.108 * np.exp(17.26939 * temp_c / (temp_c + 237.3))
    rhosat = 216.7 * es_hpa / (temp_c + 273.3)
    return coefficient * (day_length_h / 12.0) * rhosat


class PETCalculatorMixin:
    """
    Mixin class providing PET calculation methods.

    This mixin provides three common PET calculation methods that can be
    used by model preprocessors. All methods:
    - Auto-detect temperature units (Kelvin vs Celsius)
    - Handle both lumped and distributed (multi-HRU) configurations
    - Return xarray DataArrays with proper metadata

    Usage:
        class MyModelPreProcessor(BaseModelPreProcessor, PETCalculatorMixin):
            def prepare_forcing(self):
                pet = self.calculate_pet_oudin(temp_data, latitude)

    Note:
        Requires self.logger to be available (provided by BaseModelPreProcessor)
    """

    def _get_robust_temp_c(self, temp_data: xr.DataArray) -> xr.DataArray:
        """
        Helper to get temperature in Celsius with robust unit detection.

        Args:
            temp_data: Input temperature DataArray

        Returns:
            Temperature in Celsius
        """
        # Check metadata units attribute if available
        meta_units = str(temp_data.attrs.get('units', '')).lower()

        # Convert to Celsius if needed based on metadata
        if 'k' in meta_units and 'c' not in meta_units:
            self.logger.info(f"Metadata indicates temperature is in Kelvin ({meta_units}), converting to Celsius")
            temp_C = temp_data - PhysicalConstants.KELVIN_OFFSET
            temp_C.attrs['units'] = 'degC'
            return temp_C
        elif 'c' in meta_units:
            self.logger.info(f"Metadata indicates temperature is in Celsius ({meta_units}), using as-is")
            return temp_data

        # If metadata is missing or ambiguous, we need to check the data range.
        # We only compute the mean to minimize overhead if it's a dask array.
        self.logger.debug("Metadata units missing or ambiguous, checking data range for unit detection")
        temp_mean = float(temp_data.mean().compute()) if hasattr(temp_data.data, 'compute') else float(temp_data.mean())

        self.logger.debug(f"Input temperature mean: {temp_mean:.2f}")

        if temp_mean > 100:  # Likely Kelvin based on range
            self.logger.info("Temperature appears to be in Kelvin based on range, converting to Celsius")
            temp_C = temp_data - PhysicalConstants.KELVIN_OFFSET
        elif -60 < temp_mean < 60:  # Likely Celsius based on range
            self.logger.info("Temperature appears to be in Celsius based on range, using as-is")
            temp_C = temp_data
        elif temp_mean <= -100:
            # Case where data might be Celsius but was incorrectly treated as Kelvin before
            if -60 < (temp_mean + PhysicalConstants.KELVIN_OFFSET) < 60:
                self.logger.info("Temperature appears to have been double-converted. Reversing.")
                temp_C = temp_data + PhysicalConstants.KELVIN_OFFSET
            else:
                raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")
        else:
            raise ValueError(f"Temperature data has unexpected range. Mean={temp_mean:.2f}")

        # Verification (only in debug)
        if self.logger.isEnabledFor(10):  # DEBUG level
            temp_mean_C = float(temp_C.mean().compute()) if hasattr(temp_C.data, 'compute') else float(temp_C.mean())
            self.logger.debug(f"Temperature in Celsius: Mean={temp_mean_C:.2f}°C")
            if temp_mean_C < -60 or temp_mean_C > 60:
                raise ValueError(f"Unrealistic temperature after conversion: {temp_mean_C:.2f}°C")

        return temp_C

    def calculate_pet_oudin(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate potential evapotranspiration using Oudin's formula.

        Oudin's formula is a simple temperature-based method:
        PET = Ra * (T + 5) / 100 when T > -5°C, else 0

        Reference:
            Oudin et al. (2005). "Which potential evapotranspiration input
            for a lumped rainfall-runoff model?"

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata
        """
        self.logger.info("Calculating PET using Oudin's formula")

        temp_C = self._get_robust_temp_c(temp_data)

        # Get day of year using xarray accessor (dask-friendly)
        doy = temp_data.time.dt.dayofyear

        # Calculate solar radiation components
        lat_rad = np.deg2rad(lat)

        # Solar declination (radians)
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle with numerical stability
        cos_arg = (-np.tan(lat_rad) * np.tan(solar_decl)).clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

        # Oudin's formula: PET = Ra / (lambda * rho) * (T + 5) / 100 when T + 5 > 0
        # lambda = 2.45 MJ/kg (latent heat of vaporization)
        # rho = 1000 kg/m³ (density of water) — cancels with mm↔m conversion
        pet = xr.where(temp_C + 5.0 > 0.0, Ra / (100.0 * 2.45) * (temp_C + 5.0), 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Oudin et al. (2005)',
            'latitude': lat
        }

        # Log results (only summary)
        self.logger.debug("PET calculation complete (Oudin)")

        return pet

    def calculate_pet_hamon(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hamon's method.

        Hamon's method uses temperature and daylight hours to estimate PET.
        Thin xarray wrapper around :func:`hamon_pet` (Lu et al. 2005 form,
        2.12 mm/day at 15 degC with a 12 h day). Sub-zero temperatures are
        not cut off and NaN temperature stays NaN.

        Reference:
            Hamon (1961). "Estimating Potential Evapotranspiration";
            Lu et al. (2005), JAWRA 41(3).

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata
        """
        self.logger.debug("Calculating PET using Hamon's method")

        temp_C = self._get_robust_temp_c(temp_data)

        # Day of year using xarray accessor
        doy = temp_data.time.dt.dayofyear

        # Lu et al. (2005) form; see hamon_pet() for the formula and units.
        pet = hamon_pet(temp_C, day_length_hours(lat, doy))

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'method': 'Hamon (1961), Lu et al. (2005) form',
            'latitude': lat
        }

        self.logger.debug("PET calculation complete (Hamon)")

        return pet

    def calculate_pet_hargreaves(self, temp_data: xr.DataArray, lat: float) -> xr.DataArray:
        """
        Calculate PET using Hargreaves method (simplified version).

        This is a simplified Hargreaves method that uses only mean temperature.
        The full method requires Tmin and Tmax; here we assume a typical
        diurnal temperature range of 10°C.

        Reference:
            Hargreaves & Samani (1985). "Reference Crop Evapotranspiration
            from Temperature"

        Args:
            temp_data: Temperature data in either Kelvin or Celsius
            lat: Latitude of the catchment centroid in degrees

        Returns:
            Calculated PET in mm/day with proper metadata
        """
        self.logger.info("Calculating PET using Hargreaves method (simplified)")

        temp_C = self._get_robust_temp_c(temp_data)

        # Get day of year
        doy = temp_data.time.dt.dayofyear

        # Calculate extraterrestrial radiation (Ra)
        lat_rad = np.deg2rad(lat)

        # Solar declination
        solar_decl = 0.409 * np.sin((2.0 * np.pi / 365.0) * doy - 1.39)

        # Sunset hour angle
        cos_arg = (-np.tan(lat_rad) * np.tan(solar_decl)).clip(-1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)

        # Inverse relative distance Earth-Sun
        dr = 1.0 + 0.033 * np.cos((2.0 * np.pi / 365.0) * doy)

        # Extraterrestrial radiation (MJ/m²/day)
        Ra = ((24.0 * 60.0 / np.pi) * 0.082 * dr *
              (sunset_angle * np.sin(lat_rad) * np.sin(solar_decl) +
               np.cos(lat_rad) * np.cos(solar_decl) * np.sin(sunset_angle)))

        # Hargreaves formula (simplified without Tmin/Tmax)
        # PET = 0.0023 * Ra * (Tmean + 17.8) * TD^0.5
        # TD = 10.0  # Assumed temperature range (°C) when min/max not available
        pet = 0.0023 * (Ra * 0.408) * (temp_C + 17.8) * np.sqrt(10.0)

        # Ensure non-negative
        pet = pet.where(pet > 0, 0.0)

        # Add metadata
        pet.attrs = {
            'units': 'mm/day',
            'long_name': 'Potential evapotranspiration',
            'standard_name': 'water_potential_evaporation_flux',
            'method': 'Hargreaves (simplified)',
            'latitude': lat,
            'note': 'Simplified version using assumed diurnal temperature range of 10°C'
        }

        self.logger.debug("PET calculation complete (Hargreaves)")

        return pet

    # ------------------------------------------------------------------
    # Numpy-based static methods for models that work with raw arrays
    # ------------------------------------------------------------------

    @staticmethod
    def hamon_pet_numpy(
        temp_c: np.ndarray,
        day_of_year: np.ndarray,
        lat: float,
        coefficient: float = HAMON_LU_COEFFICIENT,
    ) -> np.ndarray:
        """Hamon PET (mm/day) from numpy arrays, Lu et al. (2005) form.

        ``PET = coefficient * (N / 12) * RHOSAT``; see :func:`hamon_pet`.
        *coefficient* is Lu's calibration coefficient (default 0.1651).

        The former default ``0.55 * (D/12)^2 * e_sat[kPa]`` was neither
        Hamon (1961) nor Lu et al. (2005) and gave less than half the
        reference PET; 0.55 is the inches-per-day coefficient of Hamon (1961)
        and is refused here rather than silently fed into the mm/day form.

        Args:
            temp_c: Temperature in Celsius.
            day_of_year: Day-of-year integers, broadcastable with *temp_c*.
            lat: Latitude in degrees (scalar or broadcastable array).
            coefficient: Coefficient of the Lu et al. (2005) form.

        Returns:
            PET array (mm/day), broadcast shape of the inputs.
        """
        if coefficient == 0.55:
            raise ValueError(
                "hamon_pet_numpy: coefficient=0.55 is the inches/day coefficient "
                "of the Hamon (1961) (D/12)^2 form, which is no longer implemented; "
                f"use the Lu et al. (2005) coefficient ({HAMON_LU_COEFFICIENT}) or a "
                "calibrated value of it.")
        pet = hamon_pet(np.asarray(temp_c, dtype=float), day_length_hours(lat, day_of_year), coefficient)
        return np.maximum(pet, 0.0)

    @staticmethod
    def oudin_pet_numpy(
        temp_c: np.ndarray,
        day_of_year: np.ndarray,
        lat: float,
    ) -> np.ndarray:
        """Oudin PET (mm/day) from numpy arrays.

        Reference:
            Oudin et al. (2005). "Which potential evapotranspiration input
            for a lumped rainfall-runoff model?"

        Args:
            temp_c: Temperature in Celsius (1-D array).
            day_of_year: Day-of-year integers (1-D array).
            lat: Latitude in degrees.

        Returns:
            PET array (mm/day).
        """
        lat_rad = np.deg2rad(lat)
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * day_of_year - 1.39)
        cos_arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
        ws = np.arccos(cos_arg)
        dr = 1.0 + 0.033 * np.cos(2.0 * np.pi / 365.0 * day_of_year)

        # Extraterrestrial radiation (MJ/m²/day)
        ra = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(decl)
            + np.cos(lat_rad) * np.cos(decl) * np.sin(ws)
        )

        pet = np.where(temp_c + 5.0 > 0.0, ra / (100.0 * 2.45) * (temp_c + 5.0), 0.0)
        return np.maximum(pet, 0.0)

    @staticmethod
    def thornthwaite_pet_numpy(
        temp_c: np.ndarray,
        time: pd.DatetimeIndex,
        lat: float,
    ) -> np.ndarray:
        """Thornthwaite PET (mm/day) from numpy arrays.

        Args:
            temp_c: Daily mean temperature in Celsius (1-D array).
            time: DatetimeIndex matching *temp_c*.
            lat: Latitude in degrees.

        Returns:
            PET array (mm/day).
        """
        # Monthly average temperature for heat index
        df = pd.DataFrame({'temp': temp_c}, index=time)
        monthly_temp = df.resample('ME').mean()
        monthly_temp_pos = np.maximum(monthly_temp['temp'].values, 0.0)
        heat_index = np.sum((monthly_temp_pos / 5.0) ** 1.514)

        # Thornthwaite exponent
        a = (6.75e-7 * heat_index ** 3
             - 7.71e-5 * heat_index ** 2
             + 1.79e-2 * heat_index
             + 0.492)

        # Daily PET (mm/month equivalent)
        pet = np.where(temp_c > 0.0, 16.0 * (10.0 * temp_c / heat_index) ** a, 0.0)

        # Day-length correction
        doy = np.asarray(time.dayofyear)
        lat_rad = np.deg2rad(lat)
        decl = 0.409 * np.sin(2.0 * np.pi / 365.0 * doy - 1.39)
        cos_arg = np.clip(-np.tan(lat_rad) * np.tan(decl), -1.0, 1.0)
        sunset_angle = np.arccos(cos_arg)
        day_length = 24.0 / np.pi * sunset_angle
        correction = day_length / 12.0

        # Convert from mm/month to mm/day and apply correction
        pet = pet / 30.0 * correction

        return np.maximum(pet, 0.0)
