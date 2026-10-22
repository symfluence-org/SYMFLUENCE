# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Hydrology attribute processor.

Handles water balance calculations, streamflow signatures, baseflow analysis,
and river network characteristics.
"""
from __future__ import annotations

from typing import Any, Dict, cast

import numpy as np
import pandas as pd

from symfluence.core.mixins.project import resolve_data_subdir

from .base import BaseAttributeProcessor


class HydrologyProcessor(BaseAttributeProcessor):
    """Processor for hydrological attributes."""

    def calculate_water_balance(self) -> Dict[str, Any]:
        """
        Calculate water balance components and hydrological indices.

        Returns:
            Dictionary of water balance metrics
        """
        results: Dict[str, Any] = {}

        # Look for required data
        forcing_dir = resolve_data_subdir(self.project_dir, 'forcing')
        observations_dir = resolve_data_subdir(self.project_dir, 'observations')
        precip_path = forcing_dir / f"{self.domain_name}_precipitation.csv"
        pet_path = forcing_dir / f"{self.domain_name}_pet.csv"
        streamflow_path = observations_dir / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"

        if not streamflow_path.exists():
            self.logger.warning("Cannot calculate water balance: missing streamflow data")
            return results

        if not precip_path.exists():
            self.logger.warning("Cannot calculate water balance: missing precipitation data")
            return results

        try:
            # Read data
            precip_df = pd.read_csv(precip_path, parse_dates=['date'])
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])

            # Read PET if available
            if pet_path.exists():
                pet_df = pd.read_csv(pet_path, parse_dates=['date'])
            else:
                pet_df = None

            # Set index and align data
            precip_df.set_index('date', inplace=True)
            streamflow_df.set_index('date', inplace=True)
            if pet_df is not None:
                pet_df.set_index('date', inplace=True)

            # Get column names (handle different naming conventions)
            precip_col = 'precipitation_mm' if 'precipitation_mm' in precip_df.columns else 'precipitation'
            flow_col = 'flow_cms' if 'flow_cms' in streamflow_df.columns else 'flow'

            # Define common time period
            common_period = precip_df.index.intersection(streamflow_df.index)
            if pet_df is not None:
                common_period = common_period.intersection(pet_df.index)

            if len(common_period) == 0:
                self.logger.warning("No common time period between precipitation and streamflow data")
                return results

            # Subset data to common period
            precip = precip_df.loc[common_period, precip_col].copy()
            streamflow = streamflow_df.loc[common_period, flow_col].copy()
            if pet_df is not None:
                pet_col = 'pet' if 'pet' in pet_df.columns else 'pet_mm'
                pet = pet_df.loc[common_period, pet_col].copy()
            else:
                pet = None

            # Calculate annual values
            annual_precip = precip.resample('YE').sum()
            annual_streamflow = streamflow.resample('YE').sum()

            # Align years
            common_years = annual_precip.index.intersection(annual_streamflow.index)

            if len(common_years) > 0:
                # 1. Runoff ratio (Q/P)
                runoff_ratio = annual_streamflow.loc[common_years].mean() / annual_precip.loc[common_years].mean()
                results["runoff_ratio"] = runoff_ratio

                # 2. Mean annual values
                results["mean_annual_precip_mm"] = annual_precip.mean()
                results["mean_annual_streamflow_mm"] = annual_streamflow.mean()

                # 3. Aridity index and Budyko analysis
                if pet is not None:
                    annual_pet = pet.resample('YE').sum()
                    common_years_pet = common_years.intersection(annual_pet.index)

                    if len(common_years_pet) > 0:
                        mean_precip = annual_precip.loc[common_years_pet].mean()
                        mean_pet = annual_pet.loc[common_years_pet].mean()

                        # Aridity index (PET/P)
                        aridity_index = mean_pet / mean_precip if mean_precip > 0 else np.nan
                        results["aridity_index"] = aridity_index

                        # PET mean
                        results["pet_mean_mm_per_year"] = mean_pet

                        # Budyko parameter estimation
                        from scipy.optimize import minimize

                        # Fu equation: ET/P = 1 + PET/P - [1 + (PET/P)^w]^(1/w)
                        def fu_equation(w, pet_p):
                            if w <= 0:
                                return np.nan
                            return 1 + pet_p - (1 + pet_p**w)**(1/w)

                        # Estimate AET from water balance: AET = P - Q
                        aet = annual_precip.loc[common_years_pet].mean() - annual_streamflow.loc[common_years_pet].mean()
                        et_p_ratio = aet / mean_precip if mean_precip > 0 else 0

                        def objective(w):
                            predicted = fu_equation(w[0], aridity_index)
                            if np.isnan(predicted):
                                return 1e10
                            return (predicted - et_p_ratio)**2

                        # Optimize for w parameter
                        try:
                            result = minimize(objective, [2.6], bounds=[(0.5, 10.0)])
                            if result.success:
                                results["budyko_w_parameter"] = result.x[0]
                        except (ValueError, RuntimeError):
                            pass  # Optimization may fail for certain data, non-critical

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error calculating water balance: {str(e)}", exc_info=True)

        return results

    def calculate_streamflow_signatures(self) -> Dict[str, Any]:
        """
        Calculate streamflow signatures including flow duration curve metrics.

        Returns:
            Dictionary of streamflow signature metrics
        """
        results: Dict[str, Any] = {}

        streamflow_path = resolve_data_subdir(self.project_dir, 'observations') / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"

        if not streamflow_path.exists():
            return results

        try:
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            flow_col = 'flow_cms' if 'flow_cms' in streamflow_df.columns else 'flow'
            flow = np.asarray(streamflow_df[flow_col].dropna().values)

            if len(flow) == 0:
                return results

            # Flow duration curve percentiles
            percentiles = [5, 25, 50, 75, 95]
            for p in percentiles:
                results[f"q{p:02d}"] = np.percentile(flow, 100 - p)

            # Half-flow date (day of year when 50% of annual flow has passed)
            streamflow_df['date'] = pd.to_datetime(streamflow_df['date'])
            streamflow_df['cumsum'] = streamflow_df[flow_col].cumsum()
            annual_groups = streamflow_df.groupby(streamflow_df['date'].dt.year)

            half_flow_dates = []
            for year, group in annual_groups:
                total_flow = group[flow_col].sum()
                half_flow = total_flow / 2
                idx = (group['cumsum'] >= half_flow).idxmax()
                if idx in group.index:
                    half_flow_dates.append(group.loc[cast(Any, idx), 'date'].dayofyear)

            if half_flow_dates:
                results["half_flow_date"] = np.mean(half_flow_dates)

            # High flow duration (days above 9x median)
            median_flow = np.median(flow)
            high_flow_threshold = 9 * median_flow
            high_flow_days = np.sum(flow > high_flow_threshold)
            results["high_flow_duration"] = high_flow_days

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error calculating streamflow signatures: {str(e)}", exc_info=True)

        return results

    def calculate_baseflow_attributes(self) -> Dict[str, Any]:
        """
        Calculate baseflow attributes using Eckhardt method.

        Returns:
            Dictionary of baseflow metrics
        """
        results: Dict[str, Any] = {}

        try:
            import baseflow
        except ImportError:
            self.logger.warning("baseflow library not available")
            return results

        streamflow_path = resolve_data_subdir(self.project_dir, 'observations') / "streamflow" / "preprocessed" / f"{self.domain_name}_streamflow_processed.csv"

        if not streamflow_path.exists():
            return results

        try:
            streamflow_df = pd.read_csv(streamflow_path, parse_dates=['date'])
            flow_col = 'flow_cms' if 'flow_cms' in streamflow_df.columns else 'flow'
            flow = streamflow_df[flow_col].values

            # Run Eckhardt filter
            bf = baseflow.separation(flow, method='Eckhardt')

            # Calculate baseflow index
            baseflow_index = np.sum(bf) / np.sum(flow) if np.sum(flow) > 0 else 0
            results["baseflow_index"] = baseflow_index

            # Seasonal baseflow indices
            streamflow_df['baseflow'] = bf
            streamflow_df['month'] = pd.to_datetime(streamflow_df['date']).dt.month

            seasons = {
                'winter': [12, 1, 2],
                'spring': [3, 4, 5],
                'summer': [6, 7, 8],
                'fall': [9, 10, 11]
            }

            for season_name, months in seasons.items():
                season_data = streamflow_df[streamflow_df['month'].isin(months)]
                if len(season_data) > 0:
                    season_bf = season_data['baseflow'].sum()
                    season_total = season_data[flow_col].sum()
                    if season_total > 0:
                        results[f"baseflow_index_{season_name}"] = season_bf / season_total

        except Exception as e:  # noqa: BLE001 — preprocessing resilience
            self.logger.error(f"Error calculating baseflow attributes: {str(e)}", exc_info=True)

        return results

    def enhance_river_network_analysis(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance results with river network analysis.

        Args:
            current_results: Existing results dictionary

        Returns:
            Enhanced results with network metrics
        """
        results: Dict[str, Any] = {}

        # Calculate bifurcation ratio from stream orders
        stream_orders = {k: v for k, v in current_results.items() if 'stream_order_' in k and '_count' in k}

        if len(stream_orders) >= 2:
            # Extract order numbers and counts
            order_counts = {}
            for key, value in stream_orders.items():
                order_num = int(key.split('_')[2])
                order_counts[order_num] = value

            # Calculate bifurcation ratios
            bifurcation_ratios = []
            orders_sorted = sorted(order_counts.keys())

            for i in range(len(orders_sorted) - 1):
                lower_order = orders_sorted[i]
                higher_order = orders_sorted[i + 1]
                if order_counts[higher_order] > 0:
                    rb = order_counts[lower_order] / order_counts[higher_order]
                    bifurcation_ratios.append(rb)

            if bifurcation_ratios:
                results["bifurcation_ratio_mean"] = np.mean(bifurcation_ratios)
                results["bifurcation_ratio_std"] = np.std(bifurcation_ratios)

        # Calculate drainage density
        if "total_stream_length_km" in current_results and "catchment_area_km2" in current_results:
            drainage_density = current_results["total_stream_length_km"] / current_results["catchment_area_km2"]
            results["drainage_density_km_per_km2"] = drainage_density

        return results

    def process(self) -> Dict[str, Any]:
        """
        Process all hydrological attributes.

        Returns:
            Dictionary of hydrological attributes
        """
        results: Dict[str, Any] = {}

        # Water balance
        wb_results = self.calculate_water_balance()
        results.update(wb_results)

        # Streamflow signatures
        ss_results = self.calculate_streamflow_signatures()
        results.update(ss_results)

        # Baseflow
        bf_results = self.calculate_baseflow_attributes()
        results.update(bf_results)

        # River network (needs existing results)
        rn_results = self.enhance_river_network_analysis(results)
        results.update(rn_results)

        return results
