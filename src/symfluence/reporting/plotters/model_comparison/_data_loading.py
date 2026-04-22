# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""Data collection and file-loading helpers for model comparison plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from symfluence.core.constants import ConfigKeys


class ModelComparisonDataLoadingMixin:
    """Load model outputs/observations and normalize them for plotting."""

    def _collect_model_data(
        self,
        experiment_id: str,
        context: str = "run_model",
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Load model results and observations for comparison plotting."""
        try:
            if context == "calibrate_model":
                calibrated_results_file = self.project_dir / "results" / f"{experiment_id}_calibrated_results.csv"
                if calibrated_results_file.exists():
                    results_df = pd.read_csv(calibrated_results_file, index_col=0, parse_dates=True)
                    self.logger.info(f"Loaded calibrated results from: {calibrated_results_file}")
                else:
                    self.logger.info("Calibrated results file not found, auto-detecting from final_evaluation...")
                    results_df = self._auto_collect_model_outputs(experiment_id, context)
            else:
                results_file = self.project_dir / "results" / f"{experiment_id}_results.csv"
                if results_file.exists():
                    results_df = pd.read_csv(results_file, index_col=0, parse_dates=True)
                else:
                    self.logger.info("Results file not found, auto-detecting model outputs...")
                    results_df = self._auto_collect_model_outputs(experiment_id, context)

                if results_df is not None and not results_df.empty:
                    results_file.parent.mkdir(parents=True, exist_ok=True)
                    results_df.to_csv(results_file)
                    self.logger.info(f"Auto-generated results saved to: {results_file}")
                else:
                    self.logger.warning("No model outputs found to collect")
                    return None, None

            if context == "calibrate_model" and results_df is not None and not results_df.empty:
                calibrated_results_file = self.project_dir / "results" / f"{experiment_id}_calibrated_results.csv"
                calibrated_results_file.parent.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(calibrated_results_file)
                self.logger.info(f"Calibrated results saved to: {calibrated_results_file}")

            obs_series = None
            for col in results_df.columns:
                if "obs" in col.lower() or "observed" in col.lower():
                    obs_series = results_df[col]
                    break

            if obs_series is None:
                obs_series = self._load_observations(results_df.index)
                if obs_series is not None:
                    results_df["observed_discharge_cms"] = obs_series

            spinup_end = self._get_spinup_end_date()
            if spinup_end is not None and results_df is not None:
                original_len = len(results_df)
                results_df = results_df[results_df.index > spinup_end]
                if obs_series is not None:
                    obs_series = obs_series[obs_series.index > spinup_end]
                filtered_count = original_len - len(results_df)
                if filtered_count > 0:
                    self.logger.info(
                        f"Filtered {filtered_count} spinup timesteps (before {spinup_end.strftime('%Y-%m-%d')})"
                    )

            return results_df, obs_series

        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.error(f"Error collecting model data: {e}")
            return None, None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.error(f"Error collecting model data: {e}")
            return None, None

    def _auto_collect_model_outputs(
        self,
        experiment_id: str,
        context: str = "run_model",
    ) -> Optional[pd.DataFrame]:
        """Auto-detect and load model outputs from simulations directories."""
        if context == "calibrate_model":
            final_eval_dir = self.project_dir / "optimization" / "final_evaluation"
            if final_eval_dir.exists():
                self.logger.info(f"Loading calibrated outputs from: {final_eval_dir}")
                results_df = self._load_from_directory(final_eval_dir, experiment_id, label_suffix="_calibrated")
                if results_df is not None:
                    return results_df
                self.logger.info("No outputs in final_evaluation, checking simulations directory...")

        sim_dir = self.project_dir / "simulations" / experiment_id
        if not sim_dir.exists():
            return None

        basin_area_m2 = self._get_basin_area_m2()
        spinup_end = self._get_spinup_end_date()

        loaders = [
            lambda: self._load_mizuroute_from_simulations(sim_dir),
            lambda: self._load_troute_from_simulations(sim_dir),
            lambda: self._load_summa_from_simulations(sim_dir, basin_area_m2),
            lambda: self._load_fuse_from_simulations(sim_dir, basin_area_m2),
            lambda: self._load_gr_from_simulations(sim_dir),
            lambda: self._load_clm_from_simulations(sim_dir),
        ]

        results_df = None
        for load in loaders:
            results_df = load()
            if results_df is not None:
                break

        if results_df is None:
            return None

        obs_series = self._load_observations(results_df.index)
        if obs_series is not None:
            results_df["observed_discharge_cms"] = obs_series

        if spinup_end is not None:
            results_df = results_df[results_df.index > spinup_end]
        return results_df

    def _select_outlet_series(
        self,
        var_data,
        dataset=None,
        sim_reach_id: Optional[Any] = None,
    ) -> pd.Series:
        if "seg" in var_data.dims:
            if sim_reach_id and dataset is not None and "reachID" in dataset:
                segment_mask = dataset["reachID"].values == int(sim_reach_id)
                if segment_mask.any():
                    idx = int(segment_mask.argmax())
                    return var_data.isel(seg=idx).to_pandas()
            outlet_idx = int(np.argmax(var_data.mean(dim="time").values))
            return var_data.isel(seg=outlet_idx).to_pandas()

        if "reachID" in var_data.dims:
            outlet_idx = int(np.argmax(var_data.mean(dim="time").values))
            return var_data.isel(reachID=outlet_idx).to_pandas()

        if "feature_id" in var_data.dims:
            outlet_idx = int(np.argmax(var_data.mean(dim="time").values))
            return var_data.isel(feature_id=outlet_idx).to_pandas()

        return var_data.squeeze().to_pandas()

    def _load_mizuroute_from_simulations(self, sim_dir: Path) -> Optional[pd.DataFrame]:
        import xarray as xr

        mizu_dir_candidates = [
            sim_dir / "mizuRoute",
            sim_dir / "SUMMA" / "mizuRoute",
        ]
        mizu_files: List[Path] = []
        for candidate_dir in mizu_dir_candidates:
            if candidate_dir.exists():
                candidate_files = list(candidate_dir.glob("*.nc"))
                if candidate_files:
                    mizu_files = candidate_files
                    self.logger.debug(f"Found mizuRoute output in: {candidate_dir}")
                    break

        if not mizu_files:
            return None

        self.logger.info(f"Found {len(mizu_files)} mizuRoute output files")
        try:
            if len(mizu_files) > 1:
                ds = xr.open_mfdataset(
                    sorted(mizu_files),
                    combine="by_coords",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="override",
                )
                self.logger.info(f"Loaded {len(mizu_files)} files with open_mfdataset")
            else:
                ds = xr.open_dataset(mizu_files[0])

            with ds:
                routing_vars = [
                    "IRFroutedRunoff",
                    "KWTroutedRunoff",
                    "sumUpstreamRunoff",
                    "dlayRunoff",
                    "averageRoutedRunoff",
                ]
                routing_var = next((name for name in routing_vars if name in ds), None)
                if routing_var is None:
                    self.logger.warning(
                        "No recognized routing variable found in mizuRoute output. "
                        f"Available: {list(ds.data_vars)}"
                    )
                    return None

                self.logger.info(f"Using mizuRoute variable: {routing_var}")
                sim_reach_id = self._get_config_value(
                    lambda: self.config.routing.sim_reach_id,
                    default=None,
                    dict_key=ConfigKeys.SIM_REACH_ID,
                )
                streamflow = self._select_outlet_series(ds[routing_var], dataset=ds, sim_reach_id=sim_reach_id)
                streamflow.index = streamflow.index.round("h")
                streamflow = streamflow.resample("D").mean()

                results_df = pd.DataFrame(index=streamflow.index)
                results_df["SUMMA_discharge_cms"] = streamflow
                self.logger.info(f"Loaded {len(streamflow)} discharge values from mizuRoute")
                return results_df
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Error loading mizuRoute output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading mizuRoute output: {e}")
            return None

    def _load_troute_from_simulations(self, sim_dir: Path) -> Optional[pd.DataFrame]:
        import xarray as xr

        troute_dir = sim_dir / "TRoute"
        troute_files = list(troute_dir.glob("troute_output.nc")) if troute_dir.exists() else []
        if not troute_files and troute_dir.exists():
            troute_files = list(troute_dir.glob("*.nc"))
        if not troute_files:
            return None

        self.logger.info("Found T-Route output")
        try:
            with xr.open_dataset(troute_files[0]) as ds:
                flow_var = next((name for name in ["flow", "streamflow", "discharge", "q_lateral"] if name in ds), None)
                if flow_var is None:
                    self.logger.warning(
                        f"No recognized flow variable in T-Route output. Available: {list(ds.data_vars)}"
                    )
                    return None

                streamflow = self._select_outlet_series(ds[flow_var])
                if hasattr(streamflow.index, "freq") or len(streamflow) > 366 * 10:
                    streamflow.index = streamflow.index.round("h")
                    streamflow = streamflow.resample("D").mean()

                results_df = pd.DataFrame(index=streamflow.index)
                results_df["TRoute_discharge_cms"] = streamflow
                self.logger.info(
                    f"Loaded {len(streamflow)} discharge values from T-Route (variable: {flow_var})"
                )
                return results_df
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Error loading T-Route output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading T-Route output: {e}")
            return None

    def _load_summa_from_simulations(
        self,
        sim_dir: Path,
        basin_area_m2: Optional[float],
    ) -> Optional[pd.DataFrame]:
        import xarray as xr

        summa_dir = sim_dir / "SUMMA"
        summa_files = list(summa_dir.glob("*_timestep.nc")) if summa_dir.exists() else []
        if not summa_files:
            return None

        self.logger.info("Found SUMMA lumped output")
        try:
            with xr.open_dataset(summa_files[0]) as ds:
                if "averageRoutedRunoff" not in ds:
                    return None

                var = ds["averageRoutedRunoff"]
                if "hru" in var.dims and var.sizes["hru"] > 1:
                    if "HRUarea" in ds:
                        weights = ds["HRUarea"] / ds["HRUarea"].sum()
                        streamflow = (var * weights).sum(dim="hru").to_pandas()
                    else:
                        streamflow = var.mean(dim="hru").to_pandas()
                else:
                    streamflow = var.squeeze().to_pandas()

                if basin_area_m2:
                    streamflow = streamflow * basin_area_m2

                results_df = pd.DataFrame(index=streamflow.index)
                results_df["SUMMA_discharge_cms"] = streamflow
                return results_df
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Error loading SUMMA output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading SUMMA output: {e}")
            return None

    def _load_fuse_from_simulations(
        self,
        sim_dir: Path,
        basin_area_m2: Optional[float],
    ) -> Optional[pd.DataFrame]:
        import xarray as xr

        fuse_dir = sim_dir / "FUSE"
        fuse_files = list(fuse_dir.glob("*_runs_best.nc")) if fuse_dir.exists() else []
        if not fuse_files:
            return None

        self.logger.info("Found FUSE output")
        try:
            with xr.open_dataset(fuse_files[0]) as ds:
                if "q_routed" not in ds:
                    return None
                streamflow = ds["q_routed"].isel(param_set=0, latitude=0, longitude=0).to_pandas()
                if basin_area_m2:
                    streamflow = streamflow * (basin_area_m2 / 1e6) / 86400

                results_df = pd.DataFrame(index=streamflow.index)
                results_df["FUSE_discharge_cms"] = streamflow
                return results_df
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Error loading FUSE output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading FUSE output: {e}")
            return None

    def _load_gr_from_simulations(self, sim_dir: Path) -> Optional[pd.DataFrame]:
        gr_dir = sim_dir / "GR"
        gr_files = list(gr_dir.glob("GR_results.csv")) if gr_dir.exists() else []
        if not gr_files:
            return None

        self.logger.info("Found GR output")
        try:
            gr_df = pd.read_csv(gr_files[0], parse_dates=["Date"])
            gr_df.set_index("Date", inplace=True)
            flow_col = next((col for col in gr_df.columns if "sim" in col.lower() or "qsim" in col.lower()), None)
            if flow_col is None:
                return None

            results_df = pd.DataFrame(index=gr_df.index)
            results_df["GR_discharge_cms"] = gr_df[flow_col]
            return results_df
        except (OSError, KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"Error loading GR output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading GR output: {e}")
            return None

    def _load_clm_from_simulations(self, sim_dir: Path) -> Optional[pd.DataFrame]:
        import xarray as xr

        clm_dir = sim_dir / "CLM"
        clm_files = sorted(clm_dir.glob("*.clm2.h0.*.nc")) if clm_dir.exists() else []
        if not clm_files:
            return None

        self.logger.info(f"Found {len(clm_files)} CLM history file(s)")
        try:
            with xr.open_mfdataset(
                [str(path) for path in clm_files],
                combine="by_coords",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            ) as ds:
                if "QRUNOFF" not in ds:
                    return None
                qrunoff = ds["QRUNOFF"].values.squeeze()
                area_m2 = float(
                    self._get_config_value(
                        lambda: self.config.domain.catchment_area,
                        default=2210.0,
                    )
                ) * 1e6
                streamflow = qrunoff * area_m2 / 1000.0
                time_idx = pd.DatetimeIndex(ds.time.values)
                return pd.DataFrame({"CLM_discharge_cms": streamflow}, index=time_idx)
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
            self.logger.warning(f"Error loading CLM output: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading CLM output: {e}")
            return None

    def _load_from_directory(
        self,
        output_dir: Path,
        experiment_id: str,
        label_suffix: str = "",
    ) -> Optional[pd.DataFrame]:
        """Load model outputs from an arbitrary directory."""
        del experiment_id

        import xarray as xr

        if not output_dir.exists():
            return None

        results_df = None
        basin_area_m2 = self._get_basin_area_m2()

        spinup_period = self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default="",
            dict_key=ConfigKeys.SPINUP_PERIOD,
        )
        spinup_end = None
        if spinup_period and "," in str(spinup_period):
            try:
                spinup_end = pd.to_datetime(str(spinup_period).split(",")[1].strip())
            except (ValueError, pd.errors.ParserError, OverflowError):
                pass

        summa_files = list(output_dir.glob("*_timestep.nc"))
        if summa_files:
            self.logger.info(f"Found SUMMA output in {output_dir}")
            try:
                ds = xr.open_dataset(summa_files[0])
                if "averageRoutedRunoff" in ds:
                    var_data = ds["averageRoutedRunoff"]

                    if "hru" in var_data.dims:
                        streamflow = var_data.isel(hru=0).to_pandas()
                    elif "gru" in var_data.dims:
                        streamflow = var_data.isel(gru=0).to_pandas()
                    else:
                        streamflow = var_data.to_pandas()

                    if basin_area_m2:
                        streamflow = streamflow * basin_area_m2

                    if hasattr(streamflow.index, "freq") or len(streamflow) > 365 * 4:
                        streamflow.index = pd.to_datetime(streamflow.index)
                        streamflow = streamflow.resample("D").mean()

                    results_df = pd.DataFrame(index=streamflow.index)
                    col_name = f"SUMMA{label_suffix}_discharge_cms"
                    results_df[col_name] = streamflow
                    self.logger.info(f"Loaded {len(streamflow)} discharge values from calibrated SUMMA output")
                ds.close()
            except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
                self.logger.warning(f"Error loading SUMMA output from {output_dir}: {e}")
            except (ImportError, AttributeError, IndexError) as e:
                self.logger.warning(f"Error loading SUMMA output from {output_dir}: {e}")

        if results_df is None:
            mizu_files = list(output_dir.glob("*.nc"))
            for mizu_file in mizu_files:
                try:
                    ds = xr.open_dataset(mizu_file)
                    routing_vars = ["IRFroutedRunoff", "KWTroutedRunoff", "sumUpstreamRunoff", "dlayRunoff"]
                    routing_var = None
                    for var_name in routing_vars:
                        if var_name in ds:
                            routing_var = var_name
                            break

                    if routing_var:
                        self.logger.info(f"Found mizuRoute output: {routing_var}")
                        var_data = ds[routing_var]

                        if "seg" in var_data.dims:
                            segment_means = var_data.mean(dim="time").values
                            outlet_idx = int(np.argmax(segment_means))
                            streamflow = var_data.isel(seg=outlet_idx).to_pandas()
                        elif "reachID" in var_data.dims:
                            reach_means = var_data.mean(dim="time").values
                            outlet_idx = int(np.argmax(reach_means))
                            streamflow = var_data.isel(reachID=outlet_idx).to_pandas()
                        else:
                            streamflow = var_data.to_pandas()

                        streamflow.index = pd.to_datetime(streamflow.index)
                        streamflow = streamflow.resample("D").mean()

                        results_df = pd.DataFrame(index=streamflow.index)
                        col_name = f"SUMMA{label_suffix}_discharge_cms"
                        results_df[col_name] = streamflow
                        ds.close()
                        break
                    ds.close()
                except (OSError, KeyError, ValueError, TypeError):
                    continue

        if results_df is None:
            troute_files = list(output_dir.glob("troute_output.nc"))
            if not troute_files:
                troute_files = [f for f in output_dir.glob("*.nc") if "troute" in f.name.lower()]
            if troute_files:
                self.logger.info(f"Found T-Route output in {output_dir}")
                try:
                    ds = xr.open_dataset(troute_files[0])
                    flow_var = None
                    for vname in ["flow", "streamflow", "discharge"]:
                        if vname in ds:
                            flow_var = vname
                            break

                    if flow_var:
                        var_data = ds[flow_var]
                        if "feature_id" in var_data.dims:
                            feature_means = var_data.mean(dim="time").values
                            outlet_idx = int(np.argmax(feature_means))
                            streamflow = var_data.isel(feature_id=outlet_idx).to_pandas()
                        elif "seg" in var_data.dims:
                            seg_means = var_data.mean(dim="time").values
                            outlet_idx = int(np.argmax(seg_means))
                            streamflow = var_data.isel(seg=outlet_idx).to_pandas()
                        else:
                            streamflow = var_data.squeeze().to_pandas()

                        streamflow.index = pd.to_datetime(streamflow.index)
                        streamflow = streamflow.resample("D").mean()

                        results_df = pd.DataFrame(index=streamflow.index)
                        col_name = f"TRoute{label_suffix}_discharge_cms"
                        results_df[col_name] = streamflow
                    ds.close()
                except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
                    self.logger.warning(f"Error loading T-Route output: {e}")
                except (ImportError, AttributeError, IndexError) as e:
                    self.logger.warning(f"Error loading T-Route output from {output_dir}: {e}")

        if results_df is None:
            clm_files = sorted(output_dir.glob("*.clm2.h0.*.nc"))
            if clm_files:
                self.logger.info(f"Found {len(clm_files)} CLM history file(s)")
                try:
                    ds = xr.open_mfdataset(
                        [str(f) for f in clm_files],
                        combine="by_coords",
                        data_vars="minimal",
                        coords="minimal",
                        compat="override",
                    )
                    if "QRUNOFF" in ds:
                        qrunoff = ds["QRUNOFF"].values.squeeze()
                        area = basin_area_m2 or (
                            float(self._get_config_value(lambda: self.config.domain.catchment_area, default=2210.0))
                            * 1e6
                        )
                        streamflow = qrunoff * area / 1000.0
                        time_idx = pd.DatetimeIndex(ds.time.values)
                        results_df = pd.DataFrame(index=time_idx)
                        col_name = f"CLM{label_suffix}_discharge_cms"
                        results_df[col_name] = streamflow
                    ds.close()
                except (OSError, KeyError, ValueError, TypeError, RuntimeError) as e:
                    self.logger.warning(f"Error loading CLM output: {e}")
                except (ImportError, AttributeError, IndexError) as e:
                    self.logger.warning(f"Error loading CLM output from {output_dir}: {e}")

        if results_df is not None:
            obs_series = self._load_observations(results_df.index)
            if obs_series is not None:
                results_df["observed_discharge_cms"] = obs_series

            if spinup_end is not None:
                results_df = results_df[results_df.index > spinup_end]

        return results_df

    def _load_observations(self, target_index: pd.DatetimeIndex) -> Optional[pd.Series]:
        domain_name = self._get_config_value(
            lambda: self.config.domain.name,
            dict_key=ConfigKeys.DOMAIN_NAME,
        )
        obs_path = self.project_observations_dir / "streamflow" / "preprocessed" / f"{domain_name}_streamflow_processed.csv"

        if not obs_path.exists():
            self.logger.warning(f"Observations file not found: {obs_path}")
            return None

        try:
            obs_df = pd.read_csv(obs_path, parse_dates=["datetime"])
            obs_df.set_index("datetime", inplace=True)

            for col in obs_df.columns:
                if "discharge" in col.lower() or col.lower() in ["q", "flow"]:
                    return obs_df[col].reindex(target_index)

            return None
        except (OSError, KeyError, ValueError, TypeError) as e:
            self.logger.warning(f"Error loading observations: {e}")
            return None
        except (ImportError, AttributeError, IndexError) as e:
            self.logger.warning(f"Error loading observations: {e}")
            return None

    def _get_basin_area_m2(self) -> Optional[float]:
        try:
            import geopandas as gpd

            domain_name = self._get_config_value(
                lambda: self.config.domain.name,
                dict_key=ConfigKeys.DOMAIN_NAME,
            )
            domain_method = self._get_config_value(
                lambda: self.config.domain.definition_method,
                dict_key=ConfigKeys.DOMAIN_DEFINITION_METHOD,
            )

            basin_path = self.project_dir / "shapefiles" / "river_basins" / f"{domain_name}_riverBasins_{domain_method}.shp"

            if not basin_path.exists():
                for alt_method in ["lumped", "delineate", "subset"]:
                    alt_path = self.project_dir / "shapefiles" / "river_basins" / f"{domain_name}_riverBasins_{alt_method}.shp"
                    if alt_path.exists():
                        basin_path = alt_path
                        break

            if basin_path.exists():
                gdf = gpd.read_file(str(basin_path))
                gdf_proj = gdf.to_crs("EPSG:32611")
                return float(gdf_proj.geometry.area.sum())

            return None
        except (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError, IndexError) as e:
            self.logger.warning(f"Could not determine basin area: {e}")
            return None

    def _get_spinup_end_date(self) -> Optional[pd.Timestamp]:
        spinup_period = self._get_config_value(
            lambda: self.config.domain.spinup_period,
            default="",
            dict_key=ConfigKeys.SPINUP_PERIOD,
        )

        if not spinup_period or not isinstance(spinup_period, str):
            return None
        if "," not in spinup_period:
            return None

        try:
            spinup_end_str = spinup_period.split(",")[1].strip()
            return pd.to_datetime(spinup_end_str)
        except (ValueError, pd.errors.ParserError, OverflowError) as e:
            self.logger.debug(f"Could not parse spinup period: {e}")
            return None
