# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
GNN Model Preprocessor.

Handles data loading and graph structure construction for the GNN model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from symfluence.core.exceptions import OptionalDependencyError, ValidationError

try:
    import torch
except ImportError as _err:
    raise OptionalDependencyError("The GNN model", "ml", dependency="torch") from _err

# Import LSTM Preprocessor to inherit/reuse data loading logic
from ..lstm.preprocessor import LSTMPreProcessor

if TYPE_CHECKING:
    from symfluence.core.config.models import SymfluenceConfig


class GNNPreProcessor(LSTMPreProcessor):
    """
    Handles data preprocessing and graph construction for the GNN model.

    Extends LSTMPreprocessor to add graph structure loading from river network
    shapefiles and alignment of forcing/target data to graph node ordering.

    The preprocessor:
    1. Loads river network topology from shapefiles
    2. Constructs sparse adjacency matrix for message passing
    3. Aligns forcing data to graph node ordering
    4. Identifies outlet nodes for target assignment
    """


    MODEL_NAME = "GNN"
    def __init__(
        self,
        config: Union['SymfluenceConfig', Dict[str, Any]],
        logger: logging.Logger,
        project_dir: Optional[Path] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the GNN preprocessor.

        Args:
            config: SymfluenceConfig instance or configuration dictionary
                containing GNN model settings, lookback window, and feature
                scaling parameters.
            logger: Logger instance for status messages.
            project_dir: Optional path to project directory containing shapefiles.
                If provided, overrides the path derived from config.
            device: Optional PyTorch device for tensor allocation (CPU or CUDA).
                Defaults to CPU if not provided.

        Note:
            Graph structure is loaded lazily on first call to load_graph_structure()
            or process_data().
        """
        super().__init__(config, logger, project_dir, device)
        self.adj_matrix: Optional[torch.Tensor] = None
        self.node_mapping: Dict[int, int] = {}  # LINKNO -> Index
        self.hru_to_node: Dict[Any, int] = {}  # HRU_ID -> Index
        self.ordered_hru_ids: List[Any] = []
        self.outlet_indices: List[int] = []
        self.outlet_hru_ids: List[Any] = []

    def load_graph_structure(self) -> torch.Tensor:
        """
        Load the river network shapefile and build the adjacency matrix.

        Reads the river network shapefile, extracts connectivity from LINKNO
        and DSLINKNO columns, and constructs a row-normalized sparse adjacency
        matrix for GNN message passing.

        The adjacency matrix represents upstream-to-downstream flow:
        A[i,j] = 1/degree means node j flows to node i.

        Returns:
            torch.Tensor: Row-normalized sparse adjacency matrix (N, N) on device.

        Raises:
            FileNotFoundError: If no river network shapefile found.
            ValueError: If shapefile missing required columns (LINKNO, DSLINKNO, GRU_ID).

        Side effects:
            Sets self.adj_matrix, self.node_mapping, self.hru_to_node, self.ordered_hru_ids.
        """
        self.logger.info("Loading river network graph structure")

        # Path to river network shapefile
        # Assuming standard directory structure: data/domain/shapefiles/river_network/*.shp
        # Or using the config
        shapefile_dir = self.project_dir / 'shapefiles' / 'river_network'
        shapefiles = list(shapefile_dir.glob('*_riverNetwork_*.shp'))

        if not shapefiles:
            raise FileNotFoundError(f"No river network shapefile found in {shapefile_dir}")

        shp_path = shapefiles[0]
        self.logger.info(f"Reading shapefile: {shp_path}")

        gdf = gpd.read_file(shp_path)

        # Ensure we have required columns
        required_cols = ['LINKNO', 'DSLINKNO', 'GRU_ID'] # GRU_ID maps to HRU
        for col in required_cols:
            if col not in gdf.columns:
                raise ValueError(f"Shapefile missing required column: {col}")

        # Sort by LINKNO to have deterministic ordering
        gdf = gdf.sort_values('LINKNO').reset_index(drop=True)

        nodes = gdf['LINKNO'].values
        n_nodes = len(nodes)

        # Create mapping LINKNO -> Index
        self.node_mapping = {link: i for i, link in enumerate(nodes)}

        # Create mapping GRU_ID (HRU) -> Index
        # Assuming 1-to-1 mapping between River Segments and Subbasins (HRUs)
        self.hru_to_node = {row['GRU_ID']: self.node_mapping[row['LINKNO']]
                            for _, row in gdf.iterrows()}
        self.ordered_hru_ids = [row['GRU_ID'] for _, row in gdf.iterrows()]

        # Build Adjacency Indices
        # Flow is Upstream (j) -> Downstream (i)
        # So A_ij = 1 if j flows to i.
        rows = [] # Downstream (i)
        cols = [] # Upstream (j)

        for _, row in gdf.iterrows():
            u_link = row['LINKNO']     # This is the upstream node (source of flow)
            d_link = row['DSLINKNO']   # This is the downstream node (destination)

            # Map to indices
            if u_link in self.node_mapping and d_link in self.node_mapping:
                u_idx = self.node_mapping[u_link]
                d_idx = self.node_mapping[d_link]

                # Edge: u_idx -> d_idx
                # In Adjacency matrix A:
                # If we use A @ X (Row-normalized or standard agg), A_ij = 1 usually means j->i
                # So row = destination (d_idx), col = source (u_idx)
                rows.append(d_idx)
                cols.append(u_idx)

        indices = torch.LongTensor([rows, cols])
        values = torch.ones(len(rows))

        # Row-normalize adjacency for stable aggregation
        if rows:
            row_counts = torch.bincount(indices[0], minlength=n_nodes).float()
            values = values / row_counts[indices[0]]

        self.adj_matrix = torch.sparse_coo_tensor(
            indices, values, (n_nodes, n_nodes), dtype=torch.float32
        ).coalesce().to(self.device)

        self.logger.info(f"Graph constructed with {n_nodes} nodes and {len(rows)} edges.")

        assert self.adj_matrix is not None
        return self.adj_matrix

    def process_data(
        self,
        forcing_df: pd.DataFrame,
        streamflow_df: pd.DataFrame,
        snow_df: Optional[pd.DataFrame] = None,
        fit_scalers: bool = True,
        train_end_idx: Optional[int] = None,
        scaler_fit_dates: Optional[pd.DatetimeIndex] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex, pd.DataFrame, List[int]]:
        """
        Preprocess data and align it with the graph nodes.

        Overrides LSTMPreprocessor to ensure forcing data ordering matches
        graph node ordering (self.ordered_hru_ids). Also handles target
        assignment to outlet nodes for streamflow prediction.

        Args:
            forcing_df: Multi-indexed DataFrame (time, hruId) with forcing variables
                (precipitation, temperature, etc.).
            streamflow_df: DataFrame with 'streamflow' column indexed by time.
            snow_df: Optional DataFrame with SWE observations indexed by time.
            fit_scalers: If True, fit scalers to data; if False, use existing scalers.
            train_end_idx: If provided and fit_scalers=True, scalers are fitted
                only on data up to this index to prevent data leakage from
                validation/test data. This should be the number of unique
                timesteps in the training set.

            scaler_fit_dates: Explicit training dates for both scalers; takes
                precedence over train_end_idx.

        Returns:
            Tuple containing:
            - X_tensor: Input sequences (batch, lookback, nodes, features)
            - y_tensor: Target values (batch, nodes, outputs)
            - dates: DatetimeIndex of common dates
            - forcing_df: Aligned forcing DataFrame
            - ordered_hru_ids: List of HRU IDs in graph node order

        Raises:
            ValueError: If forcing data missing HRUs required by graph structure.
        """
        self.logger.info("Preprocessing data for GNN (Aligned with Graph Nodes)")

        # 1. Ensure graph is loaded
        if not self.node_mapping:
            self.load_graph_structure()

        # 2. Align DataFrames to common dates (Reuse parent logic manually or call it)
        # Calling parent logic is tricky because it returns tensors.
        # We need the aligned DataFrames first. Let's replicate the alignment logic.

        common_dates = forcing_df.index.get_level_values('time').unique().intersection(streamflow_df.index)
        if snow_df is not None and not snow_df.empty:
            common_dates = common_dates.intersection(snow_df.index)

        forcing_df = forcing_df.loc[pd.IndexSlice[common_dates, :], :]
        streamflow_df = streamflow_df.loc[common_dates]
        if snow_df is not None and not snow_df.empty:
            snow_df = snow_df.loc[common_dates]

        # 3. Reorder Forcing Data to match Graph Node Order
        # self.ordered_hru_ids contains the HRU IDs in the order of graph nodes (0..N-1)

        # Pivot forcing to (Time, HRU, Feature)
        forcing_df = forcing_df.reset_index()

        # Check if all graph HRUs are in forcing
        avail_hrus = set(forcing_df['hruId'].unique())
        missing = set(self.ordered_hru_ids) - avail_hrus
        if missing:
            raise ValueError(f"Forcing data missing HRUs required by graph: {missing}")

        # Filter forcing to only graph HRUs
        forcing_df = forcing_df[forcing_df['hruId'].isin(self.ordered_hru_ids)]

        # Sort/Pivot
        feature_columns = forcing_df.columns.drop(
            ['time', 'hruId', 'hru', 'latitude', 'longitude']
            if 'time' in forcing_df.columns and 'hruId' in forcing_df.columns else []
        )

        # Reindex to ensure order: time (primary), hruId (secondary matching ordered_hru_ids)
        # Make hruId categorical with specific order
        forcing_df['hruId'] = pd.Categorical(forcing_df['hruId'], categories=self.ordered_hru_ids, ordered=True)
        forcing_df = forcing_df.sort_values(['time', 'hruId'])

        # Now features_to_scale
        features_to_scale = forcing_df[feature_columns]

        # Scale features
        if fit_scalers:
            if scaler_fit_dates is not None:
                fit_mask = forcing_df['time'].isin(scaler_fit_dates)
                if not fit_mask.any():
                    raise ValidationError('No GNN feature samples within scaler training dates')
                self.feature_scaler.fit(features_to_scale.loc[fit_mask])
            elif train_end_idx is not None:
                # Fit scaler only on training data to prevent data leakage
                # For GNN, features_to_scale is (n_timesteps * n_nodes, n_features)
                n_nodes = len(self.ordered_hru_ids)
                train_samples = train_end_idx * n_nodes
                self.feature_scaler.fit(features_to_scale[:train_samples])
            else:
                self.feature_scaler.fit(features_to_scale)
            scaled_features = self.feature_scaler.transform(features_to_scale)
        else:
            scaled_features = self.feature_scaler.transform(features_to_scale)

        scaled_features = np.clip(scaled_features, -10, 10)

        # 4. Prepare Targets
        # Streamflow is usually at the outlet.
        # We need to identify which node is the outlet or if we have distributed targets.
        # For training, we construct a target tensor (B, N, O).
        # Most nodes will have missing data (NaN) or 0 if not observed.
        # We will use a mask in the loss function, or just fill with 0 and ignore.

        # Create full target array (Time, Nodes, Output)
        n_timesteps = len(common_dates)
        n_nodes = len(self.ordered_hru_ids)

        if snow_df is not None and not snow_df.empty:
            self.output_size = 2
            self.target_names = ['streamflow', 'SWE']
        else:
            self.output_size = 1
            self.target_names = ['streamflow']

        targets_full = np.zeros((n_timesteps, n_nodes, self.output_size))

        # Find Outlet Node(s) - Assumed to be where streamflow_df data applies
        # We identify outlets as nodes that do not appear as sources (upstream) in the adjacency matrix.

        # Recover adjacency indices
        adj_indices = self.adj_matrix.indices().cpu().numpy()
        sources = set(adj_indices[1]) # Nodes that flow to someone (upstream ends of edges)
        all_nodes = set(range(n_nodes))
        outlets = list(all_nodes - sources)

        if not outlets:
            self.logger.warning("No outlet (sink) node found in graph! Circular or infinite?")
            # Fallback: Use the last node
            outlets = [n_nodes - 1]

        self.logger.info(f"Identified outlet nodes (indices): {outlets}")
        self.outlet_indices = outlets
        self.outlet_hru_ids = [self.ordered_hru_ids[idx] for idx in outlets]

        # Fill targets
        # streamflow_df is (Time, 1) or Series
        q_vals = streamflow_df['streamflow'].values
        if fit_scalers:
            if scaler_fit_dates is not None:
                target_fit_mask = np.asarray(common_dates.isin(scaler_fit_dates), dtype=bool)
                self.target_scaler.fit(q_vals[target_fit_mask].reshape(-1, 1))
            elif train_end_idx is not None:
                # Fit scaler only on training data to prevent data leakage
                self.target_scaler.fit(q_vals[:train_end_idx].reshape(-1, 1))
            else:
                self.target_scaler.fit(q_vals.reshape(-1, 1))

        q_scaled = self.target_scaler.transform(q_vals.reshape(-1, 1)).flatten()

        # Assign to outlet nodes
        for out_idx in outlets:
            targets_full[:, out_idx, 0] = q_scaled

        # 5. Create Sequences (B, T, N, F)
        X, y = [], []

        # scaled_features is (Time * HRU, Features). Reshape to (Time, HRU, Features)
        feat_reshaped = scaled_features.reshape(n_timesteps, n_nodes, -1)

        for i in range(n_timesteps - self.lookback):
            X.append(feat_reshaped[i:(i + self.lookback)])
            y.append(targets_full[i + self.lookback]) # (N, O)

        X_tensor = torch.FloatTensor(np.array(X)).to(self.device)
        y_tensor = torch.FloatTensor(np.array(y)).to(self.device)

        return X_tensor, y_tensor, pd.DatetimeIndex(common_dates), forcing_df, self.ordered_hru_ids
