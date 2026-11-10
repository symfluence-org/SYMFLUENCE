# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>


"""
GNN Model Runner.

Orchestrates the GNN model workflow: data loading, graph construction, training, and simulation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from symfluence.core.exceptions import OptionalDependencyError

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as _err:
    raise OptionalDependencyError("The GNN model", "ml", dependency="torch") from _err

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.execution import SpatialOrchestrator
from symfluence.core.registries import R

from .model import GNNModel
from .postprocessor import GNNPostProcessor
from .preprocessor import GNNPreProcessor


@R.runners.add('GNN', runner_method='run_gnn')
class GNNRunner(BaseModelRunner, SpatialOrchestrator):  # type: ignore[misc]
    """Runner for the Spatio-Temporal GNN Hydrological Model.

    Orchestrates the complete GNN workflow: data loading, graph construction,
    model training or loading, and forward simulation. Handles GPU/CPU device
    selection and manages model checkpoints.

    The runner expects a distributed domain (not lumped) and issues a warning
    if lumped mode is detected, as GNN requires spatial structure to function.

    Workflow:
        1. Load forcing data (precipitation, temperature) and observations
        2. Load river network graph and construct adjacency matrix
        3. Preprocess and normalize data into PyTorch tensors
        4. Train model on historical data (or load pre-trained model)
        5. Run forward simulation to generate streamflow predictions
        6. Post-process results and save to output directory

    Attributes:
        device: torch.device (cuda if available, else cpu)
        preprocessor: GNNPreProcessor for data loading and normalization
        postprocessor: GNNPostProcessor for result formatting
        model: GNNModel instance (None until initialized)
        hru_ids: List of HRU identifiers in model
        outlet_indices: List of node indices at watersheds outlets
        outlet_hru_ids: List of HRU IDs at outlets (for output mapping)
    """

    MODEL_NAME = "GNN"

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the GNN model runner.

        Sets up the GNN execution environment including device selection
        (GPU if available), preprocessor for data/graph loading, and
        postprocessor for result formatting.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                GNN hyperparameters (hidden_size, epochs, learning_rate, etc.),
                paths, and domain settings.
            logger: Logger instance for status messages and debugging output.
            reporting_manager: Optional reporting manager for experiment tracking
                and visualization.

        Note:
            Issues warning if DOMAIN_DEFINITION_METHOD is 'lumped' since GNN
            requires spatial graph structure to function properly.
        """
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized GNN runner with device: {self.device}")

        # Check spatial mode
        domain_method = self._get_config_value(
            lambda: self.config.domain.definition_method, default='lumped'
        )
        if domain_method == 'lumped':
            self.logger.warning(
                "⚠️  GNN model requested in 'lumped' mode. GNN is designed for spatially distributed modeling "
                "with a graph structure. Consider using 'LSTM' for lumped modeling or change "
                "DOMAIN_DEFINITION_METHOD to 'delineate'."
            )

        self.preprocessor = GNNPreProcessor(
            self.config_dict,
            self.logger,
            self.project_dir,
            self.device
        )
        self.postprocessor = GNNPostProcessor(
            self.config_dict,
            self.logger,
            reporting_manager=self.reporting_manager
        )

        self.model: Optional[GNNModel] = None
        self.hru_ids: list[Any] = []
        self.outlet_indices: list[Any] = []
        self.outlet_hru_ids: list[Any] = []

    def run_gnn(self):
        """Run the complete GNN model workflow.

        Main orchestration method that coordinates data loading, preprocessing,
        model training/loading, and forward simulation. Wrapped with error
        handling to catch and report GNN-specific execution issues.

        Workflow:
            1. Load forcing data and observations from preprocessed files
            2. Load river network adjacency matrix (defines watershed connectivity)
            3. Decide: train new model vs. load pre-trained (GNN_LOAD config)
            4. Train: Run training loop with validation on 80/20 split
            5. Simulate: Generate forward predictions for full time period
            6. Save: Post-process and save results to NetCDF

        Configuration Dependencies:
            GNN_USE_SNOW (bool): Include snow cover as model input
            GNN_LOAD (bool): Load pre-trained model from checkpoint
            GNN_HIDDEN_SIZE (int): LSTM hidden dimension
            GNN_OUTPUT_SIZE (int): GNN layer output dimension
            GNN_EPOCHS (int): Training epochs
            GNN_BATCH_SIZE (int): Training batch size
            GNN_LEARNING_RATE (float): Adam optimizer learning rate
            GNN_DROPOUT (float): Dropout rate for regularization

        Raises:
            ModelExecutionError: If any step fails (data loading, training, simulation)
        """
        self.logger.info("Starting GNN model run")

        with symfluence_error_handler(
            "GNN model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # 1. Load Data & Graph
            forcing_df, streamflow_df, snow_df = self.preprocessor.load_data()

            # Load Graph to get adjacency
            adj_matrix = self.preprocessor.load_graph_structure()

            # 2. Preprocess
            use_snow = self._get_config_value(
                lambda: self.config.model.gnn.use_snow if self.config.model and self.config.model.gnn else None,
                default=False
            )
            snow_df_input = snow_df if use_snow else pd.DataFrame()

            load_existing_model = self._get_config_value(
                lambda: self.config.model.gnn.load if self.config.model and self.config.model.gnn else None,
                default=False
            )
            model_save_path = self.project_dir / 'models' / 'gnn_model.pt'

            if load_existing_model:
                self.logger.info("Loading pre-trained GNN model")
                checkpoint = self._load_model_checkpoint(model_save_path)

                self.preprocessor.set_scalers(
                    checkpoint['feature_scaler'],
                    checkpoint['target_scaler'],
                    checkpoint['output_size'],
                    checkpoint['target_names']
                )

                # Check if graph matches
                if checkpoint['adj_matrix_shape'] != list(adj_matrix.shape):
                    self.logger.warning("Loaded model graph shape mismatch! This may cause errors.")

                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=False
                )
                self.hru_ids = hru_ids
                self.outlet_indices = self.preprocessor.outlet_indices
                self.outlet_hru_ids = self.preprocessor.outlet_hru_ids

                self._create_model_instance(
                    input_size=X_tensor.shape[-1],
                    hidden_size=self._get_config_value(
                        lambda: self.config.model.gnn.hidden_size if self.config.model and self.config.model.gnn else None,
                        default=64
                    ),
                    gnn_output_size=self._get_config_value(
                        lambda: self.config.model.gnn.output_size if self.config.model and self.config.model.gnn else None,
                        default=32
                    ),
                    adjacency_matrix=adj_matrix
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])

            else:
                # Train
                # Compute train/val split BEFORE preprocessing to prevent data leakage
                # The scaler should only be fit on training data
                common_times = forcing_df.index.get_level_values('time').unique().intersection(
                    streamflow_df.index
                )
                if snow_df_input is not None and not snow_df_input.empty:
                    common_times = common_times.intersection(snow_df_input.index)
                n_timesteps = len(common_times)
                train_end_idx = int(0.8 * n_timesteps)
                self.logger.info(f"Training on first {train_end_idx} of {n_timesteps} timesteps (80/20 split)")

                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=True,
                    train_end_idx=train_end_idx
                )
                self.hru_ids = hru_ids
                self.outlet_indices = self.preprocessor.outlet_indices
                self.outlet_hru_ids = self.preprocessor.outlet_hru_ids

                self._create_model_instance(
                    input_size=X_tensor.shape[-1],
                    hidden_size=self._get_config_value(
                        lambda: self.config.model.gnn.hidden_size if self.config.model and self.config.model.gnn else None,
                        default=64
                    ),
                    gnn_output_size=self._get_config_value(
                        lambda: self.config.model.gnn.output_size if self.config.model and self.config.model.gnn else None,
                        default=32
                    ),
                    adjacency_matrix=adj_matrix
                )

                self._train_model(
                    X_tensor,
                    y_tensor,
                    epochs=self._get_config_value(
                        lambda: self.config.model.gnn.epochs if self.config.model and self.config.model.gnn else None,
                        default=100
                    ),
                    batch_size=self._get_config_value(
                        lambda: self.config.model.gnn.batch_size if self.config.model and self.config.model.gnn else None,
                        default=16
                    ),  # Smaller batch due to graph size
                    learning_rate=self._get_config_value(
                        lambda: self.config.model.gnn.learning_rate if self.config.model and self.config.model.gnn else None,
                        default=0.005
                    )
                )

                self.project_dir.joinpath('models').mkdir(exist_ok=True)
                self._save_model_checkpoint(model_save_path, adj_matrix)

            # 3. Simulate
            results = self._simulate(X_tensor, common_dates, hru_ids)

            # 4. Save Results
            output_file = self.postprocessor.save_results(
                results,
                hru_ids=self.hru_ids,
                outlet_hru_ids=self.outlet_hru_ids
            )
            self.logger.info(f"Results saved to {output_file}")

            # Self-training: score the run and write the standard
            # final_evaluation.json/best_params.json without a calibrate step.
            from symfluence.core.calibration.optimizers.final_evaluation.self_training import (
                persist_run_evaluation,
            )
            persist_run_evaluation(self, 'GNN')

    def _create_model_instance(self, input_size, hidden_size, gnn_output_size, adjacency_matrix):
        """Create the GNN model instance."""
        self.logger.info(f"Creating GNN model: In={input_size}, Hidden={hidden_size}, GNN_Out={gnn_output_size}")
        self.model = GNNModel(
            input_size=input_size,
            hidden_size=hidden_size,
            gnn_output_size=gnn_output_size,
            adjacency_matrix=adjacency_matrix,
            dropout_rate=float(self._get_config_value(
                lambda: self.config.model.gnn.dropout if self.config.model and self.config.model.gnn else None,
                default=0.2
            ))
        ).to(self.device)

    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, learning_rate: float):
        """Train the GNN model with adaptive masking for outlet nodes.

        This training routine handles the key challenge: observational data only
        exists at watershed outlets, not at internal nodes. The solution is to
        create a "mask" that isolates loss computation to observed outlet locations.

        Without masking, the model would learn to predict zero flow at unobserved
        internal nodes, which is physically wrong (water must exist internally even
        if not observed). The mask prevents this by down-weighting or ignoring
        internal node predictions during loss computation.

        Training Details:
            - 80/20 train/validation split of temporal dimension
            - Batch size should be small (16-32) because graph operations are memory-intensive
            - AdamW optimizer with gradient clipping to prevent exploding gradients
            - Loss computed only on observed (outlet) nodes via masking
            - Validation every epoch for early stopping heuristics

        Masking Strategy:
            If outlet_indices are known: Create one-hot mask on those nodes
            Otherwise: Infer from data variance (nodes with non-zero activity)

        Args:
            X: Training input tensor (Batch, Time, Nodes, Features).
                Batch dimension is random times from full dataset.
            y: Training target tensor (Batch, Nodes, Outputs).
                Only outlet nodes have meaningful values; internal nodes are zero.
            epochs: Number of training passes over the data
            batch_size: Number of time steps per batch (small due to graph structure)
            learning_rate: Adam optimizer learning rate (typically 0.001-0.01)

        Side Effects:
            - Updates self.model in-place via backpropagation
            - Logs epoch-wise training and validation loss
        """
        assert self.model is not None
        self.logger.info(f"Training GNN with {epochs} epochs, batch_size: {batch_size}")

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Mask: Identify which nodes have valid targets.
        # Assuming we only have data at outlets, and non-outlets are 0.
        # But scaling might have shifted 0.
        # A robust way is to check the variance of targets or use explicit mask.
        # Here we assume any node with non-constant target is observed?
        # Simpler: In preprocessor, we set outlets.
        # Let's derive mask from y: if y is constant (0-like) across time, maybe unobserved?
        # Actually, streamflow varies.
        # We will assume only the Outlet nodes contribute to loss.
        # To do this generically, we compute loss on all nodes but weight them?
        # Or just compute loss on nodes where we have data.
        # Since we don't pass a mask explicitly, we will assume y contains valid data where it matters.
        # If internal nodes are 0, the model will learn to predict 0 there? That's BAD.

        # FIX: We need a mask.
        # Let's create a mask based on variance of y in training set?
        # Or just use the 'outlets' logic again.
        # For now, let's assume y is correctly populated for outlets and we want to fit those.
        # If y is 0 for internal nodes, minimizing MSE will force flow to 0, which is physically wrong (water exists).
        # We should MASK out internal nodes from the loss.

        # Heuristic: Nodes with sum(abs(y)) > epsilon are observed.
        # Or pass mask from preprocessor.
        # Let's compute a mask on the whole dataset once.
        # y is (B, N, O). Sum over B and O.
        # Phase 4.3: Improved outlet selection with better error messages
        if self.outlet_indices:
            mask = torch.zeros(y.size(1), device=self.device)
            mask[self.outlet_indices] = 1.0
            self.logger.info(
                f"Using {len(self.outlet_indices)} configured outlet node(s) for loss calculation. "
                f"Outlet HRU IDs: {self.outlet_hru_ids[:5]}{'...' if len(self.outlet_hru_ids) > 5 else ''}"
            )
        else:
            # Fallback: infer outlets from data variance
            self.logger.warning(
                "No outlet_indices configured. Inferring observed nodes from target variance. "
                "For better results, configure outlet locations explicitly via pour point shapefile "
                "or CALIBRATION_OUTLET_ID config key."
            )
            y_activity = y.abs().sum(dim=(0, 2))
            mask = (y_activity > 1e-6).float().to(self.device)  # (N,)

            # Check if inference found any nodes
            active_count = mask.sum().item()
            if active_count == 0:
                raise ValueError(
                    "No active observation nodes found in target data. This typically means:\n"
                    "1. Target data contains all zeros or NaN - check observation preprocessing\n"
                    "2. Outlet nodes not properly configured - provide pour point shapefile\n"
                    "3. Data alignment issue - check time periods match between forcing and observations"
                )
            elif active_count > y.size(1) * 0.5:
                self.logger.warning(
                    f"Unusually high number of 'active' nodes ({int(active_count)}/{y.size(1)}). "
                    f"This may indicate improper target data setup. GNN expects observations "
                    f"only at watershed outlets, not all nodes."
                )

        self.logger.info(f"Training mask active for {mask.sum().item():.0f} nodes out of {len(mask)}")

        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss(reduction='none') # We will apply mask

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            # Shuffle batches
            indices = torch.randperm(X_train.size(0))

            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]

                optimizer.zero_grad()
                outputs = self.model(batch_X) # (B, N, 1)

                # Reshape batch_y if needed (B, N, O) -> (B, N, 1) for streamflow
                target = batch_y[:, :, 0:1] # Take streamflow

                loss_raw = criterion(outputs, target) # (B, N, 1)

                # Apply mask: (B, N, 1) * (N, 1 broadcast)
                masked_loss = loss_raw * mask.view(1, -1, 1)

                loss = masked_loss.sum() / (mask.sum() * batch_X.size(0) + 1e-6)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

            # Validation (batched to avoid OOM)
            self.model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_batch_size = min(batch_size * 4, 256)  # Larger batches OK for inference
                for j in range(0, X_val.size(0), val_batch_size):
                    val_batch_X = X_val[j:j + val_batch_size]
                    val_batch_y = y_val[j:j + val_batch_size]
                    val_out = self.model(val_batch_X)
                    val_target = val_batch_y[:, :, 0:1]
                    val_loss_raw = criterion(val_out, val_target)
                    val_loss_sum += (val_loss_raw * mask.view(1, -1, 1)).sum().item()
                val_loss = val_loss_sum / (mask.sum().item() * X_val.size(0) + 1e-6)

            avg_train_loss = total_loss / max(total_samples, 1)

            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

            # Phase 4.3: Periodic checkpoint saving every 50 epochs
            checkpoint_interval = self._get_config_value(
                lambda: self.config.model.gnn.checkpoint_interval if self.config.model and self.config.model.gnn else None,
                default=50
            )
            if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = self.sim_dir / f'gnn_checkpoint_epoch_{epoch+1}.pt'
                try:
                    # Get adjacency matrix for checkpoint
                    adj_matrix = self.model.gnn.adj
                    self._save_model_checkpoint(checkpoint_path, adj_matrix)
                    self.logger.info(f"Checkpoint saved at epoch {epoch+1}: {checkpoint_path}")
                except Exception as e:  # noqa: BLE001 — model execution resilience
                    self.logger.warning(f"Failed to save checkpoint at epoch {epoch+1}: {e}", exc_info=True)

    def _simulate(self, X: torch.Tensor, common_dates: pd.DatetimeIndex, hru_ids: List[int]) -> pd.DataFrame:
        """Run full forward simulation and return streamflow time series.

        Applies the trained model to generate streamflow predictions for the
        entire period (training + evaluation + forecast). Handles:
        - Batch processing for memory efficiency
        - Inverse scaling to convert predictions back to physical units
        - Multi-index DataFrame construction for time and HRU dimensions

        Why Separate Simulation from Training?
            Training uses a subset of data for computational efficiency. Simulation
            applies the model to the full time period to generate output for analysis,
            evaluation, and forecasting.

        Args:
            X: Full input tensor (Time_steps, Nodes, Features).
                Normalized by feature scaler used during training.
            common_dates: DatetimeIndex of valid times (length = X.shape[0] - lookback)
                The lookback period is excluded because the LSTM needs historical
                context to make its first prediction.
            hru_ids: List of HRU identifiers corresponding to node indices.

        Returns:
            pd.DataFrame: Multi-indexed results with shape (Time_steps, Nodes).
                Index: MultiIndex(time, hruId)
                Column: 'predicted_streamflow' in physical units (mm or m^3/s)

        Note:
            - Predictions are inverse-scaled using preprocessor.target_scaler
            - Batch size=50 for memory efficiency on large watersheds
            - Operates in eval mode with torch.no_grad() to disable gradients
        """
        assert self.model is not None
        self.logger.info("Running GNN simulation")
        self.model.eval()

        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                out = self.model(batch[0]) # (B, N, 1)
                predictions.append(out.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0) # (Total_Steps, N, 1)

        # Inverse transform
        # Target scaler was fitted on (Total, 1) or (Total*N, 1)
        # We need to apply inverse transform.
        # Our target scaler expects shape (..., 1) usually.
        B, N, n_out = predictions.shape
        preds_flat = predictions.reshape(-1, 1)
        preds_unscaled = self.preprocessor.target_scaler.inverse_transform(preds_flat)
        preds_restored = preds_unscaled.reshape(B, N, n_out)

        # Create DataFrame
        # MultiIndex: (Time, HRU)
        lookback = self.preprocessor.lookback
        time_idx = common_dates[lookback:]

        dfs = []
        for i, hru_id in enumerate(hru_ids):
            # Extract time series for this HRU
            data = preds_restored[:, i, 0]
            df = pd.DataFrame({'predicted_streamflow': data}, index=time_idx)
            df['hruId'] = hru_id
            dfs.append(df)

        result = pd.concat(dfs).reset_index().rename(columns={'index': 'time'})
        result = result.set_index(['time', 'hruId']).sort_index()

        return result

    def _save_model_checkpoint(self, path: Path, adj_matrix: torch.Tensor):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.preprocessor.feature_scaler,
            'target_scaler': self.preprocessor.target_scaler,
            'lookback': self.preprocessor.lookback,
            'output_size': self.preprocessor.output_size,
            'target_names': self.preprocessor.target_names,
            'adj_matrix_shape': list(adj_matrix.shape)
        }, path)

    def _load_model_checkpoint(self, path: Path):
        # weights_only=True prevents arbitrary code execution when deserializing
        # an untrusted checkpoint (CVE-2025-32434 lineage).
        return torch.load(path, map_location=self.device, weights_only=True)
