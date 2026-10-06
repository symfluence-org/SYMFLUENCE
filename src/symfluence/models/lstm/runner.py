# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
LSTM (Flow and Snow Hydrological LSTM) model runner.

An LSTM-based model for hydrological predictions, specifically for streamflow
and snow water equivalent (SWE).
"""
from __future__ import annotations

import logging

# Security rationale: Used for trusted internal model serialization
import pickle  # nosec B403
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import psutil

from symfluence.core.exceptions import OptionalDependencyError, ValidationError

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as _err:
    raise OptionalDependencyError("The LSTM model", "ml", dependency="torch") from _err

try:
    import droute
    HAS_DROUTE = True
except ImportError:
    HAS_DROUTE = False

from symfluence.core.exceptions import ModelExecutionError, symfluence_error_handler
from symfluence.core.modeling.base import BaseModelRunner
from symfluence.core.modeling.execution import RoutingModel, SpatialOrchestrator
from symfluence.core.modeling.mixins import SpatialModeDetectionMixin
from symfluence.core.modeling.spatial_modes import SpatialMode
from symfluence.core.registries import R

from ..mizuroute.mixins import MizuRouteConfigMixin
from .model import LSTMModel
from .postprocessor import LSTMPostProcessor
from .preprocessor import LSTMPreProcessor


@R.runners.add('LSTM', runner_method='run_lstm')
class LSTMRunner(BaseModelRunner, SpatialOrchestrator, MizuRouteConfigMixin, SpatialModeDetectionMixin):  # type: ignore[misc]
    """
    LSTM: Flow and Snow Hydrological LSTM Runner.

    Orchestrates the LSTM model workflow: data loading, preprocessing,
    model training (or loading), simulation, and postprocessing.
    Supports both lumped and distributed modes with dRoute integration.
    """

    MODEL_NAME = "LSTM"

    @property
    def lstm_config(self):
        """Access LSTM-specific configuration."""
        return self.config.model.lstm

    def __init__(self, config: Dict[str, Any], logger: logging.Logger, reporting_manager: Optional[Any] = None):
        """
        Initialize the LSTM model runner.

        Sets up the LSTM execution environment including device selection
        (GPU if available), preprocessor for data loading/scaling, and
        postprocessor for result formatting.

        Args:
            config: Configuration dictionary or SymfluenceConfig object containing
                LSTM hyperparameters (hidden_size, num_layers, epochs, learning_rate),
                use_snow flag, and domain settings.
            logger: Logger instance for status messages and debugging output.
            reporting_manager: Optional reporting manager for experiment tracking
                and visualization.

        Note:
            Issues warning if DOMAIN_DEFINITION_METHOD is 'delineate' since GNN
            is better suited for distributed modeling with explicit topology.
        """
        # Call base class
        super().__init__(config, logger, reporting_manager=reporting_manager)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Initialized LSTM runner with device: {self.device}")

        # Determine spatial mode using mixin
        # LSTM has model-specific warnings for distributed mode in MODEL_SPATIAL_CAPABILITIES
        self.spatial_mode = self.detect_spatial_mode('LSTM')

        # Initialize components
        self.preprocessor = LSTMPreProcessor(
            self.config_dict,
            self.logger,
            self.project_dir,
            self.device
        )
        self.postprocessor = LSTMPostProcessor(
            self.config,
            self.logger,
            reporting_manager=self.reporting_manager
        )

        self.model: Optional[LSTMModel] = None
        self.hru_ids: list[Any] = []

    def run_lstm(self):
        """
        Run the complete LSTM model workflow.

        Main orchestration method that coordinates data loading, preprocessing,
        model training/loading, simulation, optional routing, and result saving.

        Workflow:
            1. Load forcing data, streamflow observations, and optional snow data
            2. Preprocess: scale features, create sequences with lookback window
            3. Either load pre-trained model or train new model:
               - Training: 80/20 train/validation split, Adam optimizer, MSE loss
               - Loading: Restore model weights and scalers from checkpoint
            4. Run forward simulation to generate predictions
            5. For distributed mode: optionally run dRoute or mizuRoute for routing
            6. Post-process results and save to NetCDF

        Configuration Dependencies:
            LSTM_LOAD (bool): Load pre-trained model vs train new
            LSTM_USE_SNOW (bool): Include SWE as output target
            LSTM_HIDDEN_SIZE (int): LSTM hidden dimension (default: 64)
            LSTM_NUM_LAYERS (int): Number of stacked LSTM layers (default: 2)
            LSTM_EPOCHS (int): Training epochs (default: 100)
            LSTM_BATCH_SIZE (int): Training batch size (default: 32)
            LSTM_LEARNING_RATE (float): Adam optimizer learning rate (default: 0.001)
            LSTM_DROPOUT (float): Dropout rate for regularization (default: 0.2)

        Raises:
            ModelExecutionError: If any step fails (data loading, training, simulation).
        """
        self.logger.info(f"Starting LSTM model run in {self.spatial_mode} mode")
        seed = self.config.system.random_seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.logger.info("LSTM initialization and training seed: %s", seed)

        with symfluence_error_handler(
            "LSTM model execution",
            self.logger,
            error_type=ModelExecutionError
        ):
            # 1. Load Data
            forcing_df, streamflow_df, snow_df = self.preprocessor.load_data()

            # Check if snow data should be used based on config
            use_snow = self.lstm_config.use_snow
            snow_df_input = snow_df if use_snow else pd.DataFrame() # Use empty DF if not using snow

            # 2. Preprocess Data
            # Decide if we are training (fit scalers) or just simulating (load scalers)
            load_existing_model = self.lstm_config.load
            model_save_path = self.project_dir / 'models' / 'lstm_model.pt'

            if load_existing_model:
                # Load pre-trained model state and scalers first
                self.logger.info("Loading pre-trained LSTM model")
                checkpoint = self._load_model_checkpoint(model_save_path)

                # Set scalers in preprocessor from checkpoint
                self.preprocessor.set_scalers(
                    checkpoint['feature_scaler'],
                    checkpoint['target_scaler'],
                    checkpoint['output_size'],
                    checkpoint['target_names']
                )

                # Preprocess data using loaded scalers
                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=False
                )
                self.hru_ids = hru_ids

                # Create model structure
                input_size = X_tensor.shape[-1]
                self._create_model_instance(
                    input_size,
                    checkpoint['output_size'],
                    hidden_size=self.lstm_config.hidden_size,
                    num_layers=self.lstm_config.num_layers
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])

            else:
                # Training mode: Fit scalers
                # Compute train/val split BEFORE preprocessing to prevent data leakage
                # The scaler should only be fit on training data
                common_times = forcing_df.index.get_level_values('time').unique().intersection(
                    streamflow_df.index
                )
                if snow_df_input is not None and not snow_df_input.empty:
                    common_times = common_times.intersection(snow_df_input.index)
                period = self.config.domain.calibration_period
                training_dates = None
                scaler_dates = None
                if period and self.spatial_mode == SpatialMode.LUMPED:
                    start_date, end_date = [pd.Timestamp(v.strip()) for v in period.split(',')]
                    target_dates = common_times[self.preprocessor.lookback:]
                    training_dates = target_dates[(target_dates >= start_date) & (target_dates < end_date + pd.Timedelta(days=1))]
                    if len(training_dates) < 10:
                        raise ValidationError("Insufficient LSTM target samples in calibration period")
                    scaler_dates = training_dates[:int(0.8 * len(training_dates))]
                    self.logger.info(
                        "LSTM calibration targets %s to %s (%d); scaler/gradient training ends %s; validation remains inside calibration",
                        training_dates.min(), training_dates.max(), len(training_dates), scaler_dates.max())
                train_end_idx = int(0.8 * len(common_times))
                X_tensor, y_tensor, common_dates, features_avg, hru_ids = self.preprocessor.process_data(
                    forcing_df, streamflow_df, snow_df_input, fit_scalers=True,
                    train_end_idx=train_end_idx, scaler_fit_dates=scaler_dates
                )
                fit_X, fit_y = X_tensor, y_tensor
                if training_dates is not None:
                    mask = torch.as_tensor(common_dates[self.preprocessor.lookback:].isin(training_dates), device=X_tensor.device)
                    fit_X, fit_y = X_tensor[mask], y_tensor[mask]
                self.hru_ids = hru_ids

                input_size = X_tensor.shape[-1]
                hidden_size = self.lstm_config.hidden_size
                num_layers = self.lstm_config.num_layers
                output_size = self.preprocessor.output_size

                # Create and Train
                self._create_model_instance(input_size, output_size, hidden_size, num_layers)

                # Check if we should train through routing
                train_through_routing = (
                    self.lstm_config.train_through_routing and
                    self.requires_routing() and
                    self.get_spatial_config('LSTM').routing.model == RoutingModel.DROUTE
                )

                if train_through_routing:
                    self.logger.info("Training LSTM through dRoute routing...")
                    self._train_model_with_routing(
                        X_tensor,
                        streamflow_df,  # Need full streamflow for outlet comparison
                        common_dates,
                        epochs=self.lstm_config.epochs,
                        learning_rate=self.lstm_config.learning_rate
                    )
                elif self.spatial_mode == SpatialMode.LUMPED or X_tensor.ndim == 3:
                    self._train_model(
                        fit_X,
                        fit_y,
                        epochs=self.lstm_config.epochs,
                        batch_size=self.lstm_config.batch_size,
                        learning_rate=self.lstm_config.learning_rate
                    )
                else:
                    self._train_model_distributed(
                        X_tensor,
                        y_tensor,
                        epochs=self.lstm_config.epochs,
                        batch_size=self.lstm_config.batch_size,
                        learning_rate=self.lstm_config.learning_rate
                    )

                # Save model
                self.project_dir.joinpath('models').mkdir(exist_ok=True)
                self._save_model_checkpoint(model_save_path)

            # 3. Simulate (Run Inference on all data)
            results = self._simulate(X_tensor, common_dates, features_avg)

            # 4. Postprocess & Routing
            # Save results (this produces the netCDF needed for routing)
            output_file = self.postprocessor.save_results(results, use_snow, self.hru_ids)

            # Handle routing if needed
            if self.requires_routing():
                self.logger.info(f"Routing LSTM output using {self.routing_model}")
                # Routing will infer source model from HYDROLOGICAL_MODEL config
                routed_output = self.route_model_output(output_file)
                if routed_output:
                    self.logger.info(f"Routing completed. Routed results: {routed_output}")

            self.logger.info("LSTM model run completed successfully")

    def _create_model_instance(self, input_size: int, output_size: int, hidden_size: int, num_layers: int):
        """
        Create the LSTM model instance with specified architecture.

        Args:
            input_size: Number of input features per timestep.
            output_size: Number of output variables (1 for streamflow, 2 with SWE).
            hidden_size: Dimensionality of LSTM hidden state.
            num_layers: Number of stacked LSTM layers.
        """
        dropout_rate = self.lstm_config.dropout
        self.logger.info(
            f"Creating LSTM model with input_size: {input_size}, hidden_size: {hidden_size}, "
            f"num_layers: {num_layers}, output_size: {output_size}"
        )
        self.model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate).to(self.device)

    def _train_model(self, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the LSTM model in lumped mode.

        Uses an 80/20 train/validation split with early stopping based on
        validation loss. Training uses SmoothL1Loss (Huber loss) for robustness
        to outliers and AdamW optimizer with learning rate scheduling.

        Args:
            X: Input tensor of shape (samples, lookback, features).
            y: Target tensor of shape (samples, outputs).
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per gradient update.
            learning_rate: Initial learning rate for AdamW optimizer.

        Note:
            Implements gradient clipping (max_norm=1.0) to prevent exploding
            gradients and uses ReduceLROnPlateau scheduler for adaptive learning.
        """
        assert self.model is not None
        self.logger.info(
            f"Training LSTM model with {epochs} epochs, batch_size: {batch_size}, learning_rate: {learning_rate}"
        )

        train_size = int(0.8 * len(X))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        criterion = nn.SmoothL1Loss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.lstm_config.l2_regularization
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float('inf')
        best_state = None
        patience = self.lstm_config.learning_patience
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for i in range(0, X_train.size(0), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                if torch.isnan(loss):
                    self.logger.warning(f"NaN loss encountered in epoch {epoch}, batch {i // batch_size}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            # Validation (batched to avoid OOM)
            self.model.eval()
            with torch.no_grad():
                val_loss_sum = 0.0
                val_samples = 0
                val_batch_size = min(batch_size * 4, 1000)  # Larger batches OK for inference
                for j in range(0, X_val.size(0), val_batch_size):
                    val_batch_X = X_val[j:j + val_batch_size]
                    val_batch_y = y_val[j:j + val_batch_size]
                    val_outputs = self.model(val_batch_X)
                    val_loss_sum += criterion(val_outputs, val_batch_y).item() * val_batch_X.size(0)
                    val_samples += val_batch_X.size(0)
                val_loss = val_loss_sum / max(val_samples, 1)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            n_batches = max(X_train.size(0) // batch_size, 1)
            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {total_loss / n_batches:.4f}, Val Loss: {val_loss:.4f}')

        if best_state is None:
            raise ValidationError("LSTM training produced no finite validation loss")
        self.model.load_state_dict(best_state)
        self.logger.info("LSTM model training completed; restored best validation weights")

    def _train_model_distributed(self, X: torch.Tensor, y: torch.Tensor, epochs: int, batch_size: int, learning_rate: float):
        """
        Train the LSTM model in distributed mode.

        Flattens the spatial (HRU) dimension into the batch dimension to allow
        standard LSTM training across all HRUs simultaneously. Each HRU is
        treated as an independent sequence sample.

        Args:
            X: Input tensor of shape (batch, lookback, n_hrus, features).
            y: Target tensor of shape (batch, n_hrus, outputs).
            epochs: Maximum number of training epochs.
            batch_size: Number of samples per gradient update.
            learning_rate: Initial learning rate for AdamW optimizer.

        Note:
            The flattening transforms X from (B, T, N, F) to (B*N, T, F),
            allowing each HRU to be processed as a separate batch sample.
        """
        # X: (B, T, N, F) -> (B*N, T, F)
        # y: (B, N, n_out) -> (B*N, n_out)
        B, T, N, F = X.shape
        n_out = y.shape[-1]

        X_flattened = X.transpose(1, 2).reshape(B * N, T, F)
        y_flattened = y.reshape(B * N, n_out)

        self.logger.debug(f"Training distributed LSTM: Flattened shape {X_flattened.shape}")
        self._train_model(X_flattened, y_flattened, epochs, batch_size, learning_rate)

    def _train_model_with_routing(self, X: torch.Tensor, obs_df: pd.DataFrame, common_dates: pd.DatetimeIndex, epochs: int, learning_rate: float):
        """
        Train LSTM through dRoute routing.
        Uses dRoute's internal AD or numerical gradients to backpropagate
        outlet streamflow error to individual HRU runoff predictions.
        """
        assert self.model is not None

        if not HAS_DROUTE:
            raise ImportError("droute required for training through routing")

        # 1. Load network
        network_data_path = self.project_dir / "settings" / "dRoute" / 'dRoute_network.pkl'
        with open(network_data_path, 'rb') as f:
            # Security rationale: Loading trusted internal model data
            network_data = pickle.load(f)  # nosec B301

        network = network_data['network']
        network_data['seg_areas']
        outlet_idx = network_data['outlet_idx']
        hru_to_seg_idx = network_data['hru_to_seg_idx']

        # 2. Setup targets
        lookback = self.preprocessor.lookback
        target_dates = common_dates[lookback:]
        observed = obs_df.loc[target_dates, 'streamflow'].values
        torch.FloatTensor(observed).to(self.device)

        # 3. Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # 4. Training loop
        B, T, N, F = X.shape
        self.logger.info(f"Starting training through routing: {B} batches, {N} HRUs")

        # Map HRU index in tensor to reach index in network
        hru_idx_to_reach = [hru_to_seg_idx[hru_id] for hru_id in self.hru_ids]

        # Router config
        routing_method = self._get_config_value(lambda: None, default='mc', dict_key='DROUTE_METHOD').lower()
        router_classes = {
            'mc': droute.MuskingumCungeRouter,
            'lag': droute.LagRouter,
            'irf': droute.IRFRouter,
            'kwt': droute.SoftGatedKWT,
        }
        RouterClass = router_classes.get(routing_method, droute.MuskingumCungeRouter)

        config = droute.RouterConfig()
        config.dt = float(self.mizu_routing_dt)
        config.enable_gradients = True # Enable AD

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            # Predict runoff for all HRUs
            # Flatten HRUs into batch for efficient LSTM processing
            X_flattened = X.transpose(1, 2).reshape(B * N, T, F)
            runoff_scaled = self.model(X_flattened) # (B*N, 1)

            # Unscale runoff
            # We need to do this in a differentiable way if possible,
            # but for now we'll do it manually
            mean = torch.tensor(self.preprocessor.target_scaler.mean_[0]).to(self.device)
            scale = torch.tensor(self.preprocessor.target_scaler.scale_[0]).to(self.device)
            runoff = runoff_scaled[:, 0] * scale + mean # (B*N,)

            # Reshape to (B, N)
            runoff_reshaped = runoff.reshape(B, N)

            # Convert to CMS (assuming LSTM predicts runoff in same units as routing expects)
            # Actually runoff_reshaped is runoff depth or rate.
            # If LSTM was trained on streamflow, it predicts streamflow.
            # Usually in distributed mode, LSTM predicts runoff per unit area.

            # Run Routing (Forward)
            # Since droute is not a torch module, we'll run it and then manually compute gradients
            runoff_np = runoff_reshaped.detach().cpu().numpy()

            # Create router and record
            router = RouterClass(network, config)
            router.start_recording()

            sim_outlet = np.zeros(B)
            for t in range(B):
                # Set lateral inflows
                for i in range(N):
                    reach_idx = hru_idx_to_reach[i]
                    router.set_lateral_inflow(reach_idx, float(runoff_np[t, i]))

                router.route_timestep()
                router.record_output(outlet_idx)
                sim_outlet[t] = router.get_discharge(outlet_idx)

            router.stop_recording()

            # Compute Loss (MSE at outlet)
            loss_val = np.mean((sim_outlet - observed) ** 2)

            # Compute gradients dL/dQ_outlet
            dL_dQ = (2.0 / B) * (sim_outlet - observed)

            # Backprop through routing using dRoute (Reverse AD)
            router.compute_gradients_timeseries(outlet_idx, dL_dQ.tolist())
            droute_grads = router.get_gradients()

            # Extract gradients dL/dq_hru where q_hru is lateral inflow
            # We need dL/dq_hru(t, i)
            # KNOWN LIMITATION: dRoute timestep-specific gradient support is partial.
            # Current workaround extracts gradients per reach/timestep key if available,
            # falling back to zero when keys are missing. Full gradient support requires
            # dRoute API extension for complete reverse-mode AD through routing.

            # Assuming droute_grads contains 'lateral_inflow_reach_R_step_T'
            grad_runoff = torch.zeros((B, N)).to(self.device)
            for t in range(B):
                for i in range(N):
                    reach_idx = hru_idx_to_reach[i]
                    grad_key = f"lateral_inflow_reach_{reach_idx}_step_{t}"
                    grad_runoff[t, i] = droute_grads.get(grad_key, 0.0)

            # Manual backward pass through LSTM
            # dL/drunoff_scaled = dL/drunoff * drunoff/drunoff_scaled = grad_runoff * scale
            runoff_scaled.backward(grad_runoff.flatten().unsqueeze(1) * scale)

            optimizer.step()

            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss (Outlet MSE): {loss_val:.4f}")

    def _simulate(self, X_tensor: torch.Tensor, common_dates: pd.DatetimeIndex, features_avg: pd.DataFrame) -> pd.DataFrame:
        """
        Run full simulation with trained LSTM model.

        Performs forward pass through the model on all input sequences,
        inverse-transforms predictions to physical units, and formats
        results as a DataFrame.

        Args:
            X_tensor: Input sequences tensor (lumped: (B, T, F) or
                distributed: (B, T, N, F)).
            common_dates: DatetimeIndex of all input timesteps.
            features_avg: DataFrame of forcing features for result joining.

        Returns:
            pd.DataFrame: Results with predicted_streamflow (and predicted_SWE
                if configured) columns joined to forcing features. Indexed by
                time for lumped mode, or MultiIndex (time, hruId) for distributed.

        Note:
            Uses batched DataLoader for memory-efficient inference.
            Predictions are clipped to handle numerical instabilities.
        """
        assert self.model is not None
        self.logger.info(f"Running full simulation with LSTM model (mode={self.spatial_mode})")

        self._log_memory_usage()

        if self.spatial_mode == SpatialMode.LUMPED or X_tensor.ndim == 3:
            X_input = X_tensor
        else:
            # Flatten HRUs for inference: (B, T, N, F) -> (B*N, T, F)
            B, T, N, F = X_tensor.shape
            X_input = X_tensor.transpose(1, 2).reshape(B * N, T, F)

        dataset = TensorDataset(X_input)
        dataloader = DataLoader(dataset, batch_size=1000, shuffle=False)

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                batch_predictions = self.model(batch[0])
                predictions.append(batch_predictions.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)

        # Inverse transform the predictions
        predictions = self.preprocessor.target_scaler.inverse_transform(predictions)

        # Handle NaN values
        predictions = np.nan_to_num(predictions, nan=0.0, posinf=1e15, neginf=-1e15)

        # Create column names based on number of targets
        if self.preprocessor.output_size == 2:
            columns = ['predicted_streamflow', 'predicted_SWE']
        else:
            columns = ['predicted_streamflow']

        lookback = self.preprocessor.lookback

        if self.spatial_mode == SpatialMode.LUMPED or X_tensor.ndim == 3:
            # Create a DataFrame for predictions
            pred_df = pd.DataFrame(predictions, columns=columns, index=common_dates[lookback:])
            # Join predictions with the original averaged features
            result = features_avg.join(pred_df, how='outer')
        else:
            # Distributed mode: Unflatten results
            # predictions is (B*N, n_out)
            B = X_tensor.shape[0]
            N = len(self.hru_ids)
            n_out = self.preprocessor.output_size

            # Reshape to (B, N, n_out)
            preds_reshaped = predictions.reshape(B, N, n_out)

            # Convert to a DataFrame with MultiIndex [time, hruId]
            time_idx = common_dates[lookback:]

            all_preds = []
            for i, hru_id in enumerate(self.hru_ids):
                hru_preds = pd.DataFrame(preds_reshaped[:, i, :], columns=columns, index=time_idx)
                hru_preds['hruId'] = hru_id
                all_preds.append(hru_preds)

            pred_df = pd.concat(all_preds).reset_index().rename(columns={'index': 'time'})
            pred_df = pred_df.set_index(['time', 'hruId']).sort_index()

            # Join with features_avg (which is the original forcing_df subset)
            result = features_avg.join(pred_df, how='outer')

        self.logger.debug(f"Shape of final result: {result.shape}")
        self._log_memory_usage()
        return result

    def _save_model_checkpoint(self, path: Path):
        """
        Save the LSTM model and scalers to disk.

        Saves a checkpoint containing model weights, fitted scalers, and
        metadata needed to reload and resume predictions without retraining.

        Args:
            path: File path for the checkpoint (.pt file).

        Saved contents:
            - model_state_dict: PyTorch model weights
            - feature_scaler: Fitted StandardScaler for inputs
            - target_scaler: Fitted StandardScaler for outputs
            - lookback: Sequence length used during training
            - output_size: Number of output variables
            - target_names: List of output variable names
        """
        self.logger.info(f"Saving LSTM model to {path}")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_scaler': self.preprocessor.feature_scaler,
            'target_scaler': self.preprocessor.target_scaler,
            'lookback': self.preprocessor.lookback,
            'output_size': self.preprocessor.output_size,
            'target_names': self.preprocessor.target_names
        }, path)
        self.logger.info("Model saved successfully")

    def _load_model_checkpoint(self, path: Path) -> Dict[str, Any]:
        """
        Load a LSTM model checkpoint from disk.

        Args:
            path: File path to the checkpoint (.pt file).

        Returns:
            dict: Checkpoint dictionary containing model_state_dict, scalers,
                and metadata (lookback, output_size, target_names).

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
        """
        self.logger.info(f"Loading LSTM model from {path}")
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {path}")
        # weights_only=True prevents arbitrary code execution when deserializing
        # an untrusted checkpoint (CVE-2025-32434 lineage).
        return torch.load(path, map_location=self.device, weights_only=True)

    def _log_memory_usage(self):
        """Log current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
