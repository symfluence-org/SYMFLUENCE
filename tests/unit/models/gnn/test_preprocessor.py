"""
Tests for GNN preprocessor.
"""
from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

pytest.importorskip("torch", reason="requires the ml extra (pip install 'symfluence[ml]')")


class TestGNNPreProcessorInitialization:
    """Tests for GNN preprocessor initialization."""

    def test_preprocessor_can_be_imported(self):
        """Test that GNNPreProcessor can be imported."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor
        assert GNNPreProcessor is not None

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_preprocessor_initialization(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test preprocessor initializes with config."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        assert preprocessor is not None


class TestGNNGraphLoading:
    """Tests for GNN graph structure loading."""

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_graph_structure_initialization(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test graph structure is initialized from river network."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # GNN should have adjacency matrix attribute
        assert hasattr(preprocessor, 'adj_matrix')

    @patch('symfluence.models.gnn.preprocessor.torch')
    @patch('symfluence.models.gnn.preprocessor.gpd')
    def test_load_river_network_for_adjacency(self, mock_gpd, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test loading river network to build adjacency matrix."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        mock_gdf = MagicMock()
        mock_gpd.read_file.return_value = mock_gdf

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # River network defines graph edges
        assert hasattr(preprocessor, 'node_mapping')


class TestGNNFeaturePreparation:
    """Tests for GNN feature preparation."""

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_node_features_preparation(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test node features are prepared correctly."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Each node (catchment) should have HRU mapping
        assert hasattr(preprocessor, 'hru_to_node')

    @patch('symfluence.models.gnn.preprocessor.torch')
    def test_edge_features_preparation(self, mock_torch, gnn_config, mock_logger, setup_gnn_directories):
        """Test edge features are prepared if needed."""
        from symfluence.models.gnn.preprocessor import GNNPreProcessor

        mock_device = MagicMock()
        mock_torch.device.return_value = mock_device

        config_dict = gnn_config.model_dump()
        project_dir = setup_gnn_directories['domain_dir']

        preprocessor = GNNPreProcessor(config_dict, mock_logger, project_dir, mock_device)
        # Should have outlet indices for target assignment
        assert hasattr(preprocessor, 'outlet_indices')


def test_explicit_scaler_dates_exclude_holdout():
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.preprocessing import StandardScaler

    from symfluence.models.gnn.preprocessor import GNNPreProcessor

    preprocessor = object.__new__(GNNPreProcessor)
    preprocessor.logger = Mock()
    preprocessor.node_mapping = {10: 0}
    preprocessor.ordered_hru_ids = [10]
    preprocessor.adj_matrix = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.long), torch.empty(0), (1, 1)
    ).coalesce()
    preprocessor.device = torch.device('cpu')
    preprocessor.lookback = 1
    preprocessor.feature_scaler = StandardScaler()
    preprocessor.target_scaler = StandardScaler()
    dates = pd.date_range('2000-01-01', periods=4)
    forcing = pd.DataFrame(
        {'temperature': [1., 3., 100., 200.], 'hru': 0, 'latitude': 0., 'longitude': 0.},
        index=pd.MultiIndex.from_product([dates, [10]], names=['time', 'hruId']),
    )
    flow = pd.DataFrame({'streamflow': [2., 4., 100., 200.]}, index=dates)
    X, y, _, _, _ = preprocessor.process_data(
        forcing, flow, train_end_idx=4, scaler_fit_dates=dates[:2]
    )
    np.testing.assert_allclose(preprocessor.feature_scaler.mean_, [2.])
    np.testing.assert_allclose(preprocessor.target_scaler.mean_, [3.])
    assert X.shape == (3, 1, 1, 1)
    assert y.shape == (3, 1, 1)
