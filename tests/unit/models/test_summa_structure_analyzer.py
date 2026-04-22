"""
Tests for SUMMA Structure Analyzer lazy loading and routing logic.

Tests the refactored lazy loading pattern that prevents circular dependencies.
"""

from unittest.mock import Mock, patch

import pytest


class TestSummaStructureAnalyzerLazyLoading:
    """Test lazy loading of MizuRoute runner."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config_dict = {
            'SYMFLUENCE_DATA_DIR': '/tmp/test_data',
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'SUMMA_DECISION_OPTIONS': {
                'snowIncept': ['stickySnow', 'lightSnow'],
                'windPrfile': ['exponential', 'logarithmic']
            }
        }
        self.logger = Mock()

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_init_without_routing(self, mock_summa_runner):
        """Test that __init__ does NOT instantiate MizuRoute runner."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        analyzer = SummaStructureAnalyzer(self.config_dict, self.logger)

        # SUMMA runner should be instantiated
        assert mock_summa_runner.called

        # MizuRoute runner should NOT be instantiated yet
        assert analyzer._mizuroute_runner is None

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_routing_needed_with_mizuroute_config(self, mock_summa_runner):
        """Test _needs_routing() returns True when ROUTING_MODEL=mizuroute."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_with_routing = {
            **self.config_dict,
            'ROUTING_MODEL': 'mizuroute'
        }

        analyzer = SummaStructureAnalyzer(config_with_routing, self.logger)

        assert analyzer._needs_routing() is True

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_routing_needed_with_distributed_domain(self, mock_summa_runner):
        """Test _needs_routing() returns True for distributed domains."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_distributed = {
            **self.config_dict,
            'DOMAIN_DEFINITION_METHOD': 'distributed',
            'ROUTING_DELINEATION': 'lumped'
        }

        analyzer = SummaStructureAnalyzer(config_distributed, self.logger)

        assert analyzer._needs_routing() is True

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_routing_not_needed_lumped(self, mock_summa_runner):
        """Test _needs_routing() returns False for lumped domain."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_lumped = {
            **self.config_dict,
            'DOMAIN_DEFINITION_METHOD': 'lumped',
            'ROUTING_DELINEATION': 'lumped'
        }

        analyzer = SummaStructureAnalyzer(config_lumped, self.logger)

        assert analyzer._needs_routing() is False

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    @patch('symfluence.models.summa.structure_analyzer.MizuRouteRunner')
    def test_mizuroute_runner_property_lazy_loads(self, mock_mizu_runner, mock_summa_runner):
        """Test mizuroute_runner property lazy-loads when routing is needed."""
        # Need to patch at the import location inside the property
        with patch('symfluence.models.summa.structure_analyzer.MizuRouteRunner') as mock_mizu:
            from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

            config_with_routing = {
                **self.config_dict,
                'ROUTING_MODEL': 'mizuroute'
            }

            analyzer = SummaStructureAnalyzer(config_with_routing, self.logger)

            # MizuRoute should not be imported yet
            assert analyzer._mizuroute_runner is None

            # Access the property
            runner = analyzer.mizuroute_runner

            # Now it should be loaded
            assert mock_mizu.called
            assert analyzer._mizuroute_runner is not None

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_mizuroute_runner_raises_when_not_configured(self, mock_summa_runner):
        """Test accessing mizuroute_runner raises error when routing not configured."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_no_routing = {
            **self.config_dict,
            'ROUTING_MODEL': 'none'
        }

        analyzer = SummaStructureAnalyzer(config_no_routing, self.logger)

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="MizuRoute runner requested but routing is not configured"):
            _ = analyzer.mizuroute_runner

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_run_model_skips_routing_when_disabled(self, mock_summa_runner):
        """Test run_model() skips mizuRoute when routing is disabled."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_no_routing = {
            **self.config_dict,
            'ROUTING_MODEL': 'none'
        }

        analyzer = SummaStructureAnalyzer(config_no_routing, self.logger)

        # Mock summa runner
        analyzer.summa_runner.run_summa = Mock()

        # Run model
        analyzer.run_model()

        # SUMMA should run
        analyzer.summa_runner.run_summa.assert_called_once()

        # MizuRoute should NOT be instantiated
        assert analyzer._mizuroute_runner is None

        # Logger should indicate routing was skipped
        log_messages = [call[0][0] for call in self.logger.info.call_args_list]
        assert any('Skipping mizuRoute routing' in msg for msg in log_messages)

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_run_model_executes_routing_when_enabled(self, mock_summa_runner):
        """Test run_model() executes mizuRoute when routing is enabled."""
        with patch('symfluence.models.summa.structure_analyzer.MizuRouteRunner') as mock_mizu:
            from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

            config_with_routing = {
                **self.config_dict,
                'ROUTING_MODEL': 'mizuroute'
            }

            analyzer = SummaStructureAnalyzer(config_with_routing, self.logger)

            # Mock runners
            analyzer.summa_runner.run_summa = Mock()
            mock_mizu_instance = Mock()
            mock_mizu.return_value = mock_mizu_instance

            # Run model
            analyzer.run_model()

            # SUMMA should run
            analyzer.summa_runner.run_summa.assert_called_once()

            # MizuRoute should be instantiated and run
            assert mock_mizu.called
            mock_mizu_instance.run_mizuroute.assert_called_once()

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_needs_routing_cached(self, mock_summa_runner):
        """Test _needs_routing() caches result and only checks once."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config_with_routing = {
            **self.config_dict,
            'ROUTING_MODEL': 'mizuroute'
        }

        analyzer = SummaStructureAnalyzer(config_with_routing, self.logger)

        # First call
        result1 = analyzer._needs_routing()

        # Clear logger calls
        self.logger.reset_mock()

        # Second call
        result2 = analyzer._needs_routing()

        # Results should be the same
        assert result1 == result2 == True

        # Logger should NOT have been called again (cached)
        assert not any('Routing (mizuRoute)' in str(call) for call in self.logger.info.call_args_list)


class TestCalculatePerformanceMetricsDispatch:
    """Test that calculate_performance_metrics dispatches by OPTIMIZATION_TARGET."""

    def setup_method(self):
        self.base_config = {
            'SYMFLUENCE_DATA_DIR': '/tmp/test_data',
            'DOMAIN_NAME': 'test_domain',
            'EXPERIMENT_ID': 'test_exp',
            'SUMMA_DECISION_OPTIONS': {},
        }
        self.logger = Mock()

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_default_target_is_streamflow(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        analyzer = SummaStructureAnalyzer(self.base_config, self.logger)
        assert analyzer._get_optimization_target() == 'streamflow'

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_swe_target_detected(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'SWE'}
        analyzer = SummaStructureAnalyzer(config, self.logger)
        assert analyzer._get_optimization_target() == 'swe'

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_streamflow_target_calls_streamflow_metrics(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'streamflow'}
        analyzer = SummaStructureAnalyzer(config, self.logger)
        analyzer._calculate_streamflow_metrics = Mock(return_value={'kge': 0.8})
        analyzer._calculate_evaluator_metrics = Mock()

        result = analyzer.calculate_performance_metrics()

        analyzer._calculate_streamflow_metrics.assert_called_once()
        analyzer._calculate_evaluator_metrics.assert_not_called()
        assert result == {'kge': 0.8}

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_swe_target_calls_evaluator_metrics(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'SWE'}
        analyzer = SummaStructureAnalyzer(config, self.logger)
        analyzer._calculate_streamflow_metrics = Mock()
        analyzer._calculate_evaluator_metrics = Mock(return_value={'kge': 0.7})

        result = analyzer.calculate_performance_metrics()

        analyzer._calculate_evaluator_metrics.assert_called_once_with('swe')
        analyzer._calculate_streamflow_metrics.assert_not_called()
        assert result == {'kge': 0.7}

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_flow_target_calls_streamflow_metrics(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'flow'}
        analyzer = SummaStructureAnalyzer(config, self.logger)
        analyzer._calculate_streamflow_metrics = Mock(return_value={'kge': 0.9})

        result = analyzer.calculate_performance_metrics()
        analyzer._calculate_streamflow_metrics.assert_called_once()

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_et_target_calls_evaluator_metrics(self, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'ET'}
        analyzer = SummaStructureAnalyzer(config, self.logger)
        analyzer._calculate_evaluator_metrics = Mock(return_value={'kge': 0.6})

        result = analyzer.calculate_performance_metrics()
        analyzer._calculate_evaluator_metrics.assert_called_once_with('et')

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    @patch('symfluence.evaluation.registry.EvaluationRegistry.get_evaluator')
    def test_evaluator_metrics_delegates_to_registry(self, mock_get_evaluator, mock_summa_runner):
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        mock_evaluator = Mock()
        mock_evaluator.calculate_metrics.return_value = {
            'KGE': 0.75, 'KGEp': 0.78, 'NSE': 0.70, 'MAE': 1.2, 'RMSE': 2.1,
        }
        mock_get_evaluator.return_value = mock_evaluator

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'SWE'}
        analyzer = SummaStructureAnalyzer(config, self.logger)

        result = analyzer._calculate_evaluator_metrics('swe')

        mock_get_evaluator.assert_called_once_with(
            'SNOW', analyzer.config, analyzer.logger, analyzer.project_dir,
            target='swe',
        )
        mock_evaluator.calculate_metrics.assert_called_once()
        assert result['kge'] == 0.75
        assert result['nse'] == 0.70

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    @patch('symfluence.evaluation.registry.EvaluationRegistry.get_evaluator')
    def test_evaluator_metrics_returns_nan_when_no_evaluator(self, mock_get_evaluator, mock_summa_runner):
        import numpy as np

        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        mock_get_evaluator.return_value = None

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'unknown_var'}
        analyzer = SummaStructureAnalyzer(config, self.logger)

        result = analyzer._calculate_evaluator_metrics('unknown_var')

        assert np.isnan(result['kge'])
        assert np.isnan(result['nse'])

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    @patch('symfluence.evaluation.registry.EvaluationRegistry.get_evaluator')
    def test_evaluator_metrics_returns_nan_when_metrics_fail(self, mock_get_evaluator, mock_summa_runner):
        import numpy as np

        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        mock_evaluator = Mock()
        mock_evaluator.calculate_metrics.return_value = None
        mock_get_evaluator.return_value = mock_evaluator

        config = {**self.base_config, 'OPTIMIZATION_TARGET': 'SWE'}
        analyzer = SummaStructureAnalyzer(config, self.logger)

        result = analyzer._calculate_evaluator_metrics('swe')

        assert np.isnan(result['kge'])


class TestBackwardCompatibility:
    """Test that existing code patterns still work."""

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_can_still_import_structure_analyzer(self, mock_summa_runner):
        """Test structure analyzer can still be imported."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        assert SummaStructureAnalyzer is not None

    @patch('symfluence.models.summa.structure_analyzer.SummaRunner')
    def test_existing_initialization_pattern_works(self, mock_summa_runner):
        """Test that existing initialization patterns still work."""
        from symfluence.models.summa.structure_analyzer import SummaStructureAnalyzer

        config = {'SYMFLUENCE_DATA_DIR': '/tmp', 'DOMAIN_NAME': 'test', 'EXPERIMENT_ID': 'exp1'}
        logger = Mock()

        # Should not raise
        analyzer = SummaStructureAnalyzer(config, logger)
        assert analyzer is not None
