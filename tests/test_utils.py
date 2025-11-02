import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.utils import DataLoader, ConfigManager, PerformanceTracker


class TestDataLoader:
    def test_calculate_returns(self):
        prices = pd.Series([100, 105, 103, 108, 110])
        returns = DataLoader.calculate_returns(prices)

        expected = pd.Series([0.05, -0.019047619, 0.048543689, 0.018518519])
        expected.index = [1, 2, 3, 4]

        pd.testing.assert_series_equal(returns, expected, check_exact=False, rtol=1e-7)

    def test_calculate_log_returns(self):
        prices = pd.Series([100, 105, 103, 108, 110])
        log_returns = DataLoader.calculate_log_returns(prices)

        expected = pd.Series([
            np.log(105/100),
            np.log(103/105),
            np.log(108/103),
            np.log(110/108)
        ])
        expected.index = [1, 2, 3, 4]

        pd.testing.assert_series_equal(log_returns, expected, check_exact=False)

    def test_resample_data(self):
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        data = pd.DataFrame({
            'price': np.random.randn(10),
            'volume': np.random.randint(100, 1000, 10)
        }, index=dates)

        resampled = DataLoader.resample_data(data, freq='D')

        assert len(resampled) <= len(data)
        assert resampled.index.freq is None or 'D' in str(resampled.index.freq)

    @patch('yfinance.download')
    def test_fetch_yahoo_data_single_symbol(self, mock_download):
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        mock_download.return_value = mock_data

        result = DataLoader.fetch_yahoo_data("AAPL", period="1mo")

        assert len(result.columns) == 5
        assert all(col.startswith("AAPL_") for col in result.columns)

    @patch('yfinance.download')
    def test_prepare_simulation_data(self, mock_download):
        # Mock yfinance response
        mock_data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 105, 103, 108, 110],
            ('Volume', 'AAPL'): [1000, 1100, 900, 1200, 1300]
        })
        mock_data.index = pd.date_range('2023-01-01', periods=5)
        mock_download.return_value = mock_data

        result = DataLoader.prepare_simulation_data(['AAPL'], period="1mo")

        assert 'AAPL' in result
        assert 'prices' in result['AAPL']
        assert 'log_returns' in result['AAPL']
        assert 'current_price' in result['AAPL']
        assert 'mean_return' in result['AAPL']
        assert 'volatility' in result['AAPL']


class TestConfigManager:
    def test_default_config_creation(self):
        config_manager = ConfigManager()

        assert 'simulation' in config_manager.config
        assert 'assets' in config_manager.config
        assert 'portfolio' in config_manager.config
        assert 'display' in config_manager.config

    def test_config_get_set(self):
        config_manager = ConfigManager()

        # Test setting a value
        config_manager.set('simulation.num_simulations', 5000)
        assert config_manager.get('simulation.num_simulations') == 5000

        # Test getting with default
        assert config_manager.get('nonexistent.key', 'default') == 'default'

    def test_config_save_load(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path)
            config_manager.set('test.value', 42)
            config_manager.save_config()

            # Create new instance and load
            config_manager2 = ConfigManager(config_path)
            config_manager2.load_config()

            assert config_manager2.get('test.value') == 42

        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)

    def test_asset_config_management(self):
        config_manager = ConfigManager()

        # Add asset config
        asset_params = {'drift': 0.1, 'volatility': 0.3}
        config_manager.add_asset_config('AAPL', asset_params)

        # Get asset config
        retrieved_config = config_manager.get_asset_config('AAPL')

        assert retrieved_config['drift'] == 0.1
        assert retrieved_config['volatility'] == 0.3

    def test_get_asset_config_with_defaults(self):
        config_manager = ConfigManager()

        # Get config for non-existent asset (should return defaults)
        config = config_manager.get_asset_config('UNKNOWN')

        default_params = config_manager.get('assets.default_params')
        assert config == default_params


class TestPerformanceTracker:
    def test_track_simulation(self):
        tracker = PerformanceTracker()

        metrics = {
            'var_95': -0.05,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.15
        }

        tracker.track_simulation('sim_001', metrics, 2.5)

        assert len(tracker.metrics_history) == 1
        assert tracker.metrics_history[0]['simulation_id'] == 'sim_001'
        assert tracker.metrics_history[0]['duration'] == 2.5
        assert tracker.metrics_history[0]['metrics'] == metrics

    def test_performance_summary(self):
        tracker = PerformanceTracker()

        # Add multiple simulations
        for i in range(3):
            metrics = {'var_95': -0.05 - i*0.01, 'duration': 2.0 + i}
            tracker.track_simulation(f'sim_{i:03d}', metrics, 2.0 + i)

        summary = tracker.get_performance_summary()

        assert len(summary) == 3
        assert 'simulation_id' in summary.columns
        assert 'duration' in summary.columns
        assert 'metrics' in summary.columns

    def test_export_metrics(self):
        tracker = PerformanceTracker()

        metrics = {'var_95': -0.05, 'sharpe_ratio': 1.2}
        tracker.track_simulation('sim_001', metrics, 2.5)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            export_path = f.name

        try:
            tracker.export_metrics(export_path)

            # Verify file was created and has content
            assert os.path.exists(export_path)

            # Load and verify content
            df = pd.read_csv(export_path)
            assert len(df) == 1
            assert df.iloc[0]['simulation_id'] == 'sim_001'

        finally:
            if os.path.exists(export_path):
                os.unlink(export_path)

    def test_empty_tracker(self):
        tracker = PerformanceTracker()

        summary = tracker.get_performance_summary()
        assert len(summary) == 0
        assert isinstance(summary, pd.DataFrame)


if __name__ == '__main__':
    pytest.main([__file__])