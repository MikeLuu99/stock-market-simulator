import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import stock_sim_core
    from python.simulator import StockMarketSimulator
    COMPILED = True
except ImportError:
    COMPILED = False


@pytest.mark.skipif(not COMPILED, reason="C++ module not compiled")
class TestMCMCEngine:
    def setup_method(self):
        self.simulator = StockMarketSimulator(seed=42)

    def test_simulation_params_creation(self):
        params = self.simulator.create_simulation_params(
            initial_price=100.0,
            drift=0.05,
            volatility=0.2,
            num_steps=10,
            num_simulations=5
        )

        assert params.initial_price == 100.0
        assert params.drift == 0.05
        assert params.volatility == 0.2
        assert params.num_steps == 10
        assert params.num_simulations == 5

    def test_single_asset_simulation(self):
        params = self.simulator.create_simulation_params(
            initial_price=100.0,
            num_steps=10,
            num_simulations=5
        )

        paths = self.simulator.simulate_single_asset(params)

        assert paths.shape == (5, 11)  # num_simulations x (num_steps + 1)
        assert np.all(paths[:, 0] == 100.0)  # Initial prices
        assert np.all(paths > 0)  # All prices positive

    def test_portfolio_simulation(self):
        assets = [
            {
                'symbol': 'ASSET1',
                'weight': 0.6,
                'params': {
                    'initial_price': 100.0,
                    'drift': 0.05,
                    'volatility': 0.2,
                    'num_steps': 10,
                    'num_simulations': 5
                }
            },
            {
                'symbol': 'ASSET2',
                'weight': 0.4,
                'params': {
                    'initial_price': 50.0,
                    'drift': 0.03,
                    'volatility': 0.25,
                    'num_steps': 10,
                    'num_simulations': 5
                }
            }
        ]

        correlation_matrix = np.array([[1.0, 0.3], [0.3, 1.0]])

        paths, portfolio = self.simulator.simulate_portfolio(assets, correlation_matrix)

        # Check portfolio weights sum to 1
        assert abs(portfolio.get_total_weight() - 1.0) < 1e-10

        # Check portfolio values are calculated
        portfolio_values = portfolio.calculate_portfolio_values()
        assert len(portfolio_values) == 11  # num_steps + 1

    def test_risk_metrics(self):
        # Generate some sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        metrics = self.simulator.calculate_risk_metrics(returns)

        assert 'var_95' in metrics
        assert 'var_99' in metrics
        assert 'cvar_95' in metrics
        assert 'cvar_99' in metrics
        assert 'sortino_ratio' in metrics

        # VaR 99% should be more extreme than VaR 95%
        assert metrics['var_99'] <= metrics['var_95']
        assert metrics['cvar_99'] <= metrics['cvar_95']

    def test_market_parameter_estimation(self):
        # Create synthetic price data
        np.random.seed(42)
        initial_price = 100.0
        num_days = 252
        true_drift = 0.05
        true_vol = 0.2

        prices = [initial_price]
        for _ in range(num_days):
            ret = np.random.normal(true_drift/252, true_vol/np.sqrt(252))
            prices.append(prices[-1] * np.exp(ret))

        price_array = np.array(prices)
        estimated_params = self.simulator.estimate_market_parameters(price_array)

        # Check that estimated parameters are reasonable
        assert 0.0 <= estimated_params.volatility <= 1.0
        assert -1.0 <= estimated_params.drift <= 1.0
        assert estimated_params.current_price == prices[-1]

    def test_correlation_matrix_generation(self):
        # Generate correlated returns
        np.random.seed(42)
        returns1 = np.random.normal(0, 0.02, 100)
        returns2 = 0.5 * returns1 + 0.5 * np.random.normal(0, 0.02, 100)
        returns3 = np.random.normal(0, 0.02, 100)

        multi_asset_returns = [returns1, returns2, returns3]
        corr_matrix = self.simulator.get_correlation_matrix(multi_asset_returns)

        assert corr_matrix.shape == (3, 3)

        # Diagonal should be 1.0
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0], decimal=10)

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T, decimal=10)

        # Correlation values should be between -1 and 1
        assert np.all(corr_matrix >= -1.0)
        assert np.all(corr_matrix <= 1.0)


@pytest.mark.skipif(not COMPILED, reason="C++ module not compiled")
class TestPortfolio:
    def test_portfolio_creation(self):
        simulator = StockMarketSimulator()

        assets = [
            {'symbol': 'AAPL', 'weight': 0.5, 'initial_price': 150.0},
            {'symbol': 'GOOGL', 'weight': 0.5, 'initial_price': 2500.0}
        ]

        portfolio = simulator.create_portfolio("test_portfolio", assets)

        assert portfolio.get_total_weight() == 1.0
        assert len(portfolio.get_assets()) == 2

    def test_portfolio_metrics(self):
        simulator = StockMarketSimulator(seed=42)

        assets = [
            {
                'symbol': 'ASSET1',
                'weight': 1.0,
                'params': {
                    'initial_price': 100.0,
                    'drift': 0.05,
                    'volatility': 0.2,
                    'num_steps': 252,
                    'num_simulations': 1
                }
            }
        ]

        paths, portfolio = simulator.simulate_portfolio(assets)

        # Calculate metrics
        sharpe_ratio = portfolio.calculate_sharpe_ratio()
        max_drawdown = portfolio.calculate_max_drawdown()
        volatility = portfolio.calculate_volatility()

        assert isinstance(sharpe_ratio, float)
        assert isinstance(max_drawdown, float)
        assert isinstance(volatility, float)

        assert 0.0 <= max_drawdown <= 1.0  # Drawdown should be between 0 and 1