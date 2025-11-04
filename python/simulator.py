import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import stock_sim_core
import logging
import time

logger = logging.getLogger(__name__)


class StockMarketSimulator:
    def __init__(self, seed: Optional[int] = None):
        self.engine = stock_sim_core.MCMCEngine(seed if seed is not None else np.random.randint(0, 2**32))
        self.portfolios: Dict[str, stock_sim_core.Portfolio] = {}

    def create_simulation_params(
        self,
        initial_price: float = 100.0,
        drift: float = 0.05,
        volatility: float = 0.2,
        jump_intensity: float = 0.1,
        jump_mean: float = 0.0,
        jump_std: float = 0.05,
        num_steps: int = 252,
        num_simulations: int = 1000,
        dt: float = 1.0 / 252.0
    ) -> stock_sim_core.SimulationParams:
        params = stock_sim_core.SimulationParams()
        params.initial_price = initial_price
        params.drift = drift
        params.volatility = volatility
        params.jump_intensity = jump_intensity
        params.jump_mean = jump_mean
        params.jump_std = jump_std
        params.num_steps = num_steps
        params.num_simulations = num_simulations
        params.dt = dt
        return params

    def simulate_single_asset(self, params: stock_sim_core.SimulationParams) -> np.ndarray:
        logger.info(f"[Python Wrapper] simulate_single_asset called")
        logger.info(f"[Python Wrapper] Parameters: num_simulations={params.num_simulations}, "
                   f"num_steps={params.num_steps}, initial_price={params.initial_price}")
        logger.info(f"[Python Wrapper] Calling C++ engine.simulate_gbm_with_jumps()...")

        start_time = time.time()
        paths = self.engine.simulate_gbm_with_jumps(params)
        elapsed = time.time() - start_time

        logger.info(f"[Python Wrapper] C++ simulation returned in {elapsed:.4f}s")
        logger.info(f"[Python Wrapper] Converting C++ result to numpy array...")
        result = np.array(paths)
        logger.info(f"[Python Wrapper] Result shape: {result.shape}, dtype: {result.dtype}")
        return result

    def simulate_portfolio(
        self,
        assets: List[Dict],
        correlation_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, stock_sim_core.Portfolio]:
        logger.info(f"[Python Wrapper] simulate_portfolio called with {len(assets)} assets")

        sim_params = []
        portfolio_assets = []

        logger.info(f"[Python Wrapper] Creating parameters for each asset...")
        for idx, asset in enumerate(assets):
            params = self.create_simulation_params(**asset.get('params', {}))
            sim_params.append(params)
            logger.info(f"[Python Wrapper] Asset {idx}: {asset.get('symbol', 'UNKNOWN')}, "
                       f"weight={asset.get('weight', 1.0 / len(assets)):.2f}")

            portfolio_asset = stock_sim_core.Asset()
            portfolio_asset.symbol = asset.get('symbol', 'UNKNOWN')
            portfolio_asset.weight = asset.get('weight', 1.0 / len(assets))
            portfolio_asset.initial_price = params.initial_price
            portfolio_assets.append(portfolio_asset)

        if correlation_matrix is None:
            correlation_matrix = np.eye(len(assets))

        logger.info(f"[Python Wrapper] Correlation matrix shape: {correlation_matrix.shape}")
        corr_matrix_list = [correlation_matrix[i].tolist() for i in range(len(assets))]

        logger.info(f"[Python Wrapper] Calling C++ engine.simulate_multi_asset()...")
        start_time = time.time()
        paths = self.engine.simulate_multi_asset(sim_params, corr_matrix_list)
        elapsed = time.time() - start_time
        logger.info(f"[Python Wrapper] C++ multi-asset simulation returned in {elapsed:.4f}s")

        logger.info(f"[Python Wrapper] Creating portfolio object...")
        portfolio = stock_sim_core.Portfolio(portfolio_assets)

        logger.info(f"[Python Wrapper] Processing asset paths...")
        asset_paths = []
        for i in range(len(assets)):
            asset_path = [paths[sim * len(assets) + i] for sim in range(sim_params[0].num_simulations)]
            asset_paths.append(asset_path[0] if asset_path else [])

        logger.info(f"[Python Wrapper] Updating portfolio prices...")
        portfolio.update_prices(asset_paths)
        logger.info(f"[Python Wrapper] Portfolio simulation complete")

        return np.array(paths), portfolio

    def calculate_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        logger.info(f"[Python Wrapper] calculate_risk_metrics called")
        logger.info(f"[Python Wrapper] Returns array shape: {returns.shape}")

        returns_list = returns.flatten().tolist()
        logger.info(f"[Python Wrapper] Flattened returns list size: {len(returns_list)}")

        logger.info(f"[Python Wrapper] Calling C++ RiskAnalysis methods...")
        start_time = time.time()
        metrics = {
            'var_95': stock_sim_core.RiskAnalysis.calculate_var(returns_list, 0.05),
            'var_99': stock_sim_core.RiskAnalysis.calculate_var(returns_list, 0.01),
            'cvar_95': stock_sim_core.RiskAnalysis.calculate_cvar(returns_list, 0.05),
            'cvar_99': stock_sim_core.RiskAnalysis.calculate_cvar(returns_list, 0.01),
            'sortino_ratio': stock_sim_core.RiskAnalysis.calculate_sortino_ratio(returns_list),
        }
        elapsed = time.time() - start_time
        logger.info(f"[Python Wrapper] Risk metrics calculated in {elapsed:.4f}s")

        return metrics

    def estimate_market_parameters(self, price_data: np.ndarray) -> stock_sim_core.MarketParams:
        prices_list = price_data.tolist()
        return stock_sim_core.MarketData.estimate_parameters(prices_list)

    def get_correlation_matrix(self, multi_asset_returns: List[np.ndarray]) -> np.ndarray:
        returns_list = [returns.tolist() for returns in multi_asset_returns]
        corr_matrix = stock_sim_core.MarketData.generate_correlation_matrix(returns_list)
        return np.array(corr_matrix)

    def create_portfolio(self, name: str, assets: List[Dict]) -> stock_sim_core.Portfolio:
        portfolio_assets = []

        for asset in assets:
            portfolio_asset = stock_sim_core.Asset()
            portfolio_asset.symbol = asset.get('symbol', 'UNKNOWN')
            portfolio_asset.weight = asset.get('weight', 1.0 / len(assets))
            portfolio_asset.initial_price = asset.get('initial_price', 100.0)
            portfolio_assets.append(portfolio_asset)

        portfolio = stock_sim_core.Portfolio(portfolio_assets)
        self.portfolios[name] = portfolio
        return portfolio

    def get_portfolio(self, name: str) -> Optional[stock_sim_core.Portfolio]:
        return self.portfolios.get(name)

    def run_monte_carlo_var(
        self,
        simulated_paths: np.ndarray,
        confidence_level: float = 0.05
    ) -> np.ndarray:
        paths_list = [path.tolist() for path in simulated_paths]
        var_estimates = stock_sim_core.RiskAnalysis.monte_carlo_var(paths_list, confidence_level)
        return np.array(var_estimates)