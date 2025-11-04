import pandas as pd
import numpy as np
import yfinance as yf
from typing import List, Dict, Optional, Union
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def fetch_yahoo_data(
        symbols: Union[str, List[str]],
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(f"[DataLoader] Fetching Yahoo Finance data for {len(symbols)} symbols: {symbols}")
        logger.info(f"[DataLoader] Parameters: period={period}, interval={interval}")

        start_time = time.time()
        data = yf.download(symbols, period=period, interval=interval)
        elapsed = time.time() - start_time

        logger.info(f"[DataLoader] Yahoo Finance download completed in {elapsed:.4f}s")
        logger.info(f"[DataLoader] Downloaded data shape: {data.shape}")

        if len(symbols) == 1:
            data.columns = [f"{symbols[0]}_{col}" for col in data.columns]
        else:
            data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]

        logger.info(f"[DataLoader] Data columns after formatting: {list(data.columns)}")

        return data

    @staticmethod
    def load_csv_data(file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        return prices.pct_change().dropna()

    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def resample_data(data: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
        return data.resample(freq).last().dropna()

    @staticmethod
    def prepare_simulation_data(
        symbols: List[str],
        period: str = "1y"
    ) -> Dict[str, Dict]:
        logger.info(f"[DataLoader] prepare_simulation_data called for {len(symbols)} symbols")

        start_time = time.time()
        data = DataLoader.fetch_yahoo_data(symbols, period=period)
        fetch_elapsed = time.time() - start_time
        logger.info(f"[DataLoader] Data fetching took {fetch_elapsed:.4f}s")

        simulation_data = {}

        logger.info(f"[DataLoader] Processing data for each symbol...")
        for symbol in symbols:
            try:
                logger.info(f"[DataLoader] Processing symbol: {symbol}")
                close_col = f"{symbol}_Close"

                if close_col in data.columns:
                    process_start = time.time()
                    prices = data[close_col].dropna()
                    logger.info(f"[DataLoader] {symbol}: Found {len(prices)} price points")

                    log_returns = DataLoader.calculate_log_returns(prices)
                    logger.info(f"[DataLoader] {symbol}: Calculated {len(log_returns)} log returns")

                    simulation_data[symbol] = {
                        'prices': prices.values,
                        'log_returns': log_returns.values,
                        'current_price': prices.iloc[-1],
                        'mean_return': log_returns.mean(),
                        'volatility': log_returns.std(),
                        'dates': prices.index.tolist()
                    }

                    process_elapsed = time.time() - process_start
                    logger.info(f"[DataLoader] {symbol}: Processed in {process_elapsed:.4f}s "
                              f"(mean_return={log_returns.mean():.6f}, vol={log_returns.std():.6f})")
                else:
                    logger.warning(f"[DataLoader] {symbol}: Close column '{close_col}' not found in data")

            except Exception as e:
                logger.error(f"[DataLoader] Error processing {symbol}: {e}", exc_info=True)

        total_elapsed = time.time() - start_time
        logger.info(f"[DataLoader] prepare_simulation_data completed in {total_elapsed:.4f}s")
        logger.info(f"[DataLoader] Successfully processed {len(simulation_data)} out of {len(symbols)} symbols")

        return simulation_data


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config.json"
        self.config = self._load_default_config()

    def _load_default_config(self) -> Dict:
        return {
            "simulation": {
                "num_simulations": 1000,
                "num_steps": 252,
                "dt": 1.0 / 252.0,
                "confidence_levels": [0.01, 0.05, 0.1]
            },
            "assets": {
                "default_params": {
                    "drift": 0.05,
                    "volatility": 0.2,
                    "jump_intensity": 0.1,
                    "jump_mean": 0.0,
                    "jump_std": 0.05
                }
            },
            "portfolio": {
                "rebalancing": "monthly",
                "risk_free_rate": 0.02
            },
            "display": {
                "chart_height": 400,
                "chart_width": 800,
                "color_scheme": "plotly"
            }
        }

    def load_config(self, file_path: Optional[str] = None) -> Dict:
        path = file_path or self.config_path

        try:
            with open(path, 'r') as f:
                loaded_config = json.load(f)
                self.config.update(loaded_config)
        except FileNotFoundError:
            print(f"Config file {path} not found. Using default configuration.")
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}. Using default configuration.")

        return self.config

    def save_config(self, file_path: Optional[str] = None):
        path = file_path or self.config_path

        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

    def get(self, key: str, default=None):
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value):
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def get_asset_config(self, symbol: str) -> Dict:
        asset_configs = self.get("assets.configs", {})
        default_params = self.get("assets.default_params", {})

        if symbol in asset_configs:
            config = default_params.copy()
            config.update(asset_configs[symbol])
            return config

        return default_params

    def add_asset_config(self, symbol: str, params: Dict):
        if "assets" not in self.config:
            self.config["assets"] = {}
        if "configs" not in self.config["assets"]:
            self.config["assets"]["configs"] = {}

        self.config["assets"]["configs"][symbol] = params


class PerformanceTracker:
    def __init__(self):
        self.metrics_history = []

    def track_simulation(self, simulation_id: str, metrics: Dict, duration: float):
        entry = {
            'simulation_id': simulation_id,
            'timestamp': pd.Timestamp.now(),
            'duration': duration,
            'metrics': metrics
        }
        self.metrics_history.append(entry)

    def get_performance_summary(self) -> pd.DataFrame:
        if not self.metrics_history:
            return pd.DataFrame()

        df = pd.DataFrame(self.metrics_history)
        return df

    def export_metrics(self, file_path: str):
        df = self.get_performance_summary()
        df.to_csv(file_path, index=False)