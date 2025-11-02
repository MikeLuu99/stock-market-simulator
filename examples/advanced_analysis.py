#!/usr/bin/env python3
"""
Advanced Stock Market Analysis Example

This example demonstrates advanced features:
- Parameter estimation from historical data
- Scenario analysis and stress testing
- Advanced risk metrics and Monte Carlo VaR
- Performance comparison between different models
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from python.simulator import StockMarketSimulator
    from python.utils import DataLoader, ConfigManager, PerformanceTracker
    print("✓ Advanced modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("Please ensure all dependencies are installed and the C++ module is compiled.")
    sys.exit(1)


def generate_synthetic_market_data(num_days: int = 500) -> pd.DataFrame:
    """Generate synthetic historical data for testing"""
    print(f"Generating {num_days} days of synthetic market data...")

    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=num_days, freq='D')

    # Generate correlated asset returns
    base_return = np.random.normal(0.0008, 0.02, num_days)  # Market factor

    # Asset 1: Tech stock (high volatility, high correlation with market)
    tech_returns = 0.7 * base_return + 0.3 * np.random.normal(0.001, 0.03, num_days)

    # Asset 2: Utility stock (low volatility, moderate correlation)
    utility_returns = 0.4 * base_return + 0.6 * np.random.normal(0.0005, 0.012, num_days)

    # Asset 3: Commodity ETF (negative correlation during stress)
    commodity_returns = -0.2 * base_return + np.random.normal(0.0003, 0.025, num_days)

    # Convert to prices
    tech_prices = 100 * np.cumprod(1 + tech_returns)
    utility_prices = 50 * np.cumprod(1 + utility_returns)
    commodity_prices = 75 * np.cumprod(1 + commodity_returns)

    # Add some jumps
    jump_indices = np.random.choice(num_days, size=int(num_days * 0.02), replace=False)
    for idx in jump_indices:
        jump_size = np.random.normal(-0.05, 0.02)
        tech_prices[idx:] *= (1 + jump_size)
        utility_prices[idx:] *= (1 + jump_size * 0.5)
        commodity_prices[idx:] *= (1 - jump_size * 0.3)

    data = pd.DataFrame({
        'TECH_Close': tech_prices,
        'UTILITY_Close': utility_prices,
        'COMMODITY_Close': commodity_prices,
        'TECH_Volume': np.random.randint(1000000, 5000000, num_days),
        'UTILITY_Volume': np.random.randint(500000, 2000000, num_days),
        'COMMODITY_Volume': np.random.randint(800000, 3000000, num_days)
    }, index=dates)

    return data


def parameter_estimation_example(market_data: pd.DataFrame) -> Dict:
    """Demonstrate parameter estimation from historical data"""
    print("\n" + "="*50)
    print("PARAMETER ESTIMATION FROM HISTORICAL DATA")
    print("="*50)

    simulator = StockMarketSimulator()
    estimated_params = {}

    symbols = ['TECH', 'UTILITY', 'COMMODITY']

    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")

        price_col = f'{symbol}_Close'
        prices = market_data[price_col].values

        # Estimate parameters using our C++ implementation
        market_params = simulator.estimate_market_parameters(prices)

        # Calculate additional statistics
        log_returns = DataLoader.calculate_log_returns(market_data[price_col])

        print(f"  Current Price: ${market_params.current_price:.2f}")
        print(f"  Estimated Drift: {market_params.drift:.4f}")
        print(f"  Estimated Volatility: {market_params.volatility:.4f}")
        print(f"  Jump Intensity: {market_params.jump_intensity:.4f}")

        estimated_params[symbol] = {
            'current_price': market_params.current_price,
            'drift': market_params.drift,
            'volatility': market_params.volatility,
            'jump_intensity': market_params.jump_intensity,
            'log_returns': log_returns.values
        }

    # Estimate correlation matrix
    returns_data = [estimated_params[symbol]['log_returns'] for symbol in symbols]
    correlation_matrix = simulator.get_correlation_matrix(returns_data)

    print(f"\nEstimated Correlation Matrix:")
    correlation_df = pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)
    print(correlation_df.round(4))

    return estimated_params, correlation_matrix


def scenario_analysis_example(estimated_params: Dict, correlation_matrix: np.ndarray) -> Dict:
    """Demonstrate scenario analysis and stress testing"""
    print("\n" + "="*50)
    print("SCENARIO ANALYSIS AND STRESS TESTING")
    print("="*50)

    simulator = StockMarketSimulator(seed=456)

    scenarios = {
        'Base Case': {'stress_factor': 1.0, 'description': 'Normal market conditions'},
        'Market Stress': {'stress_factor': 2.0, 'description': 'Doubled volatility'},
        'Crisis Scenario': {'stress_factor': 3.0, 'description': 'Tripled volatility, increased jumps'},
        'Bull Market': {'stress_factor': 0.5, 'description': 'Reduced volatility, positive drift'}
    }

    scenario_results = {}

    for scenario_name, scenario_config in scenarios.items():
        print(f"\nRunning {scenario_name}: {scenario_config['description']}")

        # Adjust parameters for scenario
        stress_factor = scenario_config['stress_factor']

        assets = []
        symbols = ['TECH', 'UTILITY', 'COMMODITY']
        weights = [0.5, 0.3, 0.2]

        for i, symbol in enumerate(symbols):
            params = estimated_params[symbol]

            # Apply stress factors
            if scenario_name == 'Crisis Scenario':
                adjusted_drift = params['drift'] - 0.1  # Negative drift
                adjusted_volatility = params['volatility'] * stress_factor
                adjusted_jump_intensity = params['jump_intensity'] * 2
            elif scenario_name == 'Bull Market':
                adjusted_drift = params['drift'] + 0.05  # Positive boost
                adjusted_volatility = params['volatility'] * stress_factor
                adjusted_jump_intensity = params['jump_intensity'] * 0.5
            else:
                adjusted_drift = params['drift']
                adjusted_volatility = params['volatility'] * stress_factor
                adjusted_jump_intensity = params['jump_intensity']

            assets.append({
                'symbol': symbol,
                'weight': weights[i],
                'params': {
                    'initial_price': params['current_price'],
                    'drift': adjusted_drift,
                    'volatility': adjusted_volatility,
                    'jump_intensity': adjusted_jump_intensity,
                    'num_steps': 252,
                    'num_simulations': 2000
                }
            })

        # Run simulation
        paths, portfolio = simulator.simulate_portfolio(assets, correlation_matrix)

        # Calculate metrics
        portfolio_values = portfolio.calculate_portfolio_values()
        portfolio_returns = portfolio.calculate_returns()

        if portfolio_returns:
            risk_metrics = simulator.calculate_risk_metrics(np.array(portfolio_returns))
            sharpe_ratio = portfolio.calculate_sharpe_ratio()
            max_drawdown = portfolio.calculate_max_drawdown()

            scenario_results[scenario_name] = {
                'final_value': portfolio_values[-1],
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'var_95': risk_metrics['var_95'],
                'cvar_95': risk_metrics['cvar_95'],
                'volatility': portfolio.calculate_volatility()
            }

            print(f"  Portfolio Performance:")
            print(f"    Final Value: {portfolio_values[-1]:.4f}")
            print(f"    Sharpe Ratio: {sharpe_ratio:.4f}")
            print(f"    Max Drawdown: {max_drawdown:.4f}")
            print(f"    VaR (95%): {risk_metrics['var_95']:.4f}")
            print(f"    CVaR (95%): {risk_metrics['cvar_95']:.4f}")

    return scenario_results


def monte_carlo_var_analysis(estimated_params: Dict, correlation_matrix: np.ndarray) -> np.ndarray:
    """Demonstrate Monte Carlo VaR analysis"""
    print("\n" + "="*50)
    print("MONTE CARLO VALUE-AT-RISK ANALYSIS")
    print("="*50)

    simulator = StockMarketSimulator(seed=789)

    # Create high-precision simulation for VaR
    assets = []
    symbols = ['TECH', 'UTILITY', 'COMMODITY']
    weights = [0.5, 0.3, 0.2]

    for i, symbol in enumerate(symbols):
        params = estimated_params[symbol]
        assets.append({
            'symbol': symbol,
            'weight': weights[i],
            'params': {
                'initial_price': params['current_price'],
                'drift': params['drift'],
                'volatility': params['volatility'],
                'jump_intensity': params['jump_intensity'],
                'num_steps': 21,  # 21 days (1 month)
                'num_simulations': 10000  # High precision
            }
        })

    print("Running high-precision Monte Carlo simulation for VaR analysis...")
    paths, portfolio = simulator.simulate_portfolio(assets, correlation_matrix)

    # Calculate portfolio paths for each simulation
    portfolio_paths = []
    num_assets = len(assets)

    for sim in range(assets[0]['params']['num_simulations']):
        asset_paths = []
        for asset_idx in range(num_assets):
            path_idx = sim * num_assets + asset_idx
            if path_idx < len(paths):
                asset_paths.append(paths[path_idx])

        if len(asset_paths) == num_assets:
            # Calculate portfolio value path for this simulation
            portfolio_path = []
            for step in range(assets[0]['params']['num_steps'] + 1):
                portfolio_value = sum(
                    assets[i]['weight'] * (asset_paths[i][step] / assets[i]['params']['initial_price'])
                    for i in range(num_assets)
                )
                portfolio_path.append(portfolio_value)
            portfolio_paths.append(portfolio_path)

    if portfolio_paths:
        portfolio_paths_array = np.array(portfolio_paths)

        # Calculate VaR estimates for different confidence levels
        confidence_levels = [0.01, 0.05, 0.1]
        var_results = {}

        for confidence in confidence_levels:
            var_estimates = simulator.run_monte_carlo_var(portfolio_paths_array, confidence)
            var_results[confidence] = var_estimates

            print(f"\nVaR Analysis ({(1-confidence)*100:.0f}% confidence):")
            print(f"  1-day VaR: {var_estimates[1]:.4f}")
            print(f"  1-week VaR: {var_estimates[5]:.4f}")
            print(f"  1-month VaR: {var_estimates[-1]:.4f}")

        # Visualization
        plt.figure(figsize=(15, 10))

        # Plot 1: Sample portfolio paths
        plt.subplot(2, 3, 1)
        time_axis = np.arange(portfolio_paths_array.shape[1])
        for i in range(min(50, portfolio_paths_array.shape[0])):
            plt.plot(time_axis, portfolio_paths_array[i], alpha=0.3, color='blue', linewidth=0.5)

        mean_path = np.mean(portfolio_paths_array, axis=0)
        plt.plot(time_axis, mean_path, 'r-', linewidth=2, label='Mean Path')
        plt.title('Portfolio Simulation Paths')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2-4: VaR estimates over time
        for i, confidence in enumerate(confidence_levels):
            plt.subplot(2, 3, i + 2)
            plt.plot(time_axis, var_results[confidence], 'r-', linewidth=2)
            plt.title(f'VaR {(1-confidence)*100:.0f}% Over Time')
            plt.xlabel('Days')
            plt.ylabel('Value at Risk')
            plt.grid(True, alpha=0.3)

        # Plot 5: Final value distribution
        plt.subplot(2, 3, 5)
        final_values = portfolio_paths_array[:, -1]
        plt.hist(final_values, bins=100, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
        plt.axvline(np.percentile(final_values, 5), color='red', linestyle='--', label='5th Percentile')
        plt.axvline(np.percentile(final_values, 1), color='orange', linestyle='--', label='1st Percentile')
        plt.title('Final Portfolio Value Distribution')
        plt.xlabel('Portfolio Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 6: Tail risk analysis
        plt.subplot(2, 3, 6)
        percentiles = np.arange(0.1, 5.1, 0.1)
        tail_values = [np.percentile(final_values, p) for p in percentiles]
        plt.plot(percentiles, tail_values, 'ro-', linewidth=2, markersize=4)
        plt.title('Tail Risk Analysis')
        plt.xlabel('Percentile (%)')
        plt.ylabel('Portfolio Value')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('advanced_analysis_results.png', dpi=300, bbox_inches='tight')
        print(f"\nAdvanced analysis visualization saved as 'advanced_analysis_results.png'")

        return portfolio_paths_array
    else:
        print("Warning: No portfolio paths generated")
        return np.array([])


def performance_comparison(scenario_results: Dict):
    """Compare performance across different scenarios"""
    print("\n" + "="*50)
    print("SCENARIO PERFORMANCE COMPARISON")
    print("="*50)

    if not scenario_results:
        print("No scenario results to compare")
        return

    # Create comparison table
    metrics = ['final_value', 'sharpe_ratio', 'max_drawdown', 'var_95', 'cvar_95', 'volatility']
    comparison_df = pd.DataFrame(scenario_results).T

    print("Performance Comparison Across Scenarios:")
    print(comparison_df.round(4))

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    scenarios = list(scenario_results.keys())

    for i, metric in enumerate(metrics):
        ax = axes[i // 3, i % 3]
        values = [scenario_results[scenario][metric] for scenario in scenarios]

        bars = ax.bar(scenarios, values, alpha=0.7)

        # Color code bars
        colors = ['green', 'yellow', 'red', 'blue']
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Scenario comparison saved as 'scenario_comparison.png'")


def main():
    """Run advanced analysis examples"""
    print("Stock Market Simulator - Advanced Analysis Examples")
    print("="*70)

    # Initialize performance tracker
    tracker = PerformanceTracker()

    try:
        # Generate synthetic market data
        market_data = generate_synthetic_market_data(500)

        # Parameter estimation
        estimated_params, correlation_matrix = parameter_estimation_example(market_data)
        tracker.track_simulation('parameter_estimation', {'status': 'completed'}, 0.0)

        # Scenario analysis
        scenario_results = scenario_analysis_example(estimated_params, correlation_matrix)
        tracker.track_simulation('scenario_analysis', scenario_results, 0.0)

        # Monte Carlo VaR analysis
        portfolio_paths = monte_carlo_var_analysis(estimated_params, correlation_matrix)
        tracker.track_simulation('monte_carlo_var', {'num_paths': len(portfolio_paths)}, 0.0)

        # Performance comparison
        performance_comparison(scenario_results)

        # Export performance metrics
        tracker.export_metrics('performance_log.csv')

        print("\n" + "="*70)
        print("ADVANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Generated files:")
        print("  - advanced_analysis_results.png")
        print("  - scenario_comparison.png")
        print("  - performance_log.csv")
        print("="*70)

    except Exception as e:
        print(f"\nError running advanced analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()