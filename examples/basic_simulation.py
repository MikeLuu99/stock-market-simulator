#!/usr/bin/env python3
"""
Basic Stock Market Simulation Example

This example demonstrates the core functionality of the stock market simulator:
- Single asset simulation using MCMC methods
- Portfolio construction and analysis
- Risk metric calculations
- Basic visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from python.simulator import StockMarketSimulator
    from python.utils import ConfigManager
    print("✓ Simulator modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    print("Please compile the C++ module first: pip install -e .")
    sys.exit(1)


def basic_single_asset_simulation():
    """Demonstrate basic single asset simulation"""
    print("\n" + "="*50)
    print("BASIC SINGLE ASSET SIMULATION")
    print("="*50)

    # Initialize simulator with fixed seed for reproducible results
    simulator = StockMarketSimulator(seed=42)

    # Create simulation parameters
    params = simulator.create_simulation_params(
        initial_price=100.0,
        drift=0.08,           # 8% annual drift
        volatility=0.25,      # 25% annual volatility
        jump_intensity=0.2,   # 20% chance of jump per year
        jump_mean=-0.02,      # Average jump size -2%
        jump_std=0.03,        # Jump standard deviation 3%
        num_steps=252,        # Daily steps for 1 year
        num_simulations=1000  # 1000 Monte Carlo paths
    )

    print(f"Running {params.num_simulations} simulations with {params.num_steps} steps each...")

    # Run simulation
    paths = simulator.simulate_single_asset(params)

    print(f"Simulation completed. Generated {paths.shape[0]} paths of {paths.shape[1]} steps each.")

    # Analyze results
    final_prices = paths[:, -1]
    initial_price = params.initial_price

    print(f"\nSimulation Results:")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Mean Final Price: ${np.mean(final_prices):.2f}")
    print(f"Median Final Price: ${np.median(final_prices):.2f}")
    print(f"Std Final Price: ${np.std(final_prices):.2f}")
    print(f"Min Final Price: ${np.min(final_prices):.2f}")
    print(f"Max Final Price: ${np.max(final_prices):.2f}")

    # Calculate returns
    log_returns = np.diff(np.log(paths), axis=1).flatten()

    # Risk metrics
    risk_metrics = simulator.calculate_risk_metrics(log_returns)

    print(f"\nRisk Metrics:")
    print(f"Value at Risk (95%): {risk_metrics['var_95']:.4f}")
    print(f"Value at Risk (99%): {risk_metrics['var_99']:.4f}")
    print(f"Conditional VaR (95%): {risk_metrics['cvar_95']:.4f}")
    print(f"Conditional VaR (99%): {risk_metrics['cvar_99']:.4f}")
    print(f"Sortino Ratio: {risk_metrics['sortino_ratio']:.4f}")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot 1: Sample paths
    plt.subplot(2, 2, 1)
    time_axis = np.arange(paths.shape[1])
    for i in range(min(20, paths.shape[0])):  # Show first 20 paths
        plt.plot(time_axis, paths[i], alpha=0.3, color='blue', linewidth=0.5)

    # Plot mean path
    mean_path = np.mean(paths, axis=0)
    plt.plot(time_axis, mean_path, 'r-', linewidth=2, label='Mean Path')

    # Plot confidence intervals
    percentile_95 = np.percentile(paths, 95, axis=0)
    percentile_5 = np.percentile(paths, 5, axis=0)
    plt.fill_between(time_axis, percentile_5, percentile_95, alpha=0.2, color='red', label='90% Confidence Interval')

    plt.title('Monte Carlo Simulation Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Final price distribution
    plt.subplot(2, 2, 2)
    plt.hist(final_prices, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(final_prices), color='red', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(final_prices):.2f}')
    plt.axvline(initial_price, color='green', linestyle='--', linewidth=2, label=f'Initial: ${initial_price:.2f}')
    plt.title('Distribution of Final Prices')
    plt.xlabel('Final Price ($)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Returns distribution
    plt.subplot(2, 2, 3)
    plt.hist(log_returns, bins=100, alpha=0.7, density=True, color='lightgreen', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Return')
    plt.axvline(risk_metrics['var_95'], color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {risk_metrics["var_95"]:.4f}')
    plt.title('Log Returns Distribution')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Cumulative returns
    plt.subplot(2, 2, 4)
    cumulative_returns = np.cumprod(1 + np.diff(paths, axis=1) / paths[:, :-1], axis=1)
    mean_cumulative = np.mean(cumulative_returns, axis=0)
    plt.plot(time_axis[1:], mean_cumulative, 'b-', linewidth=2, label='Mean Cumulative Return')

    percentile_95_cum = np.percentile(cumulative_returns, 95, axis=0)
    percentile_5_cum = np.percentile(cumulative_returns, 5, axis=0)
    plt.fill_between(time_axis[1:], percentile_5_cum, percentile_95_cum, alpha=0.2, color='blue', label='90% Confidence Interval')

    plt.title('Cumulative Returns')
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('basic_simulation_results.png', dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as 'basic_simulation_results.png'")

    return paths, risk_metrics


def portfolio_simulation_example():
    """Demonstrate multi-asset portfolio simulation"""
    print("\n" + "="*50)
    print("PORTFOLIO SIMULATION")
    print("="*50)

    simulator = StockMarketSimulator(seed=123)

    # Define portfolio assets with different characteristics
    assets = [
        {
            'symbol': 'TECH_STOCK',
            'weight': 0.4,
            'params': {
                'initial_price': 150.0,
                'drift': 0.12,
                'volatility': 0.35,
                'jump_intensity': 0.3,
                'num_steps': 252,
                'num_simulations': 500
            }
        },
        {
            'symbol': 'BLUE_CHIP',
            'weight': 0.4,
            'params': {
                'initial_price': 85.0,
                'drift': 0.08,
                'volatility': 0.18,
                'jump_intensity': 0.1,
                'num_steps': 252,
                'num_simulations': 500
            }
        },
        {
            'symbol': 'BOND_ETF',
            'weight': 0.2,
            'params': {
                'initial_price': 50.0,
                'drift': 0.03,
                'volatility': 0.08,
                'jump_intensity': 0.05,
                'num_steps': 252,
                'num_simulations': 500
            }
        }
    ]

    # Define correlation matrix
    correlation_matrix = np.array([
        [1.00, 0.65, -0.20],  # TECH_STOCK correlations
        [0.65, 1.00, -0.10],  # BLUE_CHIP correlations
        [-0.20, -0.10, 1.00]  # BOND_ETF correlations
    ])

    print("Portfolio Configuration:")
    for asset in assets:
        print(f"  {asset['symbol']}: {asset['weight']*100:.1f}% weight")

    # Run portfolio simulation
    paths, portfolio = simulator.simulate_portfolio(assets, correlation_matrix)

    # Calculate portfolio metrics
    portfolio_values = portfolio.calculate_portfolio_values()
    portfolio_returns = portfolio.calculate_returns()

    sharpe_ratio = portfolio.calculate_sharpe_ratio(risk_free_rate=0.02)
    max_drawdown = portfolio.calculate_max_drawdown()
    volatility = portfolio.calculate_volatility()

    print(f"\nPortfolio Performance Metrics:")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Annualized Volatility: {volatility:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")
    print(f"Final Portfolio Value: {portfolio_values[-1]:.4f}")

    # Portfolio risk metrics
    if portfolio_returns:
        portfolio_risk_metrics = simulator.calculate_risk_metrics(np.array(portfolio_returns))
        print(f"\nPortfolio Risk Metrics:")
        print(f"Portfolio VaR (95%): {portfolio_risk_metrics['var_95']:.4f}")
        print(f"Portfolio CVaR (95%): {portfolio_risk_metrics['cvar_95']:.4f}")

    # Visualization
    plt.figure(figsize=(12, 8))

    # Portfolio value over time
    plt.subplot(2, 2, 1)
    time_axis = np.arange(len(portfolio_values))
    plt.plot(time_axis, portfolio_values, 'b-', linewidth=2)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.grid(True, alpha=0.3)

    # Portfolio returns distribution
    plt.subplot(2, 2, 2)
    if portfolio_returns:
        plt.hist(portfolio_returns, bins=50, alpha=0.7, density=True, color='lightcoral', edgecolor='black')
        plt.title('Portfolio Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)

    # Drawdown analysis
    plt.subplot(2, 2, 3)
    cumulative_max = np.maximum.accumulate(portfolio_values)
    drawdown = (cumulative_max - portfolio_values) / cumulative_max
    plt.plot(time_axis, drawdown, 'r-', linewidth=2)
    plt.fill_between(time_axis, 0, drawdown, alpha=0.3, color='red')
    plt.title('Portfolio Drawdown')
    plt.xlabel('Time Steps')
    plt.ylabel('Drawdown')
    plt.grid(True, alpha=0.3)

    # Asset weight contribution
    plt.subplot(2, 2, 4)
    asset_symbols = [asset['symbol'] for asset in assets]
    asset_weights = [asset['weight'] for asset in assets]
    plt.pie(asset_weights, labels=asset_symbols, autopct='%1.1f%%', startangle=90)
    plt.title('Portfolio Asset Allocation')

    plt.tight_layout()
    plt.savefig('portfolio_simulation_results.png', dpi=300, bbox_inches='tight')
    print(f"Portfolio visualization saved as 'portfolio_simulation_results.png'")

    return portfolio, portfolio_values


def main():
    """Run all examples"""
    print("Stock Market Simulator - Basic Examples")
    print("="*60)

    try:
        # Basic simulation
        paths, risk_metrics = basic_single_asset_simulation()

        # Portfolio simulation
        portfolio, portfolio_values = portfolio_simulation_example()

        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("Check the generated PNG files for visualizations.")
        print("="*60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()