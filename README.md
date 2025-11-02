# Stock Market Simulator

A high-performance stock market simulator using Monte Carlo Markov Chain methods with C++ core algorithms and Python interface.

## Features

- High-performance C++ core with optimized MCMC algorithms
- Geometric Brownian Motion with jump diffusion processes
- Multi-asset portfolio simulation with correlation modeling
- Risk analysis tools including VaR, CVaR, Sharpe ratio, and Sortino ratio
- Interactive Streamlit dashboard for real-time visualization
- Python API with NumPy and Pandas integration

## Installation

### Prerequisites

- Python 3.8+
- C++17 compatible compiler
- CMake 3.12+
- uv (for package management)

### Setup

1. Clone and enter the project directory:
   ```bash
   git clone <repository-url>
   cd stock-market-simulator
   ```

2. Install Python dependencies using uv:
   ```bash
   uv sync
   ```

3. Compile the C++ extension:
   ```bash
   uv pip install -e .
   ```

4. Verify installation:
   ```bash
   source .venv/bin/activate
   python -c "import stock_sim_core; print('Installation successful!')"
   ```

## Quick Start

### Basic Simulation

```python
from python.simulator import StockMarketSimulator
import numpy as np

# Initialize simulator
simulator = StockMarketSimulator(seed=42)

# Create simulation parameters
params = simulator.create_simulation_params(
    initial_price=100.0,
    drift=0.08,
    volatility=0.25,
    jump_intensity=0.1,
    num_steps=252,
    num_simulations=1000
)

# Run simulation
paths = simulator.simulate_single_asset(params)
print(f"Generated {paths.shape[0]} paths with {paths.shape[1]} steps each")
```

### Portfolio Simulation

```python
# Define portfolio assets
assets = [
    {
        'symbol': 'TECH',
        'weight': 0.6,
        'params': {
            'initial_price': 150.0,
            'drift': 0.12,
            'volatility': 0.30
        }
    },
    {
        'symbol': 'BONDS',
        'weight': 0.4,
        'params': {
            'initial_price': 100.0,
            'drift': 0.04,
            'volatility': 0.10
        }
    }
]

# Define correlation
correlation_matrix = np.array([[1.0, -0.2], [-0.2, 1.0]])

# Simulate portfolio
paths, portfolio = simulator.simulate_portfolio(assets, correlation_matrix)

# Calculate performance metrics
sharpe_ratio = portfolio.calculate_sharpe_ratio()
max_drawdown = portfolio.calculate_max_drawdown()
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
```

## Streamlit Dashboard

Launch the interactive dashboard:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Features:
- Real-time parameter adjustment
- Interactive visualizations
- Monte Carlo simulation controls
- Risk analysis dashboard
- Portfolio construction tools

## Examples

Run the included examples:

```bash
source .venv/bin/activate
python examples/basic_simulation.py
python examples/advanced_analysis.py
```

## Testing

Run the test suite:

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Architecture

```
stock-market-simulator/
├── cpp/                     # C++ core implementation
│   ├── include/            # Header files
│   │   ├── mcmc_engine.h   # MCMC simulation engine
│   │   ├── portfolio.h     # Portfolio management
│   │   ├── risk_analysis.h # Risk metrics calculation
│   │   └── market_data.h   # Market data utilities
│   └── src/                # Source files
├── python/                 # Python interface
│   ├── bindings/           # pybind11 bindings
│   ├── simulator.py        # Main simulator class
│   └── utils.py           # Utilities and data loading
├── tests/                  # Test suite
├── examples/               # Usage examples
├── streamlit_app.py        # Streamlit dashboard
└── pyproject.toml         # Project configuration
```

## Core Algorithms

### MCMC Engine
- Geometric Brownian Motion with jump diffusion
- Correlated multi-asset simulation
- Efficient random number generation
- Optimized C++ implementation

### Risk Analysis
- Value at Risk (VaR) and Conditional VaR
- Monte Carlo VaR estimation
- Sharpe and Sortino ratios
- Maximum drawdown calculation
- Beta calculation

### Portfolio Management
- Multi-asset portfolio construction
- Weight normalization
- Performance attribution
- Risk-return optimization

## Configuration

The simulator uses a flexible configuration system. Example configuration:

```json
{
  "simulation": {
    "num_simulations": 1000,
    "num_steps": 252,
    "confidence_levels": [0.01, 0.05, 0.1]
  },
  "assets": {
    "default_params": {
      "drift": 0.05,
      "volatility": 0.2,
      "jump_intensity": 0.1
    }
  }
}
```

## Performance

The C++ core provides significant performance advantages:
- 10-100x faster than pure Python implementations
- Efficient memory usage for large simulations
- Optimized linear algebra operations
- Parallel-friendly design

## API Reference

### StockMarketSimulator

Main simulator class providing high-level interface.

Methods:
- `simulate_single_asset(params)`: Single asset simulation
- `simulate_portfolio(assets, correlation_matrix)`: Multi-asset simulation
- `calculate_risk_metrics(returns)`: Risk analysis
- `estimate_market_parameters(prices)`: Parameter estimation

### SimulationParams

Configuration for simulation parameters.

Attributes:
- `initial_price`: Starting price
- `drift`: Annual drift rate
- `volatility`: Annual volatility
- `jump_intensity`: Jump frequency
- `num_steps`: Simulation steps
- `num_simulations`: Number of Monte Carlo paths
