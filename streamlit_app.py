import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import our simulator (will work after compilation)
try:
    from python.simulator import StockMarketSimulator
    from python.utils import DataLoader, ConfigManager

    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False
    st.error("Please compile the C++ module first by running: pip install -e .")

st.set_page_config(
    page_title="Stock Market Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Stock Market Simulator")
st.markdown(
    "Monte Carlo Markov Chain (MCMC) based stock market simulation with real-time analysis"
)

if not SIMULATOR_AVAILABLE:
    st.stop()


@st.cache_resource
def get_simulator():
    return StockMarketSimulator()


@st.cache_resource
def get_config_manager():
    return ConfigManager()


def sidebar_controls():
    st.sidebar.header("Simulation Parameters")

    sim_type = st.sidebar.selectbox(
        "Simulation Type", ["Single Asset", "Multi-Asset Portfolio"]
    )

    st.sidebar.subheader("Market Parameters")
    initial_price = st.sidebar.number_input("Initial Price", value=100.0, min_value=1.0)
    drift = st.sidebar.slider("Annual Drift", -0.5, 0.5, 0.05, 0.01)
    volatility = st.sidebar.slider("Volatility", 0.01, 1.0, 0.2, 0.01)

    st.sidebar.subheader("Jump Process")
    jump_intensity = st.sidebar.slider("Jump Intensity", 0.0, 1.0, 0.1, 0.01)
    jump_mean = st.sidebar.slider("Jump Mean", -0.1, 0.1, 0.0, 0.01)
    jump_std = st.sidebar.slider("Jump Std", 0.01, 0.2, 0.05, 0.01)

    st.sidebar.subheader("Simulation Settings")
    num_simulations = st.sidebar.selectbox(
        "Number of Simulations", [100, 500, 1000, 5000], index=2
    )
    num_steps = st.sidebar.selectbox("Time Steps", [63, 126, 252, 504], index=2)

    return {
        "sim_type": sim_type,
        "initial_price": initial_price,
        "drift": drift,
        "volatility": volatility,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
        "num_simulations": num_simulations,
        "num_steps": num_steps,
    }


@st.cache_data(ttl=3600)
def compute_path_statistics(paths: np.ndarray):
    """Compute and cache expensive statistics."""
    return {
        "mean_path": np.mean(paths, axis=0),
        "percentile_95": np.percentile(paths, 95, axis=0),
        "percentile_5": np.percentile(paths, 5, axis=0),
    }


@st.cache_data(ttl=3600)
def compute_returns(paths: np.ndarray):
    """Compute and cache returns calculation."""
    return np.diff(np.log(paths), axis=1).flatten()


@st.cache_data(ttl=3600)
def calculate_risk_metrics_cached(returns: np.ndarray, _simulator) -> Dict[str, float]:
    """Cache risk metrics calculation. _simulator is not hashed."""
    return _simulator.calculate_risk_metrics(returns)


@st.cache_data(ttl=3600)
def plot_simulation_paths(
    paths: np.ndarray, title: str = "Simulated Price Paths", num_paths_to_show: int = 10
):
    """Optimized plotting with WebGL and caching."""
    fig = go.Figure()

    time_axis = np.arange(paths.shape[1])

    # Downsample time axis if too many points (keep max 100 points per path)
    downsample_factor = max(1, paths.shape[1] // 100)
    time_axis_ds = time_axis[::downsample_factor]

    # Show subset of paths
    paths_to_show = min(num_paths_to_show, paths.shape[0])
    indices = np.random.choice(paths.shape[0], paths_to_show, replace=False)

    # Use Scattergl for WebGL rendering (much faster)
    for i in indices[:10]:  # Show first 10 for clarity
        fig.add_trace(
            go.Scattergl(  # Changed to Scattergl for WebGL
                x=time_axis_ds,
                y=paths[i][::downsample_factor],  # Downsampled
                mode="lines",
                opacity=0.3,
                line=dict(width=1),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Compute statistics (will be cached)
    stats = compute_path_statistics(paths)

    # Add mean path (full resolution)
    fig.add_trace(
        go.Scattergl(  # Changed to Scattergl
            x=time_axis,
            y=stats["mean_path"],
            mode="lines",
            line=dict(color="red", width=3),
            name="Mean Path",
        )
    )

    # Add confidence intervals (full resolution)
    fig.add_trace(
        go.Scattergl(  # Changed to Scattergl
            x=time_axis,
            y=stats["percentile_95"],
            mode="lines",
            line=dict(color="rgba(255,0,0,0)", width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scattergl(  # Changed to Scattergl
            x=time_axis,
            y=stats["percentile_5"],
            fill="tonexty",
            mode="lines",
            line=dict(color="rgba(255,0,0,0)", width=0),
            name="90% Confidence Interval",
            fillcolor="rgba(255,0,0,0.1)",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Steps",
        yaxis_title="Price",
        height=500,
        showlegend=True,
        hovermode="closest",
    )

    return fig


@st.cache_data(ttl=3600)
def plot_returns_distribution(
    returns: np.ndarray, title: str = "Returns Distribution", max_points: int = 10000
):
    """Optimized returns distribution with downsampling and WebGL."""
    from scipy import stats

    # Downsample if too many points
    if len(returns) > max_points:
        downsample_indices = np.random.choice(len(returns), max_points, replace=False)
        returns_display = returns[downsample_indices]
    else:
        returns_display = returns

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Returns Distribution", "Q-Q Plot vs Normal"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]],
    )

    # Histogram with downsampled data
    fig.add_trace(
        go.Histogram(x=returns_display, nbinsx=50, opacity=0.7, name="Returns"),
        row=1,
        col=1,
    )

    # Q-Q plot with downsampled data
    qq_result = stats.probplot(returns_display, dist="norm")

    fig.add_trace(
        go.Scattergl(  # Use Scattergl for better performance
            x=qq_result[0][0],
            y=qq_result[0][1],
            mode="markers",
            name="Sample Quantiles",
            marker=dict(size=4),
        ),
        row=1,
        col=2,
    )

    # Add reference line for Q-Q plot
    line_x = np.array([qq_result[0][0].min(), qq_result[0][0].max()])
    line_y = qq_result[1][1] + qq_result[1][0] * line_x

    fig.add_trace(
        go.Scattergl(  # Use Scattergl
            x=line_x,
            y=line_y,
            mode="lines",
            name="Normal Reference",
            line=dict(color="red", dash="dash"),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=title,
        height=400,
        showlegend=True,
        hovermode="closest",
    )

    return fig


def display_risk_metrics(metrics: Dict[str, float]):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("VaR (95%)", f"{metrics.get('var_95', 0):.4f}")
        st.metric("VaR (99%)", f"{metrics.get('var_99', 0):.4f}")

    with col2:
        st.metric("CVaR (95%)", f"{metrics.get('cvar_95', 0):.4f}")
        st.metric("CVaR (99%)", f"{metrics.get('cvar_99', 0):.4f}")

    with col3:
        st.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.4f}")

    with col4:
        if "sharpe_ratio" in metrics:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}")


def main():
    simulator = get_simulator()
    config_manager = get_config_manager()

    params = sidebar_controls()

    if st.sidebar.button("Run Simulation", type="primary"):
        logger.info("=== Run Simulation Button Clicked ===")

        # Track completion status and time
        simulation_complete = False
        total_time = 0

        with st.spinner("Running Monte Carlo simulation..."):
            logger.info("=== Entered spinner context ===")
            start_time = time.time()
            logger.info("=== Starting Simulation ===")
            logger.info(f"Simulation type: {params['sim_type']}")

            if params["sim_type"] == "Single Asset":
                # Single asset simulation
                logger.info("Creating simulation parameters...")
                logger.info(
                    f"Parameters: num_simulations={params['num_simulations']}, num_steps={params['num_steps']}, "
                    f"drift={params['drift']}, volatility={params['volatility']}"
                )

                param_start = time.time()
                sim_params = simulator.create_simulation_params(
                    initial_price=params["initial_price"],
                    drift=params["drift"],
                    volatility=params["volatility"],
                    jump_intensity=params["jump_intensity"],
                    jump_mean=params["jump_mean"],
                    jump_std=params["jump_std"],
                    num_steps=params["num_steps"],
                    num_simulations=params["num_simulations"],
                )
                logger.info(f"Parameters created in {time.time() - param_start:.4f}s")

                logger.info("Starting single asset simulation (C++ execution)...")
                sim_start = time.time()
                paths = simulator.simulate_single_asset(sim_params)
                sim_elapsed = time.time() - sim_start
                logger.info(f"Simulation completed in {sim_elapsed:.4f}s")
                logger.info(f"Generated paths shape: {paths.shape}")

                st.subheader("Simulation Results")

                # Compute statistics and returns using cached functions
                logger.info("Computing cached statistics and returns...")
                stats_start = time.time()
                final_prices = paths[:, -1]
                returns = compute_returns(paths)  # Use cached function
                logger.info(
                    f"Statistics calculated in {time.time() - stats_start:.4f}s"
                )
                logger.info(f"Returns array size: {returns.shape}")

                # Display summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Final Price", f"${np.mean(final_prices):.2f}")
                with col2:
                    st.metric("Std Final Price", f"${np.std(final_prices):.2f}")
                with col3:
                    st.metric("Simulation Time", f"{sim_elapsed:.2f}s")

                # Use tabs for lazy loading of visualizations
                tab1, tab2, tab3 = st.tabs(
                    ["📈 Price Paths", "📊 Returns Distribution", "⚠️ Risk Analysis"]
                )

                # Plotly config for better performance
                plotly_config = {"displayModeBar": False, "responsive": True}

                with tab1:
                    try:
                        logger.info("Creating price paths plot...")
                        plot_start = time.time()
                        fig_paths = plot_simulation_paths(
                            paths, "Monte Carlo Price Paths"
                        )
                        logger.info(
                            f"Price paths plot created in {time.time() - plot_start:.4f}s"
                        )

                        logger.info("Rendering price paths chart...")
                        render_start = time.time()
                        st.plotly_chart(
                            fig_paths, width="stretch", config=plotly_config
                        )
                        logger.info(
                            f"Price paths chart rendered in {time.time() - render_start:.4f}s"
                        )
                    except Exception as e:
                        logger.error(f"Error plotting price paths: {e}", exc_info=True)
                        st.error(f"Error creating price paths plot: {e}")

                with tab2:
                    try:
                        logger.info("Creating returns distribution plot...")
                        plot_start = time.time()
                        fig_dist = plot_returns_distribution(
                            returns, "Log Returns Distribution"
                        )
                        logger.info(
                            f"Returns distribution plot created in {time.time() - plot_start:.4f}s"
                        )

                        logger.info("Rendering returns distribution chart...")
                        render_start = time.time()
                        st.plotly_chart(fig_dist, width="stretch", config=plotly_config)
                        logger.info(
                            f"Returns distribution chart rendered in {time.time() - render_start:.4f}s"
                        )
                    except Exception as e:
                        logger.error(
                            f"Error plotting returns distribution: {e}", exc_info=True
                        )
                        st.error(f"Error creating returns distribution plot: {e}")

                with tab3:
                    try:
                        logger.info("Calculating risk metrics...")
                        risk_start = time.time()
                        risk_metrics = calculate_risk_metrics_cached(
                            returns, simulator
                        )  # Use cached function
                        logger.info(
                            f"Risk metrics calculated in {time.time() - risk_start:.4f}s"
                        )

                        logger.info("Displaying risk metrics...")
                        display_risk_metrics(risk_metrics)
                        logger.info("Risk metrics displayed successfully")
                    except Exception as e:
                        logger.error(f"Error with risk metrics: {e}", exc_info=True)
                        st.error(f"Error calculating risk metrics: {e}")

                # Mark completion - DON'T call st.success here, do it outside spinner
                total_time = time.time() - start_time
                simulation_complete = True
                logger.info(
                    f"=== Single Asset Simulation Complete === Total time: {total_time:.4f}s"
                )
                logger.info("=== About to exit spinner context ===")

            elif params["sim_type"] == "Multi-Asset Portfolio":
                st.subheader("Portfolio Simulation")

                # Portfolio setup
                st.write("Configure your portfolio assets:")

                if "portfolio_assets" not in st.session_state:
                    st.session_state.portfolio_assets = [
                        {"symbol": "AAPL", "weight": 0.4, "params": {}},
                        {"symbol": "GOOGL", "weight": 0.3, "params": {}},
                        {"symbol": "MSFT", "weight": 0.3, "params": {}},
                    ]

                # Simple portfolio display for demo
                for i, asset in enumerate(st.session_state.portfolio_assets):
                    col1, col2 = st.columns(2)
                    with col1:
                        asset["symbol"] = st.text_input(
                            f"Asset {i + 1} Symbol",
                            value=asset["symbol"],
                            key=f"symbol_{i}",
                        )
                    with col2:
                        asset["weight"] = st.number_input(
                            f"Weight",
                            value=asset["weight"],
                            min_value=0.0,
                            max_value=1.0,
                            key=f"weight_{i}",
                        )

                    # Use main parameters for all assets
                    asset["params"] = {
                        "initial_price": params["initial_price"],
                        "drift": params["drift"],
                        "volatility": params["volatility"],
                        "jump_intensity": params["jump_intensity"],
                        "jump_mean": params["jump_mean"],
                        "jump_std": params["jump_std"],
                        "num_steps": params["num_steps"],
                        "num_simulations": params["num_simulations"],
                    }

                # Correlation matrix
                num_assets = len(st.session_state.portfolio_assets)
                correlation = st.slider("Asset Correlation", -1.0, 1.0, 0.3, 0.1)
                correlation_matrix = np.full((num_assets, num_assets), correlation)
                np.fill_diagonal(correlation_matrix, 1.0)

                # Run portfolio simulation
                logger.info(
                    f"Starting portfolio simulation with {num_assets} assets..."
                )
                logger.info(f"Correlation: {correlation}")
                portfolio_start = time.time()
                paths, portfolio = simulator.simulate_portfolio(
                    st.session_state.portfolio_assets, correlation_matrix
                )
                logger.info(
                    f"Portfolio simulation completed in {time.time() - portfolio_start:.4f}s"
                )

                # Display results
                logger.info("Calculating portfolio values...")
                values_start = time.time()
                portfolio_values = portfolio.calculate_portfolio_values()
                logger.info(
                    f"Portfolio values calculated in {time.time() - values_start:.4f}s"
                )

                fig_portfolio = go.Figure()
                fig_portfolio.add_trace(
                    go.Scattergl(  # Use Scattergl for better performance
                        x=list(range(len(portfolio_values))),
                        y=portfolio_values,
                        mode="lines",
                        name="Portfolio Value",
                    )
                )

                fig_portfolio.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Time Steps",
                    yaxis_title="Portfolio Value",
                    height=400,
                    hovermode="closest",
                )

                # Plotly config for better performance
                plotly_config = {"displayModeBar": False, "responsive": True}

                st.plotly_chart(fig_portfolio, width="stretch", config=plotly_config)

                # Portfolio metrics
                st.subheader("Portfolio Metrics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Sharpe Ratio", f"{portfolio.calculate_sharpe_ratio():.4f}"
                    )
                with col2:
                    st.metric("Volatility", f"{portfolio.calculate_volatility():.4f}")
                with col3:
                    st.metric(
                        "Max Drawdown", f"{portfolio.calculate_max_drawdown():.4f}"
                    )
                with col4:
                    st.metric("Final Value", f"{portfolio_values[-1]:.4f}")

                # Mark completion - DON'T call st.success here, do it outside spinner
                total_time = time.time() - start_time
                simulation_complete = True
                logger.info(
                    f"=== Portfolio Simulation Complete === Total time: {total_time:.4f}s"
                )
                logger.info("=== About to exit spinner context ===")

        logger.info("=== Exited spinner context successfully ===")

        # Now display success message AFTER spinner closes
        if simulation_complete:
            logger.info("=== Displaying success message ===")
            st.success(f"✓ Simulation completed successfully in {total_time:.2f}s")
            logger.info("=== Success message displayed ===")

            # Force UI update
            logger.info("=== Forcing UI update ===")
            st.write("")  # Empty write to force render
            logger.info("=== All operations complete ===")

    # Real-time data section
    st.sidebar.subheader("Real Market Data")
    if st.sidebar.button("Fetch Market Data"):
        with st.spinner("Fetching market data..."):
            try:
                logger.info("=== Fetching Market Data ===")
                data_loader = DataLoader()
                symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
                logger.info(f"Fetching data for symbols: {symbols}")
                fetch_start = time.time()
                market_data = data_loader.prepare_simulation_data(symbols, period="6mo")
                logger.info(f"Market data fetched in {time.time() - fetch_start:.4f}s")

                st.subheader("Market Data Analysis")

                for symbol, data in market_data.items():
                    with st.expander(f"{symbol} Analysis"):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Current Price", f"${data['current_price']:.2f}")
                        with col2:
                            st.metric("Mean Return", f"{data['mean_return']:.4f}")
                        with col3:
                            st.metric("Volatility", f"{data['volatility']:.4f}")

                        # Price chart
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scattergl(  # Use Scattergl for better performance
                                x=list(range(len(data["prices"]))),
                                y=data["prices"],
                                mode="lines",
                                name=f"{symbol} Price",
                            )
                        )
                        fig.update_layout(
                            title=f"{symbol} Price History",
                            height=300,
                            hovermode="closest",
                        )

                        # Plotly config for better performance
                        plotly_config = {"displayModeBar": False, "responsive": True}

                        st.plotly_chart(fig, width="stretch", config=plotly_config)

            except Exception as e:
                st.error(f"Error fetching market data: {e}")


if __name__ == "__main__":
    main()
