#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mcmc_engine.h"
#include "portfolio.h"
#include "risk_analysis.h"
#include "market_data.h"

namespace py = pybind11;

PYBIND11_MODULE(stock_sim_core, m) {
    m.doc() = "High-performance stock market simulator using MCMC methods";

    // MCMCEngine
    py::class_<MCMCEngine>(m, "MCMCEngine")
        .def(py::init<unsigned int>(), py::arg("seed") = std::random_device{}())
        .def("simulate_gbm_with_jumps", &MCMCEngine::simulate_gbm_with_jumps)
        .def("simulate_multi_asset", &MCMCEngine::simulate_multi_asset)
        .def("generate_correlated_normals", &MCMCEngine::generate_correlated_normals);

    py::class_<MCMCEngine::SimulationParams>(m, "SimulationParams")
        .def(py::init<>())
        .def_readwrite("initial_price", &MCMCEngine::SimulationParams::initial_price)
        .def_readwrite("drift", &MCMCEngine::SimulationParams::drift)
        .def_readwrite("volatility", &MCMCEngine::SimulationParams::volatility)
        .def_readwrite("jump_intensity", &MCMCEngine::SimulationParams::jump_intensity)
        .def_readwrite("jump_mean", &MCMCEngine::SimulationParams::jump_mean)
        .def_readwrite("jump_std", &MCMCEngine::SimulationParams::jump_std)
        .def_readwrite("num_steps", &MCMCEngine::SimulationParams::num_steps)
        .def_readwrite("num_simulations", &MCMCEngine::SimulationParams::num_simulations)
        .def_readwrite("dt", &MCMCEngine::SimulationParams::dt);

    // Portfolio
    py::class_<Portfolio::Asset>(m, "Asset")
        .def(py::init<>())
        .def_readwrite("symbol", &Portfolio::Asset::symbol)
        .def_readwrite("weight", &Portfolio::Asset::weight)
        .def_readwrite("initial_price", &Portfolio::Asset::initial_price)
        .def_readwrite("price_path", &Portfolio::Asset::price_path);

    py::class_<Portfolio>(m, "Portfolio")
        .def(py::init<const std::vector<Portfolio::Asset>&>())
        .def("update_prices", &Portfolio::update_prices)
        .def("calculate_portfolio_values", &Portfolio::calculate_portfolio_values)
        .def("calculate_returns", &Portfolio::calculate_returns)
        .def("calculate_sharpe_ratio", &Portfolio::calculate_sharpe_ratio, py::arg("risk_free_rate") = 0.02)
        .def("calculate_max_drawdown", &Portfolio::calculate_max_drawdown)
        .def("calculate_volatility", &Portfolio::calculate_volatility)
        .def("get_assets", &Portfolio::get_assets, py::return_value_policy::reference)
        .def("get_total_weight", &Portfolio::get_total_weight);

    // RiskAnalysis
    py::class_<RiskAnalysis>(m, "RiskAnalysis")
        .def_static("calculate_var", &RiskAnalysis::calculate_var, py::arg("returns"), py::arg("confidence_level") = 0.05)
        .def_static("calculate_cvar", &RiskAnalysis::calculate_cvar, py::arg("returns"), py::arg("confidence_level") = 0.05)
        .def_static("calculate_sortino_ratio", &RiskAnalysis::calculate_sortino_ratio, py::arg("returns"), py::arg("target_return") = 0.0)
        .def_static("calculate_correlation_matrix", &RiskAnalysis::calculate_correlation_matrix)
        .def_static("calculate_beta", &RiskAnalysis::calculate_beta)
        .def_static("monte_carlo_var", &RiskAnalysis::monte_carlo_var, py::arg("simulated_paths"), py::arg("confidence_level") = 0.05);

    // MarketData
    py::class_<MarketData::HistoricalData>(m, "HistoricalData")
        .def(py::init<>())
        .def_readwrite("prices", &MarketData::HistoricalData::prices)
        .def_readwrite("volumes", &MarketData::HistoricalData::volumes)
        .def_readwrite("dates", &MarketData::HistoricalData::dates);

    py::class_<MarketData::MarketParams>(m, "MarketParams")
        .def(py::init<>())
        .def_readwrite("drift", &MarketData::MarketParams::drift)
        .def_readwrite("volatility", &MarketData::MarketParams::volatility)
        .def_readwrite("jump_intensity", &MarketData::MarketParams::jump_intensity)
        .def_readwrite("current_price", &MarketData::MarketParams::current_price);

    py::class_<MarketData>(m, "MarketData")
        .def_static("estimate_parameters", &MarketData::estimate_parameters)
        .def_static("calculate_log_returns", &MarketData::calculate_log_returns)
        .def_static("estimate_volatility", &MarketData::estimate_volatility)
        .def_static("estimate_drift", &MarketData::estimate_drift)
        .def_static("generate_correlation_matrix", &MarketData::generate_correlation_matrix)
        .def_static("detect_jumps", &MarketData::detect_jumps, py::arg("log_returns"), py::arg("threshold") = 3.0);
}