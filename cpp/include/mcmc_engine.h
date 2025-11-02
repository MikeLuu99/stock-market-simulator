#pragma once

#include <vector>
#include <random>
#include <memory>

class MCMCEngine {
public:
    struct SimulationParams {
        double initial_price = 100.0;
        double drift = 0.05;
        double volatility = 0.2;
        double jump_intensity = 0.1;
        double jump_mean = 0.0;
        double jump_std = 0.05;
        int num_steps = 252;
        int num_simulations = 1000;
        double dt = 1.0 / 252.0;
    };

    MCMCEngine(unsigned int seed = std::random_device{}());

    std::vector<std::vector<double>> simulate_gbm_with_jumps(const SimulationParams& params);

    std::vector<std::vector<double>> simulate_multi_asset(
        const std::vector<SimulationParams>& asset_params,
        const std::vector<std::vector<double>>& correlation_matrix
    );

    std::vector<double> generate_correlated_normals(
        const std::vector<double>& correlations,
        size_t num_assets
    );

private:
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::exponential_distribution<double> exp_dist_;
    std::uniform_real_distribution<double> uniform_dist_;

    double generate_jump_time();
    double generate_jump_size(double mean, double std);
};