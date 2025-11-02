#include "mcmc_engine.h"
#include <cmath>
#include <algorithm>

MCMCEngine::MCMCEngine(unsigned int seed)
    : rng_(seed), normal_dist_(0.0, 1.0), exp_dist_(1.0), uniform_dist_(0.0, 1.0) {}

std::vector<std::vector<double>> MCMCEngine::simulate_gbm_with_jumps(const SimulationParams& params) {
    std::vector<std::vector<double>> paths(params.num_simulations, std::vector<double>(params.num_steps + 1));

    for (int sim = 0; sim < params.num_simulations; ++sim) {
        paths[sim][0] = params.initial_price;

        for (int step = 1; step <= params.num_steps; ++step) {
            double prev_price = paths[sim][step - 1];

            // Geometric Brownian Motion component
            double drift_component = (params.drift - 0.5 * params.volatility * params.volatility) * params.dt;
            double diffusion_component = params.volatility * std::sqrt(params.dt) * normal_dist_(rng_);

            // Jump component (Poisson process)
            double jump_component = 0.0;
            double jump_prob = params.jump_intensity * params.dt;
            if (uniform_dist_(rng_) < jump_prob) {
                jump_component = generate_jump_size(params.jump_mean, params.jump_std);
            }

            // Apply log-normal process with jumps
            double log_return = drift_component + diffusion_component + jump_component;
            paths[sim][step] = prev_price * std::exp(log_return);
        }
    }

    return paths;
}

std::vector<std::vector<double>> MCMCEngine::simulate_multi_asset(
    const std::vector<SimulationParams>& asset_params,
    const std::vector<std::vector<double>>& correlation_matrix) {

    size_t num_assets = asset_params.size();
    size_t num_steps = asset_params[0].num_steps;
    size_t num_sims = asset_params[0].num_simulations;

    std::vector<std::vector<double>> combined_paths(num_assets * num_sims);

    for (size_t sim = 0; sim < num_sims; ++sim) {
        for (size_t asset = 0; asset < num_assets; ++asset) {
            combined_paths[sim * num_assets + asset].resize(num_steps + 1);
            combined_paths[sim * num_assets + asset][0] = asset_params[asset].initial_price;
        }

        for (size_t step = 1; step <= num_steps; ++step) {
            std::vector<double> correlated_randoms = generate_correlated_normals(correlation_matrix[0], num_assets);

            for (size_t asset = 0; asset < num_assets; ++asset) {
                const auto& params = asset_params[asset];
                double prev_price = combined_paths[sim * num_assets + asset][step - 1];

                double drift_component = (params.drift - 0.5 * params.volatility * params.volatility) * params.dt;
                double diffusion_component = params.volatility * std::sqrt(params.dt) * correlated_randoms[asset];

                double jump_component = 0.0;
                double jump_prob = params.jump_intensity * params.dt;
                if (uniform_dist_(rng_) < jump_prob) {
                    jump_component = generate_jump_size(params.jump_mean, params.jump_std);
                }

                double log_return = drift_component + diffusion_component + jump_component;
                combined_paths[sim * num_assets + asset][step] = prev_price * std::exp(log_return);
            }
        }
    }

    return combined_paths;
}

std::vector<double> MCMCEngine::generate_correlated_normals(
    const std::vector<double>& correlations, size_t num_assets) {

    std::vector<double> independent_normals(num_assets);
    for (size_t i = 0; i < num_assets; ++i) {
        independent_normals[i] = normal_dist_(rng_);
    }

    // Simple Cholesky-like transformation for demo
    std::vector<double> correlated_normals(num_assets);
    correlated_normals[0] = independent_normals[0];

    for (size_t i = 1; i < num_assets; ++i) {
        double correlation = (i < correlations.size()) ? correlations[i] : 0.0;
        correlated_normals[i] = correlation * correlated_normals[0] +
                               std::sqrt(1.0 - correlation * correlation) * independent_normals[i];
    }

    return correlated_normals;
}

double MCMCEngine::generate_jump_time() {
    return exp_dist_(rng_);
}

double MCMCEngine::generate_jump_size(double mean, double std) {
    return mean + std * normal_dist_(rng_);
}