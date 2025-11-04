#include "mcmc_engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <chrono>

MCMCEngine::MCMCEngine(unsigned int seed)
    : rng_(seed), normal_dist_(0.0, 1.0), exp_dist_(1.0), uniform_dist_(0.0, 1.0) {}

std::vector<std::vector<double>> MCMCEngine::simulate_gbm_with_jumps(const SimulationParams& params) {
    std::cout << "[C++ Engine] simulate_gbm_with_jumps starting..." << std::endl;
    std::cout << "[C++ Engine] Parameters: num_simulations=" << params.num_simulations
              << ", num_steps=" << params.num_steps << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> paths(params.num_simulations, std::vector<double>(params.num_steps + 1));

    int checkpoint_interval = std::max(1, params.num_simulations / 10); // Log every 10%

    for (int sim = 0; sim < params.num_simulations; ++sim) {
        if (sim % checkpoint_interval == 0 && sim > 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();
            double progress = 100.0 * sim / params.num_simulations;
            std::cout << "[C++ Engine] Progress: " << progress << "% (" << sim << "/"
                     << params.num_simulations << " simulations, " << elapsed_sec << "s elapsed)" << std::endl;
        }

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

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[C++ Engine] simulate_gbm_with_jumps completed in " << total_elapsed << "s" << std::endl;

    return paths;
}

std::vector<std::vector<double>> MCMCEngine::simulate_multi_asset(
    const std::vector<SimulationParams>& asset_params,
    const std::vector<std::vector<double>>& correlation_matrix) {

    size_t num_assets = asset_params.size();
    size_t num_steps = asset_params[0].num_steps;
    size_t num_sims = asset_params[0].num_simulations;

    std::cout << "[C++ Engine] simulate_multi_asset starting..." << std::endl;
    std::cout << "[C++ Engine] Parameters: num_assets=" << num_assets
              << ", num_simulations=" << num_sims << ", num_steps=" << num_steps << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<double>> combined_paths(num_assets * num_sims);

    int checkpoint_interval = std::max(1, static_cast<int>(num_sims / 10)); // Log every 10%

    for (size_t sim = 0; sim < num_sims; ++sim) {
        if (sim % checkpoint_interval == 0 && sim > 0) {
            auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();
            double progress = 100.0 * sim / num_sims;
            std::cout << "[C++ Engine] Progress: " << progress << "% (" << sim << "/"
                     << num_sims << " simulations, " << elapsed_sec << "s elapsed)" << std::endl;
        }

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

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[C++ Engine] simulate_multi_asset completed in " << total_elapsed << "s" << std::endl;

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