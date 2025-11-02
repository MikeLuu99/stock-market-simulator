#include "risk_analysis.h"
#include <algorithm>
#include <numeric>
#include <cmath>

double RiskAnalysis::calculate_var(const std::vector<double>& returns, double confidence_level) {
    if (returns.empty()) {
        return 0.0;
    }

    auto sorted_returns = sort_returns(returns);
    return percentile(sorted_returns, confidence_level);
}

double RiskAnalysis::calculate_cvar(const std::vector<double>& returns, double confidence_level) {
    if (returns.empty()) {
        return 0.0;
    }

    auto sorted_returns = sort_returns(returns);
    size_t var_index = static_cast<size_t>(confidence_level * sorted_returns.size());

    if (var_index == 0) {
        return sorted_returns[0];
    }

    double sum = 0.0;
    for (size_t i = 0; i < var_index; ++i) {
        sum += sorted_returns[i];
    }

    return sum / var_index;
}

double RiskAnalysis::calculate_sortino_ratio(const std::vector<double>& returns, double target_return) {
    if (returns.empty()) {
        return 0.0;
    }

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double downside_deviation_sq = 0.0;
    size_t downside_count = 0;

    for (double ret : returns) {
        if (ret < target_return) {
            double diff = ret - target_return;
            downside_deviation_sq += diff * diff;
            ++downside_count;
        }
    }

    if (downside_count == 0) {
        return 0.0;
    }

    double downside_deviation = std::sqrt(downside_deviation_sq / downside_count);
    double annualized_excess_return = (mean_return - target_return) * 252;
    double annualized_downside_deviation = downside_deviation * std::sqrt(252);

    return (annualized_downside_deviation > 0) ? annualized_excess_return / annualized_downside_deviation : 0.0;
}

std::vector<double> RiskAnalysis::calculate_correlation_matrix(
    const std::vector<std::vector<double>>& asset_returns) {

    size_t num_assets = asset_returns.size();
    std::vector<double> correlation_matrix(num_assets * num_assets, 0.0);

    if (num_assets == 0 || asset_returns[0].empty()) {
        return correlation_matrix;
    }

    // Calculate means
    std::vector<double> means(num_assets);
    for (size_t i = 0; i < num_assets; ++i) {
        means[i] = std::accumulate(asset_returns[i].begin(), asset_returns[i].end(), 0.0) / asset_returns[i].size();
    }

    // Calculate correlation matrix
    for (size_t i = 0; i < num_assets; ++i) {
        for (size_t j = 0; j < num_assets; ++j) {
            if (i == j) {
                correlation_matrix[i * num_assets + j] = 1.0;
            } else {
                double covariance = 0.0;
                double var_i = 0.0;
                double var_j = 0.0;

                size_t min_size = std::min(asset_returns[i].size(), asset_returns[j].size());

                for (size_t k = 0; k < min_size; ++k) {
                    double diff_i = asset_returns[i][k] - means[i];
                    double diff_j = asset_returns[j][k] - means[j];

                    covariance += diff_i * diff_j;
                    var_i += diff_i * diff_i;
                    var_j += diff_j * diff_j;
                }

                double std_i = std::sqrt(var_i / min_size);
                double std_j = std::sqrt(var_j / min_size);

                if (std_i > 0 && std_j > 0) {
                    correlation_matrix[i * num_assets + j] = (covariance / min_size) / (std_i * std_j);
                }
            }
        }
    }

    return correlation_matrix;
}

double RiskAnalysis::calculate_beta(
    const std::vector<double>& asset_returns,
    const std::vector<double>& market_returns) {

    if (asset_returns.empty() || market_returns.empty()) {
        return 0.0;
    }

    size_t min_size = std::min(asset_returns.size(), market_returns.size());

    double asset_mean = std::accumulate(asset_returns.begin(), asset_returns.begin() + min_size, 0.0) / min_size;
    double market_mean = std::accumulate(market_returns.begin(), market_returns.begin() + min_size, 0.0) / min_size;

    double covariance = 0.0;
    double market_variance = 0.0;

    for (size_t i = 0; i < min_size; ++i) {
        double asset_diff = asset_returns[i] - asset_mean;
        double market_diff = market_returns[i] - market_mean;

        covariance += asset_diff * market_diff;
        market_variance += market_diff * market_diff;
    }

    return (market_variance > 0) ? covariance / market_variance : 0.0;
}

std::vector<double> RiskAnalysis::monte_carlo_var(
    const std::vector<std::vector<double>>& simulated_paths,
    double confidence_level) {

    if (simulated_paths.empty() || simulated_paths[0].empty()) {
        return {};
    }

    size_t num_steps = simulated_paths[0].size();
    std::vector<double> var_estimates(num_steps);

    for (size_t step = 0; step < num_steps; ++step) {
        std::vector<double> step_values;
        step_values.reserve(simulated_paths.size());

        for (const auto& path : simulated_paths) {
            if (step < path.size()) {
                step_values.push_back(path[step]);
            }
        }

        if (!step_values.empty()) {
            std::sort(step_values.begin(), step_values.end());
            var_estimates[step] = percentile(step_values, confidence_level);
        }
    }

    return var_estimates;
}

std::vector<double> RiskAnalysis::sort_returns(const std::vector<double>& returns) {
    std::vector<double> sorted = returns;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

double RiskAnalysis::percentile(const std::vector<double>& sorted_data, double p) {
    if (sorted_data.empty()) {
        return 0.0;
    }

    if (p <= 0.0) {
        return sorted_data.front();
    }
    if (p >= 1.0) {
        return sorted_data.back();
    }

    double index = p * (sorted_data.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));

    if (lower == upper) {
        return sorted_data[lower];
    }

    double weight = index - lower;
    return sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight;
}