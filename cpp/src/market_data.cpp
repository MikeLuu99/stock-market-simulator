#include "market_data.h"
#include <cmath>
#include <algorithm>
#include <numeric>

MarketData::MarketParams MarketData::estimate_parameters(const std::vector<double>& prices) {
    MarketParams params{};

    if (prices.size() < 2) {
        return params;
    }

    auto log_returns = calculate_log_returns(prices);

    params.current_price = prices.back();
    params.drift = estimate_drift(log_returns);
    params.volatility = estimate_volatility(log_returns);

    auto jumps = detect_jumps(log_returns);
    params.jump_intensity = static_cast<double>(jumps.size()) / log_returns.size();

    return params;
}

std::vector<double> MarketData::calculate_log_returns(const std::vector<double>& prices) {
    if (prices.size() < 2) {
        return {};
    }

    std::vector<double> log_returns(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i - 1] > 0 && prices[i] > 0) {
            log_returns[i - 1] = std::log(prices[i] / prices[i - 1]);
        }
    }

    return log_returns;
}

double MarketData::estimate_volatility(const std::vector<double>& log_returns) {
    if (log_returns.empty()) {
        return 0.0;
    }

    double mean = calculate_mean(log_returns);
    double variance = calculate_variance(log_returns, mean);

    return std::sqrt(variance * 252); // Annualized
}

double MarketData::estimate_drift(const std::vector<double>& log_returns) {
    if (log_returns.empty()) {
        return 0.0;
    }

    double mean_return = calculate_mean(log_returns);
    return mean_return * 252; // Annualized
}

std::vector<std::vector<double>> MarketData::generate_correlation_matrix(
    const std::vector<std::vector<double>>& multi_asset_returns) {

    size_t num_assets = multi_asset_returns.size();
    std::vector<std::vector<double>> correlation_matrix(num_assets, std::vector<double>(num_assets, 0.0));

    if (num_assets == 0) {
        return correlation_matrix;
    }

    // Calculate means
    std::vector<double> means(num_assets);
    for (size_t i = 0; i < num_assets; ++i) {
        means[i] = calculate_mean(multi_asset_returns[i]);
    }

    // Calculate correlation matrix
    for (size_t i = 0; i < num_assets; ++i) {
        for (size_t j = 0; j < num_assets; ++j) {
            if (i == j) {
                correlation_matrix[i][j] = 1.0;
            } else {
                const auto& returns_i = multi_asset_returns[i];
                const auto& returns_j = multi_asset_returns[j];

                size_t min_size = std::min(returns_i.size(), returns_j.size());

                if (min_size == 0) {
                    continue;
                }

                double covariance = 0.0;
                double var_i = 0.0;
                double var_j = 0.0;

                for (size_t k = 0; k < min_size; ++k) {
                    double diff_i = returns_i[k] - means[i];
                    double diff_j = returns_j[k] - means[j];

                    covariance += diff_i * diff_j;
                    var_i += diff_i * diff_i;
                    var_j += diff_j * diff_j;
                }

                double std_i = std::sqrt(var_i / min_size);
                double std_j = std::sqrt(var_j / min_size);

                if (std_i > 0 && std_j > 0) {
                    correlation_matrix[i][j] = (covariance / min_size) / (std_i * std_j);
                }
            }
        }
    }

    return correlation_matrix;
}

std::vector<double> MarketData::detect_jumps(const std::vector<double>& log_returns, double threshold) {
    if (log_returns.empty()) {
        return {};
    }

    double mean = calculate_mean(log_returns);
    double std_dev = std::sqrt(calculate_variance(log_returns, mean));

    std::vector<double> jumps;
    for (double ret : log_returns) {
        double z_score = std::abs((ret - mean) / std_dev);
        if (z_score > threshold) {
            jumps.push_back(ret);
        }
    }

    return jumps;
}

double MarketData::calculate_mean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }

    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

double MarketData::calculate_variance(const std::vector<double>& data, double mean) {
    if (data.empty()) {
        return 0.0;
    }

    double variance = 0.0;
    for (double value : data) {
        double diff = value - mean;
        variance += diff * diff;
    }

    return variance / data.size();
}