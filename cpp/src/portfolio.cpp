#include "portfolio.h"
#include <cmath>
#include <algorithm>
#include <numeric>

Portfolio::Portfolio(const std::vector<Asset>& assets) : assets_(assets) {
    normalize_weights();
}

void Portfolio::update_prices(const std::vector<std::vector<double>>& price_paths) {
    if (price_paths.size() != assets_.size()) {
        return;
    }

    for (size_t i = 0; i < assets_.size(); ++i) {
        assets_[i].price_path = price_paths[i];
    }

    portfolio_values_ = calculate_portfolio_values();
}

std::vector<double> Portfolio::calculate_portfolio_values() const {
    if (assets_.empty() || assets_[0].price_path.empty()) {
        return {};
    }

    size_t num_steps = assets_[0].price_path.size();
    std::vector<double> values(num_steps, 0.0);

    for (size_t step = 0; step < num_steps; ++step) {
        for (const auto& asset : assets_) {
            if (step < asset.price_path.size()) {
                double position_value = asset.weight * (asset.price_path[step] / asset.initial_price);
                values[step] += position_value;
            }
        }
    }

    return values;
}

std::vector<double> Portfolio::calculate_returns() const {
    if (portfolio_values_.size() < 2) {
        return {};
    }

    std::vector<double> returns(portfolio_values_.size() - 1);
    for (size_t i = 1; i < portfolio_values_.size(); ++i) {
        returns[i - 1] = (portfolio_values_[i] - portfolio_values_[i - 1]) / portfolio_values_[i - 1];
    }

    return returns;
}

double Portfolio::calculate_sharpe_ratio(double risk_free_rate) const {
    auto returns = calculate_returns();
    if (returns.empty()) {
        return 0.0;
    }

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();

    double volatility = std::sqrt(variance);
    double annualized_excess_return = (mean_return * 252) - risk_free_rate;
    double annualized_volatility = volatility * std::sqrt(252);

    return (annualized_volatility > 0) ? annualized_excess_return / annualized_volatility : 0.0;
}

double Portfolio::calculate_max_drawdown() const {
    if (portfolio_values_.empty()) {
        return 0.0;
    }

    double max_value = portfolio_values_[0];
    double max_drawdown = 0.0;

    for (double value : portfolio_values_) {
        if (value > max_value) {
            max_value = value;
        }

        double drawdown = (max_value - value) / max_value;
        max_drawdown = std::max(max_drawdown, drawdown);
    }

    return max_drawdown;
}

double Portfolio::calculate_volatility() const {
    auto returns = calculate_returns();
    if (returns.empty()) {
        return 0.0;
    }

    double mean_return = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();

    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - mean_return) * (ret - mean_return);
    }
    variance /= returns.size();

    return std::sqrt(variance * 252); // Annualized
}

double Portfolio::get_total_weight() const {
    double total = 0.0;
    for (const auto& asset : assets_) {
        total += asset.weight;
    }
    return total;
}

void Portfolio::normalize_weights() {
    double total_weight = get_total_weight();
    if (total_weight > 0.0) {
        for (auto& asset : assets_) {
            asset.weight /= total_weight;
        }
    }
}