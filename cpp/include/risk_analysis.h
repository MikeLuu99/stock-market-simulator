#pragma once

#include <vector>

class RiskAnalysis {
public:
    static double calculate_var(const std::vector<double>& returns, double confidence_level = 0.05);

    static double calculate_cvar(const std::vector<double>& returns, double confidence_level = 0.05);

    static double calculate_sortino_ratio(const std::vector<double>& returns, double target_return = 0.0);

    static std::vector<double> calculate_correlation_matrix(
        const std::vector<std::vector<double>>& asset_returns
    );

    static double calculate_beta(
        const std::vector<double>& asset_returns,
        const std::vector<double>& market_returns
    );

    static std::vector<double> monte_carlo_var(
        const std::vector<std::vector<double>>& simulated_paths,
        double confidence_level = 0.05
    );

private:
    static std::vector<double> sort_returns(const std::vector<double>& returns);
    static double percentile(const std::vector<double>& sorted_data, double p);
};