#pragma once

#include <vector>
#include <string>
#include <map>

class MarketData {
public:
    struct HistoricalData {
        std::vector<double> prices;
        std::vector<double> volumes;
        std::vector<std::string> dates;
    };

    struct MarketParams {
        double drift;
        double volatility;
        double jump_intensity;
        double current_price;
    };

    static MarketParams estimate_parameters(const std::vector<double>& prices);

    static std::vector<double> calculate_log_returns(const std::vector<double>& prices);

    static double estimate_volatility(const std::vector<double>& log_returns);

    static double estimate_drift(const std::vector<double>& log_returns);

    static std::vector<std::vector<double>> generate_correlation_matrix(
        const std::vector<std::vector<double>>& multi_asset_returns
    );

    static std::vector<double> detect_jumps(
        const std::vector<double>& log_returns,
        double threshold = 3.0
    );

private:
    static double calculate_mean(const std::vector<double>& data);
    static double calculate_variance(const std::vector<double>& data, double mean);
};