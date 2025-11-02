#pragma once

#include <vector>
#include <string>
#include <map>

class Portfolio {
public:
    struct Asset {
        std::string symbol;
        double weight;
        double initial_price;
        std::vector<double> price_path;
    };

    Portfolio(const std::vector<Asset>& assets);

    void update_prices(const std::vector<std::vector<double>>& price_paths);

    std::vector<double> calculate_portfolio_values() const;
    std::vector<double> calculate_returns() const;
    double calculate_sharpe_ratio(double risk_free_rate = 0.02) const;
    double calculate_max_drawdown() const;
    double calculate_volatility() const;

    const std::vector<Asset>& get_assets() const { return assets_; }
    double get_total_weight() const;

private:
    std::vector<Asset> assets_;
    std::vector<double> portfolio_values_;

    void normalize_weights();
};