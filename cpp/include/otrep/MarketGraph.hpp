/**
 * @file MarketGraph.hpp
 * @brief Graph Signal Processing for Market Topology Alpha
 * 
 * OTREP-X PRIME - Phase III Graph Alpha
 * 
 * Implements Laplacian Diffusion on market correlation networks to detect
 * relative mispricing between connected assets. When a stock's return
 * deviates from what its neighbors suggest (the "expected" return from
 * diffusion), this creates a tradeable signal.
 * 
 * Mathematical Framework:
 * 1. Build correlation matrix from returns
 * 2. Threshold to create adjacency matrix W
 * 3. Compute Laplacian L = D - W
 * 4. Solve diffusion: (I + αL)h = x  for expected return h
 * 5. Signal s = x - h (residual indicates mispricing)
 * 
 * @author OTREP-X Development Team
 * @date December 2025
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace otrep {

/**
 * @brief Graph signal processing parameters
 */
struct GraphParams {
    double correlation_threshold = 0.5;  // Hard threshold for adjacency
    double diffusion_alpha = 1.0;        // Laplacian diffusion strength
    int lookback_bars = 50;              // Bars for correlation calc
};

/**
 * @brief Result of graph signal calculation
 */
struct GraphSignalResult {
    Eigen::VectorXd signals;             // Per-asset signals
    Eigen::MatrixXd correlation_matrix;  // N x N correlation matrix
    Eigen::MatrixXd adjacency_matrix;    // Thresholded adjacency W
    int num_edges;                       // Number of edges in graph
    double avg_correlation;              // Average correlation
    int64_t latency_ns;                  // Calculation time
    
    GraphSignalResult() : num_edges(0), avg_correlation(0.0), latency_ns(0) {}
};

/**
 * @brief Market correlation graph for Laplacian diffusion signals
 * 
 * Analyzes the market as a connected network where edges represent
 * correlation between assets. Detects mispricing via diffusion residuals.
 * 
 * Performance: O(N² * T) for correlation, O(N³) for linear solve
 * Typical latency: ~0.5-2ms for 40 assets
 */
class MarketGraph {
public:
    /**
     * @brief Construct with default parameters
     */
    MarketGraph();
    
    /**
     * @brief Construct with custom parameters
     */
    explicit MarketGraph(const GraphParams& params);
    
    /**
     * @brief Calculate graph-based signals from returns matrix
     * 
     * @param returns Matrix of shape (T x N) where:
     *                T = number of time periods (rows)
     *                N = number of assets (columns)
     *                Values are log returns
     * @return GraphSignalResult with per-asset signals
     * 
     * The signal for asset i indicates:
     *   - s[i] > 0: Asset returned MORE than neighbors expect → expensive → SELL
     *   - s[i] < 0: Asset returned LESS than neighbors expect → cheap → BUY
     */
    [[nodiscard]] GraphSignalResult calculate_signals(const Eigen::MatrixXd& returns);
    
    /**
     * @brief Calculate signals from returns with symbol mapping (zero-copy from NumPy)
     * 
     * @param returns Matrix of shape (T x N) in Row-Major storage (NumPy default)
     * @param symbols Vector of N symbol names for output mapping
     * @return Map of symbol -> signal value
     * 
     * Uses Eigen::Ref with RowMajor storage for zero-copy mapping from NumPy arrays.
     */
    using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    [[nodiscard]] std::unordered_map<std::string, double> calculate_signals_map(
        const Eigen::Ref<const RowMajorMatrixXd>& returns,
        const std::vector<std::string>& symbols
    );
    
    /**
     * @brief Calculate signals (column-major overload for internal C++ use)
     */
    [[nodiscard]] std::unordered_map<std::string, double> calculate_signals_map(
        const Eigen::MatrixXd& returns,
        const std::vector<std::string>& symbols
    );
    
    /**
     * @brief Get last calculation latency in nanoseconds
     */
    [[nodiscard]] int64_t last_latency_ns() const noexcept { return last_latency_ns_; }
    
    /**
     * @brief Get current parameters
     */
    [[nodiscard]] const GraphParams& params() const noexcept { return params_; }
    
    /**
     * @brief Update parameters
     */
    void set_params(const GraphParams& params) { params_ = params; }

private:
    GraphParams params_;
    int64_t last_latency_ns_;
    
    /**
     * @brief Compute Pearson correlation matrix from returns
     * 
     * @param returns (T x N) matrix
     * @return (N x N) correlation matrix
     */
    [[nodiscard]] Eigen::MatrixXd compute_correlation_matrix(
        const Eigen::MatrixXd& returns) const;
    
    /**
     * @brief Apply hard threshold to create adjacency matrix
     * 
     * @param correlation (N x N) correlation matrix
     * @return (N x N) adjacency matrix W with zeros on diagonal
     */
    [[nodiscard]] Eigen::MatrixXd compute_adjacency_matrix(
        const Eigen::MatrixXd& correlation) const;
    
    /**
     * @brief Compute graph Laplacian L = D - W
     * 
     * @param adjacency (N x N) adjacency matrix W
     * @return (N x N) Laplacian matrix L
     */
    [[nodiscard]] Eigen::MatrixXd compute_laplacian(
        const Eigen::MatrixXd& adjacency) const;
    
    /**
     * @brief Solve Laplacian diffusion (I + αL)h = x
     * 
     * @param laplacian (N x N) Laplacian matrix
     * @param current_returns (N,) vector of most recent returns
     * @return (N,) vector of expected returns
     */
    [[nodiscard]] Eigen::VectorXd solve_diffusion(
        const Eigen::MatrixXd& laplacian,
        const Eigen::VectorXd& current_returns) const;
};

} // namespace otrep
