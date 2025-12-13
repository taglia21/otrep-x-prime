/**
 * @file MarketGraph.cpp
 * @brief Implementation of Graph Signal Processing for Market Topology Alpha
 * 
 * OTREP-X PRIME - Phase III Graph Alpha
 * 
 * @author OTREP-X Development Team
 * @date December 2025
 */

#include "otrep/MarketGraph.hpp"
#include <cmath>
#include <numeric>

namespace otrep {

// =============================================================================
// CONSTRUCTION
// =============================================================================

MarketGraph::MarketGraph()
    : MarketGraph(GraphParams{}) {}

MarketGraph::MarketGraph(const GraphParams& params)
    : params_(params)
    , last_latency_ns_(0)
{}

// =============================================================================
// MAIN SIGNAL CALCULATION
// =============================================================================

GraphSignalResult MarketGraph::calculate_signals(const Eigen::MatrixXd& returns) {
    const auto start = std::chrono::high_resolution_clock::now();
    
    GraphSignalResult result;
    
    const int T = returns.rows();  // Time periods
    const int N = returns.cols();  // Assets
    
    if (T < 2 || N < 2) {
        // Not enough data for correlation
        result.signals = Eigen::VectorXd::Zero(N);
        const auto end = std::chrono::high_resolution_clock::now();
        result.latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        last_latency_ns_ = result.latency_ns;
        return result;
    }
    
    // Step 1: Compute correlation matrix
    result.correlation_matrix = compute_correlation_matrix(returns);
    
    // Step 2: Threshold to create adjacency matrix
    result.adjacency_matrix = compute_adjacency_matrix(result.correlation_matrix);
    
    // Count edges and compute average correlation
    result.num_edges = 0;
    double total_corr = 0.0;
    int corr_count = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (result.adjacency_matrix(i, j) > 0) {
                ++result.num_edges;
            }
            total_corr += std::abs(result.correlation_matrix(i, j));
            ++corr_count;
        }
    }
    result.avg_correlation = corr_count > 0 ? total_corr / corr_count : 0.0;
    
    // Step 3: Compute Laplacian L = D - W
    Eigen::MatrixXd laplacian = compute_laplacian(result.adjacency_matrix);
    
    // Step 4: Get most recent return vector x (last row)
    Eigen::VectorXd current_returns = returns.row(T - 1).transpose();
    
    // Step 5: Solve diffusion (I + αL)h = x for expected return h
    Eigen::VectorXd expected_returns = solve_diffusion(laplacian, current_returns);
    
    // Step 6: Signal s = x - h (residual indicates mispricing)
    // If actual > expected → stock is expensive relative to network → SELL (positive signal)
    // If actual < expected → stock is cheap relative to network → BUY (negative signal)
    result.signals = current_returns - expected_returns;
    
    // Normalize signals to [-1, 1] range
    double max_abs = result.signals.cwiseAbs().maxCoeff();
    if (max_abs > 0) {
        result.signals /= max_abs;
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    result.latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    last_latency_ns_ = result.latency_ns;
    
    return result;
}

// Row-Major overload for zero-copy NumPy integration
std::unordered_map<std::string, double> MarketGraph::calculate_signals_map(
    const Eigen::Ref<const RowMajorMatrixXd>& returns,
    const std::vector<std::string>& symbols
) {
    // Convert to column-major for internal processing (Eigen optimized for ColMajor)
    Eigen::MatrixXd returns_col = returns;
    GraphSignalResult result = calculate_signals(returns_col);
    
    std::unordered_map<std::string, double> signal_map;
    
    const int N = std::min(static_cast<int>(symbols.size()), 
                          static_cast<int>(result.signals.size()));
    
    for (int i = 0; i < N; ++i) {
        signal_map[symbols[i]] = result.signals(i);
    }
    
    return signal_map;
}

// Column-Major overload for internal C++ use
std::unordered_map<std::string, double> MarketGraph::calculate_signals_map(
    const Eigen::MatrixXd& returns,
    const std::vector<std::string>& symbols
) {
    GraphSignalResult result = calculate_signals(returns);
    
    std::unordered_map<std::string, double> signal_map;
    
    const int N = std::min(static_cast<int>(symbols.size()), 
                          static_cast<int>(result.signals.size()));
    
    for (int i = 0; i < N; ++i) {
        signal_map[symbols[i]] = result.signals(i);
    }
    
    return signal_map;
}

// =============================================================================
// CORRELATION MATRIX
// =============================================================================

Eigen::MatrixXd MarketGraph::compute_correlation_matrix(
    const Eigen::MatrixXd& returns) const 
{
    const int T = returns.rows();
    const int N = returns.cols();
    
    // Center the returns (subtract mean)
    Eigen::RowVectorXd means = returns.colwise().mean();
    Eigen::MatrixXd centered = returns.rowwise() - means;
    
    // Compute covariance matrix: C = X'X / (T-1)
    Eigen::MatrixXd cov = (centered.transpose() * centered) / (T - 1);
    
    // Compute standard deviations
    Eigen::VectorXd stds = cov.diagonal().cwiseSqrt();
    
    // Compute correlation: corr(i,j) = cov(i,j) / (std[i] * std[j])
    Eigen::MatrixXd correlation = Eigen::MatrixXd::Zero(N, N);
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double denom = stds(i) * stds(j);
            if (denom > 1e-10) {
                correlation(i, j) = cov(i, j) / denom;
            } else {
                correlation(i, j) = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    
    // Ensure diagonal is exactly 1
    for (int i = 0; i < N; ++i) {
        correlation(i, i) = 1.0;
    }
    
    return correlation;
}

// =============================================================================
// ADJACENCY MATRIX (HARD THRESHOLD WITH NONNEGATIVE WEIGHTS)
// =============================================================================

Eigen::MatrixXd MarketGraph::compute_adjacency_matrix(
    const Eigen::MatrixXd& correlation) const 
{
    const int N = correlation.rows();
    Eigen::MatrixXd adjacency = Eigen::MatrixXd::Zero(N, N);
    
    const double threshold = params_.correlation_threshold;
    
    // CTO Directive (Dec 2025): Use nonnegative adjacency weights
    // A_ij = max(abs(rho_ij), 0) if |rho_ij| >= threshold
    // 
    // Rationale: 
    // 1. Ensures Laplacian L = D - A is positive semi-definite
    // 2. Guarantees LDLT decomposition numerical stability
    // 3. Threshold increased to 0.5 for sparser, more stable graph
    
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                double abs_corr = std::abs(correlation(i, j));
                if (abs_corr >= threshold) {
                    // Use absolute value for nonnegative weights
                    adjacency(i, j) = abs_corr;
                }
            }
        }
    }
    
    return adjacency;
}

// =============================================================================
// LAPLACIAN MATRIX
// =============================================================================

Eigen::MatrixXd MarketGraph::compute_laplacian(
    const Eigen::MatrixXd& adjacency) const 
{
    const int N = adjacency.rows();
    
    // Degree matrix D: D[i,i] = sum of row i in adjacency
    Eigen::VectorXd degrees = adjacency.rowwise().sum();
    
    // Laplacian L = D - W
    Eigen::MatrixXd laplacian = -adjacency;
    for (int i = 0; i < N; ++i) {
        laplacian(i, i) = degrees(i);
    }
    
    return laplacian;
}

// =============================================================================
// LAPLACIAN DIFFUSION SOLVE
// =============================================================================

Eigen::VectorXd MarketGraph::solve_diffusion(
    const Eigen::MatrixXd& laplacian,
    const Eigen::VectorXd& current_returns) const 
{
    const int N = laplacian.rows();
    
    // Build system matrix: A = I + α * L
    Eigen::MatrixXd A = Eigen::MatrixXd::Identity(N, N) 
                       + params_.diffusion_alpha * laplacian;
    
    // Solve A * h = x for h (expected returns)
    // Using LDLT decomposition (robust for positive semi-definite matrices)
    Eigen::LDLT<Eigen::MatrixXd> solver(A);
    
    if (solver.info() != Eigen::Success) {
        // Fallback: return current returns if solve fails
        return current_returns;
    }
    
    Eigen::VectorXd expected = solver.solve(current_returns);
    
    if (solver.info() != Eigen::Success) {
        return current_returns;
    }
    
    return expected;
}

} // namespace otrep
