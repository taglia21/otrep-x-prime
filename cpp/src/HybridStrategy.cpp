/**
 * @file HybridStrategy.cpp
 * @brief Implementation of the low-latency hybrid trading strategy
 * 
 * OTREP-X PRIME C++ Port - Phase 1
 * 
 * Performance-critical implementation using:
 * - Eigen for SIMD-optimized vectorized math
 * - Pre-allocated buffers to avoid heap allocations
 * - Inline math utilities for hot path optimization
 * 
 * @author OTREP-X Development Team
 * @date December 2025
 */

#include "otrep/HybridStrategy.hpp"
#include <cmath>
#include <numeric>

namespace otrep {

// =============================================================================
// CONSTRUCTION
// =============================================================================

HybridStrategy::HybridStrategy()
    : HybridStrategy(StrategyParams{}) {}

HybridStrategy::HybridStrategy(const StrategyParams& params)
    : params_(params)
    , bars_(params.max_history_bars)
    , closes_(params.max_history_bars)
    , work_buffer_()
    , eigen_buffer_()
    , volatility_history_()
    , last_latency_ns_(0)
    , total_latency_ns_(0)
    , calc_count_(0)
{
    // Pre-allocate working buffers to avoid heap allocation in hot path
    const size_t max_lookback = static_cast<size_t>(
        std::max({params_.momentum_lookback, 
                  params_.mean_reversion_lookback,
                  params_.low_vol_lookback}) + 10
    );
    work_buffer_.reserve(max_lookback);
    eigen_buffer_.resize(max_lookback);
    volatility_history_.reserve(100);
}

// =============================================================================
// DATA INPUT
// =============================================================================

void HybridStrategy::on_bar(const Bar& bar) {
    bars_.push(bar);
    closes_.push(bar.close);
}

// =============================================================================
// SIGNAL CALCULATION - HOT PATH
// =============================================================================

SignalResult HybridStrategy::calculate_signal(double external_graph_signal) {
    // Start high-precision timer
    const auto start = std::chrono::high_resolution_clock::now();
    
    SignalResult result;
    result.graph_signal = external_graph_signal;
    
    // Need minimum bars for any calculation
    const size_t min_bars = static_cast<size_t>(
        std::max(params_.momentum_lookback, params_.mean_reversion_lookback)
    );
    
    if (closes_.size() < min_bars) {
        // Not enough data yet
        const auto end = std::chrono::high_resolution_clock::now();
        last_latency_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        result.latency_ns = last_latency_ns_;
        return result;
    }
    
    // Calculate individual signals
    int lookback_used = 0;
    result.momentum_signal = calculate_momentum_signal(lookback_used);
    result.lookback_used = lookback_used;
    result.reversion_signal = calculate_mean_reversion_signal();
    result.volatility = calculate_volatility();
    
    // Combine signals with weights (now including graph signal)
    const double total_weight = params_.momentum_weight 
                               + params_.mean_reversion_weight 
                               + params_.graph_weight;
    
    if (total_weight > 0.0) {
        // Note: Graph signal convention is inverted for trading:
        // graph_signal > 0 means expensive → SELL (negative trading signal)
        // graph_signal < 0 means cheap → BUY (positive trading signal)
        const double graph_contribution = -external_graph_signal * params_.graph_weight;
        
        result.signal = (
            (result.momentum_signal * params_.momentum_weight) +
            (result.reversion_signal * params_.mean_reversion_weight) +
            graph_contribution
        ) / total_weight;
    } else {
        result.signal = 0.0;
    }
    
    // Clip to [-1, 1]
    result.signal = clip(result.signal, -1.0, 1.0);
    
    // Record timing
    const auto end = std::chrono::high_resolution_clock::now();
    last_latency_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_latency_ns_ += last_latency_ns_;
    ++calc_count_;
    result.latency_ns = last_latency_ns_;
    
    return result;
}

// =============================================================================
// MOMENTUM SIGNAL
// =============================================================================

double HybridStrategy::calculate_momentum_signal(int& lookback_used) const {
    // Get adaptive lookback based on volatility regime
    const int lookback = get_adaptive_lookback();
    lookback_used = lookback;
    
    const size_t n = static_cast<size_t>(lookback);
    if (closes_.size() < n) {
        return 0.0;
    }
    
    // Extract closes to work buffer (chronological order: oldest to newest)
    closes_.extract_last_n(n, work_buffer_);
    
    // Map to Eigen for SIMD-optimized calculations
    Eigen::Map<const Eigen::ArrayXd> closes_arr(work_buffer_.data(), n);
    
    const double current = closes_arr(n - 1);  // Most recent
    const double average = closes_arr.mean();
    
    if (average == 0.0) {
        return 0.0;
    }
    
    // Momentum: normalized deviation from mean
    const double momentum = (current - average) / average;
    
    // Scale to approximately [-1, 1] range
    // Factor of 10 matches Python implementation
    const double raw_signal = momentum * 10.0;
    
    return clip(raw_signal, -1.0, 1.0);
}

// =============================================================================
// MEAN REVERSION SIGNAL (BOLLINGER BANDS)
// =============================================================================

double HybridStrategy::calculate_mean_reversion_signal() const {
    if (!params_.mean_reversion_enabled) {
        return 0.0;
    }
    
    const size_t n = static_cast<size_t>(params_.mean_reversion_lookback);
    if (closes_.size() < n) {
        return 0.0;
    }
    
    // Extract closes to work buffer
    closes_.extract_last_n(n, work_buffer_);
    
    // Map to Eigen for vectorized SMA and STD calculation
    Eigen::Map<const Eigen::ArrayXd> closes_arr(work_buffer_.data(), n);
    
    const double sma = closes_arr.mean();
    
    if (sma == 0.0) {
        return 0.0;
    }
    
    // Calculate standard deviation using Eigen
    // std = sqrt(mean((x - mean)^2))
    const double variance = (closes_arr - sma).square().mean();
    const double std_dev = std::sqrt(variance);
    
    if (std_dev == 0.0 || std::isnan(std_dev)) {
        return 0.0;
    }
    
    const double current_price = closes_arr(n - 1);
    
    // Mean reversion signal formula:
    // Signal_MR = -(Close - SMA) / (STD * BB_MULTIPLIER)
    //
    // When price > SMA: negative signal (expect reversion down -> SELL)
    // When price < SMA: positive signal (expect reversion up -> BUY)
    const double signal = -(current_price - sma) / 
                          (std_dev * params_.bb_std_dev_multiplier);
    
    return clip(signal, -1.0, 1.0);
}

// =============================================================================
// VOLATILITY CALCULATION
// =============================================================================

double HybridStrategy::calculate_volatility() const {
    constexpr size_t vol_window = 20;
    
    if (closes_.size() < vol_window + 1) {
        return 0.0;
    }
    
    // Extract last vol_window + 1 closes for log returns
    closes_.extract_last_n(vol_window + 1, work_buffer_);
    
    // Calculate log returns manually to avoid temporary allocations
    // log_return[i] = log(close[i+1] / close[i])
    double sum = 0.0;
    double sum_sq = 0.0;
    
    for (size_t i = 0; i < vol_window; ++i) {
        if (work_buffer_[i] <= 0.0) continue;
        const double log_ret = std::log(work_buffer_[i + 1] / work_buffer_[i]);
        sum += log_ret;
        sum_sq += log_ret * log_ret;
    }
    
    const double mean = sum / static_cast<double>(vol_window);
    const double variance = (sum_sq / static_cast<double>(vol_window)) - (mean * mean);
    
    return std::sqrt(std::max(0.0, variance));
}

// =============================================================================
// ADAPTIVE LOOKBACK
// =============================================================================

int HybridStrategy::get_adaptive_lookback() const {
    if (!params_.adaptive_enabled) {
        return params_.momentum_lookback;
    }
    
    // Need at least 50 bars for volatility regime detection
    if (closes_.size() < 50) {
        return params_.momentum_lookback;
    }
    
    // Current volatility (last 20 bars)
    constexpr size_t current_window = 20;
    constexpr size_t historical_start = 20;
    constexpr size_t historical_end = 50;
    constexpr size_t historical_window = historical_end - historical_start;
    
    // Extract data for current volatility
    closes_.extract_last_n(current_window + 1, work_buffer_);
    
    double current_sum = 0.0;
    double current_sum_sq = 0.0;
    for (size_t i = 0; i < current_window; ++i) {
        if (work_buffer_[i] <= 0.0) continue;
        const double log_ret = std::log(work_buffer_[i + 1] / work_buffer_[i]);
        current_sum += log_ret;
        current_sum_sq += log_ret * log_ret;
    }
    const double current_mean = current_sum / static_cast<double>(current_window);
    const double current_var = (current_sum_sq / static_cast<double>(current_window)) 
                               - (current_mean * current_mean);
    const double current_vol = std::sqrt(std::max(0.0, current_var));
    
    // Historical volatility (bars 20-50, i.e., older data)
    // Extract bars from position [size-50] to [size-20]
    if (closes_.size() < historical_end + 1) {
        return params_.momentum_lookback;
    }
    
    // We need to extract a different range for historical
    // Extract last 51 closes, then use indices 0-30 for historical
    closes_.extract_last_n(historical_end + 1, work_buffer_);
    
    double hist_sum = 0.0;
    double hist_sum_sq = 0.0;
    for (size_t i = 0; i < historical_window; ++i) {
        if (work_buffer_[i] <= 0.0) continue;
        const double log_ret = std::log(work_buffer_[i + 1] / work_buffer_[i]);
        hist_sum += log_ret;
        hist_sum_sq += log_ret * log_ret;
    }
    const double hist_mean = hist_sum / static_cast<double>(historical_window);
    const double hist_var = (hist_sum_sq / static_cast<double>(historical_window)) 
                            - (hist_mean * hist_mean);
    const double hist_vol = std::sqrt(std::max(0.0, hist_var));
    
    if (hist_vol <= 0.0) {
        return params_.momentum_lookback;
    }
    
    // Volatility regime detection
    const double vol_ratio = current_vol / hist_vol;
    
    if (vol_ratio >= params_.vol_multiplier) {
        // High volatility: use shorter lookback (more reactive)
        return params_.high_vol_lookback;
    } else if (vol_ratio <= 1.0 / params_.vol_multiplier) {
        // Low volatility: use longer lookback (more stable)
        return params_.low_vol_lookback;
    }
    
    return params_.momentum_lookback;
}

// =============================================================================
// TRADE ACTION
// =============================================================================

TradeAction HybridStrategy::get_trade_action(double signal) const noexcept {
    if (signal > params_.signal_threshold) {
        return TradeAction::BUY;
    } else if (signal < -params_.signal_threshold) {
        return TradeAction::SELL;
    } else if (std::abs(signal) < params_.neutral_threshold) {
        return TradeAction::CLOSE;
    }
    return TradeAction::HOLD;
}

// =============================================================================
// PROFILING
// =============================================================================

double HybridStrategy::avg_latency_ns() const noexcept {
    if (calc_count_ == 0) return 0.0;
    return static_cast<double>(total_latency_ns_) / static_cast<double>(calc_count_);
}

void HybridStrategy::reset_profiling() noexcept {
    last_latency_ns_ = 0;
    total_latency_ns_ = 0;
    calc_count_ = 0;
}

void HybridStrategy::clear() {
    bars_.clear();
    closes_.clear();
    volatility_history_.clear();
    reset_profiling();
}

} // namespace otrep
