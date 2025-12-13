/**
 * @file HybridStrategy.hpp
 * @brief Low-latency hybrid trading strategy combining Momentum and Mean Reversion
 * 
 * OTREP-X PRIME C++ Port - Phase 1
 * 
 * This implementation is optimized for:
 * - Zero heap allocations in the hot path
 * - SIMD-optimized math via Eigen
 * - Cache-friendly data access patterns
 * 
 * Optimized Parameters (from Phase IV validation):
 * - momentum_lookback: 20
 * - mean_reversion_lookback: 15
 * - bb_std_dev_multiplier: 1.5
 * 
 * @author OTREP-X Development Team
 * @date December 2025
 */

#pragma once

#include <Eigen/Dense>
#include <chrono>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace otrep {

// =============================================================================
// DATA STRUCTURES
// =============================================================================

/**
 * @brief OHLCV Bar representation
 */
struct Bar {
    std::chrono::system_clock::time_point timestamp;
    double open;
    double high;
    double low;
    double close;
    int64_t volume;
    
    Bar() : open(0), high(0), low(0), close(0), volume(0) {}
    
    Bar(double o, double h, double l, double c, int64_t v)
        : open(o), high(h), low(l), close(c), volume(v) {}
    
    Bar(std::chrono::system_clock::time_point ts, 
        double o, double h, double l, double c, int64_t v)
        : timestamp(ts), open(o), high(h), low(l), close(c), volume(v) {}
};

/**
 * @brief Fixed-capacity circular buffer with O(1) push and access
 * 
 * Optimized for rolling window calculations:
 * - Contiguous memory for cache efficiency
 * - No heap allocations after construction
 * - Supports Eigen::Map for zero-copy vectorization
 */
template<typename T>
class CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : buffer_(capacity)
        , head_(0)
        , size_(0)
        , capacity_(capacity) {}
    
    /**
     * @brief Push a new value, overwriting oldest if full
     */
    void push(const T& value) noexcept {
        buffer_[head_] = value;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) ++size_;
    }
    
    /**
     * @brief Access element by index (0 = newest, size-1 = oldest)
     */
    [[nodiscard]] const T& operator[](size_t idx) const {
        if (idx >= size_) throw std::out_of_range("CircularBuffer: index out of bounds");
        size_t actual = (head_ - 1 - idx + capacity_) % capacity_;
        return buffer_[actual];
    }
    
    /**
     * @brief Access element by index from oldest (0) to newest (size-1)
     * More intuitive for time-series analysis
     */
    [[nodiscard]] const T& at_chronological(size_t idx) const {
        if (idx >= size_) throw std::out_of_range("CircularBuffer: index out of bounds");
        size_t oldest_idx = (head_ - size_ + capacity_) % capacity_;
        size_t actual = (oldest_idx + idx) % capacity_;
        return buffer_[actual];
    }
    
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool full() const noexcept { return size_ == capacity_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    
    /**
     * @brief Get pointer to underlying data for Eigen::Map
     * WARNING: Data is circular, use extract_to_vector for contiguous access
     */
    [[nodiscard]] const T* data() const noexcept { return buffer_.data(); }
    [[nodiscard]] T* data() noexcept { return buffer_.data(); }
    
    /**
     * @brief Extract last N elements to a contiguous vector (chronological order)
     * Pre-allocated destination avoids heap allocation in hot path
     */
    void extract_last_n(size_t n, std::vector<T>& dest) const {
        n = std::min(n, size_);
        dest.resize(n);
        for (size_t i = 0; i < n; ++i) {
            dest[i] = at_chronological(size_ - n + i);
        }
    }
    
    /**
     * @brief Get the most recent (newest) element
     */
    [[nodiscard]] const T& front() const {
        if (size_ == 0) throw std::out_of_range("CircularBuffer: empty");
        return (*this)[0];
    }
    
    /**
     * @brief Get the oldest element
     */
    [[nodiscard]] const T& back() const {
        if (size_ == 0) throw std::out_of_range("CircularBuffer: empty");
        return (*this)[size_ - 1];
    }
    
    void clear() noexcept {
        head_ = 0;
        size_ = 0;
    }

private:
    std::vector<T> buffer_;
    size_t head_;
    size_t size_;
    size_t capacity_;
};

// =============================================================================
// STRATEGY PARAMETERS
// =============================================================================

/**
 * @brief Configuration parameters for HybridStrategy
 * 
 * Default values are the optimized parameters from Phase IV.
 */
struct StrategyParams {
    // Momentum parameters
    int momentum_lookback = 20;        // Optimized value
    int trend_lookback = 15;
    double signal_threshold = 0.15;
    double neutral_threshold = 0.05;
    double momentum_weight = 0.4;      // Reduced for graph alpha
    double trend_weight = 0.0;         // Deprecated, set to 0
    
    // Adaptive volatility parameters
    bool adaptive_enabled = true;
    int high_vol_lookback = 10;
    int low_vol_lookback = 30;
    double vol_multiplier = 1.5;
    
    // Mean Reversion parameters (Bollinger Bands)
    bool mean_reversion_enabled = true;
    double mean_reversion_weight = 0.4; // Reduced for graph alpha
    int mean_reversion_lookback = 15;  // Optimized value
    double bb_std_dev_multiplier = 1.5; // Optimized value
    double reversion_threshold = 0.01;
    
    // Graph Alpha parameters (Laplacian Diffusion)
    double graph_weight = 0.2;         // Weight for graph signal
    
    // Buffer sizing
    size_t max_history_bars = 200;     // Enough for all lookback calculations
};

// =============================================================================
// SIGNAL RESULT
// =============================================================================

/**
 * @brief Result of signal calculation with profiling info
 */
struct SignalResult {
    double signal;              // Combined signal in [-1, 1]
    double momentum_signal;     // Individual momentum component
    double reversion_signal;    // Individual mean reversion component
    double graph_signal;        // External graph alpha signal
    int lookback_used;          // Actual lookback period used
    double volatility;          // Current volatility estimate
    int64_t latency_ns;         // Calculation time in nanoseconds
    
    SignalResult()
        : signal(0.0)
        , momentum_signal(0.0)
        , reversion_signal(0.0)
        , graph_signal(0.0)
        , lookback_used(0)
        , volatility(0.0)
        , latency_ns(0) {}
};

/**
 * @brief Trade action derived from signal
 */
enum class TradeAction {
    HOLD,
    BUY,
    SELL,
    CLOSE
};

// =============================================================================
// HYBRID STRATEGY
// =============================================================================

/**
 * @brief High-performance hybrid trading strategy
 * 
 * Combines momentum and mean reversion signals using optimized
 * Eigen-based vectorized calculations.
 * 
 * Performance targets:
 * - Signal calculation: < 0.01 ms/bar
 * - Zero heap allocations in calculate_signal()
 */
class HybridStrategy {
public:
    /**
     * @brief Construct with default optimized parameters
     */
    HybridStrategy();
    
    /**
     * @brief Construct with custom parameters
     */
    explicit HybridStrategy(const StrategyParams& params);
    
    /**
     * @brief Add a new bar to the price history
     * Call this for each incoming market data update
     */
    void on_bar(const Bar& bar);
    
    /**
     * @brief Calculate the combined trading signal
     * 
     * This is the HOT PATH - optimized for minimal latency.
     * 
     * @param external_graph_signal Optional external graph alpha signal
     *        from MarketGraph Laplacian diffusion (default: 0.0)
     * @return SignalResult containing signal value and profiling info
     */
    [[nodiscard]] SignalResult calculate_signal(double external_graph_signal = 0.0);
    
    /**
     * @brief Convert signal to trade action
     */
    [[nodiscard]] TradeAction get_trade_action(double signal) const noexcept;
    
    /**
     * @brief Get last calculation latency in nanoseconds
     */
    [[nodiscard]] int64_t last_latency_ns() const noexcept { return last_latency_ns_; }
    
    /**
     * @brief Get last calculation latency in milliseconds
     */
    [[nodiscard]] double last_latency_ms() const noexcept { 
        return static_cast<double>(last_latency_ns_) / 1e6; 
    }
    
    /**
     * @brief Get average latency over all calculations
     */
    [[nodiscard]] double avg_latency_ns() const noexcept;
    
    /**
     * @brief Get current parameters
     */
    [[nodiscard]] const StrategyParams& params() const noexcept { return params_; }
    
    /**
     * @brief Get number of bars in history
     */
    [[nodiscard]] size_t bar_count() const noexcept { return closes_.size(); }
    
    /**
     * @brief Reset profiling statistics
     */
    void reset_profiling() noexcept;
    
    /**
     * @brief Clear all price history
     */
    void clear();

private:
    // Configuration
    StrategyParams params_;
    
    // Price history (circular buffers for O(1) operations)
    CircularBuffer<Bar> bars_;
    CircularBuffer<double> closes_;
    
    // Pre-allocated working buffers (avoid heap allocation in hot path)
    mutable std::vector<double> work_buffer_;
    mutable Eigen::ArrayXd eigen_buffer_;
    
    // Volatility tracking
    std::vector<double> volatility_history_;
    
    // Profiling
    mutable int64_t last_latency_ns_;
    mutable int64_t total_latency_ns_;
    mutable size_t calc_count_;
    
    // Internal calculation methods
    [[nodiscard]] double calculate_momentum_signal(int& lookback_used) const;
    [[nodiscard]] double calculate_mean_reversion_signal() const;
    [[nodiscard]] double calculate_volatility() const;
    [[nodiscard]] int get_adaptive_lookback() const;
    
    // Math utilities (inlined for performance)
    static double clip(double value, double min_val, double max_val) noexcept {
        return std::max(min_val, std::min(max_val, value));
    }
};

} // namespace otrep
