/**
 * @file benchmark_main.cpp
 * @brief Performance benchmark for HybridStrategy
 * 
 * Measures signal calculation latency to validate C++ port performance.
 * Target: < 0.01 ms/bar (10x improvement over Python)
 */

#include <otrep/HybridStrategy.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace otrep;

/**
 * @brief Generate synthetic price data similar to real market data
 */
std::vector<Bar> generate_synthetic_bars(size_t count, double start_price = 100.0) {
    std::vector<Bar> bars;
    bars.reserve(count);
    
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<double> returns(0.0005, 0.02);  // ~0.05% drift, 2% vol
    
    double price = start_price;
    auto now = std::chrono::system_clock::now();
    
    for (size_t i = 0; i < count; ++i) {
        double ret = returns(rng);
        price *= (1.0 + ret);
        
        // Generate OHLC from close
        double close = price;
        double range = price * std::abs(returns(rng));
        double open = close - range * (returns(rng) > 0 ? 0.5 : -0.5);
        double high = std::max(open, close) + range * 0.3;
        double low = std::min(open, close) - range * 0.3;
        int64_t volume = 1000000 + static_cast<int64_t>(rng() % 9000000);
        
        bars.emplace_back(now + std::chrono::minutes(i * 5), open, high, low, close, volume);
    }
    
    return bars;
}

/**
 * @brief Run benchmark and collect statistics
 */
void run_benchmark(size_t num_bars, size_t warmup_bars, size_t measurement_iterations) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "OTREP-X PRIME C++ HybridStrategy Benchmark\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    // Create strategy with optimized parameters
    StrategyParams params;
    params.momentum_lookback = 20;        // Optimized
    params.mean_reversion_lookback = 15;  // Optimized
    params.bb_std_dev_multiplier = 1.5;   // Optimized
    
    HybridStrategy strategy(params);
    
    std::cout << "Configuration:\n";
    std::cout << "  momentum_lookback:        " << params.momentum_lookback << "\n";
    std::cout << "  mean_reversion_lookback:  " << params.mean_reversion_lookback << "\n";
    std::cout << "  bb_std_dev_multiplier:    " << params.bb_std_dev_multiplier << "\n";
    std::cout << "  Total bars to process:    " << num_bars << "\n";
    std::cout << "  Warmup bars:              " << warmup_bars << "\n";
    std::cout << "  Measurement iterations:   " << measurement_iterations << "\n\n";
    
    // Generate synthetic data
    std::cout << "Generating synthetic price data...\n";
    auto bars = generate_synthetic_bars(num_bars);
    
    std::vector<int64_t> latencies;
    latencies.reserve(measurement_iterations * (num_bars - warmup_bars));
    
    // Run multiple iterations for statistical significance
    for (size_t iter = 0; iter < measurement_iterations; ++iter) {
        strategy.clear();
        strategy.reset_profiling();
        
        // Feed bars and measure signal calculation
        for (size_t i = 0; i < num_bars; ++i) {
            strategy.on_bar(bars[i]);
            
            if (i >= warmup_bars) {
                auto result = strategy.calculate_signal();
                latencies.push_back(result.latency_ns);
            }
        }
    }
    
    // Calculate statistics
    std::sort(latencies.begin(), latencies.end());
    
    const size_t n = latencies.size();
    const int64_t min_ns = latencies.front();
    const int64_t max_ns = latencies.back();
    const int64_t p50_ns = latencies[n / 2];
    const int64_t p95_ns = latencies[static_cast<size_t>(n * 0.95)];
    const int64_t p99_ns = latencies[static_cast<size_t>(n * 0.99)];
    
    const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    const double mean_ns = sum / static_cast<double>(n);
    
    // Convert to microseconds for readability
    auto ns_to_us = [](double ns) { return ns / 1000.0; };
    auto ns_to_ms = [](double ns) { return ns / 1e6; };
    
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "LATENCY RESULTS (per signal calculation)\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Samples:      " << n << "\n";
    std::cout << "  Mean:         " << ns_to_us(mean_ns) << " µs  (" << ns_to_ms(mean_ns) << " ms)\n";
    std::cout << "  Min:          " << ns_to_us(min_ns) << " µs\n";
    std::cout << "  Max:          " << ns_to_us(max_ns) << " µs\n";
    std::cout << "  P50 (median): " << ns_to_us(p50_ns) << " µs\n";
    std::cout << "  P95:          " << ns_to_us(p95_ns) << " µs\n";
    std::cout << "  P99:          " << ns_to_us(p99_ns) << " µs\n";
    
    // Performance comparison with Python
    const double python_latency_ms = 0.03;  // From Phase IV profiling
    const double cpp_latency_ms = ns_to_ms(mean_ns);
    const double speedup = python_latency_ms / cpp_latency_ms;
    
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "PERFORMANCE COMPARISON\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "  Python latency:  " << python_latency_ms << " ms/bar\n";
    std::cout << "  C++ latency:     " << cpp_latency_ms << " ms/bar\n";
    std::cout << "  Speedup:         " << std::setprecision(1) << speedup << "x\n";
    
    if (cpp_latency_ms < 0.01) {
        std::cout << "\n  ✅ TARGET ACHIEVED: < 0.01 ms/bar\n";
    } else {
        std::cout << "\n  ⚠️ Target not met: " << cpp_latency_ms << " ms > 0.01 ms\n";
    }
    
    // Run a final signal calculation and show result
    strategy.clear();
    for (const auto& bar : bars) {
        strategy.on_bar(bar);
    }
    auto final_result = strategy.calculate_signal();
    
    std::cout << "\n" << std::string(60, '-') << "\n";
    std::cout << "SAMPLE SIGNAL OUTPUT\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::setprecision(4);
    std::cout << "  Combined Signal:    " << final_result.signal << "\n";
    std::cout << "  Momentum Signal:    " << final_result.momentum_signal << "\n";
    std::cout << "  Reversion Signal:   " << final_result.reversion_signal << "\n";
    std::cout << "  Lookback Used:      " << final_result.lookback_used << "\n";
    std::cout << "  Volatility:         " << final_result.volatility << "\n";
    std::cout << "  Trade Action:       ";
    
    switch (strategy.get_trade_action(final_result.signal)) {
        case TradeAction::BUY:   std::cout << "BUY\n"; break;
        case TradeAction::SELL:  std::cout << "SELL\n"; break;
        case TradeAction::CLOSE: std::cout << "CLOSE\n"; break;
        case TradeAction::HOLD:  std::cout << "HOLD\n"; break;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Benchmark complete.\n\n";
}

int main() {
    try {
        // Run benchmark with realistic parameters
        // 1000 bars ≈ ~35 trading days of 5-min bars
        run_benchmark(
            1000,   // Total bars
            50,     // Warmup bars (fill lookback windows)
            10      // Measurement iterations
        );
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
