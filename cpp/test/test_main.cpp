/**
 * @file test_main.cpp
 * @brief Unit tests for HybridStrategy
 * 
 * Verifies correctness of signal calculations against expected behavior.
 */

#include <otrep/HybridStrategy.hpp>
#include <iostream>
#include <cmath>
#include <cassert>

using namespace otrep;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "✅ PASSED\n"; \
} while(0)

#define ASSERT_NEAR(a, b, eps) do { \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "\n  ASSERT_NEAR failed: " << (a) << " != " << (b) << " (eps=" << (eps) << ")\n"; \
        assert(false); \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "\n  ASSERT_TRUE failed: " #cond "\n"; \
        assert(false); \
    } \
} while(0)

// =============================================================================
// CIRCULAR BUFFER TESTS
// =============================================================================

TEST(circular_buffer_basic) {
    CircularBuffer<double> buf(5);
    
    ASSERT_TRUE(buf.empty());
    ASSERT_TRUE(buf.size() == 0);
    ASSERT_TRUE(buf.capacity() == 5);
    
    buf.push(1.0);
    buf.push(2.0);
    buf.push(3.0);
    
    ASSERT_TRUE(buf.size() == 3);
    ASSERT_TRUE(!buf.full());
    
    // Index 0 = newest (3.0), index 2 = oldest (1.0)
    ASSERT_NEAR(buf[0], 3.0, 0.001);
    ASSERT_NEAR(buf[1], 2.0, 0.001);
    ASSERT_NEAR(buf[2], 1.0, 0.001);
    
    // Chronological access
    ASSERT_NEAR(buf.at_chronological(0), 1.0, 0.001);
    ASSERT_NEAR(buf.at_chronological(1), 2.0, 0.001);
    ASSERT_NEAR(buf.at_chronological(2), 3.0, 0.001);
}

TEST(circular_buffer_wraparound) {
    CircularBuffer<int> buf(3);
    
    buf.push(1);
    buf.push(2);
    buf.push(3);
    ASSERT_TRUE(buf.full());
    
    buf.push(4);  // Overwrites 1
    buf.push(5);  // Overwrites 2
    
    ASSERT_TRUE(buf.size() == 3);
    ASSERT_TRUE(buf[0] == 5);  // Newest
    ASSERT_TRUE(buf[1] == 4);
    ASSERT_TRUE(buf[2] == 3);  // Oldest
}

TEST(circular_buffer_extract) {
    CircularBuffer<double> buf(10);
    
    for (int i = 1; i <= 10; ++i) {
        buf.push(static_cast<double>(i));
    }
    
    std::vector<double> extracted;
    buf.extract_last_n(5, extracted);
    
    ASSERT_TRUE(extracted.size() == 5);
    ASSERT_NEAR(extracted[0], 6.0, 0.001);  // Oldest of last 5
    ASSERT_NEAR(extracted[4], 10.0, 0.001); // Newest
}

// =============================================================================
// STRATEGY TESTS
// =============================================================================

TEST(strategy_default_params) {
    HybridStrategy strategy;
    const auto& params = strategy.params();
    
    ASSERT_TRUE(params.momentum_lookback == 20);
    ASSERT_TRUE(params.mean_reversion_lookback == 15);
    ASSERT_NEAR(params.bb_std_dev_multiplier, 1.5, 0.001);
}

TEST(strategy_insufficient_data) {
    HybridStrategy strategy;
    
    // Add only 5 bars (less than lookback)
    for (int i = 0; i < 5; ++i) {
        strategy.on_bar(Bar(100.0 + i, 101.0, 99.0, 100.0 + i, 1000000));
    }
    
    auto result = strategy.calculate_signal();
    
    ASSERT_NEAR(result.signal, 0.0, 0.001);
    ASSERT_NEAR(result.momentum_signal, 0.0, 0.001);
    ASSERT_NEAR(result.reversion_signal, 0.0, 0.001);
}

TEST(strategy_momentum_signal) {
    StrategyParams params;
    params.momentum_lookback = 5;
    params.mean_reversion_enabled = false;  // Isolate momentum
    params.momentum_weight = 1.0;
    params.mean_reversion_weight = 0.0;
    
    HybridStrategy strategy(params);
    
    // Add trending up prices
    double prices[] = {100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0};
    for (double p : prices) {
        strategy.on_bar(Bar(p, p + 0.5, p - 0.5, p, 1000000));
    }
    
    auto result = strategy.calculate_signal();
    
    // With uptrend, momentum should be positive
    ASSERT_TRUE(result.signal > 0.0);
    std::cout << "(signal=" << result.signal << ") ";
}

TEST(strategy_mean_reversion_signal) {
    StrategyParams params;
    params.mean_reversion_lookback = 5;
    params.momentum_weight = 0.0;  // Isolate mean reversion
    params.mean_reversion_weight = 1.0;
    params.mean_reversion_enabled = true;
    params.bb_std_dev_multiplier = 2.0;
    
    HybridStrategy strategy(params);
    
    // Price at mean (no signal)
    for (int i = 0; i < 10; ++i) {
        strategy.on_bar(Bar(100.0, 101.0, 99.0, 100.0, 1000000));
    }
    
    auto result = strategy.calculate_signal();
    
    // All prices same, std=0, so signal should be 0
    ASSERT_NEAR(result.signal, 0.0, 0.001);
    
    // Now add a price spike above mean
    strategy.clear();
    for (int i = 0; i < 5; ++i) {
        strategy.on_bar(Bar(100.0, 101.0, 99.0, 100.0, 1000000));
    }
    strategy.on_bar(Bar(110.0, 111.0, 109.0, 110.0, 1000000));  // Price spike
    
    result = strategy.calculate_signal();
    
    // Price above mean -> negative signal (expect reversion down)
    ASSERT_TRUE(result.signal < 0.0);
    std::cout << "(signal=" << result.signal << ") ";
}

TEST(strategy_trade_action) {
    HybridStrategy strategy;
    
    ASSERT_TRUE(strategy.get_trade_action(0.5) == TradeAction::BUY);
    ASSERT_TRUE(strategy.get_trade_action(-0.5) == TradeAction::SELL);
    ASSERT_TRUE(strategy.get_trade_action(0.02) == TradeAction::CLOSE);
    ASSERT_TRUE(strategy.get_trade_action(0.1) == TradeAction::HOLD);
}

TEST(strategy_latency_tracking) {
    HybridStrategy strategy;
    
    // Add enough bars
    for (int i = 0; i < 50; ++i) {
        strategy.on_bar(Bar(100.0 + i * 0.1, 101.0, 99.0, 100.0 + i * 0.1, 1000000));
    }
    
    auto result = strategy.calculate_signal();
    
    ASSERT_TRUE(result.latency_ns > 0);
    ASSERT_TRUE(strategy.last_latency_ns() > 0);
    
    strategy.reset_profiling();
    ASSERT_TRUE(strategy.last_latency_ns() == 0);
}

// =============================================================================
// MAIN
// =============================================================================

int main() {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "OTREP-X PRIME C++ Unit Tests\n";
    std::cout << std::string(50, '=') << "\n\n";
    
    try {
        RUN_TEST(circular_buffer_basic);
        RUN_TEST(circular_buffer_wraparound);
        RUN_TEST(circular_buffer_extract);
        RUN_TEST(strategy_default_params);
        RUN_TEST(strategy_insufficient_data);
        RUN_TEST(strategy_momentum_signal);
        RUN_TEST(strategy_mean_reversion_signal);
        RUN_TEST(strategy_trade_action);
        RUN_TEST(strategy_latency_tracking);
        
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "All tests passed! ✅\n";
        std::cout << std::string(50, '=') << "\n\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
