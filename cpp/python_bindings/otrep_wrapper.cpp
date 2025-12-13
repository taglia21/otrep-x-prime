/**
 * @file otrep_wrapper.cpp
 * @brief Pybind11 bindings for OTREP-X PRIME C++ core
 * 
 * Exposes the high-performance HybridStrategy and MarketGraph to Python
 * for use in the MVT Trader production system.
 * 
 * @author OTREP-X Development Team
 * @date December 2025
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <otrep/HybridStrategy.hpp>
#include <otrep/MarketGraph.hpp>

namespace py = pybind11;

PYBIND11_MODULE(otrep_core, m) {
    m.doc() = "OTREP-X PRIME C++ Core - High-performance trading strategy engine";
    
    // =========================================================================
    // Bar struct
    // =========================================================================
    py::class_<otrep::Bar>(m, "Bar", "OHLCV Bar representation")
        .def(py::init<>(), "Default constructor")
        .def(py::init<double, double, double, double, int64_t>(),
             py::arg("open"), py::arg("high"), py::arg("low"), 
             py::arg("close"), py::arg("volume"),
             "Construct Bar with OHLCV values")
        .def_readwrite("open", &otrep::Bar::open, "Open price")
        .def_readwrite("high", &otrep::Bar::high, "High price")
        .def_readwrite("low", &otrep::Bar::low, "Low price")
        .def_readwrite("close", &otrep::Bar::close, "Close price")
        .def_readwrite("volume", &otrep::Bar::volume, "Volume")
        .def("__repr__", [](const otrep::Bar& b) {
            return "<Bar O=" + std::to_string(b.open) + 
                   " H=" + std::to_string(b.high) + 
                   " L=" + std::to_string(b.low) + 
                   " C=" + std::to_string(b.close) + 
                   " V=" + std::to_string(b.volume) + ">";
        });

    // =========================================================================
    // StrategyParams struct
    // =========================================================================
    py::class_<otrep::StrategyParams>(m, "StrategyParams", 
        "Configuration parameters for HybridStrategy (Phase IV optimized defaults)")
        .def(py::init<>(), "Default constructor with optimized Phase IV parameters")
        .def_readwrite("momentum_lookback", &otrep::StrategyParams::momentum_lookback,
                       "Momentum lookback period (default: 20)")
        .def_readwrite("trend_lookback", &otrep::StrategyParams::trend_lookback,
                       "Trend lookback period (default: 15)")
        .def_readwrite("signal_threshold", &otrep::StrategyParams::signal_threshold,
                       "Signal threshold for trade decisions (default: 0.15)")
        .def_readwrite("neutral_threshold", &otrep::StrategyParams::neutral_threshold,
                       "Neutral zone threshold (default: 0.05)")
        .def_readwrite("momentum_weight", &otrep::StrategyParams::momentum_weight,
                       "Weight of momentum signal (default: 0.5)")
        .def_readwrite("trend_weight", &otrep::StrategyParams::trend_weight,
                       "Weight of trend signal [deprecated] (default: 0.0)")
        .def_readwrite("adaptive_enabled", &otrep::StrategyParams::adaptive_enabled,
                       "Enable adaptive lookback based on volatility (default: true)")
        .def_readwrite("high_vol_lookback", &otrep::StrategyParams::high_vol_lookback,
                       "Lookback period during high volatility (default: 10)")
        .def_readwrite("low_vol_lookback", &otrep::StrategyParams::low_vol_lookback,
                       "Lookback period during low volatility (default: 30)")
        .def_readwrite("vol_multiplier", &otrep::StrategyParams::vol_multiplier,
                       "Volatility multiplier for regime detection (default: 1.5)")
        .def_readwrite("mean_reversion_enabled", &otrep::StrategyParams::mean_reversion_enabled,
                       "Enable mean reversion signal (default: true)")
        .def_readwrite("mean_reversion_weight", &otrep::StrategyParams::mean_reversion_weight,
                       "Weight of mean reversion signal (default: 0.5)")
        .def_readwrite("mean_reversion_lookback", &otrep::StrategyParams::mean_reversion_lookback,
                       "Mean reversion lookback period (default: 15)")
        .def_readwrite("bb_std_dev_multiplier", &otrep::StrategyParams::bb_std_dev_multiplier,
                       "Bollinger Bands std dev multiplier (default: 1.5)")
        .def_readwrite("reversion_threshold", &otrep::StrategyParams::reversion_threshold,
                       "Threshold for mean reversion signal (default: 0.01)")
        .def_readwrite("graph_weight", &otrep::StrategyParams::graph_weight,
                       "Weight of graph alpha signal (default: 0.2)")
        .def_readwrite("max_history_bars", &otrep::StrategyParams::max_history_bars,
                       "Maximum bars to keep in history (default: 200)")
        .def("__repr__", [](const otrep::StrategyParams& p) {
            return "<StrategyParams momentum_lookback=" + std::to_string(p.momentum_lookback) +
                   " mean_reversion_lookback=" + std::to_string(p.mean_reversion_lookback) +
                   " graph_weight=" + std::to_string(p.graph_weight) +
                   " bb_std_dev_multiplier=" + std::to_string(p.bb_std_dev_multiplier) + ">";
        });

    // =========================================================================
    // SignalResult struct
    // =========================================================================
    py::class_<otrep::SignalResult>(m, "SignalResult",
        "Result of signal calculation with profiling info")
        .def(py::init<>())
        .def_readonly("signal", &otrep::SignalResult::signal,
                     "Combined signal in [-1, 1]")
        .def_readonly("momentum_signal", &otrep::SignalResult::momentum_signal,
                     "Individual momentum component")
        .def_readonly("reversion_signal", &otrep::SignalResult::reversion_signal,
                     "Individual mean reversion component")
        .def_readonly("graph_signal", &otrep::SignalResult::graph_signal,
                     "External graph alpha signal")
        .def_readonly("lookback_used", &otrep::SignalResult::lookback_used,
                     "Actual lookback period used")
        .def_readonly("volatility", &otrep::SignalResult::volatility,
                     "Current volatility estimate")
        .def_readonly("latency_ns", &otrep::SignalResult::latency_ns,
                     "Calculation time in nanoseconds")
        .def_property_readonly("latency_ms", [](const otrep::SignalResult& r) {
            return static_cast<double>(r.latency_ns) / 1e6;
        }, "Calculation time in milliseconds")
        .def("__repr__", [](const otrep::SignalResult& r) {
            return "<SignalResult signal=" + std::to_string(r.signal) +
                   " momentum=" + std::to_string(r.momentum_signal) +
                   " reversion=" + std::to_string(r.reversion_signal) +
                   " graph=" + std::to_string(r.graph_signal) +
                   " latency_us=" + std::to_string(r.latency_ns / 1000.0) + ">";
        });

    // =========================================================================
    // TradeAction enum
    // =========================================================================
    py::enum_<otrep::TradeAction>(m, "TradeAction", "Trade action derived from signal")
        .value("HOLD", otrep::TradeAction::HOLD, "No action - hold current position")
        .value("BUY", otrep::TradeAction::BUY, "Enter long position")
        .value("SELL", otrep::TradeAction::SELL, "Enter short position")
        .value("CLOSE", otrep::TradeAction::CLOSE, "Close current position")
        .export_values();

    // =========================================================================
    // HybridStrategy class
    // =========================================================================
    py::class_<otrep::HybridStrategy>(m, "HybridStrategy",
        R"doc(
        High-performance hybrid trading strategy combining Momentum and Mean Reversion.
        
        This C++ implementation achieves ~27x speedup over Python (0.001 ms/bar).
        
        Optimized Parameters (Phase IV validated):
        - momentum_lookback: 20
        - mean_reversion_lookback: 15
        - bb_std_dev_multiplier: 1.5
        
        Example:
            >>> import otrep_core
            >>> strategy = otrep_core.HybridStrategy()
            >>> bar = otrep_core.Bar(100.0, 101.0, 99.5, 100.5, 10000)
            >>> strategy.on_bar(bar)
            >>> result = strategy.calculate_signal()
            >>> print(f"Signal: {result.signal:.4f}, Latency: {result.latency_ns} ns")
        )doc")
        .def(py::init<>(), 
             "Construct with default optimized parameters")
        .def(py::init<const otrep::StrategyParams&>(),
             py::arg("params"),
             "Construct with custom parameters")
        
        // Core methods
        .def("on_bar", &otrep::HybridStrategy::on_bar,
             py::arg("bar"),
             "Add a new bar to the price history. Call for each incoming market data update.")
        .def("calculate_signal", &otrep::HybridStrategy::calculate_signal,
             py::arg("external_graph_signal") = 0.0,
             "Calculate the combined trading signal. Accepts optional graph alpha signal.")
        .def("get_trade_action", &otrep::HybridStrategy::get_trade_action,
             py::arg("signal"),
             "Convert signal to trade action (HOLD, BUY, SELL, CLOSE)")
        
        // Profiling
        .def("last_latency_ns", &otrep::HybridStrategy::last_latency_ns,
             "Get last calculation latency in nanoseconds")
        .def("last_latency_ms", &otrep::HybridStrategy::last_latency_ms,
             "Get last calculation latency in milliseconds")
        .def("avg_latency_ns", &otrep::HybridStrategy::avg_latency_ns,
             "Get average latency over all calculations")
        .def("reset_profiling", &otrep::HybridStrategy::reset_profiling,
             "Reset profiling statistics")
        
        // State
        .def("bar_count", &otrep::HybridStrategy::bar_count,
             "Get number of bars in history")
        .def("clear", &otrep::HybridStrategy::clear,
             "Clear all price history")
        .def_property_readonly("params", &otrep::HybridStrategy::params,
             "Get current strategy parameters (read-only)")
        
        // Convenience method for bulk loading historical data
        .def("load_bars", [](otrep::HybridStrategy& self, 
                            const std::vector<std::tuple<double, double, double, double, int64_t>>& bars) {
            for (const auto& [o, h, l, c, v] : bars) {
                self.on_bar(otrep::Bar(o, h, l, c, v));
            }
            return self.bar_count();
        }, py::arg("bars"),
        R"doc(
        Bulk load historical bars from a list of tuples.
        
        Args:
            bars: List of (open, high, low, close, volume) tuples
            
        Returns:
            Number of bars loaded
            
        Example:
            >>> strategy.load_bars([(100.0, 101.0, 99.0, 100.5, 1000), ...])
        )doc")
        
        .def("__repr__", [](const otrep::HybridStrategy& s) {
            return "<HybridStrategy bars=" + std::to_string(s.bar_count()) +
                   " avg_latency_ns=" + std::to_string(static_cast<int64_t>(s.avg_latency_ns())) + ">";
        });

    // =========================================================================
    // GraphParams struct
    // =========================================================================
    py::class_<otrep::GraphParams>(m, "GraphParams",
        "Configuration parameters for MarketGraph Laplacian diffusion")
        .def(py::init<>(), "Default constructor")
        .def_readwrite("correlation_threshold", &otrep::GraphParams::correlation_threshold,
                       "Hard threshold for adjacency matrix (default: 0.5)")
        .def_readwrite("diffusion_alpha", &otrep::GraphParams::diffusion_alpha,
                       "Laplacian diffusion strength (default: 1.0)")
        .def_readwrite("lookback_bars", &otrep::GraphParams::lookback_bars,
                       "Bars for correlation calculation (default: 50)")
        .def("__repr__", [](const otrep::GraphParams& p) {
            return "<GraphParams threshold=" + std::to_string(p.correlation_threshold) +
                   " alpha=" + std::to_string(p.diffusion_alpha) + ">";
        });

    // =========================================================================
    // GraphSignalResult struct
    // =========================================================================
    py::class_<otrep::GraphSignalResult>(m, "GraphSignalResult",
        "Result of graph signal calculation")
        .def(py::init<>())
        .def_property_readonly("signals", [](const otrep::GraphSignalResult& r) -> Eigen::VectorXd {
                     return r.signals;
                 }, "Per-asset signals as numpy array")
        .def_property_readonly("correlation_matrix", [](const otrep::GraphSignalResult& r) -> Eigen::MatrixXd {
                     return r.correlation_matrix;
                 }, "N x N correlation matrix")
        .def_property_readonly("adjacency_matrix", [](const otrep::GraphSignalResult& r) -> Eigen::MatrixXd {
                     return r.adjacency_matrix;
                 }, "Thresholded adjacency matrix W")
        .def_readonly("num_edges", &otrep::GraphSignalResult::num_edges,
                     "Number of edges in graph")
        .def_readonly("avg_correlation", &otrep::GraphSignalResult::avg_correlation,
                     "Average absolute correlation")
        .def_readonly("latency_ns", &otrep::GraphSignalResult::latency_ns,
                     "Calculation time in nanoseconds")
        .def_property_readonly("latency_ms", [](const otrep::GraphSignalResult& r) {
            return static_cast<double>(r.latency_ns) / 1e6;
        }, "Calculation time in milliseconds")
        .def("__repr__", [](const otrep::GraphSignalResult& r) {
            return "<GraphSignalResult edges=" + std::to_string(r.num_edges) +
                   " avg_corr=" + std::to_string(r.avg_correlation) +
                   " latency_ms=" + std::to_string(r.latency_ns / 1e6) + ">";
        });

    // =========================================================================
    // MarketGraph class
    // =========================================================================
    py::class_<otrep::MarketGraph>(m, "MarketGraph",
        R"doc(
        Market correlation graph for Laplacian diffusion signals.
        
        Analyzes the market as a connected network where edges represent
        correlation between assets. Detects mispricing via diffusion residuals.
        
        The signal for asset i indicates:
          - s[i] > 0: Asset returned MORE than neighbors expect → expensive → SELL
          - s[i] < 0: Asset returned LESS than neighbors expect → cheap → BUY
        
        Example:
            >>> import otrep_core
            >>> import numpy as np
            >>> graph = otrep_core.MarketGraph()
            >>> returns = np.random.randn(50, 40) * 0.01  # 50 bars, 40 assets
            >>> result = graph.calculate_signals(returns)
            >>> print(f"Edges: {result.num_edges}, Latency: {result.latency_ms:.2f}ms")
        )doc")
        .def(py::init<>(), 
             "Construct with default parameters")
        .def(py::init<const otrep::GraphParams&>(),
             py::arg("params"),
             "Construct with custom parameters")
        .def("calculate_signals", [](otrep::MarketGraph& self, 
                                      const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& returns) {
             // Pybind11 automatically maps NumPy row-major to Eigen::Ref<RowMajor> (zero-copy)
             if (returns.rows() < 2 || returns.cols() < 2) {
                 throw std::runtime_error("returns must have at least 2 rows and 2 columns");
             }
             
             // Convert to column-major for internal processing
             Eigen::MatrixXd returns_col = returns;
             return self.calculate_signals(returns_col);
        },
             py::arg("returns"),
             R"doc(
             Calculate graph-based signals from returns matrix (zero-copy from NumPy).
             
             Args:
                 returns: numpy array of shape (T, N) where T=time periods, N=assets
                         Values should be log returns
             
             Returns:
                 GraphSignalResult with per-asset signals
             )doc")
        .def("calculate_signals_map", [](otrep::MarketGraph& self, 
                                         const Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& returns,
                                         const std::vector<std::string>& symbols) {
             // Pybind11 automatically maps NumPy row-major to Eigen::Ref<RowMajor> (zero-copy)
             if (returns.rows() < 2 || returns.cols() < 2) {
                 throw std::runtime_error("returns must have at least 2 rows and 2 columns");
             }
             
             // Call the Row-Major overload directly (zero-copy mapping)
             return self.calculate_signals_map(returns, symbols);
        },
             py::arg("returns"), py::arg("symbols"),
             R"doc(
             Calculate signals and return as symbol -> signal dict (zero-copy from NumPy).
             
             Args:
                 returns: numpy array of shape (T, N)
                 symbols: list of N symbol names
             
             Returns:
                 Dict mapping symbol to signal value
             )doc")
        .def("last_latency_ns", &otrep::MarketGraph::last_latency_ns,
             "Get last calculation latency in nanoseconds")
        .def_property_readonly("params", &otrep::MarketGraph::params,
             "Get current parameters (read-only)")
        .def("set_params", &otrep::MarketGraph::set_params,
             py::arg("params"),
             "Update parameters")
        .def("__repr__", [](const otrep::MarketGraph& g) {
            return "<MarketGraph threshold=" + std::to_string(g.params().correlation_threshold) +
                   " alpha=" + std::to_string(g.params().diffusion_alpha) + ">";
        });

    // =========================================================================
    // Module-level utilities
    // =========================================================================
    m.def("get_version", []() { return "2.0.0"; }, "Get OTREP-X Core version");
    
    m.attr("__version__") = "2.0.0";
}
