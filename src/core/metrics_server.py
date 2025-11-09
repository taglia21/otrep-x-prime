from prometheus_client import Gauge, Counter, Histogram, start_http_server
import time

TRADE_COUNT = Counter('otrepx_trades_total', 'Total trades executed')
PORTFOLIO_VALUE = Gauge('otrepx_portfolio_value', 'Current portfolio market value')
ORDER_LATENCY = Histogram('otrepx_order_latency_seconds', 'Latency per order execution')
SHARPE_RATIO = Gauge('otrepx_sharpe_ratio', 'Current portfolio Sharpe ratio')

def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics endpoint accessible from host"""
    print(f"[INFO] Starting metrics server on 0.0.0.0:{port}")
    start_http_server(port, addr="0.0.0.0")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    start_metrics_server()
