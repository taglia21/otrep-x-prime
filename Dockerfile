FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev libssl-dev git && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/*
RUN pip install redis prometheus-client


COPY src /app/src
COPY data /app/data

COPY config.yaml /app/config.yaml
# Ensure Python finds all local modules
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING="utf-8"

CMD ["python", "-m", "src.core.app_stage02_backtest"]
# --- Streamlit Dashboard Support ---
RUN pip install streamlit watchdog

