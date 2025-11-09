"""
OTREP-X PRIME – Phase X (Fixed)
Real-Time Analytics Dashboard with Proper DataFrames for Streamlit Charts
"""

import os
import json
import redis
import streamlit as st
import pandas as pd
from datetime import datetime
import time

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "otrep_events")

# ---------------------------------------------------------------------
# Redis Connection
# ---------------------------------------------------------------------
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

st.set_page_config(page_title="OTREP-X PRIME Dashboard", layout="wide")
st.title("🧠 OTREP-X PRIME – System Telemetry Dashboard")

# ---------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------
heartbeat_df = pd.DataFrame(columns=["time", "count"])
price_df = pd.DataFrame(columns=["time", "price"])

col1, col2 = st.columns(2)
with col1:
    st.subheader("💓 System Heartbeats")
    hb_chart = st.line_chart(heartbeat_df, x="time", y="count", height=250)
with col2:
    st.subheader("📊 Market Ticks (Price Stream)")
    price_chart = st.line_chart(price_df, x="time", y="price", height=250)

status_box = st.empty()
last_update = st.empty()

# ---------------------------------------------------------------------
# Redis Listener
# ---------------------------------------------------------------------
def listen_events():
    pubsub = r.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(EVENT_CHANNEL)
    for msg in pubsub.listen():
        if msg and msg.get("data"):
            yield json.loads(msg["data"])

# ---------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------
heartbeat_count = 0
for event in listen_events():
    etype = event.get("type")
    payload = event.get("payload", {})
    now = datetime.utcnow().strftime("%H:%M:%S")

    if etype == "SYSTEM_HEARTBEAT":
        heartbeat_count += 1
        new_row = pd.DataFrame({"time": [now], "count": [heartbeat_count]})
        heartbeat_df = pd.concat([heartbeat_df, new_row]).tail(30)
        hb_chart.add_rows(new_row.set_index("time"))
        status_box.info(f"✅ Heartbeat @ {payload['timestamp']} – Status: {payload['status']}")

    elif etype == "MARKET_TICK":
        new_row = pd.DataFrame({"time": [now], "price": [payload["price"]]})
        price_df = pd.concat([price_df, new_row]).tail(30)
        price_chart.add_rows(new_row.set_index("time"))

    last_update.write(f"🕒 Last update: {now}")
    time.sleep(0.1)
