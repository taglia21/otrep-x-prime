import pandas as pd
import pytz

class DataFeed:
    def __init__(self, markets):
        self.markets = markets
        self.data = {}

    def load(self):
        for m in self.markets:
            df = pd.read_csv(m["data_path"])
            tz = pytz.timezone(m["timezone"])
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(tz)
            self.data[m["symbol"]] = df
            print(f"[INFO] Loaded {m['symbol']} ({m['region']}) with {len(df)} rows")

    def get(self, symbol):
        return self.data.get(symbol)
