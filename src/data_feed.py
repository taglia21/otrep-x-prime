import pandas as pd
from src.core_logger import logger

class DataFeed:
    def __init__(self, csv_path=None, dataframe=None):
        if dataframe is not None:
            df = dataframe.copy()
        elif csv_path:
            df = pd.read_csv(csv_path)
        else:
            raise ValueError("DataFeed needs either csv_path or dataframe")

        df.columns = [c.strip().lower() for c in df.columns]
        if "close" not in df.columns:
            if "price" in df.columns:
                df.rename(columns={"price": "close"}, inplace=True)
            elif "adj close" in df.columns:
                df.rename(columns={"adj close": "close"}, inplace=True)
            else:
                raise ValueError("Missing price/close/adj close column")

        if "date" not in df.columns and isinstance(df.index.name, str) and "date" in df.index.name.lower():
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]: "date"}, inplace=True)

        self.df = df
        logger.info(f"DataFeed initialized with {len(self.df)} rows")
