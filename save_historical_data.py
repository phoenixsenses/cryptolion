# save_historical_data.py
import os
import sys
import pandas as pd
from binance.client import Client
from dotenv import load_dotenv
import logging

# Setup logging
logger = logging.getLogger("SaveHistoricalData")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

def fetch_and_save(symbol: str, interval: str="1m", lookback: int=1000):
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")
    if not api_key or not api_secret:
        logger.critical("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set in .env file.")
        sys.exit(1)
    client = Client(api_key, api_secret, testnet=True)
    try:
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
        df = pd.DataFrame(klines, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        data_path = os.path.join("historical_data", f"{symbol}.csv")
        df.to_csv(data_path)
        logger.info(f"Saved historical data for {symbol} to {data_path}")
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")

def main():
    symbols = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "BTCBUSD"]  # Add more symbols as needed
    for sym in symbols:
        fetch_and_save(sym)

if __name__ == "__main__":
    main()
