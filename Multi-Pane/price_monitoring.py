from binance.client import Client
from rich.console import Console
from rich.table import Table
import time

# Replace these with your actual Binance API keys
api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

client = Client(api_key, api_secret)
console = Console()

def price_monitoring():
    while True:
        table = Table(title="Price Monitoring", style="blue")
        table.add_column("Symbol", style="cyan")
        table.add_column("Price (USDT)", style="green")

        # List of cryptocurrencies to track
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        for symbol in symbols:
            ticker = client.get_symbol_ticker(symbol=symbol)
            table.add_row(symbol, ticker['price'])

        console.clear()
        console.print(table)
        time.sleep(5)

if __name__ == "__main__":
    price_monitoring()
