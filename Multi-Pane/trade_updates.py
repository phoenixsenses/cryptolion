from rich.console import Console
from rich.table import Table
import time

console = Console()

def trade_updates():
    table = Table(title="Trade Updates", style="cyan")
    table.add_column("Time", style="green")
    table.add_column("Action", style="bold")
    table.add_column("Symbol", style="magenta")
    table.add_column("Outcome", style="yellow")

    # Mock updates for demonstration
    trades = [
        ("12:01:45", "BUY", "BTC/USDT", "+2%"),
        ("12:02:30", "SELL", "ETH/USDT", "-1%"),
        ("12:03:15", "BUY", "ADA/USDT", "+1.5%"),
    ]

    while True:
        for trade in trades:
            table.add_row(*trade)
            console.clear()
            console.print(table)
            time.sleep(2)

if __name__ == "__main__":
    trade_updates()
