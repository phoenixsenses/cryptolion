from rich.console import Console
from rich.table import Table
import time

console = Console()

def liquidation_events():
    table = Table(title="Liquidation Events", style="red")
    table.add_column("Time", style="green")
    table.add_column("Symbol", style="magenta")
    table.add_column("Amount", style="yellow")
    table.add_column("Type", style="bold")

    # Mock data for demonstration
    liquidations = [
        ("12:01:30", "BTC/USDT", "100K", "Long"),
        ("12:02:10", "ETH/USDT", "50K", "Short"),
        ("12:03:05", "ADA/USDT", "10K", "Long"),
    ]

    while True:
        for event in liquidations:
            table.add_row(*event)
            console.clear()
            console.print(table)
            time.sleep(2)

if __name__ == "__main__":
    liquidation_events()
