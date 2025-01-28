from rich.console import Console
import time
import random

console = Console()

def bot_logs():
    console.log("[green]Initializing CryptoLion Bot...[/green]")
    time.sleep(1)
    console.log("[cyan]Connecting to Binance API...[/cyan]")
    time.sleep(2)
    for i in range(1, 20):
        log_msg = f"[yellow]Trade {i}: {random.choice(['BUY', 'SELL'])} BTC/USDT at {random.uniform(20000, 30000):.2f}[/yellow]"
        console.log(log_msg)
        time.sleep(1)

if __name__ == "__main__":
    bot_logs()
