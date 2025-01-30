#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
from typing import Optional

from strategies import StrategyA, StrategyParametersModel  # or from your main code
from strategies import FeatureEngineer  # or relevant class from main code
# etc. You'd import what you need from your main code

# A minimal in-memory "bot" or environment
class BacktestBot:
    def __init__(self, starting_balance=1000.0):
        self.balance = starting_balance

    def on_position_closed(self, strategy, pnl: float):
        self.balance += pnl
        logging.info(f"[Backtest] Position closed => PnL={pnl:.2f}, new balance={self.balance:.2f}")

def run_backtest(csv_file: str):
    # 1) Load historical data from CSV (candles must have columns: open_time, open, high, low, close, volume)
    df = pd.read_csv(csv_file, parse_dates=["open_time"], index_col="open_time")
    # 2) Add features
    df = FeatureEngineer.add_technical_features(df)

    # 3) Setup strategy environment
    # For example, use StrategyA with some default parameters
    # We'll skip real sentiment, so pass None or a dummy config
    strat_params = StrategyParametersModel(sma_period=20, kelly_win_rate=0.55, kelly_win_loss_ratio=1.8)
    strategy = StrategyA(
        symbol="BTCUSDT",
        parameters=strat_params,
        adaptive_atr=True,
        atr_multiplier=1.5,
        sentiment_config=None
    )

    # Minimal "bot" to handle PnL updates
    bot = BacktestBot()
    strategy.bot = bot

    # 4) Simulate candle-by-candle
    realized_pnl = 0.0
    for i in range(len(df)):
        current_slice = df.iloc[: i+1]  # up to the current candle
        # Monitor existing position (check partial TPs, trailing stops, etc.)
        realized_pnl += strategy.monitor_positions(current_slice)

        # On final candle close, decide new trade
        if i == len(df) - 1:
            # This is the last candle, so let's skip
            break

        # "Decision" once the candle is final
        decision = strategy.decision(current_slice)
        logging.info(f"[Backtest] Candle {i}, close={current_slice['close'].iloc[-1]:.2f}, decision={decision}")

        # If it's a BUY or SELL, let's do a simplified fill
        if decision in ["BUY", "SELL"] and not strategy.is_open:
            # Example "quantity" => 0.01 BTC
            qty = 0.01  
            price = current_slice["close"].iloc[-1]
            # For a real backtest, you'd do a Kelly or risk-based calc
            stop_loss = price * (0.98 if decision == "BUY" else 1.02)
            take_profit = price * (1.02 if decision == "BUY" else 0.98)
            strategy.execute_trade(
                side=decision,
                quantity=qty,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing=True
            )

    logging.info(f"[Backtest] Final realized PnL={realized_pnl:.2f}, final balance={bot.balance:.2f}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest.py path_to_candle_data.csv")
        sys.exit(1)
    csv_file = sys.argv[1]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_backtest(csv_file)

if __name__ == "__main__":
    main()
