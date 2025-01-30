import os
import sys
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv  # For loading .env

from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger("CryptoLionBotBacktest")
logger.setLevel(logging.DEBUG)

# Prevent adding multiple handlers if they already exist
if not logger.handlers:
    # JSON Formatter for file logging
    json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
    file_handler = RotatingFileHandler("backtest.log", maxBytes=10*1024*1024, backupCount=2)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)

    # Console Formatter for terminal output
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set to INFO to reduce console clutter
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# =============================================================================
# CONFIG MODELS
# =============================================================================
class ATRThresholdModel(BaseModel):
    LOWER_BOUND: float
    UPPER_BOUND: float

class SentimentAnalysisModel(BaseModel):
    ENABLED: bool
    DATA_SOURCE: List[str]
    TWEETS_PER_SYMBOL: int
    SENTIMENT_THRESHOLD_POSITIVE: float
    SENTIMENT_THRESHOLD_NEGATIVE: float
    ANALYZER: List[str]
    TREND_ANALYSIS: bool
    TREND_WINDOW: int

class StrategyParametersModel(BaseModel):
    correlation_threshold: Optional[float] = 0.7
    reference_symbol: Optional[str] = "BTCUSDT"
    sma_period: Optional[int] = 20
    atr_period: Optional[int] = 14
    atr_multiplier: Optional[float] = 1.5
    kelly_win_rate: Optional[float] = 0.5
    kelly_win_loss_ratio: Optional[float] = 1.5

class StrategyModel(BaseModel):
    class_name: str = Field(..., alias="class")
    DEFAULT_LEVERAGE: int
    BASE_RISK_PER_TRADE: float
    ATR_MULTIPLIER: float
    RISK_REWARD_RATIO: float
    ADAPTIVE_ATR: bool
    DYNAMIC_RISK_ADJUSTMENT: bool
    ATR_THRESHOLD: ATRThresholdModel
    SENTIMENT_ANALYSIS: SentimentAnalysisModel
    TRADE_DURATION: int
    strategy_parameters: Optional[StrategyParametersModel] = None

class RiskParametersModel(BaseModel):
    DEFAULT_LEVERAGE: int
    BASE_RISK_PER_TRADE: float

class TradeParametersModel(BaseModel):
    stop_loss: float
    take_profit: float
    trailing_stop_loss: bool

class SymbolModel(BaseModel):
    symbol: str
    base_currency: str
    quote_currency: str
    strategy: str
    strategy_parameters: Optional[StrategyParametersModel] = None
    risk_parameters: RiskParametersModel
    trade_parameters: TradeParametersModel

class ConfigModel(BaseModel):
    MIN_NOTIONAL: float
    STARTING_BALANCE: float
    STRATEGIES: Dict[str, StrategyModel]
    ACTIVE_STRATEGY: str
    SYMBOLS_TO_TRADE: Dict[str, SymbolModel]
    USE_TESTNET: Optional[bool] = False

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def compute_atr(df: pd.DataFrame, period: int=14) -> float:
    if len(df) < period:
        return 0.0
    df = df.copy()
    df["previous_close"] = df["close"].shift(1)
    df["high_low"] = df["high"] - df["low"]
    df["high_pc"] = abs(df["high"] - df["previous_close"])
    df["low_pc"] = abs(df["low"] - df["previous_close"])
    df["tr"] = df[["high_low", "high_pc", "low_pc"]].max(axis=1)
    df["atr"] = df["tr"].rolling(period).mean()
    latest_atr = df["atr"].iloc[-1] if len(df) >= period else 0.0
    return 0.0 if np.isnan(latest_atr) else latest_atr

def analyze_sentiment(text: str) -> float:
    # Placeholder for sentiment analysis
    # Implement actual sentiment analysis or mock
    return 0.0

def combined_sentiment_score(symbol: str, cfg: SentimentAnalysisModel) -> float:
    if not cfg.ENABLED:
        return 0.0
    total = 0.0
    sources = 0
    if "twitter" in cfg.DATA_SOURCE:
        # Implement actual Twitter sentiment fetching
        total += analyze_sentiment(f"Mock tweet about {symbol}")
        sources += 1
    if "news" in cfg.DATA_SOURCE:
        # Implement actual news sentiment fetching
        total += analyze_sentiment(f"Mock news headline about {symbol}")
        sources += 1
    return total / sources if sources > 0 else 0.0

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        # RSI
        if "RSI" not in df.columns:
            df["RSI"] = ta.rsi(df["close"], length=14)
        # OBV
        if "OBV" not in df.columns:
            df["OBV"] = ta.obv(df["close"], df["volume"])
        # SMA
        if "SMA_50" not in df.columns:
            df["SMA_50"] = ta.sma(df["close"], length=50)
        # EMA
        if "EMA_20" not in df.columns:
            df["EMA_20"] = ta.ema(df["close"], length=20)
        # MACD
        if "MACD_12_26_9" not in df.columns or "MACDs_12_26_9" not in df.columns:
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
        # Bollinger Bands
        if "BBU_20_2.0" not in df.columns or "BBL_20_2.0" not in df.columns:
            bb = ta.bbands(df["close"], length=20, std=2)
            df = pd.concat([df, bb], axis=1)
        return df

# =============================================================================
# BACKTESTING CLASS
# =============================================================================
class Backtester:
    def __init__(self, config: ConfigModel):
        self.config = config
        self.results = {}
        self.trade_log = []
    
    def run_backtest(self):
        for sym_key, sym_cfg in self.config.SYMBOLS_TO_TRADE.items():
            logger.info(f"Starting backtest for {sym_key} using strategy {sym_cfg.strategy}")
            df = self.fetch_historical_data(sym_cfg.symbol, "1m", lookback=1000)
            if df.empty:
                logger.warning(f"No data for {sym_key}, skipping backtest.")
                continue
            df = FeatureEngineer.add_technical_features(df)
    
            strategy_cls = self.get_strategy_class(sym_cfg.strategy)
            if not strategy_cls:
                logger.warning(f"Strategy {sym_cfg.strategy} not found for {sym_key}, skipping.")
                continue
    
            strategy = strategy_cls(
                symbol=sym_key,
                parameters=sym_cfg.strategy_parameters,
                adaptive_atr=self.config.STRATEGIES[sym_cfg.strategy].ADAPTIVE_ATR,
                atr_multiplier=self.config.STRATEGIES[sym_cfg.strategy].ATR_MULTIPLIER,
                sentiment_config=self.config.STRATEGIES[sym_cfg.strategy].SENTIMENT_ANALYSIS
            )
    
            # Simulate each candle
            for idx in range(len(df)):
                current_df = df.iloc[:idx+1].copy()
                decision = strategy.decision(current_df)
                close_price = current_df["close"].iloc[-1]
    
                if decision in ["BUY", "SELL"] and not strategy.is_open:
                    qty = self.calculate_kelly_quantity(sym_key, close_price, strategy)
                    if qty > 0:
                        strategy.execute_trade(decision, qty, close_price)
                        self.trade_log.append({
                            "symbol": sym_key,
                            "action": decision,
                            "price": close_price,
                            "quantity": qty,
                            "timestamp": current_df.index[-1]
                        })
                elif strategy.is_open:
                    pnl = strategy.monitor_positions(current_df)
                    if pnl != 0.0:
                        self.trade_log.append({
                            "symbol": sym_key,
                            "action": "CLOSE",
                            "price": close_price,
                            "quantity": strategy.qty,
                            "pnl": pnl,
                            "timestamp": current_df.index[-1]
                        })
    
            # Summarize results
            total_pnl = sum([trade.get("pnl", 0.0) for trade in self.trade_log if trade["action"] == "CLOSE"])
            total_trades = len([trade for trade in self.trade_log if trade["action"] in ["BUY", "SELL"]])
            win_trades = len([trade for trade in self.trade_log if trade.get("pnl", 0.0) > 0.0])
            loss_trades = len([trade for trade in self.trade_log if trade.get("pnl", 0.0) < 0.0])
            win_rate = (win_trades / loss_trades) * 100 if loss_trades > 0 else 0.0
    
            self.results[sym_key] = {
                "Total PnL": total_pnl,
                "Total Trades": total_trades,
                "Win Trades": win_trades,
                "Loss Trades": loss_trades,
                "Win Rate (%)": win_rate
            }
    
            logger.info(f"Backtest completed for {sym_key}: PnL={total_pnl:.2f}, Trades={total_trades}, Win Rate={win_rate:.2f}%")
    
    def fetch_historical_data(self, symbol: str, interval: str="1m", lookback: int=1000) -> pd.DataFrame:
        # Implement historical data fetching similar to your bot
        # For backtesting, you can load data from CSV or use API
        # Here, we'll assume data is loaded from a CSV named 'symbol.csv'
        try:
            data_path = os.path.join("historical_data", f"{symbol}.csv")
            df = pd.read_csv(data_path, parse_dates=["open_time"])
            df.set_index("open_time", inplace=True)
            df.rename(columns={
                "open_time": "open_time",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "close_time": "close_time",
                "quote_asset_volume": "quote_asset_volume",
                "number_of_trades": "number_of_trades",
                "taker_buy_base_asset_volume": "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume": "taker_buy_quote_asset_volume",
                "ignore": "ignore"
            }, inplace=True)
            df = df.tail(lookback)
            logger.debug(f"Loaded historical data for {symbol}, {len(df)} records.")
            return df
        except FileNotFoundError:
            logger.error(f"Historical data file for {symbol} not found at 'historical_data/{symbol}.csv'.")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_strategy_class(self, strategy_name: str):
        strategy_map = {
            "StrategyA": StrategyA,
            "DivergenceStrategy": DivergenceStrategy,
            "MomentumStrategy": MomentumStrategy,
            "ArbitrageStrategy": ArbitrageStrategy
        }
        return strategy_map.get(strategy_name, None)
    
    def calculate_kelly_quantity(self, symbol: str, current_price: float, strategy) -> float:
        # Simplified Kelly Criterion for backtesting
        p, b = 0.5, 1.5  # You can adjust or calculate based on historical data
        kelly_fraction = kelly_criterion(p, b)
        if kelly_fraction < 0:
            kelly_fraction = 0.0
        kelly_fraction = min(kelly_fraction, 0.25)  # Cap to 25%
    
        notional = kelly_fraction * self.config.STARTING_BALANCE
        leverage = self.config.SYMBOLS_TO_TRADE[symbol].risk_parameters.DEFAULT_LEVERAGE
        notional *= leverage
        if current_price == 0:
            return 0.0
        qty = notional / current_price
        if (qty * current_price) < self.config.MIN_NOTIONAL:
            logger.warning(f"[{symbol}] Notional {qty*current_price:.2f} < MIN_NOTIONAL => 0.0")
            return 0.0
        return float(round(qty, 6))
    
    def plot_results(self):
        # Create a summary DataFrame
        summary_df = pd.DataFrame.from_dict(self.results, orient='index')
        print("\nBacktest Summary:")
        print(summary_df)
    
        # Plot PnL per Symbol
        plt.figure(figsize=(10, 6))
        sns.barplot(x=summary_df.index, y="Total PnL", data=summary_df)
        plt.title("Total PnL per Symbol")
        plt.xlabel("Symbol")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.savefig("pnl_per_symbol.png")
        plt.show()
    
        # Additional plots can be added as needed
    
    def generate_trade_log(self):
        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv("trade_log.csv", index=False)
        logger.info("Trade log saved to 'trade_log.csv'")
    
# =============================================================================
# STRATEGIES
# =============================================================================
class BaseStrategy:
    def __init__(self,
                 symbol: str,
                 parameters: Optional[StrategyParametersModel] = None,
                 adaptive_atr: bool = False,
                 atr_multiplier: float = 1.5,
                 sentiment_config: Optional[SentimentAnalysisModel] = None):
        self.symbol = symbol
        self.is_open = False
        self.side = None
        self.qty = 0.0
        self.entry_price = 0.0
        self.partial_tps: List[Dict[str, Any]] = []
        self.parameters = parameters or StrategyParametersModel()
        self.adaptive_atr = adaptive_atr
        self.static_atr_multiplier = atr_multiplier
        self.trailing_offset = self.static_atr_multiplier
        self.sentiment_config = sentiment_config
        self.sentiment_score = 0.0
    
    def fetch_sentiment(self):
        if self.sentiment_config:
            self.sentiment_score = combined_sentiment_score(self.symbol, self.sentiment_config)
            logger.debug(f"[Sentiment] {self.symbol} => {self.sentiment_score:.3f}")
    
    def update_trailing_offset(self, df: pd.DataFrame):
        if self.adaptive_atr:
            period = self.parameters.atr_period or 14
            atr_val = compute_atr(df.copy(), period)
            if atr_val <= 0:
                self.trailing_offset = self.static_atr_multiplier
            else:
                self.trailing_offset = atr_val * (self.parameters.atr_multiplier or 1.5)
        else:
            self.trailing_offset = self.parameters.atr_multiplier or 2.0
    
    def decision(self, df: pd.DataFrame) -> str:
        return "HOLD"
    
    def execute_trade(self, side: str, quantity: float, price: float,
                      stop_loss: Optional[float] = None, take_profit: Optional[float] = None, trailing: bool = False):
        self.is_open = True
        self.side = side
        self.qty = quantity
        self.entry_price = price
        self.partial_tps = []
        self.stop_loss_price = stop_loss
        self.take_profit_price = take_profit
        self.trailing_stop_enabled = trailing
    
        if side == "BUY":
            self.partial_tps.append({
                "price": price * 1.02,
                "fraction": 0.5,
                "triggered": False
            })
            self.partial_tps.append({
                "price": price * 1.04,
                "fraction": 0.5,
                "triggered": False
            })
        else:
            self.partial_tps.append({
                "price": price * 0.98,
                "fraction": 0.5,
                "triggered": False
            })
            self.partial_tps.append({
                "price": price * 0.96,
                "fraction": 0.5,
                "triggered": False
            })
    
        logger.info(colored(
            f"[{self.symbol}] {side} trade executed @ {price:.2f}, qty={quantity}, "
            f"SL={stop_loss}, TP={take_profit}, trailing={trailing}",
            "blue"
        ))
    
    def close_position(self, reason: str, close_price: float) -> float:
        if not self.is_open:
            return 0.0
    
        if self.side == "BUY":
            pnl = (close_price - self.entry_price) * self.qty
        else:
            pnl = (self.entry_price - close_price) * self.qty
    
        logger.info(colored(
            f"[{self.symbol}] Closing position. reason={reason}, close_price={close_price:.2f}, PnL={pnl:.2f}",
            "yellow"
        ))
        self.is_open = False
        self.side = None
        self.qty = 0.0
        self.entry_price = 0.0
        self.stop_loss_price = None
        self.take_profit_price = None
        self.partial_tps = []
        self.trailing_stop_enabled = False
    
        return pnl
    
    def close_partial_position(self, fraction: float, close_price: float) -> float:
        if not self.is_open or fraction <= 0:
            return 0.0
        close_qty = self.qty * fraction
        if close_qty <= 0:
            return 0.0
    
        if self.side == "BUY":
            pnl = (close_price - self.entry_price) * close_qty
        else:
            pnl = (self.entry_price - close_price) * close_qty
    
        logger.info(colored(
            f"[{self.symbol}] Partial close {fraction*100:.1f}%, close_price={close_price:.2f}, PnL={pnl:.2f}",
            "yellow"
        ))
        self.qty -= close_qty
        if self.qty <= 0:
            self.is_open = False
            self.side = None
            self.entry_price = 0.0
            self.stop_loss_price = None
            self.take_profit_price = None
            self.partial_tps = []
            self.trailing_stop_enabled = False
    
        return pnl
    
    def monitor_positions(self, df: pd.DataFrame) -> float:
        if not self.is_open:
            return 0.0
        current_price = df["close"].iloc[-1]
        realized_pnl = 0.0
    
        if self.stop_loss_price:
            if self.side == "BUY" and current_price <= self.stop_loss_price:
                realized_pnl += self.close_position("StopLoss-BUY", current_price)
                return realized_pnl
            elif self.side == "SELL" and current_price >= self.stop_loss_price:
                realized_pnl += self.close_position("StopLoss-SELL", current_price)
                return realized_pnl
    
        if self.take_profit_price:
            if self.side == "BUY" and current_price >= self.take_profit_price:
                realized_pnl += self.close_position("TakeProfit-BUY", current_price)
                return realized_pnl
            elif self.side == "SELL" and current_price <= self.take_profit_price:
                realized_pnl += self.close_position("TakeProfit-SELL", current_price)
                return realized_pnl
    
        for pt in self.partial_tps:
            if not pt["triggered"]:
                tp_price = pt["price"]
                frac = pt["fraction"]
                if self.side == "BUY" and current_price >= tp_price:
                    realized_pnl += self.close_partial_position(frac, current_price)
                    pt["triggered"] = True
                elif self.side == "SELL" and current_price <= tp_price:
                    realized_pnl += self.close_partial_position(frac, current_price)
                    pt["triggered"] = True
    
        if self.trailing_stop_enabled:
            self.update_trailing_offset(df.copy())
            if self.side == "BUY":
                trail_line = (self.entry_price + (current_price - self.entry_price) - self.trailing_offset)
                if current_price < trail_line:
                    realized_pnl += self.close_position("TrailingStop-BUY", current_price)
            else:
                trail_line = (self.entry_price - (self.entry_price - current_price) - self.trailing_offset)
                if current_price > trail_line:
                    realized_pnl += self.close_position("TrailingStop-SELL", current_price)
        return realized_pnl

class StrategyA(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df) < period:
            return "HOLD"

        # Basic SMA Signal
        sma_val = df["SMA_50"].iloc[-1]
        last_close = df["close"].iloc[-1]
        basic_signal = "BUY" if last_close < sma_val else "SELL"

        # ML Predictions (Simplified for Backtesting)
        # In a real scenario, train the ML models on historical data
        ml_prob = 0.5  # Placeholder
        xgb_prob = 0.5  # Placeholder
        combined_prob = (ml_prob + xgb_prob) / 2

        ml_signal = "HOLD"
        if combined_prob > 0.6:
            ml_signal = "BUY"
        elif combined_prob < 0.4:
            ml_signal = "SELL"

        final_decision = "HOLD"
        if basic_signal == "BUY" and ml_signal == "BUY":
            final_decision = "BUY"
        elif basic_signal == "SELL" and ml_signal == "SELL":
            final_decision = "SELL"

        # Sentiment Adjustment
        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                final_decision = "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                final_decision = "SELL"

        return final_decision

class DivergenceStrategy(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        if len(df) < 20:
            return "HOLD"

        # Calculate RSI and MACD
        rsi = df["RSI"].iloc[-1]
        macd = df["MACD_12_26_9"].iloc[-1]
        signal = df["MACDs_12_26_9"].iloc[-1]

        # Price and Indicator trends
        price_trend = df["close"].iloc[-1] - df["close"].iloc[-2]
        rsi_trend = df["RSI"].iloc[-1] - df["RSI"].iloc[-2]
        macd_trend = macd - signal

        # Detect Bullish Divergence
        if price_trend < 0 and rsi_trend > 0 and macd_trend > 0:
            if rsi < 30 and macd > signal:
                return "BUY"

        # Detect Bearish Divergence
        if price_trend > 0 and rsi_trend < 0 and macd_trend < 0:
            if rsi > 70 and macd < signal:
                return "SELL"

        return "HOLD"

class MomentumStrategy(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        if len(df) < 20:
            return "HOLD"

        rsi = df["RSI"].iloc[-1]
        macd = df["MACD_12_26_9"].iloc[-1]
        signal = df["MACDs_12_26_9"].iloc[-1]
        bb_upper = df["BBU_20_2.0"].iloc[-1]
        bb_lower = df["BBL_20_2.0"].iloc[-1]
        close_price = df["close"].iloc[-1]

        # Overbought/Oversold RSI
        if rsi < 30:
            rsi_signal = "BUY"
        elif rsi > 70:
            rsi_signal = "SELL"
        else:
            rsi_signal = "HOLD"

        # MACD Crossover
        if macd > signal:
            macd_signal = "BUY"
        elif macd < signal:
            macd_signal = "SELL"
        else:
            macd_signal = "HOLD"

        # Bollinger Bands
        if close_price < bb_lower:
            bb_signal = "BUY"
        elif close_price > bb_upper:
            bb_signal = "SELL"
        else:
            bb_signal = "HOLD"

        # Combine Signals
        signals = [rsi_signal, macd_signal, bb_signal]
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")

        if buy_signals >= 2:
            final_decision = "BUY"
        elif sell_signals >= 2:
            final_decision = "SELL"
        else:
            final_decision = "HOLD"

        # Sentiment Adjustment
        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                final_decision = "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                final_decision = "SELL"

        return final_decision

class ArbitrageStrategy(BaseStrategy):
    def __init__(self, symbol: str, related_symbol: str, **kwargs):
        super().__init__(symbol, **kwargs)
        self.related_symbol = related_symbol

    def decision(self, df: pd.DataFrame) -> str:
        # Placeholder for arbitrage logic
        # In backtesting, compare with historical related symbol data
        # For simplicity, we return "HOLD"
        return "HOLD"

# =============================================================================
# BACKTESTING CLASS (continued)
# =============================================================================
# Implement a method to simulate arbitrage trades if needed

# =============================================================================
# MAIN BACKTEST SCRIPT
# =============================================================================
def load_config(config_path: str) -> ConfigModel:
    try:
        with open(config_path, 'r') as f:
            config_json = json.load(f)
        config = ConfigModel(**config_json)
        logger.debug("Configuration loaded and validated successfully.")
        return config
    except FileNotFoundError:
        logger.critical(f"Configuration file '{config_path}' not found.", exc_info=True)
        sys.exit(1)
    except ValidationError as ve:
        logger.critical(f"Configuration validation error: {ve}", exc_info=True)
        sys.exit(1)
    except json.JSONDecodeError as je:
        logger.critical(f"Invalid JSON format in configuration file: {je}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error loading configuration: {e}", exc_info=True)
        sys.exit(1)

def main():
    try:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        config = load_config(config_path)
        logger.debug("Configuration loaded successfully.")

        backtester = Backtester(config)
        backtester.run_backtest()
        backtester.generate_trade_log()
        backtester.plot_results()

    except Exception as e:
        logger.critical(f"Critical error in backtest: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
