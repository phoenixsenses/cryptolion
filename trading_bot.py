#!/usr/bin/env python3
import os
import sys
import json
import logging
import csv
import time
import math
import asyncio
import random
import threading
import concurrent.futures
import pandas as pd
import numpy as np
import requests

# Advanced stats
import scipy.stats as st

from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from termcolor import colored

# Binance imports for production usage (Futures)
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance.streams import BinanceSocketManager

# Pydantic for config
from pydantic import BaseModel, ValidationError, Field

# For .env
from dotenv import load_dotenv

# Optional Sentiment
try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
except ImportError:
    HAVE_TEXTBLOB = False

##############################################################################
# Logging Setup
##############################################################################
logger = logging.getLogger("SuperBotProduction")
logger.setLevel(logging.DEBUG)

json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')

file_handler = RotatingFileHandler("trading_bot.log", maxBytes=10*1024*1024, backupCount=2)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(json_formatter)

console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

##############################################################################
# Pydantic Models for Config
##############################################################################
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
    use_spearman: Optional[bool] = True
    use_exponential_weight: Optional[bool] = False
    # Kelly
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

    class Config:
        populate_by_name = True

class RiskParametersModel(BaseModel):
    DEFAULT_LEVERAGE: int
    BASE_RISK_PER_TRADE: float

class TradeParametersModel(BaseModel):
    stop_loss: float
    take_profit: float
    trailing_stop_loss: bool

class ApiCredentialsModel(BaseModel):
    api_key: str
    secret_key: str

class SymbolModel(BaseModel):
    symbol: str
    base_currency: str
    quote_currency: str
    strategy: str
    strategy_parameters: Optional[StrategyParametersModel] = None
    risk_parameters: RiskParametersModel
    trade_parameters: TradeParametersModel
    api_credentials: ApiCredentialsModel

class ConfigModel(BaseModel):
    MIN_NOTIONAL: float
    STARTING_BALANCE: float
    STRATEGIES: Dict[str, StrategyModel]
    ACTIVE_STRATEGY: str
    SYMBOLS_TO_TRADE: Dict[str, SymbolModel]

    class Config:
        populate_by_name = True

##############################################################################
# Load & Validate Config
##############################################################################
def load_config(config_path: str) -> ConfigModel:
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        logger.info(f"Configuration file '{config_path}' loaded successfully.")
        config = ConfigModel(**data)
        logger.info("Configuration validation passed successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    except ValidationError as e:
        logger.error(f"Config validation error:\n{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading config: {str(e)}")
        sys.exit(1)

##############################################################################
# Utility: ATR Calculation
##############################################################################
def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Computes the Average True Range (ATR) for the last 'period' candles.
    If the dataframe is too short, returns 0.
    """
    if len(df) < period:
        return 0.0

    for col in ["high", "low", "close"]:
        df[col] = df[col].astype(float)

    df['previous_close'] = df['close'].shift(1)
    df['high_low'] = df['high'] - df['low']
    df['high_pc'] = abs(df['high'] - df['previous_close'])
    df['low_pc'] = abs(df['low'] - df['previous_close'])
    df['tr'] = df[['high_low', 'high_pc', 'low_pc']].max(axis=1)

    df['atr'] = df['tr'].rolling(period).mean()
    latest_atr = df['atr'].iloc[-1]

    # Cleanup
    df.drop(['previous_close', 'high_low', 'high_pc', 'low_pc', 'tr', 'atr'], axis=1, inplace=True, errors='ignore')

    return latest_atr if not np.isnan(latest_atr) else 0.0

##############################################################################
# Utility: Basic Sentiment Analysis
##############################################################################
def analyze_sentiment(text: str) -> float:
    """
    Simple sentiment analyzer using TextBlob.
    Returns polarity in range [-1.0, 1.0].
    """
    if not HAVE_TEXTBLOB:
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity

def fetch_twitter_sentiment(symbol: str, count: int = 50) -> float:
    """
    Placeholder method that simulates fetching tweets and analyzing.
    In real usage, integrate Twitter API, etc.
    Returns average sentiment polarity across tweets.
    """
    # Example: here we just generate random sentiments for demonstration.
    # Replace with actual logic, e.g., Tweepy or search from a Twitter API endpoint.
    sentiments = []
    for _ in range(count):
        random_text = f"Random tweet about {symbol} - sentiment test."
        sentiments.append(analyze_sentiment(random_text))
    if len(sentiments) == 0:
        return 0.0
    return float(np.mean(sentiments))

def fetch_news_sentiment(symbol: str) -> float:
    """
    Placeholder method that simulates news-based sentiment analysis.
    In real usage, scrape or query a news API, parse, then run `analyze_sentiment`.
    """
    random_news_text = f"News headline about {symbol} moving markets."
    return analyze_sentiment(random_news_text)

def combined_sentiment_score(symbol: str, config: SentimentAnalysisModel) -> float:
    """
    Aggregates various data sources (Twitter, news, etc.) into a single sentiment score.
    """
    if not config.ENABLED:
        return 0.0

    total_score = 0.0
    sources_used = 0

    if "twitter" in config.DATA_SOURCE:
        s = fetch_twitter_sentiment(symbol, config.TWEETS_PER_SYMBOL)
        total_score += s
        sources_used += 1

    if "news" in config.DATA_SOURCE:
        s = fetch_news_sentiment(symbol)
        total_score += s
        sources_used += 1

    if sources_used == 0:
        return 0.0
    return total_score / sources_used

##############################################################################
# Base Strategy
##############################################################################
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
        self.partial_tps = []
        self.parameters = parameters or StrategyParametersModel()
        self.adaptive_atr = adaptive_atr
        self.static_atr_multiplier = atr_multiplier
        self.trailing_offset = self.static_atr_multiplier
        self.sentiment_config = sentiment_config
        self.sentiment_score = 0.0

    def fetch_sentiment(self):
        """
        Fetch and store sentiment score for this strategy.
        Called before decision to incorporate sentiment if desired.
        """
        if self.sentiment_config:
            self.sentiment_score = combined_sentiment_score(self.symbol, self.sentiment_config)
            logger.info(colored(
                f"[{self.symbol}] Sentiment => {self.sentiment_score:.3f}",
                "cyan"
            ))

    def decision(self, df: pd.DataFrame) -> str:
        """
        Return 'BUY', 'SELL', or 'HOLD'. Child classes override for logic.
        Incorporate sentiment if relevant.
        """
        return "HOLD"

    def update_trailing_offset(self, df: pd.DataFrame):
        """
        Dynamically compute trailing offset if ADAPTIVE_ATR = True, else use static multiplier.
        """
        if self.adaptive_atr:
            period = self.parameters.atr_period or 14
            atr_val = compute_atr(df.copy(), period=period)
            if atr_val <= 0:
                self.trailing_offset = self.static_atr_multiplier
            else:
                self.trailing_offset = atr_val * (self.parameters.atr_multiplier or 1.5)
        else:
            self.trailing_offset = self.parameters.atr_multiplier or 2.0

    def execute_trade(self, side: str, quantity: float, price: float):
        self.is_open = True
        self.side = side
        self.qty = quantity
        self.entry_price = price
        self.partial_tps = []
        logger.info(colored(
            f"[{self.symbol}] {side} trade executed @ {price:.2f}, qty={quantity}",
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
        self.partial_tps = []
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
            self.partial_tps = []
        return pnl

    def monitor_positions(self, df: pd.DataFrame) -> float:
        """
        Check partial TPs, trailing stops, etc. Returns realized PnL from partial closes.
        """
        if not self.is_open:
            return 0.0

        current_price = df["close"].iloc[-1]
        realized_pnl = 0.0

        # partial TPs
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

        # Update trailing offset (if adaptive)
        self.update_trailing_offset(df.copy())

        # trailing offset logic
        if self.side == "BUY":
            # Suppose the trailing line is current highest price minus trailing offset
            # (For simplicity, let's assume the 'highest high' since the entry was the last close)
            # This can be extended for a true rolling high since position open.
            trail_line = self.entry_price + (current_price - self.entry_price) - self.trailing_offset
            if current_price < trail_line:
                realized_pnl += self.close_position("TrailingStop-BUY", current_price)
        else:  # SELL
            trail_line = self.entry_price - (self.entry_price - current_price) - self.trailing_offset
            if current_price > trail_line:
                realized_pnl += self.close_position("TrailingStop-SELL", current_price)

        return realized_pnl

##############################################################################
# Example Strategies
##############################################################################
class StrategyA(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        """
        SMA-based approach:
        - BUY if last close < SMA(period)
        - SELL if last close >= SMA(period)
        Incorporates sentiment thresholds if sentiment is enabled.
        """
        self.fetch_sentiment()  # Fetch sentiment before deciding
        period = self.parameters.sma_period or 20
        if len(df) < period:
            return "HOLD"

        sma = df["close"].rolling(period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]

        # Basic signal
        basic_signal = "BUY" if last_close < sma else "SELL"

        # Incorporate sentiment:
        # if sentiment is significantly negative, prefer SELL,
        # if sentiment is significantly positive, prefer BUY,
        # else fallback to basic_signal.
        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                return "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                return "SELL"

        return basic_signal

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
        # partial TPs
        if side == "BUY":
            self.partial_tps = [
                {"price": price + 2.0, "fraction": 0.5, "triggered": False},
                {"price": price + 4.0, "fraction": 0.5, "triggered": False},
            ]
        else:
            self.partial_tps = [
                {"price": price - 2.0, "fraction": 0.5, "triggered": False},
                {"price": price - 4.0, "fraction": 0.5, "triggered": False},
            ]

class StrategyB(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        """
        Momentum-based approach:
        - BUY if last close > last open
        - SELL if last close < last open
        Ignores sentiment in this basic example.
        """
        self.fetch_sentiment()  # fetch but not used in logic
        if len(df) == 0:
            return "HOLD"
        last_close = df["close"].iloc[-1]
        last_open = df["open"].iloc[-1]
        if last_close > last_open:
            return "BUY"
        else:
            return "SELL"

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
        if side == "BUY":
            self.partial_tps = [
                {"price": price + 1.0, "fraction": 0.5, "triggered": False},
                {"price": price + 2.0, "fraction": 0.5, "triggered": False},
            ]
        else:
            self.partial_tps = [
                {"price": price - 1.0, "fraction": 0.5, "triggered": False},
                {"price": price - 2.0, "fraction": 0.5, "triggered": False},
            ]

class StrategyBreakout(BaseStrategy):
    """
    Breakout Strategy:
    - BUY on breakout above recent high
    - SELL on breakout below recent low
    """
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df) < period:
            return "HOLD"

        highest_high = df["high"].rolling(period).max().iloc[-1]
        lowest_low = df["low"].rolling(period).min().iloc[-1]
        last_close = df["close"].iloc[-1]
        if len(df) > 1:
            previous_close = df["close"].iloc[-2]
        else:
            previous_close = last_close

        if last_close > highest_high and previous_close <= highest_high:
            return "BUY"
        elif last_close < lowest_low and previous_close >= lowest_low:
            return "SELL"
        else:
            return "HOLD"

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
        if side == "BUY":
            self.partial_tps = [
                {"price": price + 3.0, "fraction": 0.4, "triggered": False},
                {"price": price + 6.0, "fraction": 0.3, "triggered": False},
                {"price": price + 9.0, "fraction": 0.3, "triggered": False},
            ]
        else:
            self.partial_tps = [
                {"price": price - 3.0, "fraction": 0.4, "triggered": False},
                {"price": price - 6.0, "fraction": 0.3, "triggered": False},
                {"price": price - 9.0, "fraction": 0.3, "triggered": False},
            ]

class StrategyCorrelated(BaseStrategy):
    """
    Spearman or Pearson correlation, with optional exponential weighting.
    Inverts basic SMA-based signal if abs(corr) > threshold.
    """
    def __init__(self,
                 symbol: str,
                 parameters: Optional[StrategyParametersModel] = None,
                 adaptive_atr: bool = False,
                 atr_multiplier: float = 1.5,
                 sentiment_config: Optional[SentimentAnalysisModel] = None):
        super().__init__(symbol, parameters, adaptive_atr, atr_multiplier, sentiment_config)
        self.reference_symbol = self.parameters.reference_symbol or "BTCUSDT"
        self.correlation_threshold = self.parameters.correlation_threshold or 0.7
        self.use_spearman = self.parameters.use_spearman
        self.use_exponential_weight = self.parameters.use_exponential_weight

    def compute_rolling_correlation(self, df_main: pd.DataFrame, df_ref: pd.DataFrame, window: int) -> float:
        if len(df_main) < window or len(df_ref) < window:
            return 0.0

        m = df_main["close"].iloc[-window:].reset_index(drop=True)
        r = df_ref["close"].iloc[-window:].reset_index(drop=True)

        if self.use_exponential_weight:
            # Weighted correlation
            weights = np.exp(-0.1 * np.arange(window))[::-1]
            weights /= weights.sum()

            if self.use_spearman:
                m_rank = m.rank()
                r_rank = r.rank()
                cov = np.sum(weights * (m_rank - np.average(m_rank, weights=weights)) \
                                       * (r_rank - np.average(r_rank, weights=weights)))
                var_m = np.sum(weights * (m_rank - np.average(m_rank, weights=weights))**2)
                var_r = np.sum(weights * (r_rank - np.average(r_rank, weights=weights))**2)
            else:
                cov = np.sum(weights * (m - np.average(m, weights=weights)) \
                                     * (r - np.average(r, weights=weights)))
                var_m = np.sum(weights * (m - np.average(m, weights=weights))**2)
                var_r = np.sum(weights * (r - np.average(r, weights=weights))**2)

            if var_m == 0 or var_r == 0:
                return 0.0
            corr = cov / np.sqrt(var_m * var_r)
            if np.isnan(corr):
                return 0.0
            return corr
        else:
            # Unweighted correlation
            if self.use_spearman:
                corr_val = st.spearmanr(m, r).correlation
            else:
                corr_matrix = np.corrcoef(m, r)
                corr_val = corr_matrix[0,1] if corr_matrix.shape == (2,2) else 0.0
            if corr_val is None or np.isnan(corr_val):
                return 0.0
            return corr_val

    def decision(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df) < period or len(ref_df) < period:
            return "HOLD"

        # Basic SMA-based signal
        sma = df["close"].rolling(period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        basic_signal = "BUY" if last_close < sma else "SELL"

        # Sentiment override check
        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                basic_signal = "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                basic_signal = "SELL"

        # Compute correlation
        corr = self.compute_rolling_correlation(df, ref_df, period)
        logger.info(colored(
            f"[{self.symbol}] correlation with {self.reference_symbol}={corr:.2f} (window={period})",
            "magenta"
        ))

        # If correlation above threshold => invert signal
        if abs(corr) >= self.correlation_threshold:
            inverted_signal = "SELL" if basic_signal == "BUY" else "BUY"
            logger.info(colored(
                f"[{self.symbol}] abs(corr)={abs(corr):.2f} >= {self.correlation_threshold}, invert {basic_signal}->{inverted_signal}",
                "magenta"
            ))
            return inverted_signal
        else:
            return basic_signal

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
        # partial TPs
        if side == "BUY":
            self.partial_tps = [
                {"price": price + 2.0, "fraction": 0.5, "triggered": False},
                {"price": price + 5.0, "fraction": 0.5, "triggered": False},
            ]
        else:
            self.partial_tps = [
                {"price": price - 2.0, "fraction": 0.5, "triggered": False},
                {"price": price - 5.0, "fraction": 0.5, "triggered": False},
            ]

##############################################################################
# TradingBot Class
##############################################################################
class TradingBot:
    def __init__(self, config: ConfigModel):
        self.config = config

        # Load .env
        load_dotenv()
        # We'll override the symbol credentials with .env if present
        # Or we rely on config's embedded credentials, if .env not specified

        self.client = self.initialize_binance_client(config)
        self.stop_bot = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.loop = asyncio.get_event_loop()

        self.strategy_map = {
            "StrategyA": StrategyA,
            "StrategyB": StrategyB,
            "StrategyBreakout": StrategyBreakout,
            "StrategyCorrelated": StrategyCorrelated,
        }

        self.strategies_for_symbols = {}
        for sym_key, sym_cfg in config.SYMBOLS_TO_TRADE.items():
            strat_cls = self.strategy_map.get(sym_cfg.strategy, StrategyA)

            global_strat_cfg = config.STRATEGIES.get(sym_cfg.strategy)
            adaptive_atr = global_strat_cfg.ADAPTIVE_ATR if global_strat_cfg else False
            atr_multiplier = global_strat_cfg.ATR_MULTIPLIER if global_strat_cfg else 1.5
            sentiment_model = global_strat_cfg.SENTIMENT_ANALYSIS if global_strat_cfg else None

            self.strategies_for_symbols[sym_key] = strat_cls(
                symbol=sym_key,
                parameters=sym_cfg.strategy_parameters,
                adaptive_atr=adaptive_atr,
                atr_multiplier=atr_multiplier,
                sentiment_config=sentiment_model
            )

        self.bsm = BinanceSocketManager(self.client)
        self.streams = {}
        self.order_books = {}
        self.current_balance = config.STARTING_BALANCE

    def initialize_binance_client(self, config: ConfigModel) -> Client:
        # Fallback to .env if set, else use the first symbol config
        first_symbol = next(iter(config.SYMBOLS_TO_TRADE))
        sym_cfg = config.SYMBOLS_TO_TRADE[first_symbol].api_credentials

        env_api_key = os.getenv("BINANCE_API_KEY", sym_cfg.api_key)
        env_secret_key = os.getenv("BINANCE_SECRET_KEY", sym_cfg.secret_key)

        client = Client(env_api_key, env_secret_key)
        try:
            server_time = client.get_server_time()
            logger.info(f"Binance Server Time: {server_time}")
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch server time: {e}")
            sys.exit(1)
        return client

    def fetch_historical_data(self, symbol: str, interval: str = "1m", lookback: int = 100) -> pd.DataFrame:
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
            data = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            data["open_time"] = pd.to_datetime(data["open_time"], unit='ms')
            data.set_index("open_time", inplace=True)
            numeric_cols = ["open","high","low","close","volume","quote_asset_volume","taker_buy_base_asset_volume","taker_buy_quote_asset_volume"]
            for c in numeric_cols:
                data[c] = data[c].astype(float)
            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        """
        Place a real market order on Binance Futures.
        Adjust for testnet if you want to test with fake funds.
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(colored(f"[RealOrder] Placing {side} order on {symbol}, qty={quantity:.4f}", "cyan"))
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API Exception placing order on {symbol}: {e}")
            return {}
        except BinanceOrderException as e:
            logger.error(f"Binance Order Exception placing order on {symbol}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error placing order on {symbol}: {e}")
            return {}

    def monitor_order(self, order_id: int, symbol: str) -> bool:
        """
        Monitor an order until it is FILLED, CANCELED, or REJECTED.
        """
        try:
            while True:
                order = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                if order['status'] in ['FILLED', 'CANCELED', 'REJECTED']:
                    logger.info(colored(f"[RealOrder] Order {order_id} => {order['status']}", "cyan"))
                    return order['status'] == 'FILLED'
                time.sleep(1)
        except BinanceAPIException as e:
            logger.error(f"Binance API Exception monitoring order on {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error monitoring order on {symbol}: {e}")
            return False

    async def run_continuous_trading(self, symbols: List[str]):
        tasks = []
        for sym in symbols:
            t = asyncio.create_task(self.handle_symbol(sym))
            tasks.append(t)
        await asyncio.gather(*tasks)

    async def handle_symbol(self, symbol: str):
        """
        Main method for real-time trading. 
        - Grabs historical data
        - Opens websockets for real-time ticker + depth
        - Makes decisions
        - Places/monitors orders
        """
        df = self.fetch_historical_data(symbol, "1m", 200)
        strategy = self.strategies_for_symbols[symbol]

        # If correlated, fetch reference symbol data
        ref_df = None
        if isinstance(strategy, StrategyCorrelated):
            ref_symbol = strategy.reference_symbol
            ref_df = self.fetch_historical_data(ref_symbol, "1m", 200)

        # Start websockets
        ticker_socket = self.bsm.symbol_ticker_socket(symbol)
        book_socket = self.bsm.depth_socket(symbol)

        async with ticker_socket as ts, book_socket as tb:
            logger.info(colored(f"Started WebSocket for {symbol}", "green"))
            while not self.stop_bot:
                try:
                    res = await ts.recv()
                    book_res = await tb.recv()

                    new_price = float(res['c'])  # close
                    # push row to df
                    df = df.append({
                        'open': float(res['o']),
                        'high': float(res['h']),
                        'low':  float(res['l']),
                        'close': new_price
                    }, ignore_index=True)
                    if len(df) > 300:
                        df = df.iloc[-300:]

                    # Similarly for ref_df if correlated
                    if ref_df is not None:
                        # For demonstration, we just replicate main price or fetch real data
                        ref_data = self.fetch_historical_data(strategy.reference_symbol, "1m", 2)
                        if not ref_data.empty:
                            ref_df = pd.concat([ref_df, ref_data.iloc[-1:]])
                            if len(ref_df) > 300:
                                ref_df = ref_df.iloc[-300:]

                    self.order_books[symbol] = book_res

                    # Decision
                    if isinstance(strategy, StrategyCorrelated) and ref_df is not None:
                        action = strategy.decision(df.copy(), ref_df.copy())
                    else:
                        action = strategy.decision(df.copy())

                    logger.info(colored(f"[{symbol}] Decision => {action}", "green"))

                    # If we have a signal to open a trade
                    if action in ("BUY", "SELL") and not strategy.is_open:
                        last_close = df["close"].iloc[-1]
                        qty = self.calculate_kelly_quantity(symbol, last_close)
                        if qty <= 0:
                            logger.warning(f"[{symbol}] Kelly-based qty=0 => skip trade.")
                            await asyncio.sleep(1)
                            continue

                        strategy.execute_trade(action, qty, last_close)
                        order = self.place_order(symbol, action, qty)
                        if order and 'orderId' in order:
                            filled = self.monitor_order(order['orderId'], symbol)
                            if not filled:
                                logger.warning(f"[{symbol}] Order {order['orderId']} not filled, revert pos.")
                                strategy.close_position("OrderNotFilled", last_close)

                    # If open, check partial closes, trailing stops
                    if strategy.is_open:
                        realized = strategy.monitor_positions(df.copy())
                        if realized != 0:
                            logger.info(colored(f"[{symbol}] Realized PnL={realized:.2f}", "magenta"))
                            self.current_balance += realized

                except Exception as e:
                    logger.error(colored(f"Error in WebSocket for {symbol}: {e}", "red"))
                    await asyncio.sleep(5)

    def calculate_kelly_quantity(self, symbol: str, current_price: float) -> float:
        """
        Kelly Criterion-based approach. 
        Kelly fraction = p - ( (1 - p) / b )
          where p=win_rate, b=win_loss_ratio
        Caps Kelly fraction at 0.25
        Applies leverage from config.
        """
        strategy = self.strategies_for_symbols[symbol]
        # fallback if not specified
        p = strategy.parameters.kelly_win_rate or 0.5
        b = strategy.parameters.kelly_win_loss_ratio or 1.0
        kelly_fraction = p - ((1 - p) / b)
        if kelly_fraction < 0:
            kelly_fraction = 0.0
        kelly_fraction = min(kelly_fraction, 0.25)

        # position notional
        notional = kelly_fraction * self.current_balance

        # apply leverage
        leverage = self.config.SYMBOLS_TO_TRADE[symbol].risk_parameters.DEFAULT_LEVERAGE
        notional *= leverage

        qty = notional / current_price

        # check min notional
        if (qty * current_price) < self.config.MIN_NOTIONAL:
            logger.warning(f"[{symbol}] Kelly => qty {qty:.4f} * price {current_price:.2f} < min notional => 0")
            return 0.0

        return float(round(qty, 6))

    def shutdown(self):
        logger.info(colored("Shutting down SuperBot gracefully...", "blue"))
        self.stop_bot = True
        self.bsm.close()
        self.executor.shutdown(wait=False)
        logger.info(colored("SuperBot shut down.", "blue"))

##############################################################################
# Main
##############################################################################
import signal

def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    trading_bot = TradingBot(config)
    symbols_to_trade = list(config.SYMBOLS_TO_TRADE.keys())

    def signal_handler(sig, frame):
        logger.info(colored(f"Received signal {sig}, shutting down bot...", "yellow"))
        trading_bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(trading_bot.run_continuous_trading(symbols_to_trade))
    except Exception as e:
        logger.error(colored(f"Unexpected error in main loop: {e}", 'red'))
    finally:
        loop.close()
        trading_bot.shutdown()

if __name__ == "__main__":
    main()
