#!/usr/bin/env python3
import os
import sys
import json
import logging
import time
import asyncio
import random
import concurrent.futures
import signal

# Data & ML
import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Statistics, Sentiment
import scipy.stats as st
try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
except ImportError:
    HAVE_TEXTBLOB = False

# Typing
from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from termcolor import colored

# ─────────────────────────────────────────────────────────────────────────────
# BINANCE IMPORTS (for python-binance 1.0.x)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from binance import BinanceSocketManager
    from binance.client import Client
    from binance.enums import *
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    SOCKET_MANAGER_SOURCE = "binance"
    WEBSOCKET_STYLE = "1.0.x (callback-based)"
except ImportError as e:
    raise ImportError(
        f"Could not import BinanceSocketManager from binance. "
        f"Please ensure you have 'python-binance' 1.0.x installed.\n{e}"
    )

# Pydantic
from pydantic import BaseModel, ValidationError, Field

# .env support
from dotenv import load_dotenv

# Extra: Check python-binance version for debug
try:
    import pkg_resources
    BINANCE_VERSION = pkg_resources.get_distribution("python-binance").version
except Exception:
    BINANCE_VERSION = "unknown"

##############################################################################
# Logging Setup
##############################################################################
logger = logging.getLogger("SuperBotProduction")
logger.setLevel(logging.DEBUG)  # show DEBUG logs

json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')

file_handler = RotatingFileHandler("trading_bot.log", maxBytes=10*1024*1024, backupCount=2)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(json_formatter)

console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.debug(f"Detected python-binance version: {BINANCE_VERSION}")
logger.debug(f"Imported BinanceSocketManager from: {SOCKET_MANAGER_SOURCE} — {WEBSOCKET_STYLE}")

##############################################################################
# Pydantic Models
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
    USE_TESTNET: Optional[bool] = False

    class Config:
        populate_by_name = True

##############################################################################
# Configuration Loading
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
    if len(df) < period:
        return 0.0
    for col in ["high", "low", "close"]:
        df[col] = df[col].astype(float)

    df["previous_close"] = df["close"].shift(1)
    df["high_low"] = df["high"] - df["low"]
    df["high_pc"] = abs(df["high"] - df["previous_close"])
    df["low_pc"] = abs(df["low"] - df["previous_close"])
    df["tr"] = df[["high_low", "high_pc", "low_pc"]].max(axis=1)

    df["atr"] = df["tr"].rolling(period).mean()
    latest_atr = df["atr"].iloc[-1] if len(df) >= period else 0.0

    df.drop(
        ["previous_close", "high_low", "high_pc", "low_pc", "tr", "atr"],
        axis=1, inplace=True, errors="ignore"
    )
    return latest_atr if not np.isnan(latest_atr) else 0.0

##############################################################################
# Utility: Sentiment
##############################################################################
def analyze_sentiment(text: str) -> float:
    if not HAVE_TEXTBLOB:
        return 0.0
    blob = TextBlob(text)
    return blob.sentiment.polarity

def fetch_twitter_sentiment(symbol: str, count: int = 50) -> float:
    sentiments = []
    for _ in range(count):
        random_text = f"Random tweet about {symbol} - sentiment test."
        sentiments.append(analyze_sentiment(random_text))
    if len(sentiments) == 0:
        return 0.0
    return float(np.mean(sentiments))

def fetch_news_sentiment(symbol: str) -> float:
    random_news_text = f"News headline about {symbol} possibly moving markets."
    return analyze_sentiment(random_news_text)

def combined_sentiment_score(symbol: str, config: SentimentAnalysisModel) -> float:
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
# 1) Enhanced Risk Management
##############################################################################
class RiskManager:
    def __init__(self, config: ConfigModel):
        self.config = config
        self.position_metrics = {}
        self.correlation_matrix = pd.DataFrame()

    def calculate_volatility_position_size(self, symbol: str, df: pd.DataFrame) -> float:
        """
        ATR-based position sizing:
         - If ATR=0 or insufficient data, returns 0.
         - Scales position based on ATR * multiplier vs. risk capital.
        """
        if symbol not in self.config.SYMBOLS_TO_TRADE:
            return 0.0
        sym_cfg = self.config.SYMBOLS_TO_TRADE[symbol]
        sp = sym_cfg.strategy_parameters or StrategyParametersModel()

        atr_period = sp.atr_period or 14
        atr_val = compute_atr(df.copy(), period=atr_period)
        if atr_val == 0:
            return 0.0

        risk_amount = self.config.STARTING_BALANCE * sym_cfg.risk_parameters.BASE_RISK_PER_TRADE
        multiplier = sp.atr_multiplier or 1.5
        dollar_risk = risk_amount / (atr_val * multiplier)

        if len(df) == 0:
            return 0.0
        current_price = df["close"].iloc[-1]
        if current_price == 0:
            return 0.0

        return dollar_risk / current_price

    def update_correlation_matrix(self, all_data: Dict[str, pd.DataFrame]):
        """
        Example method to keep track of symbol correlations across the portfolio.
        """
        closes_dict = {}
        for sym, data in all_data.items():
            if len(data) > 0 and "close" in data.columns:
                closes_dict[sym] = data["close"]
        if not closes_dict:
            return

        closes = pd.DataFrame(closes_dict).dropna()
        self.correlation_matrix = closes.pct_change().corr()

##############################################################################
# 2) Machine Learning Integration
##############################################################################
class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                df[col] = 0.0
        df["RSI"] = ta.rsi(df["close"], length=14)
        df["OBV"] = ta.obv(df["close"], df["volume"])
        df["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["EMA_20"] = ta.ema(df["close"], length=20)
        return df

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        if df.empty or "close" not in df.columns:
            return
        features_cols = ["RSI", "OBV", "VWAP", "SMA_50", "EMA_20"]
        df = df.dropna(subset=features_cols)
        if len(df) < 2:
            return
        features = df[features_cols]
        shifted_close = df["close"].shift(-1)
        labels = np.where(shifted_close > df["close"], 1, 0)
        labels = labels[:-1]
        features = features.iloc[:-1]

        if len(features) < 10:
            return
        scaled_features = self.scaler.fit_transform(features)
        self.model.fit(scaled_features, labels)
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained:
            return 0.5
        features_cols = ["RSI", "OBV", "VWAP", "SMA_50", "EMA_20"]
        df = df.dropna(subset=features_cols)
        if len(df) == 0:
            return 0.5

        latest = df.iloc[-1][features_cols].values.reshape(1, -1)
        scaled = self.scaler.transform(latest)
        prob_up = self.model.predict_proba(scaled)[0][1]
        return prob_up

##############################################################################
# 3) Multi-Timeframe Analysis
##############################################################################
class MultiTimeframeAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.higher_timeframe_data = pd.DataFrame()

    def update_higher_timeframe(self, df: pd.DataFrame, timeframe: str = "15T"):
        if df.empty:
            return
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            if "open_time" in df.columns:
                df.set_index("open_time", inplace=True)
            else:
                return
        resampled = df.resample(timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna(how="any")
        self.higher_timeframe_data = pd.concat(
            [self.higher_timeframe_data, resampled]
        ).drop_duplicates().sort_index().tail(500)

    def get_trend_bias(self) -> str:
        if len(self.higher_timeframe_data) < 200:
            return "neutral"
        sma_50 = self.higher_timeframe_data["close"].rolling(50).mean().iloc[-1]
        sma_200 = self.higher_timeframe_data["close"].rolling(200).mean().iloc[-1]
        if pd.isna(sma_50) or pd.isna(sma_200):
            return "neutral"
        if sma_50 > sma_200:
            return "bullish"
        else:
            return "bearish"

##############################################################################
# 4) Order Flow Analysis
##############################################################################
class OrderFlowAnalyzer:
    def analyze_depth(self, order_book: dict) -> Dict[str, float]:
        if not order_book or "bids" not in order_book or "asks" not in order_book:
            return {
                "bid_ask_ratio": 1.0,
                "depth_imbalance": 0.0,
                "vwap_bid": 0.0,
                "vwap_ask": 0.0
            }
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        if len(bids) == 0 or len(asks) == 0:
            return {
                "bid_ask_ratio": 1.0,
                "depth_imbalance": 0.0,
                "vwap_bid": 0.0,
                "vwap_ask": 0.0
            }
        bids_float = [(float(p), float(q)) for p, q in bids]
        asks_float = [(float(p), float(q)) for p, q in asks]

        total_bid_qty = sum(q for _, q in bids_float)
        total_ask_qty = sum(q for _, q in asks_float)

        if total_ask_qty == 0:
            ratio = 999.0
        else:
            ratio = total_bid_qty / total_ask_qty

        depth_imb = sum(q for _, q in bids_float[:5]) - sum(q for _, q in asks_float[:5])

        vwap_bid = 0.0
        if total_bid_qty > 0:
            vwap_bid = sum(p * q for p, q in bids_float) / total_bid_qty
        vwap_ask = 0.0
        if total_ask_qty > 0:
            vwap_ask = sum(p * q for p, q in asks_float) / total_ask_qty

        return {
            "bid_ask_ratio": ratio,
            "depth_imbalance": depth_imb,
            "vwap_bid": vwap_bid,
            "vwap_ask": vwap_ask
        }

##############################################################################
# 5) Market Regime Detection
##############################################################################
class MarketRegimeDetector:
    def __init__(self):
        self.volatility_window = 20

    def determine_regime(self, df: pd.DataFrame) -> str:
        if len(df) < self.volatility_window:
            return "neutral"
        returns = df["close"].pct_change().dropna()
        if len(returns) < self.volatility_window:
            return "neutral"

        volatility = returns.rolling(self.volatility_window).std().iloc[-1]
        trend_strength = self._calculate_trend_strength(df)

        if volatility > 0.05:
            return "high_volatility"
        elif trend_strength > 0.8:
            return "strong_trend"
        elif trend_strength < 0.2:
            return "range_bound"
        return "neutral"

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        if len(df) < 200:
            return 0.0
        sma_50 = df["close"].rolling(50).mean()
        sma_200 = df["close"].rolling(200).mean()
        diff = (sma_50 - sma_200).dropna()
        if diff.empty or df["close"].std() == 0:
            return 0.0
        return abs(diff).mean() / df["close"].std()

##############################################################################
# Example Strategies
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
            logger.debug(
                colored(f"[{self.symbol}] Sentiment => {self.sentiment_score:.3f}", "cyan")
            )

    def decision(self, df: pd.DataFrame) -> str:
        return "HOLD"

    def update_trailing_offset(self, df: pd.DataFrame):
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
            f"[{self.symbol}] Closing position. reason={reason}, "
            f"close_price={close_price:.2f}, PnL={pnl:.2f}",
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
            f"[{self.symbol}] Partial close {fraction*100:.1f}%, "
            f"close_price={close_price:.2f}, PnL={pnl:.2f}",
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
        if not self.is_open:
            return 0.0
        current_price = df["close"].iloc[-1]
        realized_pnl = 0.0

        # Check partial TPs
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

        self.update_trailing_offset(df.copy())
        # Trailing logic
        if self.side == "BUY":
            trail_line = (self.entry_price
                          + (current_price - self.entry_price)
                          - self.trailing_offset)
            if current_price < trail_line:
                realized_pnl += self.close_position("TrailingStop-BUY", current_price)
        else:  # SELL
            trail_line = (self.entry_price
                          - (self.entry_price - current_price)
                          - self.trailing_offset)
            if current_price > trail_line:
                realized_pnl += self.close_position("TrailingStop-SELL", current_price)

        return realized_pnl

class StrategyA(BaseStrategy):
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df) < period:
            return "HOLD"
        sma_val = df["close"].rolling(period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        basic_signal = "BUY" if last_close < sma_val else "SELL"

        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                return "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                return "SELL"
        return basic_signal

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
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

class StrategyCorrelated(BaseStrategy):
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

    def compute_rolling_correlation(self, df_main: pd.DataFrame,
                                    df_ref: pd.DataFrame,
                                    window: int) -> float:
        if len(df_main) < window or len(df_ref) < window:
            return 0.0
        m = df_main["close"].iloc[-window:].reset_index(drop=True)
        r = df_ref["close"].iloc[-window:].reset_index(drop=True)

        if self.use_exponential_weight:
            weights = np.exp(-0.1 * np.arange(window))[::-1]
            weights /= weights.sum()
            if self.use_spearman:
                m_rank = m.rank()
                r_rank = r.rank()
                cov = np.sum(
                    weights * (m_rank - np.average(m_rank, weights=weights)) *
                    (r_rank - np.average(r_rank, weights=weights))
                )
                var_m = np.sum(weights * (m_rank - np.average(m_rank, weights=weights))**2)
                var_r = np.sum(weights * (r_rank - np.average(r_rank, weights=weights))**2)
            else:
                cov = np.sum(
                    weights * (m - np.average(m, weights=weights)) *
                    (r - np.average(r, weights=weights))
                )
                var_m = np.sum(weights * (m - np.average(m, weights=weights))**2)
                var_r = np.sum(weights * (r - np.average(r, weights=weights))**2)

            if var_m == 0 or var_r == 0:
                return 0.0
            corr = cov / np.sqrt(var_m * var_r)
            if np.isnan(corr):
                return 0.0
            return corr
        else:
            if self.use_spearman:
                corr_val = st.spearmanr(m, r).correlation
            else:
                corr_matrix = np.corrcoef(m, r)
                if corr_matrix.shape == (2, 2):
                    corr_val = corr_matrix[0, 1]
                else:
                    corr_val = 0.0
            if corr_val is None or np.isnan(corr_val):
                return 0.0
            return corr_val

    def decision(self, df: pd.DataFrame, ref_df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df) < period or len(ref_df) < period:
            return "HOLD"

        sma_val = df["close"].rolling(period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        basic_signal = "BUY" if last_close < sma_val else "SELL"

        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                basic_signal = "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                basic_signal = "SELL"

        corr = self.compute_rolling_correlation(df, ref_df, period)
        logger.debug(colored(
            f"[{self.symbol}] correlation with {self.reference_symbol}={corr:.2f} (window={period})",
            "magenta"
        ))

        if abs(corr) >= self.correlation_threshold:
            inverted_signal = "SELL" if basic_signal == "BUY" else "BUY"
            logger.debug(colored(
                f"[{self.symbol}] abs(corr)={abs(corr):.2f} >= "
                f"{self.correlation_threshold}, invert {basic_signal}->{inverted_signal}",
                "magenta"
            ))
            return inverted_signal
        else:
            return basic_signal

##############################################################################
# BaseTradingBot
##############################################################################
class BaseTradingBot:
    def __init__(self, config: ConfigModel):
        self.config = config
        load_dotenv()
        self.client = self.initialize_binance_client(config)
        self.stop_bot = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        self.strategy_map = {
            "StrategyA": StrategyA,
            "StrategyCorrelated": StrategyCorrelated,
        }
        self.strategies_for_symbols = {}
        for sym_key, sym_cfg in config.SYMBOLS_TO_TRADE.items():
            strat_cls = self.strategy_map.get(sym_cfg.strategy, StrategyA)
            global_strat_cfg = config.STRATEGIES.get(sym_cfg.strategy)
            if global_strat_cfg:
                adaptive_atr = global_strat_cfg.ADAPTIVE_ATR
                atr_multiplier = global_strat_cfg.ATR_MULTIPLIER
                sentiment_model = global_strat_cfg.SENTIMENT_ANALYSIS
            else:
                adaptive_atr = False
                atr_multiplier = 1.5
                sentiment_model = None

            self.strategies_for_symbols[sym_key] = strat_cls(
                symbol=sym_key,
                parameters=sym_cfg.strategy_parameters,
                adaptive_atr=adaptive_atr,
                atr_multiplier=atr_multiplier,
                sentiment_config=sentiment_model
            )

        # For python-binance 1.0.x, create *one* socket manager
        self.bsm = BinanceSocketManager(self.client)

        self.order_books = {}
        self.current_balance = config.STARTING_BALANCE

    def initialize_binance_client(self, config: ConfigModel) -> Client:
        try:
            first_symbol_key = next(iter(config.SYMBOLS_TO_TRADE))
            first_symbol_cfg = config.SYMBOLS_TO_TRADE[first_symbol_key].api_credentials
        except StopIteration:
            logger.error("No symbols found in configuration.")
            sys.exit(1)

        env_api_key = os.getenv("BINANCE_API_KEY", first_symbol_cfg.api_key)
        env_secret_key = os.getenv("BINANCE_SECRET_KEY", first_symbol_cfg.secret_key)

        client = Client(env_api_key, env_secret_key, testnet=config.USE_TESTNET)

        # Basic ping or time check:
        try:
            server_time = client.get_server_time()
            logger.info(f"Binance Server Time: {server_time}")
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch server time: {e}")
            sys.exit(1)

        return client

    def fetch_historical_data(self, symbol: str, interval: str = "1m", lookback: int = 100) -> pd.DataFrame:
        # If your library doesn't have .futures_klines (because it's old),
        # switch to client.get_klines(...) for spot data, or consider upgrading.
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
            data = pd.DataFrame(klines, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
                "ignore"
            ])
            data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
            data.set_index("open_time", inplace=True)

            numeric_cols = ["open","high","low","close","volume",
                            "quote_asset_volume","taker_buy_base_asset_volume",
                            "taker_buy_quote_asset_volume"]
            for c in numeric_cols:
                data[c] = data[c].astype(float)

            return data
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(colored(
                f"[RealOrder] Placing {side} order on {symbol}, qty={quantity:.4f}",
                "cyan"
            ))
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
        try:
            while True:
                order = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                if order["status"] in ["FILLED", "CANCELED", "REJECTED"]:
                    logger.info(colored(
                        f"[RealOrder] Order {order_id} => {order['status']}",
                        "cyan"
                    ))
                    return (order["status"] == "FILLED")
                time.sleep(1)
        except BinanceAPIException as e:
            logger.error(f"Binance API Exception monitoring order on {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error monitoring order on {symbol}: {e}")
            return False

    def calculate_kelly_quantity(self, symbol: str, current_price: float) -> float:
        strategy = self.strategies_for_symbols[symbol]
        p = strategy.parameters.kelly_win_rate or 0.5
        b = strategy.parameters.kelly_win_loss_ratio or 1.0
        kelly_fraction = p - ((1 - p) / b)
        if kelly_fraction < 0:
            kelly_fraction = 0.0
        kelly_fraction = min(kelly_fraction, 0.25)

        notional = kelly_fraction * self.current_balance
        leverage = self.config.SYMBOLS_TO_TRADE[symbol].risk_parameters.DEFAULT_LEVERAGE
        notional *= leverage

        if current_price == 0:
            return 0.0

        qty = notional / current_price
        if (qty * current_price) < self.config.MIN_NOTIONAL:
            logger.warning(
                f"[{symbol}] Kelly => qty {qty:.4f} * price {current_price:.2f}"
                f" < min notional => 0"
            )
            return 0.0
        return float(round(qty, 6))

def start_symbol_sockets(self, symbols: List[str]):
    """
    For python-binance 1.0.x with Futures, define callbacks and start the manager.
    """
    def ticker_callback(msg):
        symbol = msg.get("s")
        if not symbol or "c" not in msg:
            return
        close_price = float(msg["c"])
        open_price = float(msg["o"])
        high_price = float(msg["h"])
        low_price  = float(msg["l"])
        event_time = pd.to_datetime(msg["E"], unit="ms")
        # Your custom handling of ticker data...
        self.handle_ticker_data(symbol, open_price, high_price, low_price, close_price, event_time)

    def depth_callback(msg):
        symbol = msg.get("s")
        if not symbol or "b" not in msg or "a" not in msg:
            return
        self.handle_depth_data(symbol, msg)

    for sym in symbols:
        # FUTURES symbol ticker
        self.bsm.start_futures_symbol_ticker_socket(ticker_callback, sym)

        # FUTURES depth socket (partial order book)
        # You can adjust 'depth' and 'speed' as needed (5/10/20, 100/1000 ms)
        self.bsm.start_futures_depth_socket(depth_callback, sym, depth=5, speed=100)

    # Then start the manager once all sockets are registered
    self.bsm.start()


    def handle_ticker_data(self, symbol: str, open_p: float, high_p: float, low_p: float, close_p: float, ts):
        """Callback when new Ticker data arrives for a symbol."""
        # In a real bot, you'd store these updates in a DataFrame,
        # generate signals, etc.
        # For simplicity, just log the close price:
        logger.debug(f"[TickerCallback] {symbol} => O:{open_p}, H:{high_p}, L:{low_p}, C:{close_p}, time={ts}")

    def handle_depth_data(self, symbol: str, msg: dict):
        """Callback when new Depth data arrives for a symbol."""
        # Parse the bids/asks from the message
        # Typically 'b' are bids, 'a' are asks:
        bids = msg.get("b", [])
        asks = msg.get("a", [])
        self.order_books[symbol] = {"bids": bids, "asks": asks}

    def shutdown(self):
        logger.info(colored("Shutting down SuperBot gracefully...", "blue"))
        self.stop_bot = True
        try:
            self.bsm.close()
        except:
            pass
        self.executor.shutdown(wait=False)
        logger.info(colored("SuperBot shut down.", "blue"))

##############################################################################
# EnhancedTradingBot that Integrates Some Extra Features
##############################################################################
class EnhancedTradingBot(BaseTradingBot):
    def __init__(self, config: ConfigModel):
        super().__init__(config)
        self.risk_manager = RiskManager(config)
        self.max_daily_loss = config.STARTING_BALANCE * 0.02
        self.daily_pnl = 0.0

        # ML, MTF, OrderFlow, Regime
        self.ml_predictors: Dict[str, MLPredictor] = {}
        self.mtf_analyzers: Dict[str, MultiTimeframeAnalyzer] = {}
        self.orderflow_analyzer = OrderFlowAnalyzer()
        self.regime_detector = MarketRegimeDetector()

        for sym in config.SYMBOLS_TO_TRADE.keys():
            self.ml_predictors[sym] = MLPredictor()
            self.mtf_analyzers[sym] = MultiTimeframeAnalyzer(sym)

    def start_trading(self, symbols: List[str]):
        """
        This function starts the websocket manager for all symbols
        and does an initial data load/training if desired.
        """
        # 1) Load historical data for each symbol & train ML:
        for sym in symbols:
            df = self.fetch_historical_data(sym, "1m", 500)
            df = FeatureEngineer.add_technical_features(df)
            self.ml_predictors[sym].train(df)

        # 2) Start the websockets for real-time updates:
        self.start_symbol_sockets(symbols)

        # 3) Could do a main loop in another thread or simply wait:
        logger.info("Trading started. Listening to streams... (press Ctrl+C to exit)")

    # (Add additional logic for actual signals/decisions using the
    #  older callback-based approach or a separate timed loop to check signals)

##############################################################################
# Main
##############################################################################
def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    trading_bot = EnhancedTradingBot(config)
    symbols_to_trade = list(config.SYMBOLS_TO_TRADE.keys())

    def signal_handler(sig, frame):
        logger.warning(colored(f"Received signal {sig}, shutting down bot...", "yellow"))
        trading_bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start trading (websockets) 
    try:
        trading_bot.start_trading(symbols_to_trade)
        # Keep the main thread alive forever:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trading_bot.shutdown()
    except Exception as e:
        logger.error(colored(f"Unexpected error in main loop: {e}", "red"))
        trading_bot.shutdown()

if __name__ == "__main__":
    main()
