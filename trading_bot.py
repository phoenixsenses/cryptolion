#!/usr/bin/env python3
import os
import sys
import json
import logging
import time
import concurrent.futures
import signal
import asyncio
import random
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import pandas_ta as ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False

# Sentiment (optional)
try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
except ImportError:
    HAVE_TEXTBLOB = False

from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from termcolor import colored

# Binance imports
try:
    from binance import BinanceSocketManager
    from binance.client import Client
    from binance.enums import *
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except ImportError as e:
    raise ImportError(
        "Could not import BinanceSocketManager from binance. "
        "Please ensure you have python-binance installed.\n"
        f"{e}"
    )

from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv  # <-- For loading .env

##############################################################################
# LOGGING SETUP
##############################################################################
logger = logging.getLogger("SuperBotProduction")
logger.setLevel(logging.DEBUG)

json_formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
file_handler = RotatingFileHandler("trading_bot.log", maxBytes=10*1024*1024, backupCount=2)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(json_formatter)

console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set to INFO to reduce console clutter
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

##############################################################################
# ORDER FLOW & MARKET REGIME (SHORT VERSIONS)
##############################################################################
class OrderFlowAnalyzer:
    def __init__(self):
        self.last_imbalance = 0.0

    def update_order_book(self, symbol: str, bids: list, asks: list):
        if not bids or not asks:
            return
        best_bid_qty = float(bids[0][1])
        best_ask_qty = float(asks[0][1])
        total_qty = best_bid_qty + best_ask_qty
        if total_qty > 0:
            self.last_imbalance = (best_bid_qty - best_ask_qty) / total_qty

    def get_latest_imbalance(self):
        return self.last_imbalance

class MarketRegimeDetector:
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.current_regime = "neutral"

    def detect_regime(self, df: pd.DataFrame) -> str:
        if len(df) < self.lookback:
            return "neutral"
        closes = df["close"].iloc[-self.lookback:]
        x = np.arange(self.lookback)
        y = closes.values
        slope, intercept = np.polyfit(x, y, 1)
        if slope > 0:
            self.current_regime = "bull"
        elif slope < 0:
            self.current_regime = "bear"
        else:
            self.current_regime = "neutral"
        return self.current_regime

##############################################################################
# CONFIG, RISK, UTILS
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

class RiskManager:
    def __init__(self, config: ConfigModel):
        self.config = config
        self.max_daily_loss = config.STARTING_BALANCE * 0.02
        self.daily_pnl = 0.0
        self.last_reset_day = datetime.now(timezone.utc).day

    def update_daily_pnl(self, pnl: float):
        self.check_daily_reset()
        self.daily_pnl += pnl

    def check_daily_reset(self):
        current_day = datetime.now(timezone.utc).day
        if current_day != self.last_reset_day:
            logger.info(colored("[RiskManager] Reset daily PnL for new day.", "magenta"))
            self.daily_pnl = 0.0
            self.last_reset_day = current_day

    def is_daily_limit_reached(self) -> bool:
        return self.daily_pnl <= -self.max_daily_loss

    def can_open_new_position(self, potential_loss: float) -> bool:
        self.check_daily_reset()
        if (self.daily_pnl - potential_loss) < -self.max_daily_loss:
            return False
        return True

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
    if not HAVE_TEXTBLOB:
        return 0.0
    return TextBlob(text).sentiment.polarity

def fetch_twitter_sentiment(symbol: str, count: int=50) -> float:
    sentiments = []
    for _ in range(count):
        # Replace this with actual Twitter API calls or mock data
        random_text = f"Some random tweet about {symbol}"
        sentiments.append(analyze_sentiment(random_text))
    return float(np.mean(sentiments)) if sentiments else 0.0

def fetch_news_sentiment(symbol: str) -> float:
    # Replace this with actual news API calls or mock data
    fake_news_headline = f"News about {symbol}"
    return analyze_sentiment(fake_news_headline)

def combined_sentiment_score(symbol: str, cfg: SentimentAnalysisModel) -> float:
    if not cfg.ENABLED:
        return 0.0
    total = 0.0
    sources = 0
    if "twitter" in cfg.DATA_SOURCE:
        total += fetch_twitter_sentiment(symbol, cfg.TWEETS_PER_SYMBOL)
        sources += 1
    if "news" in cfg.DATA_SOURCE:
        total += fetch_news_sentiment(symbol)
        sources += 1
    return total / sources if sources > 0 else 0.0

##############################################################################
# ML + FEATURE ENGINEERING
##############################################################################
class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "RSI" not in df.columns:
            df["RSI"] = ta.rsi(df["close"], length=14)
        if "OBV" not in df.columns:
            df["OBV"] = ta.obv(df["close"], df["volume"])
        if "SMA_50" not in df.columns:
            df["SMA_50"] = ta.sma(df["close"], length=50)
        if "EMA_20" not in df.columns:
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
        feats = ["RSI", "OBV", "SMA_50", "EMA_20"]
        df = df.dropna(subset=feats)
        if len(df) < 2:
            return
        features = df[feats]
        shifted_close = df["close"].shift(-1)
        labels = np.where(shifted_close > df["close"], 1, 0)[:-1]
        features = features.iloc[:-1]
        if len(features) < 10:
            return
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained:
            return 0.5
        feats = ["RSI", "OBV", "SMA_50", "EMA_20"]
        df = df.dropna(subset=feats)
        if df.empty:
            return 0.5
        latest = df.iloc[-1][feats].values.reshape(1, -1)
        scaled = self.scaler.transform(latest)
        return self.model.predict_proba(scaled)[0][1]

class XGBPredictor:
    def __init__(self):
        if HAVE_XGBOOST:
            self.model = XGBClassifier(n_estimators=50, eval_metric='logloss')
        else:
            self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        if not HAVE_XGBOOST or self.model is None or df.empty or "close" not in df.columns:
            return
        feats = ["RSI", "OBV", "SMA_50", "EMA_20"]
        df = df.dropna(subset=feats)
        if len(df) < 2:
            return
        features = df[feats]
        shifted_close = df["close"].shift(-1)
        labels = np.where(shifted_close > df["close"], 1, 0)[:-1]
        features = features.iloc[:-1]
        if len(features) < 10:
            return
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        self.is_trained = True

    def predict(self, df: pd.DataFrame) -> float:
        if not HAVE_XGBOOST or self.model is None or not self.is_trained:
            return 0.5
        feats = ["RSI", "OBV", "SMA_50", "EMA_20"]
        df = df.dropna(subset=feats)
        if df.empty:
            return 0.5
        latest = df.iloc[-1][feats].values.reshape(1, -1)
        scaled = self.scaler.transform(latest)
        return float(self.model.predict_proba(scaled)[0][1])

##############################################################################
# STRATEGIES
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
        self.bot = None

        self.stop_loss_price = None
        self.take_profit_price = None
        self.trailing_stop_enabled = False

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
                      stop_loss: float = None, take_profit: float = None, trailing: bool = False):
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
            if self.bot:
                self.bot.place_order(self.symbol, SIDE_SELL, self.qty)
        else:
            pnl = (self.entry_price - close_price) * self.qty
            if self.bot:
                self.bot.place_order(self.symbol, SIDE_BUY, self.qty)

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

        if self.bot and hasattr(self.bot, "on_position_closed"):
            self.bot.on_position_closed(self.symbol, pnl)
        return pnl

    def close_partial_position(self, fraction: float, close_price: float) -> float:
        if not self.is_open or fraction <= 0:
            return 0.0
        close_qty = self.qty * fraction
        if close_qty <= 0:
            return 0.0

        if self.side == "BUY":
            pnl = (close_price - self.entry_price) * close_qty
            if self.bot:
                self.bot.place_order(self.symbol, SIDE_SELL, close_qty)
        else:
            pnl = (self.entry_price - close_price) * close_qty
            if self.bot:
                self.bot.place_order(self.symbol, SIDE_BUY, close_qty)

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

        if self.bot and hasattr(self.bot, "on_position_closed"):
            self.bot.on_position_closed(self.symbol, pnl)
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

        # Basic
        sma_val = df["close"].rolling(period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        basic_signal = "BUY" if last_close < sma_val else "SELL"

        # ML (RF + XGB)
        ml_prob = 0.5
        xgb_prob = 0.5
        if self.bot:
            ml_prob = self.bot.ml_predictors[self.symbol].predict(df)
            xgb_prob = self.bot.xgb_predictors[self.symbol].predict(df)
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

        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score >= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                final_decision = "BUY"
            elif self.sentiment_score <= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                final_decision = "SELL"

        if self.bot:
            imbalance = self.bot.orderflow_analyzer.get_latest_imbalance()
            if imbalance > 0.5:
                final_decision = "BUY"
            elif imbalance < -0.5:
                final_decision = "SELL"

            regime = self.bot.regime_detector.detect_regime(df)
            if regime == "bull" and final_decision == "SELL":
                final_decision = "HOLD"
            elif regime == "bear" and final_decision == "BUY":
                final_decision = "HOLD"

        return final_decision

##############################################################################
# BASE TRADING BOT
##############################################################################
class BaseTradingBot:
    def __init__(self, config: ConfigModel):
        self.config = config
        load_dotenv()  # <-- Load environment variables from .env
        self.client = self.initialize_binance_client(config)
        self.stop_bot = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)

        self.strategy_map = {
            "StrategyA": StrategyA
        }

        self.strategies_for_symbols: Dict[str, BaseStrategy] = {}
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

            sp = sym_cfg.strategy_parameters
            strategy_instance = strat_cls(
                symbol=sym_key,
                parameters=sp,
                adaptive_atr=adaptive_atr,
                atr_multiplier=atr_multiplier,
                sentiment_config=sentiment_model
            )
            strategy_instance.bot = self
            self.strategies_for_symbols[sym_key] = strategy_instance

        self.bsm = BinanceSocketManager(self.client)
        self.socket_connections = []
        self.order_books = {}
        self.current_balance = config.STARTING_BALANCE

        self.ml_retrain_interval = 300  # 5 minutes
        self.last_ml_retrain_time = time.time()

    def initialize_binance_client(self, config: ConfigModel) -> Client:
        logger.debug("Initializing Binance Client...")
        # Grab the first symbol's credentials from config
        try:
            first_sym = next(iter(config.SYMBOLS_TO_TRADE))
            creds = config.SYMBOLS_TO_TRADE[first_sym].api_credentials
        except StopIteration:
            logger.error("No symbols in config. Exiting.")
            sys.exit(1)

        # Overriding with environment variables if present
        env_api_key = os.getenv("BINANCE_API_KEY", creds.api_key)
        env_sec_key = os.getenv("BINANCE_SECRET_KEY", creds.secret_key)

        # Initialize the Client without proxies
        try:
    client.https_proxy = {}
            client = Client(
                env_api_key,
                env_sec_key,
                testnet=config.USE_TESTNET
                # Removed proxies={}
            )
            logger.debug(f"Client proxies: {client.proxies if hasattr(client, 'proxies') else 'No proxies set'}")
        except Exception as e:
            logger.error(f"Error initializing Binance client: {e}", exc_info=True)
            sys.exit(1)

        try:
            stime = client.get_server_time()
            logger.info(f"Binance Server Time: {stime}")
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch server time: {e}", exc_info=True)
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error fetching server time: {e}", exc_info=True)
            sys.exit(1)
        logger.debug("Binance Client initialized successfully.")
        return client

    def fetch_historical_data(self, symbol: str, interval: str="1m", lookback: int=100) -> pd.DataFrame:
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
            df = pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","number_of_trades",
                "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
            ])
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            numeric_cols = ["open","high","low","close","volume",
                            "quote_asset_volume","taker_buy_base_asset_volume",
                            "taker_buy_quote_asset_volume"]
            for c in numeric_cols:
                df[c] = df[c].astype(float)
            logger.debug(f"Fetched historical data for {symbol}, {len(df)} records.")
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def retrain_ml_models(self):
        logger.info("[ML] Re-training models now...")
        for sym in self.ml_predictors.keys():
            df = self.fetch_historical_data(sym, "1m", 500)
            df = FeatureEngineer.add_technical_features(df)
            self.ml_predictors[sym].train(df)
            if HAVE_XGBOOST:
                self.xgb_predictors[sym].train(df)
        logger.info("[ML] Re-training completed.")

    def shutdown(self):
        logger.info("Shutting down bot gracefully...")
        self.stop_bot = True
        try:
            for conn_key in self.socket_connections:
                logger.info(f"Closing socket: {conn_key}")
                self.bsm.close_socket(conn_key)
        except Exception as e:
            logger.error(f"Error closing sockets: {e}", exc_info=True)
        finally:
            self.executor.shutdown(wait=False)
            # Explicitly close the client session if possible
            if hasattr(self.client, 'close'):
                try:
                    self.client.close()
                except Exception as e:
                    logger.error(f"Error closing client: {e}", exc_info=True)
            logger.info("Bot shut down.")
            sys.exit(0)

##############################################################################
# ENHANCED TRADING BOT
##############################################################################
class EnhancedTradingBot(BaseTradingBot):
    def __init__(self, config: ConfigModel):
        super().__init__(config)
        self.risk_manager = RiskManager(config)
        self.orderflow_analyzer = OrderFlowAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        self.ml_predictors: Dict[str, MLPredictor] = {}
        self.xgb_predictors: Dict[str, XGBPredictor] = {}
        self.executor.submit(self.handle_shutdown_signals)

        for sym in config.SYMBOLS_TO_TRADE.keys():
            self.ml_predictors[sym] = MLPredictor()
            if HAVE_XGBOOST:
                self.xgb_predictors[sym] = XGBPredictor()

    def on_position_closed(self, symbol: str, pnl: float):
        self.risk_manager.update_daily_pnl(pnl)
        self.current_balance += pnl
        logger.info(colored(
            f"[PnL Update] {symbol} => daily PnL={self.risk_manager.daily_pnl:.2f}, balance={self.current_balance:.2f}",
            "yellow"
        ))

    def place_order(self, symbol: str, side: str, quantity: float) -> dict:
        current_price = 0.0
        df = self.fetch_historical_data(symbol, "1m", 1)
        if not df.empty:
            current_price = df["close"].iloc[-1]
        potential_loss = 0.05 * quantity * current_price

        if not self.risk_manager.can_open_new_position(potential_loss):
            logger.warning(f"[RiskManager] Cannot open new position on {symbol}, daily limit exceeded.")
            return {}
        if self.risk_manager.is_daily_limit_reached():
            logger.warning(f"[RiskManager] Daily limit reached, blocking new order on {symbol}.")
            return {}

        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.info(colored(
                f"[RealOrder] {side} {symbol} qty={quantity:.4f}",
                "cyan"
            ))
            return order
        except BinanceAPIException as e:
            logger.error(f"Binance API Exception placing order on {symbol}: {e}", exc_info=True)
            return {}
        except BinanceOrderException as e:
            logger.error(f"Binance Order Exception placing order on {symbol}: {e}", exc_info=True)
            return {}
        except Exception as e:
            logger.error(f"Unexpected error placing order on {symbol}: {e}", exc_info=True)
            return {}

    def retrain_ml_models(self):
        logger.info("[ML] Re-training models now...")
        for sym in self.ml_predictors.keys():
            df = self.fetch_historical_data(sym, "1m", 500)
            df = FeatureEngineer.add_technical_features(df)
            self.ml_predictors[sym].train(df)
            if HAVE_XGBOOST:
                self.xgb_predictors[sym].train(df)
        logger.info("[ML] Re-training completed.")

    def start_trading(self, symbols: List[str]):
        # Train once at startup
        for sym in symbols:
            df = self.fetch_historical_data(sym, "1m", 500)
            df = FeatureEngineer.add_technical_features(df)
            self.ml_predictors[sym].train(df)
            if HAVE_XGBOOST:
                self.xgb_predictors[sym].train(df)
        self.start_symbol_sockets(symbols)
        logger.info(f"EnhancedTradingBot is running websockets for: {symbols}")

    def start_symbol_sockets(self, symbols: List[str]):
        def process_message(msg):
            try:
                if not msg or "e" not in msg:
                    return
                event_type = msg.get("e")
                symbol = msg.get("s")
                if event_type == "kline":
                    k = msg.get("k", {})
                    is_final = k.get("x", False)
                    close_price = float(k.get("c", 0.0))
                    if is_final:
                        logger.debug(f"[Kline final] {symbol} => close={close_price}")
                        df = self.fetch_historical_data(symbol, "1m", 100)
                        df = FeatureEngineer.add_technical_features(df)

                        realized_pnl = self.strategies_for_symbols[symbol].monitor_positions(df)
                        if realized_pnl != 0:
                            logger.info(f"[MonitorPositions] {symbol} => realized PnL={realized_pnl:.2f}")

                        if time.time() - self.last_ml_retrain_time > self.ml_retrain_interval:
                            logger.info("[ML] Re-training models (interval passed).")
                            self.retrain_ml_models()
                            self.last_ml_retrain_time = time.time()

                        decision = self.strategies_for_symbols[symbol].decision(df)
                        logger.info(f"[Decision] {symbol} => {decision}")
                        if decision in ["BUY", "SELL"]:
                            qty = self.calculate_kelly_quantity(symbol, close_price)
                            logger.info(f"[PositionSize] {symbol} => {qty:.6f} (calc) @ {close_price:.2f}")
                            if qty > 0:
                                self.place_order(symbol, SIDE_BUY if decision=="BUY" else SIDE_SELL, qty)
                                sl_price = close_price * (0.98 if decision=="BUY" else 1.02)
                                tp_price = close_price * (1.02 if decision=="BUY" else 0.98)
                                self.strategies_for_symbols[symbol].execute_trade(
                                    side=decision,
                                    quantity=qty,
                                    price=close_price,
                                    stop_loss=sl_price,
                                    take_profit=tp_price,
                                    trailing=True
                                )
                        else:
                            logger.debug(f"[Decision] {symbol} => HOLD (no trade)")
                elif event_type == "depthUpdate":
                    bids = msg.get("b", [])
                    asks = msg.get("a", [])
                    self.order_books[symbol] = {"bids": bids, "asks": asks}
                    self.orderflow_analyzer.update_order_book(symbol, bids, asks)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)

        streams = []
        for sym in symbols:
            streams.append(f"{sym.lower()}@kline_1m")
            streams.append(f"{sym.lower()}@depth5@100ms")

        try:
            logger.info(f"Initializing multiplex socket for streams: {streams}")
            conn_key = self.bsm.futures_multiplex_socket(streams, process_message)
            self.socket_connections.append(conn_key)
            self.bsm.start()
        except AttributeError as e:
            logger.error(f"Error starting multiplex socket: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error starting multiplex socket: {e}", exc_info=True)

    def calculate_kelly_quantity(self, symbol: str, current_price: float) -> float:
        strat = self.strategies_for_symbols[symbol]
        p = strat.parameters.kelly_win_rate or 0.5
        b = strat.parameters.kelly_win_loss_ratio or 1.0
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
            logger.warning(f"[{symbol}] Notional {qty*current_price:.2f} < MIN_NOTIONAL => 0.0")
            return 0.0
        return float(round(qty, 6))

    def handle_shutdown_signals(self):
        # Handle termination signals to shutdown gracefully
        def shutdown_handler(signum, frame):
            logger.info(f"Received signal {signum}. Initiating shutdown...")
            self.shutdown()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

##############################################################################
# MAIN FUNCTION
##############################################################################
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

        bot = EnhancedTradingBot(config)
        logger.debug("EnhancedTradingBot instance created.")

        symbols = list(config.SYMBOLS_TO_TRADE.keys())
        bot.start_trading(symbols)
        logger.info("Trading bot is now running.")

        # Keep the main thread alive to allow asyncio to run
        while not bot.stop_bot:
            time.sleep(1)

    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
