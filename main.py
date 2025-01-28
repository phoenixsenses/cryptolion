#!/usr/bin/env python3
import os
import sys
import json
import logging
import time
import random
import concurrent.futures
import signal

import pandas as pd
import numpy as np
import pandas_ta as ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import scipy.stats as st

try:
    from textblob import TextBlob
    HAVE_TEXTBLOB = True
except ImportError:
    HAVE_TEXTBLOB = False

from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from termcolor import colored

# binance imports...
try:
    from binance import BinanceSocketManager
    from binance.client import Client
    from binance.enums import *
    from binance.exceptions import BinanceAPIException, BinanceOrderException
except ImportError as e:
    raise ImportError(
        "Could not import BinanceSocketManager from binance. "
        "Please ensure you have 'python-binance' 1.0.x installed.\n"
        f"{e}"
    )

from pydantic import BaseModel, ValidationError, Field
from dotenv import load_dotenv

# Optional: check python-binance version
try:
    import pkg_resources
    BINANCE_VERSION = pkg_resources.get_distribution("python-binance").version
except:
    BINANCE_VERSION = "unknown"

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
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.debug(f"Detected python-binance version: {BINANCE_VERSION}")

##############################################################################
# PLACEHOLDER CLASSES
##############################################################################
class RiskManager:
    def __init__(self, config):
        self.config = config
        # Add any risk management logic here

class OrderFlowAnalyzer:
    def __init__(self):
        pass
    # Add real order flow logic or placeholders

class MarketRegimeDetector:
    def __init__(self):
        pass
    # Add real regime logic or placeholders

class MultiTimeframeAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
    # Add real multi-timeframe logic or placeholders

##############################################################################
# CONFIG MODELS
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
# LOAD CONFIG
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
# UTILITY
##############################################################################
def compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    if len(df) < period:
        return 0.0
    for col in ["high","low","close"]:
        df[col] = df[col].astype(float)

    df["previous_close"] = df["close"].shift(1)
    df["high_low"] = df["high"] - df["low"]
    df["high_pc"]  = abs(df["high"] - df["previous_close"])
    df["low_pc"]   = abs(df["low"]  - df["previous_close"])
    df["tr"] = df[["high_low","high_pc","low_pc"]].max(axis=1)

    df["atr"] = df["tr"].rolling(period).mean()
    latest_atr = df["atr"].iloc[-1] if len(df) >= period else 0.0
    if np.isnan(latest_atr):
        latest_atr = 0.0

    # optional cleanup
    df.drop(["previous_close","high_low","high_pc","low_pc","tr","atr"],
            axis=1, errors="ignore", inplace=True)
    return latest_atr

def analyze_sentiment(text: str) -> float:
    if not HAVE_TEXTBLOB:
        return 0.0
    return TextBlob(text).sentiment.polarity

def fetch_twitter_sentiment(symbol: str, count: int=50) -> float:
    sentiments = []
    for _ in range(count):
        # Example mock tweets
        random_text = f"Random tweet about {symbol}"
        sentiments.append(analyze_sentiment(random_text))
    if len(sentiments)==0:
        return 0.0
    return float(np.mean(sentiments))

def fetch_news_sentiment(symbol: str) -> float:
    random_news_text = f"Some news headline about {symbol}"
    return analyze_sentiment(random_news_text)

def combined_sentiment_score(symbol: str, cfg: SentimentAnalysisModel) -> float:
    if not cfg.ENABLED:
        return 0.0
    total = 0.0
    sources_used=0
    if "twitter" in cfg.DATA_SOURCE:
        total += fetch_twitter_sentiment(symbol, cfg.TWEETS_PER_SYMBOL)
        sources_used +=1
    if "news" in cfg.DATA_SOURCE:
        total += fetch_news_sentiment(symbol)
        sources_used +=1
    if sources_used==0:
        return 0.0
    return total / sources_used

##############################################################################
# ML + FEATURE ENGINEERING
##############################################################################
class FeatureEngineer:
    @staticmethod
    def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for c in ["open","high","low","close","volume"]:
            if c not in df.columns:
                df[c]=0.0
        df["RSI"] = ta.rsi(df["close"], length=14)
        df["OBV"] = ta.obv(df["close"], df["volume"])
        df["SMA_50"] = ta.sma(df["close"], length=50)
        df["EMA_20"] = ta.ema(df["close"], length=20)
        return df

class MLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler= StandardScaler()
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        if df.empty or "close" not in df.columns:
            return
        feats = ["RSI","OBV","SMA_50","EMA_20"]
        df = df.dropna(subset=feats)
        if len(df) <2:
            return
        features = df[feats]
        shifted_close = df["close"].shift(-1)
        labels = np.where(shifted_close > df["close"],1,0)[:-1]
        features = features.iloc[:-1]
        if len(features)<10:
            return
        scaled= self.scaler.fit_transform(features)
        self.model.fit(scaled, labels)
        self.is_trained=True

    def predict(self, df: pd.DataFrame)-> float:
        if not self.is_trained:
            return 0.5
        feats = ["RSI","OBV","SMA_50","EMA_20"]
        df = df.dropna(subset=feats)
        if len(df)==0:
            return 0.5
        latest = df.iloc[-1][feats].values.reshape(1,-1)
        scaled = self.scaler.transform(latest)
        return self.model.predict_proba(scaled)[0][1]

##############################################################################
# STRATEGIES (Base + Example)
##############################################################################
class BaseStrategy:
    def __init__(self,
                 symbol: str,
                 parameters: Optional[StrategyParametersModel]=None,
                 adaptive_atr: bool=False,
                 atr_multiplier: float=1.5,
                 sentiment_config: Optional[SentimentAnalysisModel]=None):
        self.symbol = symbol
        self.is_open = False
        self.side    = None
        self.qty     = 0.0
        self.entry_price=0.0
        self.partial_tps: List[Dict[str,Any]]=[]
        self.parameters = parameters or StrategyParametersModel()
        self.adaptive_atr= adaptive_atr
        self.static_atr_multiplier= atr_multiplier
        self.trailing_offset = self.static_atr_multiplier
        self.sentiment_config = sentiment_config
        self.sentiment_score=0.0

    def fetch_sentiment(self):
        if self.sentiment_config:
            self.sentiment_score = combined_sentiment_score(self.symbol, self.sentiment_config)
            logger.debug(colored(f"[{self.symbol}] Sentiment => {self.sentiment_score:.3f}","cyan"))

    def decision(self, df: pd.DataFrame) -> str:
        """Return 'BUY', 'SELL', or 'HOLD'."""
        return "HOLD"

    def update_trailing_offset(self, df: pd.DataFrame):
        if self.adaptive_atr:
            period = self.parameters.atr_period or 14
            atr_val= compute_atr(df.copy(), period)
            if atr_val<=0:
                self.trailing_offset= self.static_atr_multiplier
            else:
                self.trailing_offset= atr_val*(self.parameters.atr_multiplier or 1.5)
        else:
            self.trailing_offset= self.parameters.atr_multiplier or 2.0

    def execute_trade(self, side: str, quantity: float, price: float):
        self.is_open=True
        self.side= side
        self.qty= quantity
        self.entry_price= price
        self.partial_tps=[]
        logger.info(colored(
            f"[{self.symbol}] {side} trade executed @ {price:.2f}, qty={quantity}",
            "blue"
        ))

    def close_position(self, reason: str, close_price: float)-> float:
        if not self.is_open:
            return 0.0
        if self.side=="BUY":
            pnl= (close_price - self.entry_price)*self.qty
        else:
            pnl= (self.entry_price - close_price)*self.qty

        logger.info(colored(
            f"[{self.symbol}] Closing position. reason={reason}, close_price={close_price:.2f}, PnL={pnl:.2f}",
            "yellow"
        ))
        self.is_open=False
        self.side=None
        self.qty=0.0
        self.entry_price=0.0
        self.partial_tps=[]
        return pnl

    def close_partial_position(self, fraction: float, close_price: float)-> float:
        if not self.is_open or fraction<=0:
            return 0.0
        close_qty= self.qty*fraction
        if close_qty<=0:
            return 0.0
        if self.side=="BUY":
            pnl= (close_price - self.entry_price)*close_qty
        else:
            pnl= (self.entry_price - close_price)*close_qty

        logger.info(colored(
            f"[{self.symbol}] Partial close {fraction*100:.1f}%, close_price={close_price:.2f}, PnL={pnl:.2f}",
            "yellow"
        ))
        self.qty-= close_qty
        if self.qty<=0:
            self.is_open=False
            self.side=None
            self.entry_price=0.0
            self.partial_tps=[]
        return pnl

    def monitor_positions(self, df: pd.DataFrame)-> float:
        if not self.is_open:
            return 0.0
        current_price= df["close"].iloc[-1]
        realized_pnl=0.0

        # Check partial TPs
        for pt in self.partial_tps:
            if not pt["triggered"]:
                tp_price= pt["price"]
                frac= pt["fraction"]
                if self.side=="BUY" and current_price>= tp_price:
                    realized_pnl+= self.close_partial_position(frac, current_price)
                    pt["triggered"]= True
                elif self.side=="SELL" and current_price<= tp_price:
                    realized_pnl+= self.close_partial_position(frac, current_price)
                    pt["triggered"]= True

        # Update trailing
        self.update_trailing_offset(df.copy())
        if self.side=="BUY":
            trail_line= (self.entry_price
                         + (current_price- self.entry_price)
                         - self.trailing_offset)
            if current_price< trail_line:
                realized_pnl+= self.close_position("TrailingStop-BUY", current_price)
        else:  # SELL
            trail_line= (self.entry_price
                         - (self.entry_price- current_price)
                         - self.trailing_offset)
            if current_price> trail_line:
                realized_pnl+= self.close_position("TrailingStop-SELL", current_price)

        return realized_pnl

class StrategyA(BaseStrategy):
    """Simple SMA-based example strategy."""
    def decision(self, df: pd.DataFrame) -> str:
        self.fetch_sentiment()
        period = self.parameters.sma_period or 20
        if len(df)< period:
            return "HOLD"
        sma_val= df["close"].rolling(period).mean().iloc[-1]
        last_close= df["close"].iloc[-1]
        basic_signal= "BUY" if last_close < sma_val else "SELL"

        # incorporate sentiment
        if self.sentiment_config and self.sentiment_config.ENABLED:
            if self.sentiment_score>= self.sentiment_config.SENTIMENT_THRESHOLD_POSITIVE:
                return "BUY"
            elif self.sentiment_score<= self.sentiment_config.SENTIMENT_THRESHOLD_NEGATIVE:
                return "SELL"
        return basic_signal

    def execute_trade(self, side: str, quantity: float, price: float):
        super().execute_trade(side, quantity, price)
        if side=="BUY":
            self.partial_tps= [
                {"price": price+2.0, "fraction":0.5, "triggered":False},
                {"price": price+4.0, "fraction":0.5, "triggered":False},
            ]
        else:
            self.partial_tps= [
                {"price": price-2.0, "fraction":0.5, "triggered":False},
                {"price": price-4.0, "fraction":0.5, "triggered":False},
            ]

##############################################################################
# BaseTradingBot
##############################################################################
class BaseTradingBot:
    def __init__(self, config: ConfigModel):
        self.config = config
        load_dotenv()

        self.client = self.initialize_binance_client(config)
        self.stop_bot= False
        self.executor= concurrent.futures.ThreadPoolExecutor(max_workers=10)

        # Strategy map
        self.strategy_map = {
            "StrategyA": StrategyA,
            # "StrategyCorrelated": StrategyCorrelated, ...
        }

        self.strategies_for_symbols= {}
        for sym_key, sym_cfg in config.SYMBOLS_TO_TRADE.items():
            strat_cls= self.strategy_map.get(sym_cfg.strategy, StrategyA)
            global_strat_cfg= config.STRATEGIES.get(sym_cfg.strategy)
            if global_strat_cfg:
                adaptive_atr   = global_strat_cfg.ADAPTIVE_ATR
                atr_multiplier = global_strat_cfg.ATR_MULTIPLIER
                sentiment_model= global_strat_cfg.SENTIMENT_ANALYSIS
            else:
                adaptive_atr   = False
                atr_multiplier = 1.5
                sentiment_model= None

            self.strategies_for_symbols[sym_key] = strat_cls(
                symbol=sym_key,
                parameters= sym_cfg.strategy_parameters,
                adaptive_atr= adaptive_atr,
                atr_multiplier= atr_multiplier,
                sentiment_config= sentiment_model
            )

        # Single socket manager for python-binance 1.0.x
        self.bsm= BinanceSocketManager(self.client)
        self.order_books={}
        self.current_balance= config.STARTING_BALANCE

    def initialize_binance_client(self, config: ConfigModel) -> Client:
        try:
            first_sym = next(iter(config.SYMBOLS_TO_TRADE))
            creds= config.SYMBOLS_TO_TRADE[first_sym].api_credentials
        except StopIteration:
            logger.error("No symbols in config. Exiting.")
            sys.exit(1)

        env_api_key   = os.getenv("BINANCE_API_KEY",   creds.api_key)
        env_secret_key= os.getenv("BINANCE_SECRET_KEY",creds.secret_key)

        client = Client(env_api_key, env_secret_key, testnet=config.USE_TESTNET)
        try:
            stime = client.get_server_time()
            logger.info(f"Binance Server Time: {stime}")
        except BinanceAPIException as e:
            logger.error(f"Failed to fetch server time: {e}")
            sys.exit(1)

        return client

    def fetch_historical_data(self, symbol: str, interval: str="1m", lookback: int=100)-> pd.DataFrame:
        """Futures klines (1.0.x)."""
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=lookback)
            df = pd.DataFrame(klines, columns=[
                "open_time","open","high","low","close","volume",
                "close_time","quote_asset_volume","number_of_trades",
                "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
            ])
            df["open_time"]= pd.to_datetime(df["open_time"], unit="ms")
            df.set_index("open_time", inplace=True)
            numeric_cols= ["open","high","low","close","volume",
                           "quote_asset_volume","taker_buy_base_asset_volume",
                           "taker_buy_quote_asset_volume"]
            for c in numeric_cols:
                df[c]= df[c].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def place_order(self, symbol: str, side: str, quantity: float)-> dict:
        try:
            order= self.client.futures_create_order(
                symbol=symbol,
                side= side,
                type=ORDER_TYPE_MARKET,
                quantity= quantity
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

    def calculate_kelly_quantity(self, symbol: str, current_price: float)-> float:
        """Simple Kelly formula + check min notional."""
        strat= self.strategies_for_symbols[symbol]
        p = strat.parameters.kelly_win_rate or 0.5
        b = strat.parameters.kelly_win_loss_ratio or 1.0
        kelly_fraction = p - ((1-p)/b)
        if kelly_fraction<0:
            kelly_fraction= 0.0
        kelly_fraction= min(kelly_fraction, 0.25)

        notional= kelly_fraction* self.current_balance
        leverage= self.config.SYMBOLS_TO_TRADE[symbol].risk_parameters.DEFAULT_LEVERAGE
        notional*= leverage

        if current_price==0:
            return 0.0
        qty= notional/current_price
        if (qty* current_price)< self.config.MIN_NOTIONAL:
            logger.warning(
                f"[{symbol}] Kelly => qty {qty:.4f} * price {current_price:.2f} < min notional => 0"
            )
            return 0.0
        return float(round(qty,6))

    def start_symbol_sockets(self, symbols: List[str]):
        """Start FUTURES websockets for ticker & partial depth, then bsm.start()."""

        def ticker_callback(msg):
            symbol = msg.get("s")
            if not symbol or "c" not in msg:
                return
            close_p = float(msg["c"])
            open_p  = float(msg["o"])
            high_p  = float(msg["h"])
            low_p   = float(msg["l"])
            etime   = pd.to_datetime(msg["E"], unit="ms")

            logger.debug(f"[Ticker] {symbol} => O:{open_p} H:{high_p} L:{low_p} C:{close_p} T:{etime}")

            # TODO: If you want real-time strategy decisions,
            # store these in a DataFrame or call your strategy here.

        def depth_callback(msg):
            symbol= msg.get("s")
            if not symbol or "b" not in msg or "a" not in msg:
                return
            bids = msg["b"]
            asks = msg["a"]
            self.order_books[symbol] = {"bids": bids, "asks": asks}
            logger.debug(f"[Depth] {symbol} => top2 bids: {bids[:2]}, top2 asks: {asks[:2]}")

        for sym in symbols:
            # Start the 24hr ticker for a single FUTURES symbol
            self.bsm.start_futures_ticker_socket(ticker_callback, symbol=sym)

            # Start partial-depth for that symbol
            self.bsm.start_futures_depth_socket(depth_callback, sym, depth=5, speed=100)

        self.bsm.start()

    def shutdown(self):
        logger.info(colored("Shutting down SuperBot gracefully...", "blue"))
        self.stop_bot=True
        try:
            self.bsm.close()
        except:
            pass
        self.executor.shutdown(wait=False)
        logger.info(colored("SuperBot shut down.", "blue"))

##############################################################################
# EnhancedTradingBot
##############################################################################
class EnhancedTradingBot(BaseTradingBot):
    def __init__(self, config: ConfigModel):
        super().__init__(config)
        from collections import defaultdict
        # Now that RiskManager is defined, no more NameError:
        self.risk_manager= RiskManager(config)
        self.max_daily_loss= config.STARTING_BALANCE* 0.02
        self.daily_pnl= 0.0

        # ML, MTF, OrderFlow, Regime
        self.ml_predictors: Dict[str, MLPredictor] = {}
        self.mtf_analyzers: Dict[str, Any] = {}

        self.orderflow_analyzer= OrderFlowAnalyzer()
        self.regime_detector   = MarketRegimeDetector()

        for sym in config.SYMBOLS_TO_TRADE.keys():
            self.ml_predictors[sym] = MLPredictor()
            self.mtf_analyzers[sym] = MultiTimeframeAnalyzer(sym)

    def start_trading(self, symbols: List[str]):
        """Load historical, train ML, then start websockets."""
        for sym in symbols:
            df= self.fetch_historical_data(sym, "1m", 500)
            df= FeatureEngineer.add_technical_features(df)
            self.ml_predictors[sym].train(df)

        self.start_symbol_sockets(symbols)
        logger.info("EnhancedTradingBot is running websockets for: "+ str(symbols))

        # You can add a main loop or real-time logic here if needed.

##############################################################################
# MAIN
##############################################################################
def main():
    config_path= os.path.join(os.path.dirname(__file__), "config.json")
    config= load_config(config_path)

    bot= EnhancedTradingBot(config)
    symbols= list(config.SYMBOLS_TO_TRADE.keys())

    def signal_handler(sig, frame):
        logger.warning(colored(f"Received signal {sig}, shutting down bot...", "yellow"))
        bot.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        bot.start_trading(symbols)
        logger.info("Main loop: press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.shutdown()
    except Exception as e:
        logger.error(colored(f"Unexpected error in main loop: {e}", "red"), exc_info=True)
        bot.shutdown()

if __name__=="__main__":
    main()
