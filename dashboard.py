import os
import sys
import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger("CryptoLionBotDashboard")
logger.setLevel(logging.DEBUG)

# Prevent adding multiple handlers if they already exist
if not logger.handlers:
    # Console Formatter for terminal output
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

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
def load_config(config_path: str) -> ConfigModel:
    try:
        with open(config_path, 'r') as f:
            config_json = json.load(f)
        config = ConfigModel(**config_json)
        logger.debug("Configuration loaded and validated successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        sys.exit(1)
    except ValidationError as ve:
        logger.error(f"Configuration validation error: {ve}")
        sys.exit(1)
    except json.JSONDecodeError as je:
        logger.error(f"Invalid JSON format in configuration file: {je}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        sys.exit(1)

# =============================================================================
# DASHBOARD FUNCTIONS
# =============================================================================
def load_trade_log(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        return df
    except FileNotFoundError:
        logger.error(f"Trade log file '{file_path}' not found.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading trade log: {e}")
        return pd.DataFrame()

def load_backtest_results(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logger.error(f"Backtest results file '{file_path}' not found.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading backtest results: {e}")
        return pd.DataFrame()

def plot_pnl_per_symbol(backtest_df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=backtest_df.index, y="Total PnL", data=backtest_df)
    plt.title("Total PnL per Symbol")
    plt.xlabel("Symbol")
    plt.ylabel("PnL")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def plot_trade_pnl(trade_df: pd.DataFrame):
    if trade_df.empty:
        st.write("No trades to display.")
        return
    trade_df = trade_df[trade_df["action"] == "CLOSE"]
    if trade_df.empty:
        st.write("No closed trades to display.")
        return
    plt.figure(figsize=(12, 6))
    sns.histplot(trade_df["pnl"], bins=30, kde=True)
    plt.title("Distribution of Trade PnL")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

def main_dashboard():
    st.title("CryptoLionBot Dashboard")
    
    # Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)
    
    # Load Trade Log
    trade_log_path = "trade_log.csv"
    trade_df = load_trade_log(trade_log_path)
    
    # Load Backtest Results
    backtest_results_path = "backtest_results.csv"  # Modify if different
    backtest_df = load_backtest_results(backtest_results_path)
    
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    options = ["Overview", "Trade Log", "Backtest Results", "Performance Metrics"]
    choice = st.sidebar.radio("Go to", options)
    
    if choice == "Overview":
        st.header("Bot Overview")
        st.write(f"**Starting Balance:** {config.STARTING_BALANCE}")
        st.write(f"**Minimum Notional:** {config.MIN_NOTIONAL}")
        st.write(f"**Number of Symbols Traded:** {len(config.SYMBOLS_TO_TRADE)}")
        st.write("**Strategies Implemented:**")
        for strategy in config.STRATEGIES.keys():
            st.write(f"- {strategy}")
    
    elif choice == "Trade Log":
        st.header("Trade Log")
        if trade_df.empty:
            st.write("No trade data available.")
        else:
            st.dataframe(trade_df)
            st.download_button(
                label="Download Trade Log as CSV",
                data=trade_df.to_csv(index=False).encode('utf-8'),
                file_name='trade_log.csv',
                mime='text/csv',
            )
    
    elif choice == "Backtest Results":
        st.header("Backtest Results")
        if backtest_df.empty:
            st.write("No backtest data available.")
        else:
            st.dataframe(backtest_df)
            plot_pnl_per_symbol(backtest_df)
            st.download_button(
                label="Download Backtest Results as CSV",
                data=backtest_df.to_csv(index=False).encode('utf-8'),
                file_name='backtest_results.csv',
                mime='text/csv',
            )
    
    elif choice == "Performance Metrics":
        st.header("Performance Metrics")
        if trade_df.empty:
            st.write("No trade data available.")
        else:
            total_pnl = trade_df["pnl"].sum()
            total_trades = len(trade_df)
            win_trades = len(trade_df[trade_df["pnl"] > 0])
            loss_trades = len(trade_df[trade_df["pnl"] < 0])
            win_rate = (win_trades / loss_trades) * 100 if loss_trades > 0 else 0.0
    
            st.metric("Total PnL", f"{total_pnl:.2f}")
            st.metric("Total Trades", f"{total_trades}")
            st.metric("Win Trades", f"{win_trades}")
            st.metric("Loss Trades", f"{loss_trades}")
            st.metric("Win Rate (%)", f"{win_rate:.2f}%")
    
            plot_trade_pnl(trade_df)

if __name__ == "__main__":
    main_dashboard()
