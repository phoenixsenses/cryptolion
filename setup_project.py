# setup_project.py

import os

def create_directories(base_path):
    """
    Creates the necessary directories for the CryptoLion project.
    """
    directories = [
        base_path,
        os.path.join(base_path, 'data')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def create_file(file_path, content):
    """
    Creates a file at the specified path with the given content.
    If the file already exists, it will be overwritten.
    """
    with open(file_path, 'w') as file:
        file.write(content)
    print(f"Created file: {file_path}")

def main():
    base_path = 'CryptoLion'
    
    # Step 1: Create Directories
    create_directories(base_path)
    
    # Step 2: Define File Contents
    
    # dashboard.py Content
    dashboard_py_content = '''\
import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import os
import json
from datetime import datetime
from pytz import timezone

# Function to initialize the database if it doesn't exist
def init_db(db_path='data/trading_bot.db'):
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                pnl REAL,
                reason TEXT
            )
        ''')
        conn.commit()
        conn.close()
        st.success(f"Database '{{db_path}}' created successfully.")

# Function to load trades from SQLite
@st.cache_data(ttl=60)  # Cache data for 60 seconds
def load_trades(db_path='data/trading_bot.db'):
    if not os.path.exists(db_path):
        st.error(f"Database file '{{db_path}}' does not exist!")
        return pd.DataFrame()
    
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except sqlite3.Error as e:
        st.error(f"Database error: {{e}}")
        return pd.DataFrame()

# Function to load config
@st.cache_data(ttl=300)  # Cache config for 5 minutes
def load_config(config_path='config.json'):
    if not os.path.exists(config_path):
        st.error(f"Config file '{{config_path}}' not found!")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        st.error(f"Error parsing config file: {{e}}")
        return {}

# Function to get server time
def get_server_time(tz_str='UTC'):
    try:
        tz = timezone(tz_str)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception as e:
        st.error(f"Error getting server time: {{e}}")
        return "N/A"

# Initialize database if missing
init_db()

# Load data
trades_df = load_trades()
config = load_config()

# Sidebar for navigation
st.sidebar.title("CryptoLion Dashboard")
page = st.sidebar.radio("Navigate to", ["Overview", "Trade History", "Performance Metrics", "Active Positions"])

# Add a refresh button
if st.sidebar.button("Refresh Data"):
    st.caching.clear_cache()
    st.experimental_rerun()

# Overview Page
if page == "Overview":
    st.title("CryptoLion Trading Bot Dashboard")
    st.write("Welcome to your Trading Bot Dashboard. Monitor your trades and performance in real-time.")

    # Display Current Balance
    starting_balance = config.get("STARTING_BALANCE", 1000.0)
    if not trades_df.empty:
        balance = starting_balance + trades_df['pnl'].sum()
    else:
        balance = starting_balance
    st.metric("Current Balance", f"${balance:,.2f}")

    # Display Server Time
    server_tz = config.get("SERVER_TIMEZONE", "UTC")
    server_time = get_server_time(server_tz)
    st.info(f"Server Time ({server_tz}): {server_time}")

    # Additional Overview Metrics (Optional)
    if not trades_df.empty:
        total_trades = len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        st.write("### Quick Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Trades", f"{total_trades}")
        col2.metric("Total PnL", f"${total_pnl:,.2f}")
        col3.metric("Win Rate", f"{win_rate:.2f}%")
    else:
        st.write("No trades executed yet.")

# Trade History Page
elif page == "Trade History":
    st.title("Trade History")
    if not trades_df.empty:
        st.dataframe(trades_df[['timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price', 'pnl', 'reason']])

        # Plot PnL Over Time
        pnl_over_time = trades_df.sort_values('timestamp').copy()
        pnl_over_time['Cumulative PnL'] = pnl_over_time['pnl'].cumsum()
        fig = px.line(pnl_over_time, x='timestamp', y='Cumulative PnL', title='Cumulative PnL Over Time')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No trades executed yet.")

# Performance Metrics Page
elif page == "Performance Metrics":
    st.title("Performance Metrics")
    if not trades_df.empty:
        # Win Rate
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.2f}% ({winning_trades}/{total_trades})")

        # Total PnL
        total_pnl = trades_df['pnl'].sum()
        st.metric("Total PnL", f"${total_pnl:,.2f}")

        # Average PnL per Trade
        avg_pnl = trades_df['pnl'].mean()
        st.metric("Average PnL per Trade", f"${avg_pnl:,.2f}")

        # Maximum Drawdown (Optional)
        pnl_over_time = trades_df.sort_values('timestamp').copy()
        pnl_over_time['Cumulative PnL'] = pnl_over_time['pnl'].cumsum()
        pnl_over_time['Peak'] = pnl_over_time['Cumulative PnL'].cummax()
        pnl_over_time['Drawdown'] = pnl_over_time['Peak'] - pnl_over_time['Cumulative PnL']
        max_drawdown = pnl_over_time['Drawdown'].max()
        st.metric("Maximum Drawdown", f"${max_drawdown:,.2f}")

        # Plot Distribution of PnL
        fig = px.histogram(trades_df, x='pnl', nbins=50, title='PnL Distribution', labels={'pnl': 'PnL ($)'})
        st.plotly_chart(fig, use_container_width=True)

        # Additional Metrics (Optional)
        st.write("### Trade Duration")
        if 'exit_time' in trades_df.columns:
            trades_df['duration'] = pd.to_datetime(trades_df['exit_time']) - trades_df['timestamp']
            trades_df['duration_seconds'] = trades_df['duration'].dt.total_seconds()
            fig_duration = px.histogram(trades_df, x='duration_seconds', nbins=50, title='Trade Duration Distribution', labels={'duration_seconds': 'Duration (seconds)'})
            st.plotly_chart(fig_duration, use_container_width=True)
        else:
            st.info("Trade duration data not available.")
    else:
        st.warning("No trades executed yet.")

# Active Positions Page
elif page == "Active Positions":
    st.title("Active Positions")
    if not trades_df.empty:
        # Assuming active positions have 'exit_price' as NULL or NaN
        active_positions = trades_df[trades_df['exit_price'].isnull()].copy()
        if not active_positions.empty:
            active_positions = active_positions[['timestamp', 'symbol', 'side', 'quantity', 'entry_price', 'pnl', 'reason']]
            st.dataframe(active_positions.reset_index(drop=True))

            # Plot Unrealized PnL for Active Positions
            fig = px.bar(active_positions, x='symbol', y='pnl', color='side',
                        title='Unrealized PnL by Symbol',
                        labels={'pnl': 'Unrealized PnL ($)', 'symbol': 'Symbol', 'side': 'Side'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No active positions currently.")
    else:
        st.warning("No trades data available.")

# Footer
st.markdown("---")
st.write("© 2025 CryptoLion. All rights reserved.")
'''

    # init_db.py Content
    init_db_py_content = '''\
import sqlite3
import os

def initialize_database(db_path='data/trading_bot.db'):
    if os.path.exists(db_path):
        print(f"Database '{{db_path}}' already exists.")
        return
    
    # Connect to SQLite (this will create the database file)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the 'trades' table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity REAL NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            pnl REAL,
            reason TEXT
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print(f"Database '{{db_path}}' created successfully with 'trades' table.")

if __name__ == "__main__":
    initialize_database()
'''

    # populate_db.py Content
    populate_db_py_content = '''\
import sqlite3
from datetime import datetime
import os

def populate_database(db_path='data/trading_bot.db'):
    if not os.path.exists(db_path):
        print(f"Database '{{db_path}}' does not exist. Please run 'init_db.py' first.")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Sample trade data
    sample_trades = [
        (datetime.now().isoformat(), 'BTC/USD', 'buy', 0.5, 30000.0, 31000.0, 500.0, 'Take Profit'),
        (datetime.now().isoformat(), 'ETH/USD', 'sell', 2.0, 2000.0, 1900.0, -200.0, 'Stop Loss'),
        (datetime.now().isoformat(), 'LTC/USD', 'buy', 10.0, 150.0, None, None, 'Pending')
    ]

    cursor.executemany('''
        INSERT INTO trades (timestamp, symbol, side, quantity, entry_price, exit_price, pnl, reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_trades)

    conn.commit()
    conn.close()
    print("Sample trades inserted into 'trades' table.")

if __name__ == "__main__":
    populate_database()
'''

    # config.json Content
    config_json_content = '''{
    "STARTING_BALANCE": 1000.0,
    "SERVER_TIMEZONE": "UTC"
}
'''

    # requirements.txt Content
    requirements_txt_content = '''\
streamlit==1.15.0
pandas==1.5.3
plotly==5.15.0
pytz==2023.3
watchdog==3.0.0
'''

    # Step 3: Create Files with Content
    files_to_create = {
        'dashboard.py': dashboard_py_content,
        'init_db.py': init_db_py_content,
        'populate_db.py': populate_db_py_content,
        'config.json': config_json_content,
        'requirements.txt': requirements_txt_content
    }

    for filename, content in files_to_create.items():
        file_path = os.path.join(base_path, filename)
        create_file(file_path, content)

    print("\nSetup completed successfully!")

if __name__ == "__main__":
    main()
