#!/bin/bash

# =============================================================================
# Script: fix_bot.sh
# Description: Fixes the trading bot by downgrading python-binance and numpy,
#              modifying trading_bot.py to include https_proxy attribute,
#              reinstalling dependencies, and restarting the trading bot.
# Author: ChatGPT
# Date: 2025-01-30
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e
set -x  # Enable debugging

# Function to downgrade python-binance to 1.0.25
downgrade_python_binance() {
    echo "Downgrading python-binance to version 1.0.25..."
    pip install --upgrade python-binance==1.0.25
    echo "python-binance downgraded to 1.0.25."
}

# Function to modify trading_bot.py to set client.https_proxy = {}
fix_trading_bot_py() {
    echo "Updating trading_bot.py to set client.https_proxy = {}..."

    # Determine OS type
    OS_TYPE="$(uname)"
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        # macOS uses BSD sed
        # Check if 'client.https_proxy = {}' is already present
        if ! grep -q "client.https_proxy = {}" trading_bot.py; then
            # Insert the line after 'client = Client(' line
            sed -i '' '/client = Client/i \
    client.https_proxy = {}
    ' trading_bot.py
            echo "Added 'client.https_proxy = {}' to trading_bot.py."
        else
            echo "'client.https_proxy = {}' is already present in trading_bot.py."
        fi
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        # Linux uses GNU sed
        if ! grep -q "client.https_proxy = {}" trading_bot.py; then
            sed -i '/client = Client/i \    client.https_proxy = {}' trading_bot.py
            echo "Added 'client.https_proxy = {}' to trading_bot.py."
        else
            echo "'client.https_proxy = {}' is already present in trading_bot.py."
        fi
    else
        echo "Unsupported OS: $OS_TYPE. Please modify trading_bot.py manually."
        exit 1
    fi
}

# Function to reinstall dependencies
reinstall_dependencies() {
    echo "Reinstalling dependencies from requirements.txt..."
    pip install --force-reinstall -r requirements.txt
    echo "Dependencies reinstalled."
}

# Function to resolve numpy and scipy dependency conflict
resolve_numpy_scipy_conflict() {
    echo "Resolving numpy and scipy dependency conflict..."

    # Check installed numpy version
    INSTALLED_NUMPY_VERSION=$(pip show numpy | grep Version | awk '{print $2}')
    echo "Current numpy version: $INSTALLED_NUMPY_VERSION"

    # Desired numpy version compatible with scipy 1.11.2
    DESIRED_NUMPY_VERSION="1.26.4"  # Updated to latest 1.x version

    if [[ "$INSTALLED_NUMPY_VERSION" == "2."* ]]; then
        echo "Detected incompatible numpy version: $INSTALLED_NUMPY_VERSION"
        echo "Downgrading numpy to $DESIRED_NUMPY_VERSION..."
        pip install numpy==$DESIRED_NUMPY_VERSION
        echo "numpy downgraded to $DESIRED_NUMPY_VERSION."
    elif [[ "$(echo "$INSTALLED_NUMPY_VERSION < 1.21.6" | bc)" -eq 1 ]]; then
        echo "Detected outdated numpy version: $INSTALLED_NUMPY_VERSION"
        echo "Upgrading numpy to $DESIRED_NUMPY_VERSION..."
        pip install numpy==$DESIRED_NUMPY_VERSION
        echo "numpy upgraded to $DESIRED_NUMPY_VERSION."
    else
        echo "numpy version is compatible."
    fi

    # Ensure scipy is installed at the required version
    REQUIRED_SCIPY_VERSION="1.11.2"
    INSTALLED_SCIPY_VERSION=$(pip show scipy | grep Version | awk '{print $2}')

    if [[ "$INSTALLED_SCIPY_VERSION" != "$REQUIRED_SCIPY_VERSION" ]]; then
        echo "Installing scipy version $REQUIRED_SCIPY_VERSION..."
        pip install scipy==$REQUIRED_SCIPY_VERSION
        echo "scipy installed at version $REQUIRED_SCIPY_VERSION."
    else
        echo "scipy version is already $REQUIRED_SCIPY_VERSION."
    fi
}

# Function to restart the trading bot
restart_trading_bot() {
    echo "Restarting the trading bot..."
    # Find existing bot PIDs and kill them
    BOT_PIDS=$(pgrep -f trading_bot.py)
    if [ ! -z "$BOT_PIDS" ]; then
        echo "Killing existing bot PIDs: $BOT_PIDS"
        kill -9 $BOT_PIDS
    else
        echo "No existing trading bot processes found."
    fi
    # Start the bot using setup_and_run_trading_bot.sh
    ./setup_and_run_trading_bot.sh
    echo "Trading bot restarted."
}

# Execute functions in the correct order
downgrade_python_binance
fix_trading_bot_py
resolve_numpy_scipy_conflict
reinstall_dependencies
restart_trading_bot

echo "Bot has been fixed and restarted successfully."
