#!/bin/bash

# =============================================================================
# Script: setup_and_run_trading_bot.sh
# Description: Automates the setup and execution of the CryptoLion Trading Bot.
# Author: Gorkem Berkeyuksel
# Date: 2025-01-30
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# =============================================================================
# Function Definitions
# =============================================================================

# Function to display usage information
usage() {
    echo "Usage: $0 [-d] [-t]"
    echo "  -d    Enable debug mode"
    echo "  -t    Use Binance Testnet"
    exit 1
}

# Parse command-line options
DEBUG=0
TESTNET=0

while getopts ":dt" opt; do
  case ${opt} in
    d )
      DEBUG=1
      ;;
    t )
      TESTNET=1
      ;;
    \? )
      echo "Invalid Option: -$OPTARG" 1>&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Enable debugging if requested
if [ "$DEBUG" -eq 1 ]; then
    echo "Debug mode enabled."
    set -x
fi

# Function to unset proxy environment variables
unset_proxies() {
    echo "Unsetting proxy environment variables..."
    unset HTTP_PROXY
    unset http_proxy
    unset HTTPS_PROXY
    unset https_proxy
    echo "Proxy environment variables have been unset."
}

# Function to verify that proxy variables are unset
verify_unset_proxies() {
    echo "Verifying that proxy environment variables are unset:"
    echo "HTTP_PROXY: '${HTTP_PROXY}'"
    echo "http_proxy: '${http_proxy}'"
    echo "HTTPS_PROXY: '${HTTPS_PROXY}'"
    echo "https_proxy: '${https_proxy}'"
}

# Function to add entries to .gitignore if they don't already exist
update_gitignore() {
    echo "Ensuring '.env', 'trading_bot.log', and 'bot_output.log' are in .gitignore..."

    # Define the entries to add
    local entries=(".env" "trading_bot.log" "bot_output.log")

    # Iterate over each entry
    for entry in "${entries[@]}"; do
        # Check if the entry already exists in .gitignore
        if ! grep -Fxq "$entry" .gitignore; then
            echo "$entry" >> .gitignore
            echo "Added '$entry' to .gitignore."
        else
            echo "'$entry' is already present in .gitignore."
        fi
    done
}

# Function to activate the virtual environment if not already active
activate_virtualenv() {
    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "Activating virtual environment..."
        if [ -d ".venv" ]; then
            source .venv/bin/activate
            echo "Virtual environment activated."
        else
            echo "Virtual environment '.venv' not found. Creating a new one..."
            python3 -m venv .venv
            source .venv/bin/activate
            echo "Virtual environment created and activated."
        fi
    else
        echo "Virtual environment is already activated."
    fi
}

# Function to setup environment variables in .env
setup_env() {
    local env_file=".env"

    if [ ! -f "$env_file" ]; then
        echo "Creating '$env_file' for environment variables."
        touch "$env_file"
    fi

    # Check if API_KEY is set
    if ! grep -q "^BINANCE_API_KEY=" "$env_file"; then
        read -p "Enter your Binance API Key: " API_KEY
        echo "BINANCE_API_KEY=${API_KEY}" >> "$env_file"
        echo "BINANCE_API_KEY set in '$env_file'."
    else
        echo "BINANCE_API_KEY is already set in '$env_file'."
    fi

    # Check if API_SECRET is set
    if ! grep -q "^BINANCE_SECRET_KEY=" "$env_file"; then
        read -sp "Enter your Binance API Secret: " API_SECRET
        echo
        echo "BINANCE_SECRET_KEY=${API_SECRET}" >> "$env_file"
        echo "BINANCE_SECRET_KEY set in '$env_file'."
    else
        echo "BINANCE_SECRET_KEY is already set in '$env_file'."
    fi

    # Set USE_TESTNET flag
    if [ "$TESTNET" -eq 1 ]; then
        if grep -q "^USE_TESTNET=" "$env_file"; then
            sed -i.bak 's/^USE_TESTNET=.*/USE_TESTNET=true/' "$env_file"
            echo "Updated USE_TESTNET to true in '$env_file'."
        else
            echo "USE_TESTNET=true" >> "$env_file"
            echo "Added USE_TESTNET=true to '$env_file'."
        fi
    else
        if grep -q "^USE_TESTNET=" "$env_file"; then
            sed -i.bak 's/^USE_TESTNET=.*/USE_TESTNET=false/' "$env_file"
            echo "Updated USE_TESTNET to false in '$env_file'."
        else
            echo "USE_TESTNET=false" >> "$env_file"
            echo "Added USE_TESTNET=false to '$env_file'."
        fi
    fi
}

# Function to run the trading bot and capture logs
run_trading_bot() {
    echo "Starting the trading bot..."

    # Define log file
    local log_file="bot_output.log"

    # Run the trading bot in the background, capturing stdout and stderr
    python trading_bot.py > "$log_file" 2>&1 &

    # Get the Process ID (PID) of the background process
    local BOT_PID=$!

    echo "Trading bot started with PID: $BOT_PID"
    echo "Logs are being written to '$log_file'."

    # Optionally, tail the log file to monitor in real-time
    echo "Press Ctrl+C to stop monitoring the logs."
    echo "Alternatively, check '$log_file' for detailed logs."

    # Tail the log file
    tail -f "$log_file" &
    local TAIL_PID=$!

    # Function to handle script termination
    cleanup() {
        echo "Terminating trading bot and log monitoring..."
        kill $BOT_PID 2>/dev/null || true
        kill $TAIL_PID 2>/dev/null || true
        exit 0
    }

    # Trap Ctrl+C and other termination signals
    trap cleanup SIGINT SIGTERM

    # Wait for the trading bot process to finish
    wait $BOT_PID
}

# =============================================================================
# Main Script Execution
# =============================================================================

# Ensure the script is being run from the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Unset Proxy Environment Variables
unset_proxies

# Step 2: Verify that Proxy Environment Variables are Unset
verify_unset_proxies

# Step 3: Update .gitignore with .env and trading_bot.log
update_gitignore

# Step 4: Activate Virtual Environment
activate_virtualenv

# Step 5: Setup Environment Variables in .env
setup_env

# Step 6: Install Required Python Packages
echo "Installing required Python packages..."
pip install --upgrade pip
pip install -r requirements.txt
echo "Python packages installed successfully."

# Step 7: Run the Trading Bot
run_trading_bot

# =============================================================================
# End of Script
# =============================================================================
