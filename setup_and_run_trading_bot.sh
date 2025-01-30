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
    echo "Ensuring '.env' and 'trading_bot.log' are in .gitignore..."

    # Define the entries to add
    local entries=(".env" "trading_bot.log")

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
        source .venv/bin/activate
        echo "Virtual environment activated."
    else
        echo "Virtual environment is already activated."
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

    # Optionally, you can tail the log file to monitor in real-time
    echo "Press Ctrl+C to stop monitoring the logs."
    echo "Alternatively, check '$log_file' for detailed logs."

    # Wait for the bot process to finish
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

# Step 5: Run the Trading Bot
run_trading_bot

# =============================================================================
# End of Script
# =============================================================================
