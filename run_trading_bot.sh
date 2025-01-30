#!/bin/bash

echo "Starting run_trading_bot.sh script..."

# Unset proxy environment variables correctly
unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
echo "Proxy environment variables unset."

# Verify that proxy variables are unset
echo "Verifying proxy variables:"
echo "HTTP_PROXY: $HTTP_PROXY"
echo "http_proxy: $http_proxy"
echo "HTTPS_PROXY: $HTTPS_PROXY"
echo "https_proxy: $https_proxy"

# Ensure .env is not exposed in Git
echo "Ensuring .env is in .gitignore..."
echo ".env
trading_bot.log" > .gitignore

# Activate virtual environment if not already activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment is already activated."
else
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

echo "Starting the trading bot..."

# Run the trading bot and capture any errors
python trading_bot.py
