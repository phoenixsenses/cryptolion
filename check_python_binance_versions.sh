#!/bin/bash

echo "Fetching available versions of python-binance from PyPI..."
pip install pip-search 2>/dev/null || pip install pip-search

AVAILABLE_VERSIONS=$(pip search python-binance | grep -oP 'python-binance\s*\(\K[^)]+' | tr ',' '\n' | sort -V | uniq)

echo "Available versions:"
echo "$AVAILABLE_VERSIONS"
