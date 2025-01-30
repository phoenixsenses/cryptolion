#!/bin/bash

# Define the desired version
DESIRED_VERSION="1.0.27"

echo "Installing python-binance version $DESIRED_VERSION..."
pip install "python-binance==$DESIRED_VERSION"

if [ $? -eq 0 ]; then
    echo "Successfully installed python-binance==$DESIRED_VERSION."
else
    echo "Failed to install python-binance==$DESIRED_VERSION. Please check the available versions."
    exit 1
fi
