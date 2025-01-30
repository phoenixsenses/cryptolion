#!/usr/bin/env python3

import os
import sys
import subprocess
from binance.client import Client

REQUIRED_PACKAGES = [
    "python-binance>=1.0.15",
    "pandas",
    "numpy",
    "pandas-ta",
    "scikit-learn",
    "textblob",
    "python-dotenv",
    "termcolor",
    "python-json-logger",
]

def install_missing_packages():
    """Install any missing Python packages."""
    for package in REQUIRED_PACKAGES:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            sys.exit(1)

def check_binance_testnet(api_key, secret_key):
    """Test connection to Binance Testnet."""
    client = Client(api_key, secret_key, testnet=True)
    try:
        server_time = client.get_server_time()
        print(f"✅ Connected to Binance Testnet: Server Time - {server_time['serverTime']}")
    except Exception as e:
        print(f"❌ Failed to connect to Binance Testnet: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🔍 Checking and installing missing dependencies...")
    install_missing_packages()

    print("\n🔍 Verifying Binance Testnet connection...")
    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not secret_key:
        print("❌ API key or secret key not found in environment variables.")
        sys.exit(1)

    check_binance_testnet(api_key, secret_key)
