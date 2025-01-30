import sys
from binance import BinanceSocketManager
from binance.client import Client

def main():
    try:
        # Initialize the client with dummy keys (testnet)
        client = Client(api_key='dummy', api_secret='dummy', testnet=True)
        
        # Check if 'https_proxy' attribute exists
        has_https_proxy = hasattr(client, 'https_proxy')
        https_proxy_value = getattr(client, 'https_proxy', 'Not found')
        
        print("Client has https_proxy:", has_https_proxy)
        print("Client https_proxy:", https_proxy_value)
        
        # Initialize BinanceSocketManager
        bsm = BinanceSocketManager(client)
        
        # List all callable methods in BinanceSocketManager
        methods = [method for method in dir(bsm) if callable(getattr(bsm, method)) and not method.startswith("__")]
        
        print("Available methods in BinanceSocketManager:")
        for method in methods:
            print(f"- {method}")
    
    except Exception as e:
        print(f"Error initializing BinanceSocketManager: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
