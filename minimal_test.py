from binance.client import Client
from binance.websockets import BinanceSocketManager

# Initialize Client (use testnet for safety)
client = Client('your_api_key', 'your_secret_key', testnet=True)
bsm = BinanceSocketManager(client)

# List available methods
methods = [method for method in dir(bsm) if callable(getattr(bsm, method))]
print("Available methods in BinanceSocketManager:", methods)

# Check for 'multiplex_socket' and 'stop'
if hasattr(bsm, 'multiplex_socket'):
    print("multiplex_socket method is available.")
else:
    print("multiplex_socket method is NOT available.")

if hasattr(bsm, 'stop'):
    print("stop method is available.")
else:
    print("stop method is NOT available.")
