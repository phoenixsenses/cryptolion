import os
import sys
import inspect
import logging
from binance.client import Client  # Corrected import
from binance.websockets import BinanceSocketManager  # Corrected import
from binance.exceptions import BinanceAPIException

# =============================================================================
# LOGGING SETUP
# =============================================================================
logger = logging.getLogger("DebugBinanceSocket")
logger.setLevel(logging.DEBUG)

# Console Formatter for terminal output
console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# =============================================================================
# DEBUGGING FUNCTIONS
# =============================================================================
def inspect_socket_methods(bsm_instance):
    methods = [method for method in dir(bsm_instance) if callable(getattr(bsm_instance, method))]
    logger.info(f"Available methods in BinanceSocketManager: {methods}")
    
    # Check for specific methods
    if hasattr(bsm_instance, 'start_kline_socket'):
        logger.info("start_kline_socket method is available.")
    else:
        logger.error("start_kline_socket method is NOT available.")
    
    if hasattr(bsm_instance, 'start_depth_socket'):
        logger.info("start_depth_socket method is available.")
    else:
        logger.error("start_depth_socket method is NOT available.")
    
    if hasattr(bsm_instance, 'stop'):
        logger.info("stop method is available.")
    else:
        logger.error("stop method is NOT available.")

def process_kline_message(msg):
    logger.info(f"Kline Message Received: {msg}")

def process_depth_message(msg):
    logger.info(f"Depth Message Received: {msg}")

def test_individual_sockets(bsm_instance):
    # Define the symbol and stream types
    symbol = "btcusdt"
    
    try:
        # Start Kline Socket
        logger.info(f"Starting Kline socket for {symbol}@kline_1m")
        conn_key_kline = bsm_instance.start_kline_socket(symbol, process_kline_message)
        logger.info(f"Kline socket started with connection key: {conn_key_kline}")
        
        # Start Depth Socket
        logger.info(f"Starting Depth socket for {symbol}@depth5@100ms")
        conn_key_depth = bsm_instance.start_depth_socket(symbol, process_depth_message)
        logger.info(f"Depth socket started with connection key: {conn_key_depth}")
        
        # Start the socket manager
        bsm_instance.start()
        logger.info("BinanceSocketManager started successfully with individual sockets.")
        
    except TypeError as te:
        logger.error(f"TypeError encountered: {te}")
    except BinanceAPIException as bae:
        logger.error(f"BinanceAPIException encountered: {bae}")
    except Exception as e:
        logger.error(f"Unexpected exception encountered: {e}")

def main():
    # =============================================================================
    # BINANCE CLIENT INITIALIZATION
    # =============================================================================
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not api_secret:
        logger.critical("BINANCE_API_KEY and BINANCE_SECRET_KEY must be set in environment variables.")
        sys.exit(1)

    # Initialize Binance Client
    try:
        client = Client(api_key, api_secret, testnet=True)  # Set testnet=True for testing
        logger.info("Binance Client initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize Binance Client: {e}", exc_info=True)
        sys.exit(1)

    # Initialize BinanceSocketManager
    try:
        bsm = BinanceSocketManager(client)
        logger.info("BinanceSocketManager initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize BinanceSocketManager: {e}", exc_info=True)
        sys.exit(1)

    # =============================================================================
    # INSPECT AVAILABLE SOCKET METHODS
    # =============================================================================
    inspect_socket_methods(bsm)

    # =============================================================================
    # TEST INDIVIDUAL SOCKETS
    # =============================================================================
    test_individual_sockets(bsm)

    # Keep the script running for a short period to receive any messages
    import time
    try:
        logger.info("Script will run for 10 seconds to receive messages...")
        time.sleep(10)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting...")
    finally:
        logger.info("Stopping BinanceSocketManager...")
        try:
            bsm.stop()  # Properly stop all sockets
            logger.info("BinanceSocketManager stopped successfully.")
        except AttributeError:
            logger.warning("BinanceSocketManager has no 'stop' method.")
        logger.info("Debugging script completed.")

if __name__ == "__main__":
    main()
