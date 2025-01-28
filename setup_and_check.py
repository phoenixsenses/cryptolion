import subprocess
import sys

def install_and_import(package, module_name=None):
    """
    Install the package if not already installed and import it.
    """
    module_name = module_name if module_name else package
    try:
        __import__(module_name)
        print(f"'{package}' is already installed.")
    except ImportError:
        print(f"'{package}' is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"'{package}' has been installed successfully.")

# List of required packages
required_packages = {
    "setuptools": "pkg_resources",  # For pkg_resources
    "pandas_ta": "pandas_ta",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
}

# Check and install missing packages
for package, module in required_packages.items():
    install_and_import(package, module)

print("All dependencies are resolved!")

# Optional: Ensure the environment is ready
print("Verifying 'pandas_ta' functionality...")
try:
    import pandas_ta as ta
    print("pandas_ta is functioning correctly!")
except Exception as e:
    print(f"An error occurred while verifying pandas_ta: {e}")

print("Setup complete!")
