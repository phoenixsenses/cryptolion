import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} has been installed successfully.")

# List of required packages
required_packages = {
    "pandas_ta": "pandas_ta",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost"
}

# Check and install missing packages
for package_name, import_name in required_packages.items():
    install_and_import(import_name)

print("All required packages are installed and ready!")
