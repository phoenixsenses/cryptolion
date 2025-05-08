# CryptoLion

CryptoLion is a lightweight Python library for backtesting and running cryptocurrency trading strategies locally.

---

## Installation

```bash
pip install cryptolion
```

---

## Quickstart Example

```python
import pandas as pd
from cryptolion.ultra import MACrossover10_50
from cryptolion.engine import StrategyEngine

# Load OHLCV data (DataFrame with 'close' column and datetime index)
data = pd.read_csv('data/BTC_USD.csv', parse_dates=['Date'], index_col='Date')

# Initialize and run the engine
engine = StrategyEngine()
engine.run(data, strategy=MACrossover10_50(), fee=0.001)

# Access the equity curve and summary statistics
equity = engine.equity
print(equity)
print(equity.describe())
```

---

## Release Information

The latest release (0.1.1) is available on PyPI:

[https://pypi.org/project/cryptolion/0.1.1/](https://pypi.org/project/cryptolion/0.1.1/)

---

## Contributing

For bug reports or feature requests, please open an issue on GitHub:

[https://github.com/phoenixsenses/cryptolion](https://github.com/phoenixsenses/cryptolion)

Pull requests are welcome!
