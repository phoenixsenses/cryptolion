import pandas as pd
from backtester import simulate


def test_simulate_flat_series():
    # build a trivial up‑only price series
    df = pd.DataFrame(
        {
            'timestamp': pd.date_range('2025-01-01', periods=5, freq='D'),
            'open': [1, 1, 1, 1, 1],
            'high': [1, 1, 1, 1, 1],
            'low': [1, 1, 1, 1, 1],
            'close': [1, 1, 1, 1, 1],
            'volume': [1, 1, 1, 1, 1],
        }
    )
    # run with any params—no movement means zero return
    _, metrics = simulate(df, {'fast': 2, 'slow': 3})
    assert metrics['total_return'] == 0.0
    assert isinstance(metrics['sharpe'], float)
