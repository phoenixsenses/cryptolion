import subprocess
import sys

def test_help():
    res = subprocess.run([sys.executable, '-m', 'backtester', '--help'], capture_output=True)
    assert res.returncode == 0
    assert 'backtest' in res.stdout.decode()