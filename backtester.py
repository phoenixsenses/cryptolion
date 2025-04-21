#!/usr/bin/env python3
"""
backtester.py
A quick-and-dirty crypto strategy sandbox.

Subcommands:
  backtest    – run a single parameter set (with optional exit rules) and plot results
  optimize    – grid search over params (multiprocessing)
  batch       – run optimize on each CSV in a folder
  walk        – train/test walk-forward validation with optional plot
  report      – generate heatmap + scatter from grid results
  portfolio   – aggregate many symbols into a portfolio curve (costs/cash/rf)
  resample    – resample OHLCV CSVs to a target timeframe via pandas

Add --verbose for debug logging.
"""

from __future__ import annotations
import argparse, json, itertools, multiprocessing as mp, logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------- Logging Setup ----------------
logger = logging.getLogger(__name__)

# ----------------------- Helpers -----------------------
def load_df(path: str | Path) -> pd.DataFrame:
    logger.debug(f"Loading CSV {path}")
    df = pd.read_csv(path)
    for col in ("timestamp", "date", "Datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: "timestamp"})
            break
    else:
        logger.error(f"Missing date/timestamp in {path}")
        raise ValueError(f"Missing date/timestamp in {path}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if not all(c in df.columns for c in ("open","high","low","close","volume")):
        logger.error(f"Missing OHLCV in {path}")
        raise ValueError(f"Missing OHLCV columns in {path}")
    return df

# ---------------- Simulation & Metrics ----------------
def simulate(df: pd.DataFrame, params: Dict) -> tuple[pd.DataFrame, Dict]:
    fast = params.get("fast")
    slow = params.get("slow")
    sig = params.get("sig", 9)
    fee = params.get("fee", 0.0)
    slip = params.get("slippage", 0.0)
    exit_rule = params.get("exit")
    atr_mult = params.get("atr_mult")
    profit_target = params.get("profit_target")
    time_stop = params.get("time_stop")

    df = df.copy()
    # Indicators
    df["ema_fast"] = df["close"].ewm(span=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["signal"] = df["macd"].ewm(span=sig).mean()
    # ATR calculation for exits
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    df["tr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = df["tr"].rolling(slow).mean()

    # Simulation
    df["pos"] = 0
    equity = [1.0]
    ret = [0.0]
    position = 0
    entry_price = 0.0
    bars_held = 0

    for i in range(1, len(df)):
        price = df.at[i, "close"]
        prev_price = df.at[i-1, "close"]
        macd = df.at[i, "macd"]
        sigl = df.at[i, "signal"]
        # entry
        if position == 0 and macd > sigl:
            position = 1
            entry_price = price
            bars_held = 0
        # exit
        if position == 1 and exit_rule:
            bars_held += 1
            if exit_rule == "atr" and atr_mult and not pd.isna(df.at[i, "atr"]):
                if price < entry_price - atr_mult * df.at[i, "atr"]:
                    position = 0
            if exit_rule == "profit" and profit_target:
                if price >= entry_price * (1 + profit_target):
                    position = 0
            if exit_rule == "time" and time_stop and bars_held >= time_stop:
                position = 0
        # calculate return
        r = (price - prev_price) / prev_price * position
        trades = abs(position - df.at[i-1, "pos"] if i-1 >= 0 else position)
        r -= fee * trades
        r -= slip * trades
        ret.append(r)
        equity.append(equity[-1] * (1 + r))
        df.at[i, "pos"] = position

    df["ret"] = ret
    df["equity"] = equity

    total_return = equity[-1] - 1
    daily = pd.Series(ret)
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() else np.nan
    max_dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()
    calmar = total_return / -max_dd if max_dd < 0 else np.nan
    metrics = {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }
    logger.debug(f"Metrics: {metrics}")
    return df, metrics

# Worker for optimize

def _optimize_worker(task):
    df, params = task
    _, m = simulate(df, params)
    return {**params, **m}

# ---------------- Commands ----------------
def backtest_cmd(args):
    df = load_df(args.file)
    params = json.loads(args.params)
    if args.exit:
        params['exit'] = args.exit
        if args.atr_mult is not None: params['atr_mult'] = args.atr_mult
        if args.profit_target is not None: params['profit_target'] = args.profit_target
        if args.time_stop is not None: params['time_stop'] = args.time_stop
    df_sim, metrics = simulate(df, params)
    logger.info(f"Backtest metrics: {metrics}")
    if args.save_plots:
        plt.figure(); plt.plot(df_sim['timestamp'], df_sim['equity']); plt.title('Equity'); plt.savefig('equity.png'); plt.close()
        plt.figure(); plt.hist(df_sim['ret'], bins=50); plt.title('Return Distribution'); plt.savefig('ret_hist.png'); plt.close()
        logger.info('Saved plots → equity.png, ret_hist.png')


def optimize_cmd(args):
    df = load_df(args.file)
    grid = json.loads(args.param_grid)
    combos = [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]
    tasks = [(df, combo) for combo in combos]
    with mp.Pool(args.processes or mp.cpu_count()) as pool:
        results = pool.map(_optimize_worker, tasks)
    pd.DataFrame(results).to_csv(args.output, index=False)
    logger.info(f"Saved optimization results to {args.output}")


def batch_cmd(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(args.folder).glob('*.csv'):
        try:
            ns = argparse.Namespace(file=str(f), param_grid=args.param_grid, processes=args.processes, output=out_dir/f"{f.stem}_grid.csv")
            optimize_cmd(ns)
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")


def walk_cmd(args):
    df = load_df(args.file)
    split = int(len(df)*args.train_frac)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    grid = json.loads(args.param_grid)
    combos = [dict(zip(grid.keys(), vals)) for vals in itertools.product(*grid.values())]
    best = max(combos, key=lambda c: simulate(train, c)[1][args.metric])
    logger.info(f"Best params: {best}")
    sim_tr,_ = simulate(train, best)
    sim_te,_ = simulate(test, best)
    out_df = pd.DataFrame([{**simulate(train,best)[1],'set':'train'},{**simulate(test,best)[1],'set':'test'}])
    out_df.to_csv(args.output,index=False)
    logger.info(f"Saved walk-forward results to {args.output}")
    if args.plot:
        plt.figure(); plt.plot(sim_tr['timestamp'],sim_tr['equity'],label='Train'); plt.plot(sim_te['timestamp'],sim_te['equity'],label='Test'); plt.legend(); plt.title('Walk-forward Equity'); plt.savefig(args.output.replace('.csv','.png')); plt.close()
        logger.info(f"Saved equity plot to {args.output.replace('.csv','.png')}")


def report_cmd(args):
    df = pd.read_csv(args.grid_file)
    pt = df.pivot_table(index=args.x,columns=args.y,values=args.metric)
    plt.figure(); __import__('seaborn').heatmap(pt,annot=True,fmt='.2f'); plt.title(f"{args.metric} heatmap"); plt.savefig(args.output); plt.close()
    plt.figure(); plt.scatter(df[args.x],df[args.y],c=df[args.metric]); plt.colorbar(label=args.metric); plt.savefig(args.output.replace('.png','_scatter.png')); plt.close()
    logger.info(f"Saved report: {args.output} & scatter")


def portfolio_cmd(args):
    logger.debug(f"Starting portfolio_cmd with args: {args}")
    cash_w = args.cash_weight
    rf_rate = args.rf_rate
    files = list(Path(args.folder).glob("*.csv"))
    if not files:
        logger.error(f"No CSVs found in {args.folder}")
        raise SystemExit(f"No CSVs found in {args.folder}")
    rets = []
    for f in files:
        try:
            df = load_df(f)
            s = df["close"].pct_change().rename(f.stem)
            s.index = df["timestamp"]
            rets.append(s)
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
    if not rets:
        logger.error("All files were skipped – nothing to backtest.")
        raise SystemExit("All files were skipped – nothing to backtest.")
    returns_df = pd.concat(rets, axis=1).sort_index().fillna(0)
    if cash_w > 0:
        per_period_rf = rf_rate / 365
        returns_df["__cash__"] = per_period_rf
    rebalance_idx = returns_df.resample(args.rebalance).last().index
    n_cols = returns_df.shape[1]
    asset_count = n_cols - (1 if cash_w > 0 else 0)
    base_w = [(1 - cash_w) / asset_count] * asset_count
    if cash_w > 0:
        base_w.append(cash_w)
    old_w = base_w.copy()
    equity = [1.0]
    port_rets = []
    for date, row in returns_df.iterrows():
        if date in rebalance_idx:
            if args.weight == "equal":
                new_w = [(1 - cash_w) / asset_count] * asset_count
            else:
                vol = returns_df.iloc[:, :asset_count].loc[:date].rolling(30).std().iloc[-1]
                inv = 1.0 / vol.replace(0, np.nan)
                alloc = (inv / inv.sum() * (1 - cash_w)).tolist()
                new_w = alloc
            if cash_w > 0:
                new_w.append(cash_w)
            turnover = float(np.abs(np.array(new_w) - np.array(old_w)).sum())
            cost_factor = 1 - (args.fee + args.slippage) * turnover
            equity[-1] *= cost_factor
            old_w = new_w
        r = float(np.dot(old_w, row.values))
        port_rets.append(r)
        equity.append(equity[-1] * (1 + r))
    result_df = pd.DataFrame({"timestamp": returns_df.index, "ret": port_rets, "equity": equity[1:]})
    result_df.to_csv(args.out_file, index=False)
    total_return = equity[-1] - 1
    daily = pd.Series(port_rets)
    sharpe = (daily.mean() / daily.std()) * np.sqrt(365) if daily.std() else np.nan
    max_dd = (pd.Series(equity) / pd.Series(equity).cummax() - 1).min()
    calmar = total_return / -max_dd if max_dd < 0 else np.nan
    logger.info("Portfolio metrics (with cash, costs & rf growth):")
    logger.info(f"  total_return: {total_return:.4f}")
    logger.info(f"  sharpe      : {sharpe:.4f}")
    logger.info(f"  max_drawdown: {max_dd:.4f}")
    logger.info(f"  calmar      : {calmar:.4f}")
    logger.info(f"Saved portfolio equity to {args.out_file}")


def resample_cmd(args):
    logger.debug(f"Starting resample_cmd with args: {args}")
    src = Path(args.folder)
    dst = Path(args.out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.csv"):
        try:
            df = load_df(f).set_index("timestamp")
        except Exception as e:
            logger.warning(f"Skipping {f.name}: {e}")
            continue
        rule = args.rule
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }
        res = df.resample(rule).agg(agg).dropna()
        out_file = dst / f.name
        res.reset_index().to_csv(out_file, index=False)
        logger.info(f"Resampled {f.name} -> {out_file} at rule={rule}")
    logger.info(f"Resampled files to {args.out_dir}")

# ---------------- CLI Setup ----------------
def main():
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description="Tiny crypto back-tester and portfolio tool")
    parser.add_argument('--verbose', action='store_true', help='Enable debug logging')
    sub = parser.add_subparsers(dest='command', required=True)

    # backtest
    p = sub.add_parser('backtest', help='Run a single backtest')
    p.add_argument('--file', required=True, help='Path to OHLCV CSV')
    p.add_argument('--params', required=True, help='JSON string of strategy params')
    p.add_argument('--save-plots', action='store_true', help='Save equity and return histogram plots')
    p.add_argument('--exit', choices=['atr','profit','time'], help='Exit rule for positions')
    p.add_argument('--atr-mult', type=float, help='ATR multiplier for stop-loss')
    p.add_argument('--profit-target', type=float, help='Profit target fraction to exit')
    p.add_argument('--time-stop', type=int, help='Bar count-based exit')
    p.set_defaults(func=backtest_cmd)

    # optimize
    p = sub.add_parser('optimize', help='Grid search over strategy parameters')
    p.add_argument('--file', required=True, help='Path to OHLCV CSV')
    p.add_argument('--param-grid', required=True, help='JSON dict of lists for grid search')
    p.add_argument('--processes', type=int, default=0, help='Number of parallel processes')
    p.add_argument('--output', default='grid_results.csv', help='CSV file to save grid results')
    p.set_defaults(func=optimize_cmd)

    # batch
    p = sub.add_parser('batch', help='Batch optimize on all CSVs in a folder')
    p.add_argument('--folder', required=True, help='Folder containing input CSVs')
    p.add_argument('--param-grid', required=True, help='JSON grid search parameters')
    p.add_argument('--processes', type=int, default=0, help='Parallel processes count')
    p.add_argument('--out-dir', required=True, help='Output directory for per-file results')
    p.set_defaults(func=batch_cmd)

    # walk
    p = sub.add_parser('walk', help='Train/test walk-forward validation')
    p.add_argument('--file', required=True, help='Path to OHLCV CSV')
    p.add_argument('--train-frac', type=float, default=0.7, help='Fraction of data to use for training')
    p.add_argument('--param-grid', required=True, help='JSON dict of grid search parameters')
    p.add_argument('--metric', default='sharpe', help='Metric to select best params')
    p.add_argument('--output', default='walk_results.csv', help='CSV output for train/test metrics')
    p.add_argument('--plot', action='store_true', help='Save equity plot for train vs test')
    p.set_defaults(func=walk_cmd)

    # report
    p = sub.add_parser('report', help='Generate heatmap and scatter from grid CSV')
    p.add_argument('--grid-file', required=True, help='CSV file from optimize command')
    p.add_argument('--metric', required=True, help='Metric column to visualize')
    p.add_argument('--x', required=True, help='Parameter for x-axis')
    p.add_argument('--y', required=True, help='Parameter for y-axis')
    p.add_argument('--output', default='heatmap.png', help='Heatmap output filename')
    p.set_defaults(func=report_cmd)

    # portfolio
    p = sub.add_parser('portfolio', help='Aggregate multiple symbols into portfolio equity')
    p.add_argument('--folder', required=True, help='Folder containing symbol CSVs')
    p.add_argument('--weight', choices=['equal','vol'], default='equal', help='Asset weighting scheme')
    p.add_argument('--rebalance', default='M', help='Rebalance frequency (pandas rule)')
    p.add_argument('--fee', type=float, default=0.0, help='Per-trade fee fraction')
    p.add_argument('--slippage', type=float, default=0.0, help='Per-trade slippage fraction')
    p.add_argument('--cash-weight', type=float, default=0.0, help='Fraction of portfolio in cash')
    p.add_argument('--rf-rate', type=float, default=0.0, help='Annual risk-free rate for cash returns')
    p.add_argument('--out-file', default='portfolio_equity.csv', help='CSV output for portfolio equity')
    p.set_defaults(func=portfolio_cmd)

    # resample
    p = sub.add_parser('resample', help='Resample OHLCV CSVs to new timeframe')
    p.add_argument('--folder', required=True, help='Input folder of raw CSVs')
    p.add_argument('--rule', required=True, help='Pandas resample rule, e.g. 15T, 1H, 1D')
    p.add_argument('--out-dir', required=True, help='Output folder for resampled CSVs')
    p.set_defaults(func=resample_cmd)

    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    args.func(args)

if __name__=='__main__':
    main()
