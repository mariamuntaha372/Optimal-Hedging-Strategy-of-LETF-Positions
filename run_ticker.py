"""
Run the pipeline for any configured ticker and print the hedge-ratio sweep.

    python run_ticker.py MSOX
    python run_ticker.py MSTX
    python run_ticker.py PTIR
    python run_ticker.py --all

EGARCH windows, split handling, and any legacy universe filter come from
src/config.py, so every ticker runs through the identical engine.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import TICKER_CONFIGS, get_config, RESULTS_DIR
from src.data import load_all
from src.strategy import fit_egarch_signals, run_final_strategy_v21

HR_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
DVT = 16000
BAND = 0.25
ALPHA = 0.95


def run_one(ticker, save=True):
    cfg = get_config(ticker)

    print("=" * 72)
    print(f"{ticker} — {cfg['note']}")
    print("=" * 72)

    print("\n[1/4] Loading data ...")
    data = load_all(ticker)

    px = data["df_pricing"]
    print(f"\n      price rows : {len(px)}")
    print(f"      date range : {px.index.min().date()} -> {px.index.max().date()}")
    print(f"      option rows: {len(data['df_full']):,}")

    print("\n[2/4] Fitting EGARCH signal ...")
    data = fit_egarch_signals(data)

    n_episodes = int((data["trade_state"]["action"] == "ENTER").sum())
    if n_episodes == 0:
        print(f"\n  {ticker}: no entry signals fired -- nothing to backtest.")
        print("  Try a shorter egarch.fit_window, or a lower entry_pct.")
        return None

    print(f"\n[3/4] Hedge ratio sweep ({n_episodes} episodes) ...")
    rows = []
    for hr in HR_GRID:
        metrics, pnl, state, _ = run_final_strategy_v21(
            data, hedge_ratio=hr, dvt=DVT, restrike_band=BAND)
        rows.append({"Hedge ratio": hr, **metrics})
    results = pd.DataFrame(rows)
    print()
    print(results.to_string(index=False))

    print(f"\n[4/4] CVaR at {ALPHA:.0%} ...")
    cvar_rows = []
    for hr in HR_GRID:
        _, pnl, _, _ = run_final_strategy_v21(
            data, hedge_ratio=hr, dvt=DVT, restrike_band=BAND)
        live = pnl[pnl != 0]
        if len(live) == 0:
            continue
        var = np.percentile(live, (1 - ALPHA) * 100)
        cvar_rows.append({
            "Hedge ratio": hr,
            "VaR": round(var, 0),
            "CVaR": round(live[live <= var].mean(), 0),
            "Total P&L": round(live.sum(), 0),
        })
    cvar_df = pd.DataFrame(cvar_rows)
    print()
    print(cvar_df.to_string(index=False))

    if len(cvar_df):
        best = cvar_df.loc[cvar_df["CVaR"].idxmax(), "Hedge ratio"]
        print(f"\nCVaR-optimal hedge ratio: {best}")

    # Sanity check worth surfacing: hedging should not make the worst day worse.
    naked_worst = results.loc[results["Hedge ratio"] == 0.0, "Worst day"].iloc[0]
    bad = results[results["Worst day"] < naked_worst]
    if len(bad):
        print("\n  WARNING: these hedge ratios have a WORSE worst-day than the")
        print("  naked short, which should not happen if the hedge is working:")
        for _, r in bad.iterrows():
            print(f"    h={r['Hedge ratio']:.2f}  worst day {r['Worst day']:,.0f} "
                  f"vs naked {naked_worst:,.0f}")

    if save:
        out_dir = RESULTS_DIR / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_dir / "hedge_sweep.csv", index=False)
        cvar_df.to_csv(out_dir / "cvar.csv", index=False)
        print(f"\n  saved -> results/{ticker}/")

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", nargs="?", help=f"one of {sorted(TICKER_CONFIGS)}")
    ap.add_argument("--all", action="store_true", help="run every configured ticker")
    ap.add_argument("--no-save", action="store_true", help="skip writing results/")
    args = ap.parse_args()

    if args.all:
        tickers = list(TICKER_CONFIGS)
    elif args.ticker:
        tickers = [args.ticker.upper()]
    else:
        ap.error("give a ticker or --all")

    for i, t in enumerate(tickers):
        if i:
            print("\n")
        try:
            run_one(t, save=not args.no_save)
        except Exception as exc:
            print(f"\n  {t} FAILED: {type(exc).__name__}: {exc}")
            if not args.all:
                raise


if __name__ == "__main__":
    main()
