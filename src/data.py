"""
Data loading and preparation.

Two sources, both reproducible from a clean clone:
  * prices  -- yfinance, for the LETF and its 1x underlying
  * options -- OPRA daily bars, committed as CSV under data/

The split correction lives HERE, inside load_prices(), driven by the ticker's
config. It used to be a manual patch applied after the pipeline ran, which
meant re-running the pipeline silently reverted it and corrupted every
downstream moneyness and IV calculation without raising an error.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from .config import (
    get_config, RISK_FREE_RATE, DIVIDEND_YIELD, MIN_VOLUME_FOR_IV,
    IV_SOLVER_HI, IV_CEILING_TOLERANCE, MIN_SURFACE_CONTRIBUTORS,
    BASKET_STRIKE_BAND_LOW, BASKET_STRIKE_BAND_HIGH,
)
from .pricing import assign_bucket, implied_vol


def _flatten(df):
    """yfinance returns a MultiIndex when given a list; collapse it."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def load_prices(ticker, verbose=True):
    """
    Download the LETF and its 1x underlying from yfinance.

    Returns a DataFrame indexed by date with:
      Last Price   -- RAW close. Used for option strikes, moneyness, and IV,
                      because contracts are quoted in then-current terms.
      Adj Price    -- split-adjusted close. Used for returns, vol, and
                      momentum, so a reverse split doesn't inject a fake
                      multi-hundred-percent daily return into the EGARCH fit.
      Volume, <hedge>_Price, Daily_Return, Realized_Vol, Price_MA90
    """
    cfg = get_config(ticker)
    hedge = cfg["hedge_ticker"]

    if verbose:
        print(f"[prices] downloading {ticker} + {hedge} from yfinance ...")

    raw = _flatten(yf.download(ticker, start=cfg["start"], end=cfg["end"],
                               auto_adjust=False, progress=False))
    adj = _flatten(yf.download(ticker, start=cfg["start"], end=cfg["end"],
                               auto_adjust=True, progress=False))
    hedge_raw = _flatten(yf.download(hedge, start=cfg["start"], end=cfg["end"],
                                     auto_adjust=False, progress=False))

    if len(raw) == 0:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker}. This is usually rate "
            f"limiting or a blocked datacenter IP -- retry, or run locally."
        )

    df = pd.DataFrame(index=pd.to_datetime(raw.index).normalize())
    df["Last Price"] = raw["Close"].values
    df["Adj Price"] = adj["Close"].reindex(raw.index).values
    df["Volume"] = raw["Volume"].values

    h = hedge_raw["Close"]
    h.index = pd.to_datetime(hedge_raw.index).normalize()
    df[f"{hedge}_Price"] = h.reindex(df.index)

    # ── split correction ───────────────────────────────────────────────────
    # When yfinance reports the raw close as post-split across the whole
    # history, pre-split days have to be divided down to match the strikes
    # the options were actually quoted against.
    if cfg["split_fix"] is not None:
        split_date, factor = cfg["split_fix"]
        pre = df.index < split_date
        n_adjusted = int(pre.sum())
        df.loc[pre, "Last Price"] = df.loc[pre, "Last Price"] / factor
        if verbose:
            print(f"[prices] split fix: divided {n_adjusted} pre-{split_date.date()} "
                  f"raw closes by {factor:g}")

    df["Daily_Return"] = df["Adj Price"].pct_change()
    df["Realized_Vol"] = df["Daily_Return"].rolling(20).std() * np.sqrt(252)
    df["Price_MA90"] = df["Adj Price"].rolling(90).mean()

    if verbose:
        print(f"[prices] {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    return df


def parse_osi(symbol):
    """
    Parse an OSI option symbol's trailing 15 chars: YYMMDD + C/P + 8-digit
    strike in thousandths. Returns (expiration, option_type, strike).
    """
    t = symbol.strip()[-15:]
    return (
        pd.Timestamp(f"20{t[:2]}-{t[2:4]}-{t[4:6]}"),
        t[6],
        int(t[7:]) / 1000,
    )


def load_options(ticker, verbose=True):
    """
    Load the OPRA daily-bar CSV for a ticker and derive contract fields.

    The CSV is the merged product of every OPRA pull for this ticker; merging
    happens once, in scripts/merge_opra.py, not here.
    """
    cfg = get_config(ticker)
    path = cfg["options_path"]

    if not path.exists():
        raise FileNotFoundError(
            f"Options CSV not found: {path}\n"
            f"Expected {cfg['options_csv']} in the data/ directory."
        )

    df = pd.read_csv(path)
    df["ts_event"] = pd.to_datetime(df["ts_event"], format="mixed", utc=True)
    df["ts_event"] = df["ts_event"].dt.tz_localize(None).dt.normalize()

    parsed = df["symbol"].apply(parse_osi)
    df["expiration"] = [p[0] for p in parsed]
    df["option_type"] = [p[1] for p in parsed]
    df["strike"] = [p[2] for p in parsed]
    df["dte"] = (df["expiration"] - df["ts_event"]).dt.days

    keep = ["ts_event", "symbol", "option_type", "expiration", "strike",
            "dte", "close", "volume"]
    df = df[keep]

    n_raw = len(df)

    # ── legacy filter ──────────────────────────────────────────────────────
    # Reproduces the original notebook's universe. Applied only to rows
    # BEFORE applies_before, because the OPRA extension was appended raw --
    # replicating that asymmetry is what makes the numbers match.
    lf = cfg.get("legacy_filter")
    if lf:
        cutover = lf.get("applies_before")
        if cutover is not None:
            old, new = df[df["ts_event"] < cutover], df[df["ts_event"] >= cutover]
        else:
            old, new = df, df.iloc[0:0]

        if lf.get("calls_only"):
            old = old[old["option_type"] == "C"]
        if lf.get("dte_range"):
            lo, hi = lf["dte_range"]
            old = old[(old["dte"] >= lo) & (old["dte"] <= hi)]

        df = pd.concat([old, new], ignore_index=True).sort_values("ts_event")
        df = df.reset_index(drop=True)

        if verbose:
            desc = []
            if lf.get("calls_only"):
                desc.append("calls only")
            if lf.get("dte_range"):
                desc.append(f"DTE {lf['dte_range'][0]}-{lf['dte_range'][1]}")
            where = f" before {cutover.date()}" if cutover is not None else ""
            print(f"[options] legacy filter ({', '.join(desc)}){where}: "
                  f"{n_raw:,} -> {len(df):,} rows")

    if verbose:
        print(f"[options] {len(df)} rows, "
              f"{df['ts_event'].min().date()} -> {df['ts_event'].max().date()}")
    return df


def build_iv_surface(df_options, df_prices, ticker, verbose=True):
    """
    Solve implied vol per (symbol, date) against the LETF's raw spot, then
    build a median-IV surface by (date, maturity bucket).

    Returns (df_iv_final, iv_lookup, iv_surface).
    """
    cfg = get_config(ticker)

    df = df_options
    if cfg["split_exclude"] is not None:
        lo, hi = cfg["split_exclude"]
        n_before = len(df)
        df = df[~df["ts_event"].between(lo, hi)]
        if verbose:
            print(f"[iv] excluded {n_before - len(df)} rows in split window "
                  f"{lo.date()}..{hi.date()}")

    df = df[df["volume"] >= MIN_VOLUME_FOR_IV].copy()
    df = df.merge(
        df_prices[["Last Price"]].rename(columns={"Last Price": "underlying_price"}),
        left_on="ts_event", right_index=True, how="inner",
    )

    # Collapse to one volume-weighted row per (symbol, date)
    rows = []
    for (sym, d), g in df.groupby(["symbol", "ts_event"]):
        w = g["volume"].sum()
        rows.append({
            "symbol": sym, "ts_event": d,
            "option_type": g["option_type"].iloc[0],
            "expiration": g["expiration"].iloc[0],
            "dte": g["dte"].iloc[0],
            "strike": g["strike"].iloc[0],
            "close_vwap": (np.average(g["close"], weights=g["volume"])
                           if w > 0 else g["close"].mean()),
            "volume": w,
            "underlying_price": g["underlying_price"].iloc[0],
        })
    collapsed = pd.DataFrame(rows)

    collapsed["intrinsic"] = collapsed.apply(
        lambda r: max(r["underlying_price"] - r["strike"], 0) if r["option_type"] == "C"
        else max(r["strike"] - r["underlying_price"], 0), axis=1)
    collapsed["below_intrinsic"] = collapsed["close_vwap"] < collapsed["intrinsic"]

    if verbose:
        print(f"[iv] solving {len(collapsed)} implied vols (slow) ...")

    collapsed["IV"] = collapsed.apply(
        lambda r: implied_vol(r["close_vwap"], r["underlying_price"], r["strike"],
                              r["dte"] / 365.0, RISK_FREE_RATE, DIVIDEND_YIELD,
                              r["option_type"]), axis=1)

    # Drop solutions that pinned to the solver ceiling -- those are brentq
    # failing to find a genuine root, not a real 500% vol.
    at_ceiling = (collapsed["IV"].notna()
                  & (collapsed["IV"] >= IV_SOLVER_HI - IV_CEILING_TOLERANCE))
    df_iv_final = collapsed[
        collapsed["IV"].notna() & ~collapsed["below_intrinsic"] & ~at_ceiling
    ].copy()

    iv_lookup = df_iv_final[["symbol", "ts_event", "IV"]].rename(
        columns={"ts_event": "ts_event_iv"})

    df_iv_final["bucket"] = df_iv_final["dte"].apply(assign_bucket)
    surf = (df_iv_final[df_iv_final["bucket"].notna()]
            .groupby(["ts_event", "bucket"])["IV"]
            .agg(IV_surface="median", n="count").reset_index())
    iv_surface = surf[surf["n"] >= MIN_SURFACE_CONTRIBUTORS][
        ["ts_event", "bucket", "IV_surface"]].copy()

    if verbose:
        print(f"[iv] clean IVs: {len(df_iv_final)} | surface cells: {len(iv_surface)}")
    return df_iv_final, iv_lookup, iv_surface


def build_universes(df_options, df_prices, iv_lookup, iv_surface, verbose=True):
    """
    Join options to prices and IV, then carve out the hedge candidate universe.

    Note: the hedge universe deliberately does NOT filter on Price_MA90. That
    was a lookahead leak in an earlier version -- the momentum gate belongs in
    the entry decision, not in which contracts are available to hedge with.

    Returns (df_merged, df_hedge_universe, full_indexed).
    """
    df_merged = df_options.merge(
        df_prices[["Last Price", "Price_MA90", "Realized_Vol", "Daily_Return"]],
        left_on="ts_event", right_index=True, how="inner")

    df_merged = df_merged.merge(
        iv_lookup, left_on=["ts_event", "symbol"],
        right_on=["ts_event_iv", "symbol"], how="left").drop(columns=["ts_event_iv"])
    df_merged = df_merged.rename(columns={"IV": "iv"})

    df_merged["bucket"] = df_merged["dte"].apply(assign_bucket)
    df_merged = df_merged[df_merged["bucket"].notna()].copy()
    df_merged = df_merged.merge(iv_surface, on=["ts_event", "bucket"], how="left")
    df_merged = df_merged.rename(columns={"IV_surface": "iv_surface"})

    calls = df_merged[df_merged["option_type"] == "C"].copy()
    calls["hedge_otm_pct"] = (calls["strike"] - calls["Last Price"]) / calls["Last Price"]

    def dedup(df):
        return df.groupby(["ts_event", "symbol"]).agg(
            close=("close", "last"), option_type=("option_type", "first"),
            expiration=("expiration", "first"), dte=("dte", "first"),
            strike=("strike", "first"), last_price=("Last Price", "first"),
            realized_vol=("Realized_Vol", "first"),
            daily_return=("Daily_Return", "first"),
            bucket=("bucket", "first"), volume=("volume", "sum"),
            iv=("iv", "first"), iv_surface=("iv_surface", "first")).reset_index()

    df_hedge_universe = dedup(calls[
        (calls["hedge_otm_pct"] >= BASKET_STRIKE_BAND_LOW)
        & (calls["hedge_otm_pct"] <= BASKET_STRIKE_BAND_HIGH)])

    full_indexed = df_options.set_index(["ts_event", "symbol"]).sort_index()

    if verbose:
        per_date = df_hedge_universe.groupby("ts_event").size()
        print(f"[universe] hedge universe: {len(df_hedge_universe)} rows | "
              f"median {per_date.median():.0f} candidates/date")
    return df_merged, df_hedge_universe, full_indexed


def load_all(ticker, verbose=True):
    """
    Run the full data pipeline for one ticker and return everything the
    strategy needs, as a dict.
    """
    prices = load_prices(ticker, verbose=verbose)
    options = load_options(ticker, verbose=verbose)
    df_iv_final, iv_lookup, iv_surface = build_iv_surface(
        options, prices, ticker, verbose=verbose)
    df_merged, df_hedge_universe, full_indexed = build_universes(
        options, prices, iv_lookup, iv_surface, verbose=verbose)
    return {
        "ticker": ticker,
        "df_pricing": prices,
        "df_full": options,
        "df_iv_final": df_iv_final,
        "iv_surface": iv_surface,
        "df_merged": df_merged,
        "df_hedge_universe": df_hedge_universe,
        "full_indexed": full_indexed,
    }
