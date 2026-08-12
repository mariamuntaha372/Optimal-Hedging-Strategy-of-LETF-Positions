"""
Configuration: model constants and per-ticker settings.

Everything ticker-specific lives in TICKER_CONFIGS. The rest of the package
takes a ticker name and looks the config up here, so adding a new LETF means
adding one dict entry plus a CSV in data/ -- no code changes.
"""

from pathlib import Path
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
# Resolved relative to the repo root so the package works from any working
# directory and on any machine (no hardcoded drive letters).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Raw Data"
RESULTS_DIR = REPO_ROOT / "results"

# ── model constants ────────────────────────────────────────────────────────
RISK_FREE_RATE = 0.045
DIVIDEND_YIELD = 0.0
MAX_HEDGE_MULTIPLE = 10
DEFAULT_VOL = 0.30
MIN_VOLUME_FOR_IV = 10

# IV solver bounds and sanity filters
IV_SOLVER_LO = 1e-6
IV_SOLVER_HI = 5.0
IV_CEILING_TOLERANCE = 0.05
IV_SANITY_CAP = 2.00
IV_SANITY_FLOOR = 0.15
MIN_SURFACE_CONTRIBUTORS = 3

# Execution assumptions
COMMISSION_PER_CONTRACT = 0.65          # $ per contract, per fill
SLIPPAGE_BPS = 0                        # bid-ask ignored; see borrow sweeps
MARGIN_FINANCING_ANNUAL_RATE = 0.0      # borrow handled separately in sweeps

# Hedge basket construction
BASKET_STRIKE_BAND_LOW = -0.20          # 20% ITM
BASKET_STRIKE_BAND_HIGH = 0.20          # 20% OTM
MATURITY_WEIGHTS = {
    "Weekly": 0.15,
    "Short": 0.30,
    "Medium": 0.25,
    "Long": 0.20,
    "VeryLong": 0.10,
}
MATURITY_WEIGHT_TOLERANCE = 0.15
LIQUIDITY_CAP_FRACTION = 0.50
HEDGE_ROLL_DTE_THRESHOLD = 15


# ── per-ticker configuration ───────────────────────────────────────────────
# hedge_ticker : the 1x underlying, kept for the min-variance regression.
# start / end  : yfinance download window.
# split_exclude: (start, end) window to drop from the IV pipeline, because
#                option prices around a split are quoted inconsistently.
# split_fix    : (date, factor) if yfinance's raw close needs manual division
#                before the split date to match as-traded option strikes.
#                None when yfinance's raw series is already correct.
LEGACY_FILTER_NOTE = """
`legacy_filter` reproduces the original notebook's option-universe exactly,
including an inconsistency worth knowing about.

The bulk of the MSOX options came from an Excel file that had been filtered to
calls-only, DTE 15-180. The June/July 2026 OPRA extension was appended without
that filter. So the tail of the sample contains contracts (puts, weeklies,
6-month tenors) that the rest of it structurally cannot.

That matters because MATURITY_WEIGHTS allocates 15% to the "Weekly" bucket
(5-14 DTE) and 10% to "VeryLong" (151-270 DTE). Under a DTE 15-180 filter,
Weekly is unreachable and VeryLong is truncated at 180 -- for every date
before the extension.

Set legacy_filter to None to use the full merged CSV, which makes all five
maturity buckets reachable across the whole sample. That changes the results
(roughly +14% on total P&L at h=0.5 in testing), so it is a deliberate choice,
not a default.
"""

TICKER_CONFIGS = {
    "MSOX": {
        "hedge_ticker": "MSOS",
        "start": "2022-08-24",
        "end": "2026-07-25",
        "options_csv": "MSOX.csv",
        "split_exclude": ("2024-11-01", "2024-11-27"),
        "split_fix": None,
        # Reproduces the original notebook exactly. The pre-2026-06-09 data
        # came from "MSOX filtered merged.xlsx", which had been filtered to
        # calls-only, DTE 15-180. The June/July OPRA extension was appended
        # raw, so it kept puts and all DTEs. See LEGACY_FILTER_NOTE below.
        "legacy_filter": {
            "calls_only": True,
            "dte_range": (15, 180),
            "applies_before": "2026-06-09",
        },
        # ~982 trading days of history supports the full 504-day fit.
        "egarch": {"fit_window": 504, "percentile_window": 252},
        "note": "1-for-20 reverse split Nov 2024; yfinance raw close is correct.",
    },
    "MSTX": {
        "hedge_ticker": "MSTR",
        "start": "2024-08-14",
        "end": "2026-07-25",
        "options_csv": "MSTX.csv",
        "split_exclude": ("2024-12-27", "2025-01-08"),
        "split_fix": ("2026-03-19", 10.0),
        "legacy_filter": None,
        # Only ~480 trading days since inception -- 504 would raise. 126/63
        # is what the MSTX notebook used (trade_state_126 / egarch_126).
        "egarch": {"fit_window": 126, "percentile_window": 63},
        "note": (
            "Defiance Daily Target 2X Long MSTR. Two separate events: a Dec 2024 "
            "window excluded from the IV solve, and a 10:1 split on 2026-03-19 "
            "where yfinance's raw close is post-split throughout, so pre-split "
            "days must be divided by 10 to match as-traded option strikes."
        ),
    },
    "PTIR": {
        "hedge_ticker": "PLTR",
        "start": "2024-08-01",
        "end": "2026-07-25",
        "options_csv": "PTIR.csv",
        "split_exclude": ("2025-07-02", "2025-07-16"),
        "split_fix": None,
        "legacy_filter": None,
        # Same history constraint as MSTX.
        "egarch": {"fit_window": 126, "percentile_window": 63},
        "note": "Direxion Daily PLTR Bull 2X.",
    },
}


def get_config(ticker):
    """Return the config dict for a ticker, with timestamps parsed."""
    if ticker not in TICKER_CONFIGS:
        raise KeyError(
            f"No config for {ticker!r}. Known tickers: {sorted(TICKER_CONFIGS)}"
        )
    cfg = dict(TICKER_CONFIGS[ticker])
    cfg["ticker"] = ticker
    cfg["options_path"] = DATA_DIR / cfg["options_csv"]

    if cfg["split_exclude"] is not None:
        lo, hi = cfg["split_exclude"]
        cfg["split_exclude"] = (pd.Timestamp(lo), pd.Timestamp(hi))

    if cfg["split_fix"] is not None:
        date, factor = cfg["split_fix"]
        cfg["split_fix"] = (pd.Timestamp(date), float(factor))

    if cfg.get("legacy_filter"):
        lf = dict(cfg["legacy_filter"])
        if lf.get("applies_before"):
            lf["applies_before"] = pd.Timestamp(lf["applies_before"])
        cfg["legacy_filter"] = lf

    return cfg
