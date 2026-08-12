"""
Merge multiple OPRA pulls for one ticker into a single CSV under data/.

Databento delivers option bars in separate files per request window, and in
either .csv or binary .dbn/.dbn.zst form. This script normalizes both and
concatenates them.

OVERLAP HANDLING
----------------
Where two pulls cover the same dates, the LATER file wins outright: earlier
data is truncated at the point the later file begins. This matches the
behavior the pipeline was originally built and validated against.

The alternative -- keeping both and dropping exact duplicates -- looks safer
but isn't. If two vendor pulls disagree even slightly on an overlapping day
(a revised settle, a late print), drop_duplicates() keeps BOTH rows, and the
volume-weighted collapse in build_iv_surface() then averages a stale row with
a corrected one. Truncation avoids that entirely.

Usage
-----
    python scripts/merge_opra.py MSOX raw/msox_2023.csv raw/msox_2026.dbn
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

COLUMNS = ["ts_event", "rtype", "publisher_id", "instrument_id",
           "open", "high", "low", "close", "volume", "symbol"]


def read_any(path):
    """Read an OPRA file as CSV or Databento binary, returning a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif ".dbn" in path.name.lower():
        try:
            import databento as db
        except ImportError:
            sys.exit("Reading .dbn requires: pip install databento")
        # to_df() puts the timestamp in the INDEX, not a column -- reset_index()
        # is required or every row from this file gets a null ts_event.
        df = db.DBNStore.from_file(str(path)).to_df().reset_index()
    else:
        sys.exit(f"Unrecognized file type: {path.name}")

    df["ts_event"] = pd.to_datetime(df["ts_event"], format="mixed", utc=True)
    df["ts_event"] = df["ts_event"].dt.tz_localize(None).dt.normalize()
    return df


def merge(paths, verbose=True):
    """Merge OPRA files in chronological order; later files win overlaps."""
    frames = [read_any(p) for p in paths]
    frames.sort(key=lambda d: d["ts_event"].min())

    if verbose:
        for p, f in zip(paths, frames):
            print(f"  {Path(p).name}: {len(f):>8,} rows  "
                  f"{f['ts_event'].min().date()} -> {f['ts_event'].max().date()}")

    merged = frames[0]
    for nxt in frames[1:]:
        cut = nxt["ts_event"].min()
        kept = merged[merged["ts_event"] < cut]
        dropped = len(merged) - len(kept)
        if verbose and dropped:
            print(f"  overlap at {cut.date()}: dropped {dropped:,} earlier rows")
        merged = pd.concat([kept, nxt], ignore_index=True)

    merged = merged.sort_values("ts_event").reset_index(drop=True)
    keep = [c for c in COLUMNS if c in merged.columns]
    return merged[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ticker", help="e.g. MSOX")
    ap.add_argument("files", nargs="+", help="OPRA .csv / .dbn / .dbn.zst files")
    args = ap.parse_args()

    print(f"Merging {len(args.files)} file(s) for {args.ticker}:")
    out = merge(args.files)

    missing = out["ts_event"].isna().sum()
    if missing:
        sys.exit(f"ERROR: {missing:,} rows have no ts_event -- refusing to write.")

    DATA_DIR.mkdir(exist_ok=True)
    dest = DATA_DIR / f"{args.ticker}.csv"
    out.to_csv(dest, index=False)

    print(f"\nWrote {len(out):,} rows to {dest.relative_to(REPO_ROOT)}")
    print(f"  range: {out['ts_event'].min().date()} -> {out['ts_event'].max().date()}")
    print(f"  contracts: {out['symbol'].nunique():,}")


if __name__ == "__main__":
    main()
