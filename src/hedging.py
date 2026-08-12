"""
Hedge basket construction: candidate selection, pricing, the LP solver that
picks strikes and sizes across maturity buckets, and mark-to-market.

This is the core of the hedging approach -- a multi-leg basket spread across
maturity buckets, delta-matched to the short position, rather than a single
strike.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import (
    RISK_FREE_RATE, DEFAULT_VOL, MAX_HEDGE_MULTIPLE,
    BASKET_STRIKE_BAND_LOW, BASKET_STRIKE_BAND_HIGH,
    MATURITY_WEIGHTS, MATURITY_WEIGHT_TOLERANCE,
    LIQUIDITY_CAP_FRACTION, HEDGE_ROLL_DTE_THRESHOLD,
)
from .pricing import (
    bs_call_delta, bs_call_delta_gamma, assign_bucket, resolve_vol,
)


def build_hedge_candidates(date, expiration_short, spot, df_hedge_universe,
                            exclude_symbols=None):
    c = df_hedge_universe[df_hedge_universe["ts_event"] == date].copy()
    if len(c) == 0:
        return c
    c["otm_pct"] = (c["strike"] - spot) / spot
    c = c[(c["otm_pct"] >= BASKET_STRIKE_BAND_LOW) & (c["otm_pct"] <= BASKET_STRIKE_BAND_HIGH)]
    c = c[c["bucket"].notna()]
    if exclude_symbols:
        c = c[~c["symbol"].isin(exclude_symbols)]
    return c

def price_candidates(candidates, spot, r=RISK_FREE_RATE):
    cols = ["delta", "gamma", "vol_used", "vol_src"]
    if len(candidates) == 0:
        out = candidates.copy()
        for col in cols:
            out[col] = pd.Series(dtype=float if col != "vol_src" else object)
        return out
    out = candidates.copy()
    out["T"] = out["dte"] / 365.0
    deltas, gammas, vols, srcs = [], [], [], []
    for _, row in out.iterrows():
        rv = row["realized_vol"] if "realized_vol" in row and pd.notna(row["realized_vol"]) and row["realized_vol"] > 0 else np.nan
        vol, src = resolve_vol(row.get("iv", np.nan), row.get("iv_surface", np.nan), rv)
        d, g = bs_call_delta_gamma(spot, row["strike"], row["T"], vol, r)
        deltas.append(d); gammas.append(g); vols.append(vol); srcs.append(src)
    out["delta"], out["gamma"] = deltas, gammas
    out["vol_used"], out["vol_src"] = vols, srcs
    return out

def solve_hedge_basket(candidates, spot, target_delta_dollars, m=100):
    """LP basket + post-rounding delta repair (the FIXED version)."""
    empty = candidates.iloc[0:0].copy()
    empty["n_contracts"] = pd.Series(dtype=int)
    if len(candidates) == 0 or target_delta_dollars <= 0:
        return empty

    c = candidates.reset_index(drop=True).copy()
    ddpc = c["delta"].values * spot * m
    prem = c["close"].values * m
    usable = ddpc > 1.0
    if usable.sum() == 0:
        return empty
    c = c[usable].reset_index(drop=True)
    ddpc, prem = ddpc[usable], prem[usable]

    caps = np.maximum(1.0, np.floor(c["volume"].values * LIQUIDITY_CAP_FRACTION))
    bounds = [(0, cap) for cap in caps]

    A_ub, b_ub = [], []
    for bucket, tw in MATURITY_WEIGHTS.items():
        mask = (c["bucket"].values == bucket).astype(float)
        if mask.sum() == 0:
            continue
        lo, hi = max(tw - MATURITY_WEIGHT_TOLERANCE, 0.0), tw + MATURITY_WEIGHT_TOLERANCE
        A_ub.append((mask * ddpc).tolist());  b_ub.append(hi * target_delta_dollars)
        A_ub.append((-mask * ddpc).tolist()); b_ub.append(-lo * target_delta_dollars)

    res = linprog(prem, A_ub=A_ub or None, b_ub=b_ub or None,
                   A_eq=[ddpc.tolist()], b_eq=[target_delta_dollars],
                   bounds=bounds, method="highs")
    if not res.success:
        res = linprog(prem, A_eq=[ddpc.tolist()], b_eq=[target_delta_dollars],
                       bounds=bounds, method="highs")
        if not res.success:
            return empty

    out = c.copy()
    out["raw"] = res.x
    out = out[out["raw"] >= 0.5].copy()
    if len(out) == 0:
        return empty
    out["n_contracts"] = out["raw"].round().astype(int)
    out = out[out["n_contracts"] > 0].drop(columns=["raw"]).reset_index(drop=True)
    if len(out) == 0:
        return empty

    # delta repair: greedy +/-1 on whichever leg most reduces |residual|
    dd = out["delta"].values * spot * m
    cap_arr = np.maximum(1.0, np.floor(out["volume"].values * LIQUIDITY_CAP_FRACTION))
    counts = out["n_contracts"].values.astype(float)
    resid = lambda cts: (cts * dd).sum() - target_delta_dollars
    for _ in range(50):
        r0 = resid(counts)
        best_i, best_gain, best_step = None, 0.0, 0.0
        for i in range(len(counts)):
            for step in (1.0, -1.0):
                nc = counts[i] + step
                if nc < 0 or nc > cap_arr[i]:
                    continue
                trial = counts.copy(); trial[i] = nc
                gain = abs(r0) - abs(resid(trial))
                if gain > best_gain + 1e-9:
                    best_i, best_gain, best_step = i, gain, step
        if best_i is None:
            break
        counts[best_i] += best_step

    out["n_contracts"] = counts.astype(int)
    out = out[out["n_contracts"] > 0].copy()
    if len(out) == 0:
        return empty
    realized = (out["n_contracts"].values * out["delta"].values * spot * m).sum()
    out["delta_match_error_pct"] = (realized - target_delta_dollars) / target_delta_dollars
    out["entry_spot"] = spot
    return out

def roll_single_hedge_leg(old_leg, date, new_spot, df_hedge_universe, r=RISK_FREE_RATE):
    entry_spot = old_leg.get("entry_spot", new_spot)
    if not (pd.notna(entry_spot) and entry_spot > 0):
        entry_spot = new_spot
    target_strike = new_spot * (1 + (old_leg["strike"] - entry_spot) / entry_spot)
    c = df_hedge_universe[(df_hedge_universe["ts_event"] == date) &
                           (df_hedge_universe["bucket"] == old_leg.get("bucket"))].copy()
    if len(c) == 0:
        c = df_hedge_universe[df_hedge_universe["ts_event"] == date].copy()
    if len(c) == 0:
        return None
    c["dist"] = (c["strike"] - target_strike).abs()
    nr = c.loc[c["dist"].idxmin()]
    rv = nr["realized_vol"] if pd.notna(nr.get("realized_vol", np.nan)) and nr.get("realized_vol", 0) > 0 else np.nan
    vol, src = resolve_vol(nr.get("iv", np.nan), nr.get("iv_surface", np.nan), rv)
    nd, ng = bs_call_delta_gamma(new_spot, nr["strike"], nr["dte"] / 365.0, vol, r)
    if nd <= 1e-4:
        return None
    return {"symbol": nr["symbol"], "strike": nr["strike"], "bucket": nr["bucket"],
            "dte": nr["dte"], "expiration": nr["expiration"], "close": nr["close"],
            "delta": nd, "gamma": ng, "vol_used": vol, "vol_src": src,
            "n_contracts": int(old_leg["n_contracts"]),
            "entry_price": nr["close"], "entry_spot": new_spot}

def mark_hedge_basket(basket, date, full_indexed, df_full):
    marks, missing = [], False
    for leg in basket:
        sym = leg["symbol"]
        try:
            px = full_indexed.loc[(date, sym), "close"]
            if isinstance(px, pd.Series):
                px = px.iloc[0]
        except KeyError:
            known = df_full[(df_full["symbol"] == sym) & (df_full["ts_event"] <= date)]
            px = known["close"].iloc[-1] if len(known) else leg.get("entry_price", 0.0)
            if len(known) == 0:
                missing = True
        leg2 = dict(leg); leg2["mark"] = px
        marks.append(leg2)
    return marks, missing
