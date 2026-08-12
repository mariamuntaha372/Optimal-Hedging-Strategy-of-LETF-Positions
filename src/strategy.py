"""
Strategy layer: EGARCH volatility regime signal, the entry/exit state machine,
and the delta-band restriking backtest.

The EGARCH fit runs on ADJUSTED returns so a reverse split doesn't inject a
fake multi-hundred-percent return into the variance estimate. Option strikes
and moneyness use raw prices -- see data.load_prices().
"""

import numpy as np
import pandas as pd
from arch import arch_model

from .config import RISK_FREE_RATE, DEFAULT_VOL
from .pricing import (
    bs_call_delta, bs_call_delta_gamma, execution_cost, financing_cost,
    resolve_vol, assign_bucket,
)
from .hedging import (
    build_hedge_candidates, price_candidates, solve_hedge_basket,
    roll_single_hedge_leg, mark_hedge_basket,
)


# ── EGARCH signal ──────────────────────────────────────────────────────────
def compute_egarch_vol_regime(df_pricing, price_col="Adj Price", fit_window=504,
                                refit_every=21, percentile_window=252,
                                low_pct=0.33, high_pct=0.67, dist="t"):
    ret = df_pricing[price_col].pct_change().dropna() * 100
    dates = ret.index
    if len(ret) < fit_window + 10:
        raise ValueError(f"only {len(ret)} returns, need ~{fit_window+10}")
    vol = pd.Series(index=dates, dtype=float)
    flag = pd.Series(False, index=dates)
    last = None
    for i in range(fit_window, len(ret)):
        need = last is None or ((i - fit_window) % refit_every == 0)
        w = ret.iloc[max(0, i - fit_window):i]
        try:
            am = arch_model(w, vol="EGARCH", p=1, o=1, q=1, dist=dist)
            if need:
                res = am.fit(disp="off", show_warning=False); last = res.params; flag.iloc[i] = True
            else:
                res = am.fix(last)
            v = res.forecast(horizon=1, reindex=False).variance.values[-1, 0]
            vol.iloc[i] = np.sqrt(v) * np.sqrt(252) / 100
        except Exception:
            continue
    out = pd.DataFrame(index=df_pricing.index)
    out["egarch_vol"]  = vol.reindex(df_pricing.index)
    out["refit_flag"]  = flag.reindex(df_pricing.index).fillna(False).astype(bool)
    out["egarch_percentile"] = out["egarch_vol"].rolling(percentile_window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    out["vol_regime"] = out["egarch_percentile"].apply(
        lambda p: "Insufficient History" if pd.isna(p) else
                  ("Low" if p <= low_pct else ("High" if p >= high_pct else "Normal")))
    return out

def build_egarch_trade_state(eg, entry_pct=0.75, exit_pct=0.35):
    pct = eg["egarch_percentile"].shift(1)
    in_trade = pd.Series(False, index=pct.index)
    action = pd.Series("FLAT", index=pct.index, dtype=object)
    state = False
    for d in pct.index:
        p = pct.loc[d]
        if pd.isna(p):
            in_trade.loc[d] = state; action.loc[d] = "HOLD" if state else "FLAT"; continue
        if not state and p >= entry_pct:
            state = True; action.loc[d] = "ENTER"
        elif state and p <= exit_pct:
            state = False; action.loc[d] = "EXIT"
        else:
            action.loc[d] = "HOLD" if state else "FLAT"
        in_trade.loc[d] = state
    return pd.DataFrame({"pctile_lag": pct, "in_trade": in_trade, "action": action})

def egarch_size_multiplier(eg, low_pct=0.33, high_pct=0.67,
                             mult_low=1.25, mult_normal=1.0, mult_high=0.5):
    pct = eg["egarch_percentile"].shift(1)
    return pct.apply(lambda p: 0.0 if pd.isna(p) else
                      (mult_low if p <= low_pct else (mult_high if p >= high_pct else mult_normal))
                      ).rename("size_mult")

def fit_egarch_signals(data, fit_window=None, percentile_window=None,
                       entry_pct=0.75, exit_pct=0.35, verbose=True):
    """
    Fit the EGARCH regime signal and build the trade state machine, then
    attach both to the data dict so run_final_strategy_v21 can find them.

    When fit_window / percentile_window are None, they come from the ticker's
    config. Tickers with under ~2 years of history need a shorter window --
    504 requires ~514 returns and will raise otherwise. A shorter window
    trades fit quality for more usable episodes.
    """
    from .config import get_config

    if fit_window is None or percentile_window is None:
        eg_cfg = get_config(data["ticker"])["egarch"]
        fit_window = eg_cfg["fit_window"] if fit_window is None else fit_window
        percentile_window = (eg_cfg["percentile_window"]
                             if percentile_window is None else percentile_window)

    n_returns = data["df_pricing"]["Adj Price"].pct_change().dropna().shape[0]
    if n_returns < fit_window + 10:
        raise ValueError(
            f"{data['ticker']}: only {n_returns} returns available but "
            f"fit_window={fit_window} needs ~{fit_window + 10}. "
            f"Lower 'egarch.fit_window' in config.py for this ticker."
        )

    if verbose:
        print(f"[egarch] fitting (fit={fit_window}, pctile={percentile_window}) ...")
    eg = compute_egarch_vol_regime(data["df_pricing"],
                                   fit_window=fit_window,
                                   percentile_window=percentile_window)
    ts = build_egarch_trade_state(eg, entry_pct=entry_pct, exit_pct=exit_pct)
    data["egarch_signal"] = eg
    data["trade_state"] = ts
    data["size_mult"] = egarch_size_multiplier(eg)
    if verbose:
        print(f"[egarch] actions: {ts['action'].value_counts().to_dict()}")
    return data


# ── backtest ───────────────────────────────────────────────────────────────
def run_final_strategy_v21(data, hedge_ratio=0.5, dvt=16000, hard_stop=1.5,
                           restrike_band=0.25, starting_capital=25_000,
                           momentum_gate="neg5", borrow_rate=0.0,
                           ts=None, eg=None, r=RISK_FREE_RATE):
    """
    Delta-band restriking backtest.

    Parameters
    ----------
    data : dict
        Output of data.load_all(ticker), plus "trade_state" and
        "egarch_signal" from fit_egarch_signals().
    hedge_ratio : float
        Fraction of short delta to hedge. 0.0 = naked short.
    dvt : float
        Dollar volatility target -- scales position size.
    restrike_band : float
        Restrike when spot moves this far from the basket's entry spot.
    ts, eg : optional
        Override the trade state / EGARCH signal in `data`.
    """
    df_pricing        = data["df_pricing"]
    df_merged         = data["df_merged"]
    df_full           = data["df_full"]
    df_hedge_universe = data["df_hedge_universe"]
    full_indexed      = data["full_indexed"]

    ts = data["trade_state"] if ts is None else ts
    eg = data["egarch_signal"] if eg is None else eg

    price = df_pricing.copy()
    src = "Adj Price" if "Adj Price" in price.columns else "Last Price"
    price["Mom5"]  = price[src].pct_change(5).shift(1)
    price["Mom20"] = price[src].pct_change(20).shift(1)

    def gate_ok(d):
        if momentum_gate is None: return True
        if d not in price.index:  return False
        m5, m20 = price.loc[d, "Mom5"], price.loc[d, "Mom20"]
        if momentum_gate == "neg5":  return pd.notna(m5) and m5 < 0
        if momentum_gate == "neg20": return pd.notna(m20) and m20 < 0
        return pd.notna(m5) and pd.notna(m20) and m5 < 0 and m20 < 0

    pctile_lag, evol_lag = eg["egarch_percentile"].shift(1), eg["egarch_vol"].shift(1)
    sm = lambda p: 1.0 if pd.isna(p) else (1.25 if p <= 0.33 else (0.5 if p >= 0.67 else 1.0))

    def leg_vol(sym, d, fb):
        row = df_merged[(df_merged["symbol"] == sym) & (df_merged["ts_event"] == d)]
        iv  = row["iv"].iloc[0] if len(row) and pd.notna(row["iv"].iloc[0]) else np.nan
        ivs = row["iv_surface"].iloc[0] if len(row) and pd.notna(row["iv_surface"].iloc[0]) else np.nan
        return resolve_vol(iv, ivs, fb)[0]

    def mark(leg, d, spot):
        m, _ = mark_hedge_basket([leg], d, full_indexed, df_full)
        return max(m[0]["mark"], max(spot - leg["strike"], 0.0))

    def build(d, spot, n_sh, hr):
        cands = price_candidates(build_hedge_candidates(d, None, spot, df_hedge_universe), spot, r)
        bdf = solve_hedge_basket(cands, spot, hr * n_sh * spot)
        legs, cost = [], 0.0
        for _, l in bdf.iterrows():
            legs.append({"symbol": l["symbol"], "strike": l["strike"], "bucket": l["bucket"],
                          "expiration": pd.Timestamp(l["expiration"]),
                          "n_contracts": int(l["n_contracts"]), "prev_mark": l["close"],
                          "entry_spot": spot, "delta": l["delta"]})
            cost += execution_cost(l["close"], int(l["n_contracts"]))
        return legs, cost

    def coverage(legs, d, spot, n_sh, fb):
        if not legs or n_sh <= 0 or spot <= 0: return 0.0
        dd = 0.0
        for lg in legs:
            T = max((lg["expiration"] - d).days, 0) / 365.0
            dd += lg["n_contracts"] * bs_call_delta(spot, lg["strike"], T,
                                                     leg_vol(lg["symbol"], d, fb), r) * spot * 100
        return dd / (n_sh * spot)

    in_regime = ts["in_trade"]
    dates = [d for d in price.index if d in in_regime.index]
    daily, subs = {}, []
    in_pos = False; n_sh = entry_spot = 0.0; legs = []; sub_entry = None
    n_rs = 0; rs_fric = 0.0; borrow_pos = 0.0; total_borrow = 0.0
    cov_after, cov_trig = [], []
    prev_spot = None

    for d in dates:
        spot = price.loc[d, "Last Price"]
        pnl = 0.0
        rv = price.loc[d, "Realized_Vol"]
        fb = rv if pd.notna(rv) and rv > 0 else DEFAULT_VOL

        if in_pos:
            if prev_spot is not None:
                pnl += -n_sh * (spot - prev_spot)
            for lg in legs:
                px = mark(lg, d, spot)
                pnl += (px - lg["prev_mark"]) * 100 * lg["n_contracts"]
                lg["prev_mark"] = px

            b = n_sh * spot * (borrow_rate / 252.0)
            pnl -= b; borrow_pos += b; total_borrow += b

            if hedge_ratio > 0:
                cov = coverage(legs, d, spot, n_sh, fb)
                near = any((lg["expiration"] - d).days <= 3 for lg in legs)
                if abs(cov - hedge_ratio) > restrike_band or near:
                    cov_trig.append(cov)
                    for lg in legs:
                        c = execution_cost(lg["prev_mark"], lg["n_contracts"]); pnl -= c; rs_fric += c
                    legs, oc = build(d, spot, n_sh, hedge_ratio); pnl -= oc; rs_fric += oc
                    n_rs += 1
                cov_after.append(coverage(legs, d, spot, n_sh, fb))

            why = None
            if spot >= entry_spot * hard_stop:      why = "hard_stop"
            elif not bool(in_regime.get(d, False)): why = "regime_exit"
            if why:
                for lg in legs:
                    pnl -= execution_cost(lg["prev_mark"], lg["n_contracts"])
                subs.append({"entry": sub_entry, "exit": d, "why": why,
                              "entry_spot": entry_spot, "exit_spot": spot, "n_sh": n_sh,
                              "restrikes": n_rs, "restrike_friction": round(rs_fric, 0),
                              "borrow": round(borrow_pos, 0),
                              "cov_mean": round(np.mean(cov_after), 3) if cov_after else np.nan,
                              "cov_min": round(np.min(cov_after), 3) if cov_after else np.nan,
                              "worst_drift": round(min(cov_trig), 3) if cov_trig else np.nan})
                in_pos, legs = False, []
                n_rs, rs_fric, borrow_pos = 0, 0.0, 0.0
                cov_after, cov_trig = [], []
        else:
            if bool(in_regime.get(d, False)) and gate_ok(d):
                ev = evol_lag.get(d, np.nan)
                vol = ev if pd.notna(ev) and ev > 0 else fb
                n_sh = max(1, round(dvt * sm(pctile_lag.get(d, np.nan)) / (spot * vol)))
                entry_spot, sub_entry = spot, d
                legs, oc = build(d, spot, n_sh, hedge_ratio); pnl -= oc
                in_pos = True; cov_after, cov_trig = [], []

        daily[d] = daily.get(d, 0.0) + pnl
        prev_spot = spot if in_pos else None

    if in_pos:
        d = dates[-1]
        for lg in legs:
            daily[d] -= execution_cost(lg["prev_mark"], lg["n_contracts"])
        subs.append({"entry": sub_entry, "exit": d, "why": "window_end",
                      "entry_spot": entry_spot, "exit_spot": price.loc[d, "Last Price"],
                      "n_sh": n_sh, "restrikes": n_rs, "restrike_friction": round(rs_fric, 0),
                      "borrow": round(borrow_pos, 0),
                      "cov_mean": round(np.mean(cov_after), 3) if cov_after else np.nan,
                      "cov_min": round(np.min(cov_after), 3) if cov_after else np.nan,
                      "worst_drift": round(min(cov_trig), 3) if cov_trig else np.nan})

    s = pd.Series(daily).sort_index()
    eq = starting_capital + s.cumsum()
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
    dret = eq.pct_change().fillna(0.0)
    sharpe = round((dret.mean()/dret.std())*np.sqrt(252), 2) if dret.std() > 0 else 0.0
    st = pd.DataFrame(subs)
    yrs = ((st["exit"].max() - st["entry"].min()).days / 365.25) if len(st) else np.nan
    tot = s.sum() / starting_capital
    cagr = ((1 + tot) ** (1/yrs) - 1) if (yrs and yrs > 0) else np.nan

    metrics = {"hedge_ratio": hedge_ratio, "dvt": dvt, "band": restrike_band,
               "borrow_rate": borrow_rate,
               "Total P&L": round(s.sum(), 0), "Max DD": f"{max_dd:.2%}", "Sharpe": sharpe,
               "CAGR": f"{cagr:+.2%}" if pd.notna(cagr) else "n/a",
               "Worst day": round(s.min(), 0), "Worst day date": s.idxmin().date(),
               "Sub-trades": len(st),
               "Total restrikes": int(st["restrikes"].sum()) if len(st) else 0,
               "Restrike friction": f"${st['restrike_friction'].sum():,.0f}" if len(st) else "$0",
               "Borrow paid": round(total_borrow, 0),
               "Coverage mean": round(st["cov_mean"].mean(), 3) if len(st) else np.nan}
    return metrics, s, st, None