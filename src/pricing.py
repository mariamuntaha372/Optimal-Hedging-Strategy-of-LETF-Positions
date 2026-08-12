"""
Option pricing primitives: Black-Scholes greeks, the implied-vol solver,
maturity bucketing, and the vol-resolution fallback chain.

Nothing here knows about tickers or files -- pure functions only.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

from .config import (
    RISK_FREE_RATE, DIVIDEND_YIELD, DEFAULT_VOL,
    COMMISSION_PER_CONTRACT, SLIPPAGE_BPS, MARGIN_FINANCING_ANNUAL_RATE,
    IV_SOLVER_LO, IV_SOLVER_HI, IV_SANITY_CAP, IV_SANITY_FLOOR,
)


def execution_cost(price, n_contracts, commission=COMMISSION_PER_CONTRACT,
                    slippage_bps=SLIPPAGE_BPS):
    if n_contracts <= 0 or pd.isna(price):
        return 0.0
    return commission * n_contracts + (slippage_bps / 10000.0) * price * 100 * n_contracts

def financing_cost(capital, holding_days, annual_rate=MARGIN_FINANCING_ANNUAL_RATE):
    if capital <= 0 or holding_days <= 0:
        return 0.0
    return capital * annual_rate * (holding_days / 365.0)

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call_delta(S, K, T, sigma, r=RISK_FREE_RATE):
    if S <= 0 or K <= 0:
        return 0.5
    if T <= 1e-6 or sigma <= 1e-6:
        return 1.0 if S > K else (0.0 if S < K else 0.5)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)

def bs_call_delta_gamma(S, K, T, sigma, r=RISK_FREE_RATE):
    if S <= 0 or K <= 0:
        return 0.5, 0.0
    if T <= 1e-6 or sigma <= 1e-6:
        return (1.0 if S > K else (0.0 if S < K else 0.5)), 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    gamma = math.exp(-d1 * d1 / 2.0) / (math.sqrt(2 * math.pi) * S * sigma * math.sqrt(T))
    return norm_cdf(d1), gamma

def assign_bucket(dte):
    if 5 <= dte <= 14:      return "Weekly"
    elif 15 <= dte <= 45:   return "Short"
    elif 46 <= dte <= 90:   return "Medium"
    elif 91 <= dte <= 150:  return "Long"
    elif 151 <= dte <= 270: return "VeryLong"
    return None

def resolve_vol(primary, surface, fallback, default=DEFAULT_VOL,
                 sanity_cap=IV_SANITY_CAP, sanity_floor=IV_SANITY_FLOOR):
    if pd.notna(primary) and sanity_floor <= primary <= sanity_cap:
        return primary, "IV"
    if pd.notna(surface) and sanity_floor <= surface <= sanity_cap:
        return surface, "IV_surface"
    if pd.notna(fallback) and fallback > 0:
        return fallback, "RV"
    return default, "default"

# ── implied vol ────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, q=DIVIDEND_YIELD, option_type="C"):
    """Black-Scholes price for a European call or put."""
    if T <= 0 or sigma <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "C":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def implied_vol(px, S, K, T, r, q=DIVIDEND_YIELD, option_type="C"):
    """Solve for implied vol via brentq. Returns NaN if it fails to converge."""
    if T <= 0 or px <= 0:
        return np.nan
    try:
        return brentq(
            lambda s: bs_price(S, K, T, r, s, q, option_type) - px,
            IV_SOLVER_LO, IV_SOLVER_HI, maxiter=200,
        )
    except ValueError:
        return np.nan
