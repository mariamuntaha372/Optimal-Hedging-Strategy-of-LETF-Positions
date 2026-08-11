# Optimal Hedging Strategy of a Leveraged ETF Position

This project investigates whether a dynamically hedged short position can 
capture the well-documented volatility decay of Leveraged ETFs (LETFs) 
while controlling the tail risk of a sharp rally in the underlying. LETFs 
are designed to deliver a multiple of an underlying security's daily 
return, but compounding and daily rebalancing erode capital over longer 
holding periods even when the underlying is flat — making them a natural 
short candidate, particularly on high-volatility names. The danger is 
symmetric: an adverse move can be severe and fast, even for an 
experienced trader.

The core structure is a short LETF position hedged with a long call 
option basket, sized and timed dynamically rather than statically.

## Research Objectives
- **Quantify volatility decay** across LETFs with different underlyings 
  (a sector ETF, a single stock, and a bitcoin-proxy stock) and different 
  volatility regimes.
- **Determine whether timing adds value**: test if a volatility-regime 
  entry/exit signal (EGARCH-based) outperforms simply being short, via 
  random-entry null-hypothesis testing.
- **Optimize the hedge ratio mathematically**: derive it via Conditional 
  Value-at-Risk (CVaR) minimization rather than assuming a fixed level, 
  and test its sensitivity across position sizes.
- **Test generalization**: apply an identical framework, unmodified, to 
  three structurally different LETFs to see whether the design travels.

## Methodology
- **Entry/exit timing**: EGARCH(1,1) volatility-regime signal (rolling 
  refit) combined with a momentum filter.
- **Hedge construction**: multi-maturity, multi-strike call basket, 
  optimized via linear programming to minimize premium subject to a 
  target dollar-delta and liquidity constraints.
- **Dynamic rebalancing**: delta-band restriking — the basket is rebuilt 
  whenever realized coverage drifts outside a tolerance band around the 
  target hedge ratio, plus forced rolls near expiry.
- **Bias-corrected backtesting**: fully lagged signals, path-aware daily 
  stop-loss, daily mark-to-market accounting on a fixed capital base.

## Key Deliverables & Analysis
- Backtesting engine (Python — pandas, numpy, scipy, arch)
- CVaR-optimal hedge-ratio derivation and sensitivity table
- Overnight-gap stress testing (synthetic tail scenarios)
- Random-entry null-hypothesis testing with block-bootstrap resampling 
  and confidence intervals
- Cross-ticker validation across MSOX, PTIR, and MSTX

## Key Findings
- Hedge mechanics generalized across all three tickers; realized 
  effectiveness scaled directly with option-chain liquidity.
- EGARCH-based entry timing did **not** clearly outperform random entry 
  on two of three tickers tested — a substantial share of realized 
  returns reflects structural LETF decay and the convexity hedge, not 
  timing skill.
- A data-integrity error (a mislabeled price series) was identified and 
  corrected via independent cross-validation before being propagated 
  into final results.