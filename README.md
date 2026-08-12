# Optimal Hedging of Short LETF Positions

EGARCH-timed short positions in leveraged ETFs, hedged with an LP-optimized
basket of long calls spread across maturity buckets. Backtested on three 2x
LETFs — MSOX, MSTX, and PTIR — with contract-level implied vol solved from
OPRA daily bars, CVaR-derived hedge ratios, and overnight-gap stress testing.

---

## 🤝 Industry Mentorship

This project was developed with guidance from an industry mentor in
quantitative finance. Their feedback helped shape the project's approach to
quantitative trading, strategy development, and analysis.

---

## The idea

Leveraged ETFs decay. Daily rebalancing plus volatility drag means a 2x LETF
loses value against 2x its underlying's return over time, which makes a short
position structurally attractive.

The problem is the tail. A short LETF position has unbounded loss, and the
move that kills it — a violent overnight gap up — is exactly the move no stop
can protect against, because the fill *is* the gapped price. Long calls are
the only instrument that functions in that scenario.

So the question isn't whether to hedge. It's **how much**, and **with which
contracts** — and whether the premium paid destroys the edge that made the
short worth taking.

## How it works

**1. Entry timing — EGARCH volatility regime.** An EGARCH(1,1,1) with
Student-t errors is fit on a rolling window of adjusted returns, refit every
21 days. Its one-day-ahead forecast is ranked into a rolling percentile. Entry
fires when the percentile crosses 0.75, exit when it falls below 0.35 —
hysteresis, so the position doesn't churn at the threshold. A momentum gate
(5-day return < 0) filters entries further.

**2. Implied vol surface.** IV is solved per contract-day by Brent's method
against the LETF's *raw* close, since option contracts are quoted in
then-current terms. Solutions that pin to the solver's ceiling are discarded
as non-convergence rather than treated as genuine 500% vol. A median-IV
surface by (date, maturity bucket) provides the fallback when a specific
contract's IV is unusable.

**3. Hedge basket — linear program.** Rather than picking one strike, the
hedge is a basket. An LP selects contracts within ±20% of spot, delta-matched
to a target fraction of the short's dollar delta, subject to maturity-bucket
weights (Weekly through VeryLong), a liquidity cap, and integer contract
counts with post-rounding delta repair.

**4. Restriking.** When spot drifts more than 25% from the basket's entry
spot, the basket is rebuilt. Legs within 15 DTE of expiry are rolled
individually rather than replacing the whole basket.

**5. Risk measurement.** Episode-level P&L feeds VaR and CVaR at 95%. Separate
overnight-gap stress tests inject +50% / +75% / +100% gaps early in each
position's life and revalue the basket at intrinsic — no IV spike credited,
which is deliberately conservative.

## Results

$25,000 base capital, dollar-volatility target $16,000, restrike band 0.25,
commissions at $0.65/contract. Bid-ask spread excluded; borrow cost handled in
separate sweeps.

### MSOX — 2x MSOS · 6 episodes · Feb 2023–Jul 2026

| Hedge ratio | Total P&L | Max DD | Sharpe | CVaR 95% | Worst day |
|---|---|---|---|---|---|
| 0.00 (naked) | $4,215 | −17.00% | 0.37 | −$1,229 | −$2,583 |
| 0.25 | $4,114 | −10.37% | 0.42 | −$928 | −$1,635 |
| **0.50** | $2,313 | −7.92% | 0.26 | **−$845** | −$1,314 |
| 0.75 | $1,463 | −11.17% | 0.19 | −$902 | −$1,473 |
| 1.00 | $4,785 | −15.33% | 0.45 | −$964 | −$3,233 ⚠️ |

### PTIR — 2x PLTR · 12 episodes · Oct 2024–Jul 2026

| Hedge ratio | Total P&L | Max DD | Sharpe | CVaR 95% | Worst day |
|---|---|---|---|---|---|
| 0.00 (naked) | $1,535 | −10.95% | 0.40 | −$712 | −$791 |
| 0.25 | $1,561 | −8.54% | 0.42 | −$608 | −$784 |
| 0.50 | $982 | −7.16% | 0.32 | −$540 | −$679 |
| 0.75 | $1,053 | −4.54% | 0.37 | −$453 | −$562 |
| **1.00** | $434 | −7.20% | 0.18 | **−$422** | −$579 |

### MSTX — 2x MSTR · 10 episodes · Aug 2024–Jul 2026

| Hedge ratio | Total P&L | Max DD | Sharpe | CVaR 95% | Worst day |
|---|---|---|---|---|---|
| 0.00 (naked) | $8,906 | −16.92% | 1.10 | −$1,188 | −$1,729 |
| 0.25 | $7,670 | −14.44% | 1.08 | −$952 | −$1,348 |
| 0.50 | $6,703 | −12.02% | 1.05 | −$787 | −$1,440 |
| 0.75 | $6,899 | −7.39% | 1.27 | −$580 | −$960 |
| **1.00** | $8,144 | **−5.24%** | **1.45** | **−$561** | −$1,019 |

## The main finding

The hedge behaves as theory predicts **only when the option chain is deep
enough to build a real basket.**

| | MSOX | PTIR | MSTX |
|---|---|---|---|
| Hedge candidates/date (median) | **3** | 22 | **51** |
| Hedge universe rows | 1,334 | 6,617 | 27,441 |
| Max DD falls monotonically in h? | ✗ | ✗ | ✓ |
| CVaR improves monotonically in h? | ✗ | ✓ | ✓ |
| Worst day ever worse than naked? | **yes, at h=1.0** | no | no |

On MSTX, with a median of 51 candidates per date, everything works: drawdown
falls from −16.92% to −5.24%, CVaR improves at every step, and Sharpe rises
to 1.45 — the hedged book strictly dominates the naked short on risk while
giving up little return.

On MSOX, with a median of **3**, the LP has almost no freedom. At a full
delta target it is forced into whatever contracts exist regardless of
maturity-bucket fit, and the result breaks down: worst day at h=1.0 is
−$3,233, *worse* than the naked short's −$2,583. A hedge that makes the worst
day worse is not hedging.

This is a constraint on where the strategy applies, not a tuning problem.
Thin option chains cannot support basket hedging, and no parameter choice
fixes that.

## Reproducing

```bash
git clone https://github.com/mariamuntaha372/Optimal-Hedging-Strategy-of-LETF-Positions.git
cd Optimal-Hedging-Strategy-of-LETF-Positions
pip install -r requirements.txt

python run_ticker.py MSOX      # one ticker
python run_ticker.py --all     # all three
```

Prices download from yfinance at runtime. Option data is committed under
`Raw Data/`. Results write to `results/<TICKER>/`.

Runtime is dominated by the IV solve and scales with chain size — MSOX takes
a few minutes, MSTX closer to half an hour (601,844 option rows, 59,081 IVs).

## Repository layout

```
├── src/
│   ├── config.py       per-ticker settings: splits, EGARCH windows, filters
│   ├── data.py         yfinance prices, OPRA loader, IV surface, universes
│   ├── pricing.py      Black-Scholes greeks, Brent IV solver, bucketing
│   ├── hedging.py      basket candidate selection, LP solver, mark-to-market
│   └── strategy.py     EGARCH regime signal, state machine, backtest
├── scripts/
│   └── merge_opra.py   merge multiple OPRA pulls into one CSV per ticker
├── Raw Data/           committed option data (MSOX.csv, MSTX.csv, PTIR.csv)
├── results/            generated sweeps per ticker
└── run_ticker.py       CLI entry point
```

Every ticker runs through the same engine. Adding a fourth LETF means one
`TICKER_CONFIGS` entry and one CSV — no code changes.

## Data

Option bars are OPRA daily OHLCV via Databento, merged per ticker by
`scripts/merge_opra.py`. Where two pulls overlap, the later file wins
outright — earlier data is truncated at the point the later file begins,
rather than deduplicated, so a revised settle never gets volume-averaged
against a stale one.

Prices come from yfinance, downloaded twice per ticker on purpose: **raw**
close for strikes, moneyness, and IV, since contracts are quoted in
then-current terms; **split-adjusted** close for returns, volatility, and
momentum, so a reverse split doesn't inject a fake several-hundred-percent
daily return into the EGARCH fit.

Split handling is per-ticker in `config.py`:

- **MSOX** — 1-for-20 reverse split Nov 2024; Nov 1–27 excluded from the IV
  solve.
- **MSTX** — Dec 27 2024–Jan 8 2025 excluded from the IV solve, *and* a 10:1
  split on Mar 19 2026 where yfinance reports post-split closes throughout,
  so pre-split raw closes are divided by 10 to match as-traded strikes.
- **PTIR** — Jul 2–16 2025 excluded from the IV solve.

## Limitations

**Small samples.** Six episodes on MSOX, ten on MSTX, twelve on PTIR. CVaR at
95% on ten episodes rests on the worst one or two. Every number here should be
read as indicative, not established.

**Short, favorable window.** MSTX and PTIR have under two years of history —
both launched in 2024. Their EGARCH fits use a 126-day window because 504
isn't available. That period was unusually volatile for both underlyings,
which flatters a short-volatility-regime strategy.

**Execution assumptions are optimistic.** Bid-ask spread is excluded entirely.
On thin LETF option chains — precisely the MSOX case above — realistic spreads
could consume a meaningful share of the reported edge. Borrow cost is set to
zero in the headline runs and swept separately; hard-to-borrow names would
change the picture.

**Marking convention.** Hedge legs are marked at the greater of model value
and intrinsic. Under gap stress they are marked at intrinsic with no IV spike
credited — conservative for the hedge, but it means real-world gap performance
would likely be somewhat better than shown.

**No live trading.** This is a backtest. Nothing here has been traded.

## License

MIT
