"""LETF hedging strategy: EGARCH-timed short with an LP-optimized call basket."""

from .config import TICKER_CONFIGS, get_config
from .data import load_all
from .strategy import fit_egarch_signals, run_final_strategy_v21

__all__ = [
    "TICKER_CONFIGS", "get_config", "load_all",
    "fit_egarch_signals", "run_final_strategy_v21",
]
