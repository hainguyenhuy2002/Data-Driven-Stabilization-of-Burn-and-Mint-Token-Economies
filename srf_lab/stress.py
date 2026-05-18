"""
Synthetic stress-test return generators.

Each function returns a length-T numpy array of daily returns.
"""
from __future__ import annotations
import numpy as np


def gbm_returns(T: int = 365, drift: float = 0.002, sigma: float = 0.04,
                seed: int = 42) -> np.ndarray:
    """Plain Geometric Brownian Motion returns."""
    rng = np.random.default_rng(seed)
    return rng.normal(drift, sigma, T)


def engineered_crash(T: int = 365, drift: float = 0.002, sigma: float = 0.04,
                     crash_t: int = 150, crash_len: int = 10,
                     crash_mean: float = -0.12, crash_sigma: float = 0.05,
                     seed: int = 42) -> np.ndarray:
    """GBM returns with a flash-crash window superimposed."""
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, sigma, T)
    crash_t = max(0, min(T-1, crash_t))
    end = min(T, crash_t + crash_len)
    r[crash_t:end] = rng.normal(crash_mean, crash_sigma, end - crash_t)
    return r


def vol_clustering(T: int = 365, drift: float = 0.002, sigma_low: float = 0.02,
                    sigma_high: float = 0.08, regime_len: int = 30,
                    seed: int = 42) -> np.ndarray:
    """
    Returns drawn from a two-state regime-switching model — alternating low- and
    high-volatility windows of length `regime_len`. Captures GARCH-style clusters.
    """
    rng = np.random.default_rng(seed)
    r = np.zeros(T)
    state = 0
    t = 0
    while t < T:
        sig = sigma_low if state == 0 else sigma_high
        end = min(T, t + regime_len)
        r[t:end] = rng.normal(drift, sig, end - t)
        state = 1 - state
        t = end
    return r


def oracle_delay(observed_prices: np.ndarray, lag: int = 1) -> np.ndarray:
    """
    Apply oracle delay: the controller observes P_{t-lag} instead of P_t.
    Returns the delayed price series (left-padded with the first sample).
    """
    if lag <= 0:
        return observed_prices
    out = np.empty_like(observed_prices)
    out[:lag] = observed_prices[0]
    out[lag:] = observed_prices[:-lag]
    return out
