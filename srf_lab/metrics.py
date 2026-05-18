"""
Evaluation metrics — far broader than the original two (Section 4.4 review).

All metrics take a TrajectoryResult (returned by simulator.simulate) and
return a flat dict so they can be aggregated into a pandas DataFrame.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Mapping


def annualized_volatility(P: np.ndarray, periods_per_year: int = 365) -> float:
    """std of daily log-returns × sqrt(N)."""
    r = np.diff(np.log(np.clip(P, 1e-12, None)))
    if len(r) < 2:
        return 0.0
    return float(np.std(r, ddof=1) * np.sqrt(periods_per_year))


def downside_volatility(P: np.ndarray, periods_per_year: int = 365) -> float:
    r = np.diff(np.log(np.clip(P, 1e-12, None)))
    neg = r[r < 0]
    if len(neg) < 2:
        return 0.0
    return float(np.std(neg, ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(P: np.ndarray) -> float:
    """(peak - trough) / peak using running max."""
    if len(P) == 0:
        return 0.0
    peak = np.maximum.accumulate(P)
    dd = (peak - P) / np.where(peak > 0, peak, 1.0)
    return float(np.max(dd))


def time_under_drawdown(P: np.ndarray, threshold: float = 0.10) -> int:
    """Number of days the price spends more than `threshold` below the running peak."""
    peak = np.maximum.accumulate(P)
    dd = (peak - P) / np.where(peak > 0, peak, 1.0)
    return int(np.sum(dd > threshold))


def recovery_time(P: np.ndarray) -> int:
    """Days from the trough back above the peak before it. -1 if not recovered."""
    if len(P) == 0:
        return -1
    peak_idx = int(np.argmax(P[:int(np.argmin(P)) + 1])) if len(P) > 0 else 0
    trough_idx = int(np.argmin(P))
    peak_val = P[peak_idx]
    for i in range(trough_idx, len(P)):
        if P[i] >= peak_val:
            return i - trough_idx
    return -1


def expected_shortfall(P: np.ndarray, q: float = 0.05) -> float:
    """ES_q (≡ CVaR_q) on daily log-returns. q is the *tail* probability."""
    r = np.diff(np.log(np.clip(P, 1e-12, None)))
    if len(r) < 2:
        return 0.0
    var = np.quantile(r, q)
    tail = r[r <= var]
    if len(tail) == 0:
        return float(var)
    return float(np.mean(tail))


def reserve_depletion_ratio(R: np.ndarray) -> float:
    """1 - R_T / R_0  (positive means reserve was drawn down)."""
    if len(R) < 2 or R[0] <= 0:
        return 0.0
    return float(1.0 - R[-1] / R[0])


def intervention_cost(u_act_usd: np.ndarray) -> float:
    """Total absolute USD traded by the SRF over the run."""
    return float(np.sum(np.abs(u_act_usd)))


def slippage_adjusted_cost(u_act_usd: np.ndarray, delta_P_srf: np.ndarray,
                           P: np.ndarray) -> float:
    """
    Sum of price-impact-induced losses to the protocol when buying:
        cost ≈ Σ |u_act| · |ΔP_srf|.
    """
    return float(np.sum(np.abs(u_act_usd) * np.abs(delta_P_srf)))


def avg_lock(ell: np.ndarray) -> float:
    return float(np.mean(ell))


def peak_lock(ell: np.ndarray) -> float:
    return float(np.max(ell))


def avg_tax(eta: np.ndarray) -> float:
    return float(np.mean(eta))


def num_interventions(u_act_usd: np.ndarray) -> int:
    return int(np.sum(np.abs(u_act_usd) > 0))


# ------------------------------------------------------------------ #
# Short-window metrics — measure SUDDEN-MOVE damping rather than long-run anchor.
# ------------------------------------------------------------------ #
def rolling_max_drawdown(P: np.ndarray, window: int = 14) -> float:
    """
    Mean of the maximum drawdown computed inside each rolling W-day window.
    This captures *sudden* peak-to-trough moves rather than the global
    peak-to-trough of the entire run.
    """
    if len(P) < window + 1:
        return 0.0
    n = len(P)
    rolling_dd = np.empty(n - window + 1)
    for i in range(n - window + 1):
        seg = P[i:i+window]
        peak = np.maximum.accumulate(seg)
        dd = (peak - seg) / np.where(peak > 0, peak, 1.0)
        rolling_dd[i] = float(np.max(dd))
    return float(np.mean(rolling_dd))


def worst_daily_return(P: np.ndarray) -> float:
    """Most negative one-day simple return (a clean 'worst spike' indicator)."""
    if len(P) < 2:
        return 0.0
    r = np.diff(P) / np.maximum(P[:-1], 1e-9)
    return float(np.min(r))


def worst_daily_return_pct(P: np.ndarray) -> float:
    """Same as above but as a positive percentage drop magnitude."""
    return float(-worst_daily_return(P))


def short_window_volatility(P: np.ndarray, window: int = 14,
                              periods_per_year: int = 365) -> float:
    """
    Mean of the rolling W-day std of daily simple returns (annualised).
    A direct proxy for 'how volatile is each W-day window'.
    """
    if len(P) < window + 1:
        return 0.0
    r = np.diff(P) / np.maximum(P[:-1], 1e-9)
    if len(r) < window:
        return 0.0
    rs = np.lib.stride_tricks.sliding_window_view(r, window)
    return float(np.mean(np.std(rs, axis=1, ddof=1)) * np.sqrt(periods_per_year))


def long_run_drift_preservation(P_srf: np.ndarray, P_base: np.ndarray) -> float:
    """
    Log-distance between the regulated and unregulated terminal prices:
        |log(P_T(SRF)) - log(P_T(base))|.
    Symmetric and not skewed by tiny baseline denominators. Values near 0
    mean the SRF preserves long-run drift; values >>1 mean the SRF strongly
    diverges from the natural trajectory.
    """
    if len(P_srf) < 1 or len(P_base) < 1 or P_base[-1] <= 0 or P_srf[-1] <= 0:
        return float("nan")
    return float(abs(np.log(P_srf[-1]) - np.log(P_base[-1])))


def signed_terminal_log_excess(P_srf: np.ndarray, P_base: np.ndarray) -> float:
    """log(P_T(SRF)) - log(P_T(base)). Positive: SRF ended above baseline."""
    if len(P_srf) < 1 or len(P_base) < 1 or P_base[-1] <= 0 or P_srf[-1] <= 0:
        return float("nan")
    return float(np.log(P_srf[-1]) - np.log(P_base[-1]))


# ------------------------------------------------------------------ #
def compute_metrics(traj, baseline_traj=None) -> dict:
    """
    Build the full metric dict for a TrajectoryResult.

    If `baseline_traj` is given (no-intervention run on the same seed),
    we also report relative reductions vs that baseline.
    """
    P = traj.P
    R = traj.R
    ell = traj.lock_ratio
    eta = traj.eta
    u_act = traj.u_actual_usd
    dP_srf = traj.delta_P_srf

    out = {
        "ann_vol":              annualized_volatility(P),
        "downside_vol":         downside_volatility(P),
        "max_drawdown":         max_drawdown(P),                    # global peak-to-trough
        "rolling_dd_14":        rolling_max_drawdown(P, 14),        # short-window MaxDD
        "rolling_vol_14":       short_window_volatility(P, 14),     # short-window vol
        "worst_day_pct":        worst_daily_return_pct(P),          # largest one-day drop
        "time_under_dd_10pct":  time_under_drawdown(P, 0.10),
        "recovery_days":        recovery_time(P),
        "expected_shortfall_5": expected_shortfall(P, 0.05),
        "reserve_depletion":    reserve_depletion_ratio(R),
        "intervention_cost":    intervention_cost(u_act),
        "slippage_cost":        slippage_adjusted_cost(u_act, dP_srf, P),
        "avg_lock":             avg_lock(ell),
        "peak_lock":            peak_lock(ell),
        "avg_tax":              avg_tax(eta),
        "n_interventions":      num_interventions(u_act),
        "P_T":                  float(P[-1]),
        "R_T":                  float(R[-1]),
    }

    if baseline_traj is not None:
        base_vol = annualized_volatility(baseline_traj.P)
        base_dd = max_drawdown(baseline_traj.P)
        base_rolling_dd = rolling_max_drawdown(baseline_traj.P, 14)
        base_rolling_vol = short_window_volatility(baseline_traj.P, 14)
        base_worst_day = worst_daily_return_pct(baseline_traj.P)
        if base_vol > 0:
            out["vol_reduction"] = float((base_vol - out["ann_vol"]) / base_vol)
        if base_dd > 0:
            out["dd_reduction"]  = float((base_dd - out["max_drawdown"]) / base_dd)
        if base_rolling_dd > 0:
            out["rolling_dd_reduction"] = float(
                (base_rolling_dd - out["rolling_dd_14"]) / base_rolling_dd)
        if base_rolling_vol > 0:
            out["rolling_vol_reduction"] = float(
                (base_rolling_vol - out["rolling_vol_14"]) / base_rolling_vol)
        if base_worst_day > 0:
            out["worst_day_reduction"] = float(
                (base_worst_day - out["worst_day_pct"]) / base_worst_day)
        out["drift_preservation"] = long_run_drift_preservation(P, baseline_traj.P)
        out["terminal_log_excess"] = signed_terminal_log_excess(P, baseline_traj.P)
    return out
