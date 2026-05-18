"""
Daily Euler-Maruyama simulator that wires all the pieces together.

Key design notes
----------------
* The natural market force `r_mkt[t]` is exogenous (provided by the caller).
* Organic BME mint/burn flux changes the *circulating* supply S_t before the
  controller acts, addressing the professor's first comment.
* The controller is opaque: it implements the .step() interface, so the
  same loop runs every baseline.
* The caller chooses the market-impact model (linear vs AMM, with or
  without liquidity shock).
* The "floor" (controller target) supports two modes:
    - "ema":     classic EMA tracker (drifts down with the price).
    - "hybrid":  max(EMA, lower-confidence bound, governance floor).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import Config
from .bme import BMEScenario, bme_flux, make_scenario
from .market_impact import LiquidityShock, make_impact, _BaseImpact
from .controllers import _ControllerBase, NoController
from .stress import (gbm_returns, engineered_crash, vol_clustering, oracle_delay)


# ------------------------------------------------------------------ #
@dataclass
class TrajectoryResult:
    P:           np.ndarray
    P_target:    np.ndarray
    S:           np.ndarray            # circulating supply
    R:           np.ndarray            # reserve balance (USD)
    lock_ratio:  np.ndarray
    eta:         np.ndarray            # daily tax rate
    delta_P_srf: np.ndarray            # SRF-induced %-change per day
    u_actual_usd: np.ndarray           # USD traded (signed)
    r_mkt:       np.ndarray            # exogenous market returns used


# ------------------------------------------------------------------ #
def _build_returns(cfg: Config, override: Optional[np.ndarray]) -> np.ndarray:
    if override is not None:
        return np.asarray(override)[:cfg.T]
    if cfg.stress_kind == "gbm":
        return gbm_returns(cfg.T, cfg.drift, cfg.sigma, seed=cfg.seed)
    if cfg.stress_kind == "gbm_crash":
        return engineered_crash(cfg.T, cfg.drift, cfg.sigma,
                                 cfg.crash_t, cfg.crash_len,
                                 cfg.crash_mean, cfg.crash_sigma, seed=cfg.seed)
    if cfg.stress_kind == "vol_cluster":
        return vol_clustering(cfg.T, cfg.drift, cfg.sigma * 0.5, cfg.sigma * 2.0,
                                regime_len=30, seed=cfg.seed)
    raise ValueError(f"Unknown stress_kind: {cfg.stress_kind}")


def _local_reference(cfg: Config, P_hist: list[float], P_ema_fast: float) -> float:
    """
    Compute the controller's *local* reference price for the current step.

    Unlike a long-run floor, the local reference TRACKS the recent trend so that
    the SRF dampens sudden short-window deviations without fighting long-run drift.

    target_kind:
      "sma_short"    — simple moving average over a short window (default).
      "ema_fast"     — exponential MA with a short half-life (a few days).
      "hybrid_short" — max(SMA, short-window LCB, optional governance floor).
      "ema"          — legacy long-half-life EMA (kept for ablation comparisons).
    """
    if cfg.target_kind == "ema_fast" or cfg.target_kind == "ema":
        return float(P_ema_fast)

    if cfg.target_kind == "sma_short":
        if len(P_hist) >= cfg.sma_window:
            return float(np.mean(P_hist[-cfg.sma_window:]))
        return float(np.mean(P_hist))

    if cfg.target_kind == "hybrid_short":
        # SMA component
        if len(P_hist) >= cfg.sma_window:
            sma = float(np.mean(P_hist[-cfg.sma_window:]))
        else:
            sma = float(np.mean(P_hist))
        target = sma
        # LCB component on a short window
        if len(P_hist) >= cfg.hybrid_lookback:
            recent = np.asarray(P_hist[-cfg.hybrid_lookback:])
            lcb = float(np.quantile(recent, cfg.hybrid_q_lcb))
            target = max(target, lcb)
        # Optional emergency governance floor
        if cfg.governance_floor > 0:
            target = max(target, cfg.governance_floor)
        return float(target)

    raise ValueError(f"Unknown target_kind: {cfg.target_kind}")


def _realized_vol(P_hist: list[float], window: int) -> float:
    """Std of the last `window` daily simple returns."""
    if len(P_hist) < window + 1:
        return 0.0
    arr = np.asarray(P_hist[-(window+1):])
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
    return float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0


def _excess_vol(P_hist: list[float], short_window: int, baseline_window: int) -> float:
    """
    Returns max(0, σ_short - σ_baseline). This is a "vol-of-vol" signal: it
    fires only when recent volatility EXCEEDS the token's slower baseline,
    so naturally-turbulent tokens are not over-locked.
    """
    rv_short = _realized_vol(P_hist, short_window)
    rv_base = _realized_vol(P_hist, baseline_window)
    return max(0.0, rv_short - rv_base)


# ------------------------------------------------------------------ #
def simulate(cfg: Config,
             controller: _ControllerBase,
             *,
             r_mkt: Optional[np.ndarray] = None,
             impact: Optional[_BaseImpact] = None,
             scenario: Optional[BMEScenario] = None,
             daily_volume_usd: Optional[Sequence[float]] = None) -> TrajectoryResult:
    """
    Run one closed-loop simulation. Returns full trajectories.

    Parameters
    ----------
    cfg            simulation configuration
    controller     a controllers._ControllerBase instance
    r_mkt          optional override return series (e.g., real BAND returns)
    impact         optional pre-built impact model (else built from cfg)
    scenario       optional pre-built BMEScenario (else built from cfg)
    daily_volume_usd  optional volume series for the dynamic-tax mechanism;
                       if None, defaults to 5% of MC each day (synthetic).
    """
    rng = np.random.default_rng(cfg.seed)
    r_mkt_arr = _build_returns(cfg, r_mkt)
    T = len(r_mkt_arr)

    # impact model
    if impact is None:
        shock = LiquidityShock(enabled=cfg.liq_shock,
                                start=cfg.liq_shock_t,
                                length=cfg.liq_shock_len,
                                drop_frac=cfg.liq_shock_drop)
        impact = make_impact(cfg.impact_model,
                              liquidity_frac=cfg.liquidity_frac,
                              sell_cap_frac=cfg.sell_cap_frac,
                              shock=shock,
                              kappa=cfg.kappa)

    # BME scenario
    if scenario is None:
        scenario = make_scenario(cfg.bme_scenario)

    # Allocations
    P  = np.zeros(T); P_target = np.zeros(T)
    S  = np.zeros(T); R = np.zeros(T)
    ell = np.zeros(T); eta = np.zeros(T)
    dP_srf = np.zeros(T); u_act = np.zeros(T)

    P[0] = cfg.P0
    P_target[0] = cfg.P0
    S[0] = cfg.S0
    MC0 = cfg.P0 * cfg.S0
    controller.R = cfg.rho_0 * MC0
    R[0] = controller.R

    # synthetic daily volume series (if real not provided)
    if daily_volume_usd is None:
        # 5% of running MC, with mild noise
        vol_default = 0.05
    else:
        daily_volume_usd = np.asarray(daily_volume_usd)

    P_hist: list[float] = [cfg.P0]

    # observation lag (oracle delay)
    obs_lag = max(0, int(cfg.oracle_delay))

    # EMA tracker state
    ema_fast = P[0]

    for t in range(1, T):
        # --- 1. Update LOCAL REFERENCE on the OBSERVED price history ---
        # observed price for the controller (with optional oracle lag)
        idx_obs = max(0, t - 1 - obs_lag)
        P_obs = P[idx_obs]
        # fast EMA for ema_fast / legacy ema target_kind
        ema_fast = ema_fast + cfg.ema_lambda * (P_obs - ema_fast)
        P_target[t] = _local_reference(cfg, P_hist, ema_fast)

        # short-window EXCESS realized vol drives the lock — fires only on
        # spikes above the token's own slow baseline volatility.
        rvol_t = _excess_vol(P_hist, cfg.lock_vol_window, cfg.lock_baseline_window)

        # --- 2. Apply organic BME flux to supply BEFORE the controller acts ---
        flux = bme_flux(scenario, S[t-1], P[t-1], P_target[t], t, T)
        S[t] = max(1.0, S[t-1] + flux)

        # current market cap and volume
        MC_t = P[t-1] * S[t]
        Vol_t = (daily_volume_usd[t] if daily_volume_usd is not None
                  else vol_default * MC_t)

        # --- 3. Controller step ---
        res = controller.step(
            P=P_obs, P_prev=P[max(0, idx_obs-1)],
            P_target=P_target[t], MC=MC_t, Vol=Vol_t, t=t,
            impact=impact,
            ell_max=cfg.ell_max, gamma_lock=cfg.gamma,
            deadband=cfg.deadband, rate_limit=cfg.rate_limit,
            tax_max=cfg.tax_max, rho_target=cfg.rho_target,
            omega_p=cfg.omega_p, omega_d=cfg.omega_d,
            realized_vol=rvol_t,
            lock_use_realized_vol=cfg.lock_use_realized_vol,
        )
        ell[t]    = res.lock_ratio
        eta[t]    = res.eta
        dP_srf[t] = res.delta_P_srf
        u_act[t]  = res.u_actual_usd
        R[t]      = controller.R

        # --- 4. Price update (Euler-Maruyama) ---
        # dampened market force
        f_mkt = r_mkt_arr[t] * (1.0 - ell[t])
        P[t] = max(P[t-1] * (1.0 + f_mkt + dP_srf[t]), 1e-3)

        P_hist.append(P[t])

    return TrajectoryResult(
        P=P, P_target=P_target, S=S, R=R,
        lock_ratio=ell, eta=eta,
        delta_P_srf=dP_srf, u_actual_usd=u_act,
        r_mkt=r_mkt_arr,
    )
