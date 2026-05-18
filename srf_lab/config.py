"""
Simulation configuration dataclass.
Keep all defaults here so experiments only override what differs.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # Horizon
    T: int = 365                       # number of daily steps
    seed: int = 42

    # Initial state
    P0: float = 1.0                    # initial price ($)
    S0: float = 5_000_000              # initial circulating supply
    rho_0: float = 0.10                # R_0 / MC_0   (initial reserve fraction)

    # PD controller --- derivative-dominated:
    # ω_p is INTENTIONALLY tiny so the SRF does not anchor the price; the
    # primary signal is the velocity term, which fires only when the daily
    # return |v_t| spikes. ω_p ≈ 0.1 keeps a mild self-stabilising tendency
    # without pinning the price to its initial level.
    omega_p: float = 0.10               # very small restoring pull
    omega_d: float = 4.0                # strong derivative damping
    deadband: float = 0.04              # ignore short-window deviations < 4%
    rate_limit: float = 0.04            # max |Delta P_SRF| per day

    # Volatility-triggered escrow lock
    # The lock activates on EXCESS short-window vol over a slow baseline,
    # so a token's natural turbulence is treated as the new "calm" --- only
    # spikes above that baseline trigger circuit-breaker behaviour.
    gamma: float = 20.0                 # lock-sensitivity to excess vol
    ell_max: float = 0.55               # capped at 55%
    lock_vol_window: int = 5            # very short window for realized vol
    lock_baseline_window: int = 45      # slow baseline (45-day rolling vol)
    lock_use_realized_vol: bool = True  # if False, fall back to |e_t|+|v_t| (legacy)

    # Self-financing dynamic tax
    tax_max: float = 0.05
    rho_target: float = 0.15           # R_target / MC

    # Local-reference target. The mechanism is a short-horizon circuit breaker:
    # the target TRACKS the local recent trend so it does not anchor the price
    # to its initial level. With sma_window=7 the reference moves with the
    # market over ~one week, leaving long-run drift undamped.
    target_kind: str = "sma_short"     # "sma_short" | "ema_fast" | "hybrid_short" | "ema" (legacy)
    sma_window: int = 7                # 7-day SMA: short enough to follow the trend
    ema_lambda: float = 0.35           # fast EMA, half-life ~1.6 days
    hybrid_q_lcb: float = 0.25         # quantile for the short-window LCB component
    hybrid_lookback: int = 7           # short lookback
    governance_floor: float = 0.0      # OPTIONAL emergency absolute floor (0 disables)

    # Liquidity model
    impact_model: str = "linear"       # "linear" | "amm"
    liquidity_frac: float = 0.10       # LD_t = liquidity_frac * MC_t
    kappa: float = 0.5                 # linear-impact slippage
    sell_cap_frac: float = 0.20        # sell cap as fraction of LD_t

    # Liquidity shock (optional)
    liq_shock: bool = False
    liq_shock_t: int = 150
    liq_shock_len: int = 20
    liq_shock_drop: float = 0.6        # liquidity drops to (1-drop) * normal during shock

    # BME organic dynamics
    bme_scenario: str = "neutral"      # "off" | "neutral" | "emission_heavy" | "burn_heavy" | "reward_inflation" | "demand_collapse"
    bme_mu: float = 0.0                # base daily mint rate (fraction of S)
    bme_beta: float = 0.0              # base daily burn rate (fraction of S)
    bme_demand_elasticity: float = 1.0 # how strongly utility demand U(t) responds to price

    # Stress (when no historical returns supplied)
    stress_kind: str = "gbm_crash"     # "gbm" | "gbm_crash" | "vol_cluster" | "real"
    drift: float = 0.002
    sigma: float = 0.04
    crash_t: int = 150
    crash_len: int = 10
    crash_mean: float = -0.12
    crash_sigma: float = 0.05

    # Oracle delay (lag between true price and the price the controller observes)
    oracle_delay: int = 0
