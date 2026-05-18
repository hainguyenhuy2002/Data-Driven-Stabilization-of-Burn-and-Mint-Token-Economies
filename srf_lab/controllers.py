"""
SRF controller hierarchy.

All controllers expose the same step() signature so the simulator and metrics
modules can treat them uniformly.

Controllers (Section 4.3 baselines):

    NoController                — no intervention, BME runs open-loop.
    PassiveThresholdController  — buy at fixed % drop, sell at fixed % rise (no PD).
    ProportionalController      — P-only (omega_d = 0), no lock, no tax.
    PDController                — full PD, no lock, no tax.
    LockOnlyController          — only the volatility-triggered escrow lock active.
    SRFNoTaxController          — PD + lock + safeguards, no dynamic tax.
    FullSRFController           — all four mechanisms enabled (proposed model).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepResult:
    delta_P_srf: float = 0.0
    lock_ratio: float = 0.0
    eta: float = 0.0
    u_actual_usd: float = 0.0
    R_after: float = 0.0


# ------------------------------------------------------------------ #
class _ControllerBase:
    name: str = "base"
    label: str = "base"

    def __init__(self, R0: float):
        self.R = R0
        self.prev_R = R0

    # main step
    def step(self, *, P, P_prev, P_target, MC, Vol, t,
             impact, ell_max, gamma_lock, deadband, rate_limit,
             tax_max, rho_target,
             omega_p, omega_d) -> StepResult:
        raise NotImplementedError

    # helpers
    def _maybe_tax(self, MC, Vol, tax_max, rho_target):
        """Optional self-financing tax. Returns the tax rate η_t actually applied."""
        return 0.0   # default: no tax

    def _maybe_lock(self, e, v, gamma_lock, ell_max):
        return 0.0

    def _execute_pd(self, *, P, P_prev, P_target, MC, t,
                    impact, deadband, rate_limit, omega_p, omega_d):
        """Compute PD demand, run it through the impact model, update reserve.
        Returns (delta_P_srf_pct, u_actual_usd).
        """
        e = (P_target - P) / max(P_target, 1e-9)
        v = (P - P_prev) / max(P_prev, 1e-9)
        if abs(e) < deadband:
            return 0.0, 0.0
        u_demand = self.R * (omega_p * e - omega_d * v)
        LD = impact.liquidity_depth(MC, t)
        delta_pct, u_act = impact.price_impact(u_demand, LD, P, self.R)
        # rate limiter (clip absolute pct impact)
        delta_pct = float(np.clip(delta_pct, -rate_limit, rate_limit))
        # update reserve
        if u_act > 0:                       # buy spends reserve
            self.R = max(0.0, self.R - u_act)
        elif u_act < 0:                     # sell adds USD to reserve
            self.R += abs(u_act)
        return delta_pct, u_act


# ------------------------------------------------------------------ #
class NoController(_ControllerBase):
    name = "none"
    label = "No intervention"

    def step(self, **kw) -> StepResult:
        return StepResult(delta_P_srf=0.0, lock_ratio=0.0, eta=0.0,
                          u_actual_usd=0.0, R_after=self.R)


class PassiveThresholdController(_ControllerBase):
    """
    Threshold reserve: buys a fixed USD chunk whenever price drops below
    `threshold_buy` (e.g. -10% from EMA target), sells when above
    `threshold_sell`. No PD, no lock, no tax. Used as a passive baseline.
    """
    name = "passive_threshold"
    label = "Passive threshold reserve"

    def __init__(self, R0, threshold_buy=-0.10, threshold_sell=0.10,
                 chunk_frac=0.02):
        super().__init__(R0)
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.chunk_frac = chunk_frac        # fraction of R to spend per intervention

    def step(self, *, P, P_prev, P_target, MC, t, impact,
             rate_limit, **kw) -> StepResult:
        e = (P_target - P) / max(P_target, 1e-9)
        u_demand = 0.0
        # buy if price below threshold
        if e > -self.threshold_buy:        # e > 0.10 means 10% below target
            u_demand = self.R * self.chunk_frac
        elif e < -self.threshold_sell:     # 10% above target
            u_demand = -self.R * self.chunk_frac
        if u_demand == 0.0:
            return StepResult(R_after=self.R)
        LD = impact.liquidity_depth(MC, t)
        delta_pct, u_act = impact.price_impact(u_demand, LD, P, self.R)
        delta_pct = float(np.clip(delta_pct, -rate_limit, rate_limit))
        if u_act > 0:
            self.R = max(0.0, self.R - u_act)
        else:
            self.R += abs(u_act)
        return StepResult(delta_P_srf=delta_pct, u_actual_usd=u_act, R_after=self.R)


class ProportionalController(_ControllerBase):
    """P-only (no derivative damping, no lock, no tax)."""
    name = "p_only"
    label = "P-only controller"

    def step(self, **kw):
        kw["omega_d"] = 0.0   # force derivative gain off
        delta, u_act = self._execute_pd(**_pd_kwargs(kw))
        return StepResult(delta_P_srf=delta, u_actual_usd=u_act, R_after=self.R)


class PDController(_ControllerBase):
    """Full PD, no lock, no tax."""
    name = "pd"
    label = "PD controller (no lock, no tax)"

    def step(self, **kw):
        delta, u_act = self._execute_pd(**_pd_kwargs(kw))
        return StepResult(delta_P_srf=delta, u_actual_usd=u_act, R_after=self.R)


def _compute_lock(*, P, P_prev, P_target, gamma_lock, ell_max,
                    realized_vol=0.0, lock_use_realized_vol=True):
    """Lock ratio derived from short-window realized volatility (preferred)
    or, in the legacy fallback, from |e_t|+|v_t|.
    """
    if lock_use_realized_vol:
        # rvol is the std of the last W daily returns (a "sudden-move" indicator).
        return float(min(ell_max, gamma_lock * realized_vol))
    e = (P_target - P) / max(P_target, 1e-9)
    v = (P - P_prev) / max(P_prev, 1e-9)
    return float(min(ell_max, gamma_lock * (abs(e) + abs(v))))


class LockOnlyController(_ControllerBase):
    """Only the volatility-triggered escrow lock; no trade engine, no tax."""
    name = "lock_only"
    label = "Volatility-triggered lock only"

    def step(self, *, P, P_prev, P_target, MC, t, gamma_lock, ell_max,
             realized_vol=0.0, lock_use_realized_vol=True, **kw) -> StepResult:
        ell = _compute_lock(P=P, P_prev=P_prev, P_target=P_target,
                             gamma_lock=gamma_lock, ell_max=ell_max,
                             realized_vol=realized_vol,
                             lock_use_realized_vol=lock_use_realized_vol)
        return StepResult(lock_ratio=ell, R_after=self.R)


class SRFNoTaxController(_ControllerBase):
    """PD + lock + safeguards, no dynamic tax (Section 3 ablation)."""
    name = "srf_no_tax"
    label = "SRF (no dynamic tax)"

    def step(self, **kw):
        ell = _compute_lock(P=kw["P"], P_prev=kw["P_prev"], P_target=kw["P_target"],
                             gamma_lock=kw["gamma_lock"], ell_max=kw["ell_max"],
                             realized_vol=kw.get("realized_vol", 0.0),
                             lock_use_realized_vol=kw.get("lock_use_realized_vol", True))
        delta, u_act = self._execute_pd(**_pd_kwargs(kw))
        return StepResult(delta_P_srf=delta, lock_ratio=ell,
                          u_actual_usd=u_act, R_after=self.R)


class FullSRFController(_ControllerBase):
    """Proposed model: PD + safeguards + lock + dynamic tax."""
    name = "full_srf"
    label = "Full SRF (PD + lock + tax)"

    def step(self, **kw):
        # 1) Self-financing tax refill BEFORE the trade so the engine sees fresh reserve.
        eta = self._tax_refill(MC=kw["MC"], Vol=kw["Vol"],
                                tax_max=kw["tax_max"], rho_target=kw["rho_target"])
        # 2) lock from short-window realized vol
        ell = _compute_lock(P=kw["P"], P_prev=kw["P_prev"], P_target=kw["P_target"],
                             gamma_lock=kw["gamma_lock"], ell_max=kw["ell_max"],
                             realized_vol=kw.get("realized_vol", 0.0),
                             lock_use_realized_vol=kw.get("lock_use_realized_vol", True))
        # 3) trade
        delta, u_act = self._execute_pd(**_pd_kwargs(kw))
        return StepResult(delta_P_srf=delta, lock_ratio=ell, eta=eta,
                          u_actual_usd=u_act, R_after=self.R)

    def _tax_refill(self, *, MC, Vol, tax_max, rho_target):
        R_target = rho_target * MC
        if R_target <= 0:
            return 0.0
        eta = tax_max * max(0.0, 1.0 - self.R / R_target)
        self.R += eta * Vol
        return float(eta)


# ------------------------------------------------------------------ #
def _pd_kwargs(kw: dict) -> dict:
    """Strip kwargs that _execute_pd does not need."""
    keep = ("P", "P_prev", "P_target", "MC", "t", "impact",
            "deadband", "rate_limit", "omega_p", "omega_d")
    return {k: kw[k] for k in keep if k in kw}
