"""
Market-impact models used by the SRF trade engine.

Two models per the professor's review:

  * LinearImpact — baseline, ΔP ≈ κ · u / LD with hard cap.
  * AMMImpact    — constant-product (xy=k) AMM with explicit slippage and
                    in-pool reserves. Closed-form: after a buy of size u (USD),
                    the spot price becomes P · (x+u)^2 / x^2  (approximately).

Both impact models accept a LiquidityShock that temporarily reduces the
on-chain liquidity depth (LD_t) — modelling a flight of LPs during a crash.

API:
    impact.price_impact(u_usd, LD_t, P, R) -> (delta_P_pct, executed_usd)
    impact.liquidity_depth(MC_t, t)        -> LD_t

`u_usd > 0` means the SRF buys; `u_usd < 0` means the SRF sells.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


# ------------------------------------------------------------------ #
@dataclass
class LiquidityShock:
    enabled: bool = False
    start: int = 150
    length: int = 20
    drop_frac: float = 0.6   # liquidity drops to (1 - drop_frac) * normal during shock

    def multiplier(self, t: int) -> float:
        if not self.enabled:
            return 1.0
        if self.start <= t < self.start + self.length:
            return max(1e-3, 1.0 - self.drop_frac)
        return 1.0


# ------------------------------------------------------------------ #
class _BaseImpact:
    name: str = "base"

    def __init__(self, liquidity_frac: float = 0.10,
                 sell_cap_frac: float = 0.20,
                 shock: LiquidityShock | None = None):
        self.liquidity_frac = liquidity_frac
        self.sell_cap_frac = sell_cap_frac
        self.shock = shock or LiquidityShock(enabled=False)

    def liquidity_depth(self, MC_t: float, t: int) -> float:
        return self.liquidity_frac * MC_t * self.shock.multiplier(t)

    def price_impact(self, u_usd: float, LD_t: float,
                     P: float, R: float):
        raise NotImplementedError


# ------------------------------------------------------------------ #
class LinearImpact(_BaseImpact):
    """Linear baseline:  ΔP/P  = sign(u) · min(|u|, cap) · κ / LD."""
    name = "linear"

    def __init__(self, kappa: float = 0.5, **kw):
        super().__init__(**kw)
        self.kappa = kappa

    def price_impact(self, u_usd, LD_t, P, R):
        if LD_t <= 0:
            return 0.0, 0.0
        if u_usd > 0:                # buy
            cap = R                   # cannot spend more than reserve
        elif u_usd < 0:              # sell
            cap = self.sell_cap_frac * LD_t
        else:
            return 0.0, 0.0
        u_act = np.sign(u_usd) * min(abs(u_usd), cap)
        delta_pct = u_act * self.kappa / LD_t
        return float(delta_pct), float(u_act)


# ------------------------------------------------------------------ #
class AMMImpact(_BaseImpact):
    """
    Constant-product (xy=k) AMM impact.

        Pool reserves: x (USD), y (token). Spot price P_pool = x / y.
        Setting x = LD_t   and  y = LD_t / P_pool   (==> P_pool = P).
        After a buy of u USD, new reserves are (x+u, y - dy) where
        (x+u)(y-dy) = xy  =>  dy = u·y/(x+u). New spot = (x+u)/(y-dy)
        = (x+u)^2 / x^2 · P. ΔP/P = (x+u)^2/x^2 − 1.

        Sell of |u| USD pulls liquidity in the opposite direction.
    """
    name = "amm"

    def price_impact(self, u_usd, LD_t, P, R):
        if LD_t <= 0:
            return 0.0, 0.0
        x = LD_t
        if u_usd > 0:
            cap = R
            u_act = min(u_usd, cap)
            new_ratio = (x + u_act) ** 2 / x ** 2
            return float(new_ratio - 1.0), float(u_act)
        if u_usd < 0:
            cap = self.sell_cap_frac * LD_t
            u_act_abs = min(abs(u_usd), cap)
            new_ratio = x ** 2 / (x + u_act_abs) ** 2
            return float(new_ratio - 1.0), float(-u_act_abs)
        return 0.0, 0.0


def make_impact(model: str = "linear", **kw) -> _BaseImpact:
    model = model.lower()
    if model == "linear":
        return LinearImpact(**kw)
    if model == "amm":
        # AMMImpact doesn't take kappa (no linear slippage parameter)
        kw.pop("kappa", None)
        return AMMImpact(**kw)
    raise ValueError(f"Unknown market-impact model: {model}")
