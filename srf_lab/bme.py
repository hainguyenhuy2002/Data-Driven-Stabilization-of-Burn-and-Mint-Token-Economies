"""
Burn-and-Mint Equilibrium organic dynamics.

Five canonical scenarios per the methodology:
    1. neutral             — mint rate matches burn rate (mu = beta), supply ~ constant.
    2. emission_heavy      — mu > beta, dilutive emissions (e.g., Render Network early phase).
    3. burn_heavy          — mu < beta, deflationary BME (utility-driven burn dominates).
    4. reward_inflation    — emissions scale with price drop to "compensate" stakers
                              (pathological feedback that triggers death spirals).
    5. demand_collapse     — utility demand U(t) drops sharply, burn evaporates.

Returns the net mint - burn flux (Δ supply per day) given the current state.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BMEScenario:
    name: str
    mu: float           # base daily mint rate (fraction of S)
    beta: float         # base daily burn rate (fraction of S)
    elasticity: float   # demand sensitivity
    description: str = ""

    def utility_demand(self, P: float, P_target: float, t: int, T: int) -> float:
        """Token-utility demand U(t) (in token units / day). Subclasses override."""
        raise NotImplementedError


# --------- Concrete scenarios ---------
class _Neutral(BMEScenario):
    def utility_demand(self, P, P_target, t, T):
        return 1.0


class _EmissionHeavy(BMEScenario):
    def utility_demand(self, P, P_target, t, T):
        # Demand is constant; emissions outpace burns.
        return 1.0


class _BurnHeavy(BMEScenario):
    def utility_demand(self, P, P_target, t, T):
        # Demand grows linearly during the year; burn dominates.
        return 1.0 + 0.5 * t / T


class _RewardInflation(BMEScenario):
    """Emissions inflate when the price falls below target (the BME pathology)."""
    def utility_demand(self, P, P_target, t, T):
        return 1.0


class _DemandCollapse(BMEScenario):
    """Utility demand collapses partway through the year."""
    def utility_demand(self, P, P_target, t, T):
        if t < 0.4 * T:
            return 1.0
        return max(0.05, 1.0 - 3.0 * (t / T - 0.4))


_SCENARIO_MAP = {
    "off":              ("Off",                _Neutral, 0.0,    0.0,    0.0,
                          "BME organic flux disabled (constant supply ablation)."),
    "neutral":          ("Neutral",            _Neutral, 0.0010, 0.0010, 0.5,
                          "Balanced mint/burn, supply roughly constant."),
    "emission_heavy":   ("Emission-heavy",     _EmissionHeavy, 0.0030, 0.0010, 0.0,
                          "Dilutive emissions exceed burn (typical early-phase DePIN)."),
    "burn_heavy":       ("Burn-heavy",         _BurnHeavy, 0.0010, 0.0030, 1.0,
                          "Deflationary regime: utility-driven burn dominates emissions."),
    "reward_inflation": ("Reward-driven inflation", _RewardInflation, 0.0010, 0.0010, 0.0,
                          "Emissions inflate when the price falls (pathological feedback)."),
    "demand_collapse":  ("Demand collapse",    _DemandCollapse, 0.0010, 0.0010, 1.0,
                          "Utility demand evaporates partway through, burn vanishes."),
}


def make_scenario(kind: str) -> BMEScenario:
    spec = _SCENARIO_MAP.get(kind)
    if spec is None:
        raise ValueError(f"Unknown BME scenario: {kind}")
    name, cls, mu, beta, ela, desc = spec
    return cls(name=name, mu=mu, beta=beta, elasticity=ela, description=desc)


def bme_flux(scenario: BMEScenario, S: float, P: float, P_target: float,
             t: int, T: int) -> float:
    """
    Daily net (mint - burn) flux applied to supply.

        dS_org/dt = M(t) - B(P, U)

    where M(t) is the protocol's emission schedule and B(P, U) is the
    utility-driven burn that depends on price and demand U.

    Returns: signed token delta for day t.
    """
    if scenario.name == "Off":
        return 0.0

    # Base mint
    mint = scenario.mu * S

    # Reward-inflation pathology: emissions ramp up when price is below target.
    if isinstance(scenario, _RewardInflation):
        if P < P_target:
            mint *= (1.0 + 5.0 * (P_target - P) / max(P_target, 1e-9))

    # Utility-driven burn: B = beta * S * U * (P / P_target)^elasticity
    #   higher price -> more demand burns more tokens (typical BME).
    U = scenario.utility_demand(P, P_target, t, T)
    burn = scenario.beta * S * U * (max(P, 1e-9) / max(P_target, 1e-9)) ** scenario.elasticity

    return mint - burn


def list_scenarios():
    return list(_SCENARIO_MAP.keys())
