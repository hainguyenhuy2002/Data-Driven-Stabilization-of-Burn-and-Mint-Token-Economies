"""
srf_lab — Simulation framework for the Stabilization Reserve Fund (SRF).

Public API:
    from srf_lab import (
        # core
        Config, simulate, monte_carlo,
        # bme
        BMEScenario, bme_flux,
        # market impact
        LinearImpact, AMMImpact, LiquidityShock,
        # controllers
        NoController, PassiveThresholdController,
        ProportionalController, PDController,
        LockOnlyController, SRFNoTaxController, FullSRFController,
        # stress
        gbm_returns, engineered_crash, vol_clustering, oracle_delay,
        # metrics
        compute_metrics,
    )
"""
from .config import Config
from .bme import BMEScenario, bme_flux
from .market_impact import LinearImpact, AMMImpact, LiquidityShock
from .controllers import (
    NoController, PassiveThresholdController,
    ProportionalController, PDController,
    LockOnlyController, SRFNoTaxController, FullSRFController,
)
from .stress import gbm_returns, engineered_crash, vol_clustering, oracle_delay
from .metrics import compute_metrics
from .simulator import simulate
from .montecarlo import monte_carlo

__all__ = [
    "Config", "simulate", "monte_carlo",
    "BMEScenario", "bme_flux",
    "LinearImpact", "AMMImpact", "LiquidityShock",
    "NoController", "PassiveThresholdController",
    "ProportionalController", "PDController",
    "LockOnlyController", "SRFNoTaxController", "FullSRFController",
    "gbm_returns", "engineered_crash", "vol_clustering", "oracle_delay",
    "compute_metrics",
]
