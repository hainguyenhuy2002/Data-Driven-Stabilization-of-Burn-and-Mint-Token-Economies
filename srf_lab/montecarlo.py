"""
Monte Carlo wrapper — run a controller across N seeds and aggregate metrics.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import replace
from typing import Optional, Type

from .config import Config
from .controllers import _ControllerBase, NoController
from .simulator import simulate
from .metrics import compute_metrics


def monte_carlo(cfg: Config,
                ControllerCls: Type[_ControllerBase],
                n_runs: int = 50,
                seed_offset: int = 0,
                controller_kwargs: Optional[dict] = None,
                baseline_cls: Type[_ControllerBase] = NoController,
                **simulate_kwargs) -> pd.DataFrame:
    """
    Run `ControllerCls` and `baseline_cls` on N seeded copies of cfg, and
    return a DataFrame of per-run metrics for the regulated trajectory
    (with baseline-relative reductions where computable).
    """
    rows = []
    controller_kwargs = controller_kwargs or {}
    for i in range(n_runs):
        seed = cfg.seed + seed_offset + i
        run_cfg = replace(cfg, seed=seed)

        # baseline (for comparison metrics)
        b_ctrl = baseline_cls(R0=run_cfg.rho_0 * run_cfg.P0 * run_cfg.S0)
        traj_base = simulate(run_cfg, b_ctrl, **simulate_kwargs)

        # treated
        ctrl = ControllerCls(R0=run_cfg.rho_0 * run_cfg.P0 * run_cfg.S0,
                              **controller_kwargs)
        traj = simulate(run_cfg, ctrl, **simulate_kwargs)

        m = compute_metrics(traj, baseline_traj=traj_base)
        m["seed"] = seed
        m["run"] = i
        rows.append(m)
    return pd.DataFrame(rows)
