#!/usr/bin/env python3
"""
Run the SRF on the 10-token panel and aggregate metrics by group.

For each token we run N_SEEDS independent realisations of the volatility
profile (different rng seeds drive the calibrated synthetic returns).
The SRF is then evaluated on each realisation against an unregulated baseline
that uses *the same* return realisation, so paired metrics are like-for-like.

Outputs
-------
results/tables/table_token_panel.csv      (per-token, mean over seeds)
results/tables/table_token_groups.csv     (group means, well-fit subset)
results/tables/table_unfit_tokens.csv     (tokens classified as limitation)
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from srf_lab import (
    Config, simulate,
    NoController, FullSRFController, SRFNoTaxController,
    LockOnlyController, PDController,
    compute_metrics,
)
from srf_lab.data_loader import (
    PANEL, panel_groups, fetch_real, load_calibrated,
)

TBL_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TBL_DIR, exist_ok=True)

# ---------------------------------------------------------------- #
# Suitability criteria for the short-window dampening mechanism
#   well-fit: drift change < 50% AND rolling-DD reduction > 15%
# ---------------------------------------------------------------- #
DRIFT_THRESHOLD = 1.00         # |log(P_T_srf) - log(P_T_base)| < 1.0  (≈ ratio 0.37–2.71)
RDD_THRESHOLD   = 0.15         # rolling-window MaxDD reduction must exceed 15%


def evaluate_token(spec, n_seeds: int = 10) -> list[dict]:
    """Run controllers on synthetic realisations of one token's profile."""
    rows = []
    controllers = [
        ('No intervention', NoController),
        ('Lock-only',       LockOnlyController),
        ('PD',              PDController),
        ('SRF (no tax)',    SRFNoTaxController),
        ('Full SRF',        FullSRFController),
    ]
    for ctrl_label, cls in controllers:
        per = []
        for s in range(n_seeds):
            data = load_calibrated(panel=[spec], seed=42 + 17*s)
            df = data[spec.coin_id]
            r_mkt = df['Return_hist'].to_numpy()
            vol_t = df['Vol_hist'].to_numpy()
            P0 = float(df['P_hist'].iloc[0])
            S0 = float(df['MC_hist'].iloc[0] / P0)
            cfg = Config(T=len(df), seed=42+s, P0=P0, S0=S0,
                          stress_kind='real', bme_scenario='neutral')
            R0 = cfg.rho_0 * P0 * S0
            base = simulate(cfg, NoController(R0=R0), r_mkt=r_mkt, daily_volume_usd=vol_t)
            traj = simulate(cfg, cls(R0=R0),         r_mkt=r_mkt, daily_volume_usd=vol_t)
            m = compute_metrics(traj, baseline_traj=base)
            per.append(m)
        agg = pd.DataFrame(per).mean(numeric_only=True).to_dict()
        agg.update({'token': spec.ticker, 'group': spec.group, 'controller': ctrl_label})
        rows.append(agg)
    return rows


def is_well_fit(full_row: dict) -> bool:
    drift = full_row.get('drift_preservation', float('nan'))
    rdd   = full_row.get('rolling_dd_reduction', float('nan'))
    if not np.isfinite(drift) or not np.isfinite(rdd):
        return False
    return (drift < DRIFT_THRESHOLD) and (rdd > RDD_THRESHOLD)


def main():
    print('[token panel] running per-token simulations, 10 seeds each ...')
    all_rows = []
    for spec in PANEL:
        rows = evaluate_token(spec, n_seeds=10)
        all_rows.extend(rows)
        full = next(r for r in rows if r['controller'] == 'Full SRF')
        fit = '[fit]    ' if is_well_fit(full) else '[unfit]  '
        print(f'  {fit}{spec.ticker:5s} ({spec.group:14s})  '
              f'drift={full["drift_preservation"]*100:5.1f}%  '
              f'r-DD={-full.get("rolling_dd_reduction",0)*100:+5.1f}%  '
              f'r-vol={-full.get("rolling_vol_reduction",0)*100:+5.1f}%  '
              f'worst={-full.get("worst_day_reduction",0)*100:+5.1f}%')

    df_all = pd.DataFrame(all_rows)
    keep = ['token', 'group', 'controller',
             'ann_vol', 'rolling_vol_14', 'rolling_dd_14', 'worst_day_pct',
             'max_drawdown', 'expected_shortfall_5',
             'drift_preservation', 'terminal_log_excess',
             'rolling_vol_reduction', 'rolling_dd_reduction', 'worst_day_reduction',
             'avg_lock', 'peak_lock', 'avg_tax', 'reserve_depletion']
    df_all = df_all.reindex(columns=keep)
    df_all.to_csv(os.path.join(TBL_DIR, 'table_token_panel.csv'), index=False)

    # Classify each token
    full = df_all[df_all['controller'] == 'Full SRF'].copy()
    full['fit'] = full.apply(lambda r: 'fit' if is_well_fit(r.to_dict()) else 'unfit', axis=1)

    # Group-level for well-fit tokens
    fit_tbl = full[full['fit'] == 'fit']
    grp = fit_tbl.groupby('group')[
        ['rolling_vol_14', 'rolling_dd_14', 'worst_day_pct',
         'rolling_vol_reduction', 'rolling_dd_reduction', 'worst_day_reduction',
         'drift_preservation', 'reserve_depletion', 'peak_lock', 'avg_tax']
    ].mean()
    grp.to_csv(os.path.join(TBL_DIR, 'table_token_groups.csv'))

    unfit = full[full['fit'] == 'unfit'][['token', 'group',
        'drift_preservation', 'rolling_dd_reduction', 'rolling_vol_reduction',
        'worst_day_reduction']]
    unfit.to_csv(os.path.join(TBL_DIR, 'table_unfit_tokens.csv'), index=False)

    print('\n[token panel] WELL-FIT group means (Full SRF):')
    print(grp.round(4).to_string())
    print('\n[token panel] UN-FIT (limitation) tokens:')
    print(unfit.round(4).to_string())
    print(f'\nSaved to {TBL_DIR}')


if __name__ == '__main__':
    main()
