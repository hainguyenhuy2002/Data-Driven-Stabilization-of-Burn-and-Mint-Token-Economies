#!/usr/bin/env python3
"""
Run the full SRF experimental study and save tables + figures.

Usage:
    python notebooks/run_experiments.py           # full run (~2 min)
    python notebooks/run_experiments.py --quick   # reduced MC sample (faster)

Outputs are written to:
    results/figs/         (PDF + PNG figures)
    results/tables/       (CSV + LaTeX tables)
"""
from __future__ import annotations
import os, sys, argparse, time
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # headless
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from srf_lab import (
    Config, simulate, monte_carlo,
    NoController, PassiveThresholdController, ProportionalController,
    PDController, LockOnlyController, SRFNoTaxController, FullSRFController,
    compute_metrics,
)

FIG_DIR = os.path.join(ROOT, 'results', 'figs')
TBL_DIR = os.path.join(ROOT, 'results', 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TBL_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 200})

PALETTE = {
    'baseline':    '#9aa0a6',
    'passive':     '#bb6588',
    'p_only':      '#e9c46a',
    'pd':          '#f4a261',
    'lock_only':   '#9b5de5',
    'srf_no_tax':  '#264653',
    'full_srf':    '#2a9d8f',
    'crash':       '#e76f51',
}


def save_table(df: pd.DataFrame, stem: str, caption: str, label: str, **fmt):
    df.to_csv(os.path.join(TBL_DIR, stem + '.csv'))
    try:
        df.to_latex(os.path.join(TBL_DIR, stem + '.tex'),
                    caption=caption, label=label,
                    float_format=lambda x: f'{x:.4f}', **fmt)
    except Exception as e:
        print(f'  (latex export skipped: {e})')


# ====================================================================
# STAGE 1: Headline result + Figure 4
# ====================================================================
def run_headline(N_RUNS):
    print('\n[1/8] Headline result (no SRF vs Full SRF, single seed) ...')
    cfg = Config(T=365, seed=42, stress_kind='gbm_crash', bme_scenario='neutral')
    R0 = cfg.rho_0 * cfg.P0 * cfg.S0
    traj_base = simulate(cfg, NoController(R0=R0))
    traj_full = simulate(cfg, FullSRFController(R0=R0))
    m_base = compute_metrics(traj_base)
    m_full = compute_metrics(traj_full, baseline_traj=traj_base)
    summary = pd.DataFrame({'No intervention': m_base, 'Full SRF': m_full}).T
    save_table(summary, 'table2_main_result',
                'Main result on the engineered-crash scenario (single seed).',
                'tab:main')

    # Figure 4
    t = np.arange(cfg.T)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 8.5),
                                       gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    a1.plot(t, traj_base.P, label='No intervention', color=PALETTE['baseline'], linestyle='--', alpha=0.85)
    a1.plot(t, traj_full.P, label='Full SRF (PD + lock + tax)', color=PALETTE['full_srf'], lw=2)
    a1.plot(t, traj_full.P_target, label='Local SMA target', color='#264653', linestyle=':', alpha=0.7)
    a1.axvspan(150, 160, color=PALETTE['crash'], alpha=0.10, label='Engineered crash')
    a1.set_ylabel('Price (USD)'); a1.legend(loc='lower left')
    a1.set_title('Figure 4. Price trajectory under engineered crash')

    a2.plot(t, traj_full.R/1e6, color='#9b5de5', lw=1.6, label='Reserve $R_t$ (M USD)')
    a2.fill_between(t, traj_full.R/1e6, color='#9b5de5', alpha=0.20)
    a2.set_ylabel('Reserve (M USD)'); a2.legend(loc='upper left')

    a3.plot(t, traj_full.lock_ratio*100, color='#f4a261', lw=1.6, label=r'Lock ratio $\ell_t$ (%)')
    a3.plot(t, traj_full.eta*100, color='#2a9d8f', lw=1.4, label=r'Tax rate $\eta_t$ (%)')
    a3.set_ylabel('% '); a3.set_xlabel('day'); a3.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_price_trajectory.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig4_price_trajectory.png'), bbox_inches='tight')
    plt.close(fig)

    # Figure 5: reserve + tax
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    traj_no_tax = simulate(cfg, SRFNoTaxController(R0=R0))
    a1.plot(t, traj_no_tax.R/1e6, label='SRF (no dynamic tax)',  color='#264653', lw=1.6)
    a1.plot(t, traj_full.R/1e6,   label='Full SRF (+ dynamic tax)', color='#2a9d8f', lw=1.8)
    a1.axhline(traj_full.R[0]/1e6, color='gray', linestyle=':', alpha=0.6, label='Initial $R_0$')
    a1.set_ylabel('Reserve (M USD)'); a1.legend(loc='lower left')
    a1.set_title('Figure 5. Reserve balance and dynamic tax rate')

    a2.plot(t, traj_full.eta*100, color='#2a9d8f', lw=1.6, label=r'$\eta_t$ (Full SRF)')
    a2.fill_between(t, traj_full.eta*100, color='#2a9d8f', alpha=0.20)
    a2.set_ylabel('Tax rate (%)'); a2.set_xlabel('day'); a2.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_reserve_tax.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig5_reserve_tax.png'), bbox_inches='tight')
    plt.close(fig)

    # Figure 6: lock ratio + effective supply
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    a1.plot(t, traj_full.lock_ratio*100, color='#9b5de5', lw=1.7, label=r'Lock ratio $\ell_t$ (%)')
    a1.fill_between(t, traj_full.lock_ratio*100, color='#9b5de5', alpha=0.2)
    a1.set_ylabel('Lock ratio (%)'); a1.legend(loc='upper left')
    a1.set_title('Figure 6. Volatility-triggered lock ratio and effective supply')
    S_eff = traj_full.S * (1 - traj_full.lock_ratio)
    a2.plot(t, traj_full.S/1e6,    label=r'Total supply $S_t$', color='#264653', linestyle='--')
    a2.plot(t, S_eff/1e6,          label=r'Effective supply $S_{\mathrm{eff},t}$', color='#2a9d8f', lw=1.8)
    a2.set_ylabel('Supply (M tokens)'); a2.set_xlabel('day'); a2.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_lock_supply.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig6_lock_supply.png'), bbox_inches='tight')
    plt.close(fig)
    return summary


# ====================================================================
# STAGE 2: Baseline ablation
# ====================================================================
def run_ablation(N_RUNS):
    print('\n[2/8] Baseline ablation across', N_RUNS, 'seeds ...')
    cfg = Config(T=365, seed=42, stress_kind='gbm_crash', bme_scenario='neutral')
    controllers = {
        'No intervention':         (NoController,                {}),
        'Passive threshold':        (PassiveThresholdController,  {}),
        'P-only':                   (ProportionalController,      {}),
        'PD':                       (PDController,                {}),
        'Lock-only':                (LockOnlyController,          {}),
        'SRF (no tax)':             (SRFNoTaxController,          {}),
        'Full SRF':                 (FullSRFController,           {}),
    }
    rows = []
    for label, (cls, kw) in controllers.items():
        df = monte_carlo(cfg, cls, n_runs=N_RUNS, controller_kwargs=kw)
        row = df.mean(numeric_only=True).to_dict()
        row['controller'] = label
        rows.append(row)
        print(f'    - {label:25s}  ann_vol={row["ann_vol"]*100:5.2f}%   max_dd={row["max_drawdown"]*100:5.2f}%')
    table = pd.DataFrame(rows).set_index('controller')
    keep = ['ann_vol', 'rolling_vol_14', 'max_drawdown', 'rolling_dd_14',
             'worst_day_pct', 'downside_vol', 'expected_shortfall_5',
             'time_under_dd_10pct', 'reserve_depletion', 'avg_lock', 'peak_lock',
             'avg_tax', 'intervention_cost']
    save_table(table[keep], 'table4_baseline_ablation',
                f'Baseline ablation under engineered crash (mean over {N_RUNS} seeds).',
                'tab:abl')
    return table[keep]


# ====================================================================
# STAGE 3: BME scenario sweep
# ====================================================================
def run_bme_scenarios(N_RUNS):
    print('\n[3/8] BME organic-flux scenarios ...')
    cfg = Config(T=365, seed=42, stress_kind='gbm_crash')
    scenarios = ['neutral', 'emission_heavy', 'burn_heavy', 'reward_inflation', 'demand_collapse']
    labels = {
        'neutral':           'Neutral',
        'emission_heavy':    'Emission-heavy',
        'burn_heavy':        'Burn-heavy',
        'reward_inflation':  'Reward-driven inflation',
        'demand_collapse':   'Demand collapse',
    }
    rows = []
    for sc in scenarios:
        cfg_sc = replace(cfg, bme_scenario=sc)
        base_df = monte_carlo(cfg_sc, NoController, n_runs=N_RUNS)
        full_df = monte_carlo(cfg_sc, FullSRFController, n_runs=N_RUNS)
        rows.append({
            'scenario':           labels[sc],
            'base_ann_vol':       base_df['ann_vol'].mean(),
            'base_max_dd':        base_df['max_drawdown'].mean(),
            'srf_ann_vol':        full_df['ann_vol'].mean(),
            'srf_max_dd':         full_df['max_drawdown'].mean(),
            'vol_reduction':      full_df['vol_reduction'].mean() if 'vol_reduction' in full_df else float('nan'),
            'dd_reduction':       full_df['dd_reduction'].mean() if 'dd_reduction' in full_df else float('nan'),
            'reserve_depletion':  full_df['reserve_depletion'].mean(),
            'peak_lock':          full_df['peak_lock'].mean(),
        })
        print(f'    - {labels[sc]:25s}  base_dd={rows[-1]["base_max_dd"]*100:5.2f}%   srf_dd={rows[-1]["srf_max_dd"]*100:5.2f}%')
    table = pd.DataFrame(rows).set_index('scenario')
    save_table(table, 'table3_bme_scenarios',
                f'BME organic-flux scenarios (mean over {N_RUNS} seeds).',
                'tab:bme')
    return table


# ====================================================================
# STAGE 4: Impact-model + liquidity-shock comparison
# ====================================================================
def run_impact(N_RUNS):
    print('\n[4/8] Impact model and liquidity-shock comparison ...')
    cfg = Config(T=365, seed=42, stress_kind='gbm_crash', bme_scenario='neutral')
    rows = []
    for impact_kind in ['linear', 'amm']:
        for liq_shock in [False, True]:
            cfg_im = replace(cfg, impact_model=impact_kind, liq_shock=liq_shock,
                              liq_shock_t=148, liq_shock_len=20, liq_shock_drop=0.7)
            df = monte_carlo(cfg_im, FullSRFController, n_runs=N_RUNS)
            row = df.mean(numeric_only=True).to_dict()
            row['impact_model'] = impact_kind
            row['liquidity_shock'] = liq_shock
            rows.append(row)
            print(f'    - impact={impact_kind:6s}  shock={str(liq_shock):5s}  max_dd={row["max_drawdown"]*100:5.2f}%')
    table = pd.DataFrame(rows).set_index(['impact_model', 'liquidity_shock'])
    keep = ['ann_vol', 'max_drawdown', 'reserve_depletion', 'peak_lock', 'intervention_cost', 'slippage_cost']
    save_table(table[keep], 'table_impact_models',
                'Impact-model comparison with optional liquidity shock.',
                'tab:impact')
    return table[keep]


# ====================================================================
# STAGE 5: Sensitivity heatmap (omega_p, omega_d) - Figure 7
# ====================================================================
def run_sensitivity(MC):
    print('\n[5/8] Sensitivity heatmap over (omega_p, omega_d) ...')
    # Use vol_cluster + small jitter so we measure SHORT-WINDOW dampening,
    # which is the proper objective for a circuit breaker.
    cfg = Config(T=365, seed=42, stress_kind='vol_cluster', bme_scenario='neutral')
    omega_p_grid = np.array([0.05, 0.10, 0.20, 0.40, 0.60, 1.00])
    omega_d_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Z = np.zeros((len(omega_p_grid), len(omega_d_grid)))
    for i, wp in enumerate(omega_p_grid):
        for j, wd in enumerate(omega_d_grid):
            cfg_g = replace(cfg, omega_p=float(wp), omega_d=float(wd))
            df = monte_carlo(cfg_g, FullSRFController, n_runs=MC)
            Z[i, j] = df['rolling_dd_14'].mean()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    im = ax.imshow(Z*100, origin='lower', aspect='auto', cmap='RdYlGn_r',
                    extent=[omega_d_grid[0]-0.5, omega_d_grid[-1]+0.5,
                            omega_p_grid[0]-0.025, omega_p_grid[-1]+0.025])
    ax.set_xticks(omega_d_grid); ax.set_yticks(omega_p_grid)
    ax.set_xlabel(r'$\omega_d$ (derivative gain)')
    ax.set_ylabel(r'$\omega_p$ (proportional gain)')
    ax.set_title(r'Sensitivity surface — rolling-14 MaxDD (%) by $(\omega_p, \omega_d)$')
    for i in range(len(omega_p_grid)):
        for j in range(len(omega_d_grid)):
            ax.text(omega_d_grid[j], omega_p_grid[i], f'{Z[i,j]*100:.1f}',
                    ha='center', va='center', color='black', fontsize=9)
    fig.colorbar(im, ax=ax, label='Rolling-14 MaxDD (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_sensitivity.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig7_sensitivity.png'), bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(Z, index=[f'wp={x}' for x in omega_p_grid],
                 columns=[f'wd={x}' for x in omega_d_grid]).to_csv(
        os.path.join(TBL_DIR, 'sensitivity_heatmap.csv'))
    print(f'    matrix shape {Z.shape}, min={Z.min()*100:.2f}%, max={Z.max()*100:.2f}%')


# ====================================================================
# STAGE 6: Failure boundary (rho_0, liq-shock drop) - Figure 8
# ====================================================================
def run_failure(MC):
    print('\n[6/8] Failure boundary under liquidity shock ...')
    cfg = Config(T=365, seed=42, stress_kind='vol_cluster', bme_scenario='neutral')
    drop_grid = np.linspace(0.0, 0.9, 6)
    rho_grid  = np.array([0.05, 0.075, 0.10, 0.15, 0.20])
    Z2 = np.zeros((len(rho_grid), len(drop_grid)))
    for i, rho in enumerate(rho_grid):
        for j, drop in enumerate(drop_grid):
            cfg_f = replace(cfg, rho_0=float(rho), liq_shock=True,
                             liq_shock_t=148, liq_shock_len=20,
                             liq_shock_drop=float(drop))
            df = monte_carlo(cfg_f, FullSRFController, n_runs=MC)
            Z2[i, j] = df['rolling_dd_14'].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z2*100, origin='lower', aspect='auto', cmap='RdYlGn_r',
                    extent=[drop_grid[0]-0.05, drop_grid[-1]+0.05,
                            rho_grid[0]-0.0125, rho_grid[-1]+0.0125])
    ax.set_xticks(drop_grid); ax.set_yticks(rho_grid)
    ax.set_xlabel('Liquidity-shock severity (drop fraction)')
    ax.set_ylabel(r'Initial reserve fraction $\rho_0$')
    ax.set_title('Failure boundary — rolling-14 MaxDD (%) under liquidity shock')
    for i in range(len(rho_grid)):
        for j in range(len(drop_grid)):
            ax.text(drop_grid[j], rho_grid[i], f'{Z2[i,j]*100:.1f}', ha='center', va='center', fontsize=9)
    fig.colorbar(im, ax=ax, label='Rolling-14 MaxDD (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig8_failure_boundary.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig8_failure_boundary.png'), bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(Z2, index=[f'rho={x}' for x in rho_grid],
                 columns=[f'drop={x:.2f}' for x in drop_grid]).to_csv(
        os.path.join(TBL_DIR, 'failure_boundary.csv'))
    print(f'    Z2 shape {Z2.shape}, min={Z2.min()*100:.2f}%, max={Z2.max()*100:.2f}%')


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Reduced MC samples')
    parser.add_argument('--stage', type=int, default=None,
                         help='Run only one stage (1..6); default runs all')
    args = parser.parse_args()

    N_RUNS_MAIN = 8 if args.quick else 30
    N_RUNS_SCEN = 6 if args.quick else 20
    MC_SENS    = 4 if args.quick else 12
    MC_FAIL    = 4 if args.quick else 8

    t0 = time.time()
    if args.stage in (None, 1):
        run_headline(N_RUNS_MAIN)
    if args.stage in (None, 2):
        run_ablation(N_RUNS_MAIN)
    if args.stage in (None, 3):
        run_bme_scenarios(N_RUNS_SCEN)
    if args.stage in (None, 4):
        run_impact(N_RUNS_MAIN)
    if args.stage in (None, 5):
        run_sensitivity(MC_SENS)
    if args.stage in (None, 6):
        run_failure(MC_FAIL)

    print(f'\nDone in {time.time()-t0:.1f}s. Outputs saved under results/.')
