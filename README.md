# Stabilizing Token-Mediated Digital Marketplaces through Closed-Loop Reserve Funds and Dynamic Supply Locking

**Authors:** Huy Hai Nguyen (Tsinghua University), Binh Minh Nguyen (Hanoi University of Science and Technology)

> This repository contains the full simulation framework, experiment scripts, and LaTeX source for the paper:
>
> *"Stabilizing Token-Mediated Digital Marketplaces through Closed-Loop Reserve Funds and Dynamic Supply Locking"*

---

## Overview

Token-mediated digital marketplaces (storage, compute, data, oracle, liquidity) use native tokens as payment, incentive, and coordination instruments. When these tokens follow **Burn-and-Mint Equilibrium (BME)** designs, adverse price movements can trigger self-reinforcing volatility spirals.

This paper develops and evaluates a closed-loop **Stabilization Reserve Fund (SRF)** that combines four mutually reinforcing mechanisms:

| Mechanism | What it does | Paper section |
|---|---|---|
| **Reserve-funded PD feedback** | Buys/sells from a protocol reserve proportional to price deviation and momentum | §3.1 |
| **Volatility-triggered supply lock** | Temporarily reduces the free float of eligible supply when short-window vol spikes above baseline | §3.2 |
| **Intervention safeguards** | Deadband + rate limiter prevent the SRF itself from adding noise | §3.1 |
| **Self-financing dynamic tax** | Transaction fee replenishes the reserve when it falls below its target ratio | §3.3 |

### Key results (paper §5)

Under the upper-bound eligible-supply setting (θ = 1.00), the Full SRF achieves:
- **−17.6 %** rolling 14-day MaxDD (vs. no intervention)
- **−18.6 %** worst one-day drop
- **−22.5 %** global maximum drawdown
- Average **~26 %** rolling-DrawDown reduction across well-fit panel tokens

---

## Repository structure

```
.
├── README.md                        ← this file
├── requirements.txt                 ← Python dependencies
├── .gitignore
│
├── srf_lab/                         ← Core simulation library (§3)
│   ├── __init__.py                  ← Public API
│   ├── config.py                    ← Config dataclass — all hyperparameters (Table 1)
│   ├── simulator.py                 ← Euler-Maruyama daily loop (§3, Algorithm 1)
│   ├── controllers.py               ← Full SRF + all baselines (§3.1, §4.3)
│   ├── bme.py                       ← BME organic mint/burn dynamics (§3.3)
│   ├── market_impact.py             ← Linear & AMM price-impact models (§3.4)
│   ├── stress.py                    ← Synthetic stress generators (§4.1)
│   ├── metrics.py                   ← Evaluation metrics (§4.4)
│   ├── montecarlo.py                ← Multi-seed Monte Carlo wrapper
│   └── data_loader.py               ← Token panel + CoinGecko live fetch (§4.2)
│
├── experiments/                     ← Experiment entry points
│   ├── run_experiments.py           ← Reproduces Figures 4–8 and Tables 2–4
│   ├── run_token_panel.py           ← Token-panel backtest (§4.5, Table 5)
│   └── srf_experiments.ipynb        ← Interactive walkthrough notebook
│
├── results/                         ← Generated outputs (reproducible; git-ignored by default)
│   ├── figs/                        ← fig4_price_trajectory, fig5_reserve_tax, fig6_lock_supply,
│   │                                    fig7_sensitivity, fig8_failure_boundary  (PDF + PNG)
│   └── tables/                      ← table2_main_result, table3_bme_scenarios,
│                                        table4_baseline_ablation, … (CSV + LaTeX)
│
├── paper/                           ← LaTeX source
│   ├── main.tex                     ← Main manuscript
│   ├── cas-refs.bib                 ← Bibliography
│   ├── cas-dc.cls                   ← Elsevier CAS double-column class
│   ├── cas-sc.cls
│   ├── cas-common.sty
│   └── cas-model2-names.bst
│
└── figs/                            ← Architecture diagrams (used in paper)
    ├── srf_architecture.{pdf,png,svg}
    ├── srf_pipeline.{pdf,drawio}
    ├── srf_control_loop.pdf
    └── srf_integration_loop.pdf
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>

# (recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, NumPy ≥ 1.24, Pandas ≥ 2.0, Matplotlib ≥ 3.7, Requests ≥ 2.28 (only needed for the live CoinGecko fetch), Jupyter (for the notebook).

---

## Quick start

```python
from srf_lab import Config, simulate, compute_metrics
from srf_lab import NoController, FullSRFController

cfg = Config(T=365, seed=42, stress_kind="gbm_crash", bme_scenario="neutral")
R0  = cfg.rho_0 * cfg.P0 * cfg.S0   # initial reserve (USD)

# unregulated baseline
traj_base = simulate(cfg, NoController(R0=R0))

# Full SRF (PD + lock + dynamic tax)
traj_srf  = simulate(cfg, FullSRFController(R0=R0))

# evaluate
m = compute_metrics(traj_srf, baseline_traj=traj_base)
print(f"Rolling-14 MaxDD reduction: {m['rolling_dd_reduction']*100:.1f}%")
print(f"Worst-day drop reduction:   {m['worst_day_reduction']*100:.1f}%")
print(f"Reserve depletion:          {m['reserve_depletion']*100:.1f}%")
```

---

## Reproducing paper results

All experiment outputs are written to `results/figs/` and `results/tables/`. Run from the repository root.

### Full reproduction (~2 minutes)

```bash
python experiments/run_experiments.py
python experiments/run_token_panel.py
```

### Fast approximation (~20 seconds, fewer MC seeds)

```bash
python experiments/run_experiments.py --quick
```

### Run a single stage

```bash
# stage 1 → Figure 4 + Figure 5 + Figure 6 + Table 2
python experiments/run_experiments.py --stage 1

# stage 2 → Table 4 (baseline ablation)
python experiments/run_experiments.py --stage 2

# stage 3 → Table 3 (BME scenario sweep)
python experiments/run_experiments.py --stage 3

# stage 4 → Table (impact-model comparison)
python experiments/run_experiments.py --stage 4

# stage 5 → Figure 7 (sensitivity heatmap)
python experiments/run_experiments.py --stage 5

# stage 6 → Figure 8 (failure boundary)
python experiments/run_experiments.py --stage 6
```

---

## Figures and tables — code mapping

Every figure and table in the paper is generated by a specific function in `experiments/run_experiments.py`. The table below shows the exact correspondence.

| Paper item | Function | Output file |
|---|---|---|
| **Figure 4** — Price trajectory under engineered crash | `run_headline()` | `results/figs/fig4_price_trajectory.{pdf,png}` |
| **Figure 5** — Reserve balance and dynamic tax rate | `run_headline()` | `results/figs/fig5_reserve_tax.{pdf,png}` |
| **Figure 6** — Lock ratio and effective supply | `run_headline()` | `results/figs/fig6_lock_supply.{pdf,png}` |
| **Figure 7** — Sensitivity heatmap (ω_p, ω_d) | `run_sensitivity()` | `results/figs/fig7_sensitivity.{pdf,png}` |
| **Figure 8** — Failure boundary (ρ₀, liq-shock severity) | `run_failure()` | `results/figs/fig8_failure_boundary.{pdf,png}` |
| **Table 2** — Main result (no SRF vs. Full SRF) | `run_headline()` | `results/tables/table2_main_result.{csv,tex}` |
| **Table 3** — BME organic-flux scenarios | `run_bme_scenarios()` | `results/tables/table3_bme_scenarios.{csv,tex}` |
| **Table 4** — Baseline ablation | `run_ablation()` | `results/tables/table4_baseline_ablation.{csv,tex}` |
| **Table (impact models)** | `run_impact()` | `results/tables/table_impact_models.{csv,tex}` |
| **Token panel tables** | `experiments/run_token_panel.py` | `results/tables/table_token_panel.csv`, `table_token_groups.csv`, `table_unfit_tokens.csv` |

---

## Code-to-paper mapping (module level)

### `srf_lab/config.py` → Paper Table 1 (Hyperparameter table)

`Config` is a Python dataclass holding every tunable parameter used across the paper. The defaults match the values reported in Table 1.

```python
@dataclass
class Config:
    T: int = 365          # horizon (daily steps)
    P0: float = 1.0       # initial price
    S0: float = 5_000_000 # initial circulating supply
    rho_0: float = 0.10   # R_0 / MC_0 — initial reserve fraction

    # PD controller (§3.1)
    omega_p: float = 0.10  # proportional gain (small: circuit-breaker, not anchor)
    omega_d: float = 4.0   # derivative gain (strong velocity damping)
    deadband: float = 0.04 # ignore deviations < 4%
    rate_limit: float = 0.04

    # Volatility-triggered escrow lock (§3.2)
    gamma: float = 20.0         # lock sensitivity to excess vol
    ell_max: float = 0.55       # maximum lock fraction (55%)
    lock_vol_window: int = 5    # short window for realized vol
    lock_baseline_window: int = 45

    # Self-financing dynamic tax (§3.3)
    tax_max: float = 0.05
    rho_target: float = 0.15   # target reserve-to-MC ratio

    # Market impact (§3.4)
    impact_model: str = "linear"   # "linear" | "amm"
    liquidity_frac: float = 0.10
    kappa: float = 0.5
    ...
```

---

### `srf_lab/controllers.py` → Paper §3.1 (SRF control law) and §4.3 (baselines)

All controllers implement the same `.step()` interface so the simulator loop is controller-agnostic. The full hierarchy is:

```
_ControllerBase
├── NoController              — no intervention (§4.3 baseline B0)
├── PassiveThresholdController— buy/sell on fixed % threshold (§4.3 baseline B1)
├── ProportionalController    — proportional-only, ω_d = 0 (§4.3 ablation A1)
├── PDController              — PD only, no lock, no tax (§4.3 ablation A2)
├── LockOnlyController        — escrow lock only, no trading (§4.3 ablation A3)
├── SRFNoTaxController        — PD + lock + safeguards, no tax (§4.3 ablation A4)
└── FullSRFController         — all four mechanisms (§3, proposed model)
```

The **core PD trade rule** (§3.1, Eq. 2–4) lives in `_ControllerBase._execute_pd()`:

```
u_demand = R · (ω_p · e_t  −  ω_d · v_t)
```

where `e_t = (P_target − P) / P_target` is the signed deviation and `v_t = (P − P_prev) / P_prev` is the velocity. The demand is then passed through the market-impact model (see below), clipped by the rate limiter, and the reserve is updated.

The **lock ratio** (§3.2, Eq. 6) is computed by `_compute_lock()`:

```
ℓ_t = min(ℓ_max,  γ · max(0,  σ_short − σ_baseline))
```

where `σ_short` is the realized volatility of the last 5 daily returns and `σ_baseline` is the 45-day rolling baseline. The lock fires only on *excess* volatility — naturally turbulent tokens are not over-locked.

The **dynamic tax** (§3.3, Eq. 7) in `FullSRFController._tax_refill()`:

```
η_t = η_max · max(0, 1 − R_t / R_target)
```

---

### `srf_lab/simulator.py` → Paper §3 (Algorithm 1, Euler-Maruyama loop)

`simulate(cfg, controller, ...)` is the main entry point. Each daily step:

1. **Local reference price** `P_target[t]` — a short-window SMA (7-day by default) that *tracks* the trend instead of anchoring at a fixed level. This is the key design choice that allows long-run price discovery while dampening sudden moves.
2. **Excess volatility** `σ_excess` — fires the lock only when short-window vol exceeds the token's own 45-day baseline.
3. **Organic BME flux** (§3.3) — mint/burn changes supply *before* the controller acts.
4. **Controller `.step()`** — produces `(ΔP_SRF, ℓ_t, η_t, u_actual)`.
5. **Price update** (Euler-Maruyama):

```
P_{t+1} = P_t · [1  +  r_mkt[t] · (1 − ℓ_t)  +  ΔP_SRF,t]
```

The market shock `r_mkt` is *attenuated* by the lock fraction `ℓ_t` before being applied.

---

### `srf_lab/bme.py` → Paper §3.3 (BME organic dynamics)

Implements five canonical BME scenarios (paper Table in §4.1):

| Scenario key | Description | μ (daily mint) | β (daily burn) |
|---|---|---|---|
| `"off"` | Constant supply (ablation baseline) | 0 | 0 |
| `"neutral"` | Balanced mint/burn | 0.10 % | 0.10 % |
| `"emission_heavy"` | Dilutive emissions exceed burn | 0.30 % | 0.10 % |
| `"burn_heavy"` | Deflationary, utility-driven burn dominates | 0.10 % | 0.30 % |
| `"reward_inflation"` | **Pathological**: emissions inflate when price falls | adaptive | 0.10 % |
| `"demand_collapse"` | Utility demand collapses mid-run | 0.10 % | adaptive |

The net supply flux per day (§3.3, Eq. 1):

```
ΔS_org = M(t) − B(P, U)
       = μ·S  −  β·S·U(t)·(P / P_target)^elasticity
```

---

### `srf_lab/market_impact.py` → Paper §3.4 (Market-impact models)

Two models, selected via `Config.impact_model`:

**`LinearImpact`** (default):
```
ΔP/P = sign(u) · min(|u|, cap) · κ / LD_t
```
where `LD_t = liquidity_frac · MC_t` and κ = 0.5 (slippage coefficient).

**`AMMImpact`** (constant-product AMM):
```
ΔP/P = (x + u)² / x²  − 1      [for a buy of u USD, x = LD_t]
```

Both accept a `LiquidityShock` that temporarily drops `LD_t` by a configurable fraction (e.g., 60%) over a window of days — modelling LP flight during a crash.

---

### `srf_lab/stress.py` → Paper §4.1 (Stress scenarios)

Three synthetic return generators:

| Function | Scenario | Paper name |
|---|---|---|
| `gbm_returns()` | Plain Geometric Brownian Motion | GBM baseline |
| `engineered_crash()` | GBM + superimposed flash crash at day `crash_t` | **Benchmark scenario** (Figures 4–6) |
| `vol_clustering()` | Two-state regime-switching (GARCH-style) | Volatility-cluster scenario (Figures 7–8) |

The benchmark crash (§4.1): crash at day 150 over 10 days, `crash_mean = −12%/day`, `crash_sigma = 5%`.

---

### `srf_lab/metrics.py` → Paper §4.4 (Evaluation metrics)

`compute_metrics(traj, baseline_traj=None)` returns a flat dict with:

| Key | Metric | Equation |
|---|---|---|
| `ann_vol` | Annualized volatility | std(log-returns) × √365 |
| `max_drawdown` | Global peak-to-trough drawdown | max((peak − P) / peak) |
| `rolling_dd_14` | **Primary metric**: mean rolling-14-day MaxDD | mean over windows |
| `rolling_vol_14` | Mean rolling-14-day annualized vol | mean over windows |
| `worst_day_pct` | Largest single-day drop | max(|negative return|) |
| `downside_vol` | Downside deviation | std(negative log-returns) × √365 |
| `expected_shortfall_5` | CVaR at 5% tail | mean of worst 5% daily returns |
| `reserve_depletion` | 1 − R_T / R_0 | fraction of reserve consumed |
| `intervention_cost` | Σ|u_actual| (USD) | total reserve-side turnover |
| `avg_lock` / `peak_lock` | Average/peak supply lock fraction | mean/max of ℓ_t |
| `avg_tax` | Average daily tax rate | mean of η_t |
| `drift_preservation` | \|log(P_T^SRF) − log(P_T^base)\| | terminal log-price distance |
| `rolling_dd_reduction` | Relative rolling-DD improvement vs. baseline | (base − srf) / base |

The **rolling-14-day MaxDD** (`rolling_dd_14`) is the paper's *primary* metric for short-horizon protection because it captures sudden spike-to-trough moves rather than cumulative long-run decline.

---

### `srf_lab/data_loader.py` → Paper §4.2 (Token panel)

Contains the 10-token panel (Table in §4.2) and two data modes:

**`fetch_real(coin_ids, days=365)`** — pulls live OHLCV from CoinGecko v3 with exponential back-off on rate limits. Caches to CSV if `cache_dir` is set.

**`load_calibrated(panel, days=365, seed=42)`** — deterministic synthetic series calibrated to each token's published 2024 annualised volatility, drift, and volume-to-MC ratio. Used for offline / reproducible evaluation. Returns the identical DataFrame schema as `fetch_real`.

**Token panel** (5 BME/DePIN + 5 volatile DeFi):

| Ticker | Group | σ_ann | Notes |
|---|---|---|---|
| RNDR | BME/DePIN | 130% | Render Network — GPU compute |
| HNT  | BME/DePIN | 120% | Helium — wireless |
| FIL  | BME/DePIN |  95% | Filecoin — storage |
| AKT  | BME/DePIN | 140% | Akash — cloud compute |
| AR   | BME/DePIN | 110% | Arweave — permanent storage |
| AAVE | DeFi-volatile | 85% | Lending |
| COMP | DeFi-volatile | 95% | Compound |
| UNI  | DeFi-volatile | 80% | Uniswap |
| CRV  | DeFi-volatile | 105% | Curve |
| BAND | DeFi-volatile | 115% | Band Protocol oracle |

**Suitability criteria** (§4.5): a token is classified *well-fit* if:
- Drift preservation `|log(P_T^SRF) − log(P_T^base)| < 1.0` (ratio stays in [0.37, 2.71])
- Rolling-DD reduction > 15%

---

### `srf_lab/montecarlo.py` → Paper §4.1 (Monte Carlo evaluation)

`monte_carlo(cfg, ControllerCls, n_runs=50)` runs the controller on `n_runs` independently seeded copies of the configuration and returns a DataFrame of per-run metrics with baseline-relative reductions.

```python
from srf_lab import Config, monte_carlo, FullSRFController

cfg = Config(T=365, stress_kind="gbm_crash", bme_scenario="neutral")
df  = monte_carlo(cfg, FullSRFController, n_runs=30)
print(df[["ann_vol", "rolling_dd_14", "rolling_dd_reduction"]].describe())
```

---

## Configuration reference

The most commonly changed parameters (all in `Config`):

```python
cfg = Config(
    # Horizon & initial state
    T      = 365,         # simulation days
    P0     = 1.0,         # initial price (USD)
    S0     = 5_000_000,   # initial circulating supply
    rho_0  = 0.10,        # initial reserve as fraction of market cap

    # PD gains — derivative-dominated is the paper default
    omega_p = 0.10,       # proportional gain (small keeps long-run drift free)
    omega_d = 4.0,        # derivative gain   (strong velocity damping)
    deadband    = 0.04,   # ignore |e_t| < 4%
    rate_limit  = 0.04,   # max |ΔP_SRF| per day

    # Supply lock
    gamma              = 20.0,  # lock sensitivity
    ell_max            = 0.55,  # max lock fraction
    lock_vol_window    = 5,     # short vol window (days)
    lock_baseline_window = 45,  # slow baseline window (days)

    # Dynamic tax
    tax_max    = 0.05,   # η_max
    rho_target = 0.15,  # target R / MC

    # Local reference target
    target_kind = "sma_short",  # 7-day SMA (default)
    sma_window  = 7,

    # Market impact
    impact_model   = "linear",  # "linear" | "amm"
    liquidity_frac = 0.10,
    kappa          = 0.5,

    # BME scenario
    bme_scenario = "neutral",
    # options: "off" | "neutral" | "emission_heavy" | "burn_heavy"
    #          | "reward_inflation" | "demand_collapse"

    # Stress scenario
    stress_kind = "gbm_crash",
    # options: "gbm" | "gbm_crash" | "vol_cluster" | "real"
    crash_t     = 150,
    crash_len   = 10,
    crash_mean  = -0.12,
    crash_sigma = 0.05,
)
```

---

## Extending the framework

### Adding a new controller

Subclass `_ControllerBase` in `srf_lab/controllers.py` and implement `step()`:

```python
from srf_lab.controllers import _ControllerBase, StepResult

class MyController(_ControllerBase):
    name  = "my_ctrl"
    label = "My custom controller"

    def step(self, *, P, P_prev, P_target, MC, Vol, t, impact,
             ell_max, gamma_lock, deadband, rate_limit,
             tax_max, rho_target, omega_p, omega_d, **kw) -> StepResult:
        # your logic here
        delta, u_act = self._execute_pd(
            P=P, P_prev=P_prev, P_target=P_target, MC=MC, t=t,
            impact=impact, deadband=deadband, rate_limit=rate_limit,
            omega_p=omega_p, omega_d=omega_d,
        )
        return StepResult(delta_P_srf=delta, u_actual_usd=u_act, R_after=self.R)
```

### Adding a new BME scenario

Add an entry to `_SCENARIO_MAP` in `srf_lab/bme.py` and subclass `BMEScenario` with a custom `utility_demand()` method.

### Adding a new stress scenario

Add a function to `srf_lab/stress.py` returning a `np.ndarray` of length `T` daily returns, then add a branch in `simulator._build_returns()`.

---

## Compiling the paper

The LaTeX source is in `paper/`. It references figure files relative to the **repository root** (e.g., `\includegraphics{figs/srf_architecture}`), so compile from the root:

```bash
cd <repo-root>
pdflatex paper/main.tex
bibtex paper/main
pdflatex paper/main.tex
pdflatex paper/main.tex
```

Or use `latexmk`:

```bash
latexmk -pdf -outdir=paper paper/main.tex
```

**Required packages:** `svg`, `listings`, `xcolor`, `algorithm`, `algpseudocode`, `amsmath`, `natbib` (numbers style), Elsevier CAS class (`cas-dc.cls`, included).

---

## Citation

If you use this code or build on the ideas in the paper, please cite:

```bibtex
@article{nguyen2025srf,
  title   = {Stabilizing Token-Mediated Digital Marketplaces through
             Closed-Loop Reserve Funds and Dynamic Supply Locking},
  author  = {Nguyen, Huy Hai and Nguyen, Binh Minh},
  year    = {2025},
  note    = {Manuscript submitted for publication}
}
```

---

## License

The simulation code (`srf_lab/`, `experiments/`) is released under the **MIT License**.
The LaTeX source (`paper/`) is subject to the Elsevier CAS class license.
