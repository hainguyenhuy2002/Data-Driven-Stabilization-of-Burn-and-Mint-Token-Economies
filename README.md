# Closed-Loop Reserve Fund and Dynamic Supply Locking for Token-Mediated Marketplaces

**Authors:** Huy Hai Nguyen (Tsinghua University) · Binh Minh Nguyen (Hanoi University of Science and Technology)

[![GitHub](https://img.shields.io/badge/GitHub-hainguyenhuy2002-181717?logo=github)](https://github.com/hainguyenhuy2002/Data-Driven-Stabilization-of-Burn-and-Mint-Token-Economies)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)


---

## What this is

Token-mediated digital marketplaces — storage, compute, data, oracle, liquidity — use a native token as payment, incentive, and coordination medium. Many of them follow a **Burn-and-Mint Equilibrium (BME)** design: tokens are burned when users consume services and minted to reward suppliers and validators. This keeps supply loosely tied to activity, but it creates a feedback problem: when the token price falls sharply, emissions continue or even accelerate, demand burn drops, and liquidity thins — all at once. The result is a self-reinforcing volatility spiral.

This repository implements a **Stabilization Reserve Fund (SRF)** — a closed-loop mechanism that sits on top of a BME token economy and dampens short-horizon price shocks. The SRF is not an algorithmic peg and does not try to fix the token price at a predetermined level. Instead it acts as an adaptive circuit breaker: it observes recent market conditions and decides whether to intervene, how hard, and at what cost to the protocol.

---

## The four mechanisms

### 1. Reserve-funded PD feedback

The protocol holds a treasury reserve `R` (in USD or a stable asset). When the token price moves sharply relative to its recent trend, the reserve buys or sells tokens in the open market to dampen the move.

The intervention size is determined by a **proportional-derivative (PD) rule**:

```
u_demand = R · (ω_p · e_t  −  ω_d · v_t)
```

- `e_t = (P_target − P) / P_target` — how far the current price is from a short-window local reference
- `v_t = (P − P_prev) / P_prev` — price velocity (the derivative term)
- `ω_p` is kept intentionally small (0.10) so the SRF does not anchor the price
- `ω_d` is large (4.0) so the mechanism fires primarily on sudden *moves*, not on sustained drift

The local reference `P_target` is a 7-day simple moving average of recent prices. Because it tracks the trend, the SRF dampens spikes without fighting long-run price discovery.

Two **safeguards** keep the feedback from becoming noisy itself:
- **Deadband** — interventions are skipped when `|e_t| < 4%` to avoid acting on noise
- **Rate limiter** — the SRF-induced price change is capped at ±4% per day

### 2. Volatility-triggered supply lock

When short-window realized volatility spikes above the token's own slow baseline, a fraction `ℓ_t` of the eligible supply (treasury holdings, opted-in staking positions, enrolled LP positions) is temporarily locked from trading:

```
ℓ_t = min(ℓ_max,  γ · max(0,  σ_short − σ_baseline))
```

- `σ_short` — realized vol of the last 5 daily returns
- `σ_baseline` — 45-day rolling baseline
- The lock fires only on *excess* volatility above the token's own norm, so naturally turbulent tokens are not over-locked
- `ℓ_max = 0.55` caps the lock at 55% of eligible supply

The locked supply is not confiscated or burned. It re-enters circulation once volatility subsides. The effect in the price equation is that the market return is attenuated:

```
P_{t+1} = P_t · [1  +  r_mkt[t] · (1 − ℓ_t)  +  ΔP_SRF,t]
```

### 3. Self-financing dynamic tax

Reserve trading depletes `R` over time. To keep the mechanism self-sustaining, a small transaction tax `η_t` is levied on marketplace volume and routed back into the reserve:

```
η_t = η_max · max(0,  1 − R_t / R_target)
```

The tax is zero when the reserve is healthy (`R_t ≥ R_target`), and rises toward `η_max = 5%` only when the reserve is depleted. This makes the cost of stabilization visible and proportional to how much the reserve has been used.

### 4. Organic BME flux

Independent of the SRF, the token's supply evolves each day through its native mint/burn schedule:

```
ΔS_org = M(t) − B(P, U)
       = μ·S  −  β·S·U(t)·(P / P_target)^elasticity
```

where `M(t)` is the emission schedule and `B(P, U)` is utility-driven burn that depends on price and demand `U(t)`. The SRF controller acts *after* this organic flux has been applied to supply, so the two mechanisms interact realistically.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
│
├── srf_lab/                    ← core simulation library
│   ├── __init__.py             ← public API
│   ├── config.py               ← all tunable parameters in one dataclass
│   ├── simulator.py            ← daily Euler-Maruyama loop
│   ├── controllers.py          ← SRF controller + comparison baselines
│   ├── bme.py                  ← BME organic mint/burn dynamics
│   ├── market_impact.py        ← linear and AMM price-impact models
│   ├── stress.py               ← synthetic return generators (GBM, crash, vol-cluster)
│   ├── metrics.py              ← evaluation metrics
│   ├── montecarlo.py           ← multi-seed Monte Carlo wrapper
│   └── data_loader.py          ← 10-token panel + CoinGecko live fetch
│
├── experiments/
│   ├── run_experiments.py      ← crash simulation, ablation, BME sweep, sensitivity, failure boundary
│   ├── run_token_panel.py      ← per-token backtest across 10 real tokens
│   └── srf_experiments.ipynb  ← interactive walkthrough notebook
│
├── results/
│   ├── figs/                   ← generated figures (PDF + PNG)
│   └── tables/                 ← generated tables (CSV + LaTeX)
│
├── paper/                      ← LaTeX manuscript source
│   ├── main.tex
│   ├── cas-refs.bib
│   └── cas-dc.cls  cas-sc.cls  cas-common.sty  cas-model2-names.bst
│
└── figs/                       ← architecture diagrams
    ├── srf_architecture.{pdf,png,svg}
    ├── srf_pipeline.{pdf,drawio}
    ├── srf_control_loop.pdf
    └── srf_integration_loop.pdf
```

---

## Installation

```bash
git clone https://github.com/hainguyenhuy2002/Data-Driven-Stabilization-of-Burn-and-Mint-Token-Economies.git
cd Data-Driven-Stabilization-of-Burn-and-Mint-Token-Economies

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.9, NumPy, Pandas, Matplotlib, Requests (live data only), Jupyter (notebook only).

---

## Quick start

```python
from srf_lab import Config, simulate, compute_metrics
from srf_lab import NoController, FullSRFController

# configure a 365-day simulation: GBM with an engineered flash crash at day 150
cfg = Config(T=365, seed=42, stress_kind="gbm_crash", bme_scenario="neutral")
R0  = cfg.rho_0 * cfg.P0 * cfg.S0   # initial reserve = 10% of market cap

# run with no intervention and with the full SRF
traj_base = simulate(cfg, NoController(R0=R0))
traj_srf  = simulate(cfg, FullSRFController(R0=R0))

# compare
m = compute_metrics(traj_srf, baseline_traj=traj_base)
print(f"Rolling-14 MaxDD reduction : {m['rolling_dd_reduction']*100:+.1f}%")
print(f"Worst one-day drop reduction: {m['worst_day_reduction']*100:+.1f}%")
print(f"Reserve depletion           : {m['reserve_depletion']*100:.1f}%")
```

---

## Running the experiments

All outputs go to `results/figs/` and `results/tables/`. Run from the repository root.

```bash
# full run — all experiments (~2 min)
python experiments/run_experiments.py

# quick mode — fewer Monte Carlo seeds (~20 sec)
python experiments/run_experiments.py --quick

# individual stages
python experiments/run_experiments.py --stage 1   # crash simulation + headline figures
python experiments/run_experiments.py --stage 2   # baseline ablation table
python experiments/run_experiments.py --stage 3   # BME scenario sweep
python experiments/run_experiments.py --stage 4   # impact-model comparison
python experiments/run_experiments.py --stage 5   # sensitivity heatmap over (ω_p, ω_d)
python experiments/run_experiments.py --stage 6   # failure boundary under liquidity shock

# token panel backtest across 10 real tokens
python experiments/run_token_panel.py
```

---

## Module guide

### `srf_lab/config.py` — simulation parameters

A single `Config` dataclass holds every tunable knob. Pass it to `simulate()` or `monte_carlo()`. Override only what you need:

```python
from srf_lab import Config

cfg = Config(
    T      = 365,          # simulation horizon (days)
    P0     = 1.0,          # initial token price (USD)
    S0     = 5_000_000,    # initial circulating supply
    rho_0  = 0.10,         # initial reserve as a fraction of market cap

    # PD controller
    omega_p    = 0.10,     # proportional gain — keep small so price stays free to drift
    omega_d    = 4.0,      # derivative gain — large for strong velocity damping
    deadband   = 0.04,     # ignore deviations smaller than 4%
    rate_limit = 0.04,     # maximum SRF-induced price move per day

    # supply lock
    gamma                = 20.0,   # how aggressively excess vol triggers the lock
    ell_max              = 0.55,   # maximum lock fraction (55% of eligible supply)
    lock_vol_window      = 5,      # short window for realized vol (days)
    lock_baseline_window = 45,     # slow baseline window (days)

    # dynamic tax
    tax_max    = 0.05,     # maximum daily tax rate (5%)
    rho_target = 0.15,     # target reserve-to-market-cap ratio

    # local reference target type
    target_kind = "sma_short",  # 7-day SMA (tracks trend, doesn't anchor price)
    sma_window  = 7,

    # market impact model
    impact_model   = "linear",  # "linear" | "amm"
    liquidity_frac = 0.10,      # on-chain liquidity as fraction of market cap
    kappa          = 0.5,       # slippage coefficient

    # BME organic dynamics
    bme_scenario = "neutral",
    # choices: "off" | "neutral" | "emission_heavy" | "burn_heavy"
    #          | "reward_inflation" | "demand_collapse"

    # stress scenario for the exogenous return series
    stress_kind = "gbm_crash",
    # choices: "gbm" | "gbm_crash" | "vol_cluster" | "real"
    crash_t     = 150,      # crash starts at day 150
    crash_len   = 10,       # lasts 10 days
    crash_mean  = -0.12,    # −12%/day during the crash
    crash_sigma = 0.05,
)
```

---

### `srf_lab/controllers.py` — SRF and comparison baselines

All controllers share the same `.step()` interface so they drop into the simulator without any other changes.

```
_ControllerBase
├── NoController                — no intervention at all
├── PassiveThresholdController  — buy/sell a fixed chunk whenever price crosses a threshold
├── ProportionalController      — P-only (ω_d = 0), no lock, no tax
├── PDController                — full PD trading, no lock, no tax
├── LockOnlyController          — only the supply lock, no trading
├── SRFNoTaxController          — PD + lock + safeguards, without the dynamic tax
└── FullSRFController           — all four mechanisms (the proposed design)
```

Running a comparison is straightforward:

```python
from srf_lab import Config, simulate, compute_metrics
from srf_lab import NoController, LockOnlyController, FullSRFController

cfg = Config()
R0  = cfg.rho_0 * cfg.P0 * cfg.S0

for cls in [NoController, LockOnlyController, FullSRFController]:
    traj = simulate(cfg, cls(R0=R0))
    m    = compute_metrics(traj)
    print(f"{cls.__name__:30s}  MaxDD={m['max_drawdown']*100:.1f}%  "
          f"rolling-DD={m['rolling_dd_14']*100:.1f}%")
```

---

### `srf_lab/simulator.py` — daily simulation loop

`simulate(cfg, controller)` steps through `T` days. Each day:

1. Compute the local reference price `P_target` (7-day SMA by default).
2. Compute excess realized volatility `σ_excess = max(0, σ_5day − σ_45day)`.
3. Apply organic BME mint/burn flux to supply.
4. Call `controller.step()` → returns `(ΔP_SRF, ℓ_t, η_t, u_actual)`.
5. Update price: `P_{t+1} = P_t · [1 + r_mkt[t] · (1 − ℓ_t) + ΔP_SRF,t]`.

Returns a `TrajectoryResult` with full daily arrays for price, supply, reserve, lock ratio, tax rate, and trade sizes.

The exogenous return series `r_mkt` can be synthetic (GBM, crash, vol-cluster) or real historical returns passed in by the caller. Oracle delay (lag between true price and observed price) is also supported.

---

### `srf_lab/bme.py` — BME organic dynamics

Six built-in supply scenarios cover the range from stable to pathological:

| Key | Description |
|---|---|
| `"off"` | Constant supply — organic BME disabled |
| `"neutral"` | Balanced mint and burn, supply roughly stable |
| `"emission_heavy"` | Dilutive emissions outpace utility burn (typical early-phase DePIN) |
| `"burn_heavy"` | Deflationary: utility-driven burn dominates emissions |
| `"reward_inflation"` | **Pathological** — emissions ramp up when price falls, amplifying spirals |
| `"demand_collapse"` | Utility demand evaporates midway through the run, burn disappears |

Each scenario is a subclass of `BMEScenario` with a `utility_demand(P, P_target, t, T)` method. You can add custom scenarios the same way.

---

### `srf_lab/market_impact.py` — how SRF trades move the price

Two price-impact models are available:

**Linear** (default) — slippage proportional to trade size relative to on-chain liquidity:
```
ΔP/P = sign(u) · min(|u|, cap) · κ / LD_t
```

**AMM** (constant-product) — closed-form slippage from a `xy = k` pool:
```
ΔP/P = (LD_t + u)² / LD_t²  − 1
```

Both support a `LiquidityShock` — a temporary drop in on-chain liquidity depth (e.g., LPs withdrawing during a crash). This lets you stress-test the SRF under degraded market conditions.

---

### `srf_lab/stress.py` — synthetic return generators

| Function | What it produces |
|---|---|
| `gbm_returns()` | Plain GBM — drift + noise |
| `engineered_crash()` | GBM with a superimposed flash crash at a chosen day |
| `vol_clustering()` | Two-state regime-switching: alternating low- and high-vol windows |

Pass real historical returns directly to `simulate(..., r_mkt=array)` to run backtests on actual token data.

---

### `srf_lab/metrics.py` — evaluation

`compute_metrics(traj, baseline_traj=None)` returns a flat dictionary. The most important metrics:

| Key | What it measures |
|---|---|
| `rolling_dd_14` | Mean of rolling 14-day peak-to-trough drawdown — the primary short-horizon metric |
| `rolling_vol_14` | Mean of rolling 14-day annualized volatility |
| `worst_day_pct` | Largest single-day price drop |
| `max_drawdown` | Global peak-to-trough over the full run |
| `ann_vol` | Annualized volatility of daily log-returns |
| `expected_shortfall_5` | CVaR at the 5% tail |
| `reserve_depletion` | Fraction of the initial reserve consumed (`1 − R_T / R_0`) |
| `intervention_cost` | Total absolute USD traded by the SRF |
| `avg_lock` / `peak_lock` | Average and peak supply lock fraction |
| `avg_tax` | Average daily tax rate |
| `drift_preservation` | `|log(P_T^SRF) − log(P_T^base)|` — how far the regulated run drifts from the unregulated one |
| `rolling_dd_reduction` | Relative rolling-DD improvement versus the baseline run |

When `baseline_traj` is supplied (a no-intervention run on the same seed), the reduction metrics are computed automatically.

---

### `srf_lab/data_loader.py` — token data

Two modes with an identical output schema:

**Live fetch** — pulls daily price, market cap, and volume from CoinGecko v3 with exponential back-off and optional local CSV cache:
```python
from srf_lab.data_loader import fetch_real
data = fetch_real(["render-token", "helium"], days=365, cache_dir="data/cache")
```

**Calibrated synthetic** — seed-deterministic series whose annualized volatility and drift match each token's published 2024 profile. Reproduces experiments offline without any network access:
```python
from srf_lab.data_loader import load_calibrated, PANEL
data = load_calibrated(PANEL, seed=42)
```

The built-in 10-token panel covers two groups:

| Ticker | Type | Ann. vol |
|---|---|---|
| RNDR | BME/DePIN (GPU compute) | 130% |
| HNT  | BME/DePIN (wireless) | 120% |
| FIL  | BME/DePIN (storage) | 95% |
| AKT  | BME/DePIN (cloud compute) | 140% |
| AR   | BME/DePIN (permanent storage) | 110% |
| AAVE | DeFi-volatile (lending) | 85% |
| COMP | DeFi-volatile (lending) | 95% |
| UNI  | DeFi-volatile (DEX) | 80% |
| CRV  | DeFi-volatile (DEX) | 105% |
| BAND | DeFi-volatile (oracle) | 115% |

---

### `srf_lab/montecarlo.py` — multi-seed evaluation

`monte_carlo(cfg, ControllerCls, n_runs=50)` runs the same configuration across `n_runs` independent random seeds and returns a DataFrame of per-run metrics — useful for checking robustness rather than cherry-picking a single seed.

```python
from srf_lab import Config, monte_carlo, FullSRFController

cfg = Config(T=365, stress_kind="gbm_crash", bme_scenario="neutral")
df  = monte_carlo(cfg, FullSRFController, n_runs=30)
print(df[["rolling_dd_14", "rolling_dd_reduction", "reserve_depletion"]].describe())
```

---

## Extending the framework

### New controller

Subclass `_ControllerBase` in `srf_lab/controllers.py` and implement `step()`. The helper `_execute_pd()` handles PD demand calculation and reserve accounting for you:

```python
from srf_lab.controllers import _ControllerBase, StepResult

class MyController(_ControllerBase):
    name  = "my_ctrl"
    label = "My custom controller"

    def step(self, *, P, P_prev, P_target, MC, Vol, t, impact,
             ell_max, gamma_lock, deadband, rate_limit,
             tax_max, rho_target, omega_p, omega_d, **kw) -> StepResult:
        delta, u_act = self._execute_pd(
            P=P, P_prev=P_prev, P_target=P_target, MC=MC, t=t,
            impact=impact, deadband=deadband, rate_limit=rate_limit,
            omega_p=omega_p, omega_d=omega_d,
        )
        return StepResult(delta_P_srf=delta, u_actual_usd=u_act, R_after=self.R)
```

### New BME scenario

Add an entry to `_SCENARIO_MAP` in `srf_lab/bme.py` and subclass `BMEScenario` with a `utility_demand()` method.

### New stress scenario

Add a function to `srf_lab/stress.py` that returns a `np.ndarray` of length `T` daily returns, then add a branch in `simulator._build_returns()`.

---

## Compiling the LaTeX manuscript

The source is in `paper/`. Figure files are referenced relative to the repository root, so compile from there:

```bash
cd <repo-root>
pdflatex paper/main.tex
bibtex paper/main
pdflatex paper/main.tex
pdflatex paper/main.tex
# or simply:
latexmk -pdf -outdir=paper paper/main.tex
```

---

## Citation

```bibtex
@article{nguyen2025srf,
  title   = {Stabilizing Token-Mediated Digital Marketplaces through
             Closed-Loop Reserve Funds and Dynamic Supply Locking},
  author  = {Nguyen, Huy Hai and Nguyen, Binh Minh},
  year    = {2025},
  note    = {Manuscript submitted for publication}
}
```

For the code itself, please also cite the repository:

```bibtex
@software{nguyen2025srf_code,
  author    = {Nguyen, Huy Hai and Nguyen, Binh Minh},
  title     = {Closed-Loop Reserve Fund and Dynamic Supply Locking
               for Token-Mediated Marketplaces — Simulation Code},
  year      = {2025},
  url       = {https://github.com/hainguyenhuy2002/Data-Driven-Stabilization-of-Burn-and-Mint-Token-Economies},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

> Replace `10.5281/zenodo.XXXXXXX` with your actual Zenodo DOI before publishing.

---

## License

Simulation code (`srf_lab/`, `experiments/`) — MIT License.
LaTeX source (`paper/`) — subject to the Elsevier CAS class license.
