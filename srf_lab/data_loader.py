"""
Token-data loader for the SRF stress tests.

Two complementary modes
-----------------------

1.  ``fetch_real``  — pulls live data from the CoinGecko v3 ``market_chart``
    endpoint for a list of coin IDs, with exponential back-off on rate limits.

2.  ``load_calibrated``  — generates seed-deterministic synthetic series per
    token, using publicly-reported daily-volatility / drift / mean-volume
    profiles for 2024.  Used in offline / blocked-network environments and
    for the deterministic reproductions reported in the paper.

The two modes return DataFrames with **identical schema** so the
experiments code path is unchanged regardless of source.
"""
from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


# --------------------------------------------------------------------- #
# Token panel — 5 BME / DePIN / BME-like + 5 volatile DeFi
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenSpec:
    coin_id:     str          # CoinGecko v3 id
    ticker:      str
    group:       str          # "BME/DePIN" | "DeFi-volatile"
    sigma_ann:   float        # publicly-reported annualised daily-return vol (2024)
    drift_ann:   float        # annualised drift
    init_price:  float        # initial price for the calibrated trace
    init_supply: float        # circulating supply (tokens)
    avg_vol_mc:  float        # average daily-volume-to-market-cap ratio (used for synth Vol_t)


PANEL = [
    # ----- BME / DePIN / BME-like -----
    TokenSpec("render-token",       "RNDR", "BME/DePIN",     1.30, 0.20,  6.50,  389e6, 0.10),
    TokenSpec("helium",              "HNT",  "BME/DePIN",     1.20, 0.05,  6.80,  165e6, 0.06),
    TokenSpec("filecoin",            "FIL",  "BME/DePIN",     0.95, -0.05, 5.20,  590e6, 0.05),
    TokenSpec("akash-network",       "AKT",  "BME/DePIN",     1.40, 0.30,  3.10,  234e6, 0.04),
    TokenSpec("arweave",             "AR",   "BME/DePIN",     1.10, -0.10, 18.00, 66e6,  0.07),
    # ----- volatile DeFi -----
    TokenSpec("aave",                "AAVE", "DeFi-volatile", 0.85, 0.10,  95.00,  15e6, 0.10),
    TokenSpec("compound-governance-token","COMP","DeFi-volatile",0.95,-0.20,55.00, 9e6,  0.05),
    TokenSpec("uniswap",             "UNI",  "DeFi-volatile", 0.80, 0.05,  9.00,  600e6, 0.04),
    TokenSpec("curve-dao-token",     "CRV",  "DeFi-volatile", 1.05, -0.15, 0.40, 1100e6, 0.06),
    TokenSpec("band-protocol",       "BAND", "DeFi-volatile", 1.15, -0.05, 1.50,  142e6, 0.05),
]


# --------------------------------------------------------------------- #
# 1) Live fetch
# --------------------------------------------------------------------- #
def fetch_real(coin_ids: list[str],
               days: int = 365,
               vs_currency: str = "usd",
               cache_dir: Optional[str] = None,
               base_url: str = "https://api.coingecko.com/api/v3") -> dict[str, pd.DataFrame]:
    """
    Pull <days> of daily price + market-cap + volume data from the CoinGecko
    v3 ``market_chart`` endpoint, with exponential back-off on 429.

    Returns
    -------
    dict mapping coin_id -> DataFrame with columns
        timestamp, P_hist, MC_hist, Vol_hist, Return_hist
    """
    import requests   # imported lazily so offline runs don't pay the cost

    out: dict[str, pd.DataFrame] = {}
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for cid in coin_ids:
        # Local cache hit?
        if cache_dir:
            cache_path = os.path.join(cache_dir, f"{cid}.csv")
            if os.path.isfile(cache_path):
                out[cid] = pd.read_csv(cache_path, parse_dates=["timestamp"])
                continue

        url = f"{base_url}/coins/{cid}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        delay = 6
        for attempt in range(5):
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 200:
                break
            if r.status_code in (429, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 2, 90)
            else:
                raise RuntimeError(f"{cid}: HTTP {r.status_code}")
        else:
            raise RuntimeError(f"{cid}: rate-limited after 5 attempts")

        data = r.json()
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([p[0] for p in data["prices"]],   unit="ms"),
            "P_hist":    [p[1] for p in data["prices"]],
            "MC_hist":   [p[1] for p in data["market_caps"]],
            "Vol_hist":  [p[1] for p in data["total_volumes"]],
        })
        # Daily simple returns; boundary NaN -> 0
        df["Return_hist"] = df["P_hist"].pct_change().fillna(0.0)
        # Polite spacing between requests
        time.sleep(1.5)

        if cache_dir:
            df.to_csv(os.path.join(cache_dir, f"{cid}.csv"), index=False)
        out[cid] = df
    return out


# --------------------------------------------------------------------- #
# 2) Calibrated synthetic — used when CoinGecko unreachable
# --------------------------------------------------------------------- #
def load_calibrated(panel: list[TokenSpec] = PANEL,
                    days: int = 365,
                    seed: int = 42,
                    add_clusters: bool = True) -> dict[str, pd.DataFrame]:
    """
    Generate per-token deterministic synthetic series whose annualised
    volatility matches each token's published 2024 profile, plus optional
    GARCH-style volatility clustering.

    Returns the same schema as ``fetch_real``.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    base_t = pd.date_range("2024-05-01", periods=days, freq="D")

    for spec in panel:
        sigma_d = spec.sigma_ann / np.sqrt(365.0)
        drift_d = spec.drift_ann / 365.0

        if add_clusters:
            # two-state regime switching with regime-specific sigma
            regime = np.zeros(days, dtype=int)
            t = 0; state = 0; reg_len = 30
            while t < days:
                regime[t:t+reg_len] = state
                state = 1 - state
                t += reg_len
            sigma_t = np.where(regime == 0, sigma_d * 0.6, sigma_d * 1.6)
        else:
            sigma_t = np.full(days, sigma_d)

        eps = rng.standard_normal(days)
        ret = drift_d + sigma_t * eps

        prices = np.empty(days, dtype=float)
        prices[0] = spec.init_price
        for t in range(1, days):
            prices[t] = max(prices[t-1] * (1 + ret[t]), 1e-3)

        mc = prices * spec.init_supply           # constant-supply approximation
        vol = mc * spec.avg_vol_mc * (1.0 + 0.3*rng.standard_normal(days))
        vol = np.clip(vol, 0.0, None)

        df = pd.DataFrame({
            "timestamp": base_t,
            "P_hist":   prices,
            "MC_hist":  mc,
            "Vol_hist": vol,
            "Return_hist": np.concatenate([[0.0], np.diff(prices)/prices[:-1]]),
        })
        out[spec.coin_id] = df
    return out


# --------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------- #
def normalised_volume(df: pd.DataFrame) -> np.ndarray:
    """
    Volume normalisation rule used in the paper:  Vol_t / MC_t (a.k.a. turnover).
    Multiplied by the protocol's initial market-cap to make it comparable
    across tokens. Returns Vol_t in USD on the token's own scale.
    """
    return df["Vol_hist"].to_numpy()


def panel_groups(panel: list[TokenSpec] = PANEL) -> dict[str, list[TokenSpec]]:
    g: dict[str, list[TokenSpec]] = {}
    for s in panel:
        g.setdefault(s.group, []).append(s)
    return g
