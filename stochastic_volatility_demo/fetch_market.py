#!/usr/bin/env python3
"""Download listed option quotes and build a clean IV panel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CACHE_CSV = ART / "market_quotes.csv"
CACHE_META = ART / "market_meta.json"

# Tried in order; first symbol with enough live tenors wins.
DEFAULT_CANDIDATES = ("AAPL", "QQQ", "SPY", "^SPX")


def _spot_from_ticker(ticker) -> float:
    fi = getattr(ticker, "fast_info", None)
    if fi is not None:
        for key in ("lastPrice", "last_price", "regularMarketPrice"):
            try:
                v = fi[key] if hasattr(fi, "__getitem__") else getattr(fi, key, None)
            except Exception:
                v = None
            if v is not None:
                return float(v)
    info = ticker.info or {}
    for key in ("regularMarketPrice", "currentPrice", "previousClose"):
        if info.get(key) is not None:
            return float(info[key])
    raise RuntimeError("Could not resolve spot price")


def _bs_price(S, K, tau, r, sigma, call: bool) -> float:
    if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * tau) / vol
    d2 = d1 - vol
    if call:
        return float(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2))
    return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1))


def imply_bs_iv(
    price: float,
    S: float,
    K: float,
    tau: float,
    r: float = 0.0,
    call: bool = True,
) -> float:
    """Black–Scholes IV from a traded price (handles after-hours last prints)."""
    if not np.isfinite(price) or price <= 0 or tau <= 1e-6 or S <= 0 or K <= 0:
        return np.nan
    disc = np.exp(-r * tau)
    # Intrinsic / upper bounds
    if call:
        lo_p, hi_p = max(S - K * disc, 0.0), S
    else:
        lo_p, hi_p = max(K * disc - S, 0.0), K * disc
    if price < lo_p - 1e-8 or price > hi_p + 1e-4:
        return np.nan

    def objective(sig):
        return _bs_price(S, K, tau, r, sig, call=call) - price

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=80))
    except Exception:
        return np.nan


def _subsample_surface(
    df: pd.DataFrame,
    max_tenors: int = 8,
    max_strikes_per_tenor: int = 25,
) -> pd.DataFrame:
    """Thin a dense strip (e.g. SPX) so fits/plots stay readable."""
    tenors = np.array(sorted(df["tau"].unique()))
    if len(tenors) > max_tenors:
        idx = np.unique(np.linspace(0, len(tenors) - 1, max_tenors).astype(int))
        keep_t = set(tenors[idx])
        df = df[df["tau"].isin(keep_t)].copy()
    parts = []
    for _, g in df.groupby("expiry", sort=False):
        g = g.sort_values("k")
        if len(g) > max_strikes_per_tenor:
            idx = np.unique(
                np.linspace(0, len(g) - 1, max_strikes_per_tenor).astype(int)
            )
            g = g.iloc[idx]
        parts.append(g)
    return pd.concat(parts, ignore_index=True).sort_values(["tau", "k"]).reset_index(
        drop=True
    )


def fetch_yahoo_options(
    symbol: str = "AAPL",
    tau_min: float = 0.05,
    tau_max: float = 1.6,
    abs_k_max: float = 0.30,
    min_iv: float = 0.05,
    max_iv: float = 1.5,
    r: float = 0.0,
    min_quotes_per_tenor: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Pull Yahoo Finance option chains and keep an OTM IV strip.

    After hours, single-name chains often have bid=ask=0 and a junk Yahoo IV
    (~1e-5). In that case we fall back to ``lastPrice`` and invert Black–Scholes
    IV so AAPL/QQQ still yield usable surfaces.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    spot = _spot_from_ticker(ticker)
    today = datetime.now(timezone.utc).date()
    rows = []
    for expiry in ticker.options:
        d = datetime.strptime(expiry, "%Y-%m-%d").date()
        tau = (d - today).days / 365.25
        if tau < tau_min or tau > tau_max:
            continue
        chain = ticker.option_chain(expiry)
        for side, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            part = df.copy()
            part["side"] = side
            part["expiry"] = expiry
            part["tau"] = float(tau)
            rows.append(part)
    if not rows:
        raise RuntimeError(f"No option chains returned for {symbol}")

    raw = pd.concat(rows, ignore_index=True)
    raw["spot"] = spot
    bid = raw["bid"].astype(float)
    ask = raw["ask"].astype(float)
    last = raw["lastPrice"].astype(float)
    raw["mid"] = np.where((bid > 0) & (ask > 0), 0.5 * (bid + ask), np.nan)
    raw["px"] = np.where(np.isfinite(raw["mid"]) & (raw["mid"] > 0), raw["mid"], last)
    raw["k"] = np.log(raw["strike"].astype(float) / spot)
    yahoo_iv = raw["impliedVolatility"].astype(float)

    # Recover IV when Yahoo marks are missing / collapsed after the close.
    need_invert = (
        (~np.isfinite(yahoo_iv))
        | (yahoo_iv < min_iv)
        | (yahoo_iv > max_iv)
    ) & (raw["px"] > 0)
    recovered = []
    for i, row in raw.loc[need_invert].iterrows():
        recovered.append(
            (
                i,
                imply_bs_iv(
                    price=float(row["px"]),
                    S=spot,
                    K=float(row["strike"]),
                    tau=float(row["tau"]),
                    r=r,
                    call=(row["side"] == "call"),
                ),
            )
        )
    iv = yahoo_iv.copy()
    n_recovered = 0
    for i, sig in recovered:
        if np.isfinite(sig):
            iv.loc[i] = sig
            n_recovered += 1
    raw["iv"] = iv
    raw["iv_source"] = np.where(need_invert & np.isfinite(iv), "bs_invert", "yahoo")

    liq = raw[(raw["px"] > 0) & np.isfinite(raw["iv"])].copy()
    liq = liq[(liq["iv"] > min_iv) & (liq["iv"] < max_iv)]
    liq = liq[liq["k"].abs() <= abs_k_max]
    otm = liq[
        ((liq["side"] == "put") & (liq["k"] <= 0))
        | ((liq["side"] == "call") & (liq["k"] >= 0))
    ].copy()
    counts = otm.groupby("expiry").size()
    keep_exp = counts[counts >= min_quotes_per_tenor].index
    otm = otm[otm["expiry"].isin(keep_exp)].copy()
    if otm.empty:
        raise RuntimeError("No liquid OTM quotes after filters")

    # Dense index surfaces (SPX) → thin for fitting / plotting
    n_before = len(otm)
    if len(otm["tau"].unique()) > 10 or len(otm) > 400:
        otm = _subsample_surface(otm, max_tenors=8, max_strikes_per_tenor=25)

    keep = [
        "expiry",
        "tau",
        "side",
        "strike",
        "k",
        "spot",
        "bid",
        "ask",
        "mid",
        "px",
        "iv",
        "iv_source",
        "volume",
        "openInterest",
    ]
    for c in keep:
        if c not in otm.columns:
            otm[c] = np.nan
    out = otm[keep].sort_values(["tau", "k"]).reset_index(drop=True)
    meta = {
        "symbol": symbol,
        "spot": spot,
        "source": "yahoo_finance",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_quotes": int(len(out)),
        "n_quotes_before_subsample": int(n_before),
        "n_iv_recovered_from_last": int(n_recovered),
        "tenors": sorted({float(t) for t in out["tau"].unique()}),
        "tau_min": tau_min,
        "tau_max": tau_max,
    }
    return out, meta


def make_synthetic_surface(seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """
    Offline fallback: equity-like surface with strong short smile + persistent
    long skew — exactly the regime where 2-factor Bergomi helps.
    """
    rng = np.random.default_rng(seed)
    spot = 500.0
    taus = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0])
    ks = np.linspace(-0.22, 0.22, 21)
    rows = []
    for tau in taus:
        w_fast = np.exp(-10.0 * tau)
        w_slow = np.exp(-0.45 * tau)
        atm = 0.18 * (1.0 + 0.15 * w_fast + 0.08 * w_slow)
        for k in ks:
            skew = (-0.22 * w_fast - 0.18 * w_slow) * k
            smile = 0.55 * w_fast * k**2 + 0.12 * w_slow * k**2
            iv = float(np.clip(atm + skew + smile + rng.normal(0, 0.002), 0.07, 0.9))
            side = "put" if k < 0 else "call"
            strike = spot * np.exp(k)
            rows.append(
                dict(
                    expiry=f"synthetic_{tau:.2f}",
                    tau=float(tau),
                    side=side,
                    strike=float(strike),
                    k=float(k),
                    spot=spot,
                    bid=np.nan,
                    ask=np.nan,
                    mid=np.nan,
                    px=np.nan,
                    iv=iv,
                    iv_source="synthetic",
                    volume=np.nan,
                    openInterest=np.nan,
                )
            )
    df = pd.DataFrame(rows)
    meta = {
        "symbol": "SYNTHETIC",
        "spot": spot,
        "source": "synthetic_equity_like",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_quotes": int(len(df)),
        "tenors": sorted({float(t) for t in df["tau"].unique()}),
    }
    return df, meta


def choose_richest_live(
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
    min_tenors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Probe candidate underliers and return the best usable live strip.

    Score (higher is better):
      1. number of tenors
      2. share of *native* Yahoo IVs (not BS-inverted last prints)
      3. quote count before subsampling

    AAPL is first in ``DEFAULT_CANDIDATES`` so it wins ties when quality is
    comparable; ^SPX usually wins on native-IV share during the US session.
    """
    attempts = []
    best = None
    for sym in candidates:
        try:
            df, meta = fetch_yahoo_options(symbol=sym)
            n_ten = len(meta.get("tenors", []))
            n_quotes = int(meta.get("n_quotes", 0))
            n_before = int(meta.get("n_quotes_before_subsample", n_quotes))
            if "iv_source" in df.columns and len(df):
                native_frac = float((df["iv_source"] == "yahoo").mean())
            else:
                native_frac = 0.0
            # Prefer many tenors, then clean IVs, then depth
            score = (n_ten, native_frac, n_before)
            attempts.append(
                {
                    "symbol": sym,
                    "n_tenors": n_ten,
                    "n_quotes": n_quotes,
                    "n_quotes_before_subsample": n_before,
                    "n_iv_recovered_from_last": meta.get("n_iv_recovered_from_last"),
                    "native_iv_fraction": native_frac,
                }
            )
            print(
                f"[fetch] {sym}: {n_quotes} quotes, {n_ten} tenors, "
                f"native IV={native_frac:.0%}, "
                f"recovered={meta.get('n_iv_recovered_from_last', 0)}"
            )
            if n_ten >= min_tenors and (best is None or score > best[0]):
                best = (score, df, meta)
        except Exception as exc:
            attempts.append({"symbol": sym, "error": str(exc)})
            print(f"[fetch] {sym} failed: {exc}")

    if best is None:
        raise RuntimeError(f"No live surface with ≥{min_tenors} tenors: {attempts}")

    _, df, meta = best
    meta["candidate_attempts"] = attempts
    return df, meta


def load_or_fetch(
    symbol: str | None = None,
    prefer_cache: bool = False,
    min_tenors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a live Yahoo IV strip.

    - ``symbol=None`` (default): auto-pick among AAPL → QQQ → SPY → ^SPX
    - If the download fails or is too thin, fall back to synthetic.
    """
    ART.mkdir(parents=True, exist_ok=True)
    if prefer_cache and CACHE_CSV.exists():
        df = pd.read_csv(CACHE_CSV)
        meta = json.loads(CACHE_META.read_text()) if CACHE_META.exists() else {}
        return df, meta

    live_df, live_meta = None, None
    try:
        if symbol is None:
            live_df, live_meta = choose_richest_live(min_tenors=min_tenors)
        else:
            live_df, live_meta = fetch_yahoo_options(symbol=symbol)
        live_df.to_csv(ART / "market_quotes_live.csv", index=False)
        (ART / "market_meta_live.json").write_text(json.dumps(live_meta, indent=2))
    except Exception as exc:
        print(f"[fetch] live download failed ({exc})")

    if live_df is not None and len(live_meta.get("tenors", [])) >= min_tenors:
        df, meta = live_df, live_meta
    else:
        n_ten = len((live_meta or {}).get("tenors", []))
        print(
            f"[fetch] live surface has {n_ten} tenor(s); "
            f"using synthetic equity-like surface for the demo fit"
        )
        df, meta = make_synthetic_surface()
        if live_meta is not None:
            meta["live_attempt"] = live_meta
        meta["note"] = (
            "Synthetic surface used so short- and long-dated smiles are both "
            "populated; live Yahoo snapshot saved alongside when available."
        )

    df.to_csv(CACHE_CSV, index=False)
    CACHE_META.write_text(json.dumps(meta, indent=2))
    return df, meta


if __name__ == "__main__":
    quotes, meta = load_or_fetch(None)
    print(json.dumps({k: meta[k] for k in meta if k != "candidate_attempts"}, indent=2))
    print("attempts:", json.dumps(meta.get("candidate_attempts"), indent=2))
    print(quotes.head())
