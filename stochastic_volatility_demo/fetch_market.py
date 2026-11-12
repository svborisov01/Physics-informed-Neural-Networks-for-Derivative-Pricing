#!/usr/bin/env python3
"""Download listed option quotes and build a clean IV panel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
CACHE_CSV = ART / "market_quotes.csv"
CACHE_META = ART / "market_meta.json"


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


def fetch_yahoo_options(
    symbol: str = "SPY",
    tau_min: float = 0.05,
    tau_max: float = 1.6,
    abs_k_max: float = 0.30,
    min_iv: float = 0.05,
    max_iv: float = 1.5,
) -> tuple[pd.DataFrame, dict]:
    """
    Pull Yahoo Finance option chains and keep liquid OTM quotes.

    Uses put IV for k < 0 and call IV for k ≥ 0 (standard surface strip).
    Prefers monthly-ish tenors (≥7 quotes after filters) so short and long
    ends of the surface are both visible.
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
    raw["mid"] = (raw["bid"].astype(float) + raw["ask"].astype(float)) / 2.0
    raw["k"] = np.log(raw["strike"].astype(float) / spot)
    raw["iv"] = raw["impliedVolatility"].astype(float)

    # Liquidity: require a positive mid; bid/ask may be one-sided after hours
    liq = raw[(raw["mid"] > 0) | ((raw["bid"] > 0) & (raw["ask"] > 0))].copy()
    liq = liq[(liq["iv"] > min_iv) & (liq["iv"] < max_iv)]
    liq = liq[liq["k"].abs() <= abs_k_max]
    # Drop dead quotes with neither volume nor open interest when available
    if "volume" in liq.columns and "openInterest" in liq.columns:
        vol = liq["volume"].fillna(0)
        oi = liq["openInterest"].fillna(0)
        # Keep if any activity signal OR mid/IV look market-like
        liq = liq[(vol + oi > 0) | (liq["mid"] > 0)]
    otm = liq[
        ((liq["side"] == "put") & (liq["k"] <= 0))
        | ((liq["side"] == "call") & (liq["k"] >= 0))
    ].copy()
    # Keep tenors with enough points for a smile
    counts = otm.groupby("expiry").size()
    keep_exp = counts[counts >= 5].index
    otm = otm[otm["expiry"].isin(keep_exp)].copy()
    if otm.empty:
        raise RuntimeError("No liquid OTM quotes after filters")

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
        "iv",
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
                    iv=iv,
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


def load_or_fetch(
    symbol: str = "SPY",
    prefer_cache: bool = False,
    min_tenors: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Prefer a live Yahoo strip. If the download fails *or* yields fewer than
    ``min_tenors`` maturities (common after hours / for sparse chains), fall
    back to a synthetic equity-like surface that still demonstrates why two
    Bergomi timescales matter.
    """
    ART.mkdir(parents=True, exist_ok=True)
    if prefer_cache and CACHE_CSV.exists():
        df = pd.read_csv(CACHE_CSV)
        meta = json.loads(CACHE_META.read_text()) if CACHE_META.exists() else {}
        return df, meta

    live_df, live_meta = None, None
    try:
        live_df, live_meta = fetch_yahoo_options(symbol=symbol)
        # Always keep a raw live snapshot for inspection
        live_df.to_csv(ART / "market_quotes_live.csv", index=False)
        (ART / "market_meta_live.json").write_text(json.dumps(live_meta, indent=2))
    except Exception as exc:  # network / empty chain
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
    quotes, meta = load_or_fetch("SPY")
    print(json.dumps(meta, indent=2))
    print(quotes.head())
