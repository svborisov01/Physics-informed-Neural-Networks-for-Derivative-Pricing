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


def _bs_price_fwd(F, K, tau, r, sigma, call: bool) -> float:
    """Black price given forward F (discounted expectation)."""
    disc = np.exp(-r * max(tau, 0.0))
    if tau <= 0 or sigma <= 0 or F <= 0 or K <= 0:
        return float(disc * max(F - K, 0.0) if call else disc * max(K - F, 0.0))
    vol = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * vol * vol) / vol
    d2 = d1 - vol
    if call:
        return float(disc * (F * norm.cdf(d1) - K * norm.cdf(d2)))
    return float(disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1)))


def _bs_price(S, K, tau, r, q, sigma, call: bool) -> float:
    F = S * np.exp((r - q) * max(tau, 0.0))
    return _bs_price_fwd(F, K, tau, r, sigma, call=call)


def imply_bs_iv(
    price: float,
    S: float,
    K: float,
    tau: float,
    r: float = 0.04,
    q: float = 0.0,
    call: bool = True,
    F: float | None = None,
) -> float:
    """Black–Scholes IV from mid/last price (forward measure)."""
    if not np.isfinite(price) or price <= 0 or tau <= 1e-6 or K <= 0:
        return np.nan
    if F is None:
        if S <= 0:
            return np.nan
        F = S * np.exp((r - q) * tau)
    if not np.isfinite(F) or F <= 0:
        return np.nan
    disc = np.exp(-r * tau)
    if call:
        lo_p, hi_p = max(disc * (F - K), 0.0), disc * F
    else:
        lo_p, hi_p = max(disc * (K - F), 0.0), disc * K
    # Keep a tiny cushion inside arbitrage bounds
    if price <= lo_p + 1e-6 or price >= hi_p - 1e-6:
        return np.nan

    def objective(sig):
        return _bs_price_fwd(F, K, tau, r, sig, call=call) - price

    try:
        return float(brentq(objective, 1e-4, 5.0, maxiter=100))
    except Exception:
        return np.nan


def _forward_from_parity(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    tau: float,
    r: float,
    S: float,
    q_fallback: float,
) -> float:
    """
    Imply forward from near-ATM put-call parity: C − P = e^{-rτ}(F − K).
    Falls back to F = S e^{(r−q)τ} when no usable pairs exist.
    """
    F0 = S * np.exp((r - q_fallback) * tau)
    if calls.empty or puts.empty:
        return float(F0)
    # Match strikes present on both sides with usable mids
    c = calls[["strike", "px"]].rename(columns={"px": "c"})
    p = puts[["strike", "px"]].rename(columns={"px": "p"})
    both = c.merge(p, on="strike", how="inner")
    both = both[(both["c"] > 0) & (both["p"] > 0)]
    if both.empty:
        return float(F0)
    both["dist"] = (both["strike"] - F0).abs()
    both = both.sort_values("dist").head(7)
    disc = np.exp(-r * tau)
    Fs = both["strike"] + (both["c"] - both["p"]) / disc
    Fs = Fs[(Fs > 0.5 * S) & (Fs < 1.5 * S)]
    if Fs.empty:
        return float(F0)
    return float(Fs.median())


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
    symbol: str = "^SPX",
    tau_min: float = 0.05,
    tau_max: float = 1.6,
    abs_k_max: float = 0.25,
    min_iv: float = 0.05,
    max_iv: float = 1.2,
    r: float = 0.04,
    q: float | None = None,
    min_quotes_per_tenor: int = 6,
    max_rel_spread: float = 0.35,
) -> tuple[pd.DataFrame, dict]:
    """
    Pull Yahoo Finance option chains and build an OTM IV strip.

    IVs are **re-implied from mid (else last) prices** with Black–Scholes.
    Per-expiry forward F is taken from put-call parity (fallback:
    S*exp((r-q)*tau)). Yahoo's ``impliedVolatility`` is diagnostic only.
    """
    import yfinance as yf

    # Rough continuous dividend yields by underlier family
    if q is None:
        q = 0.013 if symbol in {"^SPX", "SPY", "QQQ", "IWM"} else 0.005

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
    raw["spot"] = float(spot)
    bid = raw["bid"].astype(float)
    ask = raw["ask"].astype(float)
    last = raw["lastPrice"].astype(float)
    raw["mid"] = np.where((bid > 0) & (ask > 0), 0.5 * (bid + ask), np.nan)
    raw["px"] = np.where(np.isfinite(raw["mid"]) & (raw["mid"] > 0), raw["mid"], last)
    raw["rel_spread"] = np.where(
        np.isfinite(raw["mid"]) & (raw["mid"] > 0),
        (ask - bid) / raw["mid"],
        np.nan,
    )
    raw["yahoo_iv"] = raw["impliedVolatility"].astype(float)

    # Per-expiry forward from put-call parity (fixes ATM put/call IV gap).
    F_map: dict[str, float] = {}
    for expiry, g in raw.groupby("expiry"):
        tau_e = float(g["tau"].iloc[0])
        calls = g[g["side"] == "call"]
        puts = g[g["side"] == "put"]
        F_map[str(expiry)] = _forward_from_parity(
            calls, puts, tau_e, r, float(spot), q
        )
    raw["F"] = raw["expiry"].map(lambda e: F_map[str(e)]).astype(float)
    raw["k"] = np.log(raw["strike"].astype(float) / raw["F"])
    # Effective dividend yield implied by the parity forward
    raw["q_eff"] = r - np.log(raw["F"] / raw["spot"]) / raw["tau"].astype(float)

    # Pre-filter before expensive IV inversion (OTM strip, tight spreads, px>0).
    cand = raw[(raw["px"] > 0) & (raw["k"].abs() <= abs_k_max)].copy()
    wide = np.isfinite(cand["rel_spread"]) & (cand["rel_spread"] > max_rel_spread)
    cand = cand[~wide]
    # Drop junk far OTM lasts without a mid (penny prints)
    junk = (~np.isfinite(cand["mid"])) & (cand["px"] < 0.25)
    cand = cand[~junk]
    cand = cand[
        ((cand["side"] == "put") & (cand["k"] <= 0))
        | ((cand["side"] == "call") & (cand["k"] >= 0))
    ].copy()

    # Always imply our own IV from mid (else last) using the parity forward.
    ivs = np.full(len(cand), np.nan)
    sources = np.array(["fail"] * len(cand), dtype=object)
    for j, (_, row) in enumerate(cand.iterrows()):
        px = float(row["px"])
        src = "mid" if np.isfinite(row["mid"]) and float(row["mid"]) > 0 else "last"
        sig = imply_bs_iv(
            price=px,
            S=float(row["spot"]),
            K=float(row["strike"]),
            tau=float(row["tau"]),
            r=r,
            q=float(row["q_eff"]),
            call=(row["side"] == "call"),
            F=float(row["F"]),
        )
        if np.isfinite(sig) and min_iv < sig < max_iv:
            ivs[j] = sig
            sources[j] = src
    cand["iv"] = ivs
    cand["iv_source"] = sources
    otm = cand[np.isfinite(cand["iv"])].copy()
    counts = otm.groupby("expiry").size()
    keep_exp = counts[counts >= min_quotes_per_tenor].index
    otm = otm[otm["expiry"].isin(keep_exp)].copy()
    if otm.empty:
        raise RuntimeError("No liquid OTM quotes after filters")

    n_before = len(otm)
    if len(otm["tau"].unique()) > 10 or len(otm) > 400:
        otm = _subsample_surface(otm, max_tenors=8, max_strikes_per_tenor=25)

    # Diagnostic: disagreement vs Yahoo IV on the kept strip
    both = otm[np.isfinite(otm["yahoo_iv"]) & (otm["yahoo_iv"] > min_iv)]
    yahoo_mae = (
        float(np.mean(np.abs(both["iv"] - both["yahoo_iv"]))) if len(both) else None
    )
    # ATM put/call continuity check (should be small after parity forward)
    atm_gap = []
    for _, g in otm.groupby("expiry"):
        puts = g[(g["side"] == "put") & (g["k"] > -0.03)]
        calls = g[(g["side"] == "call") & (g["k"] < 0.03)]
        if len(puts) and len(calls):
            atm_gap.append(abs(float(puts.iloc[-1]["iv"]) - float(calls.iloc[0]["iv"])))
    atm_put_call_gap = float(np.median(atm_gap)) if atm_gap else None

    keep = [
        "expiry",
        "tau",
        "side",
        "strike",
        "k",
        "F",
        "q_eff",
        "spot",
        "bid",
        "ask",
        "mid",
        "px",
        "rel_spread",
        "iv",
        "yahoo_iv",
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
        "r": r,
        "q_fallback": q,
        "forward": "put_call_parity",
        "moneyness": "log(K/F)",
        "iv_method": "bs_invert_from_mid_else_last",
        "source": "yahoo_finance",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_quotes": int(len(out)),
        "n_quotes_before_subsample": int(n_before),
        "n_iv_from_mid": int((out["iv_source"] == "mid").sum()),
        "n_iv_from_last": int((out["iv_source"] == "last").sum()),
        "yahoo_iv_mae_vs_ours": yahoo_mae,
        "atm_put_call_iv_gap_median": atm_put_call_gap,
        "forwards_by_expiry": {e: float(F_map[e]) for e in sorted(F_map)},
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
    r, q = 0.04, 0.013
    taus = np.array([0.08, 0.17, 0.25, 0.5, 0.75, 1.0])
    ks = np.linspace(-0.22, 0.22, 21)
    rows = []
    for tau in taus:
        F = spot * np.exp((r - q) * tau)
        w_fast = np.exp(-10.0 * tau)
        w_slow = np.exp(-0.45 * tau)
        atm = 0.18 * (1.0 + 0.15 * w_fast + 0.08 * w_slow)
        for k in ks:
            skew = (-0.22 * w_fast - 0.18 * w_slow) * k
            smile = 0.55 * w_fast * k**2 + 0.12 * w_slow * k**2
            iv = float(np.clip(atm + skew + smile + rng.normal(0, 0.002), 0.07, 0.9))
            side = "put" if k < 0 else "call"
            strike = F * np.exp(k)
            rows.append(
                dict(
                    expiry=f"synthetic_{tau:.2f}",
                    tau=float(tau),
                    side=side,
                    strike=float(strike),
                    k=float(k),
                    F=float(F),
                    spot=spot,
                    bid=np.nan,
                    ask=np.nan,
                    mid=np.nan,
                    px=np.nan,
                    rel_spread=np.nan,
                    iv=iv,
                    yahoo_iv=np.nan,
                    iv_source="synthetic",
                    volume=np.nan,
                    openInterest=np.nan,
                )
            )
    df = pd.DataFrame(rows)
    meta = {
        "symbol": "SYNTHETIC",
        "spot": spot,
        "r": r,
        "q": q,
        "moneyness": "log(K/F)",
        "iv_method": "synthetic",
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
      2. share of mid-implied IVs (vs last-print fallback)
      3. quote count before subsampling
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
                mid_frac = float((df["iv_source"] == "mid").mean())
            else:
                mid_frac = 0.0
            score = (n_ten, mid_frac, n_before)
            attempts.append(
                {
                    "symbol": sym,
                    "n_tenors": n_ten,
                    "n_quotes": n_quotes,
                    "n_quotes_before_subsample": n_before,
                    "n_iv_from_mid": meta.get("n_iv_from_mid"),
                    "n_iv_from_last": meta.get("n_iv_from_last"),
                    "mid_iv_fraction": mid_frac,
                    "yahoo_iv_mae_vs_ours": meta.get("yahoo_iv_mae_vs_ours"),
                }
            )
            print(
                f"[fetch] {sym}: {n_quotes} quotes, {n_ten} tenors, "
                f"mid IV={mid_frac:.0%}, "
                f"Yahoo MAE={meta.get('yahoo_iv_mae_vs_ours')}"
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
