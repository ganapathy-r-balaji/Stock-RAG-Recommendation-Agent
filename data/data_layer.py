"""
Data layer — price history and news.
Price: Alpha Vantage (primary, works from cloud IPs) → yfinance (local fallback).
News:  Finnhub (optional) → yfinance.
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

_AV_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")

# Finnhub is optional — only used if installed and API key is present
try:
    import finnhub as _finnhub_mod
    _finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    _client = _finnhub_mod.Client(api_key=_finnhub_key) if _finnhub_key else None
except ImportError:
    _client = None


# ── Price history ──────────────────────────────────────────────────────────────

def _av_price_history(ticker: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV from Alpha Vantage (works from any cloud IP)."""
    import requests
    outputsize = "compact" if days <= 100 else "full"
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={ticker}&outputsize={outputsize}&apikey={_AV_KEY}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise ValueError(data.get("Note") or data.get("Information") or "No data from Alpha Vantage")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for date_str, vals in ts.items():
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if dt < cutoff:
            continue
        rows.append({
            "date":   dt,
            "open":   float(vals["1. open"]),
            "high":   float(vals["2. high"]),
            "low":    float(vals["3. low"]),
            "close":  float(vals["5. adjusted close"]),
            "volume": float(vals["6. volume"]),
        })

    if not rows:
        raise ValueError(f"No Alpha Vantage data for {ticker!r}")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _yf_price_history(ticker: str, days: int) -> pd.DataFrame:
    """Fetch OHLCV from yfinance (fallback — may be blocked on cloud IPs)."""
    period_map = {30: "1mo", 60: "3mo", 90: "3mo", 180: "6mo", 365: "1y"}
    period = period_map.get(days) or ("1y" if days >= 365 else "3mo")
    raw = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().rename(columns={
        "Date": "date", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def get_price_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """Return OHLCV DataFrame for *ticker* covering the last *days* calendar days.
    Uses Alpha Vantage when key is set (cloud-safe), falls back to yfinance.
    """
    if _AV_KEY:
        try:
            return _av_price_history(ticker, days)
        except Exception:
            pass  # fall through to yfinance
    return _yf_price_history(ticker, days)


# ── News ───────────────────────────────────────────────────────────────────────

def get_news(ticker: str, days: int = 7) -> list[dict]:
    """Return a list of recent news articles for *ticker* (last *days* days).

    Uses Finnhub if available, otherwise falls back to yfinance news.
    """
    # Try Finnhub first (richer summaries)
    if _client is not None:
        try:
            end_dt   = datetime.now(timezone.utc)
            start_dt = end_dt - timedelta(days=days)
            articles = _client.company_news(ticker, _fmt(start_dt), _fmt(end_dt))
            if articles:
                return [
                    {
                        "headline": a.get("headline", ""),
                        "summary":  a.get("summary", ""),
                        "url":      a.get("url", ""),
                        "datetime": datetime.fromtimestamp(a["datetime"], tz=timezone.utc).isoformat(),
                        "source":   a.get("source", ""),
                    }
                    for a in articles
                ]
        except Exception:
            pass

    # Fallback: yfinance news
    info = yf.Ticker(ticker).news or []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    results = []
    for item in info:
        # yfinance news item structure varies; handle both old and new formats
        content = item.get("content", item)
        ts = content.get("pubDate") or item.get("providerPublishTime")
        if ts:
            try:
                if isinstance(ts, (int, float)):
                    pub_dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                else:
                    pub_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                if pub_dt < cutoff:
                    continue
                dt_str = pub_dt.isoformat()
            except Exception:
                dt_str = str(ts)
        else:
            dt_str = ""

        results.append({
            "headline": content.get("title", item.get("title", "")),
            "summary":  content.get("summary", ""),
            "url":      (content.get("canonicalUrl") or {}).get("url", content.get("url", item.get("link", ""))),
            "datetime": dt_str,
            "source":   (content.get("provider") or {}).get("displayName", ""),
        })
    return results


# ── Company profile ────────────────────────────────────────────────────────────

def get_company_profile(ticker: str) -> dict:
    """Return basic company metadata (name, sector, industry, peers).

    Uses Finnhub if available, otherwise falls back to yfinance info.
    """
    if _client is not None:
        try:
            profile = _client.company_profile2(symbol=ticker)
            peers   = _client.company_peers(ticker)
            if profile:
                return {
                    "ticker":  ticker,
                    "name":    profile.get("name", ticker),
                    "sector":  profile.get("finnhubIndustry", "Unknown"),
                    "country": profile.get("country", ""),
                    "peers":   peers or [],
                }
        except Exception:
            pass

    # Fallback: yfinance
    info = yf.Ticker(ticker).info or {}
    return {
        "ticker":  ticker,
        "name":    info.get("longName", ticker),
        "sector":  info.get("sector", "Unknown"),
        "country": info.get("country", ""),
        "peers":   [],   # yfinance doesn't provide peers
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")
