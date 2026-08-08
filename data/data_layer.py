"""
Data layer — price history and news via yfinance (free, no key needed);
Finnhub used optionally for news if available and key is set.
Swap the provider by changing only this file.
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Finnhub is optional — only used if installed and API key is present
try:
    import finnhub as _finnhub_mod
    _finnhub_key = os.getenv("FINNHUB_API_KEY", "")
    _client = _finnhub_mod.Client(api_key=_finnhub_key) if _finnhub_key else None
except ImportError:
    _client = None


# ── Price history (yfinance — free, no rate limit issues) ─────────────────────

def get_price_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """Return OHLCV DataFrame for *ticker* covering the last *days* calendar days."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker!r}")

    # yfinance may return MultiIndex columns when downloading a single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().rename(columns={
        "Date":   "date",
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })
    df = df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
    return df


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
