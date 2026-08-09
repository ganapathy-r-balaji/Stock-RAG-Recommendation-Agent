"""
Agent tools — three tools exposed to the LangGraph ReAct agent.
"""

import json

from langchain_core.tools import tool

from data.data_layer import get_price_history
from forecast.forecaster import forecast_price
from retrieval.vector_store import index_news, retrieve_news


@tool
def tool_forecast_price(ticker: str) -> str:
    """
    Return a 3-day-ahead price forecast for *ticker*.
    Output includes current price, predicted price, % change, direction, and confidence.
    Always call this when the user asks whether to buy, hold, or avoid a stock.
    """
    try:
        result = forecast_price(ticker.upper())
        return json.dumps(result.__dict__)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def tool_retrieve_news(ticker: str, query: str) -> str:
    """
    Retrieve the most relevant recent news snippets for *ticker* given *query*.
    Indexes fresh news on first call. Returns up to 5 results with source URLs.
    Always call this to ground your answer in recent news.
    """
    try:
        ticker = ticker.upper()
        index_news(ticker)
        snippets = retrieve_news(ticker, query)
        return json.dumps(snippets)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def tool_price_history(ticker: str, days: int = 60) -> str:
    """
    Return historical price performance for *ticker* over the last *days* calendar days.
    Includes start price, end price, % change, high, and low.
    Call this whenever the user asks about past performance or how much a stock has moved.
    """
    try:
        df = get_price_history(ticker.upper(), days=days)
        if df.empty:
            return json.dumps({"error": f"No price data available for {ticker}"})

        start_price = round(float(df["close"].iloc[0]), 2)
        end_price   = round(float(df["close"].iloc[-1]), 2)
        pct_change  = round((end_price - start_price) / start_price * 100, 2)

        return json.dumps({
            "ticker":      ticker.upper(),
            "days":        days,
            "start_date":  str(df["date"].iloc[0])[:10],
            "end_date":    str(df["date"].iloc[-1])[:10],
            "start_price": start_price,
            "end_price":   end_price,
            "abs_change":  round(end_price - start_price, 2),
            "pct_change":  pct_change,
            "period_high": round(float(df["high"].max()), 2),
            "period_low":  round(float(df["low"].min()), 2),
            "direction":   "up" if pct_change > 0 else "down" if pct_change < 0 else "flat",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


ALL_TOOLS = [tool_forecast_price, tool_retrieve_news, tool_price_history]
