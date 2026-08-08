"""
LangGraph agent — orchestrates retrieve → forecast → guardrail → LLM synthesis.
Supports multi-turn conversation via persistent message history.
LangSmith tracing is enabled automatically when LANGSMITH_API_KEY is set.
"""

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agent.guardrails import (
    GROUNDING_SYSTEM_PROMPT,
    GuardrailViolation,
    StockRecommendation,
    safe_fallback,
    validate_recommendation,
)
from agent.tools import ALL_TOOLS

load_dotenv()

# LangSmith tracing — set env vars if key present
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "stock-rag-agent"))

# ── LLM ───────────────────────────────────────────────────────────────────────

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)

# ── Agent graph ────────────────────────────────────────────────────────────────

_agent = create_react_agent(_llm, ALL_TOOLS)

# ── Public API ─────────────────────────────────────────────────────────────────

def run_agent(ticker: str, question: str, history: list[dict] | None = None) -> dict:
    """
    Run the agent for *ticker* and *question*.

    *history* is a list of prior {"role": "user"|"assistant", "content": str} dicts
    for multi-turn continuity.

    Returns a dict with keys: recommendation (StockRecommendation) | error (str),
    plus raw_response.
    """
    ticker = ticker.upper()

    # Build message list
    messages = [SystemMessage(content=GROUNDING_SYSTEM_PROMPT)]
    for msg in (history or []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        # assistant messages are implicitly part of prior turns; skip for simplicity

    # Inject ticker context into the question
    full_question = (
        f"Ticker: {ticker}\n\n"
        f"Question: {question}\n\n"
        "Use the available tools to gather data, then respond with ONLY a valid JSON object "
        "— no prose before or after it — matching this exact schema:\n"
        "{\"ticker\": str, \"recommendation\": 'buy'|'hold'|'avoid'|'insufficient_data', "
        "\"forecast_price_used\": float|null, \"confidence_used\": float|null, "
        "\"sources_cited\": [str], \"reasoning\": str}"
    )
    messages.append(HumanMessage(content=full_question))

    # Run the agent
    result = _agent.invoke({"messages": messages})
    raw_response = result["messages"][-1].content

    # Pull actuals from tool messages for guardrail cross-check (needed in both branches)
    forecast, retrieved_docs = _extract_tool_actuals(result["messages"])

    # Parse and validate the structured output
    try:
        data = _extract_json(raw_response)
        rec  = StockRecommendation(**data)
        validate_recommendation(rec, forecast, retrieved_docs)
        return {"recommendation": rec, "raw_response": raw_response}

    except GuardrailViolation as e:
        fallback = safe_fallback(ticker, str(e), forecast=forecast)
        return {"recommendation": fallback, "raw_response": raw_response, "guardrail_error": str(e)}
    except ValueError as e:
        # LLM returned prose instead of JSON — treat the raw text as the reasoning
        fallback = safe_fallback(ticker, f"Could not parse structured response: {e}", forecast=forecast)
        fallback.reasoning = raw_response  # surface what the LLM actually said
        return {"recommendation": fallback, "raw_response": raw_response, "guardrail_error": str(e)}
    except Exception as e:
        return {"error": str(e), "raw_response": raw_response}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of *text*."""
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(text[start:end])


def _extract_tool_actuals(messages) -> tuple[dict, list[dict]]:
    """
    Scan tool result messages to extract the forecast dict and retrieved_docs
    list expected by validate_recommendation.
    """
    forecast: dict = {}
    retrieved_docs: list[dict] = []
    doc_counter = 0

    for msg in messages:
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue

        # ForecastResult fields — map to the dict keys guardrails expects
        if isinstance(data, dict) and "forecast_price" in data:
            forecast = {
                "predicted_price": data.get("forecast_price"),
                "confidence":      data.get("confidence"),
                "current_price":   data.get("current_price"),
            }

        # News snippets → retrieved_docs with id, date, text
        if isinstance(data, list):
            for item in data:
                doc_counter += 1
                retrieved_docs.append({
                    "id":   f"doc-{doc_counter}",
                    "date": item.get("datetime", ""),
                    "text": item.get("text", ""),
                    "url":  item.get("url", ""),
                })

    return forecast, retrieved_docs
