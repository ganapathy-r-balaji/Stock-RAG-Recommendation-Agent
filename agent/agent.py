"""
RAG + Forecast agent — LangGraph ReAct loop.
Model: Claude Sonnet 4.6
Supports multi-turn conversation via persistent message history.
LangSmith tracing enabled automatically when LANGSMITH_API_KEY is set.
"""

import json
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from agent.tools import ALL_TOOLS

load_dotenv()

# LangSmith tracing
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "stock-rag-agent"))

_SYSTEM_PROMPT = """You are a stock research assistant with access to three tools:
- tool_forecast_price: ML-based 3-day price forecast with confidence score
- tool_retrieve_news: semantic search over recent news articles
- tool_price_history: historical price performance over any number of days

Always call the relevant tools before answering. Base every claim strictly on
tool output — never invent prices, forecasts, or news. If data is insufficient,
say so clearly. Be concise and factual.

⚠️ This is not financial advice. For informational purposes only."""

# Lazy-initialised agent (avoids import-time API calls)
_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatAnthropic(
            model="claude-sonnet-4-6",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.2,
        )
        _agent = create_react_agent(
            llm,
            tools=ALL_TOOLS,
            prompt=_SYSTEM_PROMPT,
        )
    return _agent


def run_agent(
    ticker: str,
    question: str,
    history: list[dict] | None = None,
    run_evals: bool = True,
) -> tuple[str, dict | None]:
    """
    Run the agent for *ticker* and *question*.

    *history* is a list of prior {"role": "user"|"assistant", "content": str} dicts
    for multi-turn continuity.

    Returns (answer: str, eval_scores: dict | None).
    eval_scores is None when run_evals=False or evaluation fails.
    """
    messages: list = []
    for turn in (history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=f"Ticker: {ticker}\n\nQuestion: {question}"))

    try:
        result   = _get_agent().invoke({"messages": messages})
        answer   = result["messages"][-1].content

        eval_scores = None
        if run_evals:
            # Extract news snippets returned by tool_retrieve_news (if called)
            retrieved_docs: list[dict] = []
            for msg in result["messages"]:
                if isinstance(msg, ToolMessage):
                    try:
                        data = json.loads(msg.content)
                        if isinstance(data, list):          # news snippets are a list
                            retrieved_docs.extend(data)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Only run evals when there are retrieved docs (RAG path was triggered)
            if retrieved_docs:
                from evals.metrics import evaluate
                scores = evaluate(
                    ticker=ticker,
                    query=question,
                    retrieved_docs=retrieved_docs,
                    answer=answer,
                )
                eval_scores = scores.to_dict()

        return answer, eval_scores
    except Exception as e:
        return f"❌ Agent error: {e}", None
