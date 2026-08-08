"""
Guardrails — three-layer hallucination mitigation.

1. Grounding system prompt  (used when building the agent)
2. Structured output schema (StockRecommendation)
3. Code-level validator     (validate_recommendation)

Wire into a LangGraph node: build_grounded_prompt() → LLM (structured output)
→ validate_recommendation(). On failure, retry once with the failure reason
appended, or fall back to a safe "insufficient data" response.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── 1. System prompt ───────────────────────────────────────────────────────────
# The point of this prompt isn't "please don't hallucinate" (models ignore
# vague pleas). It's specific, falsifiable rules the model can actually follow,
# plus an explicit permission to say "insufficient data" instead of guessing.

GROUNDING_SYSTEM_PROMPT = """You are a stock analysis assistant. You have exactly
two sources of truth: the FORECAST block and the RETRIEVED_CONTEXT block below.

Rules:
1. Every numeric claim (price, % change, date) must come verbatim from FORECAST
   or RETRIEVED_CONTEXT. Never compute, round, or estimate a number that wasn't
   given to you.
2. Every factual claim about news, earnings, or events must cite a source_id
   from RETRIEVED_CONTEXT using the format [source: <id>]. If you make a claim
   with no matching source_id, that is a violation of your instructions.
3. If RETRIEVED_CONTEXT is empty or FORECAST confidence is below 0.5, you must
   set recommendation="insufficient_data" rather than guessing. Saying "I don't
   have enough information" is a correct, complete answer.
4. Do not use outside knowledge about the company, ticker, or market beyond what
   is in FORECAST and RETRIEVED_CONTEXT, even if you recognize the ticker.
5. State your reasoning as a chain from source_id(s) -> claim -> conclusion, not
   as a general market opinion.

This is a demo/educational tool, not licensed financial advice. Never state or
imply certainty about future price movement.
"""

# Keep the old name as an alias so existing callers don't break.
SYSTEM_PROMPT = GROUNDING_SYSTEM_PROMPT


def build_grounded_prompt(ticker: str, forecast: dict, retrieved_docs: list[dict]) -> str:
    """Assemble the user-turn prompt with explicit, numbered, citable context blocks."""
    docs_block = "\n".join(
        f"[source: {d['id']}] ({d['date']}) {d['text']}" for d in retrieved_docs
    ) or "(no documents retrieved)"

    return f"""FORECAST:
ticker: {ticker}
predicted_price_3d: {forecast['predicted_price']}
confidence: {forecast['confidence']}
current_price: {forecast['current_price']}

RETRIEVED_CONTEXT:
{docs_block}

Based only on the above, should the user buy, hold, or avoid {ticker}?"""


# ── 2. Structured output schema ────────────────────────────────────────────────
# A free-text answer is easy to hallucinate in. A schema that REQUIRES a
# sources_cited list and a forecast_price_used field makes fabrication visible
# and checkable — the model has to commit to specific, verifiable claims.

class StockRecommendation(BaseModel):
    ticker: str
    recommendation: Literal["buy", "hold", "avoid", "insufficient_data"]

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalise_recommendation(cls, v: Any) -> str:
        """Lowercase and normalise the recommendation value from the LLM."""
        if not isinstance(v, str):
            return "insufficient_data"
        v = v.lower().strip().replace(" ", "_")
        if v in ("buy", "hold", "avoid", "insufficient_data"):
            return v
        # Map common LLM variants to valid values
        if v.startswith("buy"):
            return "buy"
        if v.startswith("hold") or v.startswith("neutral"):
            return "hold"
        if v.startswith("avoid") or v.startswith("sell"):
            return "avoid"
        return "insufficient_data"
    forecast_price_used: float | None = Field(
        default=None,
        description="The predicted_price_3d value the model actually used — must match FORECAST exactly",
    )
    confidence_used: float | None = Field(
        default=None,
        description="The confidence value from FORECAST that was used",
    )
    sources_cited: list[str] = Field(
        default_factory=list,
        description="source_id values from RETRIEVED_CONTEXT that support the reasoning",
    )
    reasoning: str = Field(
        description="Chain from cited sources + forecast to the conclusion",
    )

    @field_validator("reasoning", mode="before")
    @classmethod
    def coerce_reasoning_to_str(cls, v: Any) -> str:
        """LLMs sometimes return reasoning as a dict — flatten it to a string."""
        if isinstance(v, dict):
            return " | ".join(f"{k}: {val}" for k, val in v.items())
        return str(v)

    @field_validator("sources_cited", mode="before")
    @classmethod
    def coerce_sources_to_list(cls, v: Any) -> list:
        """Handle sources being None or a single string."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v else []
        return list(v)


# ── 3. Code-level validator ────────────────────────────────────────────────────
# This is the layer that matters most. The LLM can still lie in a structured
# field; this function catches it before the user ever sees the output.

class GuardrailViolation(Exception):
    pass

# Alias so agent.py and other callers that import GuardrailError still work.
GuardrailError = GuardrailViolation


def validate_recommendation(
    rec: StockRecommendation,
    forecast: dict,
    retrieved_docs: list[dict],
    price_tolerance: float = 0.01,
) -> None:
    """
    Raise GuardrailViolation if the LLM's output doesn't match ground truth.
    Call this before returning anything to the user.
    """
    valid_ids = {d["id"] for d in retrieved_docs}

    # 1. No fabricated citations
    fabricated = set(rec.sources_cited) - valid_ids
    if fabricated:
        raise GuardrailViolation(f"Cited sources that don't exist: {fabricated}")

    # 2. Forecast number must match what was actually computed
    if rec.forecast_price_used is not None and forecast.get("predicted_price") is not None:
        if abs(rec.forecast_price_used - forecast["predicted_price"]) > price_tolerance:
            raise GuardrailViolation(
                f"Forecast price mismatch: model said {rec.forecast_price_used}, "
                f"tool returned {forecast['predicted_price']}"
            )

    # 3. Confidence must match — catches the model inventing certainty
    if rec.confidence_used is not None and forecast.get("confidence") is not None:
        if abs(rec.confidence_used - forecast["confidence"]) > 0.01:
            raise GuardrailViolation(
                f"Confidence mismatch: model said {rec.confidence_used}, "
                f"tool returned {forecast['confidence']}"
            )

    # 4. Low-confidence forecast must not produce a confident buy/avoid
    if (
        forecast.get("confidence") is not None
        and forecast["confidence"] < 0.5
        and rec.recommendation != "insufficient_data"
    ):
        raise GuardrailViolation(
            f"Confidence {forecast['confidence']} is below threshold but model "
            "gave a confident recommendation instead of insufficient_data"
        )

    # 5. Only buy/avoid recommendations strictly require cited sources
    if rec.recommendation in ("buy", "avoid") and not rec.sources_cited:
        raise GuardrailViolation(
            "Buy/avoid recommendation given with no cited sources — likely ungrounded"
        )


def safe_fallback(ticker: str, reason: str, forecast: dict | None = None) -> StockRecommendation:
    """Return a safe fallback recommendation when guardrails fail."""
    return StockRecommendation(
        ticker=ticker,
        recommendation="insufficient_data",
        reasoning=f"Guardrail triggered — {reason}",
        forecast_price_used=forecast.get("predicted_price") if forecast else None,
        confidence_used=forecast.get("confidence") if forecast else None,
        sources_cited=[],
    )


# ── LangGraph node sketch ──────────────────────────────────────────────────────

def grounded_recommendation_node(
    ticker: str, forecast: dict, retrieved_docs: list[dict], llm
) -> StockRecommendation:
    """
    Sketch of the LangGraph node. `llm` is a chat model bound to
    StockRecommendation via .with_structured_output().
    """
    prompt = build_grounded_prompt(ticker, forecast, retrieved_docs)
    rec: StockRecommendation = llm.invoke(
        [
            {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
    )

    try:
        validate_recommendation(rec, forecast, retrieved_docs)
    except GuardrailViolation as e:
        return safe_fallback(
            ticker,
            reason=str(e),
            forecast=forecast,
        )

    return rec
