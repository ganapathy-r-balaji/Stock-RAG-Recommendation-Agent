"""
Evaluation module — measures retrieval and generation quality using LLM-as-judge.

Metrics
-------
Retrieval
  - context_relevance   : are the retrieved snippets relevant to the query?  [0–1]
  - context_coverage    : does the context contain enough info to answer?     [0–1]

Generation
  - faithfulness        : is the answer grounded in the retrieved context?    [0–1]
  - answer_relevance    : does the answer actually address the question?       [0–1]

Each metric is scored 0–1 by Claude (LLM-as-judge) with a brief rationale.
Scores are logged to LangSmith as run feedback when a run_id is provided.

Usage
-----
    from evals.metrics import evaluate
    scores = evaluate(
        ticker="AAPL",
        query="Should I buy this stock?",
        retrieved_docs=[{"text": "...", "url": "..."}],
        answer="Based on the forecast...",
        run_id="<langsmith-run-id>",   # optional
    )
    print(scores)
"""

import json
import os
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
_MODEL  = "claude-haiku-4-5-20251001"   # fast + cheap for evals


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class EvalScores:
    context_relevance:  float   # 0–1
    context_coverage:   float   # 0–1
    faithfulness:       float   # 0–1
    answer_relevance:   float   # 0–1
    rationales:         dict    # metric → explanation string

    @property
    def overall(self) -> float:
        """Simple unweighted average of all four metrics."""
        return round(
            (self.context_relevance + self.context_coverage +
             self.faithfulness + self.answer_relevance) / 4,
            3,
        )

    def to_dict(self) -> dict:
        return {
            "context_relevance": self.context_relevance,
            "context_coverage":  self.context_coverage,
            "faithfulness":      self.faithfulness,
            "answer_relevance":  self.answer_relevance,
            "overall":           self.overall,
            "rationales":        self.rationales,
        }


# ── LLM judge ─────────────────────────────────────────────────────────────────

def _judge(prompt: str) -> dict:
    """Call Claude and parse a JSON response with {score: float, rationale: str}."""
    response = _client.messages.create(
        model=_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    # Extract JSON from response
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1:
        return {"score": 0.5, "rationale": text}
    return json.loads(text[start:end])


# ── Individual metric prompts ──────────────────────────────────────────────────

def _score_context_relevance(query: str, docs: list[dict]) -> dict:
    context = "\n---\n".join(d.get("text", "") for d in docs)
    return _judge(f"""You are an evaluation judge.

Query: {query}

Retrieved context:
{context}

Score how relevant the retrieved context is to the query on a scale of 0.0 to 1.0.
- 1.0 = all retrieved snippets are directly relevant
- 0.5 = some relevant, some off-topic
- 0.0 = completely irrelevant

Respond with ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<1 sentence>"}}""")


def _score_context_coverage(query: str, docs: list[dict]) -> dict:
    context = "\n---\n".join(d.get("text", "") for d in docs)
    return _judge(f"""You are an evaluation judge.

Query: {query}

Retrieved context:
{context}

Score whether the context contains SUFFICIENT information to answer the query (0.0–1.0).
- 1.0 = context fully covers what's needed to answer
- 0.5 = partial coverage, some key info missing
- 0.0 = context is insufficient to answer at all

Respond with ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<1 sentence>"}}""")


def _score_faithfulness(answer: str, docs: list[dict]) -> dict:
    context = "\n---\n".join(d.get("text", "") for d in docs)
    return _judge(f"""You are an evaluation judge.

Retrieved context:
{context}

Agent answer:
{answer}

Score how faithfully the answer is grounded in the retrieved context (0.0–1.0).
- 1.0 = every claim in the answer is directly supported by the context
- 0.5 = mostly grounded but contains some unsupported claims
- 0.0 = answer contradicts or ignores the context entirely

Respond with ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<1 sentence>"}}""")


def _score_answer_relevance(query: str, answer: str) -> dict:
    return _judge(f"""You are an evaluation judge.

Question: {query}

Answer:
{answer}

Score how well the answer addresses the question (0.0–1.0).
- 1.0 = answer directly and completely addresses the question
- 0.5 = partially addresses the question
- 0.0 = answer is off-topic or refuses to engage

Respond with ONLY valid JSON: {{"score": <float 0-1>, "rationale": "<1 sentence>"}}""")


# ── Public API ─────────────────────────────────────────────────────────────────

def evaluate(
    ticker: str,
    query: str,
    retrieved_docs: list[dict],
    answer: str,
    run_id: str | None = None,
) -> EvalScores:
    """
    Run all four eval metrics and return an EvalScores object.
    If *run_id* is provided, scores are posted to LangSmith as run feedback.
    """
    full_query = f"[{ticker}] {query}"

    cr  = _score_context_relevance(full_query, retrieved_docs)
    cc  = _score_context_coverage(full_query, retrieved_docs)
    fth = _score_faithfulness(answer, retrieved_docs)
    ar  = _score_answer_relevance(full_query, answer)

    scores = EvalScores(
        context_relevance = round(float(cr.get("score",  0.5)), 3),
        context_coverage  = round(float(cc.get("score",  0.5)), 3),
        faithfulness      = round(float(fth.get("score", 0.5)), 3),
        answer_relevance  = round(float(ar.get("score",  0.5)), 3),
        rationales={
            "context_relevance": cr.get("rationale",  ""),
            "context_coverage":  cc.get("rationale",  ""),
            "faithfulness":      fth.get("rationale", ""),
            "answer_relevance":  ar.get("rationale",  ""),
        },
    )

    # Log to LangSmith if run_id is available
    if run_id:
        _log_to_langsmith(run_id, scores)

    return scores


def _log_to_langsmith(run_id: str, scores: EvalScores) -> None:
    """Post eval scores as feedback to a LangSmith run."""
    try:
        from langsmith import Client as LangSmithClient
        ls = LangSmithClient()
        for key, value in [
            ("context_relevance", scores.context_relevance),
            ("context_coverage",  scores.context_coverage),
            ("faithfulness",      scores.faithfulness),
            ("answer_relevance",  scores.answer_relevance),
            ("overall",           scores.overall),
        ]:
            ls.create_feedback(run_id=run_id, key=key, score=value)
    except Exception:
        pass  # don't crash the app if LangSmith logging fails
