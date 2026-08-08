# Stock RAG Recommendation Agent

A portfolio/demo project that combines price forecasting, vector RAG over news, and a
knowledge graph to produce grounded **buy / hold / avoid** recommendations via a
conversational Streamlit interface.

> ⚠️ **NOT FINANCIAL ADVICE** — for educational and demo purposes only.

---

## Project structure

```
├── app.py                       # Streamlit UI
├── data/
│   └── data_layer.py            # Finnhub API wrapper
├── forecast/
│   └── forecaster.py            # LightGBM 3-day-ahead price forecaster
├── retrieval/
│   ├── vector_store.py          # Chroma vector store over Finnhub news
│   └── knowledge_graph.py      # NetworkX ticker/sector/peer graph
├── agent/
│   ├── tools.py                 # LangChain tools (forecast, retrieve, graph)
│   ├── guardrails.py            # Structured output schema + validator
│   └── agent.py                 # LangGraph ReAct agent
├── requirements.txt
└── .env.example
```

## Setup

```bash
# 1. Clone & create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in your API keys
cp .env.example .env
#   → FINNHUB_API_KEY  (free at finnhub.io)
#   → ANTHROPIC_API_KEY
#   → LANGSMITH_API_KEY  (optional, enables tracing)

# 4. Run
streamlit run app.py
```

## How it works

1. **Data layer** — fetches OHLCV price history and news from Finnhub.
2. **Forecaster** — engineers lag/MA/RSI features, trains a LightGBM model with walk-forward
   validation, and returns a 3-day-ahead price prediction with a confidence score.
3. **Vector store** — embeds news articles into Chroma; the agent queries it semantically.
4. **Knowledge graph** — NetworkX graph of ticker → sector → peers for peer comparison.
5. **Agent** — LangGraph ReAct loop: calls tools → validates output via guardrails →
   synthesises a `StockRecommendation` with citations.
6. **Guardrails** — cross-checks cited prices/confidence against actual tool output;
   falls back to `insufficient_data` on mismatch or low confidence.
7. **LangSmith** — every run, tool call, and guardrail event is traced automatically
   when `LANGSMITH_API_KEY` is set.
