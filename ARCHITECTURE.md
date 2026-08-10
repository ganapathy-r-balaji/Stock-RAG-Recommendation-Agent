# Architecture

```mermaid
flowchart TD
    U[User: ticker + question] --> UI[Streamlit UI\nprice chart · multi-turn chat]
    UI --> AG[LangGraph ReAct Agent\nClaude Sonnet 4.6]

    subgraph Tools
        T1[tool_forecast_price]
        T2[tool_retrieve_news]
        T3[tool_price_history]
    end

    AG -->|tool call| T1
    AG -->|tool call| T2
    AG -->|tool call| T3

    subgraph Data
        YF[yfinance] --> PH[Price History]
        YF --> NW[News fallback]
        FH[Finnhub API\noptional] --> NW
    end

    subgraph Forecast
        PH --> FE[Feature Engineering\nlags · MA · RSI]
        FE --> LGB[LightGBM Regressor\nwalk-forward validated]
        LGB --> FC[3-day Forecast + confidence]
    end

    subgraph RAG
        NW --> VEC[Chroma Vector Store\nin-memory]
        NW --> BM25[BM25 Keyword Index\nin-memory]
        VEC --> SEM[Semantic Retrieval]
        BM25 --> KW[Keyword Retrieval]
        SEM --> RRF[RRF Fusion\nReciprocal Rank Fusion]
        KW --> RRF
    end

    T1 --> LGB
    T2 --> RRF
    T3 --> PH

    FC --> AG
    RET --> AG
    PH --> AG

    AG --> LLM[Claude Sonnet 4.6\nfinal answer]
    LLM --> UI
    PH --> SNAP[Claude Opus 4.5\nstock snapshot · 3-4 bullets]
    FC --> SNAP
    SNAP --> UI
    AG -.->|trace| LS[LangSmith]
```

## Components

| Layer | What it does | Key file |
|---|---|---|
| **Data** | Price history via Alpha Vantage (primary, cloud-safe) → yfinance fallback; news via Finnhub optional → yfinance | `data/data_layer.py` |
| **Forecast** | LightGBM with lag/MA/RSI features, walk-forward backtested | `forecast/forecaster.py` |
| **Vector store** | FAISS IndexFlatIP (OpenAI embeddings, cosine similarity) + BM25 keyword, fused via RRF | `retrieval/vector_store.py` |
| **Agent tools** | 3 LangChain tools: forecast · news RAG · price history | `agent/tools.py` |
| **Agent** | LangGraph ReAct loop with multi-turn memory; LangSmith tracing | `agent/agent.py` |
| **Snapshot** | Claude Opus 4.5 generates 3-4 bullet price + forecast summary shown before chat | `app.py` |
| **UI** | Streamlit: ticker input, price chart, snapshot, persistent multi-turn chat | `app.py` |


```mermaid
flowchart TD
    U[User: ticker + question] --> AG[LangGraph Agent]

    subgraph Data Layer
        YF[yfinance] --> PD[Price / Volume History]
        FH[Finnhub API\noptional] --> NW[Company News]
        YF --> NW
    end

    subgraph Forecast
        PD --> FE[Feature Engineering\nlags · MA · RSI · volatility]
        FE --> LGB[LightGBM Regressor\nwalk-forward validated]
        LGB --> FC[3-day Price Forecast\n+ confidence score]
    end

    subgraph Retrieval
        NW --> VEC[Vector Store\nChroma in-memory]
        KG[Knowledge Graph\nNetworkX: ticker → sector → peers]
    end

    subgraph Agent Tools
        T1[tool_forecast_price]
        T2[tool_retrieve_news]
        T3[tool_query_graph]
        T4[tool_price_history]
    end

    AG -->|tool call| T1
    AG -->|tool call| T2
    AG -->|tool call| T3
    AG -->|tool call| T4

    T1 --> LGB
    T2 --> VEC
    T3 --> KG
    T4 --> PD

    FC --> AG
    VEC --> AG
    KG --> AG

    AG --> GR[Guardrail Validator\ncitation check · price check\nconfidence threshold]
    GR -->|pass| LLM[Claude claude-haiku-4-5]
    GR -->|fail| SAFE[Fallback: insufficient_data]
    LLM --> REC[StockRecommendation\nbuy · hold · avoid · insufficient_data]
    SAFE --> REC

    REC --> UI[Streamlit UI\nprice chart · chat · disclaimer]
    AG -.->|trace| LS[LangSmith]
```

## Components

| Layer | What it does | Key file |
|---|---|---|
| **Data** | Price history via yfinance; news via Finnhub (falls back to yfinance) | `data/data_layer.py` |
| **Forecast** | LightGBM regressor with lag/MA/RSI features, walk-forward backtested | `forecast/forecaster.py` |
| **Vector store** | Chroma in-memory store; news embedded and retrieved by semantic similarity | `retrieval/vector_store.py` |
| **Knowledge graph** | NetworkX graph of ticker → sector → peer companies | `retrieval/knowledge_graph.py` |
| **Agent tools** | 4 LangChain tools the agent can call: forecast, news, graph, price history | `agent/tools.py` |
| **Guardrails** | 3-layer hallucination mitigation: system prompt + Pydantic schema + code validator | `agent/guardrails.py` |
| **Agent** | LangGraph ReAct loop; LangSmith tracing wired in from the start | `agent/agent.py` |
| **UI** | Streamlit: ticker input, close price chart, chat interface, persistent disclaimer | `app.py` |
