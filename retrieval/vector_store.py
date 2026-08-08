"""
Vector store — embeds Finnhub/yfinance news articles into Chroma so the agent
can retrieve relevant context by semantic similarity.
Uses chromadb's default embedding (no heavy torch/onnx dependencies).
"""

import chromadb
from chromadb.utils import embedding_functions

from data.data_layer import get_news

_COLLECTION = "stock_news"
_embed_fn = embedding_functions.DefaultEmbeddingFunction()
_client   = chromadb.Client()  # in-memory; fine for Streamlit Cloud


def _collection():
    return _client.get_or_create_collection(_COLLECTION, embedding_function=_embed_fn)


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_news(ticker: str, days: int = 7) -> int:
    """
    Fetch recent news for *ticker* and upsert into the vector store.
    Returns the number of articles indexed.
    """
    articles = get_news(ticker, days=days)
    if not articles:
        return 0

    col  = _collection()
    docs = [f"{a['headline']}\n{a['summary']}" for a in articles]
    ids  = [f"{ticker}_{a['datetime']}_{i}" for i, a in enumerate(articles)]
    metas = [{"ticker": ticker, "url": a["url"], "datetime": a["datetime"]} for a in articles]

    col.upsert(documents=docs, ids=ids, metadatas=metas)
    return len(docs)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def retrieve_news(ticker: str, query: str, k: int = 5) -> list[dict]:
    """
    Return the *k* most relevant news snippets for *query* filtered to *ticker*.
    Each result dict has: text, url, datetime.
    """
    col = _collection()
    results = col.query(
        query_texts=[query],
        n_results=k,
        where={"ticker": ticker},
    )
    docs   = results.get("documents", [[]])[0]
    metas  = results.get("metadatas", [[]])[0]
    return [
        {"text": doc, "url": meta.get("url", ""), "datetime": meta.get("datetime", "")}
        for doc, meta in zip(docs, metas)
    ]
