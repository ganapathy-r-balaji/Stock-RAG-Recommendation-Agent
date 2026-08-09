"""
Hybrid retrieval — combines semantic search (Chroma) with keyword search (BM25)
and merges results using Reciprocal Rank Fusion (RRF).

Why hybrid?
- Semantic search: catches meaning/paraphrasing ("earnings beat" ≈ "profit exceeded expectations")
- BM25 keyword:    catches exact matches (ticker symbols, CEO names, specific numbers)
- RRF fusion:      re-ranks by combining both rank lists without needing score normalisation
"""

import re
from collections import defaultdict

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

from data.data_layer import get_news

_COLLECTION = "stock_news"
_embed_fn = embedding_functions.DefaultEmbeddingFunction()
_client   = chromadb.Client()  # in-memory

# In-memory BM25 index per ticker: {ticker: {"corpus": [...], "metas": [...], "bm25": BM25Okapi}}
_bm25_index: dict = {}


def _collection():
    return _client.get_or_create_collection(_COLLECTION, embedding_function=_embed_fn)


def _tokenise(text: str) -> list[str]:
    """Simple lowercase word tokeniser for BM25."""
    return re.findall(r"\w+", text.lower())


# ── Indexing ───────────────────────────────────────────────────────────────────

def index_news(ticker: str, days: int = 7) -> int:
    """
    Fetch recent news for *ticker*, upsert into the Chroma vector store,
    and rebuild the BM25 index for that ticker.
    Returns the number of articles indexed.
    """
    articles = get_news(ticker, days=days)
    if not articles:
        return 0

    docs  = [f"{a['headline']}\n{a['summary']}" for a in articles]
    ids   = [f"{ticker}_{a['datetime']}_{i}" for i, a in enumerate(articles)]
    metas = [{"ticker": ticker, "url": a["url"], "datetime": a["datetime"]} for a in articles]

    # Semantic index
    col = _collection()
    col.upsert(documents=docs, ids=ids, metadatas=metas)

    # Keyword index
    _bm25_index[ticker] = {
        "corpus": docs,
        "metas":  metas,
        "bm25":   BM25Okapi([_tokenise(d) for d in docs]),
    }

    return len(docs)


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """
    Merge multiple ranked lists of doc indices using RRF.
    Returns a single list of indices sorted by fused score (best first).
    """
    scores: dict[int, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


# ── Hybrid retrieval ───────────────────────────────────────────────────────────

def retrieve_news(ticker: str, query: str, k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: semantic (Chroma) + keyword (BM25), fused via RRF.
    Returns the top *k* deduplicated results, each with: text, url, datetime.
    """
    fetch_n = min(k * 3, 20)  # fetch more than k from each source before fusion

    # ── Semantic results ──────────────────────────────────────────────────────
    col = _collection()
    try:
        sem_results = col.query(
            query_texts=[query],
            n_results=fetch_n,
            where={"ticker": ticker},
        )
        sem_docs  = sem_results.get("documents", [[]])[0]
        sem_metas = sem_results.get("metadatas", [[]])[0]
    except Exception:
        sem_docs, sem_metas = [], []

    # ── BM25 keyword results ──────────────────────────────────────────────────
    bm25_docs: list[str] = []
    bm25_metas: list[dict] = []
    if ticker in _bm25_index:
        idx = _bm25_index[ticker]
        scores = idx["bm25"].get_scores(_tokenise(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_n]
        bm25_docs  = [idx["corpus"][i] for i in top_indices]
        bm25_metas = [idx["metas"][i]  for i in top_indices]

    # ── Merge via RRF ─────────────────────────────────────────────────────────
    # Build a unified pool of (doc, meta) deduped by text
    pool: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for doc, meta in list(zip(sem_docs, sem_metas)) + list(zip(bm25_docs, bm25_metas)):
        if doc not in seen:
            seen.add(doc)
            pool.append((doc, meta))

    n = len(pool)
    if n == 0:
        return []

    # Rank each source over the pool
    sem_set   = {d: i for i, (d, _) in enumerate(pool)}
    bm25_set  = {d: i for i, (d, _) in enumerate(pool)}
    sem_rank  = [sem_set[d]  for d in sem_docs  if d in sem_set]
    bm25_rank = [bm25_set[d] for d in bm25_docs if d in bm25_set]

    fused_indices = _rrf([sem_rank, bm25_rank])

    # Fill any pool items not appearing in either ranked list at the end
    ranked_set = set(fused_indices)
    fused_indices += [i for i in range(n) if i not in ranked_set]

    return [
        {"text": pool[i][0], "url": pool[i][1].get("url", ""), "datetime": pool[i][1].get("datetime", "")}
        for i in fused_indices[:k]
    ]
