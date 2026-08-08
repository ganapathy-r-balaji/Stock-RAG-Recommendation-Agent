"""
Knowledge graph — lightweight NetworkX graph of ticker → sector → peers.
Populated once per session from Finnhub company profiles.
"""

import networkx as nx

from data.data_layer import get_company_profile

_graph: nx.Graph = nx.Graph()


# ── Graph population ───────────────────────────────────────────────────────────

def build_graph(ticker: str) -> None:
    """
    Add *ticker* and its sector/peers to the graph.
    Safe to call multiple times; duplicate nodes/edges are ignored.
    """
    profile = get_company_profile(ticker)

    sector = profile["sector"]
    peers  = profile["peers"]

    _graph.add_node(ticker, type="ticker", name=profile["name"])
    _graph.add_node(sector, type="sector")
    _graph.add_edge(ticker, sector, relation="belongs_to")

    for peer in peers:
        if peer == ticker:
            continue
        _graph.add_node(peer, type="ticker")
        _graph.add_edge(peer, sector, relation="belongs_to")
        _graph.add_edge(ticker, peer, relation="peer")


# ── Query helpers ──────────────────────────────────────────────────────────────

def get_sector(ticker: str) -> str:
    """Return the sector for *ticker*, or 'Unknown' if not in graph."""
    for neighbor in _graph.neighbors(ticker):
        if _graph.nodes[neighbor].get("type") == "sector":
            return str(neighbor)
    return "Unknown"


def get_peers(ticker: str) -> list[str]:
    """Return list of peer tickers for *ticker*."""
    return [
        n for n in _graph.neighbors(ticker)
        if _graph.nodes[n].get("type") == "ticker" and n != ticker
    ]


def query_graph(ticker: str) -> dict:
    """Return a summary dict with sector and peers for the agent to consume."""
    if ticker not in _graph:
        build_graph(ticker)
    return {
        "ticker": ticker,
        "sector": get_sector(ticker),
        "peers":  get_peers(ticker),
    }
