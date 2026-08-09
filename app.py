"""
Streamlit UI — Stock RAG Recommendation Agent
⚠️ NOT FINANCIAL ADVICE — for educational/demo purposes only.
"""

import streamlit as st
import plotly.graph_objects as go

from data.data_layer import get_price_history
from agent.agent import run_agent

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stock RAG Agent",
    page_icon="📈",
    layout="centered",
)

st.title("📈 Stock RAG Recommendation Agent")
st.caption("Powered by LightGBM forecasting · RAG news retrieval · Claude Sonnet")

# ── Ticker input ───────────────────────────────────────────────────────────────

ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10).upper().strip()

# ── Session state ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = ticker

# Reset chat when the user switches tickers
if st.session_state.active_ticker != ticker:
    st.session_state.history = []
    st.session_state.active_ticker = ticker

# ── Price chart ────────────────────────────────────────────────────────────────

if ticker:
    with st.spinner(f"Loading {ticker} price history…"):
        try:
            df = get_price_history(ticker, days=90)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"],
                mode="lines", name="Close",
                line=dict(color="#00b4d8", width=2),
            ))
            fig.update_layout(
                title=f"{ticker} — 90-day Close Price",
                xaxis_title="Date", yaxis_title="Price (USD)",
                margin=dict(l=0, r=0, t=40, b=0), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load price data: {e}")

# ── Chat ───────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Ask the agent")

# Render existing history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# New input
if prompt := st.chat_input(f"Ask about {ticker}…"):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = run_agent(
                ticker,
                prompt,
                history=st.session_state.history[:-1],
            )
        st.markdown(response)

    st.session_state.history.append({"role": "assistant", "content": response})

if st.session_state.history:
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.rerun()

# ── Disclaimer ─────────────────────────────────────────────────────────────────

st.divider()
st.caption("⚠️ This is not financial advice. For educational and demo purposes only.")
