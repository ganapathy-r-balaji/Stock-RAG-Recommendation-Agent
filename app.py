"""
Streamlit UI — Stock RAG Recommendation Agent
⚠️ NOT FINANCIAL ADVICE — for educational/demo purposes only.
"""

import streamlit as st
import plotly.graph_objects as go

from data.data_layer import get_price_history

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
            from agent.agent import run_agent  # lazy import — loads LangChain only on first query
            response, eval_scores = run_agent(
                ticker,
                prompt,
                history=st.session_state.history[:-1],
            )
        st.markdown(response)

        if eval_scores:
            with st.expander(f"📊 Eval scores — overall {eval_scores['overall']:.2f}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Context relevance",  f"{eval_scores['context_relevance']:.2f}")
                col2.metric("Context coverage",   f"{eval_scores['context_coverage']:.2f}")
                col3.metric("Faithfulness",        f"{eval_scores['faithfulness']:.2f}")
                col4.metric("Answer relevance",    f"{eval_scores['answer_relevance']:.2f}")
                st.caption("Scores are 0–1, judged by Claude Haiku (LLM-as-judge). Higher = better.")
                if rationales := eval_scores.get("rationales"):
                    for metric, reason in rationales.items():
                        if reason:
                            st.markdown(f"**{metric.replace('_', ' ').title()}**: {reason}")

    st.session_state.history.append({"role": "assistant", "content": response})

if st.session_state.history:
    if st.button("🗑 Clear chat"):
        st.session_state.history = []
        st.rerun()

# ── Disclaimer ─────────────────────────────────────────────────────────────────

st.divider()
st.caption("⚠️ This is not financial advice. For educational and demo purposes only.")
