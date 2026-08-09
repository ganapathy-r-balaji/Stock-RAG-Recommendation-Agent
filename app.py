"""
Streamlit UI — Stock RAG Recommendation Agent
⚠️ NOT FINANCIAL ADVICE — for educational/demo purposes only.
"""

import streamlit as st
import plotly.graph_objects as go

from data.data_layer import get_price_history
from agent.agent import run_agent
from forecast.forecaster import forecast_price


@st.cache_data(ttl=3600)
def _cached_price_history(ticker: str):
    return get_price_history(ticker, days=90)


@st.cache_data(ttl=3600)
def _cached_snapshot(ticker: str) -> str:
    """Generate a 3-4 bullet stock snapshot using Claude — cached 1 hour per ticker."""
    import os
    from anthropic import Anthropic

    # Gather data
    try:
        from forecast.forecaster import forecast_price
        fc = forecast_price(ticker)
        forecast_text = (
            f"3-day forecast: {fc.predicted_price:.2f} (current: {fc.current_price:.2f}, "
            f"{fc.pct_change:+.1f}%, confidence: {fc.confidence:.0%})"
        )
    except Exception as e:
        forecast_text = f"Forecast unavailable: {e}"

    try:
        df = get_price_history(ticker, days=180)
        start_6m = round(float(df["close"].iloc[0]), 2)
        end_6m   = round(float(df["close"].iloc[-1]), 2)
        pct_6m   = round((end_6m - start_6m) / start_6m * 100, 1)
        high_6m  = round(float(df["high"].max()), 2)
        low_6m   = round(float(df["low"].min()), 2)
        price_text = (
            f"6-month price history: start ${start_6m}, current ${end_6m}, "
            f"change {pct_6m:+.1f}%, 6m high ${high_6m}, 6m low ${low_6m}"
        )
    except Exception as e:
        price_text = f"Price history unavailable: {e}"

    prompt = f"""You are a concise stock analyst. Given the following data for {ticker}, write exactly 3-4 bullet points summarising:
1. Recent price performance
2. Short-term forecast signal
3. What this suggests for the next 6 months (up/down/sideways and why)

Data:
- {price_text}
- {forecast_text}

Rules:
- Use bullet points (•)
- Be concise — one sentence per bullet
- End with a directional outlook for the next 6 months
- Do NOT give financial advice, just analytical observations
- Do NOT use markdown headers
- Write prices as plain numbers with USD suffix (e.g. 313.33 USD) — do NOT use $ signs"""

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()

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
            df = _cached_price_history(ticker)
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

    # ── Stock snapshot ─────────────────────────────────────────────────────────
    with st.spinner(f"Analysing {ticker}…"):
        try:
            snapshot = _cached_snapshot(ticker)
            # Escape $ signs to prevent Streamlit rendering them as LaTeX
            snapshot_escaped = snapshot.replace("$", "\\$")
            st.info(snapshot_escaped)
        except Exception:
            pass  # silently skip if snapshot fails

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
                st.caption("Scores are 0–1, judged by GPT-4o-mini (LLM-as-judge). Higher = better.")
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
