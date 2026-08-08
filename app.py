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

# ── Persistent disclaimer ──────────────────────────────────────────────────────

st.warning(
    "⚠️ **NOT FINANCIAL ADVICE** — This tool is for educational and demo purposes only. "
    "Do not make investment decisions based on its output.",
    icon="🚨",
)

st.title("📈 Stock RAG Recommendation Agent")

# ── Session state ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []   # list of {"role", "content"}
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = ""

# ── Sidebar: ticker input ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL", max_chars=10).upper().strip()
    days   = st.slider("Price history (days)", 30, 365, 90)

    if st.button("🔄 Clear chat"):
        st.session_state.history = []
        st.rerun()

# ── Price chart ────────────────────────────────────────────────────────────────

if ticker:
    with st.spinner(f"Loading price history for {ticker}…"):
        try:
            df = get_price_history(ticker, days=days)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["close"],
                mode="lines", name="Close",
                line=dict(color="#1f77b4", width=2),
            ))
            fig.update_layout(
                title=f"{ticker} — {days}-day Close Price",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                margin=dict(l=0, r=0, t=40, b=0),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load price data: {e}")

# ── Chat interface ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("Ask the agent")

# Display existing chat history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input(f"Ask about {ticker}…"):
    # Show user message
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            output = run_agent(ticker, prompt, history=st.session_state.history[:-1])

        if "error" in output:
            response_text = f"❌ Error: {output['error']}"
            st.error(response_text)

        else:
            rec = output["recommendation"]

            # Colour-code the badge
            badge_map = {
                "buy":              "🟢 **BUY**",
                "hold":             "🟡 **HOLD**",
                "avoid":            "🔴 **AVOID**",
                "insufficient_data": "⚪ **INSUFFICIENT DATA**",
            }
            badge = badge_map.get(rec.recommendation, rec.recommendation.upper())

            lines = [f"### {badge}", "", rec.reasoning, ""]

            if rec.forecast_price_used is not None:
                lines.append(f"**Forecast price used:** ${rec.forecast_price_used:.2f}")
            if rec.confidence_used is not None:
                lines.append(f"**Model confidence:** {rec.confidence_used:.1%}")

            if rec.sources_cited:
                lines += ["", "**Sources:**"]
                for url in rec.sources_cited:
                    lines.append(f"- {url}")

            # Guardrail note
            if "guardrail_error" in output:
                lines += ["", f"> ⚠️ Guardrail triggered: {output['guardrail_error']}"]

            response_text = "\n".join(lines)
            st.markdown(response_text)

        st.session_state.history.append({"role": "assistant", "content": response_text})
