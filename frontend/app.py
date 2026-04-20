"""
app.py — Streamlit frontend for BNS Legal Research RAG chatbot.
Minimal, clean chat interface. Talks to the FastAPI backend.
"""

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="BNS Legal Research",
    page_icon="⚖️",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Minimal custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Clean header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .main-header p {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* Citation box */
    .citation-box {
        background: #f8f9fa;
        border-left: 3px solid #4a6fa5;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.82rem;
        color: #444;
        margin-top: 0.5rem;
    }
    .citation-box strong {
        color: #1a1a2e;
        display: block;
        margin-bottom: 0.3rem;
    }

    /* No-answer warning */
    .no-answer {
        background: #fff8e1;
        border-left: 3px solid #f0a500;
        padding: 0.6rem 0.9rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("""
<div style="
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
">
    <h1 style="color: #FFFFFF;">⚖️ Legal Research Assistant ⚖️</h1>
    <p style="color: #CCCCCC;">
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# ---------------------------------------------------------------------------
# Sidebar — status + ingest trigger
# ---------------------------------------------------------------------------

with st.sidebar:
    # Health check
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        if health.get("index_loaded"):
            st.success(f"✅ Index ready ({health['total_chunks']:,} chunks)")
            st.session_state.index_ready = True
        else:
            st.warning("⚠️ Index not built yet")
            st.session_state.index_ready = False
    except Exception:
        st.error("❌ Backend not reachable\nMake sure FastAPI is running")
        st.session_state.index_ready = False


    # Ingest button
    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Ingesting PDFs… this takes a few minutes"):
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", timeout=600)
                if resp.status_code == 200:
                    st.success(resp.json()["message"])
                    st.session_state.index_ready = True
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Sources:**")
    st.markdown("• Bharatiya Nyaya Sanhita, 2023 (BNS)")
    st.markdown("• Bharatiya Nagarik Suraksha Sanhita, 2023 (BNSS)")
    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            st.markdown(
                f'<div class="citation-box"><strong>📎 Sources</strong>{msg["citations"]}</div>',
                unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input(
    "" if st.session_state.index_ready else "Build the index first using the sidebar →",
    disabled=not st.session_state.index_ready,
):
    if not prompt.strip():
        st.stop()
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query backend
    with st.chat_message("assistant"):
        with st.spinner("Searching legal texts…"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": prompt, "top_k": 5},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    citations = data["citations"]
                    has_answer = data["has_answer"]

                    st.markdown(answer)

                    if has_answer and citations:
                        st.markdown(
                            f'<div class="citation-box"><strong>📎 Sources</strong>{citations}</div>',
                            unsafe_allow_html=True
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations if has_answer else "",
                        "no_answer": not has_answer,
                    })

                else:
                    err = f"Backend error {resp.status_code}: {resp.text}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

            except requests.exceptions.Timeout:
                msg = "Request timed out. The model may be slow — try again."
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as e:
                msg = f"Error: {e}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
