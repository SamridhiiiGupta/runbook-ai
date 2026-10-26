import streamlit as st

from runbook_ai.answer import CHAT_MODEL, answer_question
from runbook_ai.chunker import load_runbooks
from runbook_ai.search import EMBED_MODEL, Retriever

st.set_page_config(page_title="Runbook AI", page_icon="🛠️", layout="centered")

CSS = """
<style>
.hero {padding: 1.3rem 1.5rem; border-radius: 18px; color: #fff; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, #312e81 0%, #0f766e 100%);}
.hero h1 {margin: 0; font-size: 1.9rem;}
.hero p {margin: .35rem 0 0; opacity: .9;}
.badge {display: inline-block; padding: 2px 12px; border-radius: 999px;
        font-size: .75rem; font-weight: 600; margin-bottom: .4rem;}
.badge-ok {background: #14532d; color: #bbf7d0;}
.badge-no {background: #7c2d12; color: #fed7aa;}
.src {border: 1px solid rgba(148,163,184,.3); border-radius: 12px;
      padding: .6rem .85rem; margin: .45rem 0;}
.src small {opacity: .7;}
.bar {height: 6px; border-radius: 99px; background: rgba(148,163,184,.25);
      overflow: hidden; margin-top: .4rem;}
.bar > div {height: 100%; background: #14b8a6;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

EXAMPLES = [
    "I forgot my login and cannot sign in",
    "My VPN shows error 809, what do I do?",
    "The printer says it is not found",
    "How long must my password be?",
]

AVATARS = {"user": "🧑‍💻", "assistant": "🛠️"}

@st.cache_resource(show_spinner="Indexing the IT guides on your laptop...")
def get_retriever() -> Retriever:
    return Retriever(load_runbooks("data/runbooks"))


def render_assistant(msg: dict) -> None:
    """Draw one assistant message: badge, answer text, and sources."""
    if msg["grounded"]:
        st.markdown('<span class="badge badge-ok">Answered from IT guides</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-no">No answer found</span>',
                    unsafe_allow_html=True)
    st.markdown(msg["text"])
    if msg["sources"]:
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                pct = max(0, min(100, int(s["score"] * 100)))
                st.markdown(
                    f'<div class="src"><b>{s["section"]}</b><br>'
                    f'<small>{s["source"]} &middot; match {pct}%</small>'
                    f'<div class="bar"><div style="width:{pct}%"></div></div></div>',
                    unsafe_allow_html=True,
                )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Try asking")
    for i, example in enumerate(EXAMPLES):
        if st.button(example, key=f"ex{i}", use_container_width=True):
            st.session_state.pending = example
            st.rerun()
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Chat model: {CHAT_MODEL}")
    st.caption(f"Embedding model: {EMBED_MODEL}")
    st.caption("Runs fully on this computer. Nothing is sent online.")

# ---------- Main page ----------
st.markdown(
    '<div class="hero"><h1>Runbook AI</h1>'
    "<p>An IT helpdesk assistant that answers only from your IT guides "
    "and shows its sources.</p></div>",
    unsafe_allow_html=True,
)

try:
    retriever = get_retriever()
except Exception:
    st.error("Could not reach the local AI engine. Is Ollama running? "
             "Open the Ollama app, then refresh this page.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.info("Ask about VPN, passwords or printers, or pick an example "
            "from the sidebar. If the guides don't contain the answer, "
            "I will say so instead of guessing.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_assistant(msg)

prompt = st.chat_input("Ask an IT question...")
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the guides and writing an answer..."):
                result = answer_question(prompt, retriever)
            msg = {
                "role": "assistant",
                "text": result.text,
                "grounded": result.grounded,
                "sources": [
                    {"source": s.source, "section": s.section, "score": s.score}
                    for s in result.sources
                ],
            }
            render_assistant(msg)
            st.session_state.messages.append(msg)
        except Exception:
            st.error("The local model did not respond. Check that Ollama "
                     "is running, then try again.")