import html
import time
from pathlib import Path

import streamlit as st

from runbook_ai.answer import CHAT_MODEL, answer_question
from runbook_ai.chunker import load_runbooks
from runbook_ai.ingest import ALLOWED, IngestError, ingest_bytes
from runbook_ai.search import EMBED_MODEL, Retriever

st.set_page_config(page_title="Runbook AI", page_icon="🛠️", layout="centered")

INDEX_DIR = "data/index"
UPLOAD_TYPES = [ext.lstrip(".") for ext in sorted(ALLOWED)]

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

st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_n", 0)
st.session_state.setdefault("notices", [])


@st.cache_resource(show_spinner="Loading the document library...")
def get_retriever() -> Retriever:
    retriever = Retriever.load(INDEX_DIR)
    if not Path(INDEX_DIR, "chunks.json").exists():   # first run: add sample guides
        retriever.add_chunks(load_runbooks("data/runbooks"))
        retriever.save(INDEX_DIR)
    return retriever


def render_assistant(msg: dict) -> None:
    """Draw one assistant message: badge, answer text, and sources."""
    if msg["grounded"]:
        st.markdown('<span class="badge badge-ok">Answered from your documents</span>',
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
                    f'<div class="src"><b>{html.escape(s["section"])}</b><br>'
                    f'<small>{html.escape(s["source"])} &middot; match {pct}%</small>'
                    f'<div class="bar"><div style="width:{pct}%"></div></div></div>',
                    unsafe_allow_html=True,
                )


# ---------- Header ----------
st.markdown(
    '<div class="hero"><h1>Runbook AI</h1>'
    "<p>An IT helpdesk assistant that answers only from your documents "
    "and shows its sources.</p></div>",
    unsafe_allow_html=True,
)

try:
    retriever = get_retriever()
except Exception:
    st.error("Could not load the document library. Delete the data/index folder "
             "and refresh. If that does not help, check that Ollama is running.")
    st.stop()

for kind, text in st.session_state.notices:
    (st.success if kind == "ok" else st.error)(text)
st.session_state.notices = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your documents")
    files = st.file_uploader(
        "Add PDF, DOCX, TXT or MD files (max 10 MB each)",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_n}",
    )
    if files and st.button("Index selected files", type="primary",
                           use_container_width=True):
        for f in files:
            started = time.perf_counter()
            try:
                with st.spinner(f"Reading and indexing {f.name}..."):
                    chunks = ingest_bytes(f.getvalue(), f.name)
                    source = chunks[0].source
                    retriever.replace_source(source, chunks)
                    retriever.save(INDEX_DIR)
                secs = time.perf_counter() - started
                st.session_state.notices.append(
                    ("ok", f"{source}: {len(chunks)} chunks indexed in {secs:.1f}s"))
            except IngestError as err:
                st.session_state.notices.append(("err", f"{f.name}: {err}"))
            except Exception:
                st.session_state.notices.append(
                    ("err", f"{f.name}: indexing failed. Is Ollama running?"))
        st.session_state.uploader_n += 1    # clears the uploader box
        st.rerun()

    library = retriever.sources()
    if library:
        for name, count in library.items():
            left, right = st.columns([5, 1])
            left.text(f"{name} ({count} chunks)")
            if right.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                retriever.remove_source(name)
                retriever.save(INDEX_DIR)
                st.rerun()
    else:
        st.caption("No documents yet. Upload one above.")

    st.divider()
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

# ---------- Chat ----------
has_documents = bool(retriever.sources())

if not has_documents:
    st.info("Upload a document in the sidebar to get started.")
elif not st.session_state.messages:
    st.info("Ask a question about your documents, or pick an example from the "
            "sidebar. If the documents don't contain the answer, I will say so "
            "instead of guessing.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            render_assistant(msg)

prompt = st.chat_input("Ask a question...", disabled=not has_documents)
if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        try:
            with st.spinner("Searching the documents and writing an answer..."):
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