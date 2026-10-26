from html import escape
from pathlib import Path
import time

import streamlit as st

from runbook_ai.agent import stream_agent
from runbook_ai.answer import CHAT_MODEL
from runbook_ai.chunker import load_runbooks
from runbook_ai.conversations import ConversationStore
from runbook_ai.ingest import ALLOWED, IngestError, ingest_bytes
from runbook_ai.runtime import OllamaStatus, inspect_ollama
from runbook_ai.search import EMBED_MODEL, Retriever

st.set_page_config(page_title="Runbook AI", page_icon="✳", layout="wide", initial_sidebar_state="expanded")

INDEX_DIR = Path("data/index")
CHAT_DB = Path("data/conversations.sqlite3")
UPLOAD_TYPES = [ext.lstrip(".") for ext in sorted(ALLOWED)]
SUGGESTIONS = [
    "My VPN shows error 809. What should I try?",
    "What are the rules for my work password?",
    "The printer is not showing up on my laptop.",
]

CSS = """
<style>
:root {
  --ink: #f3f5ff; --muted: #a5afc8; --faint: #697591;
  --bg: #070b13; --panel: #0d1422; --panel-2: #111c2d;
  --line: rgba(154, 180, 224, .16); --cyan: #79e6ef; --blue: #8299ff;
  --violet: #b398ff; --green: #9ae7c7; --amber: #f0c77f;
}
.stApp {
  background:
    radial-gradient(ellipse at 70% -8%, rgba(63, 86, 156, .22), transparent 38rem),
    radial-gradient(ellipse at 4% 46%, rgba(19, 98, 119, .10), transparent 30rem),
    var(--bg);
  color: var(--ink);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(14, 21, 35, .98), rgba(8, 13, 23, .98));
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
.block-container { max-width: 1460px; padding-top: 2.1rem; padding-bottom: 7rem; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { letter-spacing: -.02em; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] small { color: var(--muted); }
.stButton > button {
  min-height: 2.65rem; border: 1px solid var(--line); border-radius: 12px;
  background: linear-gradient(180deg, rgba(21, 33, 53, .92), rgba(13, 22, 37, .95));
  color: var(--ink); transition: border-color .18s ease, transform .18s ease, background .18s ease;
}
.stButton > button:hover { border-color: rgba(121, 230, 239, .58); color: #fff; transform: translateY(-1px); }
.stButton > button:focus-visible { outline: 2px solid var(--cyan); outline-offset: 2px; }
.stButton > button[kind="primary"] {
  border-color: rgba(121, 230, 239, .42);
  background: linear-gradient(120deg, rgba(42, 120, 144, .92), rgba(75, 83, 151, .94));
}
[data-testid="stChatMessage"] {
  border: 1px solid rgba(154, 180, 224, .12); border-radius: 18px;
  background: rgba(13, 20, 34, .64); padding: 1.15rem 1.3rem;
}
[data-testid="stChatInput"] {
  border: 1px solid rgba(121, 230, 239, .28); border-radius: 18px;
  background: rgba(11, 18, 30, .96); box-shadow: 0 14px 50px rgba(0, 0, 0, .25);
}
[data-testid="stChatInput"] textarea { color: var(--ink); }
[data-testid="stFileUploader"] section { border: 1px dashed rgba(121, 230, 239, .34); border-radius: 14px; background: rgba(13, 24, 39, .52); }
[data-testid="stExpander"] { border-color: var(--line); border-radius: 13px; background: rgba(14, 23, 38, .48); }
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: var(--line); }
.brand-lockup { display:flex; align-items:center; gap:.68rem; margin:.1rem 0 1.35rem; }
.brand-mark { width:2.3rem; height:2.3rem; border-radius:11px; display:grid; place-items:center;
  color:#09111b; font-size:1.25rem; font-weight:800;
  background:linear-gradient(145deg,var(--cyan),var(--blue) 66%,var(--violet));
  box-shadow:0 0 28px rgba(121,230,239,.22); }
.brand-name { font-size:1rem; letter-spacing:.08em; font-weight:750; color:#f4f7ff; }
.brand-caption { display:block; margin-top:.1rem; font-size:.67rem; letter-spacing:.16em; color:var(--faint); text-transform:uppercase; }
.topline { display:flex; justify-content:space-between; align-items:center; margin:0 0 2.4rem; }
.eyebrow { color:var(--cyan); font-size:.7rem; font-weight:750; letter-spacing:.2em; text-transform:uppercase; }
.connection { display:inline-flex; align-items:center; gap:.55rem; border:1px solid var(--line); border-radius:99px;
  padding:.45rem .78rem; color:var(--muted); background:rgba(13,20,34,.68); font-size:.76rem; }
.connection-dot { width:.48rem; height:.48rem; border-radius:50%; background:var(--green); box-shadow:0 0 12px currentColor; }
.connection-dot.offline { background:#eb9b89; }
.welcome { position:relative; overflow:hidden; min-height:23rem; display:grid; grid-template-columns:minmax(18rem, .95fr) minmax(23rem, 1.05fr); align-items:center;
  padding:2.5rem 3.4rem; border:1px solid rgba(144,170,221,.2); border-radius:26px;
  background:linear-gradient(112deg,rgba(16,27,45,.94),rgba(10,17,29,.82) 53%,rgba(21,26,48,.86));
  box-shadow:0 24px 90px rgba(0,0,0,.24), inset 0 1px rgba(255,255,255,.035); }
.welcome::before { content:""; position:absolute; inset:-35%; pointer-events:none;
  background:radial-gradient(ellipse at 70% 45%,rgba(70,117,171,.16),transparent 40%); }
.welcome-copy { position:relative; z-index:2; max-width:36rem; }
.welcome h1 { margin:.7rem 0 1rem; font-size:clamp(2.3rem,4.1vw,4.05rem); line-height:1.04; letter-spacing:-.065em; color:#f5f7ff; }
.welcome h1 span { color:#a9b8ff; }
.welcome-copy p { max-width:34rem; color:#aeb9d0; font-size:1.02rem; line-height:1.75; }
.welcome-meta { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.35rem; }
.meta-pill { border:1px solid var(--line); border-radius:99px; padding:.42rem .7rem; color:#b8c5dc; font-size:.73rem; background:rgba(7,13,23,.45); }
.core-stage { position:relative; min-height:19rem; display:grid; place-items:center; isolation:isolate; }
.core-glow { position:absolute; width:15rem; height:15rem; border-radius:50%; filter:blur(25px); opacity:.42;
  background:radial-gradient(circle,rgba(78,203,227,.43),rgba(97,113,238,.12) 48%,transparent 72%); }
.core { position:relative; width:12.2rem; height:12.2rem; border-radius:50%; display:grid; place-items:center;
  background:radial-gradient(circle at 38% 32%,rgba(199,255,255,.96) 0 1%,rgba(113,228,241,.74) 5%,rgba(81,129,219,.6) 18%,rgba(45,57,112,.4) 43%,rgba(10,15,31,.08) 68%,transparent 72%);
  box-shadow:inset -18px -25px 35px rgba(9,13,31,.68), inset 10px 10px 27px rgba(183,248,255,.27), 0 0 65px rgba(97,169,245,.26);
  animation:core-breathe 6s ease-in-out infinite; }
.core::before,.core::after { content:""; position:absolute; border:1px solid rgba(140,222,255,.54); border-radius:50%; inset:-1.2rem; transform:rotateX(68deg) rotateZ(-24deg); box-shadow:0 0 24px rgba(111,204,250,.14); }
.core::after { inset:-2.1rem; border-color:rgba(174,153,255,.44); transform:rotateX(70deg) rotateZ(48deg); animation:orbit 18s linear infinite; }
.core-orbit { position:absolute; width:17rem; height:17rem; border:1px solid rgba(107,163,224,.18); border-radius:50%; transform:rotateX(67deg) rotateZ(-9deg); }
.core-orbit::before { content:""; position:absolute; top:13%; left:22%; width:.38rem; height:.38rem; border-radius:50%; background:#b1f6ff; box-shadow:0 0 16px #79e6ef; }
.core-caption { position:absolute; bottom:.15rem; color:#8291ad; font-size:.66rem; letter-spacing:.21em; text-transform:uppercase; }
@keyframes core-breathe { 50% { transform:scale(1.035); filter:hue-rotate(8deg); } }
@keyframes orbit { to { transform:rotateX(70deg) rotateZ(408deg); } }
.section-head { display:flex; justify-content:space-between; align-items:end; gap:1rem; margin:2.15rem 0 .85rem; }
.section-head h3 { margin:0; font-size:1rem; color:#e9efff; letter-spacing:-.02em; }
.section-head p { margin:0; color:var(--faint); font-size:.78rem; }
.knowledge-node { min-height:5.2rem; border:1px solid var(--line); border-radius:15px; padding:.85rem .95rem;
  background:linear-gradient(140deg,rgba(16,28,46,.9),rgba(10,17,29,.88)); }
.node-kicker { color:var(--cyan); font-size:.65rem; letter-spacing:.15em; text-transform:uppercase; }
.node-name { margin-top:.43rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#e1e8f8; font-size:.82rem; }
.node-meta { margin-top:.18rem; color:var(--faint); font-size:.7rem; }
.source-card { border:1px solid var(--line); border-radius:13px; padding:.75rem .9rem; margin:.45rem 0; background:rgba(13,22,37,.72); }
.source-ref { display:inline-grid; place-items:center; min-width:1.8rem; height:1.6rem; margin-right:.45rem; border-radius:7px; color:#08111b; background:var(--cyan); font-weight:800; font-size:.7rem; }
.trace-line { display:flex; gap:.65rem; align-items:flex-start; color:#bbc6dc; padding:.35rem 0; font-size:.83rem; }
.trace-mark { color:var(--green); }
.trace-mark.warning { color:var(--amber); }
.trace-mark.error { color:#ef958a; }
.empty-library { border:1px dashed var(--line); border-radius:14px; padding:1rem; color:var(--muted); font-size:.8rem; }
@media (max-width: 900px) {
  .block-container { padding-top:1.2rem; }
  .welcome { grid-template-columns:1fr; padding:2rem 1.4rem; }
  .core-stage { min-height:15rem; }
  .core { width:9.5rem; height:9.5rem; }
}
@media (max-width: 640px) {
  .topline { margin-bottom:1.4rem; align-items:flex-start; }
  .welcome { min-height:auto; }
  .core-stage { min-height:13rem; }
  .section-head { align-items:flex-start; flex-direction:column; }
}
@media (prefers-reduced-motion: reduce) {
  *,*::before,*::after { scroll-behavior:auto !important; animation-duration:.01ms !important; animation-iteration-count:1 !important; transition-duration:.01ms !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=8, show_spinner=False)
def local_status() -> OllamaStatus:
    return inspect_ollama()


@st.cache_resource(show_spinner="Opening the local knowledge index…")
def get_retriever(embedding_ready: bool):
    try:
        retriever = Retriever.load(INDEX_DIR)
    except Exception as exc:
        return Retriever(), f"Saved index could not be read ({exc.__class__.__name__})."
    if not (INDEX_DIR / "chunks.json").exists() and embedding_ready:
        try:
            retriever.add_chunks(load_runbooks("data/runbooks"))
            retriever.save(INDEX_DIR)
        except Exception as exc:
            return Retriever(), f"Starter guides could not be indexed ({exc.__class__.__name__})."
    return retriever, None


def render_brand():
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">✳</div>'
        '<div><span class="brand-name">RUNBOOK AI</span>'
        '<span class="brand-caption">Local knowledge system</span></div></div>',
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Source trail · {len(sources)} cited passage{'s' if len(sources) != 1 else ''}"):
        st.caption("Similarity helps rank passages. It is not a confidence or correctness score.")
        for source in sources:
            page = source.get("page")
            location = source.get("section") or "Document"
            if page:
                location = f"Page {page} · {location}"
            citation_index = int(source.get("citation_index", 0))
            st.markdown(
                f'<div class="source-card"><span class="source-ref">[{citation_index}]</span>'
                f'<strong>{escape(source.get("source", "Unknown source"))}</strong>'
                f'<div class="node-meta">{escape(location)}</div></div>',
                unsafe_allow_html=True,
            )
            st.text(source.get("excerpt", ""))


def render_trace(trace: list[dict]):
    if not trace:
        return
    with st.expander(f"Agent activity · {len(trace)} events"):
        for event in trace:
            status = event.get("status", "complete")
            icon = "!" if status == "warning" else "×" if status == "error" else "✓"
            klass = "warning" if status == "warning" else "error" if status == "error" else ""
            suffix = f" · {event['duration_ms']} ms" if event.get("duration_ms") is not None else ""
            st.markdown(
                f'<div class="trace-line"><span class="trace-mark {klass}">{icon}</span>'
                f'<span>{escape(event.get("label", "Activity"))}{suffix}</span></div>',
                unsafe_allow_html=True,
            )


def render_assistant(message: dict):
    kind = message.get("kind", "grounded")
    if kind == "grounded":
        st.markdown('<div class="eyebrow">Grounded answer</div>', unsafe_allow_html=True)
    elif kind == "no_answer":
        st.markdown('<div class="eyebrow" style="color:#f0c77f">Outside the indexed knowledge</div>', unsafe_allow_html=True)
    elif kind == "tool":
        st.markdown('<div class="eyebrow">Local tool result</div>', unsafe_allow_html=True)
    elif kind == "error":
        st.markdown('<div class="eyebrow" style="color:#ef958a">Local model unavailable</div>', unsafe_allow_html=True)
    st.markdown(message.get("content", ""))
    if message.get("latency_s") is not None:
        st.caption(f"Local response · {message['latency_s']:.1f}s")
    render_sources(message.get("sources", []))
    render_trace(message.get("trace", []))


def ensure_active_conversation(store: ConversationStore):
    rows = store.list_conversations()
    if not rows:
        active = store.create()
        return active, store.list_conversations()
    ids = {row["id"] for row in rows}
    active = st.session_state.get("active_conversation_id")
    if active not in ids:
        active = rows[0]["id"]
        st.session_state["active_conversation_id"] = active
    return active, rows


status = local_status()
retriever, index_error = get_retriever(status.embedding_ready)
store = ConversationStore(CHAT_DB)
active_id, conversations = ensure_active_conversation(store)

# ---------- Knowledge vault + conversation navigation ----------
with st.sidebar:
    render_brand()
    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        st.session_state.active_conversation_id = store.create()
        st.rerun()

    st.markdown("### Knowledge vault")
    uploaded = st.file_uploader(
        "Drop a guide here or browse files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.get('uploader_key', 0)}",
        help="PDF, DOCX, TXT, and Markdown. Up to 10 MB per file; image-only scanned PDFs need OCR, which is not included.",
    )
    if not status.online:
        st.warning("Start Ollama locally to index documents and chat.")
    elif not status.embedding_ready:
        st.warning(f"Install the embedding model: `ollama pull {EMBED_MODEL}`")
    if st.button("Index selected files", use_container_width=True, disabled=not uploaded or not status.embedding_ready):
        for file in uploaded:
            started = time.perf_counter()
            try:
                with st.spinner(f"Reading and indexing {escape(file.name)}…"):
                    chunks = ingest_bytes(file.getvalue(), file.name)
                    retriever.replace_source(chunks[0].source, chunks)
                    retriever.save(INDEX_DIR)
                elapsed = time.perf_counter() - started
                st.session_state.setdefault("notices", []).append(
                    ("success", f"{chunks[0].source} · {len(chunks)} passages · {elapsed:.1f}s")
                )
            except IngestError as exc:
                st.session_state.setdefault("notices", []).append(("error", f"{escape(file.name)}: {exc}"))
            except Exception:
                st.session_state.setdefault("notices", []).append(
                    ("error", f"{escape(file.name)}: indexing failed. Check Ollama and the embedding model.")
                )
        st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
        st.cache_resource.clear()
        st.rerun()

    if index_error:
        st.error(index_error)
    library = retriever.sources()
    if library:
        for name, count in sorted(library.items(), key=lambda item: item[0].casefold()):
            left, right = st.columns([5, 1])
            left.markdown(f"**{escape(name)}**  \n<span class='node-meta'>{count} indexed passages</span>", unsafe_allow_html=True)
            if right.button("×", key=f"remove_{name}", help=f"Remove {name}"):
                retriever.remove_source(name)
                retriever.save(INDEX_DIR)
                st.cache_resource.clear()
                st.rerun()
    else:
        st.markdown('<div class="empty-library">Your knowledge vault is empty. Add a guide to get started.</div>', unsafe_allow_html=True)
    st.caption("PDF · DOCX · TXT · MD  /  10 MB each")

    st.divider()
    st.markdown("### Recent conversations")
    for row in conversations[:8]:
        title = row["title"] or "New conversation"
        label = ("◉  " if row["id"] == active_id else "○  ") + title[:34]
        if st.button(label, key=f"conversation_{row['id']}", use_container_width=True, help=title):
            st.session_state.active_conversation_id = row["id"]
            st.rerun()
    if st.button("Clear this conversation", use_container_width=True, disabled=not store.get_messages(active_id)):
        store.clear(active_id)
        st.rerun()

    st.divider()
    if status.ready:
        st.markdown('<div class="connection"><span class="connection-dot"></span>Local models ready</div>', unsafe_allow_html=True)
    elif status.online:
        st.caption("Ollama is online. Install both required models to enable all features.")
    else:
        st.markdown('<div class="connection"><span class="connection-dot offline"></span>Ollama offline</div>', unsafe_allow_html=True)
    st.caption(f"Chat · {CHAT_MODEL}")
    st.caption(f"Embeddings · {EMBED_MODEL}")
    st.caption("Files and chats are stored on this computer. Prompts go to the Ollama host configured for this installation.")

# ---------- Main workspace ----------
header_left, header_right = st.columns([1, 1])
with header_left:
    st.markdown('<div class="eyebrow">A grounded helpdesk intelligence</div>', unsafe_allow_html=True)
with header_right:
    if status.ready:
        st.markdown('<div style="text-align:right"><span class="connection"><span class="connection-dot"></span>Local inference active</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:right"><span class="connection"><span class="connection-dot offline"></span>Waiting for local models</span></div>', unsafe_allow_html=True)

if "notices" in st.session_state:
    for kind, text in st.session_state.pop("notices"):
        (st.success if kind == "success" else st.error)(text)

if not status.online:
    st.warning("The workspace is ready. Start Ollama, then install the chat and embedding models shown in the vault sidebar.")
elif not status.chat_ready:
    st.warning(f"Install the chat model with `ollama pull {CHAT_MODEL}` to enable responses.")
if index_error:
    st.error("The saved knowledge index could not be opened. Your original index files were left in place; review the error in the sidebar before re-indexing.")

messages = store.get_messages(active_id)
pending_prompt = None
if not messages:
    st.markdown(
        '<section class="welcome"><div class="welcome-copy">'
        '<div class="eyebrow">Your knowledge, in conversation</div>'
        '<h1>Find the signal.<br><span>Follow the source.</span></h1>'
        '<p>Ask about your IT guides. Runbook AI searches the passages you have indexed, '
        'shows the evidence behind its answers, and says when your documents do not contain enough information.</p>'
        '<div class="welcome-meta"><span class="meta-pill">Local-first inference</span>'
        '<span class="meta-pill">Answers with citations</span><span class="meta-pill">Bounded tools</span></div>'
        '</div><div class="core-stage"><div class="core-glow"></div><div class="core-orbit"></div>'
        '<div class="core"></div><div class="core-caption">Knowledge constellation · live</div></div></section>',
        unsafe_allow_html=True,
    )
    if library:
        st.markdown('<div class="section-head"><div><h3>Knowledge constellation</h3><p>Your indexed sources, ready to explore</p></div><span class="eyebrow">%d sources</span></div>' % len(library), unsafe_allow_html=True)
        cards = st.columns(min(3, len(library)))
        for index, (name, count) in enumerate(sorted(library.items())[:3]):
            with cards[index % len(cards)]:
                st.markdown(
                    f'<div class="knowledge-node"><div class="node-kicker">Source node {index + 1:02d}</div>'
                    f'<div class="node-name">{escape(name)}</div><div class="node-meta">{count} passages indexed</div></div>',
                    unsafe_allow_html=True,
                )
    else:
        st.markdown('<div class="section-head"><div><h3>Bring your first source into the constellation</h3><p>Use the vault in the left panel to add a PDF, DOCX, TXT, or Markdown guide.</p></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-head"><div><h3>Start with a question</h3><p>Try one of these once a guide is indexed.</p></div></div>', unsafe_allow_html=True)
    prompt_cols = st.columns(3)
    for index, suggestion in enumerate(SUGGESTIONS):
        with prompt_cols[index]:
            if st.button(suggestion, key=f"suggestion_{index}", use_container_width=True, disabled=not library or not status.ready):
                pending_prompt = suggestion
else:
    title = next((row["title"] for row in conversations if row["id"] == active_id), "Conversation")
    st.markdown(f"<div class='section-head'><div><h3>{escape(title)}</h3><p>Conversation memory is stored locally on this computer.</p></div></div>", unsafe_allow_html=True)
    for message in messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_assistant(message)

if not library:
    chat_disabled_reason = "Index a guide to start asking questions."
elif not status.online:
    chat_disabled_reason = "Start Ollama to use local inference."
elif not status.chat_ready:
    chat_disabled_reason = f"Install {CHAT_MODEL} to start chatting."
else:
    chat_disabled_reason = "Ask about your indexed IT guides…"

submitted = st.chat_input(chat_disabled_reason, disabled=not library or not status.ready, max_chars=1000)
prompt = submitted or pending_prompt
if prompt:
    store.add_message(active_id, "user", prompt)
    # Show the current turn immediately; the next rerun will render it from SQLite.
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        answer_area = st.empty()
        live_text = ""
        result = None
        with st.status("Thinking with your local knowledge…", expanded=True) as activity:
            try:
                for event in stream_agent(prompt, retriever, history=messages):
                    if event["type"] == "trace":
                        label = event["event"].get("label", "Working…")
                        st.write(label)
                    elif event["type"] == "token":
                        live_text += event["text"]
                        answer_area.markdown(live_text + "▍")
                    elif event["type"] == "clear_answer":
                        live_text = ""
                        answer_area.empty()
                    elif event["type"] == "complete":
                        result = event["result"]
                activity.update(label="Response ready", state="complete", expanded=False)
            except Exception:
                activity.update(label="Local model request failed", state="error", expanded=True)
                result = {
                    "text": "I couldn't reach the local model. Check that Ollama is running and that the required models are installed, then send your question again.",
                    "grounded": False, "sources": [],
                    "trace": [{"label": "The local model request did not complete", "status": "error"}],
                    "tools_used": [], "latency_s": None, "kind": "error",
                }
        if result is None:
            result = {
                "text": "I couldn't complete that request. Please try again.",
                "grounded": False, "sources": [], "trace": [],
                "tools_used": [], "latency_s": None, "kind": "error",
            }
        answer_area.empty()
        render_assistant({
            "role": "assistant", "content": result["text"], "kind": result.get("kind"),
            "sources": result.get("sources", []), "trace": result.get("trace", []),
            "latency_s": result.get("latency_s"),
        })
    store.add_message(
        active_id, "assistant", result["text"], kind=result.get("kind", "grounded"),
        grounded=result.get("grounded", False), sources=result.get("sources", []),
        trace=result.get("trace", []), tools_used=result.get("tools_used", []),
        latency_s=result.get("latency_s"),
    )
    st.rerun()
