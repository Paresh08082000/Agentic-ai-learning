import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)

# ======================== Page Config ========================
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================== Custom CSS =========================
st.markdown("""
<style>
/* ── Sidebar background ──────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 60%, #24243e 100%);
}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
    padding-top: 1.5rem;
}

/* ── All sidebar text ────────────────────────────────── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] div {
    color: #dde1f5 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}

/* ── Sidebar buttons (conversation history) ─────────── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255, 255, 255, 0.07) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 10px !important;
    color: #dde1f5 !important;
    text-align: left !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 0.8rem !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99, 102, 241, 0.3) !important;
    border-color: #6366f1 !important;
    transform: translateX(3px);
}

/* ── Main content area ───────────────────────────────── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 1rem;
    max-width: 900px;
}

/* ── Chat input box ──────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    border-radius: 14px !important;
    font-size: 0.95rem !important;
}

/* ── Chat messages ───────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.25rem;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: rgba(99, 102, 241, 0.2) !important; }
</style>
""", unsafe_allow_html=True)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id, title="New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"][thread_id] = title


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = list(st.session_state["chat_threads"].items())
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.markdown("""
<div style='text-align:center; padding: 0.5rem 0 1.2rem 0;'>
    <div style='font-size: 2.2rem;'>🤖</div>
    <div style='font-size: 1.2rem; font-weight: 700; color: #ffffff; letter-spacing: 0.5px;'>
        AI Chatbot
    </div>
    <div style='font-size: 0.75rem; color: #9ca3af; margin-top: 2px;'>
        Powered by LangGraph + Groq
    </div>
</div>
""", unsafe_allow_html=True)

# New Chat button with gradient styling
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] .stButton:first-of-type > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.4px !important;
    color: #ffffff !important;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] .stButton:first-of-type > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.5) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

if st.sidebar.button("+ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# PDF section
st.sidebar.markdown("---")
st.sidebar.markdown("**📄 Document**")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"✅ `{latest_doc.get('filename')}`  \n"
        f"{latest_doc.get('chunks')} chunks · {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF uploaded yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF Indexed", state="complete", expanded=False)

# Past conversations
st.sidebar.markdown("---")
st.sidebar.markdown("**🗂 Past Conversations**")
if not threads:
    st.sidebar.caption("No conversations yet.")
else:
    for thread_id, title in threads:
        label = f"💬 {title}" if title != "New Chat" else "💬 New Chat"
        if st.sidebar.button(label, key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.markdown("""
<div style='margin-bottom: 1rem;'>
    <h1 style='font-size: 1.9rem; font-weight: 700; margin-bottom: 0.1rem;'>
        🤖 Multi Utility Chatbot
    </h1>
    <p style='color: #6b7280; font-size: 0.9rem; margin-top: 0;'>
        Ask anything · Upload PDFs · Web search · Stock prices · Calculator
    </p>
</div>
""", unsafe_allow_html=True)

# Welcome screen when no messages
if not st.session_state["message_history"]:
    st.markdown("""
    <div style='display: flex; gap: 1rem; flex-wrap: wrap; margin: 2rem 0;'>
        <div style='flex:1; min-width:180px; background: linear-gradient(135deg,#f0f4ff,#e8eeff);
                    border-radius:14px; padding:1.2rem; border-left: 4px solid #6366f1;'>
            <div style='font-size:1.5rem;'>📄</div>
            <div style='font-weight:600; margin: 0.3rem 0;'>PDF Chat</div>
            <div style='font-size:0.8rem; color:#6b7280;'>Upload a PDF and ask questions about its content</div>
        </div>
        <div style='flex:1; min-width:180px; background: linear-gradient(135deg,#f0fff4,#e8fef0);
                    border-radius:14px; padding:1.2rem; border-left: 4px solid #10b981;'>
            <div style='font-size:1.5rem;'>🌐</div>
            <div style='font-weight:600; margin: 0.3rem 0;'>Web Search</div>
            <div style='font-size:0.8rem; color:#6b7280;'>Get real-time info from the internet</div>
        </div>
        <div style='flex:1; min-width:180px; background: linear-gradient(135deg,#fffbf0,#fef9e8);
                    border-radius:14px; padding:1.2rem; border-left: 4px solid #f59e0b;'>
            <div style='font-size:1.5rem;'>📈</div>
            <div style='font-weight:600; margin: 0.3rem 0;'>Stock Prices</div>
            <div style='font-size:0.8rem; color:#6b7280;'>Look up live stock prices by symbol</div>
        </div>
        <div style='flex:1; min-width:180px; background: linear-gradient(135deg,#fff0f0,#fee8e8);
                    border-radius:14px; padding:1.2rem; border-left: 4px solid #ef4444;'>
            <div style='font-size:1.5rem;'>🧮</div>
            <div style='font-weight:600; margin: 0.3rem 0;'>Calculator</div>
            <div style='font-size:0.8rem; color:#6b7280;'>Perform arithmetic operations instantly</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document, search the web, check stocks…")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    # Update thread title to first user message if still default
    if st.session_state["chat_threads"].get(st.session_state["thread_id"]) == "New Chat":
        st.session_state["chat_threads"][st.session_state["thread_id"]] = user_input[:50]

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"📄 {doc_meta.get('filename')} · "
            f"{doc_meta.get('chunks')} chunks · {doc_meta.get('documents')} pages"
        )

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        temp_messages.append({"role": role, "content": msg.content})
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
