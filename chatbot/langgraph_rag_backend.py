from __future__ import annotations

import os
import re
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

_yt_api = YouTubeTranscriptApi()
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    model_kwargs={"parallel_tool_calls": False}
)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
# Tool 1 — Web Search
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for current/recent information about a topic."""
    results = search.run(query)
    return results


@tool
def get_weather(city: str) -> dict:
    """Get current weather conditions for any city."""
    try:
        r = requests.get(f"https://wttr.in/{city}?format=j1", timeout=10)
        r.raise_for_status()
        data = r.json()
        current = data["current_condition"][0]
        area = data["nearest_area"][0]
        return {
            "city": city,
            "area": area["areaName"][0]["value"],
            "country": area["country"][0]["value"],
            "temp_c": current["temp_C"],
            "temp_f": current["temp_F"],
            "feels_like_c": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "description": current["weatherDesc"][0]["value"],
            "wind_kmph": current["windspeedKmph"],
            "visibility_km": current["visibility"],
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_youtube_transcript(url: str) -> dict:
    """
    Fetch the transcript of a YouTube video from its URL.
    Use this to summarize, answer questions about, or analyse YouTube videos.
    """
    try:
        match = re.search(
            r"(?:v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url
        )
        if not match:
            return {"error": "Could not extract a valid YouTube video ID from the URL."}

        video_id = match.group(1)
        transcript_list = _yt_api.fetch(video_id)
        full_text = " ".join(entry.text for entry in transcript_list)

        # Cap at ~8 000 chars to stay within context limits
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "… [transcript truncated]"

        return {"video_id": video_id, "transcript": full_text}
    except TranscriptsDisabled:
        return {"error": "This video has captions/transcripts disabled by the uploader."}
    except NoTranscriptFound:
        return {"error": "No transcript was found for this video (it may not have captions)."}
    except VideoUnavailable:
        return {"error": "This video is unavailable (it may be private, deleted, or region-locked)."}
    except Exception as e:
        return {"error": f"Could not fetch transcript: {e}"}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [web_search, get_stock_price, get_weather, get_youtube_transcript, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, weather, "
            "and YouTube transcript tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    try:
        response = llm_with_tools.invoke(messages, config=config)
    except Exception as e:
        # Groq sometimes rejects malformed tool calls — retry without tools
        if any(s in str(e) for s in ("Failed to call a function", "tool_use_failed", "tool call validation failed")):
            response = llm.invoke(messages, config=config)
        else:
            raise
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads() -> dict:
    thread_timestamps: Dict[str, str] = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        ts = checkpoint.checkpoint.get("ts", "")
        if thread_id not in thread_timestamps or ts > thread_timestamps[thread_id]:
            thread_timestamps[thread_id] = ts

    # Most recently active thread first
    sorted_ids = sorted(thread_timestamps, key=lambda t: thread_timestamps[t], reverse=True)

    all_threads = {}
    for thread_id in sorted_ids:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = msg.content[:50]
                break
        all_threads[thread_id] = title

    return all_threads


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})