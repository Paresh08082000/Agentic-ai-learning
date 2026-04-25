# backend.py

import os

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ.get("GROQ_API_KEY"),
    model_kwargs={"parallel_tool_calls": False}
)

# ─────────────────────────────────────────
# 2. VECTOR STORE SETUP (RAG)
# ─────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"   # free, runs locally
)

# Add your own documents here
documents = [
    "LangGraph is a framework for building stateful agent workflows.",
    "Groq provides fast LLM inference using LPU hardware.",
    "RAG stands for Retrieval Augmented Generation.",
    "Vector databases store embeddings for semantic search.",
    "Agentic AI systems can plan, use tools, and remember context.",
]

vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="my_knowledge_base"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

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
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

# Tool 2 — RAG Search
@tool
def rag_search(query: str) -> str:
    """Search internal knowledge base for information about AI concepts."""
    docs = retriever.invoke(query)
    results = "\n".join([doc.page_content for doc in docs])
    return results if results else "No relevant documents found."



tools = [web_search, get_stock_price, calculator, rag_search]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a helpful assistant with access to tools: web_search, calculator, "
    "get_stock_price, and rag_search. "
    "When you need to call a tool, call exactly ONE tool at a time with valid JSON arguments. "
    "Do not include empty brackets [] in tool calls. "
    "Always pass required arguments as a proper JSON object."
))

def agent_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SYSTEM_PROMPT] + list(messages)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("agent_node", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent_node")

graph.add_conditional_edges("agent_node",tools_condition)
graph.add_edge('tools', 'agent_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in all_threads:
            all_threads[thread_id] = None

    for thread_id in all_threads:
        state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = msg.content[:50]
                break
        all_threads[thread_id] = title

    return all_threads