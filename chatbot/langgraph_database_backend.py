from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq 
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",        # ← Free Groq model
    api_key=os.environ.get("GROQ_API_KEY")
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = {}
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in all_threads:
            all_threads[thread_id] = None

    for thread_id in all_threads:
        state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
        messages = state.values.get('messages', [])
        title = "New Chat"
        for msg in messages:
            if isinstance(msg, HumanMessage):
                title = msg.content[:50]
                break
        all_threads[thread_id] = title

    return all_threads
