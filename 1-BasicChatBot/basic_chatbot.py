"""
Basic Chatbot Script
Converted from BasicChatbot.ipynb
Now supports interactive terminal chat.
"""

# ========== Imports and Setup ==========
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
import pprint

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch


# ========== Load Environment Variables ==========
load_dotenv()

# ========== State Definition ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ========== Basic Chatbot ==========
llm = ChatGroq(model="gemma2-9b-it")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

def build_basic_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("llmchatbot", chatbot)
    graph_builder.add_edge(START, "llmchatbot")
    graph_builder.add_edge("llmchatbot", END)
    return graph_builder.compile()

# ========== Chatbot with Tools (Tavily and Custom) ==========
try:
    from langchain_tavily import TavilySearch
except ImportError:
    TavilySearch = None
    print("langchain_tavily not installed. Tool functionality will be limited.")

def multiply(a: int, b: int) -> int:
    """Multiply a and b"""
    return a * b

def build_tools_graph():
    if TavilySearch is None:
        return None
    tool = TavilySearch(max_results=2)
    tools = [tool, multiply]
    llm_with_tools = llm.bind_tools(tools)
    def tool_calling_llm(state: State):
        MAX_HISTORY = 20
        messages = state["messages"][-MAX_HISTORY:]
        result = llm_with_tools.invoke(messages)
        return {"messages": [result]}
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", END)
    return builder.compile()

## Custom function
def multiply(a:int,b:int)->int:
    """Multiply a and b

    Args:
        a (int): first int
        b (int): second int

    Returns:
        int: output int
    """
    return a*b

# ========== ReAct Agent Architecture ==========
def build_react_graph():
    if TavilySearch is None:
        return None
    tool = TavilySearch(max_results=2)
    tools = [tool, multiply]
    llm_with_tool = llm.bind_tools(tools)
    def tool_calling_llm(state: State):
        MAX_HISTORY = 20
        messages = state["messages"][-MAX_HISTORY:]
        result = llm_with_tool.invoke(messages)
        return {"messages": [result]}
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    return builder.compile()

# ========== Memory in Agentic Graph ==========
def build_memory_graph():
    if TavilySearch is None:
        return None
    tool = TavilySearch(max_results=2)
    tools = [tool, multiply]
    llm_with_tool = llm.bind_tools(tools)
    memory = MemorySaver()
    pp = pprint.PrettyPrinter(indent=2)
    def tool_calling_llm(state: State):
        MAX_HISTORY = 20
        messages = state["messages"][-MAX_HISTORY:]
        # Add a system prompt to guide the LLM's behavior
        if not messages or messages[0].type != "system":
            messages = [
                SystemMessage(
                    content=(
                        "You are a helpful assistant. Your primary task is to answer user questions. "
                        "When you use the Tavily search tool, you MUST summarize the search results to provide a direct and concise answer to the user's query. "
                        "Do not just list the links or tell the user to look at the search results. Synthesize the information from the search results into a coherent response. "
                        "For example, if the user asks for the latest news, provide a summary of the top headlines from the search results."
                    )
                )
            ] + messages
        print("[DEBUG] Calling LLM with messages:")
        for m in messages:
            print(f"  - {getattr(m, 'type', str(type(m)))}: {getattr(m, 'content', m)}")
        result = llm_with_tool.invoke(messages)
        print("[DEBUG] LLM response:", getattr(result, 'content', result))
        response = {"messages": [result]}
        print("[DEBUG] Memory graph response:")
        pp.pprint(response)
        return response
    builder = StateGraph(State)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges("tool_calling_llm", tools_condition)
    builder.add_edge("tools", "tool_calling_llm")
    return builder.compile(checkpointer=memory)

# ========== Streaming Example ==========
def build_streaming_graph():
    memory = MemorySaver()
    def superbot(state: State):
        return {"messages": [llm.invoke(state['messages'])]}
    graph = StateGraph(State)
    graph.add_node("SuperBot", superbot)
    graph.add_edge(START, "SuperBot")
    graph.add_edge("SuperBot", END)
    return graph.compile(checkpointer=memory)

# ========== Human-in-the-Loop Example ==========
def build_human_in_loop_graph():
    try:
        from langchain_tavily import TavilySearch
        from langchain_core.tools import tool
        from langgraph.types import Command, interrupt
    except ImportError:
        print("langchain_tavily or langchain_core not installed. Human-in-the-loop will be limited.")
        return None
    llm = init_chat_model("groq:llama3-8b-8192")
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    graph_builder = StateGraph(State)
    @tool
    def human_assistance(query: str) -> str:
        """Request assistance from a human."""
        human_response = interrupt({"query": query})
        return human_response["data"]
    tool = TavilySearch(max_results=2)
    tools = [tool, human_assistance]
    llm_with_tools = llm.bind_tools(tools)
    def chatbot(state: State):
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

def get_graph():
    """Returns the compiled LangGraph chatbot."""
    return build_memory_graph()

if __name__ == '__main__':
    # This script is now designed to be imported by the Streamlit app.
    # You can add any direct execution logic here for testing if needed.
    pass 