import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from uuid import uuid4
import sys
import os

# Add the chatbot directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '1-BasicChatBot'))

# Import the get_graph function from your backend
from basic_chatbot import get_graph

st.set_page_config(page_title="LangGraph Chatbot", page_icon="🤖")
st.title("LangGraph Chatbot")

# Session state for messages and thread_id
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())
if "graph" not in st.session_state:
    st.session_state.graph = get_graph()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat input
if prompt := st.chat_input("What is up?"):
    # Add user message to session state and display it
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke the graph to get the response
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    
    # We pass the new message to the graph
    event = st.session_state.graph.invoke(
        {
            "messages": [
                SystemMessage(
                    content=(
                        "You are a helpful assistant. For any question about current events, news, or recent facts, "
                        "you MUST use the Tavily search tool, even if you think you know the answer. "
                        "Do NOT answer from your own knowledge for these questions. "
                        "Always call the Tavily search tool and summarize its results for the user."
                    )
                ),
                HumanMessage(content=prompt)
            ]
        },
        config=config
    )
    
    # The response from the graph contains the full conversation history
    # We take the last message, which is the AI's response
    ai_response = event['messages'][-1]
    
    # Update the session state with the full history from the graph's response
    st.session_state.messages = event['messages']
    
    with st.chat_message("assistant"):
        st.markdown(ai_response.content)
    
    # Rerun to display the updated messages list
    st.rerun() 