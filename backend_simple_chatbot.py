from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage

load_dotenv()

llm = ChatOllama(model = 'gemini-3-flash-preview:cloud')

class ChatState(TypedDict):
    
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {'messages': [response]}

# CheckPointer
checkpointer = InMemorySaver()

# Defining graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)


# Streaming


# for message_chunk, metadata in chatbot.stream(
#     {'messages': [HumanMessage(content='What is the reciepe to make Pasta')]},
#     config = {'configurable': {'thread_id': 'thread-1'}},
#     stream_mode='messages'
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=' ', flush=True)

# print(type(stream))
