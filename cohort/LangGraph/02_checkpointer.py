# checkpointer is used to remember the context and it is saved in memory.

from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class LLMState(TypedDict):
    messages: Annotated[list, add_messages]


def chatBot(state: LLMState):
    user_query = state['messages']
    # print(f'USER QUERY: {user_query.content}')
    result = model.invoke(user_query)
    # print(f"[DEBUG] : STATE : {state}")

    # [IMPORTANT]: WHEN YOU RETURN ONLY THE NEWLY GENERATED MESSAGES THEN ONLY THEY ARE ADDED TO THE STATE.
    # MESSAGES WONT BE ADDED IF THE ENTIRE STATE IS RETURNED
    return {'messages': [result]}


graph_builder = StateGraph(LLMState)

# add nodes
graph_builder.add_node("chatBot", chatBot)

# add edges
graph_builder.add_edge(START, "chatBot")
graph_builder.add_edge("chatBot", END)

# compile graph with memory
memory = MemorySaver()
config = {"configurable": {"thread_id": "1"}}

graph = graph_builder.compile(checkpointer=memory)

# stream the response (value only)

while True:
    user_query = input("> ")
    
    events = graph.stream(
        {"messages": [ {'role': 'user', 'content': user_query} ] }, 
        config = config,
        stream_mode='values',
    )
    for event in events:
        event['messages'][-1].pretty_print()
        # print(event['messages'][-1])
