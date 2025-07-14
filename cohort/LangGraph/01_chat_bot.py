from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()


if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# creating a STATE class
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    user_query = input('>  ')
    result = model.invoke(user_query)
    final_result = result.content
    state['messages'].append(final_result)
    # print(f"DEBUG: {state}")
    return state

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

final_state = graph.invoke({"messages": []})

print(final_state)