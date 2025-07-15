from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command, interrupt

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class State(TypedDict):
    user_query: str
    messages: Annotated[list, add_messages]

@tool
def humanAssistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt(
        {
            "query": query
        }
    )
    return human_response['data']


tools = [humanAssistance]
model_with_tools = model.bind_tools(tools=tools)
available_tools = {
    "humanAssistance": humanAssistance
}

def chatBot(state: State):
    user_query = state['user_query']
    result = model_with_tools.invoke(user_query)
    return {'messages': [result]}

def tool_node(state: State):
    tool_name = available_tools.get(state['messages'][-1].tool_calls[0]['name'])
    args = state['messages'][-1].tool_calls[0]['args']

    tool_response = tool_name.func(**args)

    print(f"THIS IS TOOL RESPONSE: {tool_response}")
    return {'messages': [tool_response]}
    


graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node("chatBot", chatBot)
graph_builder.add_node('tool_node', tool_node)

# add edges
graph_builder.add_edge(START, "chatBot")

graph_builder.add_conditional_edges(
    "chatBot",
    lambda state: 'tool_call' if state['messages'][-1].tool_calls else 'llm_call',
    {
        'tool_call': "tool_node",
        'llm_call': 'chatBot'
    }
)

graph_builder.add_edge('chatBot', END)
graph_builder.add_edge('tool_node', END)

checkpointer = MemorySaver()
config = {"configurable": {"thread_id": '3'}}

graph = graph_builder.compile(checkpointer=checkpointer)

user_query = input('> ')

state = graph.invoke(
    {
        "user_query": user_query
    },
    config=config
)

while "__interrupt__" in state:
    print(state["__interrupt__"][0].value)
    human_assistance = input('user needs your assistance please help him \n >')
    state = graph.invoke(Command(resume={"data": human_assistance}), config=config)

final_response = state['messages'][-1]

print(f'THIS IS FINAL RESPONSE: {final_response}')