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

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class ToolState(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def multiply(a: float, b: float) -> float:
    """
        perform multiplication operation on the provided arguments

        Args :
        a: first argument in float
        b: second argument in float
    """
    return a * b

model_with_tools = model.bind_tools([multiply])

available_tools = {
    "multiply": multiply
}

def tool_router(state: ToolState):
    user_query = state['messages'][-1]
    # print(f"USER QUERY : {user_query.content}")
    result = model_with_tools.invoke(user_query.content)
    # print(f"RESULT: {result.tool_calls}")
    if result.tool_calls:
        # pass to tool call node
        tool_call_msg = AIMessage(content='', tool_calls=result.tool_calls)
        return {"messages": [tool_call_msg]}
    else:
        # pass to general llm call edge
        return {"messages": [AIMessage(content=result.content)]}

# print(result)

def tool_node(state: ToolState):
    # print(f"STATE IN TOOL NODE: {state} \n \n \n ")
    tool = available_tools.get(state['messages'][-1].tool_calls[0]['name'])
    args = state['messages'][-1].tool_calls[0]['args']

    # [IMPORTANT]: even after getting the tool name from available tools it wont be callable because langchain modifies it into langchain function. that contains "func" which represents the callble function.
    tool_response = tool.func(**args)

    final_response = AIMessage(content=f'the final tool response is {tool_response}')
    return {'messages': [final_response]}

def llm_node(state: ToolState):
    # perform llm call based on state
    user_query = state['messages'][-1].content
    result = model.invoke(user_query)
    # print(f"STATE IN LLM NODE: {state}")
    return {'messages': [result]}


# add nodes

graph_builder = StateGraph(ToolState)


graph_builder.add_node("tool_router", tool_router)
graph_builder.add_node("tool_node", tool_node)
graph_builder.add_node("llm_node", llm_node)


# add edges

# 1 -> llm call to know if there is actual need for tool call or not
# 2 -> if yes then prepare a tool call node for calling the tool
# 3 -> if no then return with an llm call

graph_builder.add_edge(START, "tool_router")

graph_builder.add_conditional_edges(
    "tool_router",
    lambda state: 'tool_call' if state['messages'][-1].tool_calls else 'llm_call',
    {
        'tool_call': 'tool_node',
        'llm_call': 'llm_node'
    }
)

graph_builder.add_edge('tool_node', END)
graph_builder.add_edge('llm_node', END)

graph = graph_builder.compile()

user_query = input("> ")

final_state = graph.invoke(
    {"messages": [user_query]}
)

print(final_state['messages'][-1])