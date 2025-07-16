# in this file we use time travel to navigate between nodes and modify the response. goal is to summarise a text using two different methods

# HIGH LEVEL GOAL : 
# Build a graph:

# Node A: Load document
# Node B: Summarize with strategy 1
# Node C: Summarize with strategy 2
# Run the graph to Node B.
# Use time travel to revert to Node A’s state and run Node C instead.

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
    text: str
    summarize: str
    strategy: str

def load_doc(state: State):
    return {'text': 'LangGraph is a powerful framework for building agent workflows.LangGraph Platform makes it easy to get your agent running in production — whether it’s built with LangGraph or another framework — so you can focus on your app logic, not infrastructure. Deploy with one click to get a live endpoint, and use our robust APIs and built-in task queues to handle production scale.', 'strategy': 'strategy_1'}

def strategy_1(state: State):
    text = state['text']
    summary = f'summary using STRATEGY_1 {text[:20]}'
    return {'summarize': summary}


def strategy_2(state: State):
    text = state['text']
    summary = f'summary using STRATEGY_2 {text.split()[0]}'
    return {'summarize': summary}

graph_builder = StateGraph(State)

graph_builder.add_node('load_doc', load_doc)
graph_builder.add_node('strategy_1', strategy_1)
graph_builder.add_node('strategy_2', strategy_2)

# add edges

graph_builder.add_edge(START, 'load_doc')
graph_builder.add_conditional_edges(
    'load_doc',
    lambda x: 'strategy_1' if x['strategy'] == 'strategy_1' else 'strategy_2',
    {
        'strategy_1': 'strategy_1',
        'strategy_2': 'strategy_2'
    }
)
graph_builder.add_edge('strategy_1', END)
graph_builder.add_edge('strategy_2', END)

memory = MemorySaver()
config = {"configurable": {"thread_id": "3"}}

graph = graph_builder.compile(checkpointer=memory)

state = graph.invoke({}, config=config)

print(state['summarize'])

snapshots = list(graph.get_state_history(config=config))

# print(snapshots)

selected_state = None

for snapshot in snapshots:
    # print(f'{snapshot} \n \n')
    # print(f'NEXT NODE : {snapshot.next}')
    # print(f'PRESENT NODE CONFIG : {snapshot.config} \n \n')

    if snapshot.next and (snapshot.next[0]  == 'strategy_1'):
        print('inside the if condition')
        selected_state = snapshot

print(f'SELECTED STATE : {selected_state.next}\n {selected_state.values}')