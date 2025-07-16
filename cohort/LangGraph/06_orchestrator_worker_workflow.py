# an orchestrator breaks down a task and delegates each sub-task to workers.
# In the orchestrator-workers workflow, a central LLM (orchestrator) dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results.

from langgraph.checkpoint.memory import MemorySaver

from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from typing import List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel, Field
import operator
from langgraph.types import Send

from langgraph.types import Command, interrupt

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class Section(BaseModel):
    # contains name of the topic and it's brief desctiption
    name: str = Field('name of the topic in the whole report')
    description: str = Field('Brief description of the topic')

class Sections(BaseModel):
    sections: List[Section] = Field('list of all sections along with their sub topic names and their desctiptions')

model_with_structured_output = model.with_structured_output(Sections)

# MAIN GRAPH STATE : 
class State(TypedDict):
    topic: str
    sections: list[Section]   # list of sections where looping will be perfomed and worker llms are called
    completed_sections: Annotated[list, operator.add] # completed sections are appended from worker llms
    final_response: str

# WORKER STATE:
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]


def orchestrator(state: State):
    """create multiple sections based on the given topic (using structured model)"""
    response = model_with_structured_output.invoke(
        [
            SystemMessage(content='create list of sections based on the given topic'),
            HumanMessage(content=f'Here is the topic based on which you should create different sections : {state["topic"]}')
        ]
    )
    return {"sections": response.sections}

def llm_call(state: WorkerState):
    """Generate brief description on what all topic should one cover in that particular section"""
    response = model.invoke(
        [
            SystemMessage(content='Generate a reponse on what all are the important topics that i should conver for a given section'),
            HumanMessage(content=f"This is the section that you should explain with all important topics to cover: \n  SECTION NAME : {state['section'].name} \n SECTION'S DESCRIPTION : {state['section'].description}")
        ]
    )

    return {'completed_sections': [response.content]}

def synthesizer(state: State):
    """Synthesize all the data collected from parallel llm calls"""
    final_response = '\n \n'.join(content for content in state['completed_sections'])
    # print(f'THIS IS THE FINAL RESPONSE IN SYNTEHSIZER : {final_response}')

    return {'final_response': final_response}

def assign_worker(state: State):
    return [Send('llm_call', {"section": s}) for s in state['sections']]


graph_builder = StateGraph(State)

# add nodes
graph_builder.add_node('orchestrator', orchestrator)
graph_builder.add_node('llm_call', llm_call)
graph_builder.add_node('synthesizer', synthesizer)
graph_builder.add_node('assign_worker', assign_worker)

# add edges

graph_builder.add_edge(START, 'orchestrator')
graph_builder.add_conditional_edges(
    'orchestrator',
    assign_worker,
    ['llm_call']
)
graph_builder.add_edge('llm_call', 'synthesizer')
graph_builder.add_edge('synthesizer', END)

graph = graph_builder.compile()

user_query = input('> ')

final_state = graph.invoke({'topic': user_query})

print(final_state['final_response'])