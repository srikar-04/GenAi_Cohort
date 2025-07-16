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