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