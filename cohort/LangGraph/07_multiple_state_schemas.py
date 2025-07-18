# If your project has multiple types of data at multiple stages then you have to use multiple schemas(multiple states)
# Don't know why the fuck it is giving me string output when i fetch them from state. so i explicitly converted to int type

from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
load_dotenv()
import random
import math

from langgraph.types import Command, interrupt

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


class InputState(TypedDict):
    number: int

class OutputState(TypedDict):
    operated_number: float

class OverallState(InputState, OutputState):
    random_number: float

class PrivateState(TypedDict):
    private_data: int

# flow -> inputstate -> overallstate -> privatestate -> outputstate

# define nodes :
def obtain_number(state: InputState) -> OverallState:
    # generating random number between 1 to 10 and adding it to the number sent by the user
    random_number = math.floor((random.random() * 10) + 1)
    # print(f'TYPE OF RANDOM NUM IN OBTAIN NUMBER : {type(random_number)}')
    input = state['number']
    return {'random_number': int(random_number)*input}

def private_data(state: OverallState) -> PrivateState:
    # adding 100 to the obtained data
    random_number = state['random_number']
    print(f'TYPE OF RANDOM NUM IN PRIVATE DATA : {type(random_number)}')
    return {'private_data': (int(random_number) + 100)}

def output_node(state:PrivateState) -> OutputState:
    # divind with 3 from the data of output state :
    return {'operated_number': state['private_data']/3}


graph_builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)

# add nodes :
graph_builder.add_node('obtain_number', obtain_number)
graph_builder.add_node('private_data', private_data)
graph_builder.add_node('output_node', output_node)

# add edges :

graph_builder.add_edge(START, 'obtain_number')
graph_builder.add_edge('obtain_number', 'private_data')
graph_builder.add_edge('private_data', 'output_node')
graph_builder.add_edge('output_node', END)

graph = graph_builder.compile()

user_input = input('enter a number to see your lucky number related to it > ')

final_state = graph.invoke({
    'number': user_input
})

print(final_state['operated_number'])