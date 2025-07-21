import os
from dotenv import load_dotenv
load_dotenv()
from mem0 import Memory

from typing_extensions import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

mem_api_key = os.environ['MEM0_API_KEY']

config = {
    'llm': {
        'provider': 'gemini',
        'config': {
            "model": "gemini-2.0-flash-001",
            'temperature': 0.2
        }
    },
    'vector_store': {
        'provider': 'qdrant',
        'config': {
            'collection_name': 'test',
            'host': 'localhost',
            'port': 6333
        }
    },
    'embedder': {
        'provider': 'gemini',
        'config': {
            'model': 'models/text-embedding-004',
            'embedding_dims': 768
        }
    },
    "history_db_path": "./history.db",
}

user_id = 'srikar'
memory = Memory.from_config(config)

# defining state:

class MemoryState(TypedDict):
    input: str
    messages: Annotated[list, add_messages]

# defining nodes : 

def add_to_memory(state: MemoryState):
    responses = state['messages']
    result = '\n---------------\n'.join(response for response in responses)
    print(result)

def llm_call(state: MemoryState):
    user_query = state['input']
    result = model.invoke(user_query)
    return {'messages': [{'role': 'system', 'content': result.content}]}

graph_builder = StateGraph(MemoryState)

graph_builder.add_node('llm_call', llm_call)
graph_builder.add_node('add_to_memory', add_to_memory)

graph_builder.add_edge(START, 'llm_call')
graph_builder.add_edge('llm_call', 'add_to_memory')
graph_builder.add_edge('add_to_memory', END)

graph = graph_builder.compile()

user_query = input('tell us about yourself : ')

final_state = graph.invoke({
    'input': user_query
})