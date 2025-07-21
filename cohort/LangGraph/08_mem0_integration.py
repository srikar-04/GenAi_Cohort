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
            'collection_name': 'user_info',
            'host': 'localhost',
            'port': 6333,
            'embedding_model_dims': 768
        }
    },
    'embedder': {
        'provider': 'gemini',
        'config': {
            'model': 'models/text-embedding-004',
            'embedding_dims': 768
        }
    },
    # "history_db_path": "./history.db",
}

memory = Memory.from_config(config)

# defining state:

class MemoryState(TypedDict):
    input: str
    messages: Annotated[list, add_messages]
    memory_info: str

# defining nodes : 

def add_to_memory(state: MemoryState):
    responses = state['messages']
    system_result = '\n---------------\n'.join(response.content for response in responses)
    messages = [
        {'role': 'user', 'content': state['input']},
        {'role': 'system', 'content': state['messages'][-1].content}
    ]
    memory.add(messages, user_id='srikar')
    # print(f"UPDATED MEMORY : {result}")
    print(system_result)

def retreive_memory(state: MemoryState):
    user_query = state['input']
    enitre_memory = memory.search(query=user_query, user_id='srikar')
    relevant_memory = ''
    if enitre_memory['results'] and len(enitre_memory['results']) != 0:
        relevant_memory = "\n-----------\n".join(event.get('memory') for event in enitre_memory['results'])
        # print(enitre_memory['results'])
    return {'memory_info': relevant_memory}

def llm_call(state: MemoryState):
    user_query = state['input']
    relevant_memory = state['memory_info']
    result = model.invoke(f"""
        You are an helpuful ai assistant, you can generate response based on the available info about the user. 
        Here is what you know about the user : \n {relevant_memory} 
        
        Here is the query form the user :
        {user_query}
    """)
    return {'messages': {'role': 'system', 'content': result.content}}

graph_builder = StateGraph(MemoryState)

graph_builder.add_node('llm_call', llm_call)
graph_builder.add_node('add_to_memory', add_to_memory)
graph_builder.add_node('retreive_memory', retreive_memory)

graph_builder.add_edge(START, 'retreive_memory')
graph_builder.add_edge('retreive_memory', 'llm_call')
graph_builder.add_edge('llm_call', 'add_to_memory')
graph_builder.add_edge('add_to_memory', END)

graph = graph_builder.compile()

while True:
    user_query = input('> ')

    final_state = graph.invoke({
        'input': user_query
    })