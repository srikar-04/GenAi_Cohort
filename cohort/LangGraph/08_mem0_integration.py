import os
from dotenv import load_dotenv
load_dotenv()
from mem0 import Memory
import uuid

mem_api_key = os.environ['MEM0_API_KEY']

config = {
    'llm': {
        'provider': 'litellm',
        'config': {
            "model": "gemini/gemini-pro",
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
    }
}

user_id = 'srikar'
memory = Memory.from_config(config, user_id)
