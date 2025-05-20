from dotenv import load_dotenv
load_dotenv()
import os 
from mem0 import Memory

gemini_api_key = os.environ["GEMINI_API_KEY"]

config = {
    "version": "v1.1",
    "llm": {
        "provider": "gemini",
        "config": {
            "api_key": gemini_api_key,
            "model": "gemini-2.0-flash",
            "temperature": 0.2,
            "max_tokens": 2000,
        }
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "api_key": gemini_api_key,
            "model": "gemini-embedding-exp",
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "knowledge_graph",
            "host": "localhost",
            "port": 6333,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "APkfoSrEXqdWE1Rwo7NdANlVpuu-O4Mdw57cm1wZiVs"
        }
    }
}