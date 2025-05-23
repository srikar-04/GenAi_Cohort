from dotenv import load_dotenv
load_dotenv()
import os 
from mem0 import Memory
from openai import OpenAI

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "gemini-2.0-flash"

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
            "model": "models/embedding-001",
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "knowledge_graph",
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "reform-william-center-vibrate-press-5829",
        }
    }
}

mem_client = Memory.from_config(config)


messages=[]

def chat(message):
    client = OpenAI(
        api_key=gemini_api_key,
        base_url=gemini_base_url
    )

    known_memory = mem_client.search(query=message, user_id="srikar")

    memories = ""
    for memory in known_memory.get("results"):
        memories+= f"{memory.get("memory")}:{memory.get("score")}"

    SYSTEM_PROMPT = f"""
        You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
        systematically analyze input content, extract structured knowledge, and maintain an
        optimized memory store. Your primary function is information distillation
        and knowledge preservation with contextual awareness.
        Tone: Professional analytical, precision-focused, with clear uncertainty signaling
        Memory and Score:
        {memories}
    """

    # pprint.pp(f"MEMORY : \n\n {known_memory.get("results")}\n\n")

    messages.append(
        {
            "role": "assistant",
            "content": SYSTEM_PROMPT
        },
    )

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    messages.append({
        "role": "assistant",
        "content": response.choices[0].message.content
    })

    mem_client.add(
        {"role": "user", "content": message},
        user_id="srikar"
    )

    print(response.choices[0].message.content)


while True:
    print("Type `exit` to exit loop !!")
    user_query = input("> ")
    if user_query == "exit":
        break
    chat(user_query)