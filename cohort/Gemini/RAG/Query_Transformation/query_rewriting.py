from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_model = "gemini-2.0-flash"


def rewrite_query(query):
    return f"THIS QUERY IS TRANSFORMED {query}"

tools = [{
    "type": "function",
    "name": "rewrite_query",
    "description": "Rewrites the original query to add more context for better response.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Original query sent by the user which needs to be transformed."
            }
        },
        "required": [
            "query"
        ],
        "additionalProperties": False
    }
}]

available_tools = {
    "rewrite_query": rewrite_query,
}

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = f"""
    You are an helpfull ai assistant, trained in answering user queries by modifying them so that the original query has more context to generate greate responses.
    This original query rewriting can be done by choosing the appropriate tools you are provided with.

    Available tools are given below
    {available_tools}
"""

messages = [
    {
        "role": "developer",
        "content": system_prompt
    }
]

user_query = input(">> ")
messages.append(
    {
        "role": "user",
        "content": user_query
    }
)

response = client.responses.create(
    model= gemini_model,
    input = messages,
    tools=tools,
)

print(response)