from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_model = "gemini-2.0-flash"

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = f"""
    You are an helpful ai assistant
"""

messages = [
    {
        "role": "developer",
        "content": system_prompt
    }
]

while True:
    print("type `exit` to exit loop")

    user_query = input(">> ")

    if user_query == 'exit':
        break

    messages.append({
        "role": "user",
        "content": user_query
    })

    response = client.chat.completions.create(
        model=gemini_model,
        messages=messages
    )

    print(response.choices[0].message.content)