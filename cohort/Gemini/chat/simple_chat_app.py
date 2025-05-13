from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model="gemini-2.0-flash-lite"

client = OpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url
)

messages = [
    {"role":"developer","content":"you are an helpful ai assistant whose name is Garuda"},
]

user_query = input("> ")

messages.append({
    "role": "user",
    "content": user_query
})

response = client.chat.completions.create(
    model=model,
    messages=messages
)

print(response.choices[0].message.content)