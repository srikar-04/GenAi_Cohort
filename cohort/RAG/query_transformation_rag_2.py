from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that transforms user queries into a more specific and focused question."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    response_format={"type": "json_object"},
)

print(response.choices[0].message.content)

