from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

prompt = "what is a blackhole"

response = client.responses.create(
    model='gpt-3.5-turbo',
    input=prompt,
)

print(response.output_text)