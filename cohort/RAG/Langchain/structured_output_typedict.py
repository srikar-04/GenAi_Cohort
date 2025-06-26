import getpass
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict


if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


class Joke(TypedDict):
    """Telling a joke to user in this specific format"""
    setup: str
    punchline: str
    rating: int

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash",)

model_with_structure = model.with_structured_output(Joke)

response = model_with_structure.invoke('Tell me a joke')

print(response)