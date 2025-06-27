from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate



if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


prompt1 = PromptTemplate(
    template = """Write a 300 words summary on the topic \n {topic}""",
    input_variables=["topic"]
)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chain = prompt1 | model

result = chain.invoke({
    "topic": input("> ")
})

print(result.content)