from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model1 = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    template="""Give me a 5 points summary on the following topic
        {topic}
    """,
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="""
    create 5 or more questions and answers on the following topic
    {topic}
""",
input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="""
    combine both summary and quiz into single result
    summary -> {summary}, quiz -> {quiz}
""",
input_variables=["summary", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "summary": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain

result = final_chain.invoke({
    "topic": "cricket"
})

print(result)