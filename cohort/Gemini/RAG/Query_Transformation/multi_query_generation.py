from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import tool

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

@tool
def multi_query(query):
    """
        You are given with an abstract user query. Generater multiple queries to make it more specific and context rich, still keeping the user's intent.

        Args: [query]

        RULES : 
        - Donot hellucinate anything.
        - Respond with only multiple re-written queries as response
        - Donot respond with a single query
    """
    return 'TOOL CALL DONE {query}'

tools = [multi_query]

available_tools = {
    "multi_query": multi_query
}

system_prompt = PromptTemplate.from_template(
    """
        You are an intelligent ai assistant expert in analysing user queries.
        
        For a given user query analyse the ambiguity of it first. If the query is more ambiguous or you think it needs more elaboration for better responses, call the tools from the available tools.

        Available Tools: 
        {available_tools}

        Rules: 
        - Strictly pass the original query, sent by the user, to the choosen tool call as an argument.
        - Make a tool call only if it is necessary.
        - Donot hellucinate or break any rules.
    """
)

model_with_tools = model.bind_tools(tools)

user_query = input("> ")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt.format(available_tools = tools)),
        ("human", user_query)
    ]
)

chain1 = prompt | model_with_tools

result = chain1.invoke({})

print(result)