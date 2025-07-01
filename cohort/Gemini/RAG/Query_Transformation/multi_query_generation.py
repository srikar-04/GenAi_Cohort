from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class QuestionOutput(BaseModel):
    question: str = Field(description="Model generated question")
    difficulty: str = Field(description="Difficulty level of the question (easy, medium, hard)")

class ToolOutput(BaseModel):
    questions: list[QuestionOutput] = Field(description='A list of generated questions with their difficulty levels')

ouput_parser = PydanticOutputParser(pydantic_object=ToolOutput)

@tool
def multi_query(query: str) -> list:
    """
        You are given with an abstract user query. Generater multiple queries to make it more specific and context rich, still keeping the user's intent.

        Args: 
            query: original query form the user which is of string type

        IMPORTANT : Respond with only python list type, other datatypes are strictly prohibited
    """
    
    system_prompt = """
        You are given with an abstract user query. Generater multiple queries to make it more specific and context rich, still keeping the user's intent. 

        Return all the queries in a python list type, other datatypes are strictly prohibited.

        RULES : 
        - Donot hellucinate anything.
        - Respond with multiple re-written queries in a list as response.
        - Donot respond with a single query.
        - IMPORTANT : Queries should strictly be of type list not of any other data-type.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{topic}"),
        ("system", "{format_instructions}")
    ])

    # formated_prompt = prompt.format(
    #     topic = query,
    #     format_instructions = ouput_parser.get_format_instructions()
    # )

    # print(f"FORMATTED PROMPT : {formated_prompt}")

    chain = prompt | model | ouput_parser

    result = chain.invoke({
        "topic": query,
        "format_instructions": ouput_parser.get_format_instructions()
    })

    final_list_result = []

    # print(f"FUNCTION RESULT: {result.questions}")
    # print(f"FUNCTION RESULT: {type(result)}")

    for item in result.questions:
        final_list_result.append(item.question)

    # print(f"this is the final list result:  {final_list_result}")

    return final_list_result

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

while True:
    try:
        user_query = input("> ")
        if not user_query.strip():
            print("Please enter your query")
            continue
    except KeyboardInterrupt:
        break

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.format(available_tools = tools)),
            ("human", user_query)
        ]
    )

    chain1 = prompt | model_with_tools

    result = chain1.invoke({})


    if result.tool_calls:
        tool_data = result.tool_calls[0]

        function_name_string = tool_data.get("name").strip()

        function_object = available_tools[function_name_string].func

        function_args = tool_data.get('args')

        tool_response = function_object(**function_args)

        # print('TOOL RESPONSE TYPE : ', type(tool_response))  # CONFIRMED THAT THIS IS OF LIST TYPE

        print(f'🤖: {tool_response}')
    else :
        print(f'🤖: {result.content}')