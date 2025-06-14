from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import json

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_model = "gemini-2.0-flash"

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def rewrite_query(query):

    prompt = f"""
        You are given with an abstract user query. Rewrite the query to make it more specific and context rich, still keeping the user's intent.

        Original Query: {query}

        RULES : 
        - Donot hellucinate anything.
        - Respond with only single re-written query as response
        - Donot give multiple queries
    """

    response = client.chat.completions.create(
        model=gemini_model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

    # return f"THIS QUERY IS TRANSFORMED {qp


tools = [{
    "type": "function",
    "function": {
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
        },
        "strict": True
    }
}]

available_tools = {
    "rewrite_query": rewrite_query,
}


system_prompt = f"""
    You are an helpfull ai assistant, trained in answering user queries by applying first principle thinking. You are given with some tools to modify user query so that the original query has more context to generate great responses.

    This original query rewriting can be done by choosing the appropriate tools you are provided with.

    Here are the steps involved: 
    step-1: analyse the user query and see if it is more abstract and if there is any actual need for query rewriting.
    step-2: if you think there is a need then call appropriate tool.

    RULES: 
    - Donot hellucinate anyting.
    - If there is a need for query rewriting then pass the exact or original query written by the user.
    - Donot modify original query before tool calling

    Available tools are given below
    {available_tools}

    After executing the tool, the final response from the tool will be given to you. Based on that context-rich query, generate a perfect answer.
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

response = client.chat.completions.create(
    model= gemini_model,
    messages = messages,
    tools=tools,
)



result = response.choices[0].message

if result.tool_calls:
    print(f"TOOL CALL IS PRESENT")
    # print(result.tool_calls)
    name = result.tool_calls[0].function.name
    args = json.loads(result.tool_calls[0].function.arguments)

    print(f"FUNCTION NAME: {name} \n FUNCTION ARGUMENTS: {args}")

    function_name = available_tools.get(name)

    if function_name:
        function_response = function_name(**args)
        print(f"FINAL RESPONSE 🤖: {function_response}")

        messages.append(result) 

        messages.append({                               # append result message
            "role": "tool",
            "tool_call_id": result.tool_calls[0].id,
            "content": str(function_response)
        })
        
        response = client.chat.completions.create(
            model=gemini_model,
            messages=messages
        )
        print(f"FINAL RESPONSE FROM THE LLM IS 🤖 : {response.choices[0].message.content}")
    else:
        print("Function is unavailable!!!")
else:
    print("NOT TOOL CALL PRESENT, ONLY CONTENT: ")
    print(result.content)