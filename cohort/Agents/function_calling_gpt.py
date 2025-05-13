import json
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from pydantic import BaseModel


api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()


def get_weather(city: str) -> str:
    return f"the weather in {city} is 31 degree celcius!!"

function_names = {
    "get_weather": get_weather
}

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City to find weather for eg., new york."
                }
            },
            "required": ["city"]
        }
    }
]

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the weather in New York?"}
#     ], 
#     max_tokens=500,    # max_tokens working
#     temperature=0.1    # temperature working
# )

#  ans.choices[0].message.content

system_prompt_test = "Talk like a mad scientist."

messages = [
    {
        "role": "developer",
        "content": system_prompt_test
    },
    {
        "role": "user",
        "content": "what is the weather in new york?"
    }
]

response = client.responses.create(
    model="gpt-4o",
    input=messages,
    max_output_tokens=500,
    temperature=0.1,
    tools=tools
)

# model_response = response.output[0].arguments  # "arguments" ARE IN JSON FORMAT

# IF THERE IS NO TOOLS INVOLVED

# model_response = response.output[0].content[0].text 

# IF THERE ARE TOOLS INVOLVED


tool_call = response.output[0]

if tool_call.type == "function_call":
    print(tool_call)
    args = json.loads(tool_call.arguments)
    tool_string = tool_call.name
    tool = function_names.get(tool_string)
    print(type(tool))    # type is function which is fucking correct!!!!!!
    result = tool(*args)
    print(result)

    # first we have to append the original tool call response
    messages.append(tool_call)

    # then appending with the function's call id
    messages.append(
        {
            "type": "function_call_output",
            "call_id": tool_call.call_id,   
            "output": str(result)
        }
    )

    response_2 =  client.responses.create(
        model="gpt-4o",
        input=messages,
        max_output_tokens=500,
        temperature=0.1,
        tools=tools
    )

    tool_call_2 = response_2.output[0]
    if tool_call_2.type == "function_call":
        print(tool_call_2)
    else:
        print("2nd response have no function call!!")
        print(tool_call_2.content[0].text)
else:
    print("no tool call involved")
