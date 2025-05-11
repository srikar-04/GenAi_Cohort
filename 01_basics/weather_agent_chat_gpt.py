from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import json
import requests
import random
import math

api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"weather in {city} is {response.text}"
    else:
        return "something went wrong with weather api"

def random_number(range):
    random_num = math.floor(random.random() * range)
    return random_num

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
    },
    {
        "type": "function",
        "name": "random_number",
        "description": "Generationg a random number in the given range",
        "parameters": {
            "type": "object",
            "properties": {
                "range": {
                    "type": "number",
                    "description": "range in which random number is to be generated"
                }
            }
        }
    }
]

function_names = {
    "get_weather": get_weather,
    "random_number": random_number
}

system_prompt = """
You are a helpful AI assistant who is expert in breaking down complex problems and resolving user queries.
You work in the following modes: start, plan, execute, monitor, and result.
For the given user query and available tools, plan step-by-step execution. Based on planning, select the relevant tool from the available tools. Perform the task based on the selected tool.
Wait for observation and resolve the user query based on the observation from the tool call.

Follow the steps in sequence: "start", "plan", "execute", "monitor", and "result".

Rules:
1. Output should be strictly in JSON format.
2. Strictly perform only one step at a time and wait for the next input.
3. Carefully analyze the user query.
4. Donot wrap the response in a code block or fences
5. Tool call MUST happen only in the "execute" step. During this step, you must invoke the correct function using the OpenAI tool call format. Do NOT just describe it — instead, respond with a tool call so that the tool is actually invoked.

Output JSON Format:

{
    "step": "step name",
    "content": "step description",
    "function": "name of function specified in Tools section. Specify it only if the step is 'execute'",
    "input": "input to the function present in user query. Specify it only if the step is 'execute'"
}

Example:
User Query: What is the weather in New York?
- {"step": "start", "content": "Alright! The user is interested in a weather query and is asking about the weather in New York"}
- {"step": "plan", "content": "To get the weather in New York, I need to call the get_weather function"}
- {"step": "execute", "content": "Executing function call"}
- {"step": "monitor", "content": "Weather in New York is very hot today, 31 degrees Celsius."}
- {"step": "result", "content": "31 degrees Celsius."}

Tools:
- get_weather: "Returns the weather in a given location"
- input: "Takes input from the user query"
- random_number: "Generates a random number in the given range"
"""


messages = [
    {
        "role": "developer",
        "content": system_prompt
    },
]

while True:

    user_query = input("> ")

    messages.append({
        "role": "user",
        "content": user_query
    })



    while True:

        response = client.responses.create(
            model="gpt-4o",
            input=messages,
            max_output_tokens=500,
            temperature=0.1,
            tools=tools
        )

        tool_call = response.output[0]

        if tool_call.type == "function_call":
            
            args = json.loads(tool_call.arguments)
            tool_string = tool_call.name
            tool = function_names.get(tool_string)
            if tool != None:
                print(f"⛏️: calling the tool {tool_call.name}")
                result = tool(**args)

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
            else:
                print("No tool present!!!")
                break

            continue
        else:
            try:
                no_tool_response = json.loads(tool_call.content[0].text)
                current_step = no_tool_response.get("step")


                if current_step == "result":
                    print(f"🤖: {no_tool_response.get("content")}")
                    break

                messages.append({
                    "role": "assistant",
                    "content": json.dumps(no_tool_response)
                })

                print(f"🧠: {no_tool_response.get("content")}")
            except Exception as e:
                print(f"something went wrong with JSON {e}")