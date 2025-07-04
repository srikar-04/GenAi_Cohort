from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import json
import requests

api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()

def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"weather in {city} is {response.text}"
    else:
        return "something went wrong with weather api"

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

function_names = {
    "get_weather": get_weather,
}


system_prompt = """
You are a helpful AI assistant who is expert in breaking down complex problems and resolving user queries.

For the given user query and available tools, plan execution. Based on planning, select the relevant tool from the available tools. Perform the task based on the selected tool.

Rules:
1. Output should be strictly in JSON format.
2. Carefully analyze the user query.
3. Donot wrap the response in a code block or fences
4. NEVER provide an answer that requires tool information without first calling the appropriate tool.

Output JSON Format:

{
    "content": "content of the message to be sent to user",
    "function": "name of function specified in Tools section. Specify it only if the step is 'execute'", (if present)
    "input": "input to the function present in user query. Specify it only if the step is 'execute'" (if present)
}

Example:
User Query: What is the weather in New York?
- {"content": "Executing function call", "function": "get_weather", "input": {"city": "New York"}}
- {"content": "31 degrees Celsius."}

Tools:
- get_weather: "Returns the weather in a given location"
"""


messages = [
    {
        "role": "developer",
        "content": system_prompt
    },
]

user_query = input("> ")

messages.append({
    "role": "user",
    "content": user_query
})

response = client.responses.create(
    model="gpt-4.1",
    input=messages,
    max_output_tokens=500,
    temperature=0.1,
    tools=tools,
    tool_choice="auto",
)

tool_call = response.output[0].content[0].text

# print(f"INITIAL OUTPUT Type: {type(tool_call)}")

try:
    tool_call_json = json.loads(tool_call)
    print(f"Parsed JSON: {tool_call_json}")
    function_name = tool_call_json.get("function")
    if function_name in function_names:
        args = tool_call_json.get("input", {})
        result = function_names[function_name](**args)
    else:
        result = tool_call_json.get("content", "No function to execute.")
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}\nRaw output: {tool_call}")
    result = tool_call

print(f"RESULT : {result}")