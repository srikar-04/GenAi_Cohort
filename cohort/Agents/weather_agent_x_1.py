from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
import json
import requests
import re

# api_key = os.environ["OPENAI_API_KEY"]

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "gemini-2.0-flash"

client = OpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url,
)

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
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find weather for eg., new york",
                    },
                },
                "required": ["city"],
            },
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
3. IMPORTANT: Donot wrap the response in a code block or fences
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

response = client.chat.completions.create(
    messages=messages,
    model=model,
    max_tokens=500,
    temperature=0.1,
    tools=tools,
)

tool_call_code = response.choices[0].message.content

tool_call = re.sub(r"^```[a-zA-Z]*\n|```$", "", tool_call_code.strip(), flags=re.MULTILINE)

# print(f"INITIAL OUTPUT Type: {tool_call}")

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