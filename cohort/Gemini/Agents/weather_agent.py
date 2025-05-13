from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import json

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model="gemini-2.0-flash"

client = OpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url,
)

def get_weather(city: str) -> str :
    return f"weather in {city} is 31 degree celcius!!"

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

available_tools = {
    "get_weather": get_weather,
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
6. NEVER provide an answer that requires tool information without first calling the appropriate tool.

Output Format:

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
"""

messages = [
    {"role":"developer","content":system_prompt},
    {"role": "assistant", "content": json.dumps({
        "step": "start",
        "content": "The user is asking about the weather in New York.",
        "function": "null",
        "input": "null"
    })},
    {"role": "assistant", "content": json.dumps({"step": "plan", "content": "To answer the user's question, I need to use the `get_weather` function to retrieve the weather information for New York.", "function": "null", "input": "null"})},
    {
        "role": "assistant",
        "content": "ERROR: there is no tool call happened in execute step. Select and make an appropriate tool call"
    },
    {"role": "assistant", "content": json.dumps({"step": "plan", "content": "To get the weather in New York, I need to use the `get_weather` function.", "function": "null", "input": "null"})},
]

# while True:
user_query = input("> ")
messages.append({
    "role": "user",
    "content": user_query
})

# while True:
result = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.2,
    tools=tools,
)
try:
    # contains function_call=None, tool_calls=None
    response = result.choices[0].message 
    # parsed_response = json.loads(result.choices[0].message.content) # contains "step" and "content"
    print(response)

    if json.loads(response.content).get("step") == "execute":
        if response.tool_calls == None:
            messages.append({
                "role": "system",
                "content": "ERROR: You must make a function call during the execute step. Please try again and make sure to use the appropriate tool function."
            })
            print("No tool calls present")
            exit()

    if response.tool_calls:
        print('tools is available !!')
        print(response.tool_calls)
    elif response.function_call:
        print("function call is present")
        print(response.function_call)
    else:
        print("------------- still there is no tool call --------------------")
        print(response.content)
    # continue
except Exception as e:
    print(f"messed up with json: {e}")
    # break