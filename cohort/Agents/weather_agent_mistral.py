from dotenv import load_dotenv
load_dotenv()
from mistralai import Mistral
import os
import json
import functools
import re

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"

client = Mistral(api_key=api_key)


def get_weather(city: str) -> str:
    return f"weather in {city} is 31 degree celcius!!"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns the weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to get the weather for"
                    }
                },
                "required": ["city"]
            }
        }
    },
]

names_to_functions = {
    "get_weather": functools.partial(get_weather),
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
- {"step": "execute", "function": "get_weather", "input": "New York"}
- {"step": "monitor", "content": "Weather in New York is very hot today, 31 degrees Celsius."}
- {"step": "result", "content": "31 degrees Celsius."}

Tools:
- get_weather: "Returns the weather in a given location"
- input: "Takes input from the user query"
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
]

user_query = input("Enter Your Query \n > ")

messages.append(
    {
        "role": "user",
        "content": user_query
    },
)

while True:
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0.1,
        tools=tools,
        tool_choice="auto"
    )

    try:
        result = chat_response.choices[0].message
        parsed_response = None

        # If the model wants to call a tool/function
        if result.tool_calls:
            tool_call = result.tool_calls[0]
            function_name = tool_call.function.name
            function_arguments = json.loads(tool_call.function.arguments)

            # First add the assistant's message with the tool call
            # messages.append({
            #     "role": "assistant",
            #     "content": None,
            #     "tool_calls": result.tool_calls
            # })

            if function_name == tools[0].get("function").get("name"):
                output = names_to_functions.get(function_name)(**function_arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": output
                    }
                )
            continue

        else:
            try:
                parsed_response = json.loads(result.content)
            except json.JSONDecodeError:
                # print("JSON messed up")
                reg_expression = re.sub(r"```json|```", "", result.content)
                parsed_response = json.loads(reg_expression)
                
            messages.append({
                "role": "assistant",
                "content": result.content
            })

            messages.append({
                "role": "user",
                "content": "Please continue with the next step"
            })
            

            print(f"🧠: {parsed_response.get('content')}")

            if parsed_response.get("step") == "result":
                print(f"🤖: {parsed_response.get('content')}")
                break

    except Exception as e:
        print(f"Something went wrong!!:  {e}")
