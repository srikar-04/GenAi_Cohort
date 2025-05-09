from dotenv import load_dotenv
load_dotenv()
from mistralai import Mistral
import os
import json

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-small-latest"

client = Mistral(api_key=api_key)


def get_weather(city: str) -> str:
    return f"weather in {city} is 31 degree celcius!!"

tools = [
    {
        "type": "function",
        "function": {
            "name": get_weather,
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
    }
]

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

user_query = input("Enter Your Query \n >")

messages.append(
    {
        "role":"user",
        "content": user_query
    },
)

while True:

    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=500,
        temperature=0.1,
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice="auto"
    )

    try:
        result = chat_response.choices[0].message.content
        parsed_response = json.loads(result)
        # print(parsed_response)

        messages.append({
            "role": "assistant",
            "content": json.dumps(parsed_response)
        })

        # handling execute step where we need to do function call
        if parsed_response.get("step") == "execute":
            print(f"🧠: {parsed_response.get("content")}")
            tool_name = parsed_response.get("function")
            tool_input = parsed_response.get("input")

            if tools[0].get("function", {}).get("name") == tool_name:
                output = tools[0].get("function", {}).get("name")(tool_input)
                messages.append(
                    {
                        "role": "assitant",
                        "content": json.dumps({"step": "monitor", "output": output})
                    }
                )
            else:
                print(f"😱: function name did not match")
            continue

        # handling result/final step
        if parsed_response.get("step") == "result":
            print(f"🤖: {parsed_response.get("content")}")
            break
        print(f"🧠: {parsed_response.get("content")}")

    except Exception as e:
        print(f"Something went wrong!!:  {e}")

    # print(chat_response.choices[0].message.content)