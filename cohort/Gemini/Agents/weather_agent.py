from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import json
import re

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
1. Output should be strictly in json format.
2. Always perform only one step at a time and wait for the next input.
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
]

while True:
    user_query = input("> ")
    messages.append({
        "role": "user",
        "content": user_query
    })

    while True:
        result = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            tools=tools,
        )

        try:
            response = result.choices[0].message
            print(f"INITIAL RESPONSE : {response}")

            # Check if content exists before processing
            if response.content is not None:
                response_text = response.content
                # Clean possible JSON fences
                cleaned = re.sub(r"```json|```", "", response_text.strip())
                parsed_response = json.loads(cleaned)
                
                step = parsed_response.get("step")
                content = parsed_response.get("content")

                if step == "execute":
                    if response.tool_calls == None:
                        messages.append(
                            {
                                "role": "assitant",
                                "content": "ERROR: Tool call must happen in the execute step. Choose an appropriate tool and call it."
                            }
                        )
                    continue

                messages.append({
                    "role": "assistant",
                    "content": json.dumps(parsed_response)
                })
                print(f"🧠: STEP: {step} CONTENT: {content}")
                continue
                
            else:
                print("Response content is None")
                # Check for tool_calls directly
                if response.tool_calls:
                    print("✅ tool_calls detected")
                    print(f"tool_calls: {response.tool_calls}")
                    break  # Exit the inner loop
                elif response.function_call:
                    print("✅ function_call detected")
                    print(f"function_call: {response.function_call}")
                    break  # Exit the inner loop
                else:
                    print("No content and no tool calls")
                    # You might want to add some handling here
                    break  # Or continue based on your logic

        except Exception as e:
            print(f"JSON error: {e}")
            break
