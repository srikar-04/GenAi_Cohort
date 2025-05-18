from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import json
import re

gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
model = "gemini-2.0-flash"

client = OpenAI(
    api_key=gemini_api_key,
    base_url=gemini_base_url,
)

def get_weather(city: str) -> str:
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
2. Always perform only one step at a time and wait for the next input.
3. Carefully analyze the user query.
4. Do not wrap the response in a code block or fences.
5. During the "execute" step, you MUST invoke the tool by returning a tool call in the OpenAI-compatible format (not just describe it in JSON). The JSON output for "execute" should be minimal, e.g., {"step": "execute", "content": "Invoking tool"}.
6. For all other steps, output the appropriate JSON format with "step" and "content" fields.
7. NEVER provide an answer that requires tool information without first calling the appropriate tool.

Output Format for Non-Execute Steps:
{
    "step": "step name",
    "content": "step description"
}

Output Format for Execute Step (before tool call):
{
    "step": "execute",
    "content": "Invoking tool"
}

Example:
User Query: What is the weather in New York?
- {"step": "start", "content": "Alright! The user is interested in a weather query and is asking about the weather in New York"}
- {"step": "plan", "content": "To get the weather in New York, I need to call the get_weather function"}
- {"step": "execute", "content": "Invoking tool"} (followed by a tool call)
- {"step": "monitor", "content": "Weather in New York is very hot today, 31 degrees Celsius."}
- {"step": "result", "content": "31 degrees Celsius."}

Tools:
- get_weather: "Returns the weather in a given location"
- input: "Takes input from the user query"
"""

messages = [
    {"role": "system", "content": system_prompt},
]

# Get user input
user_query = input("> ")
messages.append({
    "role": "user",
    "content": user_query
})

current_step = "start"

while True:
    # Default to "auto" tool choice, but force "required" during execute step
    tool_choice = "required" if current_step == "execute" else "auto"

    result = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        tools=tools,
        tool_choice=tool_choice,
    )

    try:
        response = result.choices[0].message
        print(f"INITIAL RESPONSE: {response}")

        if response.tool_calls:
            # Handle tool call (should only happen in execute step)
            tool_call = response.tool_calls[0]
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool_result = available_tools[function_name](**arguments)

            messages.append({
                "role": "assistant",
                "content": json.dumps({"step": "execute", "content": "Invoking tool"})
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": tool_result
            })
            current_step = "monitor"  # Move to next step after tool call
            continue

        elif response.content is not None:
            # Parse JSON response for non-tool-call steps
            cleaned = re.sub(r"```json|```", "", response.content.strip())
            print(f"Cleaned content: {cleaned}")
            try:
                parsed_response = json.loads(cleaned)
                current_step = parsed_response["step"]
                messages.append({
                    "role": "assistant",
                    "content": json.dumps(parsed_response)
                })
                
                if current_step == "result":
                    print("Final Result:", parsed_response["content"])
                    break
            except json.JSONDecodeError as e:
                print("⚠️ JSON parsing error:", e)
                print("Content was:", response.content)
                break
        else:
            print("❌ Unexpected: Content is None and no tool_calls present.")
            break
    except Exception as e:
        print(f"Error: {e}")
        break