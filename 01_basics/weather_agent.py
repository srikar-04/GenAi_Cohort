# In this section we are creating a weather agent using gemini api.
# We are using a technique called "function calling" (present in gemini docs).
# Function calling lets llms to connect to external information by doing api calls, database calls, etc by using a function.

from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
import json
from typing import Optional

class Weather_Schema(BaseModel):
    step: str
    content: str
    function: Optional[str] = None
    input: Optional[str] = None

def get_weather(city: str) -> str:
    print(f"Getting weather for {city}")
    return f"weather in {city} is 31 degree celcius"

get_weather_function = {
    "name": "get_weather",
    "description": "Returns the weather for a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "Name of the city in user query (e.g., New York)"
            }
        },
        "required": ["city"]
    },
}

api_key = os.environ['GEMINI_API_KEY']

if not api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable")

client = genai.Client(api_key=api_key)
tools = types.Tool(function_declarations=[get_weather_function])

system_prompt = """
    You are an helpful AI assistant who is expert in breaking down complex problems and then resolve the user query.
    You work on start, plan, execute and monitor modes.
    For the given user query and all available tools, plan step by step execution. Based on planning, select relevant tool from the available tools. Perform the task based on the selected tool.
    Wait for observation and based on observation from tool call resolve the user query.

    Follow the steps in sequence that is "start", "plan", "execute", "monitor", and "result".
    
    Rules:
    1. Output should be strictly in JSON format.
    2. Strictly perform only one step at a time and wait for next input. Remember this rule very well
    3. Carefully analyse the user query

    Output JSON Format:
    {
        step: 'step name',
        content: 'step description',
        function: 'name of function specified in Tools section. Specify it only if the step is "execute"',
        input: 'input to the function present in user query. Specify it only if the step is "execute"'
    }

    Example:
    User Query: What is the weather in new york?
    Output: {step: "start", content: "Alright! the user is interested in weather query and he is asking about weather in new york"}
    Output: {step: "plan", content: "To get the weather in new york, i need to call get_weather function"}
    Output: {step: "execute", function: "get_weather", input: "new york"}
    Output: {step: "monitor", content: "Weather in new york is very hot today, 12 degree celcius."}
    Output: {step: "result", content: "12 degrees."}

    Tools:
    - get_weather: "Returns the weather in a given location"
    - input: "Takes input from the user query"
"""

user_defined_contents = []

user_query = input('Enter your query \n > ')

user_defined_contents.append(
    types.Content(
        role='user',
        parts=[types.Part.from_text(text=user_query)]
    ),
)

while True:
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=user_defined_contents,
        config = types.GenerateContentConfig(
            max_output_tokens=500,
            temperature=0.1,
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=list[Weather_Schema],
            tools=[tools],
        ),
    )

    try:
        result = json.loads(response.text)
        current_step = result[0]['step']


        
        # checking for the execute part
        if current_step == "execute":
            function_name = result[0].get("function")
            function_input = result[0].get("input")

            if function_name == "get_weather":
                function_result = get_weather(function_input)

                user_defined_contents.append(
                    types.Content(
                        role='assistant',
                        parts=[types.Part.from_text(text=f"function get_weather returned: {function_result}")]
                    ),
                )
                continue
            else:
                raise ValueError(f"unknown function: {function_name}")

        if current_step == 'result':
            print(f"🤖: {result[0]["content"]}")
            break

        print(f"🧠 {result[0]["content"]}")
        user_defined_contents.append(
            types.Content(
                role='assistant',
                parts=[types.Part.from_text(text=json.dumps(result[0]))]
            ),
        )        

    except Exception as e:
        print('something went wrong', e)
        break
