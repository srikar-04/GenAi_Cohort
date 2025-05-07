from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
import json

class Weather_Schema(BaseModel):
    step: str
    content: str

def get_weather():
    return "31 degrees"

api_key = os.environ['GEMINI_API_KEY']

client = genai.Client(api_key=api_key)

prompt = "what is 2 + 2"

user_defined_contents = [
    types.Content(
        role='user',
        parts=[types.Part.from_text(text=prompt)]
    ),
    types.Content(
        role='assistant',
        parts=[types.Part.from_text(text=json.dumps({'step': 'start', 'content': 'The user wants to know the result of a simple addition problem.'}))]
    ),
    types.Content(
        role='assistant',
        parts=[types.Part.from_text(text=json.dumps({'step': 'plan', 'content': 'To answer the question, I need to perform the calculation 2 + 2.'}))]
    ),
    # types.Content(
    #     role="assistant",
    #     parts=[types.Part.from_text(text=json.dumps({'step': 'execute', 'function': 'get_weather', 'input': 'new york'}))]
    # ),
]

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
        function: 'name of function if the step name is execute,
        input: 'input to the function if the step name is execute'
    }

    Example:
    User Query: What is the weather in new york?
    Output: {step: "start", content: "Alright! the user is interested in weather query and he is asking about weather in new york"}
    Output: {step: "plan", content: "To get the weather in new york, i need to call get_weather function"}
    Output: {step: "execute", function: "get_weather", input: "new york"}
    Output: {step: "monitor", content: "Weather in new york is very hot today, 12 degree celcius."}
    Output: {step: "result", content: "Weather in new york is 12 degrees."}
"""

text_response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=user_defined_contents,
    config = types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=list[Weather_Schema]
    ),
)

try:
    result = json.loads(text_response.text)
    print(result)
except Exception as e:
    print('something went wrong', e)