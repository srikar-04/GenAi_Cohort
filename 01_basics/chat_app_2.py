from dotenv import load_dotenv
load_dotenv()
import os
from google import genai
from google.genai import types
import json
from pydantic import BaseModel

class Math_Operation(BaseModel):
    step: str
    content: str

api_key = os.environ['GEMINI_API_KEY']

client = genai.Client(api_key=api_key)

system_prompt = """
    You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.

    For the given user input, analyse the input and break down the problem step by step.
    Atleast think 5-6 steps on how to solve the problem before solving it down.

    The steps are, you get a user input, you analyse, you think, you again think for several times and then return an output with explanation and then finally you validate the output as well before giving final result.

    Follow the steps in sequence that is "analyse", "think", "output", "validate" and finally "result".

    Rules:
    1. Output should be strictly in JSON format.
    2. Carefully analyse the user query

    Output Format:
    {step: 'step name', content: 'step description' }

    Example:
    Input: What is 2 + 2.
    Output: { step: "analyse", content: "Alright! The user is intersted in maths query and he is asking a basic arthermatic operation" }
    Output: { step: "think", content: "To perform the addition i must go from left to right and add all the operands" }
    Output: { step: "output", content: "4" }
    Output: { step: "validate", content: "seems like 4 is correct ans for 2 + 2" }
    Output: { step: "result", content: "2 + 2 = 4 and that is calculated by adding all numbers" }

"""

prompt = 'what is 45 * 20'

prompt_1 = "what is life?"

response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents = [prompt],
    config = types.GenerateContentConfig(
        max_output_tokens=500,
        temperature=0.1,
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=list[Math_Operation]
    ),    
)

try:
    result = json.loads(response.text)
    for res in result:
        if(res["step"] != 'result'):
            print(f"🧠 {res['content']}")
        else:
            print(f"🤖 {res['content']}")
    # print(result[0]['content'])
except:
    print('oops! ai has not given a proper json response !!')