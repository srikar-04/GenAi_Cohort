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
    2. Strictly perform one step at a time and wait for next input
    3. Carefully analyse the user query
    4. Validate the output before giving answer

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
# prompt = 'what is life?'

user_defined_contents = [
    types.Content(
        role='user',
        parts=[types.Part.from_text(text=prompt)]
    )
]


while True:
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=user_defined_contents,
        config = types.GenerateContentConfig(
            max_output_tokens=500,
            temperature=0.1,
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=list[Math_Operation]
        ),    
    )

    try:
        text_response = response.candidates[0].content.parts[0].text
        result = json.loads(text_response)
        # print(result)
        if result[0]['step'] != 'result':
            print(f"🧠 {result[0]['content']}")
            user_defined_contents.append(
                types.Content(
                    role='assistant',
                    parts=[types.Part.from_text(text=json.dumps(result[0]))]
                )
            )
            continue
        else:
            print(f"🤖 {result[0]['content']}")
            break
    except Exception as e:
        print(f"something went wrong: {e}")
        break
