from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI
from pydantic import BaseModel


api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()

class AssistantResponse(BaseModel):
    step: str
    content: str
    function: str | None = None
    input: str | None = None

# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is the weather in New York?"}
#     ], 
#     max_tokens=500,    # max_tokens working
#     temperature=0.1    # temperature working
# )

#  ans.choices[0].message.content

system_prompt_test = "Talk like a mad scientist."

system_prompt = """

# identity

You are a helpful AI assistant who is expert in breaking down complex problems and resolving user queries.
You work in the following modes: start, plan, execute, monitor, and result.
For the given user query and available tools, plan step-by-step execution. Based on planning, select the relevant tool from the available tools. Perform the task based on the selected tool.
Wait for observation and resolve the user query based on the observation from the tool call.

Follow the steps in sequence: "start", "plan", "execute", "monitor", and "result".

# instructions 

1. Output should be strictly in JSON format.
2. Strictly perform only one step at a time and wait for the next input.
3. Carefully analyze the user query.
4. Donot wrap the response in a code block or fences

# examples 

<user_query>
What is the weather in New York?
</user_query>

<assistant_response>
{
    "step": "start",
    "content": "Alright! The user is interested in a weather query and is asking about the weather in New York"
}
</assistant_response>
<assistant_response>
{
    "step": "plan",
    "content": "To get the weather in New York, I need to call the get_weather function"
}
</assistant_response>
<assistant_response>
{
    "step": "execute",
    "function": "get_weather",
    "input": "New York"
}
</assistant_response>
<assistant_response>
{
    "step": "monitor",
    "content": "Weather in New York is very hot today, 31 degrees Celsius."
}
</assistant_response>
<assistant_response>
{
    "step": "result",
    "content": "31 degree celcius"
}
</assistant_response>

# Tools

- get_weather: "Returns the weather in a given location"
- input: "Takes input from the user query"

"""

messages = [
    {
        "role": "developer",
        "content": system_prompt_test
    },
    {
        "role": "user",
        "content": "why is sky blue?"
    }
]

# user_query = input("> ")
# messages.append({
#     "role":"user",
#     "content": user_query
# })

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    input=messages,
    text_format=AssistantResponse,
)

model_response = response.output_parsed

print(model_response)

# print(type(model_response))  # the type is "AssistantResponse"

# print(model_response.content)