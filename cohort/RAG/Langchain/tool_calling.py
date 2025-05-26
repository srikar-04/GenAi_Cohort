from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import json

openai_api_key = os.environ["OPENAI_API_KEY"]

@tool
def get_weather(location:str)->str:
    """
        Gets weather based on the given location
        Args:
        [location]
    """
    return f"current weather in {location} is 31 degrees"

@tool
def run_command(command):
    """
        Execute the command without any hellucinations
        Args:
        [command]
    """
    return "command ran succesfully"

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
}

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

tools = [get_weather, run_command]

llm_with_tools = llm.bind_tools(tools)

user_query = input(">> ")

ai_message = llm_with_tools.invoke([user_query])

print(f"ACTUAL TOOL CALL RESPONSE : {ai_message.tool_calls}")

if ai_message.tool_calls:
    name = ai_message.tool_calls[0].get("name")
    function_name = available_tools.get(name)

    function_args = ai_message.tool_calls[0].get("args")
    
    print(f'FUNCTION_NAME: {function_name} \n FUNCTION_ARGUMENTS: {function_args}')
    print(f"TYPE OF ARGS: {type(function_args)}")

    tool_response = function_name.invoke(function_args)
    print(f'THIS IS THE RESPONSE FROM FUNCTION: {tool_response}')
else:
    print("OOPS! NO TOOL CALL PRESENT")