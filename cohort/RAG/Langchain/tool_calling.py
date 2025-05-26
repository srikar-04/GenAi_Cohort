from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
import requests
import subprocess

openai_api_key = os.environ["OPENAI_API_KEY"]

@tool
def get_weather(location:str)->str:
    """
        Gets weather based on the given location
        Args:
        [location]
    """
    print(f"⛏️: running get weather function")
    url = f"https://wttr.in/{location}?format=%C+%t"
    response = requests.get(url)

    if response.status_code == 200:
        return f"weather in {location} is {response.text}"
    else:
        return "something went wrong with weather api"

@tool
def run_command(command):
    """
        Execute the command without any hellucinations
        Args:
        [command]
    """
    print(f"⛏️: running execute command function !!!!")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result)
    if result.returncode == 0:
        return f"Command executed successfully:\n{result.stdout.strip()}"
    else:
        raise ValueError("Execute command function failed!!")

available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
}

tools = [get_weather, run_command]

system_prompt = f"""
    you are an helpful ai assistant.
    you can resole user queries and answer based with availbe tools you have
    Here are the available list of tools
    Available Tools:
    {available_tools}

    Responses from specific tool calls are added to "messages" list
    Based on that list give me the final answer
"""

messages = [
    {
        "role": "developer",
        "content": system_prompt
    }
]

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools(tools)

while True:
    user_query = input(">> ")

    messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    ai_message = llm_with_tools.invoke([user_query])
    # messages.append(
    #     {
    #         "role":"assistant",
    #         "content": ai_message
    #     }
    # )

    print(f"ACTUAL TOOL CALL RESPONSE : {ai_message.tool_calls}")

    if ai_message.tool_calls:
        for tool in ai_message.tool_calls:
            name = tool.get("name")
            function_name = available_tools.get(name)

            function_args = tool.get("args")

            # print(f'FUNCTION_NAME: {function_name} \n FUNCTION_ARGUMENTS: {function_args}')
            # print(f"TYPE OF ARGS: {type(function_args)}")

            tool_response = function_name.invoke(function_args)
            print(f'THIS IS THE RESPONSE FROM FUNCTION: {tool_response}')
            messages.append(
                {
                    "role": "assistant",
                    "content": tool_response
                }
            )
    else:
        print("OOPS! NO TOOL CALL PRESENT")
        # print(f"NO TOOL RESPONSE: {ai_message.content}")
        messages.append(
            {
                "role": "assistant",
                "content": ai_message.content
            }
        )

    response = llm.invoke(messages)

    print(f"FINAL RESPONSE 🤖 : {response.content}")