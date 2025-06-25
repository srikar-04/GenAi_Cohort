# TONE CHANGER USING LLM WITH FEW SHOT PROMPTING

import getpass
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder

# gemini_api_key = os.environ["GEMINI_API_KEY"]

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_message = ("system", "You are an helpful ai assistan who is a pro in changing the tone of the message sent by the user. you change the tone to look more clear and polite manner without changing the original intent of the message")

examples = [
    {
        "input": "wanna meet tommorrow?",
        "output": "would you be available to meet tommorrow?"
    },
    {
        "input": "completed today's task?",
        "output": "Did you complete the task assigned to you today?"
    },
    {
        "input": "stop bothering me!",
        "output": "could you give me a little space now? I would really appreciate that."
    },
    {
        "input": "This is your fault",
        "output": "I believe there might have been a misunderstanding, let’s work through it together."
    }
]

# This is like defining the schema for every example
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}")
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        system_message,
        MessagesPlaceholder(variable_name="examples"),
        ("human", "{input}")
    ]
)

user_input = {"input": input("> ")}

messages = final_prompt.format_messages(
    input = user_input["input"],
    examples = few_shot_prompt.format_messages()
)

response = llm.invoke(messages)

print(response.content)