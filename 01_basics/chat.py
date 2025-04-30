import os
from mistralai import Mistral
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]


api_key = os.environ["MISTRAL_API_KEY"]
model = "open-codestral-mamba"

client = Mistral(api_key=api_key)


chat_response = client.chat.complete(
    model = model,
    messages = [
        {
            "role": "user",
            "content": "what is a black hole.",
        },
    ]
)

# print('chat response', chat_response, )

print(chat_response.choices[0].message.content)


# streamline response 

# stream_response = client.chat.stream(
#     model = model,
#     messages = [
#         {
#             "role": "user",
#             "content": "What is the best French cheese?",
#         },
#     ]
# )

# for chunk in stream_response:
#     print(chunk.data.choices[0].delta.content)
