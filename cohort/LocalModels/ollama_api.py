from fastapi import FastAPI, Body
from ollama import Client

app = FastAPI()

client = Client(
    host="http://localhost:11434"
)

client.pull('gemma3:1b')

@app.post("/chat")
def chat(message: str = Body(..., description="Chat Message")):
    # passing user logic to ollama container
    response = client.chat(
        model="gemma3:1b",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )

    return response['message']['content']
