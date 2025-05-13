import os
from dotenv import load_dotenv
from mistralai import Mistral
load_dotenv()

api_key = os.environ['MISTRAL_API_KEY']

client = Mistral(api_key=api_key)

model = "mistral-embed"

prompt_1 = "hyderabadi biriyani is awesome. Nothing can beat it"
prompt_2 = "Black hole is a place where gravity is so strong that even light can't escape from it"

embeddings_response = client.embeddings.create(
    model=model,
    inputs=[prompt_1, prompt_2]
)

# print(embeddings_response.data)

print(f"Number of embedding vectors: {len(embeddings_response.data)}")   # we will be having 2 embeddings in the data because we gave 2 prompts to embedded

# this is the structure

# embedding_response -> data(list of 2 elements) -> 0th index contains embeddings of prompt_1 inside .embedding -> 1st index contains embeddings of prompt_2 inside .embedding

# try to print embedding_reponse for better understanding

for i, embedding_data in enumerate(embeddings_response.data):
    print(f"Embedding #{i+1} (first 5 dimensions): {embedding_data.embedding[:5]}")