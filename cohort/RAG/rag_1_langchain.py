# Taking a document, initializing it, loading it and then splitting the doucment as characters using RecursiveCharacterTextSplitter or as docs.
# Embed the splitted doc and then store the embeddings in qdrant db using docker container

from dotenv import load_dotenv
load_dotenv()
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


from qdrant_client import QdrantClient 
from langchain_qdrant import Qdrant

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


api_key = os.environ["OPENAI_API_KEY"]
cache_path = Path("docs_cache.pkl")

if cache_path.exists():
    with open(cache_path, "rb") as f:
        docs = pickle.load(f)
else:
    file_path = Path(__file__).parent / "introduction-to-embedded-systems-2nbsped-9339219686-9789339219680_compress.pdf"
    # Initialization
    loader = PyPDFLoader(file_path)
    # Loading
    docs = loader.load()
    with open(cache_path, "wb") as f:
        pickle.dump(docs, f)
    

# pprint.pp(docs[40].__dict__)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splitted_text = text_splitter.split_documents(docs)

# print(len(splitted_text))

# create embeddings and store in qdrant db
embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)

qdrant_client = QdrantClient(url="http://localhost:6333")

collections = [c.name for c in qdrant_client.get_collections().collections]  # gets the names of all the collections

# either the collection is not present or the collection is empty
need_ingestion = (
    ("embedded-system-rag" not in collections) or (qdrant_client.count(collection_name="embedded-system-rag").count == 0)
)

if need_ingestion:

    Qdrant.from_documents(
        documents=splitted_text,
        embedding=embed,
        url="http://localhost:6333",
        collection_name="embedded-system-rag",
        qdrant_client = qdrant_client
    )
    print(f"✅ Ingested {len(docs)} chunks into embedded-system-rag")
else:
    print(f"⏭ Skipping ingest; embedded-system-rag already has data")


qdrant_vs = Qdrant(                                     # USE qdrant_vector_store THIS IS DEPRICATED
    client=qdrant_client,
    collection_name="embedded-system-rag",
    embeddings=embed
)
retriever = qdrant_vs.as_retriever()

user_query = input(">> ")

relevant_docs = retriever.get_relevant_documents(user_query)   # USE invoke() method THIS IS DEPRICATED

context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

# print(f"CONTEXT LENGTH : {context}")

system_prompt = f"""
    You are an expert assistant. Answer using *only* the context below.
    If the answer can’t be found, say “I’m sorry, but I don't have that information in the provided context.”

    AVAILABLE CONTEXT: 
    {context}
"""

client = OpenAI()

messages = [
    {
        "role": "developer",
        "content": system_prompt
    }
]

messages.append({
    "role": "user",
    "content": user_query
})

response = client.responses.create(
    model="gpt-4.1",
    input=messages,
    max_output_tokens=500,
    temperature=0.1,
)

print("\n=== Answer ===\n")
print(response.output_text)


# PROMPT TEMPLATES OF LANGCHAIN

# system_message = SystemMessage(
#     content=(
#         "You are an expert assistant. "
#         "Answer the question *only* using the information in the context. "
#         "If the answer is not in the context, reply exactly: "
#         "'I'm sorry, but I don't have that information in the provided context.'"
#     )
# )

# # 2) Include the retrieved context + the user’s query
# human_message = HumanMessage(
#     content=f"Context:\n{context}\n\nQuestion:\n{user_query}"
# )

# # 3) Call the chat model with *only* those messages
# llm = ChatOpenAI(model="gpt-4", api_key=api_key, temperature=0)
# response = llm([system_message, human_message])

# print("\n=== Answer ===\n")
# print(response.content)
