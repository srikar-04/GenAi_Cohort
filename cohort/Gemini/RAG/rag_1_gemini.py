from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import os
import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



from qdrant_client import QdrantClient 
from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore

gemini_api_key = os.environ["GEMINI_API_KEY"]

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

splitted_text = text_splitter.split_documents(docs)

embed = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

URL = "http://localhost:6333"
COLLECTION = "embedded-system-rag-new-version-gemini"

qclient = QdrantClient(url=URL)

collections = [c.name for c in qclient.get_collections().collections]

need_ingestion = (
    (COLLECTION not in collections ) or (qclient.count(collection_name=COLLECTION).count == 0)
)


if need_ingestion:
    qdrant_vector_store = QdrantVectorStore.from_documents(
        documents=splitted_text,
        embedding=embed,
        collection_name = COLLECTION,
        url = URL,
    )
else:
    print(f"⏭ Skipping ingest; {COLLECTION} already has data")


qdrant_vector_store = QdrantVectorStore(
    client=qclient,
    collection_name=COLLECTION,
    embedding=embed
)

retriever = qdrant_vector_store.as_retriever()

client = OpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

system_prompt = """
    You are an expert assistant. Answer using *only* the context below.
    If the answer can’t be found, say “I’m sorry, but I don't have that information in the provided context.”

    AVAILABLE CONTEXT: 
    {context}
"""

while True: 
    user_query = input(">> ")

    relevant_docs = retriever.invoke(user_query)

    context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

    # print(context)
    formatted_prompt = system_prompt.format(context=context)

    messages = [
        {
            "role": "developer",
            "content": formatted_prompt
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        input=messages,
        max_output_tokens=500,
        temperature=0.1,
    )

    print("\n=== Answer ===\n")
    print(response.choices[0].message.content)