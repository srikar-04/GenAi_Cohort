# Taking a document, initializing it, loading it and then splitting the doucment as characters using RecursiveCharacterTextSplitter or as docs.
# Embed the splitted doc and then store the embeddings in qdrant db using docker container

from dotenv import load_dotenv
load_dotenv()
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient 


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
    chunk_size=5000,
    chunk_overlap=300,
)

splitted_text = text_splitter.split_documents(documents=docs)

# print(len(splitted_text))

# for chunk in texts:
#     pprint.pp(chunk.page_content)

# create embeddings and store in qdrant db
embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    collection_name="learning_langchain",
    url="http://localhost:6333",
    embedding=embed,
)

vector_store.add_documents(documents=splitted_text)
print("Injection done!!!!")