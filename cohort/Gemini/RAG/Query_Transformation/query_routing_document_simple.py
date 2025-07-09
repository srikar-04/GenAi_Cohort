from typing import List, Literal
from dotenv import load_dotenv
import langchain_core
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field, RootModel

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient 
from langchain_qdrant import QdrantVectorStore

from langchain_community.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
import math
from sklearn.cluster import KMeans


if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

openai_api_key = os.environ["OPENAI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
# model = ChatOpenAI()

cache_path = Path("docs_cache.pkl")

if cache_path.exists():
    with open(cache_path, "rb") as f:
        docs = pickle.load(f)
else:
    file_path = Path(__file__).parent / "_OceanofPDF.com_AI_Engineering_Building_Applications_-_Chip_Huyen.pdf"
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

print(f"SPLITTED TEXT TYPE {type(splitted_text[0])}")

embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key
)

# class CollectionName(BaseModel):
#     collection_name: str = Field('collection name based on the avaible summary text of the content')

# collection_output_parser = PydanticOutputParser(pydantic_object=CollectionName)

# 1. EMBEDDING THE SPLITTED TEXT : 
docs_embeddings  = embed.embed_documents([content.page_content for content in splitted_text])

print('CREATED DOCS EMBEDDINGS ✅')

# 2. PREDEFINED COLLECTION NAMES AND ITS EMBEDDINGS : 
collection_names = ["model-adaptation-and-application-development", "model-optimization", "foundation-models", "synthetic-data-strategies"]

collection_names_embeddigs = embed.embed_documents(collection_names)
print('CREATED COLLECTION NAMES EMBEDDINGS ✅')

collection_doc_map = {}

for idx, doc in enumerate(docs_embeddings):
    similarity = [cosine_similarity([doc], [collection]) for collection in collection_names_embeddigs]
    max_index = similarity.index(max(similarity))
    key = collection_names[max_index]
    # print('COLLECTION CREATED ☑️')
    if  key in collection_doc_map.keys():
        collection_doc_map[key].extend([splitted_text[idx]])
    else:
        collection_doc_map[key] = [splitted_text[idx]]

print(collection_doc_map.keys())

URL = "http://localhost:6333"
qclient = QdrantClient(url=URL)

exsisting_collection_names = [c.name for c in qclient.get_collections().collections]


for collection_name, docs in collection_doc_map.items():
    # print(collection_doc_map[collection_name][0])
    # break
    need_ingestion = (
        (collection_name not in exsisting_collection_names) or (qclient.count(collection_name=collection_name).count == 0)
    )
    if need_ingestion:
        print('⚒️ Creating New Collection')
        qdrant_vector_store = QdrantVectorStore.from_documents(
            documents=collection_doc_map[collection_name],
            embedding=embed,
            collection_name = collection_name,
            url = URL,
        )
    else:
        print(f" skipping collection, already exsists")