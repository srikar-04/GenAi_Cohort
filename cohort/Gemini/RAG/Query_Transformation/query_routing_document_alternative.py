from typing import List, Literal
from dotenv import load_dotenv
import langchain_core
from sklearn.metrics import pairwise_distances_argmin_min
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

class CollectionName(BaseModel):
    collection_name: str = Field('collection name based on the avaible summary text of the content')

collection_output_parser = PydanticOutputParser(pydantic_object=CollectionName)

# 1. EMBEDDING THE SPLITTED TEXT : 
embeddings  = embed.embed_documents([content.page_content for content in splitted_text])

print('CREATED EMBEDDINGS ✅')

# 2. CLUSTER THE EMBEDDINGS : 
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_ids = kmeans.fit_predict(embeddings)

print('CREATED EMBEDDINGS CLUSTERS AND OBTAINED IDS ✅')

# 3. MAPPING CHUNKS TO CLUSTER IDS
clustered_chunks = {}
for idx, cluster_id in enumerate(cluster_ids):
    clustered_chunks.setdefault(cluster_id, []).append(splitted_text[idx])
print('MAPPED SPLITTED TEXT TO SPECIFIC CLUSTERS ✅')

collection_prompt = PromptTemplate(
    template= """
        Generate a short collection name for the following texts: {summary_text}
        Generate only a single collection name and do not hellucinate.

        example -1:

        HUMAN : 

        summary_text : "Finetuning is the process of adapting a model to a specific task by further
        training the whole model or part of the model."

        AI RESPONSE : 
        {{
            "collection_name": "fine-tuning"
        }}

        example -2 :
        summary_text: " Prompt engineering refers to the process of crafting an instruction that gets
        a model to generate the desired outcome."

        AI RESPONSE : 
        {{
            "collection_name": "prompt-engineering"
        }}
        
        Rules : 
        1. Do not generate multiple collection names.
        2. Do not hellucinate and follow the instructions
        3. Work according to the pydantic model.
        4. IMPORTANT : Repond only in json format.

        IMPORTANT:
        - If matching an existing collection, output exactly as it is.
        - Output ONLY in JSON as per pydantic model.
        {exsisting_collections}
    """,
    input_variables=["summary_text", "exsisting_collections"]
)

collection_chain = collection_prompt | model | collection_output_parser

cluster_chunk_names = {}

# MAPPING DOC ID TO IT'S EMBEDDING
docid_to_embedding = {id(doc): emb for emb, doc in zip(embeddings, splitted_text)}

# GETTING THE DOC CLOSEST TO CENTROID TO DECIDE COLLECTION NAME 
# 4. GENERATING A COLLECTION NAME FOR USING THE CLUSTER :

for cluster_id, doc_in_cluster in clustered_chunks.items():
    # converting docs in cluster to embeddings : 
    cluster_embeddings = [docid_to_embedding[id(doc)] for doc in doc_in_cluster] 
    print(f"EMBEDDING INSIDE LOOP FOR {cluster_id} ☑️")

    # finding doc closest to cluster centroid : 
    closest_idx, _ = pairwise_distances_argmin_min(
        [kmeans.cluster_centers_[cluster_id]], cluster_embeddings
    )
    representative_doc = doc_in_cluster[closest_idx[0]]
    print(f"FOUND REPRESENTATIVE DOC FOR {cluster_id} ☑️")

    summary_text = representative_doc.page_content

    cluster_name = collection_chain.invoke({
        "summary_text": summary_text,
        "exsisting_collections": cluster_chunk_names.keys()
    })

    cluster_chunk_names[cluster_name.collection_name] = doc_in_cluster

print(cluster_chunk_names.keys())

