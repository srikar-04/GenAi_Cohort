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
    """,
    input_variables=["summary_text"]
)

collection_chain = collection_prompt | model | collection_output_parser

cluster_chunk_names = {}

# GETTING THE DOC CLOSEST TO CENTROID TO DECIDE COLLECTION NAME 
# 4. GENERATING A COLLECTION NAME FOR USING THE CLUSTER :

for cluster_id, doc_in_cluster in clustered_chunks.items():
    # converting docs in cluster to embeddings : 
    cluster_embeddings = [embed.embed_query(doc.page_content) for doc in doc_in_cluster] 
    print("EMBEDDING INSIDE LOOP FOR {cluster_id} ☑️")

    # finding doc closest to cluster centroid : 
    closest_idx, _ = pairwise_distances_argmin_min(
        [kmeans.cluster_centers_[cluster_id]], cluster_embeddings
    )
    representative_doc = doc_in_cluster[closest_idx[0]]
    print(f"FOUND REPRESENTATIVE DOC FOR {cluster_id} ☑️")

    summary_text = representative_doc.page_content

    cluster_name = collection_chain.invoke({
        "summary_text": summary_text
    })

    cluster_chunk_names[cluster_name.collection_name] = doc_in_cluster

print(cluster_chunk_names)

 
# for cluster_id, doc_in_cluster in clustered_chunks.items():
#     summary_text = "\n \n".join(doc.page_content for doc in doc_in_cluster)
#     cluster_name = collection_chain.invoke({
#         "summary_text": summary_text
#     })
#     print(f"CLUSTER {cluster_id} -> {cluster_name.collection_name}")

# class CollectionName(BaseModel):
#     collection_name: str = Field('meaningful collection name based on the context of the text')
#     splitted_text: str = Field('text splitted using text splitter, categorizing under a collection name.')

# class CollectionChunkList(RootModel[List[CollectionName]]):
#     pass

# system_prompt_collection = """
#     You are an intelligent AI document categorizer. For each given text chunk, analyze the content deeply and assign a meaningful "collection_name" or category based on the topic or concept covered.

#     At the end, after assigning proper names to every chunk, return all the chunks in a list

#     Guidelines:
#     - Create only 3-4 major categories overall to keep collections manageable.
#     - Choose intuitive, short collection names (eg: 'AI Basics', 'ML Algorithms', 'Data Engineering', 'LLMs').
#     - Do NOT hallucinate or generate irrelevant categories.
#     - IMPORTANT: Donot create more than 4 collections.
#     - return python list at the end.
# """

# collection_output_parser = PydanticOutputParser(pydantic_object=CollectionChunkList)


# collection_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_prompt_collection),
#     ("human", '{chunk}'),
#     ("system", "{format_instructions}")
# ])

# collection_chain = collection_prompt | model | collection_output_parser

# BATCH_SIZE = 50
# total_chunks = len(splitted_text_page_content)
# batches = math.ceil(total_chunks / BATCH_SIZE)

# print(f"NO OF BATCHES : {batches}")

# results = []

# for i in range(3):
#     start = i * BATCH_SIZE
#     end = min((i+1)*BATCH_SIZE, total_chunks)
#     chunk_batch = splitted_text_page_content[start:end]

#     # chunk_text = "\n \n".join(chunk.page_content for chunk in chunk_batch)

#     chunk_result = collection_chain.invoke(
#         {
#             "chunk": chunk_batch,
#             "format_instructions": collection_output_parser.get_format_instructions()
#         }
#     )

#     print(f"THIS IS THE TYPE OF CHUNK RESULT IN THE LOOP : {type(chunk_result)}")
#     results.extend(chunk_result.root)

# print(f"THIS IS THE TYPE OF FINAL RESULTS TYPE : {type(results)}")

# print(results[0])