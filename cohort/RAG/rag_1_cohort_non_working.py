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
from openai import OpenAI

from langchain_qdrant import QdrantVectorStore
from langchain.chains import retrieval_qa
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
    chunk_size=1000,
    chunk_overlap=200,
)

splitted_text = text_splitter.split_documents(docs)

print(len(splitted_text))
# print(splitted_text)  # CONTAINS PAGE_CONTENT <<---

# for chunk in texts:
#     pprint.pp(chunk.page_content)

# create embeddings and store in qdrant db
embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=api_key
)

# vector_store = QdrantVectorStore.from_documents(
#     documents=[],
#     collection_name="learning_langchain",
#     url="http://localhost:6333",
#     embedding=embed,
# )

# vector_store.add_documents(documents=splitted_text)
print("Injection done!!!!")



retrival = QdrantVectorStore.from_existing_collection(
    embedding=embed,
    collection_name="learning_langchain",
    url="http://localhost:6333",
)

qa = retrieval_qa(
    llm=OpenAI(temperature=0.1),
    chain_type="stuff",            # or "map_reduce" / "refine"
    retriever=retrival.as_retriever(),
    return_source_documents=True,  # <-- key to get back your PDF chunks
)

user_query = input(">> ")

result = qa({"query": user_query})

# 4) unpack results
print("\n=== ANSWER ===\n")
print(result["result"])

print("\n=== SOURCES ===")
for idx, doc in enumerate(result["source_documents"], 1):
    print(f"\nChunk {idx} (metadata={doc.metadata}):\n{doc.page_content[:300]}…")

# relevant_chunks = retrival.similarity_search(
#     query=user_query
# )

# print(f"RELEVANT CHUNKS LENGTH : {len(relevant_chunks)}")
# # print(relevant_chunks[0].page_content)
# for chunk in relevant_chunks:
#     print(f"CHUNKS : \n \n {chunk.page_content} \n \n \n")



# system_prompt = f"""
# You are a helpful AI assistant. Analyze the user's query and generate a response based only on the context provided below.
# You must ONLY use the text inside the "Available Context" section below to answer the user's question.
# Available Context:
# {''.join([doc.page_content for doc in relevant_chunks])}

# # Rules:
# # - If the question cannot be answered using the context, respond with: "I'm sorry, but I don't have that information in the provided context."
# # - Do not use your own knowledge, even if you know the answer.
# # """

# client = OpenAI()

# response = client.responses.create(
#     model="gpt-4.1",
#     input=user_query,
#     max_output_tokens=500,
#     temperature=0.1,
# )

# print(response.output_text)