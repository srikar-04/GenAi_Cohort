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
from qdrant_client import QdrantClient 

from langchain_community.vectorstores import Qdrant


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
    docs = []

    Qdrant.from_documents(
        documents=docs,
        embedding=embed,
        url="http://localhost:6333",
        collection_name="embedded-system-rag",
    )
    print(f"✅ Ingested {len(docs)} chunks into embedded-system-rag")
else:
    print(f"⏭ Skipping ingest; embedded-system-rag already has data")


qdrant_vs = Qdrant(
    url="http://localhost:6333",
    collection_name="embedded-system-rag",
    embedding=embed,
)
retriever = qdrant_vs.as_retriever()

user_query = input(">> ")

relevant_docs = retriever.get_relevant_documents(user_query)

context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

print(f"CONTEXT : {context}")

# system_prompt = f"""
#     You are an expert assistant. Answer using *only* the context below.
#     If the answer can’t be found, say “I’m sorry, but I don't have that information in the provided context.”

#     AVAILABLE CONTEXT: 
#     {context}
# """

# client = OpenAI()

# response = client.responses.create(
#     model="gpt-4.1",
#     input=user_query,
#     max_output_tokens=500,
#     temperature=0.1,
# )

# print("\n=== Answer ===\n")
# print(response.output_text)







# import os
# import pickle
# from pathlib import Path

# from dotenv import load_dotenv
# from qdrant_client import QdrantClient

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# # from langchain_community.vectorstores import Qdrant
# from langchain_qdrant import Qdrant

# # ─── Configuration ─────────────────────────────────────────────────────────────

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PDF_PATH       = Path(__file__).parent / "introduction-to-embedded-systems.pdf"
# CACHE_PATH     = Path("docs_cache.pkl")
# QDRANT_URL     = "http://localhost:6333"
# COLLECTION     = "embedded-system-rag"

# # ─── 1) Load or cache PDF chunks ────────────────────────────────────────────────

# if CACHE_PATH.exists():
#     docs = pickle.loads(CACHE_PATH.read_bytes())
# else:
#     loader   = PyPDFLoader(str(PDF_PATH))
#     raw_docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     docs      = splitter.split_documents(raw_docs)

#     CACHE_PATH.write_bytes(pickle.dumps(docs))
#     print(f"🔖 Cached {len(docs)} chunks")

# print(f"Total chunks to ingest (if needed): {len(docs)}")

# # ─── 2) Prepare embeddings & Qdrant client ────────────────────────────────────

# embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)
# qclient    = QdrantClient(url=QDRANT_URL)

# # ─── 3) Idempotent ingestion check ─────────────────────────────────────────────

# existing = [c.name for c in qclient.get_collections().collections]
# empty    = qclient.count(collection_name=COLLECTION).count if COLLECTION in existing else 0

# need_ingest = (COLLECTION not in existing) or (empty == 0)

# if need_ingest:
#     Qdrant.from_documents(
#         documents=docs,
#         embedding=embeddings,
#         url=QDRANT_URL,
#         collection_name=COLLECTION,
#     )
#     print(f"✅ Ingested {len(docs)} chunks into '{COLLECTION}'")
# else:
#     print(f"⏭️  Skipping ingest; '{COLLECTION}' already has data ({empty} vectors)")

# # ─── 4) Build retriever ───────────────────────────────────────────────────────

# vectorstore = Qdrant(
#     collection_name=COLLECTION,
#     embeddings=embeddings,
# )
# retriever = vectorstore.as_retriever()

# # ─── 5) Query & assemble context ──────────────────────────────────────────────

# query = input("\n🔎 Your question: ")
# relevant_docs = retriever.get_relevant_documents(query)

# context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)
# print("\n=== CONTEXT ===\n")
# print(context)
