from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel
from langchain_core.tools import tool
import json
from pydantic import BaseModel, Field

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pickle
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient 
from langchain_qdrant import QdrantVectorStore

if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

openai_api_key = os.environ["OPENAI_API_KEY"]

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class QuestionOutput(BaseModel):
    question: str = Field(description="Model generated question")
    difficulty: str = Field(description="Difficulty level of the question (easy, medium, hard)")

class ToolOutput(BaseModel):
    questions: list[QuestionOutput] = Field(description='A list of generated questions with their difficulty levels')

ouput_parser = PydanticOutputParser(pydantic_object=ToolOutput)

@tool
def multi_query(query: str) -> list:
    """
        You are given with an abstract user query. Generater multiple queries to make it more specific and context rich, still keeping the user's intent.

        Args: 
            query: original query form the user which is of string type

        IMPORTANT : Respond with only python list type, other datatypes are strictly prohibited
    """
    
    system_prompt = """
        You are given with an abstract user query. Generater multiple queries to make it more specific and context rich, still keeping the user's intent. 

        Return all the queries in a python list type, other datatypes are strictly prohibited.

        RULES : 
        - Donot hellucinate anything.
        - Respond with multiple re-written queries in a list as response.
        - Donot respond with a single query.
        - IMPORTANT : Queries should strictly be of type list not of any other data-type.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{topic}"),
        ("system", "{format_instructions}")
    ])

    # formated_prompt = prompt.format(
    #     topic = query,
    #     format_instructions = ouput_parser.get_format_instructions()
    # )

    # print(f"FORMATTED PROMPT : {formated_prompt}")

    chain = prompt | model | ouput_parser

    result = chain.invoke({
        "topic": query,
        "format_instructions": ouput_parser.get_format_instructions()
    })

    final_list_result = []

    # print(f"FUNCTION RESULT: {result.questions}")
    # print(f"FUNCTION RESULT: {type(result)}")

    for item in result.questions:
        final_list_result.append(item.question)

    # print(f"this is the final list result:  {final_list_result}")

    return final_list_result

tools = [multi_query]

available_tools = {
    "multi_query": multi_query
}

# PDF LOADING AND TEXT SPLITTING 

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

embed = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=openai_api_key
)

print(f"SPLITTED TEXT : {len(splitted_text)}")

# embed = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-exp-03-07"
# )

URL = "http://localhost:6333"
COLLECTION = "AI-engineering-rag"

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

system_prompt = PromptTemplate.from_template(
    """
        You are an intelligent ai assistant expert in analysing user queries.
        
        For a given user query analyse the ambiguity of it first. If the query is more ambiguous or you think it needs more elaboration for better responses, call the tools from the available tools.

        Available Tools: 
        {available_tools}

        Rules: 
        - Strictly pass the original query, sent by the user, to the choosen tool call as an argument.
        - Make a tool call only if it is necessary.
        - Donot hellucinate or break any rules.
    """
)

model_with_tools = model.bind_tools(tools)

while True:
    relevant_docs = []
    try:
        user_query = input("> ")
        if not user_query.strip():
            print("Please enter your query")
            continue
    except KeyboardInterrupt:
        break

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.format(available_tools = tools)),
            ("human", user_query)
        ]
    )

    chain1 = prompt | model_with_tools

    result = chain1.invoke({})


    if result.tool_calls:
        tool_data = result.tool_calls[0]

        function_name_string = tool_data.get("name").strip()

        function_object = available_tools[function_name_string].func

        function_args = tool_data.get('args')

        tool_response = function_object(**function_args)

        # print('TOOL RESPONSE TYPE : ', type(tool_response))  # CONFIRMED THAT THIS IS OF LIST TYPE

        print(f'🤖: {tool_response} {len(tool_response)} \n \n \n \n ')

        for query in tool_response:
            docs = retriever.invoke(query)
            relevant_docs.append(docs)

        # print(f"RELEVANT DOCS: {relevant_docs[:5]}")
        for docs in relevant_docs:
            # print(f"{docs} \n \n \n \n ")
            # print( f'{index}. {doc.page_content}' for index, doc in enumerate(docs) )
            # print(doc for doc in docs)
            context = "\n\n---\n\n".join(doc.page_content for doc in docs)

        print("THIS IS THE FINAL CONTEXT : 💀", context)

    else :
        print(f'🤖: {result.content}')