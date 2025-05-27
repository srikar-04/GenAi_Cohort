# from dotenv import load_dotenv
# load_dotenv()
# import os
# from openai import OpenAI

# gemini_api_key = os.environ["GEMINI_API_KEY"]
# gemini_model = "gemini-2.0-flash"

# client = OpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
# )

# system_prompt = f"""
#     You are an helpful ai assistant
# """

# messages = [
#     {
#         "role": "developer",
#         "content": system_prompt
#     }
# ]

# while True:
#     print("type `exit` to exit loop")

#     user_query = input(">> ")

#     if user_query == 'exit':
#         break

#     messages.append({
#         "role": "user",
#         "content": user_query
#     })

#     response = client.chat.completions.create(
#         model=gemini_model,
#         messages=messages
#     )

#     print(response.choices[0].message.content)






def rewrite_query(query, llm):
    """Rewrites the query using an LLM."""
    prompt = PromptTemplate(
        input_variables=["query"],
        template="Rewrite the following user query to be more clear and specific for better search results:\n\n{query}\n\nRewritten Query:",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    rewritten_query = chain.run(query)
    return rewritten_query.strip()


def process_query(query, retriever, llm, source_filter=None):
    if retriever is None:
        return "Please upload documents first."

    # Rewrite the query
    rewritten_query = rewrite_query(query, llm)
    print(f"Original Query: {query}")
    print(f"Rewritten Query: {rewritten_query}")

    if source_filter:
        results = retriever.get_relevant_documents(rewritten_query, where={"source": source_filter})
    else:
        results = retriever.get_relevant_documents(rewritten_query)

    context = "\n".join([doc.page_content for doc in results])
    return context


# Ollama setup
LLM_MODEL = "llama3"
ollama_llm = Ollama(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=LLM_MODEL)