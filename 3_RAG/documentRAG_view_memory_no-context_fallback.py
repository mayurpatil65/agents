import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory

# 1. Load all PDFs
def load_documents_from_folder(folder_path):
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
            all_docs.extend(loader.load())
    return all_docs

docs = load_documents_from_folder("pdfs")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

# 3. Vectorstore + retriever
embeddings = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# 4. LLM and Prompt
llm = ChatOllama(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use the context below to answer the user's question. If the context is empty, say you couldn't find anything.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# 5. Handle retrieval fallback logic
def safe_retriever(input_dict):
    query = input_dict["question"]
    docs = retriever.invoke(query)
    if not docs:
        return ["[NO_CONTEXT]"]  # signal for fallback
    return docs

# 6. Chain logic
rag_chain = (
    RunnableParallel({
        "context": safe_retriever,
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Memory setup
chat_histories = {}
def get_session_history(session_id: str):
    if session_id not in chat_histories:
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True
        )
        chat_histories[session_id] = memory.chat_memory
    return chat_histories[session_id]

# 8. Wrap with message history
chat_chain = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 9. CLI Chat
def run_qa():
    session_id = "user-1"
    print("🤖 3_RAG ChatBot with Summary Memory + Fallbacks")
    print("Type a question, or 'memory' to see what the bot remembers.\nType 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.strip().lower() == "exit":
            break

        if query.strip().lower() == "memory":
            history = get_session_history(session_id)
            print("\n🧠 Current Memory:\n")
            for msg in history.messages:
                role = msg.type.capitalize()
                print(f"{role}: {msg.content}")
            print("")
            continue

        response = chat_chain.invoke(
            {"question": query},
            config=RunnableConfig(configurable={"session_id": session_id})
        )

        if response.strip() == "[NO_CONTEXT]":
            print("Bot: 🤷 Sorry, I couldn't find anything related to your question in the documents.")
        else:
            print("Bot:", response)

if __name__ == "__main__":
    run_qa()
