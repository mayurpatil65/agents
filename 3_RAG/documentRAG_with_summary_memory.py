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

# Load all PDFs from ./pdfs folder
def load_documents_from_folder(folder_path):
    all_docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
            all_docs.extend(loader.load())
    return all_docs

docs = load_documents_from_folder("pdfs")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create vectorstore
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

embeddings = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question using the context below. Maintain a helpful, conversational tone.\n\nContext:\n{context}\n"),
    ("user", "{question}")
])

# Local LLM
llm = ChatOllama(model="llama3.2:1b")

# 3_RAG chain
base_chain = (
    RunnableParallel({
        "context": lambda x: retriever.invoke(x["question"]),
        "question": lambda x: x["question"]
    })
    | prompt
    | llm
    | StrOutputParser()
)

# Memory store per session
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

# Wrap with memory
chat_chain = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# CLI loop
def run_qa():
    session_id = "user-1"
    print("🤖 PDF ChatBot with Summary Memory (LangChain v0.3 + Ollama)\nType 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.strip().lower() in {"exit", "quit"}:
            break

        response = chat_chain.invoke(
            {"question": query},
            config=RunnableConfig(configurable={"session_id": session_id})
        )

        print("Bot:", response)

if __name__ == "__main__":
    run_qa()
