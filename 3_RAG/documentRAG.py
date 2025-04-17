import os
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. Load all PDFs from the ./pdfs folder
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

# 3. Extract raw text + metadata from chunks (for from_texts)
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

# 4. Create vectorstore from text chunks
embeddings = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# 5. Create a 3_RAG chain: retrieval + prompt + LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question based on the context.\n\nContext:\n{context}"),
    ("user", "{question}")
])

llm = ChatOllama(model="llama3.2:1b")

rag_chain = (
    RunnableParallel({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | prompt
    | llm
    | StrOutputParser()
)

# 6. Simple CLI for Q&A
def run_qa():
    print("📚 Document Q&A Bot (LangChain v0.3 + Ollama + PDF 3_RAG)")
    print("Ask questions about the PDFs. Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.strip().lower() in {"exit", "quit"}:
            break

        answer = rag_chain.invoke(query)
        print("Bot:", answer)

if __name__ == "__main__":
    run_qa()
