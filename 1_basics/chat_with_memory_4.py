"""
ConversationBufferMemory	Stores full conversation
ConversationSummaryMemory	Summarizes old chats
ConversationKGMemory	Stores info as knowledge graph
VectorStoreRetrieverMemory	Queries past conversation chunks
"""

from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.vectorstores import Chroma

llm = ChatOllama(model="llama3.2:1b")
embeddings = OllamaEmbeddings(model="llama3.2:1b")

vectorstore = Chroma.from_texts(["placeholder"], embedding=embeddings)
retriever = vectorstore.as_retriever()

memory = VectorStoreRetrieverMemory(retriever=retriever)

chain = ConversationChain(llm=llm, memory=memory)

chain.invoke({"input": "I like philosophy and I enjoy reading Plato."})
response = chain.invoke({"input": "Suggest a book I might enjoy."})
print(response)