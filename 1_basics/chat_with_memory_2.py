"""
ConversationBufferMemory	Stores full conversation
ConversationSummaryMemory	Summarizes old chats
ConversationKGMemory	Stores info as knowledge graph
VectorStoreRetrieverMemory	Queries past conversation chunks
"""

from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama

llm2 = ChatOllama(model="llama3.2:1b")

memory2 = ConversationSummaryMemory(llm=llm2, return_messages=True)
chain2 = ConversationChain(llm=llm2, memory=memory2)

chain2.invoke({"input": "My name is Mayur and I work for the US Navy."})
response = chain2.invoke({"input": "Where do I work?"})
print(response)