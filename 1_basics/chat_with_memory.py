from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama

memory = ConversationBufferMemory(return_messages=True)
chat_model = ChatOllama(model="llama3.2:1b")

chain = ConversationChain(llm=chat_model, memory=memory, verbose=True)
chain.invoke({"input": "Hi, I'm Mayur."})
response = chain.invoke({"input": "What’s my name?"})
print(response)
i_o = input(": ")
print(chain.invoke({"input": i_o}))

"""
ConversationBufferMemory	Stores full conversation
ConversationSummaryMemory	Summarizes old chats
ConversationKGMemory	Stores info as knowledge graph
VectorStoreRetrieverMemory	Queries past conversation chunks
"""

