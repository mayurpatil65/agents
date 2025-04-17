"""
ConversationBufferMemory	Stores full conversation
ConversationSummaryMemory	Summarizes old chats
ConversationKGMemory	Stores info as knowledge graph
VectorStoreRetrieverMemory	Queries past conversation chunks
"""
from langchain_community.graphs.networkx_graph import KnowledgeTriple
from langchain_community.memory.kg import ConversationKGMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama

llm3 = ChatOllama(model="llama3.2:1b")

memory3 = ConversationKGMemory(llm=llm3, return_messages=True)
chain3 = ConversationChain(llm=llm3, memory=memory3)

chain3.invoke({"input": "Alice is a data scientist who lives in Berlin."})

print("Extracted Triplets:", memory3.kg.get_triples())

triple = KnowledgeTriple(subject="Alice", predicate="lives in", object_="Berlin")
memory3.kg.add_triple(triple)

print("Extracted Triplets:", memory3.kg.get_triples())

response = chain3.invoke({"input": "Where does Alice live?"})
print(response)