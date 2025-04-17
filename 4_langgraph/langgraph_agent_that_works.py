import os
from typing import TypedDict, Literal, Optional, List

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import chat_agent_executor

from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Load and prepare PDFs
def load_documents_from_folder(folder_path):
    docs = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(folder_path, file_name))
            docs.extend(loader.load())
    return docs

docs = load_documents_from_folder("pdfs")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

embeddings = OllamaEmbeddings(model="llama3.2:1b")
vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# 2. LLM and prompt
llm = ChatOllama(model="llama3.2:1b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# 3. LangGraph Agent State
class AgentState(TypedDict):
    messages: List
    context: Optional[str]
    branch: Literal["answer", "no_context", "memory"]

# 4. Branch Decision Logic
def router(state: AgentState) -> Literal["answer", "no_context", "memory"]:
    last_msg = state["messages"][-1].content.lower()
    if last_msg.strip() == "memory":
        return "memory"
    if state.get("context") == "[NO_CONTEXT]":
        return "no_context"
    return "answer"

# 5. Nodes
def retrieve_context(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    if not docs:
        return {"messages": state["messages"], "context": "[NO_CONTEXT]", "branch": "no_context"}
    context = "\n\n".join(doc.page_content for doc in docs[:4])
    return {"messages": state["messages"], "context": context, "branch": "answer"}

def answer_node(state: AgentState) -> AgentState:
    question = state["messages"][-1].content
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": state["context"]})
    text_response = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": state["messages"] + [AIMessage(content=text_response)],
        "context": state["context"],
        "branch": "answer"
    }

def fallback_node(state: AgentState) -> AgentState:
    msg = "🤷 Sorry, I couldn't find anything relevant in the documents."
    return {"messages": state["messages"] + [AIMessage(content=msg)], "context": state["context"], "branch": "no_context"}

def show_memory_node(state: AgentState) -> AgentState:
    history = state["messages"]
    summary = "\n".join(f"{msg.type.capitalize()}: {msg.content}" for msg in history)
    return {"messages": state["messages"] + [AIMessage(content=summary)], "context": None, "branch": "memory"}

# 6. Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("retrieve_context", retrieve_context)
graph.add_node("answer", answer_node)
graph.add_node("no_context", fallback_node)
graph.add_node("memory", show_memory_node)

graph.set_entry_point("retrieve_context")
graph.add_conditional_edges("retrieve_context", router)
graph.add_edge("answer", END)
graph.add_edge("no_context", END)
graph.add_edge("memory", END)

app = graph.compile()

# 7. In-memory chat history per session
chat_histories = {}
def get_chat_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

# 8. CLI loop
def run_graph_chat():
    print("🤖 LangGraph PDF Agent with Memory & Branching")
    print("Type 'exit' to quit, 'memory' to see what I remember.\n")

    session_id = "user-1"
    history = get_chat_history(session_id)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        history.add_user_message(user_input)

        result = app.invoke({"messages": history.messages})

        new_msg = result["messages"][-1]
        print("Bot:", new_msg.content)

        history.add_ai_message(new_msg.content)

if __name__ == "__main__":
    run_graph_chat()
