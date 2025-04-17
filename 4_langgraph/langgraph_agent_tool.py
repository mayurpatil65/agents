import os
from typing import List, Literal, Optional, TypedDict, Union

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory
from duckduckgo_search import DDGS

# ---------- Setup LLM, Vectorstore & Retriever ----------
llm = ChatOllama(model="llama3.2:1b")
embeddings = OllamaEmbeddings(model="llama3.2:1b")

def load_docs():
    docs = []
    for file in os.listdir("pdfs"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join("pdfs", file))
            docs.extend(loader.load())
    return docs

docs = load_docs()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]

vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever()

# ---------- Tools ----------
def calculator_tool(input: str) -> str:
    try:
        return f"Answer: {eval(input)}"
    except Exception as e:
        return f"Calculator Error: {str(e)}"

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No relevant search results found."
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

# ---------- Prompt ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question using the context below.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# ---------- State Definition ----------
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: Optional[str]
    tool_used: Optional[str]
    branch: Literal["answer", "tool", "no_context", "memory", "rag"]

# ---------- Router Function ----------
def router(state: AgentState) -> str:
    user_input = state["messages"][-1].content.lower()
    if user_input.strip() == "memory":
        return "memory"
    elif any(op in user_input for op in ["+", "-", "*", "/", "%", "sqrt"]):
        return "tool"
    elif any(kw in user_input for kw in ["search", "who is", "president", "langchain", "what is"]):
        return "tool"
    return "rag"

# ---------- Nodes ----------
def retrieve_context(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs[:3]) if docs else "[NO_CONTEXT]"
    branch = "no_context" if context == "[NO_CONTEXT]" else "answer"
    return {**state, "context": context, "branch": branch}

def answer_node(state: AgentState) -> AgentState:
    question = state["messages"][-1].content
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": state["context"]})
    content = response.content if hasattr(response, "content") else str(response)
    return {**state, "messages": state["messages"] + [AIMessage(content=content)]}

def fallback_node(state: AgentState) -> AgentState:
    msg = "🤷 Sorry, I couldn't find anything relevant in the documents."
    return {**state, "messages": state["messages"] + [AIMessage(content=msg)]}

def memory_node(state: AgentState) -> AgentState:
    history = "\n".join(f"{m.type.capitalize()}: {m.content}" for m in state["messages"])
    return {**state, "messages": state["messages"] + [AIMessage(content=history)]}

def tool_node(state: AgentState) -> AgentState:
    question = state["messages"][-1].content.lower()
    if any(op in question for op in ["+", "-", "*", "/", "%", "sqrt"]):
        result = calculator_tool(question)
        return {**state, "messages": state["messages"] + [AIMessage(content=result)], "tool_used": "calculator"}
    else:
        result = duckduckgo_search(question)
        return {**state, "messages": state["messages"] + [AIMessage(content=result)], "tool_used": "duckduckgo"}

# ---------- Build LangGraph ----------
graph = StateGraph(AgentState)
graph.add_node("entry", lambda state: state)
graph.add_node("rag", retrieve_context)
graph.add_node("answer", answer_node)
graph.add_node("no_context", fallback_node)
graph.add_node("memory", memory_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("entry")
graph.add_conditional_edges("entry", router)
graph.add_conditional_edges("rag", lambda state: state["branch"])
graph.add_edge("answer", END)
graph.add_edge("no_context", END)
graph.add_edge("memory", END)
graph.add_edge("tool", END)

app = graph.compile()

# ---------- CLI ----------
chat_histories = {}
def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def run_chat():
    print("🤖 LangGraph Agent (RAG + Memory + Tools)\nType 'exit' to quit.\n")
    session_id = "user-1"
    history = get_history(session_id)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        history.add_user_message(user_input)

        result = app.invoke({"messages": history.messages})
        reply = result["messages"][-1].content
        print("Bot:", reply)

        print(f"🧭 Used branch: {result.get('branch')}", end="")
        if result.get("tool_used"):
            print(f" | 🛠️ Tool: {result['tool_used']}")
        else:
            print("")

        history.add_ai_message(reply)

if __name__ == "__main__":
    run_chat()