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
import re

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
        safe_expr = re.sub(r"[^0-9\+\-\*/%\.\(\) ]", "", input)
        result = eval(safe_expr, {"__builtins__": None}, {})
        return f"Answer: {result}"
    except Exception as e:
        return f"Calculator Error: {str(e)}"

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No relevant search results found."
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

# ---------- Prompts ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

prompt_with_context = ChatPromptTemplate.from_messages([
    ("system", "Use the context to answer the question.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# ---------- State ----------
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: Optional[str]
    tool_used: Optional[str]
    branch: Literal["llm", "rag", "tool", "fallback"]

# ---------- Agent Node (Tool → RAG → LLM Priority) ----------
def agent_node(state: AgentState) -> AgentState:
    question = state["messages"][-1].content.lower()
    new_messages = state["messages"][:]
    tool_used = None
    branch = "llm"

    # Step 1: Tool routing
    if any(op in question for op in ["+", "-", "*", "/", "%", "sqrt"]):
        content = calculator_tool(question)
        tool_used = "calculator"
        branch = "tool"

    elif any(kw in question for kw in ["search", "who is", "current president", "what is", "define"]):
        content = duckduckgo_search(question)
        tool_used = "duckduckgo"
        branch = "tool"

    else:
        # Step 2: Try RAG
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs[:3]) if docs else ""
        if context.strip():
            answer = (prompt_with_context | llm).invoke({"question": question, "context": context})
            content = answer.content
            branch = "rag_llm"
        else:
            # Step 3: LLM fallback
            answer = (prompt | llm).invoke({"question": question})
            content = answer.content
            branch = "llm_rag"

    new_messages.append(AIMessage(content=content))
    return {
        "messages": new_messages,
        "context": None,
        "tool_used": tool_used,
        "branch": branch
    }

# ---------- Graph Wiring ----------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)

graph.set_entry_point("agent")
graph.add_edge("agent", END)

app = graph.compile()

# ---------- CLI ----------
chat_histories = {}
def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def run_chat():
    print("\n🤖 LangGraph Agent (LLM → RAG → Tool cascade)")
    print("Type 'exit' to quit.\n")
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
