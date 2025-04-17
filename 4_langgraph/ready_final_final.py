import os
import re
import logging
from typing import List, Literal, Optional, TypedDict, Union

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.chat_message_histories import ChatMessageHistory
from duckduckgo_search import DDGS
from sympy import sympify
from sympy.core.sympify import SympifyError

# ------------------------- LLM Setup --------------------------------
llm = ChatOllama(model="llama3.2:1b", temperature=0.3)
embeddings = OllamaEmbeddings(model="llama3.2:1b")

# ------------------------- PDF Ingestion ----------------------------
def load_docs():
    docs = []
    for file in os.listdir("pdfs"):
        if file.endswith(".pdf"):
            path = os.path.join("pdfs", file)
            loader = PyMuPDFLoader(path)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = file.lower()
                doc.page_content = f"[{file}]\n" + doc.page_content
            docs.extend(loaded)
    return docs

chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(load_docs())
texts = [doc.page_content for doc in chunks]
metadatas = [doc.metadata for doc in chunks]
vectorstore = Chroma.from_texts(texts, embedding=embeddings, metadatas=metadatas)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ------------------------- Tools -------------------------------------
def calculator_tool(input: str) -> str:
    try:
        expr = sympify(input)
        return f"Answer: {expr.evalf()}"
    except (SympifyError, Exception) as e:
        return f"Calculator Error: {str(e)}"

def duckduckgo_search(query: str) -> str:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=3)
        if not results:
            return "No relevant search results found."
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)

# ------------------------- Prompts -----------------------------------
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use the following context to answer the user's question. "
     "If the answer cannot be found in the context, say you don’t know. "
     "Do not make up facts.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# ------------------------- Agent State -------------------------------
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    context: Optional[str]
    tool_used: Optional[str]
    branch: Literal["llm", "rag", "tool", "web_search", "fallback"]

# ------------------------- Nodes -------------------------------------
def classify_input(state: AgentState) -> AgentState:
    q = state["messages"][-1].content.lower().strip()

    if q.startswith("filename:"):
        parts = q.split(None, 1)
        if len(parts) < 2:
            return {**state, "branch": "llm"}
        filename = parts[0].replace("filename:", "").strip()
        question = parts[1]
        docs = retriever.vectorstore.similarity_search(
            question,
            k=5,
            filter={"source": {"$eq": filename.lower()}}
        )
        context = "\n\n".join(f"{doc.metadata.get('source', '')}:\n{doc.page_content}" for doc in docs) if docs else ""
        return {
            **state,
            "messages": state["messages"][:-1] + [HumanMessage(content=question)],
            "context": context,
            "branch": "rag" if context else "llm"
        }

    if any(op in q for op in ["+", "-", "*", "/", "%"]):
        return {**state, "branch": "tool"}

    if any(kw in q for kw in ["search", "lookup", "find on web"]):
        return {**state, "branch": "web_search"}

    docs = retriever.invoke(q)
    context = "\n\n".join(f"{doc.metadata.get('source', '')}:\n{doc.page_content}" for doc in docs) if docs else ""
    if context and len(set(re.findall(r"\w+", q)) & set(re.findall(r"\w+", context))) >= 2:
        return {**state, "context": context, "branch": "rag"}

    return {**state, "branch": "llm"}

def llm_node(state: AgentState) -> AgentState:
    return {**state, "branch": "llm"}

def rag_node(state: AgentState) -> AgentState:
    return {**state, "branch": "rag"}

def tool_node(state: AgentState) -> AgentState:
    q = state["messages"][-1].content
    result = calculator_tool(q)
    return {**state, "messages": state["messages"] + [AIMessage(content=result)], "tool_used": "calculator", "branch": "tool"}

def web_search_node(state: AgentState) -> AgentState:
    q = state["messages"][-1].content
    result = duckduckgo_search(q)
    return {**state, "messages": state["messages"] + [AIMessage(content=result)], "tool_used": "duckduckgo", "branch": "web_search"}

# ------------------------- LangGraph Flow -----------------------------
graph = StateGraph(AgentState)
graph.add_node("classify", classify_input)
graph.add_node("llm", llm_node)
graph.add_node("rag", rag_node)
graph.add_node("tool", tool_node)
graph.add_node("web_search", web_search_node)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", lambda state: state["branch"], {
    "llm": "llm",
    "rag": "rag",
    "tool": "tool",
    "web_search": "web_search"
})
graph.add_edge("llm", END)
graph.add_edge("rag", END)
graph.add_edge("tool", END)
graph.add_edge("web_search", END)

app = graph.compile()

# ------------------------- CLI Interface -----------------------------
chat_histories = {}
def get_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def stream_llm_response(prompt, inputs: dict):
    return (prompt | llm).stream(inputs)

def run_chat():
    print("\n🤖 Multi-Node LangGraph Agent (LLM → RAG → Tool + Verify)")
    print("Type 'exit' to quit.\n")
    session_id = "user-1"
    history = get_history(session_id)

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        history.add_user_message(user_input)
        print("Bot: ", end="", flush=True)

        state = {"messages": history.messages}

        final_state = None
        try:
            for step in app.stream(state):
                final_state = step
        except Exception as e:
            print(f"❌ Error during graph execution: {e}")
            continue

        if final_state is None:
            print("⚠️ No response from agent.")
            continue

        flattened = list(final_state.values())[0]
        branch = flattened.get("branch")
        messages = flattened.get("messages", [])

        if not branch:
            print("❌ Branch missing from final state!")
            print("🧾 Final state keys:", flattened.keys())
            continue

        state = flattened
        question = state["messages"][-1].content

        if branch in {"llm", "rag"}:
            context = state.get("context", "")
            prompt = rag_prompt if (branch == "rag" and context) else base_prompt
            inputs = {"question": question}
            if "context" in prompt.input_variables:
                inputs["context"] = context
            stream = stream_llm_response(prompt, inputs)
            response = ""
            for chunk in stream:
                print(chunk.content, end="", flush=True)
                response += chunk.content
            history.add_ai_message(response)

        elif branch in {"tool", "web_search"}:
            response = state["messages"][-1].content
            print(response, end="", flush=True)
            history.add_ai_message(response)

        print()


if __name__ == "__main__":
    run_chat()