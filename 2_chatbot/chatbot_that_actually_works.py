from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.chat_message_histories import ChatMessageHistory

# 1. Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# 2. Local Ollama LLM
llm = ChatOllama(model="llama3.2:1b")

# 3. Core LCEL chain
chain = prompt | llm | StrOutputParser()

# 4. Per-session memory manager
chat_histories = {}

def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

# 5. Wrap chain with memory
chat_chain = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 6. CLI Chat loop
def run_chat():
    session_id = "local-user"
    print("🤖 Local ChatBot with Memory (LangChain v0.3 + Ollama)\nType 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        response = chat_chain.invoke(
            {"input": user_input},
            config=RunnableConfig(configurable={"session_id": session_id})
        )

        print("Bot:", response)

if __name__ == "__main__":
    run_chat()
