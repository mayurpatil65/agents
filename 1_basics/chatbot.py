from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig

# 1. Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

# 2. Initialize the LLM
llm = ChatOllama(model="llama3.2:1b")

# 3. Create the chain by combining the prompt and LLM
chain = prompt | llm

# 4. Implement the function to retrieve or create chat history
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

# Dictionary to store chat histories by session ID
chat_histories = {}

# 5. Wrap the chain with memory using RunnableWithMessageHistory
chat_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_key="input",  # Key in the input dict for user messages
    history_key="chat_history"  # Key in the input dict for message history
)

# 6. Function to run the chat loop
def run_chat():
    session_id = "user-1"
    print(" 🤖 Local ChatBot with Memory (LangChain v0.3 + Ollama). Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Prepare input for the chain
        input_data = {
            "input": user_input,
            "chat_history": get_session_history(session_id).messages
        }

        # Invoke the chain with the input data and session configuration
        response = chat_chain.invoke(
            input_data,
            config=RunnableConfig(configurable={"session_id": session_id})
        )
        print("Bot:", response.content)

        # Update the chat history with the latest user and AI messages
        chat_histories[session_id].add_user_message(user_input)
        chat_histories[session_id].add_ai_message(response)

if __name__ == "__main__":
    run_chat()

