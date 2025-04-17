print("1")
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="llama3.2:1b")
print(llm.invoke("What is 2+2?"))

print("2")
from langchain_ollama import ChatOllama

# Initialize a local LLM
model = ChatOllama(model="llama3.2:1b")
response = model.invoke("Hello, LangChain!")
print(response.content)
print(model.invoke("Who built you").content)

print("3")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

# For chat models
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Answer the question briefly.\nQuestion: {question}\nAnswer:"
)
chain = prompt | model
output = chain.invoke({"question": "What is your name?"})
print(output)

parser = StrOutputParser()
chain = prompt | llm | parser
print(chain.invoke({"question": "Who built you?"}))

# Prompt template with LCEL
prompt = PromptTemplate.from_template("What is {a} + {b}?")
chat_chain = prompt | llm  # Functional composition: prompt → llm
output = chat_chain.invoke({"a": 2, "b": 3})
print(output)  # Direct output string from the model
