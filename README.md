# Agents — Multi-Modal, Tool-Using LangGraph Agent (LLM + RAG + Tools + Memory)

Agents is a fully local, multi-modal agent framework powered by LangChain v0.3, LangGraph, and Ollama (LLM: `llama3.2:1b`). 

It supports:

- **Document Q&A (RAG)** over PDFs using `Chroma` and `PyMuPDF`
- **Tool use**, including:
   - DuckDuckGo search for real-time info
   - Calculator via `SymPy`
- **Long-term memory** via LangGraph and persistent Chroma DB
- **Dynamic branching logic** across LLM, RAG, and tools
- `filename:` prefix-based PDF filtering
- CLI interface for interactive sessions with chat history

> Victor Agent is a fully local POC developed for research and internal use, with support for offline chat, private docs, and modular tool integration.

---
 
## Key Features

### Retrieval-Augmented Generation (RAG)
- Parses and chunks PDFs using `PyMuPDFLoader`
- Stores vectors in `Chroma` with `OllamaEmbeddings`
- Filters context using custom `filename:` syntax (e.g., `filename:doc.pdf What is this about?`)

### Tools
- **Math Calculator**: Secure expression parser using `sympy`
- **DuckDuckGo Search**: Real-time web queries with fallback to local RAG

### Memory + Chat History
- Multi-session chat support using `ChatMessageHistory`
- Long-term memory management via `LangGraph` state
- Full message replay from past sessions

### Branching Agent Logic
Custom decision node classifies user input into:
- RAG → if sufficient PDF context is found
- Tool use → if calculator expression or web keywords detected
- Web search fallback → for OOD queries
- LLM fallback → if no relevant context or tools apply

---

## Running the Agent

From the `5_langgraph_persistent_memory/` folder:

```bash
python main.py

---





## Dependencies 

LangChain v0.3+

LangGraph

Ollama + llama3.2:1b

PyMuPDF

Chroma

SymPy

duckduckgo-search

You also need to install and run Ollama locally and pull the model:

  - ollama pull llama3.2:1b

Example Interactions


You: filename:roadmap.pdf What are the key takeaways?
Bot: [RAG] Based on 'roadmap.pdf'... [summarized content]

You: 12 * (7 + 3)
Bot: [Tool] Answer: 120.000000000000

You: Search Chinya Ravishankar
Bot: [Web Search] - He is...


---

          ┌──────────────┐
          │   User CLI   │
          └─────┬────────┘
                │
         ┌──────▼───────┐
         │ Classify Node│◄────────┐
         └──────┬───────┘         │
   ┌────────────▼────────────┐    │
   │    Retrieval Node (RAG) │    │
   └────────────┬────────────┘    │
   ┌────────────▼────────────┐    │
   │  Tool Node (Calc/Search)│    │
   └────────────┬────────────┘    │
   ┌────────────▼────────────┐    │
   │     LLM Fallback Node   │    │
   └────────────┬────────────┘    │
                ▼                 │
          ┌──────────────┐        │
          │ Response Node│────────┘
          └──────────────┘


---


**Final Notes:**
- Fully local and offline-capable

- Persistent memory using Chroma

- Modular: Easily extend with new tools (e.g., Wolfram, SQL, APIs)

- Suitable for secure deployments (e.g., Internal applications)


