# 📄 Smart Document Q&A Assistant
### Project 1 of 10 | RAG · LangChain · FAISS · HuggingFace · Streamlit

> Upload any PDF or text document and ask questions in natural language. Powered by Retrieval-Augmented Generation (RAG) using free open-source models.

---

## What It Does
- Upload a PDF or TXT document
- Document is split into overlapping chunks and embedded using `all-MiniLM-L6-v2`
- Chunks are stored in a FAISS vector index
- Your question retrieves the most relevant chunks
- A free LLM (flan-t5-large) reads those chunks and generates an answer
- Sources are shown so you can verify the answer

## Architecture
```
PDF/TXT → Chunker → Embedder (MiniLM) → FAISS Index
                                              ↓
Question → Embedder → Similarity Search → Top-K Chunks
                                              ↓
                                         LLM (flan-t5) → Answer + Sources
```

## Tech Stack
| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (FREE) |
| Vector Store | FAISS (local, no server needed) |
| LLM | google/flan-t5-large (FREE) or GPT-3.5 |
| UI | Streamlit |
| PDF parsing | PyPDF2 |

## Setup

### Local Machine
```bash
git clone https://github.com/inboxofpiyush01/doc-qa-assistant
cd doc-qa-assistant
pip install -r requirements.txt
streamlit run app.py
```
### Optional: Use OpenAI instead of free model
Create a `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
```
The app auto-detects the key and switches to GPT-3.5.

## Project Structure
```
p1_doc_qa/
├── app.py              # Streamlit UI
├── app_chat_history.py # Streamlit UI with chat history
├── requirements.txt
└── README.md
```

## Key Concepts Demonstrated
- **RAG (Retrieval-Augmented Generation)** — combining retrieval with generation
- **Vector embeddings** — semantic search beyond keyword matching
- **FAISS** — efficient similarity search at scale
- **LangChain** — chaining LLM components
- **Prompt engineering** — structured prompts for accurate answers

## Results
- Accurately answers questions from uploaded documents
- Shows source chunks so answers are verifiable
- Works fully offline with free models
- Supports chat history across multiple questions

---
*Built by Piyush Sharma
