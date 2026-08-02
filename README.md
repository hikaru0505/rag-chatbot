# 📄 RAG Chatbot (FAISS / TF-IDF + Groq Llama 3.1)

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, **FAISS**, **TF-IDF**, and **Groq Llama 3.1**. Upload your own **PDF** or **TXT** documents and ask natural language questions with page-cited answers.

The application supports **switchable retrieval methods**, transparent retrieval visualization, configurable retrieval settings, document metadata, conversation export, and a built-in calculator tool.

---

# 📸 Screenshots

## Chat Interface

![Chat Interface](screenshots/01-chat-interface.png)

---

## Retrieved Context

![Retrieved Context](screenshots/02-retrieved-context.png)

---

## RAG Pipeline

![Pipeline](screenshots/03-rag-pipeline.png)

---

# ✨ Features

- 🔍 **Dual Retrieval Methods**
  - FAISS + all-MiniLM-L6-v2 embeddings (Semantic Search)
  - TF-IDF + Cosine Similarity (Keyword Search)

- 📄 Upload one or multiple PDF/TXT documents

- 📑 Automatic PDF page-number citations

- 🎯 Transparent retrieval showing
  - Retrieved chunks
  - Source document
  - Page number
  - Similarity score

- ⚙️ Adjustable retrieval settings
  - Retrieval method
  - Chunk size
  - Chunk overlap
  - Top-K chunks

- 💬 Pronoun-aware follow-up conversations using chat history

- 📥 Download conversation as Markdown

- 🧮 Built-in calculator tool for arithmetic queries

- 📂 Document metadata
  - File size
  - Number of pages

- 🧭 Interactive RAG Pipeline visualization

---

# 🛠 Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- FAISS
- Sentence Transformers
- scikit-learn
- PyPDF
- python-dotenv

---

# 📂 Project Structure

```text
rag-chatbot/
│
├── main.py
├── requirements.txt
├── .env.example
├── README.md
│
└── screenshots/
    ├── 01-chat-interface.png
    ├── 02-retrieved-context.png
    └── 03-rag-pipeline.png
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/hikaru0505/rag-chatbot.git
cd rag-chatbot
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GROQ_API_KEY=your_api_key_here
```

Run the application

```bash
streamlit run main.py
```

---

# 🚀 Usage

1. Upload one or more PDF/TXT documents.
2. Click **Process Documents**.
3. Choose **FAISS** or **TF-IDF** retrieval.
4. Configure chunk size, overlap, and Top-K if desired.
5. Ask questions about your documents.
6. View retrieved chunks in the **Retrieved Context** tab.
7. Explore the retrieval workflow in the **Pipeline** tab.
8. Download conversations as Markdown if needed.

---

# 🧠 Architecture

```text
USER Question
      │
AGENT DECISION (Calculator vs Document Search)
      │
      ├──────────── Calculator
      │               │
      │             eval()
      │               │
      │            Calculation Result
      │
      └──────────── Document Search
                      │
                      ▼
      Retrieval (FAISS Embeddings / TF-IDF)
                      │
                      ▼
      Top-K Relevant Chunks
      (Source + Page Number + Score)
                      │
                      ▼
      Groq Llama 3.1
                      │
                      ▼
      Answer + Source Citation
                      │
                      ▼
      Chat History
```

---

# 💡 Design Decisions

### Why FAISS and TF-IDF?

The application allows switching between semantic and keyword-based retrieval.

**FAISS**

- Semantic understanding
- Finds conceptually similar text
- Better for natural language questions

**TF-IDF**

- Pure keyword matching
- Fully explainable
- Lightweight
- No embedding model required

This comparison demonstrates understanding of different retrieval techniques instead of relying on a single approach.

---

### Transparent Retrieval

Unlike many RAG demos, this application exposes:

- Retrieved chunks
- Source document
- Page number
- Similarity score

making the retrieval process easy to inspect and debug.

---

### Page-Level Citations

PDFs are processed page-by-page before chunking, allowing answers to reference the exact page instead of only the document name.

---

### Session-Based Storage

Documents, indexes, and conversations are stored in `st.session_state`, keeping the application lightweight and stateless.

---

### Calculator Routing

Simple arithmetic queries bypass document retrieval and are routed directly to a calculator, demonstrating basic tool routing within the application.

---

# 🚀 Future Improvements

- Hybrid Retrieval (BM25 + FAISS)
- OCR support for scanned PDFs
- Additional LLM providers
- Chat history persistence
- Authentication and user accounts
- Docker deployment
- Cloud storage support

---

# 📄 License

This project is licensed under the MIT License.