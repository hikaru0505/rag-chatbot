# RAG Chatbot (FAISS / TF-IDF + Groq Llama 3.1)

A Retrieval-Augmented Generation chatbot with a Streamlit web UI. Upload your
own `.txt` / `.pdf` documents in-browser and ask questions about them. Supports
two switchable retrieval methods (semantic FAISS embeddings or classic TF-IDF),
shows exactly which chunks were retrieved and why, cites PDF page numbers, and
includes a basic calculator tool plus pronoun-aware follow-up questions
(e.g. "what does *he* do?").

## Demo

![RAG Chatbot Demo](demo.gif)
<!-- Replace with an actual screenshot or GIF of the app before pushing -->

## Features

- **Selectable retrieval:** FAISS + `all-MiniLM-L6-v2` sentence embeddings (semantic search) or TF-IDF cosine similarity (keyword search, fully explainable, no black-box embeddings)
- **Transparent retrieval:** a "Retrieved Context" tab shows the exact chunks used to answer the last question, with relevance scores
- **PDF page citations:** answers cite the source file *and* page number, not just the filename
- **Document metadata:** file size and page count shown for each uploaded document
- **Retrieval settings panel:** adjust chunk size, overlap, and top-K live from the sidebar
- **Conversation export:** download the chat as a Markdown file
- **Calculator tool:** simple arithmetic queries are routed to a calculator instead of retrieval
- **Pipeline tab:** a diagram of the full RAG flow, for interview / demo purposes

## Folder structure

```
rag-chatbot/
├── main.py             # Streamlit app (current version — FAISS/TF-IDF, citations, etc.)
├── requirements.txt
├── .env                (you create this, not committed to git)
├── .env.example
└── README.md
```

## Setup

1. Clone the repo and enter the project folder:
   ```bash
   git clone https://github.com/<your-username>/rag-chatbot.git
   cd rag-chatbot
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # on Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Get a free Groq API key from https://console.groq.com/keys

5. Create a `.env` file (copy `.env.example`) and paste your key in:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

## Running it

```bash
streamlit run main.py
```

This opens a browser tab at `http://localhost:8501`. From there:

1. Upload one or more `.txt` / `.pdf` files in the sidebar.
2. (Optional) Open **Retrieval settings** to pick FAISS vs. TF-IDF, or adjust chunk size / overlap / top-K.
3. Click **Process documents** to index them.
4. Ask questions in the **Chat** tab.
5. Check the **Retrieved Context** tab to see exactly which chunks (and scores) were used.
6. Try a math question like "calculate 12 * 8" to see the calculator tool trigger instead of document search.

> Note: the first FAISS query after startup will take a few seconds while the
> `all-MiniLM-L6-v2` embedding model downloads and loads. Subsequent queries
> are fast, and the model is cached for the rest of the session.

## Architecture

```
USER Question
      │
AGENT DECISION (calculator vs. document search)
      │
      ├── Calculator ──► eval() ──► Result
      │
      └── Document search
              │
              ▼
      Retrieval  (FAISS embeddings  OR  TF-IDF cosine similarity)
              │
              ▼
      Top-K relevant chunks (with source file + page number)
              │
              ▼
      Groq Llama 3.1 (llama-3.1-8b-instant)
              │
              ▼
           Answer + Source citation
              │
              ▼
      Chat history (enables pronoun-aware follow-ups)
```

## Design notes / things to mention in an interview

- **FAISS vs. TF-IDF:** FAISS embeddings capture semantic meaning (e.g. "car" and "automobile" are close in vector space); TF-IDF is pure keyword overlap but fully transparent and has no external model dependency. Making retrieval switchable lets you demonstrate you understand the tradeoff rather than just picking one.
- **`eval()` in `calculate()`** is fine for a personal/local tool but unsafe for untrusted/public input. A production version would use a restricted math parser (e.g. Python's `ast` module with a whitelisted operator set) instead.
- **Per-source deduplication:** retrieval keeps only the single best-scoring chunk per (source, page) pair before ranking, so one long document can't crowd out every other source in the answer.
- **Page-level citation for PDFs** is done by chunking per-page rather than concatenating the whole PDF into one blob first — this is what makes accurate page citations possible.
- **Session-scoped state:** documents, indexes, and chat history all live in `st.session_state`, so nothing persists to disk — re-running the app starts clean, matching how a stateless demo/portfolio app should behave.