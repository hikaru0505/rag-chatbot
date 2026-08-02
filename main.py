import os
import io
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

pronouns = {"he", "she", "it", "they", "his", "her", "their", "them"}

DEFAULTS = {
    "documents": [],        # [{file_name, text, pages, size_kb}]
    "chunks": [],            # [{content, source, page}]
    "vectorizer": None,       # TF-IDF vectorizer
    "chunk_vectors": None,     # TF-IDF matrix
    "embed_model": None,        # SentenceTransformer, loaded lazily
    "chunk_embeddings": None,    # FAISS index
    "chat_history": [],           # ["User: ...", "Assistant: ..."]
    "messages": [],                 # [{role, content, retrieved}]
    "last_retrieved": [],            # chunks retrieved for the most recent answer
    "retrieval_mode": "FAISS (embeddings)",
    "chunk_size": 500,
    "overlap": 100,
    "top_k": 3,
}
for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


@st.cache_resource(show_spinner=False)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Document loading & chunking (tracks page numbers for PDFs)
# ---------------------------------------------------------------------------
def process_uploaded_files(uploaded_files, chunk_size, overlap):
    documents = []
    all_chunks = []

    for uploaded_file in uploaded_files:
        name = uploaded_file.name
        raw_bytes = uploaded_file.read()
        size_kb = round(len(raw_bytes) / 1024, 1)
        pages_count = None

        if name.endswith(".txt"):
            text = raw_bytes.decode("utf-8", errors="ignore")
            all_chunks.extend(chunk_text(text, name, chunk_size, overlap, page=None))

        elif name.endswith(".pdf"):
            reader = PdfReader(io.BytesIO(raw_bytes))
            pages_count = len(reader.pages)
            text = ""
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                text += page_text
                all_chunks.extend(chunk_text(page_text, name, chunk_size, overlap, page=page_num))
        else:
            continue

        if text.strip():
            documents.append({
                "file_name": name,
                "text": text,
                "pages": pages_count,
                "size_kb": size_kb,
            })

    st.session_state.documents = documents
    st.session_state.chunks = all_chunks

    # Build both indexes so users can switch retrieval mode without reprocessing
    st.session_state.vectorizer, st.session_state.chunk_vectors = build_tfidf_index(all_chunks)
    st.session_state.embed_model, st.session_state.chunk_embeddings = build_faiss_index(all_chunks)

    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.last_retrieved = []


def chunk_text(text, source, chunk_size, overlap, page):
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(text), step):
        piece = text[i:i + chunk_size]
        if piece.strip():
            chunks.append({"content": piece, "source": source, "page": page})
    return chunks


# ---------------------------------------------------------------------------
# Retrieval indexes
# ---------------------------------------------------------------------------
def build_tfidf_index(chunks):
    if not chunks:
        return None, None
    texts = [c["content"] for c in chunks]
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def build_faiss_index(chunks):
    if not chunks:
        return None, None
    import faiss
    model = get_embed_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return model, index


def retrieve_chunks(query, top_k=3):
    chunks = st.session_state.chunks
    if not chunks:
        return []

    mode = st.session_state.retrieval_mode
    scored = []

    if mode == "TF-IDF":
        vectorizer = st.session_state.vectorizer
        matrix = st.session_state.chunk_vectors
        if vectorizer is None:
            return []
        query_vec = vectorizer.transform([query])
        sims = cosine_similarity(query_vec, matrix).flatten()
        ranked = sims.argsort()[::-1]
        threshold = 0.03
        for idx in ranked:
            score = float(sims[idx])
            if score < threshold:
                break
            scored.append((score, chunks[idx]))

    else:  # FAISS (embeddings)
        model, index = st.session_state.embed_model, st.session_state.chunk_embeddings
        if index is None:
            return []
        query_vec = model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        k = min(top_k * 4, len(chunks))  # over-fetch, then dedupe per source below
        sims, idxs = index.search(query_vec, k)
        threshold = 0.12
        for score, idx in zip(sims[0], idxs[0]):
            if idx == -1 or float(score) < threshold:
                continue
            scored.append((float(score), chunks[idx]))

    # Keep the single best chunk per source document, ranked by score
    best_per_doc = {}
    for score, chunk in scored:
        key = (chunk["source"], chunk.get("page"))
        if key not in best_per_doc or score > best_per_doc[key][0]:
            best_per_doc[key] = (score, chunk)

    selected = sorted(best_per_doc.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return [{"content": c["content"], "source": c["source"], "page": c.get("page"), "score": s} for s, c in selected]


# ---------------------------------------------------------------------------
# Agent logic (calculator vs document search) — unchanged from the CLI version
# ---------------------------------------------------------------------------
def decide_action(query):
    calculation_keywords = ["calculate", "find", "solve", "compute", "+", "-", "*", "/"]
    query_lower = query.lower()
    for keyword in calculation_keywords:
        if keyword in query_lower:
            return "calculation"
    return "document_search"


def calculate(query):
    """NOTE: eval() is unsafe for untrusted/public input. Kept to match the
    original CLI version; flag this as a known tradeoff if asked about it."""
    expression = query.lower().replace("calculate", "").strip()
    try:
        result = eval(expression)
        return f"Calculation result: {result}"
    except Exception:
        return "Sorry, I couldn't evaluate that expression."


def contains_pronoun(query):
    words = query.lower().split()
    return any(word in pronouns for word in words)


def build_search_query(query):
    chat_history = st.session_state.chat_history
    if chat_history and contains_pronoun(query):
        last_answer = chat_history[-1].replace("Assistant:", "", 1)
        first_sentence = last_answer.split(".")[0].strip()
        return first_sentence + " " + query
    return query


def answer_question(query, client, top_k):
    chat_history = st.session_state.chat_history
    search_query = build_search_query(query)
    relevant_chunks = retrieve_chunks(search_query, top_k=top_k)
    st.session_state.last_retrieved = relevant_chunks

    if not relevant_chunks:
        answer = "I couldn't find anything relevant to that in the documents."
        chat_history.append(f"User: {query}")
        chat_history.append(f"Assistant: {answer}")
        return answer, "N/A"

    def label(c):
        return f"{c['source']}" + (f" (p.{c['page']})" if c.get("page") else "")

    context = "\n\n".join(f"[Source: {label(c)}]\n{c['content']}" for c in relevant_chunks)
    history_text = "\n".join(chat_history[-6:])

    prompt = f"""You are a helpful assistant answering questions using only the provided context.

Chat History:
{history_text}

Context from documents:
{context}

Question: {query}

Answer using only the context above. If the answer isn't in the context, say so clearly.
Respond in EXACTLY this format:
ANSWER: <your answer>
SOURCE: <the source file name(s) and page(s) used>
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = completion.choices[0].message.content.strip()
    answer, source = raw, "Unknown"
    for line in raw.splitlines():
        if line.startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()
        elif line.startswith("SOURCE:"):
            source = line[len("SOURCE:"):].strip()

    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {answer}")
    return answer, source


def conversation_to_markdown():
    lines = ["# RAG Chatbot Conversation\n"]
    for msg in st.session_state.messages:
        role = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role}: {msg['content']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("📄 RAG Chatbot")
st.caption("Retrieval-Augmented Generation over your own documents — FAISS or TF-IDF retrieval, Groq Llama 3.1 generation.")

with st.sidebar:
    st.header("📁 Documents")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Add it to your .env file.")

    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        type=["txt", "pdf"],
        accept_multiple_files=True,
    )

    with st.expander("⚙️ Retrieval settings", expanded=False):
        st.session_state.retrieval_mode = st.radio(
            "Retrieval method",
            ["FAISS (embeddings)", "TF-IDF"],
            index=0 if st.session_state.retrieval_mode == "FAISS (embeddings)" else 1,
            help="FAISS uses semantic embeddings (understands meaning, not just keyword overlap). "
                 "TF-IDF is pure keyword/cosine similarity — simpler and fully explainable.",
        )
        st.session_state.chunk_size = st.slider("Chunk size (characters)", 200, 1500, st.session_state.chunk_size, step=50)
        st.session_state.overlap = st.slider("Chunk overlap (characters)", 0, 500, st.session_state.overlap, step=25)
        st.session_state.top_k = st.slider("Top-K chunks retrieved", 1, 8, st.session_state.top_k)

    if st.button("🔄 Process documents", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Reading and indexing documents..."):
            process_uploaded_files(uploaded_files, st.session_state.chunk_size, st.session_state.overlap)
        st.success(f"Indexed {len(st.session_state.documents)} document(s) into {len(st.session_state.chunks)} chunks.")

    if st.session_state.documents:
        st.divider()
        st.subheader("Loaded files")
        for doc in st.session_state.documents:
            meta = f"{doc['size_kb']} KB"
            if doc["pages"]:
                meta += f" · {doc['pages']} pages"
            st.markdown(f"**{doc['file_name']}**  \n<small>{meta}</small>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear chat", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.messages = []
                st.session_state.last_retrieved = []
                st.rerun()
        with col2:
            if st.button("❌ Clear all", use_container_width=True):
                for key, default in DEFAULTS.items():
                    st.session_state[key] = default
                st.rerun()

        if st.session_state.messages:
            st.download_button(
                "⬇️ Download conversation (.md)",
                data=conversation_to_markdown(),
                file_name="conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )

    st.divider()
    st.caption("Tip: try a math question like 'calculate 12 * 8' to see the calculator tool trigger instead of document search.")

tab_chat, tab_context, tab_pipeline = st.tabs(["💬 Chat", "🔍 Retrieved Context", "🧭 Pipeline"])

with tab_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# IMPORTANT: st.chat_input must be called at the TOP LEVEL of the page (not
# inside st.tabs/st.columns/st.expander) or Streamlit won't anchor it to the
# bottom of the screen — it'll render inline wherever the tab happens to sit.
query = st.chat_input("Ask a question about your documents...")

if query:
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found. Add it to your .env file before chatting.")
    elif not st.session_state.chunks:
        st.warning("Upload and process at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        action = decide_action(query)
        if action == "calculation":
            result = calculate(query)
            st.session_state.messages.append({"role": "assistant", "content": result})
        else:
            client = Groq(api_key=GROQ_API_KEY)
            with st.spinner("Thinking..."):
                answer, source = answer_question(query, client, st.session_state.top_k)
            reply = f"{answer}\n\n*Source: {source}*"
            st.session_state.messages.append({"role": "assistant", "content": reply})

        # Rerun so the Chat tab (rendered above) and Retrieved Context tab
        # both reflect the new message / retrieval immediately.
        st.rerun()

with tab_context:
    st.subheader("Chunks retrieved for the most recent question")
    if not st.session_state.last_retrieved:
        st.info("Ask a document question in the Chat tab to see retrieved chunks and their relevance scores here.")
        st.caption(
            "If you asked a question and still see nothing here, the similarity score for every "
            "chunk fell below the threshold — try rephrasing the question, or switch retrieval mode "
            "in the sidebar (FAISS handles paraphrased/casual wording better than TF-IDF)."
        )
    else:
        for i, c in enumerate(st.session_state.last_retrieved, start=1):
            page_label = f" — page {c['page']}" if c.get("page") else ""
            with st.expander(f"#{i}  {c['source']}{page_label}   ·   score {c['score']:.3f}"):
                st.write(c["content"])

with tab_pipeline:
    st.subheader("How a question is answered")
    st.markdown(
        """
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
      Chat history (enables pronoun-aware follow-ups, e.g. "what does *he* do?")
```
        """
    )
    st.caption(
        "FAISS mode embeds text with `all-MiniLM-L6-v2` (sentence-transformers) and does "
        "cosine similarity search over dense vectors — captures meaning, not just keyword overlap. "
        "TF-IDF mode is pure keyword/cosine similarity — simpler and fully explainable, no black-box embeddings."
    )