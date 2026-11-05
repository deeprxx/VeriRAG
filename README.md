# 🧠 DeepRAG AI Workspace (VeriRAG)

**Next-Gen Intelligent Document Workspace powered by Neural RAG & Auto-Adaptive Gemini Engine.**

🌐 **Live Website:** https://deep-verirag-ai.streamlit.app

---

## ✨ Overview

**DeepRAG AI Workspace** is an enterprise-grade Retrieval-Augmented Generation (RAG) application engineered to transform how students, researchers, and professionals interact with complex PDF documents and textbooks. By combining local semantic vector indexing with Google's latest Gemini models, DeepRAG eliminates hallucinations and delivers 100% source-backed answers with verifiable citations.

---

## 🚀 Key Features

* **📂 Multi-Document Ingestion:** High-speed text extraction across multiple PDF files simultaneously using `pypdf`.
* **🧩 Context-Aware Semantic Chunking:** Configurable chunk size and overlap parameters via `RecursiveCharacterTextSplitter` so zero context is lost.
* **🧬 Local Neural Embeddings:** Converts document chunks into 384-dimensional dense vectors using HuggingFace's `all-MiniLM-L6-v2` model.
* **⚡ Instant FAISS Vector Search:** Millisecond similarity search across indexed document chunks.
* **🤖 Auto-Adaptive Gemini Engine:** Dynamically detects and routes queries to the latest active Google Gemini model (`gemini-3.0-flash`) for fast, accurate synthesis.
* **📚 Verifiable Source Citations:** Displays exact text passages retrieved from your PDFs for every generated answer.
* **🎨 Ultra-Modern Glassmorphism UI:** Custom dark neon theme with letter-by-letter title animations and real-time indexing metrics.

---

## 🛠️ Technology Stack

* **Frontend & UI:** Streamlit, Custom CSS3 (Glassmorphism & Keyframe Animations)
* **RAG Framework:** LangChain (`langchain`, `langchain-community`, `langchain-text-splitters`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **Embeddings:** HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
* **LLM Engine:** Google Gemini API (`gemini-3.0-flash`)
* **PDF Processing:** `pypdf`

---

## 💻 Run Locally

* **Clone the repository:** `git clone https://github.com/deeprxx/VeriRAG.git`
* **Open project folder:** `cd VeriRAG`
* **Activate virtual environment:** `source venv/bin/activate`
* **Install dependencies:** `pip install -r requirements.txt`
* **Run the app:** `streamlit run app.py`

---

## 👨‍💻 About the Creator

**Deep Patel**  
*Founder, Lead AI Architect & Creator of DeepRAG AI*

> *"My mission is simple: turn complex artificial intelligence into a fast, reliable, and visually breathtaking experience."* — **Deep Patel**
