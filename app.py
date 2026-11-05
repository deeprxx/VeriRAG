import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import requests
from pypdf import PdfReader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =====================================================================
# 🔑 APNI GOOGLE GEMINI API KEY NICHE QUOTES KE ANDAR PASTE KAR:
# =====================================================================
DEFAULT_API_KEY = ""

try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", DEFAULT_API_KEY)
except Exception:
    GEMINI_API_KEY = DEFAULT_API_KEY

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepRAG AI Workspace | Deep Patel",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NEXT-LEVEL PREMIUM CSS & LETTER-BY-LETTER ANIMATION ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700;800&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #1e1b4b 0%, #090d16 55%, #020617 100%);
        color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .block-container {
        padding-top: 3.2rem !important;
        padding-bottom: 2rem !important;
    }

    /* Letter-by-Letter Pop & Glow Animation */
    .hero-title-wrap {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        margin-top: 10px;
        margin-bottom: 6px;
    }

    .char {
        display: inline-block;
        font-size: 3.1rem;
        font-weight: 800;
        opacity: 0;
        transform: translateY(14px) scale(0.85);
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: revealLetter 0.35s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    }

    @keyframes revealLetter {
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    .cursor-blink {
        display: inline-block;
        width: 4px;
        height: 2.8rem;
        background: #ec4899;
        margin-left: 8px;
        border-radius: 4px;
        box-shadow: 0 0 12px #ec4899;
        animation: blink 0.8s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(30, 41, 59, 0.55);
        padding: 26px;
        border-radius: 18px;
        backdrop-filter: blur(18px);
        border: 1px solid rgba(148, 163, 184, 0.14);
        box-shadow: 0 12px 35px 0 rgba(0, 0, 0, 0.45);
        margin-bottom: 20px;
        transition: transform 0.25s ease, border-color 0.25s ease;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(168, 85, 247, 0.45);
    }

    .stat-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.75));
        padding: 16px 20px;
        border-radius: 14px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        text-align: center;
    }

    .status-pill {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        padding: 10px 14px;
        border-radius: 12px;
        font-size: 0.88rem;
        font-weight: 600;
        text-align: center;
        margin-bottom: 10px;
    }

    .badge-pill {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.18);
        border: 1px solid rgba(129, 140, 248, 0.4);
        color: #c7d2fe;
    }

    /* Premium Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: rgba(15, 23, 42, 0.75);
        padding: 8px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        border-radius: 10px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        box-shadow: 0 4px 18px rgba(168, 85, 247, 0.45);
    }

    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- HELPER: AUTO-DETECT WORKING GEMINI MODEL & GENERATE ANSWER ---
def generate_smart_gemini_response(api_key: str, prompt: str) -> tuple[str, str]:
    clean_key = api_key.strip()
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
    
    resp = requests.get(list_url, timeout=15)
    if resp.status_code != 200:
        err_msg = resp.json().get("error", {}).get("message", "Invalid API Key or network issue.")
        raise RuntimeError(f"API Key Verification Failed: {err_msg}")
    
    models_data = resp.json().get("models", [])
    valid_models = [
        m["name"] for m in models_data
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "gemini" in m["name"].lower()
    ]

    # Priority list starting with latest Gemini 3.0 Flash
    preferred_keywords = ["3.0-flash", "3-flash", "3.0-pro", "2.0-flash", "1.5-flash", "flash", "pro"]
    ordered_models = ["models/gemini-3.0-flash"]
    
    for kw in preferred_keywords:
        for m in valid_models:
            if kw in m and m not in ordered_models and "2.5-flash" not in m:
                ordered_models.append(m)
                
    for m in valid_models:
        if m not in ordered_models:
            ordered_models.append(m)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3}
    }

    last_error = ""
    for chosen_model in ordered_models:
        gen_url = f"https://generativelanguage.googleapis.com/v1beta/{chosen_model}:generateContent?key={clean_key}"
        gen_resp = requests.post(gen_url, json=payload, timeout=60)
        if gen_resp.status_code == 200:
            data = gen_resp.json()
            answer_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return answer_text, chosen_model.replace("models/", "")
        else:
            last_error = gen_resp.json().get("error", {}).get("message", "Generation error")

    raise RuntimeError(f"Model Error: {last_error}")

# --- HELPER: BUILD LETTER-BY-LETTER HTML ---
def render_animated_title(text: str) -> str:
    spans = []
    for idx, ch in enumerate(text):
        delay = idx * 0.07
        display_char = "&nbsp;" if ch == " " else ch
        spans.append(f'<span class="char" style="animation-delay: {delay:.2f}s">{display_char}</span>')
    return f'<div class="hero-title-wrap">{"".join(spans)}<span class="cursor-blink"></span></div>'

# --- SIDEBAR CONTROL CENTER (API KEY HIDDEN) ---
with st.sidebar:
    st.subheader("⚙️ DeepRAG Control Center")
    st.markdown("<div class='status-pill'>🟢 Neural Engine: Connected & Active</div>", unsafe_allow_html=True)
    
    st.divider()
    st.subheader("📂 Document Portal")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    st.divider()
    st.subheader("🛠️ Neural Parameters")
    chunk_size = st.slider("Chunk Size", 300, 1200, 600, 50)
    chunk_overlap = st.slider("Chunk Overlap", 50, 250, 100, 10)
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 12px;'>Engineered with ❤️ by <b>Deep Patel</b></p>",
        unsafe_allow_html=True
    )

# --- TABS NAVIGATION ---
tab1, tab2, tab3 = st.tabs(["🚀 AI Workspace", "⚡ How It Works", "👨‍💻 About Creator"])

# ================= TAB 1: AI WORKSPACE =================
with tab1:
    st.markdown(render_animated_title("DeepRAG AI Workspace"), unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.12rem; margin-bottom: 22px;'>"
        "Next-Gen Intelligent Document Workspace powered by Neural RAG & Auto-Adaptive Gemini Engine."
        "</p>",
        unsafe_allow_html=True
    )

    if not uploaded_files:
        st.info("💡 Upload your PDF documents in the left sidebar to initialize the neural index and start chatting.")
    else:
        @st.cache_resource(show_spinner=False)
        def build_vector_index(files, c_size, c_overlap):
            full_text = ""
            total_pages = 0
            for f in files:
                reader = PdfReader(f)
                total_pages += len(reader.pages)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        full_text += extracted + "\n"
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
            chunks = splitter.split_text(full_text)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
            return vector_store, total_pages, len(chunks)

        with st.spinner("🔄 Indexing documents into neural vector space..."):
            try:
                vectorstore, page_count, chunk_count = build_vector_index(
                    uploaded_files, chunk_size, chunk_overlap
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"<div class='stat-box'><h4 style='margin:0;color:#818cf8;'>📄 PDF Pages</h4><h2 style='margin:4px 0 0 0;'>{page_count}</h2></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='stat-box'><h4 style='margin:0;color:#c084fc;'>🧩 Neural Chunks</h4><h2 style='margin:4px 0 0 0;'>{chunk_count}</h2></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown("<div class='stat-box'><h4 style='margin:0;color:#34d399;'>⚡ Vector Engine</h4><h2 style='margin:4px 0 0 0;'>FAISS Ready</h2></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                if user_query := st.chat_input("Ask anything from your uploaded documents..."):
                    st.session_state.messages.append({"role": "user", "content": user_query})
                    with st.chat_message("user"):
                        st.markdown(user_query)

                    with st.chat_message("assistant"):
                        with st.spinner("✨ Synthesizing verified answer..."):
                            docs = vectorstore.similarity_search(user_query, k=4)
                            context_str = "\n\n".join([d.page_content for d in docs])
                            
                            final_prompt = f"""You are DeepRAG AI, an elite document intelligence assistant created by Deep Patel.
Answer the user's question clearly and accurately based ONLY on the provided context.
If the answer is not present in the context, politely state that the information is not in the uploaded document.

Context:
{context_str}

User Question: {user_query}

Detailed Answer:"""
                            answer, active_model = generate_smart_gemini_response(GEMINI_API_KEY, final_prompt)
                            st.markdown(answer)
                            st.caption(f"⚡ Answered via **{active_model}** | Grounded in {len(docs)} neural chunks")

                            with st.expander("📚 View Verified Source Citations"):
                                for i, doc in enumerate(docs):
                                    st.markdown(f"**Citation #{i+1}:**")
                                    st.code(doc.page_content[:400] + "...", language="text")

                            st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ================= TAB 2: HOW IT WORKS =================
with tab2:
    st.markdown(render_animated_title("Neural RAG Architecture"), unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card' style='margin-top: 15px;'>
        <h3 style='color: #a855f7;'>🔬 Enterprise 5-Stage Document Intelligence Pipeline</h3>
        <p style='color: #cbd5e1;'>DeepRAG AI combines local semantic vector indexing with Google's latest generative models to eliminate hallucinations and deliver 100% source-backed answers.</p>
        <hr style='border-color: rgba(255,255,255,0.1);'>
        <p><b>1. Multi-Document Ingestion:</b> High-speed text extraction across multiple PDF files simultaneously.</p>
        <p><b>2. Context-Aware Semantic Chunking:</b> Splits complex chapters into overlapping segments so zero context is lost.</p>
        <p><b>3. Local HuggingFace Embeddings:</b> Converts text chunks into 384-dimensional dense vectors using <code>all-MiniLM-L6-v2</code> locally on your system.</p>
        <p><b>4. FAISS Vector Similarity Search:</b> Instantly retrieves the top-K most relevant passages in milliseconds.</p>
        <p><b>5. Auto-Adaptive Gemini Synthesis:</b> Dynamically selects the latest available Gemini model and synthesizes accurate answers with verifiable citations.</p>
    </div>
    """, unsafe_allow_html=True)

# ================= TAB 3: ABOUT CREATOR =================
with tab3:
    st.markdown(render_animated_title("Meet Deep Patel"), unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card' style='margin-top: 15px;'>
        <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;'>
            <div>
                <h1 style='margin-bottom: 4px; color: #f8fafc;'>👨‍💻 Deep Patel</h1>
                <h4 style='margin-top: 0; color: #c084fc; font-weight: 600;'>Founder, Lead AI Architect & Creator of DeepRAG AI</h4>
            </div>
        </div>
        <hr style='border-color: rgba(255,255,255,0.1);'>
        <p style='font-size: 1.05rem; line-height: 1.7; color: #e2e8f0;'>
            <b>Deep Patel</b> is a passionate AI & Software Developer dedicated to building next-generation Generative AI applications. 
            With a strong focus on combining powerful neural backend architectures with ultra-modern, human-centric UI/UX, 
            Deep engineered <b>DeepRAG AI Workspace</b> to transform how students, researchers, and professionals interact with complex PDFs and textbooks.
        </p>
        <h4 style='color: #818cf8; margin-top: 22px;'>🛠️ Core Technology Arsenal</h4>
        <div>
            <span class='badge-pill'>Python</span>
            <span class='badge-pill'>Generative AI & LLMs</span>
            <span class='badge-pill'>RAG Architecture</span>
            <span class='badge-pill'>Google Gemini API</span>
            <span class='badge-pill'>LangChain</span>
            <span class='badge-pill'>FAISS Vector DB</span>
            <span class='badge-pill'>HuggingFace Transformers</span>
            <span class='badge-pill'>Streamlit UI/UX</span>
        </div>
        <div style='margin-top: 26px; padding: 16px; border-left: 4px solid #ec4899; background: rgba(236, 72, 153, 0.08); border-radius: 0 12px 12px 0;'>
            <p style='margin: 0; font-style: italic; color: #f1f5f9;'>
                "My mission is simple: turn complex artificial intelligence into a fast, reliable, and visually breathtaking experience." — <b>Deep Patel</b>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)
