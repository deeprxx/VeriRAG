import streamlit as st
import requests
import base64
import time
import hashlib
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DeepRAG AI Workspace | Enterprise Neural RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SECURE API KEY RETRIEVAL ---
DEFAULT_API_KEY = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        API_KEY = DEFAULT_API_KEY
except Exception:
    API_KEY = DEFAULT_API_KEY

# --- ULTRA-PREMIUM CYBERPUNK GLASSMORPHISM CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #07090e;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, rgba(99, 102, 241, 0.14) 0%, transparent 35%),
                radial-gradient(circle at 85% 20%, rgba(168, 85, 247, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 50% 90%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
                #07090e;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(11, 15, 25, 0.92) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.22);
    backdrop-filter: blur(18px);
}

/* Status Pill */
.status-pill {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.08));
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34d399;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.15);
}

/* Animated Letter-by-Letter Title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
}

.char {
    display: inline-block;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    opacity: 0;
    transform: translateY(12px);
    animation: fadeUp 0.45s forwards;
}

@keyframes fadeUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.cursor-blink {
    display: inline-block;
    width: 4px;
    height: 42px;
    background: #ec4899;
    margin-left: 6px;
    animation: blink 1s infinite;
    vertical-align: middle;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.02rem;
    margin-bottom: 24px;
    font-weight: 400;
}

/* Metric Glass Cards */
.metric-card {
    background: linear-gradient(145deg, rgba(19, 25, 40, 0.88), rgba(13, 18, 30, 0.92));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px 18px;
    text-align: center;
    transition: all 0.3s ease;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 85, 247, 0.55);
    box-shadow: 0 12px 35px rgba(99, 102, 241, 0.22);
}

.metric-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #818cf8;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f8fafc;
}

/* Custom Tabs */
div[data-testid="stTabs"] button {
    background: rgba(30, 41, 59, 0.55) !important;
    border-radius: 10px !important;
    padding: 8px 18px !important;
    margin-right: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 18px rgba(139, 92, 246, 0.45) !important;
}

/* Badges & Citations */
.meta-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a5b4fc;
    font-size: 0.76rem;
    padding: 3px 10px;
    border-radius: 999px;
    margin-right: 8px;
    margin-top: 6px;
}

.citation-box {
    background: rgba(15, 23, 42, 0.85);
    border-left: 3px solid #8b5cf6;
    padding: 12px 14px;
    border-radius: 8px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #cbd5e1;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# --- HELPER: AUTO-DETECT WORKING GEMINI MODEL & GENERATE ANSWER ---
def generate_smart_gemini_response(api_key: str, prompt: str) -> tuple[str, str]:
    clean_key = api_key.strip()
    if not clean_key:
        raise RuntimeError("API Key missing! Kripya Streamlit Cloud Secrets mein GEMINI_API_KEY set karein.")

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

    preferred_keywords = ["3.0-flash", "3-flash", "3.0-pro", "2.0-flash", "1.5-flash", "flash", "pro"]
    ordered_models = ["models/gemini-3.0-flash"]

    for kw in preferred_keywords:
        for m in valid_models:
            if kw in m and m not in ordered_models and "2.5-flash" not in m:
                ordered_models.append(m)

    for m in valid_models:
        if m not in ordered_models and "2.5-flash" not in m:
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


# --- HELPER: VOICE TO TEXT USING GEMINI ---
def transcribe_voice_with_gemini(api_key: str, audio_bytes: bytes) -> str:
    clean_key = api_key.strip()
    if not clean_key:
        raise RuntimeError("API Key missing! Voice transcription ke liye GEMINI_API_KEY zaroori hai.")

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {
                    "text": (
                        "Listen to this audio carefully and transcribe the spoken question accurately. "
                        "If it is in Hindi, Hinglish, or English, write it clearly in English/Hinglish letters. "
                        "Return ONLY the transcribed question text without any extra words."
                    )
                },
                {"inline_data": {"mime_type": "audio/wav", "data": b64_audio}}
            ]
        }],
        "generationConfig": {"temperature": 0.1}
    }

    models_to_try = ["models/gemini-3.0-flash", "models/gemini-2.0-flash", "models/gemini-1.5-flash"]
    last_err = ""
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={clean_key}"
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            last_err = resp.json().get("error", {}).get("message", "Audio transcription failed")

    raise RuntimeError(f"Voice Error: {last_err}")


# --- HELPER: BUILD LETTER-BY-LETTER HTML ---
def render_animated_title(text: str) -> str:
    spans = []
    for idx, ch in enumerate(text):
        delay = idx * 0.04
        display_char = "&nbsp;" if ch == " " else ch
        spans.append(f'<span class="char" style="animation-delay: {delay:.2f}s">{display_char}</span>')
    return f'<div class="hero-title">{"".join(spans)}<span class="cursor-blink"></span></div>'


# --- HELPER: LOAD EMBEDDING MODEL CACHED ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- SESSION STATE INITIALIZATION ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_pages" not in st.session_state:
    st.session_state.pdf_pages = 0
if "neural_chunks" not in st.session_state:
    st.session_state.neural_chunks = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files_key" not in st.session_state:
    st.session_state.processed_files_key = ""
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = ""


# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown("### ⚙️ DeepRAG Control Center")
    st.markdown('<div class="status-pill">🟢 Neural Engine: Connected & Active</div>', unsafe_allow_html=True)

    st.markdown("#### 📂 Document Portal")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload textbooks, research papers, or notes."
    )

    st.markdown("---")
    st.markdown("#### 🎛️ AI Response Mode")
    response_mode = st.selectbox(
        "Select Answer Style",
        [
            "⚡ Deep Analysis (Default)",
            "🎯 Exam Ready Notes",
            "💡 Simple Explain (ELI5)",
            "📌 Quick Bullet Points"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🛠️ Neural Parameters")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=600, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=100, step=25)
    top_k = st.slider("Retrieved Citations (Top-K)", min_value=2, max_value=8, value=4, step=1)

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📥 Workspace Actions")
        export_lines = ["# 🧠 DeepRAG AI Workspace - Study Report\n\n"]
        for turn in st.session_state.chat_history:
            export_lines.append(f"### 🙋 Question:\n{turn['question']}\n\n")
            export_lines.append(f"### 🤖 DeepRAG Answer ({turn.get('model', 'Gemini')}):\n{turn['answer']}\n\n---\n\n")

        st.download_button(
            label="📄 Download Chat Report (.md)",
            data="".join(export_lines),
            file_name="DeepRAG_Study_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# --- AUTOMATIC PDF INDEXING WITH PAGE METADATA ---
if uploaded_files:
    current_key = f"{[f.name for f in uploaded_files]}_{chunk_size}_{chunk_overlap}"
    if current_key != st.session_state.processed_files_key:
        with st.spinner("🧬 Indexing PDF pages into FAISS Neural Vector Space..."):
            docs = []
            total_pages = 0
            for pdf in uploaded_files:
                reader = PdfReader(pdf)
                total_pages += len(reader.pages)
                for page_idx, page in enumerate(reader.pages):
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        docs.append(
                            Document(
                                page_content=page_text,
                                metadata={"source": pdf.name, "page": page_idx + 1}
                            )
                        )

            if docs:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                split_docs = splitter.split_documents(docs)
                embeddings = get_embedding_model()
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                st.session_state.pdf_pages = total_pages
                st.session_state.neural_chunks = len(split_docs)
                st.session_state.processed_files_key = current_key
            else:
                st.warning("Uploaded PDF mein readable text nahi mila. Kripya text-based PDF upload karein.")
else:
    st.session_state.vector_store = None
    st.session_state.pdf_pages = 0
    st.session_state.neural_chunks = 0
    st.session_state.processed_files_key = ""


# --- TOP NAVIGATION TABS ---
tab_workspace, tab_how, tab_creator = st.tabs([
    "🚀 AI Workspace",
    "⚡ How It Works",
    "👨‍💻 About Creator"
])

with tab_workspace:
    st.markdown(render_animated_title("DeepRAG AI Workspace"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Next-Gen Intelligent Document Workspace powered by Neural RAG, Voice AI & Auto-Adaptive Gemini Engine.</div>',
        unsafe_allow_html=True
    )

    # 3 Metric Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📄 PDF Pages</div>
            <div class="metric-value">{st.session_state.pdf_pages}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🧩 Neural Chunks</div>
            <div class="metric-value">{st.session_state.neural_chunks}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        engine_status = "FAISS Ready" if st.session_state.vector_store else "Standby"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Vector Engine</div>
            <div class="metric-value">{engine_status}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # --- 1-CLICK SMART ACTIONS ---
    triggered_query = None
    if st.session_state.vector_store:
        st.markdown("##### ✨ Instant Neural Actions")
        qcol1, qcol2, qcol3 = st.columns(3)
        with qcol1:
            if st.button("📝 Executive Summary", use_container_width=True):
                triggered_query = "Provide a comprehensive executive summary of the uploaded document with key chapters and core themes."
        with qcol2:
            if st.button("🔑 Key Concepts & Formulas", use_container_width=True):
                triggered_query = "Extract all important definitions, key concepts, formulas, or critical facts from the document."
        with qcol3:
            if st.button("🎓 Generate 5 Exam Questions", use_container_width=True):
                triggered_query = "Generate 5 high-probability exam or viva questions along with their concise model answers based strictly on this document."

    # --- VOICE & TEXT INPUT BAR ---
    vcol1, vcol2 = st.columns([2, 1])
    with vcol2:
        audio_value = None
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input("🎙️ Voice Question (Bol kar poochein)")

    text_query = st.chat_input("Ask anything from your uploaded documents...")

    # Determine active user query
    active_query = None
    if triggered_query:
        active_query = triggered_query
    elif text_query:
        active_query = text_query
    elif audio_value is not None:
        audio_bytes = audio_value.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if audio_hash != st.session_state.last_audio_hash:
            st.session_state.last_audio_hash = audio_hash
            with st.spinner("🎧 Aapki aawaz transcribe ki ja rahi hai..."):
                try:
                    transcribed_text = transcribe_voice_with_gemini(API_KEY, audio_bytes)
                    active_query = transcribed_text
                    st.toast(f"🎙️ Voice Recognized: {transcribed_text}")
                except Exception as e:
                    st.error(f"❌ {e}")

    # --- PROCESS QUERY ---
    if active_query:
        if not st.session_state.vector_store:
            st.warning("⚠️ Pehle left sidebar mein जाकर ek PDF document upload karein!")
        else:
            start_time = time.time()
            with st.spinner("🧠 Searching FAISS Neural Vectors & synthesizing answer..."):
                try:
                    retrieved_docs = st.session_state.vector_store.similarity_search(active_query, k=top_k)
                    context_parts = []
                    citations_list = []

                    for idx, doc in enumerate(retrieved_docs):
                        src = doc.metadata.get("source", "Document.pdf")
                        pg = doc.metadata.get("page", "?")
                        context_parts.append(f"[Source {idx+1}: {src}, Page {pg}]\n{doc.page_content}")
                        citations_list.append({
                            "source": src,
                            "page": pg,
                            "snippet": doc.page_content[:350] + "..."
                        })

                    context_block = "\n\n".join(context_parts)

                    # Include recent chat memory (last 2 turns)
                    history_context = ""
                    if st.session_state.chat_history:
                        recent_turns = st.session_state.chat_history[-2:]
                        history_context = "\n".join(
                            [f"Q: {t['question']}\nA: {t['answer']}" for t in recent_turns]
                        )

                    style_instructions = {
                        "⚡ Deep Analysis (Default)": "Provide a clear, well-structured, and comprehensive explanation.",
                        "🎯 Exam Ready Notes": "Format the answer as high-scoring exam revision notes with bold keywords, bullet points, and clear definitions.",
                        "💡 Simple Explain (ELI5)": "Explain the concept in super simple, everyday language with an easy real-world example.",
                        "📌 Quick Bullet Points": "Give a crisp, to-the-point summary in 4 to 6 short bullet points."
                    }[response_mode]

                    final_prompt = f"""You are DeepRAG AI, an elite academic and enterprise document assistant.
Answer the user's question strictly using the provided PDF Context below.
If the answer is not in the context, clearly state that it is not mentioned in the uploaded document.
Always cite the Page numbers (e.g., [Page X]) where relevant.

Style Instruction: {style_instructions}

Recent Conversation History:
{history_context}

Retrieved PDF Context:
{context_block}

User Question: {active_query}
"""
                    answer_text, used_model = generate_smart_gemini_response(API_KEY, final_prompt)
                    elapsed = round(time.time() - start_time, 2)

                    st.session_state.chat_history.append({
                        "question": active_query,
                        "answer": answer_text,
                        "model": used_model,
                        "latency": elapsed,
                        "mode": response_mode,
                        "citations": citations_list
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # --- DISPLAY CONVERSATION HISTORY (NEWEST FIRST) ---
    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user", avatar="🟠"):
            st.markdown(f"**{turn['question']}**")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(turn["answer"])
            st.markdown(
                f'<span class="meta-badge">⚡ Engine: {turn["model"]}</span>'
                f'<span class="meta-badge">⏱️ {turn["latency"]}s</span>'
                f'<span class="meta-badge">🎛️ {turn["mode"]}</span>',
                unsafe_allow_html=True
            )

            with st.expander("📚 View Verified PDF Page Citations & Sources"):
                for c_idx, cite in enumerate(turn["citations"]):
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>📌 Citation {c_idx+1} — {cite["source"]} (Page {cite["page"]})</strong><br>'
                        f'{cite["snippet"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

with tab_how:
    st.markdown("### ⚡ How DeepRAG Neural Architecture Works")
    st.markdown("""
    1. **📂 Multi-PDF Page Extraction (`pypdf`):** Har PDF page ko extract karke uske exact page number aur file name ke metadata ke saath tag kiya jata hai.
    2. **🧩 Context-Aware Semantic Chunking (`LangChain`):** `RecursiveCharacterTextSplitter` bade chapters ko overlapping neural chunks mein todta hai taaki context kabhi break na ho.
    3. **🧬 384-Dimensional Dense Embeddings (`HuggingFace`):** `all-MiniLM-L6-v2` transformer model har chunk ko high-dimensional vector space mein map karta hai.
    4. **⚡ Millisecond Vector Retrieval (`FAISS`):** Aapke text ya voice sawaal se sabse zyada match karne wale top chunks milliseconds mein dhoondhe jate hain.
    5. **🤖 Auto-Adaptive Gemini 3.0 Synthesis:** Retrieved chunks aur conversation memory ko `gemini-3.0-flash` ke paas bhej kar 100% source-backed jawab generate hota hai.
    """)

with tab_creator:
    st.markdown("### 👨‍💻 About the Creator")
    st.markdown("""
    **Deep Patel**  
    *Founder, Lead AI Architect & Creator of DeepRAG AI Workspace*

    > *"My mission is simple: turn complex artificial intelligence into a fast, reliable, and visually breathtaking experience for students, researchers, and builders."*
    """)
