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

# --- ULTRA-PREMIUM ENTERPRISE GLASSMORPHISM & ANIMATION CSS ---
custom_css = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');"
    "html, body, [class*='css'] { font-family: 'Inter', sans-serif; background-color: #06080f; color: #e2e8f0; }"
    ".stApp {"
    "  background: radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.16) 0%, transparent 38%),"
    "              radial-gradient(circle at 88% 18%, rgba(168, 85, 247, 0.14) 0%, transparent 42%),"
    "              radial-gradient(circle at 50% 88%, rgba(16, 185, 129, 0.08) 0%, transparent 45%), #06080f;"
    "}"
    "section[data-testid='stSidebar'] {"
    "  background: rgba(10, 14, 24, 0.94) !important;"
    "  border-right: 1px solid rgba(99, 102, 241, 0.25);"
    "  backdrop-filter: blur(20px);"
    "}"
    ".status-pill {"
    "  background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(5, 150, 105, 0.08));"
    "  border: 1px solid rgba(16, 185, 129, 0.45);"
    "  color: #34d399; padding: 10px 14px; border-radius: 12px; font-size: 0.84rem;"
    "  font-weight: 600; text-align: center; margin-bottom: 18px;"
    "  box-shadow: 0 0 22px rgba(16, 185, 129, 0.15); animation: pulseGlow 3s infinite ease-in-out;"
    "}"
    "@keyframes pulseGlow {"
    "  0%, 100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.15); }"
    "  50% { box-shadow: 0 0 28px rgba(16, 185, 129, 0.35); }"
    "}"
    ".hero-title {"
    "  font-family: 'Space Grotesk', sans-serif; font-size: 2.9rem; font-weight: 800;"
    "  letter-spacing: -0.02em; margin-bottom: 6px; display: flex; flex-wrap: wrap; align-items: center;"
    "}"
    ".char {"
    "  display: inline-block; background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);"
    "  -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    "  opacity: 0; transform: translateY(14px) scale(0.95);"
    "  animation: fadeUpChar 0.45s cubic-bezier(0.22, 1, 0.36, 1) forwards;"
    "}"
    "@keyframes fadeUpChar { to { opacity: 1; transform: translateY(0) scale(1); } }"
    ".cursor-blink {"
    "  display: inline-block; width: 4px; height: 40px; background: #ec4899;"
    "  margin-left: 8px; border-radius: 2px; animation: blink 1s infinite;"
    "}"
    "@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }"
    ".hero-subtitle {"
    "  color: #94a3b8; font-size: 1.04rem; margin-bottom: 20px; font-weight: 400;"
    "  animation: fadeInSlide 0.8s ease forwards;"
    "}"
    "@keyframes fadeInSlide { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }"
    ".metric-card {"
    "  background: linear-gradient(145deg, rgba(19, 25, 42, 0.88), rgba(12, 17, 30, 0.94));"
    "  border: 1px solid rgba(99, 102, 241, 0.28); border-radius: 18px; padding: 22px 18px;"
    "  text-align: center; transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);"
    "  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4); animation: fadeInSlide 0.6s ease forwards;"
    "}"
    ".metric-card:hover {"
    "  transform: translateY(-6px); border-color: rgba(168, 85, 247, 0.65);"
    "  box-shadow: 0 16px 40px rgba(99, 102, 241, 0.25);"
    "}"
    ".metric-title {"
    "  font-size: 0.92rem; font-weight: 600; color: #a5b4fc; margin-bottom: 8px;"
    "  text-transform: uppercase; letter-spacing: 0.05em;"
    "}"
    ".metric-value {"
    "  font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800;"
    "  background: linear-gradient(135deg, #ffffff, #cbd5e1);"
    "  -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    "}"
    "div[data-testid='stTabs'] button {"
    "  background: rgba(30, 41, 59, 0.55) !important; border-radius: 12px !important;"
    "  padding: 10px 20px !important; margin-right: 10px !important;"
    "  border: 1px solid rgba(148, 163, 184, 0.18) !important; color: #cbd5e1 !important;"
    "  font-weight: 600 !important; transition: all 0.3s ease !important;"
    "}"
    "div[data-testid='stTabs'] button:hover {"
    "  border-color: rgba(129, 140, 248, 0.5) !important; transform: translateY(-2px);"
    "}"
    "div[data-testid='stTabs'] button[aria-selected='true'] {"
    "  background: linear-gradient(135deg, #6366f1, #a855f7) !important; color: #ffffff !important;"
    "  border: none !important; box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45) !important;"
    "}"
    ".meta-badge {"
    "  display: inline-block; background: rgba(99, 102, 241, 0.16);"
    "  border: 1px solid rgba(99, 102, 241, 0.42); color: #c7d2fe; font-size: 0.76rem;"
    "  font-weight: 600; padding: 4px 12px; border-radius: 999px; margin-right: 8px; margin-top: 8px;"
    "}"
    ".citation-box {"
    "  background: rgba(15, 23, 42, 0.9); border-left: 4px solid #8b5cf6;"
    "  border-top: 1px solid rgba(148, 163, 184, 0.12); border-right: 1px solid rgba(148, 163, 184, 0.12);"
    "  border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding: 14px 16px;"
    "  border-radius: 10px; margin-bottom: 12px; font-size: 0.88rem; color: #cbd5e1; transition: all 0.25s ease;"
    "}"
    ".citation-box:hover { border-left-color: #ec4899; background: rgba(23, 33, 56, 0.95); }"
    ".arch-card {"
    "  background: linear-gradient(145deg, rgba(18, 24, 41, 0.9), rgba(11, 16, 28, 0.95));"
    "  border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 18px; padding: 24px; margin-bottom: 18px;"
    "  position: relative; overflow: hidden; transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);"
    "  opacity: 0; transform: translateY(20px); animation: fadeUpCard 0.6s ease forwards;"
    "}"
    "@keyframes fadeUpCard { to { opacity: 1; transform: translateY(0); } }"
    ".arch-card:hover {"
    "  transform: translateY(-5px) scale(1.01); border-color: rgba(168, 85, 247, 0.65);"
    "  box-shadow: 0 14px 36px rgba(99, 102, 241, 0.22);"
    "}"
    ".step-badge {"
    "  display: inline-block; background: linear-gradient(135deg, #6366f1, #a855f7); color: #ffffff;"
    "  font-family: 'Space Grotesk', sans-serif; font-size: 0.78rem; font-weight: 700;"
    "  padding: 4px 12px; border-radius: 999px; margin-bottom: 10px; letter-spacing: 0.06em; text-transform: uppercase;"
    "}"
    ".arch-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px; }"
    ".arch-desc { color: #94a3b8; font-size: 0.95rem; line-height: 1.6; }"
    ".creator-hero {"
    "  background: linear-gradient(135deg, rgba(20, 27, 45, 0.92), rgba(12, 16, 29, 0.96));"
    "  border: 1px solid rgba(139, 92, 246, 0.38); border-radius: 24px; padding: 40px 34px; text-align: center;"
    "  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(139, 92, 246, 0.14);"
    "  animation: fadeUpCard 0.65s ease forwards; margin-top: 12px;"
    "}"
    ".avatar-ring {"
    "  width: 96px; height: 96px; border-radius: 50%; margin: 0 auto 18px auto;"
    "  background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899); display: flex;"
    "  align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif;"
    "  font-size: 2.2rem; font-weight: 800; color: #ffffff; box-shadow: 0 0 30px rgba(168, 85, 247, 0.5);"
    "  animation: floatAvatar 3.5s ease-in-out infinite;"
    "}"
    "@keyframes floatAvatar { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }"
    ".creator-name {"
    "  font-family: 'Space Grotesk', sans-serif; font-size: 2.4rem; font-weight: 800;"
    "  background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);"
    "  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 6px;"
    "}"
    ".creator-role { color: #cbd5e1; font-size: 1.05rem; font-weight: 600; letter-spacing: 0.03em; margin-bottom: 22px; }"
    ".creator-quote {"
    "  background: rgba(15, 23, 42, 0.78); border-left: 4px solid #a855f7; border-radius: 12px;"
    "  padding: 18px 24px; max-width: 720px; margin: 0 auto 26px auto; color: #e2e8f0;"
    "  font-style: italic; font-size: 1.02rem; line-height: 1.65;"
    "}"
    ".skill-pill {"
    "  display: inline-block; background: rgba(99, 102, 241, 0.14); border: 1px solid rgba(129, 140, 248, 0.35);"
    "  color: #c7d2fe; padding: 7px 16px; border-radius: 999px; font-size: 0.85rem; font-weight: 600;"
    "  margin: 5px; transition: all 0.25s ease;"
    "}"
    ".skill-pill:hover {"
    "  background: linear-gradient(135deg, #6366f1, #a855f7); color: #ffffff;"
    "  transform: translateY(-3px); box-shadow: 0 6px 16px rgba(139, 92, 246, 0.35);"
    "}"
    ".contact-box {"
    "  background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(99, 102, 241, 0.3);"
    "  border-radius: 12px; padding: 14px; text-align: center; margin-top: 20px;"
    "}"
    ".contact-title {"
    "  font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; font-weight: 700; color: #818cf8; margin-bottom: 4px;"
    "}"
    ".contact-handle {"
    "  font-size: 0.95rem; font-weight: 600; color: #f8fafc;"
    "}"
    ".premium-footer {"
    "  margin-top: 48px; padding: 18px 0 10px 0; border-top: 1px solid rgba(99, 102, 241, 0.2);"
    "  text-align: center; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem;"
    "  font-weight: 600; color: #94a3b8; letter-spacing: 0.03em; animation: fadeInSlide 0.8s ease forwards;"
    "}"
    ".premium-footer span {"
    "  background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);"
    "  -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800;"
    "}"
    /* AGENTIC SWARM MULTI-CURSOR & MATRIX SELECTION CSS */
    ".agent-swarm-layer {"
    "  position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;"
    "  pointer-events: none; z-index: 9999; overflow: hidden;"
    "}"
    ".agent-cursor {"
    "  position: absolute; display: flex; align-items: center; gap: 6px;"
    "  filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.5));"
    "  will-change: transform; opacity: 0.9;"
    "}"
    ".agent-tag {"
    "  font-family: 'JetBrains Mono', monospace; font-size: 0.66rem; font-weight: 700;"
    "  padding: 3px 8px; border-radius: 6px; color: #06080f; white-space: nowrap;"
    "  letter-spacing: 0.04em; box-shadow: 0 4px 14px rgba(0,0,0,0.5);"
    "}"
    ".cursor-1 { animation: roamAgent1 15s infinite ease-in-out; }"
    ".cursor-2 { animation: roamAgent2 19s infinite ease-in-out; }"
    ".cursor-3 { animation: roamAgent3 17s infinite ease-in-out; }"
    ".cursor-4 { animation: roamAgent4 21s infinite ease-in-out; }"
    ".cursor-5 { animation: roamAgent5 14s infinite ease-in-out; }"
    "@keyframes roamAgent1 {"
    "  0%, 100% { transform: translate(16vw, 20vh); }"
    "  25% { transform: translate(64vw, 28vh) scale(0.95); }"
    "  50% { transform: translate(46vw, 66vh); }"
    "  75% { transform: translate(24vw, 52vh) scale(1.04); }"
    "}"
    "@keyframes roamAgent2 {"
    "  0%, 100% { transform: translate(76vw, 18vh); }"
    "  30% { transform: translate(32vw, 40vh); }"
    "  60% { transform: translate(70vw, 62vh) scale(0.95); }"
    "  85% { transform: translate(52vw, 24vh); }"
    "}"
    "@keyframes roamAgent3 {"
    "  0%, 100% { transform: translate(40vw, 76vh); }"
    "  35% { transform: translate(80vw, 44vh); }"
    "  65% { transform: translate(26vw, 26vh) scale(0.92); }"
    "}"
    "@keyframes roamAgent4 {"
    "  0%, 100% { transform: translate(62vw, 72vh); }"
    "  40% { transform: translate(20vw, 36vh); }"
    "  75% { transform: translate(56vw, 46vh); }"
    "}"
    "@keyframes roamAgent5 {"
    "  0%, 100% { transform: translate(30vw, 14vh); }"
    "  45% { transform: translate(74vw, 34vh); }"
    "  80% { transform: translate(38vw, 58vh); }"
    "}"
    ".telemetry-bar {"
    "  display: inline-flex; flex-wrap: wrap; align-items: center; gap: 12px;"
    "  background: rgba(11, 15, 28, 0.92); border: 1px solid rgba(99, 102, 241, 0.35);"
    "  padding: 7px 16px; border-radius: 999px; font-size: 0.78rem; color: #cbd5e1;"
    "  margin-bottom: 18px; box-shadow: 0 8px 25px rgba(0,0,0,0.45);"
    "}"
    ".dot-cluster { display: inline-flex; gap: 5px; align-items: center; }"
    ".dot-c { width: 7px; height: 7px; border-radius: 50%; display: inline-block; animation: blink 1.5s infinite; }"
    ".matrix-select-wrap {"
    "  position: relative; display: inline-flex; align-items: center; gap: 6px;"
    "  padding: 2px 9px; border-radius: 4px; font-family: 'JetBrains Mono', monospace;"
    "  font-size: 0.75rem; color: #e0f2fe; animation: matrixHighlight 4.5s infinite ease-in-out;"
    "}"
    "@keyframes matrixHighlight {"
    "  0%, 100% { background: rgba(56, 189, 248, 0.06); border: 1px solid rgba(56, 189, 248, 0.2); }"
    "  40%, 70% { background: rgba(56, 189, 248, 0.25); border: 1px solid #38bdf8; box-shadow: 0 0 14px rgba(56, 189, 248, 0.3); }"
    "}"
    ".crosshair-plus {"
    "  color: #38bdf8; font-weight: 800; font-size: 0.85rem; animation: blink 1.2s infinite;"
    "}"
    "footer { visibility: hidden; }"
    "</style>"
)
st.markdown(custom_css, unsafe_allow_html=True)


# --- HELPER: DYNAMICALLY FETCH ACTIVE GEMINI MODELS ---
def get_ordered_gemini_models(clean_key: str) -> list[str]:
    list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
    resp = requests.get(list_url, timeout=15)
    if resp.status_code != 200:
        err_msg = resp.json().get("error", {}).get("message", "Invalid API Key or network connection error.")
        raise RuntimeError(f"API Key Verification Failed: {err_msg}")

    models_data = resp.json().get("models", [])
    valid_models = [
        m["name"] for m in models_data
        if "generateContent" in m.get("supportedGenerationMethods", [])
        and "gemini" in m["name"].lower()
        and "embedding" not in m["name"].lower()
        and "imagen" not in m["name"].lower()
    ]

    preferred_keywords = ["3.0-flash", "3-flash", "2.5-flash", "2.0-flash", "1.5-flash", "flash", "pro"]
    ordered_models = []

    for kw in preferred_keywords:
        for m in valid_models:
            if kw in m and m not in ordered_models:
                ordered_models.append(m)

    for m in valid_models:
        if m not in ordered_models:
            ordered_models.append(m)

    return ordered_models


# --- HELPER: AUTO-DETECT WORKING GEMINI MODEL & GENERATE ANSWER ---
def generate_smart_gemini_response(api_key: str, prompt: str) -> tuple[str, str]:
    clean_key = api_key.strip()
    if not clean_key:
        raise RuntimeError("API Key is missing! Please configure GEMINI_API_KEY in Streamlit Cloud Secrets.")

    ordered_models = get_ordered_gemini_models(clean_key)

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
            last_error = gen_resp.json().get("error", {}).get("message", "Model generation failed.")

    raise RuntimeError(f"Model Synthesis Error: {last_error}")


# --- HELPER: VOICE TO TEXT USING AUTO-DETECTED GEMINI MODEL ---
def transcribe_voice_with_gemini(api_key: str, audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    clean_key = api_key.strip()
    if not clean_key:
        raise RuntimeError("API Key is missing! GEMINI_API_KEY is required for voice transcription.")

    ordered_models = get_ordered_gemini_models(clean_key)
    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    safe_mime = mime_type if mime_type else "audio/wav"

    payload = {
        "contents": [{
            "parts": [
                {
                    "text": (
                        "Listen to this audio recording carefully and transcribe the spoken question into clear English. "
                        "If the speaker asks in Hindi or Hinglish, translate or transcribe the question accurately into clear English. "
                        "Return ONLY the transcribed question text without any extra commentary."
                    )
                },
                {"inline_data": {"mime_type": safe_mime, "data": b64_audio}}
            ]
        }],
        "generationConfig": {"temperature": 0.1}
    }

    last_err = ""
    for model_name in ordered_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={clean_key}"
        resp = requests.post(url, json=payload, timeout=40)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and "text" in parts[0]:
                    return parts[0]["text"].strip()
        else:
            last_err = resp.json().get("error", {}).get("message", "Audio transcription failed.")

    raise RuntimeError(f"Voice Transcription Error: {last_err}")


# --- HELPER: BUILD LETTER-BY-LETTER HTML ---
def render_animated_title(text: str) -> str:
    spans = []
    for idx, ch in enumerate(text):
        delay = idx * 0.035
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

    enable_swarm = st.toggle(
        "🤖 Agentic Swarm Mode",
        value=True,
        help="Enable autonomous multi-agent cursors and live neural matrix telemetry."
    )

    st.markdown("---")
    st.markdown("#### 📂 Document Portal")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload textbooks, research papers, legal contracts, or lecture notes."
    )

    st.markdown("---")
    st.markdown("#### 🎛️ AI Synthesis Mode")
    response_mode = st.selectbox(
        "Select Output Style",
        [
            "⚡ Deep Analysis (Default)",
            "🎯 Exam & Revision Notes",
            "💡 Simple Explain (ELI5)",
            "📌 Executive Bullet Points"
        ]
    )

    st.markdown("---")
    st.markdown("#### 🛠️ Neural Parameters")
    chunk_size = st.slider("Chunk Size (Characters)", min_value=200, max_value=1500, value=600, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=100, step=25)
    top_k = st.slider("Retrieved Citations (Top-K)", min_value=2, max_value=8, value=4, step=1)

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📥 Workspace Actions")
        export_lines = ["# 🧠 DeepRAG AI Workspace — Research & Study Report\n\n"]
        for turn in st.session_state.chat_history:
            export_lines.append(f"### 🙋 Query:\n{turn['question']}\n\n")
            export_lines.append(f"### 🤖 DeepRAG Synthesis ({turn.get('model', 'Gemini')}):\n{turn['answer']}\n\n---\n\n")

        st.download_button(
            label="📄 Export Session Report (.md)",
            data="".join(export_lines),
            file_name="DeepRAG_Analysis_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # --- CONTACT US SECTION ---
    st.markdown("---")
    st.markdown(
        '<div class="contact-box">'
        '<div class="contact-title">📞 Contact Us</div>'
        '<div class="contact-handle">Telegram @DeepRAGAI</div>'
        '</div>',
        unsafe_allow_html=True
    )


# --- RENDER AGENTIC MULTI-CURSOR SWARM OVERLAY ---
if enable_swarm:
    swarm_overlay_html = (
        "<div class='agent-swarm-layer'>"
        "  <div class='agent-cursor cursor-1'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#38bdf8'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#38bdf8;'>Agent_01 • Retriever</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-2'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#34d399'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#34d399;'>Agent_02 • FAISS_Index</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-3'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#c084fc'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#c084fc;'>Agent_03 • Gemini_Synth</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-4'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#fb7185'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#fb7185;'>Agent_04 • Citation_Guard</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-5'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#facc15'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#facc15;'>Agent_05 • Voice_Node</span>"
        "  </div>"
        "</div>"
    )
    st.markdown(swarm_overlay_html, unsafe_allow_html=True)


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
                st.warning("No extractable text found in the uploaded PDF. Please upload a text-based PDF document.")
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

# ==========================================
# TAB 1: AI WORKSPACE
# ==========================================
with tab_workspace:
    st.markdown(render_animated_title("DeepRAG AI Workspace"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Next-Gen Intelligent Document Workspace powered by Neural RAG, Voice AI & Auto-Adaptive Gemini Engine.</div>',
        unsafe_allow_html=True
    )

    # Agentic Systems Matrix Telemetry Pill
    st.markdown(
        "<div class='telemetry-bar'>"
        "  <span class='dot-cluster'>"
        "    <span class='dot-c' style='background:#ef4444;'></span>"
        "    <span class='dot-c' style='background:#f59e0b; animation-delay:0.2s;'></span>"
        "    <span class='dot-c' style='background:#10b981; animation-delay:0.4s;'></span>"
        "    <span class='dot-c' style='background:#38bdf8; animation-delay:0.6s;'></span>"
        "    <span class='dot-c' style='background:#c084fc; animation-delay:0.8s;'></span>"
        "  </span>"
        "  <span><strong>5 Autonomous Agents</strong> Active</span>"
        "  <span style='color:#475569;'>|</span>"
        "  <span class='matrix-select-wrap'><span class='crosshair-plus'>+</span> The Systems Matrix • 384D Latent Space</span>"
        "</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">📄 Indexed PDF Pages</div>'
            f'<div class="metric-value">{st.session_state.pdf_pages}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">🧩 Neural Vector Chunks</div>'
            f'<div class="metric-value">{st.session_state.neural_chunks}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with col3:
        engine_status = "FAISS Active" if st.session_state.vector_store else "Standby"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-title">⚡ Vector Search Engine</div>'
            f'<div class="metric-value">{engine_status}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.write("")

    triggered_query = None
    if st.session_state.vector_store:
        st.markdown("##### ✨ Instant Neural Actions")
        qcol1, qcol2, qcol3, qcol4 = st.columns(4)
        with qcol1:
            if st.button("📝 Executive Summary", use_container_width=True):
                triggered_query = "Provide a comprehensive executive summary of the uploaded document, highlighting its core thesis, key sections, and main conclusions."
        with qcol2:
            if st.button("🔑 Key Concepts & Formulas", use_container_width=True):
                triggered_query = "Extract all critical definitions, core concepts, frameworks, formulas, and key technical facts from the document."
        with qcol3:
            if st.button("🎓 Top 5 Exam Questions", use_container_width=True):
                triggered_query = "Generate 5 high-value exam or interview questions along with detailed model answers strictly based on this document."
        with qcol4:
            if st.button("🚀 Actionable Insights", use_container_width=True):
                triggered_query = "Synthesize the most important actionable insights, practical takeaways, and strategic recommendations from this document."

    vcol1, vcol2 = st.columns([2, 1])
    with vcol1:
        st.markdown("##### 💬 Ask Your Documents (Text or Voice)")
        st.caption("Type your question below or record a voice query using the microphone on the right.")
    with vcol2:
        audio_value = None
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input("🎙️ Ask via Voice AI")

    text_query = st.chat_input("Ask any question about your uploaded PDF documents...")

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
            with st.spinner("🎧 Transcribing your voice query with Gemini Audio Engine..."):
                try:
                    mime_type = getattr(audio_value, "type", "audio/wav") or "audio/wav"
                    transcribed_text = transcribe_voice_with_gemini(API_KEY, audio_bytes, mime_type)
                    active_query = transcribed_text
                    st.toast(f"🎙️ Voice Query Recognized: {transcribed_text}")
                except Exception as e:
                    st.error(f"❌ {e}")

    if active_query:
        if not st.session_state.vector_store:
            st.warning("⚠️ Please upload at least one PDF document in the left sidebar to begin querying.")
        else:
            start_time = time.time()
            with st.spinner("🧠 Querying FAISS Vector Space & synthesizing verified response..."):
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
                            "snippet": doc.page_content[:360].replace("\n", " ") + "..."
                        })

                    context_block = "\n\n".join(context_parts)

                    history_context = ""
                    if st.session_state.chat_history:
                        recent_turns = st.session_state.chat_history[-2:]
                        history_context = "\n".join(
                            [f"User: {t['question']}\nAssistant: {t['answer']}" for t in recent_turns]
                        )

                    style_instructions = {
                        "⚡ Deep Analysis (Default)": "Provide a thorough, well-structured, and authoritative explanation in English.",
                        "🎯 Exam & Revision Notes": "Format the response as high-impact study notes with bold key terms, structured bullet points, and clear definitions in English.",
                        "💡 Simple Explain (ELI5)": "Explain the concept in clear, beginner-friendly English using an intuitive real-world analogy.",
                        "📌 Executive Bullet Points": "Deliver a crisp executive summary in 4 to 6 high-impact bullet points in English."
                    }[response_mode]

                    final_prompt = (
                        "You are DeepRAG AI, an elite enterprise and academic document intelligence assistant.\n"
                        "Answer the user's question strictly in English using ONLY the provided PDF Context below.\n"
                        "If the answer cannot be found within the provided context, state clearly that the information is not present in the uploaded documents.\n"
                        "Always cite the relevant page numbers inline (for example, [Page 4]) when stating facts.\n\n"
                        f"Output Style Instruction: {style_instructions}\n\n"
                        f"Recent Conversation Context:\n{history_context}\n\n"
                        f"Retrieved PDF Context:\n{context_block}\n\n"
                        f"User Question: {active_query}\n"
                    )

                    answer_text, used_model = generate_smart_gemini_response(API_KEY, final_prompt)
                    elapsed = round(time.time() - start_time, 2)
                    word_count = len(answer_text.split())

                    st.session_state.chat_history.append({
                        "question": active_query,
                        "answer": answer_text,
                        "model": used_model,
                        "latency": elapsed,
                        "words": word_count,
                        "mode": response_mode,
                        "citations": citations_list
                    })
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    for turn in reversed(st.session_state.chat_history):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{turn['question']}**")

        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(turn["answer"])
            st.markdown(
                f'<span class="meta-badge">⚡ Engine: {turn["model"]}</span>'
                f'<span class="meta-badge">⏱️ Latency: {turn["latency"]}s</span>'
                f'<span class="meta-badge">📝 {turn.get("words", 0)} Words</span>'
                f'<span class="meta-badge">🎛️ {turn["mode"]}</span>',
                unsafe_allow_html=True
            )

            with st.expander("📚 View Verified PDF Page Citations & Retrieved Passages"):
                for c_idx, cite in enumerate(turn["citations"]):
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>📌 Source {c_idx+1} — {cite["source"]} (Page {cite["page"]})</strong><br>'
                        f'{cite["snippet"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ==========================================
# TAB 2: ANIMATED HOW IT WORKS
# ==========================================
with tab_how:
    st.markdown(render_animated_title("Neural RAG Architecture"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">How DeepRAG transforms static PDF documents into an interactive, hallucination-free neural knowledge base.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="arch-card" style="animation-delay: 0.08s;">'
        '<span class="step-badge">Stage 01 • Ingestion</span>'
        '<div class="arch-title">📂 Multi-Document Parsing & Page-Level Metadata Tagging</div>'
        '<div class="arch-desc">Uploaded PDF files are processed through a high-speed <code>pypdf</code> extraction pipeline. Every single page is isolated, cleaned, and tagged with its exact source filename and page number so downstream answers retain 100% verifiable traceability.</div>'
        '</div>'
        '<div class="arch-card" style="animation-delay: 0.18s;">'
        '<span class="step-badge">Stage 02 • Semantic Splitting</span>'
        '<div class="arch-title">🧩 Context-Aware Recursive Chunking</div>'
        '<div class="arch-desc">Using LangChain\'s <code>RecursiveCharacterTextSplitter</code>, long chapters are partitioned into overlapping semantic windows. Dynamic chunk overlap ensures sentences and technical arguments are never severed mid-thought.</div>'
        '</div>'
        '<div class="arch-card" style="animation-delay: 0.28s;">'
        '<span class="step-badge">Stage 03 • Neural Encoding</span>'
        '<div class="arch-title">🧬 384-Dimensional Dense Vector Embeddings</div>'
        '<div class="arch-desc">Each text chunk is passed through HuggingFace\'s <code>all-MiniLM-L6-v2</code> Sentence-Transformer model, mapping complex concepts into a 384-dimensional dense geometric vector space where meaning is measured by spatial proximity.</div>'
        '</div>'
        '<div class="arch-card" style="animation-delay: 0.38s;">'
        '<span class="step-badge">Stage 04 • Similarity Search</span>'
        '<div class="arch-title">⚡ Millisecond FAISS Vector Indexing</div>'
        '<div class="arch-desc">Facebook AI Similarity Search (<code>FAISS</code>) indexes all neural vectors in memory. When you ask a question via text or voice, your query is embedded in real time and matched against the top-K most relevant document passages in milliseconds.</div>'
        '</div>'
        '<div class="arch-card" style="animation-delay: 0.48s;">'
        '<span class="step-badge">Stage 05 • Grounded Synthesis</span>'
        '<div class="arch-title">🤖 Auto-Adaptive Gemini 3.0 Flash Synthesis</div>'
        '<div class="arch-desc">The retrieved passages, page numbers, and conversational memory are injected into a strict grounding prompt and routed to Google\'s <code>gemini-3.0-flash</code> engine—producing fast, accurate, citation-backed responses with zero hallucination.</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ==========================================
# TAB 3: ANIMATED ABOUT CREATOR
# ==========================================
with tab_creator:
    st.markdown(render_animated_title("Meet The Creator"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Engineered with precision, modern design systems, and production-grade AI architecture.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="creator-hero">'
        '<div class="avatar-ring">DP</div>'
        '<div class="creator-name">Deep Patel</div>'
        '<div class="creator-role">Founder, Lead AI Architect & Creator of DeepRAG AI Workspace</div>'
        '<div class="creator-quote">'
        '"My mission is simple: transform complex artificial intelligence and neural retrieval pipelines into a fast, reliable, and visually breathtaking workspace that empowers students, researchers, and innovators."'
        '</div>'
        '<div>'
        '<span class="skill-pill">🧠 Generative AI & LLMs</span>'
        '<span class="skill-pill">⚡ Neural RAG Pipelines</span>'
        '<span class="skill-pill">🔍 FAISS Vector Search</span>'
        '<span class="skill-pill">🔗 LangChain Architecture</span>'
        '<span class="skill-pill">🎙️ Multimodal Voice AI</span>'
        '<span class="skill-pill">🎨 Modern Glassmorphism UI</span>'
        '<span class="skill-pill">🐍 Python & Cloud Deployment</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

# --- GLOBAL SIGNATURE FOOTER ---
st.markdown(
    '<div class="premium-footer">Engineered with ❤️ by <span>Deep</span></div>',
    unsafe_allow_html=True
)
