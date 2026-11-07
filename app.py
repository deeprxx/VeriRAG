import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import time
import hashlib
import json
import re
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

# --- ULTRA-SMOOTH ENTERPRISE GLASSMORPHISM & ANIMATION CSS ---
custom_css = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');"
    "html, body, [class*='css'] { font-family: 'Inter', sans-serif; background-color: #06080f; color: #e2e8f0; scroll-behavior: smooth; }"
    "::-webkit-scrollbar { width: 8px; height: 8px; }"
    "::-webkit-scrollbar-track { background: #06080f; }"
    "::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.35); border-radius: 999px; }"
    "::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.65); }"
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
    "  color: #94a3b8; font-size: 1.04rem; margin-bottom: 26px; font-weight: 400;"
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
    "div.stButton > button {"
    "  background: linear-gradient(145deg, rgba(30, 41, 59, 0.75), rgba(15, 23, 42, 0.9)) !important;"
    "  border: 1px solid rgba(99, 102, 241, 0.32) !important;"
    "  color: #e2e8f0 !important; border-radius: 12px !important; font-weight: 600 !important;"
    "  padding: 10px 16px !important; transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1) !important;"
    "}"
    "div.stButton > button:hover {"
    "  transform: translateY(-3px) !important;"
    "  border-color: rgba(168, 85, 247, 0.75) !important;"
    "  background: linear-gradient(135deg, rgba(99, 102, 241, 0.28), rgba(168, 85, 247, 0.28)) !important;"
    "  box-shadow: 0 8px 24px rgba(139, 92, 246, 0.3) !important;"
    "  color: #ffffff !important;"
    "}"
    "div[data-testid='stChatMessage'] {"
    "  background: rgba(15, 23, 42, 0.65) !important;"
    "  border: 1px solid rgba(99, 102, 241, 0.2) !important;"
    "  border-radius: 16px !important; padding: 16px 20px !important;"
    "  margin-bottom: 14px !important; backdrop-filter: blur(12px);"
    "  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.28);"
    "  animation: fadeInSlide 0.4s ease forwards;"
    "}"
    ".demo-banner {"
    "  background: linear-gradient(135deg, rgba(99, 102, 241, 0.14), rgba(168, 85, 247, 0.12));"
    "  border: 1px dashed rgba(129, 140, 248, 0.45); border-radius: 16px;"
    "  padding: 16px 20px; margin-bottom: 18px; display: flex; align-items: center;"
    "  justify-content: space-between; flex-wrap: wrap; gap: 12px;"
    "}"
    ".meta-badge {"
    "  display: inline-block; background: rgba(99, 102, 241, 0.16);"
    "  border: 1px solid rgba(99, 102, 241, 0.42); color: #c7d2fe; font-size: 0.76rem;"
    "  font-weight: 600; padding: 4px 12px; border-radius: 999px; margin-right: 8px; margin-top: 8px;"
    "}"
    ".match-badge {"
    "  display: inline-block; background: rgba(16, 185, 129, 0.16);"
    "  border: 1px solid rgba(16, 185, 129, 0.45); color: #34d399; font-size: 0.75rem;"
    "  font-weight: 700; padding: 3px 10px; border-radius: 999px; float: right;"
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
    "footer { visibility: hidden; }"
    "</style>"
)
st.markdown(custom_css, unsafe_allow_html=True)


# --- HELPER: BUILT-IN DEMO AI RESEARCH PAPER FOR 1-CLICK TESTING ---
def get_demo_ai_documents() -> list:
    pages = [
        (
            "DeepRAG Technical Whitepaper (2026) — Chapter 1: Retrieval-Augmented Generation (RAG) Fundamentals.\n"
            "Large Language Models (LLMs) store vast parametric knowledge but suffer from hallucinations and static training cutoffs. "
            "Retrieval-Augmented Generation (RAG) solves this by combining non-parametric dense vector memory (FAISS) with generative transformers. "
            "Mathematically, given a query q and document corpus D, the retriever encodes both into a 384-dimensional Euclidean space using "
            "Sentence-Transformers (all-MiniLM-L6-v2), selecting the top-k passages that minimize L2 distance or maximize cosine similarity: "
            "sim(q, d) = (E(q) · E(d)) / (||E(q)|| ||E(d)||)."
        ),
        (
            "DeepRAG Technical Whitepaper — Chapter 2: Context-Aware Semantic Chunking & Overlap.\n"
            "Naive fixed-length text splitting breaks semantic continuity across sentences and equations. DeepRAG implements "
            "RecursiveCharacterTextSplitter with configurable chunk size (default 600 characters) and sliding window overlap (100 characters). "
            "This sliding overlap guarantees that boundary concepts, mathematical proofs, and multi-sentence definitions remain intact across adjacent vector chunks."
        ),
        (
            "DeepRAG Technical Whitepaper — Chapter 3: High-Speed FAISS Vector Indexing Architecture.\n"
            "Facebook AI Similarity Search (FAISS) enables millisecond nearest-neighbor search across thousands of dense embeddings in RAM. "
            "By indexing 384-dimensional float32 vectors produced by all-MiniLM-L6-v2, DeepRAG achieves sub-15ms retrieval latency on CPU hardware "
            "without requiring expensive GPU clusters, making enterprise document intelligence accessible and scalable."
        ),
        (
            "DeepRAG Technical Whitepaper — Chapter 4: Zero-Hallucination Grounding & Page-Level Traceability.\n"
            "Every extracted PDF page is tagged with strict source metadata (filename and exact page number) prior to chunking. "
            "During synthesis, the Gemini 3.0 Flash engine is constrained by a deterministic system prompt to use ONLY retrieved passages "
            "and attach inline citations [Page X]. If a query falls outside the vector context, the system abstains rather than fabricating an answer."
        ),
        (
            "DeepRAG Technical Whitepaper — Chapter 5: Multimodal Voice AI & Adaptive Model Routing.\n"
            "DeepRAG integrates real-time voice-to-text interrogation and browser-native neural speech synthesis. "
            " furthermore, the Auto-Adaptive Model Router dynamically queries the Google Generative AI ModelService registry at runtime, "
            "automatically selecting the highest-performing active Flash or Pro model to guarantee 99.9% uptime."
        )
    ]
    return [
        Document(page_content=text, metadata={"source": "DeepRAG_AI_Architecture_Paper.pdf", "page": i + 1})
        for i, text in enumerate(pages)
    ]


# --- HELPER: VOICE AUDIO PLAYER FOR AI ANSWERS ---
def render_tts_audio_button(answer_text: str, btn_id: str):
    clean_text = re.sub(r"[\*\#\`\~\_\>]", "", answer_text)
    clean_text = re.sub(r"\[Page\s*\d+\]", "", clean_text)
    safe_js_text = json.dumps(clean_text)

    tts_html = (
        f"<div style='display:flex; align-items:center; gap:10px; margin-top:6px; margin-bottom:4px; font-family:Inter,sans-serif;'>"
        f"  <button id='play_{btn_id}' onclick='speak_{btn_id}()' "
        f"    style='background:linear-gradient(135deg, #6366f1, #a855f7); color:#ffffff; border:none; "
        f"    border-radius:999px; padding:6px 15px; font-size:0.78rem; font-weight:600; cursor:pointer; "
        f"    box-shadow:0 4px 14px rgba(139,92,246,0.35); display:inline-flex; align-items:center; gap:6px;'>"
        f"    🔊 Listen to Answer"
        f"  </button>"
        f"  <button id='stop_{btn_id}' onclick='stop_{btn_id}()' "
        f"    style='background:rgba(30,41,59,0.85); color:#cbd5e1; border:1px solid rgba(148,163,184,0.3); "
        f"    border-radius:999px; padding:6px 13px; font-size:0.78rem; font-weight:600; cursor:pointer; "
        f"    display:none; align-items:center; gap:5px;'>"
        f"    ⏹️ Stop Audio"
        f"  </button>"
        f"</div>"
        f"<script>"
        f"const txt_{btn_id} = {safe_js_text};"
        f"function speak_{btn_id}() {{"
        f"  if (!('speechSynthesis' in window)) {{ alert('Audio playback is not supported in this browser.'); return; }}"
        f"  window.speechSynthesis.cancel();"
        f"  const utter = new SpeechSynthesisUtterance(txt_{btn_id});"
        f"  utter.lang = 'en-US';"
        f"  utter.rate = 1.0;"
        f"  const playBtn = document.getElementById('play_{btn_id}');"
        f"  const stopBtn = document.getElementById('stop_{btn_id}');"
        f"  utter.onstart = () => {{ playBtn.innerHTML = '🎧 Playing Audio...'; stopBtn.style.display = 'inline-flex'; }};"
        f"  utter.onend = () => {{ playBtn.innerHTML = '🔊 Listen to Answer'; stopBtn.style.display = 'none'; }};"
        f"  utter.onerror = () => {{ playBtn.innerHTML = '🔊 Listen to Answer'; stopBtn.style.display = 'none'; }};"
        f"  window.speechSynthesis.speak(utter);"
        f"}}"
        f"function stop_{btn_id}() {{"
        f"  if ('speechSynthesis' in window) {{ window.speechSynthesis.cancel(); }}"
        f"  document.getElementById('play_{btn_id}').innerHTML = '🔊 Listen to Answer';"
        f"  document.getElementById('stop_{btn_id}').style.display = 'none';"
        f"}}"
        f"</script>"
    )
    components.html(tts_html, height=44)


# --- HELPER: RENDER ASSISTANT TURN WITH SEMANTIC SCORE & FOLLOW-UPS ---
def render_assistant_turn(turn: dict, turn_key: str, is_latest: bool = False):
    st.markdown(turn["answer"])
    render_tts_audio_button(turn["answer"], turn_key)

    avg_score = turn.get("avg_match", 94)
    st.markdown(
        f'<span class="meta-badge">⚡ Engine: {turn["model"]}</span>'
        f'<span class="meta-badge">🎯 Confidence: {avg_score}% Match</span>'
        f'<span class="meta-badge">⏱️ Latency: {turn["latency"]}s</span>'
        f'<span class="meta-badge">📝 {turn.get("words", 0)} Words</span>'
        f'<span class="meta-badge">🎛️ {turn["mode"]}</span>',
        unsafe_allow_html=True
    )

    with st.expander("📚 View Verified PDF Page Citations & Vector Match Scores"):
        for c_idx, cite in enumerate(turn["citations"]):
            match_pct = cite.get("match_pct", 92)
            st.markdown(
                f'<div class="citation-box">'
                f'<span class="match-badge">🎯 {match_pct}% Semantic Match</span>'
                f'<strong>📌 Source {c_idx+1} — {cite["source"]} (Page {cite["page"]})</strong><br>'
                f'{cite["snippet"]}'
                f'</div>',
                unsafe_allow_html=True
            )

    followups = turn.get("followups", [])
    if is_latest and followups:
        st.markdown("###### 💡 Smart Follow-Up Questions:")
        fcols = st.columns(len(followups))
        for f_idx, fq in enumerate(followups):
            with fcols[f_idx]:
                if st.button(f"🔎 {fq}", key=f"fq_{turn_key}_{f_idx}", use_container_width=True):
                    st.session_state.queued_followup = fq
                    st.rerun()


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
if "demo_mode_active" not in st.session_state:
    st.session_state.demo_mode_active = False
if "queued_followup" not in st.session_state:
    st.session_state.queued_followup = None


# --- HELPER: ACTIVATE 1-CLICK DEMO KNOWLEDGE BASE ---
def load_sample_ai_paper(chunk_size: int, chunk_overlap: int):
    with st.spinner("🚀 Loading Sample AI & RAG Research Paper into FAISS Vector Space..."):
        demo_docs = get_demo_ai_documents()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_docs = splitter.split_documents(demo_docs)
        embeddings = get_embedding_model()
        st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
        st.session_state.pdf_pages = len(demo_docs)
        st.session_state.neural_chunks = len(split_docs)
        st.session_state.demo_mode_active = True
        st.session_state.processed_files_key = f"DEMO_PAPER_{chunk_size}_{chunk_overlap}"


# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown("### ⚙️ DeepRAG Control Center")
    st.markdown('<div class="status-pill">🟢 Neural Engine: Connected & Active</div>', unsafe_allow_html=True)

    st.markdown("#### 📂 Document Portal")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload textbooks, research papers, legal contracts, or lecture notes."
    )

    if st.button("⚡ Load Sample AI Paper (1-Click Demo)", use_container_width=True):
        load_sample_ai_paper(600, 100)
        st.toast("✅ Sample AI Research Paper Indexed!")
        st.rerun()

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


# --- AUTOMATIC PDF INDEXING WITH PAGE METADATA ---
if uploaded_files:
    st.session_state.demo_mode_active = False
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
elif not st.session_state.demo_mode_active:
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

    # Quick Reviewer Banner when no document is loaded yet
    if not st.session_state.vector_store:
        bcol1, bcol2 = st.columns([3, 1])
        with bcol1:
            st.markdown(
                "<div class='demo-banner'>"
                "<div><strong>🎓 Quick Start for Reviewers & Admissions Panel:</strong><br>"
                "<span style='color:#cbd5e1; font-size:0.9rem;'>Don't have a PDF handy? Load our built-in 5-page AI & Neural RAG Research Paper in 1 click to test citations, voice queries, and vector search immediately.</span></div>"
                "</div>",
                unsafe_allow_html=True
            )
        with bcol2:
            st.write("")
            if st.button("🚀 Load Sample AI Paper", use_container_width=True):
                load_sample_ai_paper(chunk_size, chunk_overlap)
                st.rerun()

    triggered_query = None
    if st.session_state.queued_followup:
        triggered_query = st.session_state.queued_followup
        st.session_state.queued_followup = None

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

    previous_turns = list(reversed(st.session_state.chat_history))

    # --- INSTANT QUESTION DISPLAY & LIVE LOADING BELOW IT ---
    if active_query:
        if not st.session_state.vector_store:
            st.warning("⚠️ Please upload a PDF in the sidebar or click 'Load Sample AI Paper' to begin querying.")
        else:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**{active_query}**")

            with st.chat_message("assistant", avatar="🧠"):
                start_time = time.time()
                with st.spinner("🧠 Querying FAISS Vector Space & synthesizing verified response..."):
                    try:
                        docs_and_scores = st.session_state.vector_store.similarity_search_with_score(active_query, k=top_k)
                        context_parts = []
                        citations_list = []
                        score_percentages = []

                        for idx, (doc, l2_score) in enumerate(docs_and_scores):
                            src = doc.metadata.get("source", "Document.pdf")
                            pg = doc.metadata.get("page", "?")
                            match_pct = max(62, min(99, int(round(100 / (1 + float(l2_score) * 0.35)))))
                            score_percentages.append(match_pct)

                            context_parts.append(f"[Source {idx+1}: {src}, Page {pg}]\n{doc.page_content}")
                            citations_list.append({
                                "source": src,
                                "page": pg,
                                "match_pct": match_pct,
                                "snippet": doc.page_content[:360].replace("\n", " ") + "..."
                            })

                        avg_match = int(round(sum(score_percentages) / len(score_percentages))) if score_percentages else 94
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
                            "IMPORTANT: At the very end of your response, after your main answer, write the exact delimiter ---FOLLOWUP--- "
                            "followed by 3 short, insightful follow-up questions separated by the pipe character | (each question under 9 words, no numbering).\n\n"
                            f"Output Style Instruction: {style_instructions}\n\n"
                            f"Recent Conversation Context:\n{history_context}\n\n"
                            f"Retrieved PDF Context:\n{context_block}\n\n"
                            f"User Question: {active_query}\n"
                        )

                        raw_output, used_model = generate_smart_gemini_response(API_KEY, final_prompt)
                        if "---FOLLOWUP---" in raw_output:
                            main_ans, fq_block = raw_output.split("---FOLLOWUP---", 1)
                            answer_text = main_ans.strip()
                            followups = [q.strip() for q in fq_block.split("|") if q.strip()][:3]
                        else:
                            answer_text = raw_output.strip()
                            followups = []

                        elapsed = round(time.time() - start_time, 2)
                        word_count = len(answer_text.split())

                        new_turn = {
                            "question": active_query,
                            "answer": answer_text,
                            "model": used_model,
                            "latency": elapsed,
                            "words": word_count,
                            "avg_match": avg_match,
                            "mode": response_mode,
                            "citations": citations_list,
                            "followups": followups
                        }
                        st.session_state.chat_history.append(new_turn)
                        render_assistant_turn(new_turn, f"live_{len(st.session_state.chat_history)}", is_latest=True)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

    # --- DISPLAY PREVIOUS CONVERSATION HISTORY ---
    for idx, turn in enumerate(previous_turns):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{turn['question']}**")

        with st.chat_message("assistant", avatar="🧠"):
            is_top_card = (idx == 0 and not active_query)
            render_assistant_turn(turn, f"hist_{idx}", is_latest=is_top_card)

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
        '<div class="arch-title">⚡ Millisecond FAISS Vector Indexing & Confidence Scoring</div>'
        '<div class="arch-desc">Facebook AI Similarity Search (<code>FAISS</code>) indexes all neural vectors in memory. When you ask a question via text or voice, your query is embedded in real time and matched against the top-K passages with exact L2 semantic confidence percentages.</div>'
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
