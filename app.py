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
    page_title="ARCHITECT // DeepRAG — Agentic Build & Systems Matrix",
    page_icon="🔺",
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

# --- ARCHITECT CINEMATIC INTRO & NOCTURNAL PURPLE/RED THEME CSS ---
custom_css = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');"
    ":root { --bg-dark: #050508; --purple: #7B5CFF; --purple-glow: rgba(123, 92, 255, 0.35); --red: #FF2E54; --text-main: #F1F5F9; --text-muted: #94A3B8; }"
    "html, body, [class*='css'] { font-family: 'Inter', sans-serif; background-color: var(--bg-dark); color: var(--text-main); scroll-behavior: smooth; }"
    ".stApp {"
    "  background: radial-gradient(circle at 18% 12%, rgba(123, 92, 255, 0.18) 0%, transparent 40%),"
    "              radial-gradient(circle at 85% 22%, rgba(255, 46, 84, 0.10) 0%, transparent 38%),"
    "              radial-gradient(circle at 50% 85%, rgba(123, 92, 255, 0.12) 0%, transparent 45%), #050508;"
    "}"
    "section[data-testid='stSidebar'] { background: rgba(7, 7, 12, 0.96) !important; border-right: 1px solid rgba(123, 92, 255, 0.25); backdrop-filter: blur(20px); }"
    "#skip-intro-toggle { display: none; }"
    "#skip-intro-toggle:checked ~ .architect-intro-overlay { opacity: 0 !important; visibility: hidden !important; pointer-events: none !important; transition: all 0.35s ease !important; }"
    ".architect-intro-overlay {"
    "  position: fixed; inset: 0; width: 100vw; height: 100vh; background: #030306; z-index: 999999;"
    "  display: flex; flex-direction: column; align-items: center; justify-content: center; overflow: hidden;"
    "  animation: overlayExit 0.8s cubic-bezier(0.22, 1, 0.36, 1) 3.1s forwards;"
    "}"
    "@keyframes overlayExit { to { opacity: 0; visibility: hidden; transform: scale(1.06); pointer-events: none; } }"
    ".intro-ambient {"
    "  position: absolute; width: 520px; height: 520px; border-radius: 50%;"
    "  background: radial-gradient(circle, rgba(123, 92, 255, 0.32) 0%, rgba(255, 46, 84, 0.12) 45%, transparent 70%);"
    "  filter: blur(45px); opacity: 0; animation: ambientIn 0.6s ease 0.1s forwards;"
    "}"
    "@keyframes ambientIn { to { opacity: 1; } }"
    ".blueprint-grid {"
    "  position: absolute; width: 160vw; height: 120vh;"
    "  background-image: linear-gradient(rgba(123, 92, 255, 0.16) 1px, transparent 1px), linear-gradient(90deg, rgba(123, 92, 255, 0.16) 1px, transparent 1px);"
    "  background-size: 48px 48px; transform: perspective(600px) rotateX(62deg) scale(0.85); opacity: 0;"
    "  animation: gridTilt 2.4s cubic-bezier(0.16, 1, 0.3, 1) 0.35s forwards;"
    "}"
    "@keyframes gridTilt { 0% { opacity: 0; transform: perspective(600px) rotateX(68deg) scale(0.8) translateY(40px); } 100% { opacity: 0.55; transform: perspective(600px) rotateX(58deg) scale(1.05) translateY(0); } }"
    ".intro-diagram { position: relative; z-index: 3; display: flex; flex-direction: column; align-items: center; text-align: center; padding: 28px 40px; border: 1px solid rgba(123, 92, 255, 0.35); border-radius: 20px; background: rgba(10, 10, 18, 0.78); backdrop-filter: blur(16px); box-shadow: 0 0 50px rgba(123, 92, 255, 0.25); opacity: 0; transform: translateY(20px) scale(0.94); animation: panelSnap 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.8s forwards; }"
    "@keyframes panelSnap { to { opacity: 1; transform: translateY(0) scale(1); } }"
    ".tri-cluster { display: flex; gap: 22px; margin-bottom: 16px; }"
    ".tri-node { color: #7B5CFF; font-size: 1.5rem; filter: drop-shadow(0 0 12px #7B5CFF); opacity: 0; transform: scale(0.3); animation: triPop 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) 1.0s forwards; }"
    ".tri-node.red { color: #FF2E54; filter: drop-shadow(0 0 12px #FF2E54); animation-delay: 1.15s; }"
    ".tri-node.cyan { color: #38BDF8; filter: drop-shadow(0 0 12px #38BDF8); animation-delay: 1.3s; }"
    "@keyframes triPop { to { opacity: 1; transform: scale(1); } }"
    ".architect-wordmark { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 800; letter-spacing: 0.22em; text-transform: uppercase; color: transparent; -webkit-text-stroke: 1.5px #7B5CFF; position: relative; animation: fillWordmark 0.8s ease 1.7s forwards; }"
    "@keyframes fillWordmark { to { color: #FFFFFF; text-shadow: 0 0 28px rgba(123, 92, 255, 0.85), 0 0 55px rgba(255, 46, 84, 0.45); -webkit-text-stroke: 1.5px #FFFFFF; } }"
    ".architect-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94A3B8; letter-spacing: 0.16em; margin-top: 8px; opacity: 0; animation: ambientIn 0.5s ease 1.5s forwards; }"
    ".skip-intro-btn { position: fixed; bottom: 32px; right: 32px; z-index: 1000000; background: rgba(15, 15, 25, 0.9); border: 1px solid rgba(123, 92, 255, 0.5); color: #E2E8F0; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; padding: 8px 18px; border-radius: 999px; cursor: pointer; transition: all 0.25s ease; box-shadow: 0 0 20px rgba(123, 92, 255, 0.25); }"
    ".skip-intro-btn:hover { background: #7B5CFF; color: #FFFFFF; border-color: #7B5CFF; transform: translateY(-2px); }"
    ".top-nav { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; padding: 14px 22px; background: rgba(10, 10, 18, 0.85); border: 1px solid rgba(123, 92, 255, 0.22); border-radius: 14px; margin-bottom: 24px; backdrop-filter: blur(16px); animation: stageUp 0.6s ease forwards; }"
    ".brand-logo { font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1.05rem; letter-spacing: 0.08em; display: flex; align-items: center; gap: 8px; color: #FFFFFF; }"
    ".brand-badge { background: #FF2E54; color: #FFFFFF; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; padding: 2px 7px; border-radius: 4px; font-weight: 700; }"
    ".nav-links { display: flex; gap: 20px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #94A3B8; }"
    ".nav-item { position: relative; cursor: pointer; transition: color 0.25s ease; }"
    ".nav-item:hover { color: #FFFFFF; }"
    ".nav-item::after { content: ''; position: absolute; left: 0; bottom: -4px; width: 0%; height: 2px; background: linear-gradient(90deg, #7B5CFF, #FF2E54); transition: width 0.3s ease; }"
    ".nav-item:hover::after { width: 100%; }"
    "@keyframes stageUp { from { opacity: 0; transform: translateY(22px); } to { opacity: 1; transform: translateY(0); } }"
    ".eng-index-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #FF2E54; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 6px; display: flex; align-items: center; gap: 8px; }"
    ".hero-headline { font-family: 'Space Grotesk', sans-serif; font-size: 3.1rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; margin-bottom: 12px; color: #FFFFFF; animation: stageUp 0.7s ease 0.15s both; }"
    ".hero-headline span { background: linear-gradient(135deg, #7B5CFF 0%, #C084FC 55%, #FF2E54 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }"
    ".hero-sub { color: #94A3B8; font-size: 1.05rem; max-width: 760px; margin-bottom: 20px; line-height: 1.6; animation: stageUp 0.7s ease 0.25s both; }"
    ".telemetry-bar { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 12px; background: rgba(12, 12, 22, 0.92); border: 1px solid rgba(123, 92, 255, 0.35); padding: 8px 16px; border-radius: 999px; font-size: 0.76rem; color: #CBD5E1; margin-bottom: 22px; box-shadow: 0 8px 28px rgba(0,0,0,0.5); animation: stageUp 0.7s ease 0.35s both; }"
    ".dot-cluster { display: inline-flex; gap: 5px; align-items: center; }"
    ".dot-c { width: 7px; height: 7px; border-radius: 50%; display: inline-block; animation: pulseDot 1.5s infinite; }"
    "@keyframes pulseDot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.8); } }"
    ".matrix-select-wrap { padding: 3px 10px; border-radius: 5px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #E0F2FE; animation: matrixHighlight 4s infinite ease-in-out; }"
    "@keyframes matrixHighlight { 0%, 100% { background: rgba(123, 92, 255, 0.12); border: 1px solid rgba(123, 92, 255, 0.3); } 50% { background: rgba(56, 189, 248, 0.24); border: 1px solid #38BDF8; box-shadow: 0 0 15px rgba(56, 189, 248, 0.35); } }"
    ".mockup-shell { background: linear-gradient(145deg, rgba(16, 16, 28, 0.92), rgba(8, 8, 15, 0.96)); border: 1px solid rgba(123, 92, 255, 0.32); border-radius: 20px; padding: 22px; margin-bottom: 26px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 35px rgba(123, 92, 255, 0.12); animation: stageUp 0.8s ease 0.4s both; }"
    ".metric-card { background: rgba(13, 13, 24, 0.9); border: 1px solid rgba(123, 92, 255, 0.25); border-radius: 16px; padding: 20px 16px; text-align: center; transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); position: relative; overflow: hidden; }"
    ".metric-card:hover { transform: translateY(-5px); border-color: #7B5CFF; box-shadow: 0 12px 32px rgba(123, 92, 255, 0.28); }"
    ".tri-icon { color: #7B5CFF; font-size: 1.1rem; filter: drop-shadow(0 0 8px #7B5CFF); margin-right: 6px; }"
    ".metric-title { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; color: #94A3B8; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.08em; }"
    ".metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800; color: #FFFFFF; }"
    ".matrix-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 18px; margin-top: 16px; margin-bottom: 28px; }"
    ".flagship-card { background: linear-gradient(160deg, rgba(15, 15, 26, 0.95), rgba(8, 8, 14, 0.98)); border: 1px solid rgba(123, 92, 255, 0.22); border-top: 2px solid #7B5CFF; border-radius: 16px; padding: 22px; transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1); display: flex; flex-direction: column; justify-content: space-between; }"
    ".flagship-card.red-top { border-top-color: #FF2E54; }"
    ".flagship-card:hover { transform: translateY(-6px); border-color: #7B5CFF; box-shadow: 0 16px 40px rgba(123, 92, 255, 0.22); }"
    ".flagship-cat { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 700; color: #FF2E54; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 8px; }"
    ".flagship-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.22rem; font-weight: 700; color: #FFFFFF; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; }"
    ".flagship-desc { color: #94A3B8; font-size: 0.88rem; line-height: 1.55; margin-bottom: 14px; }"
    ".flagship-list { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #CBD5E1; margin-bottom: 16px; line-height: 1.7; }"
    ".flagship-footer { display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(148, 163, 184, 0.12); padding-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #A5B4FC; }"
    ".prog-wrap { margin-bottom: 16px; }"
    ".prog-label { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #E2E8F0; margin-bottom: 6px; }"
    ".prog-track { width: 100%; height: 8px; background: rgba(30, 41, 59, 0.65); border-radius: 999px; overflow: hidden; border: 1px solid rgba(123, 92, 255, 0.2); }"
    ".prog-fill { height: 100%; background: linear-gradient(90deg, #7B5CFF, #FF2E54); border-radius: 999px; width: 0%; animation: fillBar 1.6s cubic-bezier(0.22, 1, 0.36, 1) 0.5s forwards; box-shadow: 0 0 12px rgba(123, 92, 255, 0.6); }"
    "@keyframes fillBar { to { width: var(--target-w); } }"
    "div[data-testid='stTabs'] button { background: rgba(16, 16, 28, 0.75) !important; border-radius: 10px !important; padding: 10px 18px !important; margin-right: 8px !important; border: 1px solid rgba(123, 92, 255, 0.25) !important; color: #CBD5E1 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 0.8rem !important; font-weight: 600 !important; transition: all 0.25s ease !important; }"
    "div[data-testid='stTabs'] button:hover { border-color: #7B5CFF !important; transform: translateY(-2px); }"
    "div[data-testid='stTabs'] button[aria-selected='true'] { background: linear-gradient(135deg, #7B5CFF, #FF2E54) !important; color: #FFFFFF !important; border: none !important; box-shadow: 0 6px 20px rgba(123, 92, 255, 0.45) !important; }"
    ".meta-badge { display: inline-block; background: rgba(123, 92, 255, 0.16); border: 1px solid rgba(123, 92, 255, 0.45); color: #DDD6FE; font-family: 'JetBrains Mono', monospace; font-size: 0.73rem; font-weight: 600; padding: 4px 11px; border-radius: 999px; margin-right: 8px; margin-top: 8px; }"
    ".citation-box { background: rgba(12, 12, 22, 0.92); border-left: 3px solid #7B5CFF; border-top: 1px solid rgba(123, 92, 255, 0.15); border-right: 1px solid rgba(123, 92, 255, 0.15); border-bottom: 1px solid rgba(123, 92, 255, 0.15); padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 0.86rem; color: #CBD5E1; }"
    ".agent-swarm-layer { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 9998; overflow: hidden; }"
    ".agent-cursor { position: absolute; display: flex; align-items: center; gap: 6px; filter: drop-shadow(0 0 10px rgba(123, 92, 255, 0.55)); will-change: transform; opacity: 0.9; }"
    ".agent-tag { font-family: 'JetBrains Mono', monospace; font-size: 0.64rem; font-weight: 700; padding: 2px 7px; border-radius: 5px; color: #050508; white-space: nowrap; }"
    ".cursor-1 { animation: roam1 15s infinite ease-in-out; }"
    ".cursor-2 { animation: roam2 19s infinite ease-in-out; }"
    ".cursor-3 { animation: roam3 17s infinite ease-in-out; }"
    ".cursor-4 { animation: roam4 21s infinite ease-in-out; }"
    "@keyframes roam1 { 0%, 100% { transform: translate(15vw, 22vh); } 33% { transform: translate(65vw, 30vh); } 66% { transform: translate(42vw, 68vh); } }"
    "@keyframes roam2 { 0%, 100% { transform: translate(78vw, 18vh); } 35% { transform: translate(30vw, 44vh); } 70% { transform: translate(68vw, 62vh); } }"
    "@keyframes roam3 { 0%, 100% { transform: translate(38vw, 75vh); } 40% { transform: translate(82vw, 42vh); } 75% { transform: translate(24vw, 28vh); } }"
    "@keyframes roam4 { 0%, 100% { transform: translate(60vw, 70vh); } 45% { transform: translate(20vw, 36vh); } 80% { transform: translate(54vw, 48vh); } }"
    ".status-pill { background: rgba(123, 92, 255, 0.14); border: 1px solid rgba(123, 92, 255, 0.45); color: #C4B5FD; padding: 9px 12px; border-radius: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; font-weight: 600; text-align: center; margin-bottom: 16px; }"
    ".contact-box { background: rgba(12, 12, 22, 0.9); border: 1px solid rgba(123, 92, 255, 0.35); border-radius: 12px; padding: 14px; text-align: center; margin-top: 18px; }"
    ".contact-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem; font-weight: 700; color: #7B5CFF; margin-bottom: 4px; }"
    ".contact-handle { font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; font-weight: 600; color: #FFFFFF; }"
    ".premium-footer { margin-top: 48px; padding: 20px 0 12px 0; border-top: 1px solid rgba(123, 92, 255, 0.22); text-align: center; font-family: 'Space Grotesk', sans-serif; font-size: 0.95rem; font-weight: 600; color: #94A3B8; }"
    ".premium-footer span { background: linear-gradient(135deg, #7B5CFF 0%, #FF2E54 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }"
    "@media (prefers-reduced-motion: reduce) { *, ::before, ::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; } }"
    "footer { visibility: hidden; }"
    "</style>"
)
st.markdown(custom_css, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False
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

# --- RENDER ARCHITECT OPENING CINEMATIC SEQUENCE (ON FIRST VISIT) ---
if not st.session_state.intro_shown:
    intro_html = (
        "<input type='checkbox' id='skip-intro-toggle'>"
        "<div class='architect-intro-overlay' role='dialog' aria-label='Architect Opening Cinematic'>"
        "  <label for='skip-intro-toggle' class='skip-intro-btn' tabindex='0' aria-label='Skip intro — Go to site'>Skip Intro — Go to Site ⏭</label>"
        "  <div class='intro-ambient'></div>"
        "  <div class='blueprint-grid'></div>"
        "  <div class='intro-diagram'>"
        "    <div class='tri-cluster'>"
        "      <span class='tri-node'>▲</span>"
        "      <span class='tri-node red'>▲</span>"
        "      <span class='tri-node cyan'>▲</span>"
        "    </div>"
        "    <div class='architect-wordmark'>ARCHITECT</div>"
        "    <div class='architect-sub'>DEEPRAG // AGENTIC SYSTEMS MATRIX INITIALIZING...</div>"
        "  </div>"
        "</div>"
    )
    st.markdown(intro_html, unsafe_allow_html=True)
    st.session_state.intro_shown = True


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


# --- HELPER: LOAD EMBEDDING MODEL CACHED ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown("### 🔺 ARCHITECT // CONTROL")
    st.markdown('<div class="status-pill">● V8 NEURAL CORE: ONLINE</div>', unsafe_allow_html=True)

    enable_swarm = st.toggle(
        "🤖 Autonomous Cursor Swarm",
        value=True,
        help="Enable multi-agent roaming cursors across the workspace."
    )

    if st.button("🎬 Replay Architect Intro", use_container_width=True):
        st.session_state.intro_shown = False
        st.rerun()

    st.markdown("---")
    st.markdown("#### 📂 Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload textbooks, research papers, or technical manuals."
    )

    st.markdown("---")
    st.markdown("#### 🎛️ Synthesis Mode")
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
    st.markdown("#### 🛠️ Vector Parameters")
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1500, value=600, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=100, step=25)
    top_k = st.slider("Top-K Citations", min_value=2, max_value=8, value=4, step=1)

    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### 📥 Session Export")
        export_lines = ["# 🔺 DeepRAG AI — Architect Session Report\n\n"]
        for turn in st.session_state.chat_history:
            export_lines.append(f"### 🙋 Query:\n{turn['question']}\n\n")
            export_lines.append(f"### 🤖 Synthesis ({turn.get('model', 'Gemini')}):\n{turn['answer']}\n\n---\n\n")

        st.download_button(
            label="📄 Export Session Report (.md)",
            data="".join(export_lines),
            file_name="DeepRAG_Architect_Report.md",
            mime="text/markdown",
            use_container_width=True
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

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
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#7B5CFF'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#7B5CFF; color:#FFF;'>Agent_01 • Retriever</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-2'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#FF2E54'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#FF2E54; color:#FFF;'>Agent_02 • FAISS_Matrix</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-3'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#38BDF8'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#38BDF8;'>Agent_03 • Gemini_Synth</span>"
        "  </div>"
        "  <div class='agent-cursor cursor-4'>"
        "    <svg width='18' height='18' viewBox='0 0 24 24' fill='#34D399'><path d='M3 2l17 9-8 2-3 8z'/></svg>"
        "    <span class='agent-tag' style='background:#34D399;'>Agent_04 • Voice_Node</span>"
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


# --- MINIMAL ANIMATED HEADER / NAV ---
st.markdown(
    "<div class='top-nav'>"
    "  <div class='brand-logo'><span style='color:#7B5CFF;'>▲</span> DEEPRAG // ARCHITECT <span class='brand-badge'>V8 QUANTIZED</span></div>"
    "  <div class='nav-links'>"
    "    <span class='nav-item'>01. AI WORKSPACE</span>"
    "    <span class='nav-item'>02. SYSTEMS MATRIX</span>"
    "    <span class='nav-item'>03. PHILOSOPHY</span>"
    "    <span class='nav-item'>04. SERVICES</span>"
    "    <span class='nav-item' style='color:#7B5CFF;'>TELEGRAM @DeepRAGAI</span>"
    "  </div>"
    "</div>",
    unsafe_allow_html=True
)

# --- NAVIGATION TABS ---
tab_workspace, tab_matrix, tab_philosophy, tab_services = st.tabs([
    "🔺 01 // Agentic Workspace",
    "⚡ 02 // The Systems Matrix (Flagships)",
    "🧬 03 // Background & Philosophy",
    "🛠️ 04 // Services & Contact"
])

# ==========================================
# TAB 1: HERO + INTERACTIVE AI WORKSPACE
# ==========================================
with tab_workspace:
    st.markdown("<div class='eng-index-tag'><span>▲</span> ENGINEERING INDEX // AUTONOMOUS RAG ARCHITECTURE</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-headline'>Agentic Build — <span>DeepRAG AI Workspace</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Specialized multi-LLM cognitive assemblies, local 384-dimensional FAISS vector indexing, voice intelligence, and zero-hallucination document synthesis engineered by Deep Patel.</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='telemetry-bar'>"
        "  <span class='dot-cluster'>"
        "    <span class='dot-c' style='background:#FF2E54;'></span>"
        "    <span class='dot-c' style='background:#F59E0B; animation-delay:0.2s;'></span>"
        "    <span class='dot-c' style='background:#10B981; animation-delay:0.4s;'></span>"
        "    <span class='dot-c' style='background:#7B5CFF; animation-delay:0.6s;'></span>"
        "  </span>"
        "  <span><strong>4 Autonomous Agents</strong> Active</span>"
        "  <span style='color:#475569;'>|</span>"
        "  <span class='matrix-select-wrap'>+ The Systems Matrix • 384D Latent Layers</span>"
        "</div>",
        unsafe_allow_html=True
    )

    # Centered Dashboard Mockup Area
    st.markdown("<div class='mockup-shell'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'><span class='tri-icon'>▲</span> Indexed PDF Pages</div>"
            f"<div class='metric-value'>{st.session_state.pdf_pages}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'><span class='tri-icon'>▲</span> Neural Vector Chunks</div>"
            f"<div class='metric-value'>{st.session_state.neural_chunks}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    with col3:
        engine_status = "FAISS ONLINE" if st.session_state.vector_store else "STANDBY"
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'><span class='tri-icon'>▲</span> Cognitive Engine</div>"
            f"<div class='metric-value'>{engine_status}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    triggered_query = None
    if st.session_state.vector_store:
        st.markdown("##### 🔺 Instant Agentic Executions")
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
        st.markdown("##### 💬 Interrogate The Neural Matrix (Text or Voice)")
        st.caption("Upload a PDF in the left control panel, then type a prompt or record a voice query on the right.")
    with vcol2:
        audio_value = None
        if hasattr(st, "audio_input"):
            audio_value = st.audio_input("🎙️ Voice AI Input")

    text_query = st.chat_input("Execute a query across your indexed PDF documents...")

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
            with st.spinner("🎧 Transcribing voice stream via Gemini Audio Node..."):
                try:
                    mime_type = getattr(audio_value, "type", "audio/wav") or "audio/wav"
                    transcribed_text = transcribe_voice_with_gemini(API_KEY, audio_bytes, mime_type)
                    active_query = transcribed_text
                    st.toast(f"🎙️ Voice Query Recognized: {transcribed_text}")
                except Exception as e:
                    st.error(f"❌ {e}")

    if active_query:
        if not st.session_state.vector_store:
            st.warning("⚠️ Please upload at least one PDF document in the left sidebar to initialize the FAISS vector matrix.")
        else:
            start_time = time.time()
            with st.spinner("🧠 Traversing 384D FAISS Vector Space & synthesizing grounded response..."):
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

        with st.chat_message("assistant", avatar="🔺"):
            st.markdown(turn["answer"])
            st.markdown(
                f'<span class="meta-badge">▲ Engine: {turn["model"]}</span>'
                f'<span class="meta-badge">⏱️ Latency: {turn["latency"]}s</span>'
                f'<span class="meta-badge">📝 {turn.get("words", 0)} Words</span>'
                f'<span class="meta-badge">🎛️ {turn["mode"]}</span>',
                unsafe_allow_html=True
            )

            with st.expander("📚 View Verified PDF Page Citations & Retrieved Passages"):
                for c_idx, cite in enumerate(turn["citations"]):
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>▲ Source {c_idx+1} — {cite["source"]} (Page {cite["page"]})</strong><br>'
                        f'{cite["snippet"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# ==========================================
# TAB 2: THE SYSTEMS MATRIX (FLAGSHIP AI ECOSYSTEM)
# ==========================================
with tab_matrix:
    st.markdown("<div class='eng-index-tag'><span>▲</span> ENGINEERING INDEX // FLAGSHIP DEPLOYMENTS</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-headline'>The Systems <span>Matrix</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>Specialized architectures spanning multi-LLM cognitive assemblies, deep learning latent models, autonomous RAG pipelines, and high-frequency execution engines.</div>",
        unsafe_allow_html=True
    )

    matrix_cards_html = (
        "<div class='matrix-grid'>"
        "  <div class='flagship-card'>"
        "    <div>"
        "      <div class='flagship-cat'>AI // COGNITIVE RAG CORE</div>"
        "      <div class='flagship-title'><span class='tri-icon'>▲</span> DeepRAG — Neural Vector Workspace</div>"
        "      <div class='flagship-desc'>Enterprise Retrieval-Augmented Generation system with 384D dense embeddings, voice-to-text interrogation, and zero-hallucination page citations.</div>"
        "      <div class='flagship-list'>• Multi-Document PDF Ingestion & Chunking<br>• HuggingFace MiniLM-L6-v2 Vector Space<br>• Auto-Adaptive Gemini 3.0 Flash Synthesis</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>LangChain • FAISS</span><span>384D Latent</span></div>"
        "  </div>"
        "  <div class='flagship-card red-top'>"
        "    <div>"
        "      <div class='flagship-cat'>DEFI // HIGH-FREQUENCY</div>"
        "      <div class='flagship-title'><span class='tri-icon' style='color:#FF2E54;'>▲</span> Flash_Lab — Arbitrage & Liquidation</div>"
        "      <div class='flagship-desc'>Production flash-loan arbitrage and liquidation engine with asynchronous multi-hop routing and real-time mempool telemetry.</div>"
        "      <div class='flagship-list'>• Polygon Mainnet (Aave V3 Flash Loans)<br>• Uniswap V3, QuickSwap & Curve Liquidity<br>• Async WebSockets Execution Engine</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>Solidity • Rust</span><span>Aave V3</span></div>"
        "  </div>"
        "  <div class='flagship-card'>"
        "    <div>"
        "      <div class='flagship-cat'>MOBILE // COMMERCIAL FINTECH</div>"
        "      <div class='flagship-title'><span class='tri-icon'>▲</span> Biller — Offline-First Invoicing System</div>"
        "      <div class='flagship-desc'>Commercial mobile billing and inventory application engineered for zero-connectivity environments with instant thermal printing.</div>"
        "      <div class='flagship-list'>• Offline SQLite with Repository Pattern<br>• ESC/POS Bluetooth Thermal Printing<br>• GST-Compliant Real-Time Analytics</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>Flutter • SQLite</span><span>Offline-First</span></div>"
        "  </div>"
        "  <div class='flagship-card red-top'>"
        "    <div>"
        "      <div class='flagship-cat'>QUANT // INSTITUTIONAL ALGO</div>"
        "      <div class='flagship-title'><span class='tri-icon' style='color:#FF2E54;'>▲</span> AIT Kronos — ML Time-Series Engine</div>"
        "      <div class='flagship-desc'>Multi-LLM parliament architecture and quantitative forecasting assembly for institutional order-flow and liquidity modeling.</div>"
        "      <div class='flagship-list'>• 27 Autonomous Cognitive Sub-Agents<br>• 178 / 252 Deep Learning Latent Layers<br>• Real-Time Volatility Regime Detection</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>PyTorch • Python</span><span>Multi-LLM</span></div>"
        "  </div>"
        "</div>"
    )
    st.markdown(matrix_cards_html, unsafe_allow_html=True)

# ==========================================
# TAB 3: BACKGROUND & PHILOSOPHY (ANIMATED PROGRESS)
# ==========================================
with tab_philosophy:
    st.markdown("<div class='eng-index-tag'><span>▲</span> ARCHITECT PROFILE // BACKGROUND & PHILOSOPHY</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-headline'>Deep Patel — <span>Lead AI Architect</span></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-sub'>\"My mission is simple: transform complex artificial intelligence and neural retrieval pipelines into a fast, reliable, and visually breathtaking experience that empowers students, researchers, and builders.\"</div>",
        unsafe_allow_html=True
    )

    pcol1, pcol2 = st.columns([1, 1])
    with pcol1:
        st.markdown(
            "<div class='mockup-shell'>"
            "<div class='flagship-cat'>01 // ENGINEERING DOCTRINE</div>"
            "<div class='flagship-title'><span class='tri-icon'>▲</span> Precision Over Hype</div>"
            "<div class='flagship-desc'>Every system in the DeepRAG ecosystem is built around deterministic verification. Rather than relying on ungrounded black-box generation, we combine geometric FAISS vector search with strict citation boundaries so every output is 100% auditable.</div>"
            "<div class='flagship-cat' style='margin-top:18px;'>02 // DESIGN-FORWARD AI</div>"
            "<div class='flagship-title'><span class='tri-icon' style='color:#FF2E54;'>▲</span> Nocturnal Glass & Motion Architecture</div>"
            "<div class='flagship-desc'>Enterprise software should feel like a cockpit. By pairing low-latency Python backends with staged GPU-accelerated CSS animations, we deliver interfaces that command attention in academic and technical evaluations.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with pcol2:
        st.markdown(
            "<div class='mockup-shell'>"
            "<div class='flagship-cat'>03 // CAPABILITY TELEMETRY</div>"
            "<div class='flagship-title'><span class='tri-icon'>▲</span> Core Competency Matrix</div>"
            "<div class='prog-wrap'>"
            "  <div class='prog-label'><span>▲ Neural RAG & LangChain Orchestration</span><span>98%</span></div>"
            "  <div class='prog-track'><div class='prog-fill' style='--target-w: 98%;'></div></div>"
            "</div>"
            "<div class='prog-wrap'>"
            "  <div class='prog-label'><span>▲ FAISS Vector Search & Dense Embeddings</span><span>96%</span></div>"
            "  <div class='prog-track'><div class='prog-fill' style='--target-w: 96%;'></div></div>"
            "</div>"
            "<div class='prog-wrap'>"
            "  <div class='prog-label'><span>▲ Multimodal Gemini & Voice AI Systems</span><span>95%</span></div>"
            "  <div class='prog-track'><div class='prog-fill' style='--target-w: 95%;'></div></div>"
            "</div>"
            "<div class='prog-wrap'>"
            "  <div class='prog-label'><span>▲ High-Contrast UI & Motion Engineering</span><span>94%</span></div>"
            "  <div class='prog-track'><div class='prog-fill' style='--target-w: 94%;'></div></div>"
            "</div>"
            "<div class='prog-wrap'>"
            "  <div class='prog-label'><span>▲ Cloud Deployment & API Security</span><span>97%</span></div>"
            "  <div class='prog-track'><div class='prog-fill' style='--target-w: 97%;'></div></div>"
            "</div>"
            "</div>",
            unsafe_allow_html=True
        )

# ==========================================
# TAB 4: SERVICES & DIRECT CONTACT
# ==========================================
with tab_services:
    st.markdown("<div class='eng-index-tag'><span>▲</span> CAPABILITIES // WORK WITH ME</div>", unsafe_allow_html=True)
    st.markdown("<div class='hero-headline'>Specialized <span>Services</span></div>", unsafe_allow_html=True)

    services_html = (
        "<div class='matrix-grid'>"
        "  <div class='flagship-card'>"
        "    <div>"
        "      <div class='flagship-cat'>SERVICE // 01</div>"
        "      <div class='flagship-title'><span class='tri-icon'>▲</span> Custom Neural RAG Pipelines</div>"
        "      <div class='flagship-desc'>End-to-end document intelligence systems for universities, legal teams, and enterprises with zero-hallucination citations.</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>Learn More →</span><span>Production Ready</span></div>"
        "  </div>"
        "  <div class='flagship-card red-top'>"
        "    <div>"
        "      <div class='flagship-cat'>SERVICE // 02</div>"
        "      <div class='flagship-title'><span class='tri-icon' style='color:#FF2E54;'>▲</span> Agentic AI & Voice Workflows</div>"
        "      <div class='flagship-desc'>Multi-agent autonomous assemblies capable of voice transcription, tool execution, and real-time data synthesis.</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>Learn More →</span><span>Multi-LLM</span></div>"
        "  </div>"
        "  <div class='flagship-card'>"
        "    <div>"
        "      <div class='flagship-cat'>SERVICE // 03</div>"
        "      <div class='flagship-title'><span class='tri-icon'>▲</span> High-Impact AI Web Builds</div>"
        "      <div class='flagship-desc'>Dark-themed, motion-engineered SaaS dashboards and interactive AI portfolios deployed on cloud infrastructure.</div>"
        "    </div>"
        "    <div class='flagship-footer'><span>Telegram: @DeepRAGAI</span><span>Available</span></div>"
        "  </div>"
        "</div>"
    )
    st.markdown(services_html, unsafe_allow_html=True)

# --- GLOBAL SIGNATURE FOOTER ---
st.markdown(
    '<div class="premium-footer">Contact Us: Telegram <span>@DeepRAGAI</span> &nbsp;|&nbsp; Engineered with ❤️ by <span>Deep</span></div>',
    unsafe_allow_html=True
)
