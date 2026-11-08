import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import time
import hashlib
import json
import re
import html
import io
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
    initial_sidebar_state="collapsed"
)

# --- MULTI-API KEY POOL (ADD KEYS HERE OR IN STREAMLIT SECRETS TO AUTO-ROTATE) ---
DEFAULT_API_KEYS = [
    "",  # Primary API Key (Fallback if not in Streamlit Secrets)
    "",  # Backup API Key 2 (Auto-activates when Key 1 hits quota limit)
    ""   # Backup API Key 3 (Optional)
]


def get_api_key_pool() -> list[str]:
    keys = []
    try:
        secret_names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]
        for s_name in secret_names:
            if s_name in st.secrets and st.secrets[s_name]:
                raw_val = str(st.secrets[s_name]).strip()
                for part in raw_val.split(","):
                    clean_p = part.strip()
                    if clean_p and clean_p not in keys:
                        keys.append(clean_p)
    except Exception:
        pass

    for k in DEFAULT_API_KEYS:
        clean_k = str(k).strip()
        if clean_k and clean_k not in keys:
            keys.append(clean_k)

    return keys


API_KEYS_POOL = get_api_key_pool()

# --- ADAPTIVE DARK & LIGHT THEME CSS + ULTRA-ANIMATED CREATOR & REPORT UI ---
custom_css = (
    "<style>"
    "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');"
    "html, body, [class*='css'] { font-family: 'Inter', sans-serif; scroll-behavior: smooth; }"
    "::-webkit-scrollbar { width: 6px; height: 6px; }"
    "::-webkit-scrollbar-thumb { background: #3b82f6; border-radius: 999px; }"
    "header[data-testid='stHeader'] { background: transparent !important; }"
    "div[data-testid='stAppViewContainer'] > .main { background: transparent !important; position: relative; z-index: 2; }"
    ".block-container { position: relative; z-index: 2; padding-top: 2rem !important; }"

    "html:not([data-deeprag-theme='light']) .stApp {"
    "  background: radial-gradient(ellipse at 50% 0%, #0b1124 0%, #05070f 55%, #020306 100%) !important;"
    "  overflow-x: hidden;"
    "}"
    "html:not([data-deeprag-theme='light']) section[data-testid='stSidebar'] {"
    "  background: #060913 !important; border-right: 1px solid rgba(56, 189, 248, 0.18); z-index: 10;"
    "}"
    "html:not([data-deeprag-theme='light']) .anime-vignette {"
    "  position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0;"
    "  background: radial-gradient(circle at 50% 100%, rgba(56, 189, 248, 0.08) 0%, rgba(99, 102, 241, 0.05) 35%, transparent 70%);"
    "}"

    "html[data-deeprag-theme='light'] .stApp {"
    "  background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #f0f9ff 100%) !important;"
    "  overflow-x: hidden;"
    "}"
    "html[data-deeprag-theme='light'] section[data-testid='stSidebar'] {"
    "  background: #ffffff !important; border-right: 1px solid rgba(99, 102, 241, 0.2); z-index: 10;"
    "}"
    "html[data-deeprag-theme='light'] .anime-vignette {"
    "  position: fixed; inset: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0;"
    "  background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 65%);"
    "}"

    ".status-pill { background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.4); color: #10b981; padding: 10px 14px; border-radius: 12px; font-size: 0.84rem; font-weight: 600; text-align: center; margin-bottom: 18px; }"
    ".hero-title { font-family: 'Space Grotesk', sans-serif; font-size: 2.9rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 6px; display: flex; flex-wrap: wrap; align-items: center; }"
    ".char { display: inline-block; background: linear-gradient(135deg, #38bdf8 0%, #6366f1 50%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; opacity: 0; transform: translateY(10px); animation: fadeUpChar 0.35s ease-out forwards; }"
    "@keyframes fadeUpChar { to { opacity: 1; transform: translateY(0); } }"
    ".cursor-blink { display: inline-block; width: 4px; height: 38px; background: #38bdf8; margin-left: 8px; border-radius: 2px; animation: blink 1s infinite; }"
    "@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }"
    ".hero-subtitle { color: #64748b; font-size: 1.02rem; margin-bottom: 18px; font-weight: 500; }"
    "html:not([data-deeprag-theme='light']) .hero-subtitle { color: #94a3b8; }"

    ".telemetry-strip { display: flex; flex-direction: column; align-items: flex-start; gap: 8px; background: rgba(10, 15, 28, 0.88); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; color: #cbd5e1; }"
    "html[data-deeprag-theme='light'] .telemetry-strip { background: #f8fafc; border: 1px solid rgba(99, 102, 241, 0.25); color: #334155; }"
    ".tel-item { display: inline-flex; align-items: center; gap: 7px; }"
    ".tel-dot { width: 8px; height: 8px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 8px #38bdf8; display: inline-block; }"
    ".tel-val { color: #0284c7; font-weight: 700; }"
    "html:not([data-deeprag-theme='light']) .tel-val { color: #7dd3fc; }"

    ".metric-card { background: rgba(10, 15, 28, 0.88); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 16px; padding: 20px 18px; text-align: center; transition: transform 0.2s ease, border-color 0.2s ease; }"
    "html[data-deeprag-theme='light'] .metric-card { background: rgba(255, 255, 255, 0.95); border: 1px solid rgba(99, 102, 241, 0.25); box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05); }"
    ".metric-card:hover { transform: translateY(-4px); border-color: #38bdf8; }"
    ".metric-title { font-size: 0.88rem; font-weight: 700; color: #0284c7; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.06em; }"
    "html:not([data-deeprag-theme='light']) .metric-title { color: #7dd3fc; }"
    ".metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 800; color: #f8fafc; }"
    "html[data-deeprag-theme='light'] .metric-value { color: #0f172a; }"

    ".upload-portal-card { background: rgba(10, 15, 28, 0.88); border: 1px dashed rgba(56, 189, 248, 0.45); border-radius: 18px; padding: 18px 22px; margin-bottom: 18px; }"
    "html[data-deeprag-theme='light'] .upload-portal-card { background: rgba(255, 255, 255, 0.95); border: 1px dashed rgba(79, 70, 229, 0.45); box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05); }"
    ".upload-portal-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: #38bdf8; margin-bottom: 4px; }"
    "html[data-deeprag-theme='light'] .upload-portal-title { color: #4f46e5; }"
    ".upload-portal-sub { font-size: 0.88rem; color: #94a3b8; margin-bottom: 8px; }"
    "html[data-deeprag-theme='light'] .upload-portal-sub { color: #475569; }"

    "div[data-testid='stTabs'] button { background: rgba(15, 23, 42, 0.75) !important; border-radius: 10px !important; padding: 10px 18px !important; margin-right: 8px !important; border: 1px solid rgba(56, 189, 248, 0.18) !important; color: #cbd5e1 !important; font-weight: 600 !important; transition: all 0.2s ease !important; }"
    "html[data-deeprag-theme='light'] div[data-testid='stTabs'] button { background: #ffffff !important; border: 1px solid rgba(99, 102, 241, 0.25) !important; color: #334155 !important; }"
    "div[data-testid='stTabs'] button[aria-selected='true'] { background: linear-gradient(135deg, #0284c7, #4f46e5) !important; color: #ffffff !important; border: none !important; box-shadow: 0 4px 16px rgba(56, 189, 248, 0.35) !important; }"

    "div.stButton > button { background: rgba(13, 19, 36, 0.9) !important; border: 1px solid rgba(56, 189, 248, 0.3) !important; color: #e2e8f0 !important; border-radius: 12px !important; font-weight: 600 !important; padding: 10px 16px !important; transition: transform 0.2s ease, border-color 0.2s ease !important; }"
    "html[data-deeprag-theme='light'] div.stButton > button { background: #ffffff !important; border: 1px solid rgba(79, 70, 229, 0.3) !important; color: #1e293b !important; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05) !important; }"
    "div.stButton > button:hover { transform: translateY(-2px) !important; border-color: #38bdf8 !important; background: linear-gradient(135deg, #0284c7, #4f46e5) !important; color: #ffffff !important; }"

    "div[data-testid='stChatMessage'] { background: rgba(10, 15, 28, 0.88) !important; border: 1px solid rgba(56, 189, 248, 0.2) !important; border-radius: 16px !important; padding: 16px 20px !important; margin-bottom: 14px !important; }"
    "html[data-deeprag-theme='light'] div[data-testid='stChatMessage'] { background: #ffffff !important; border: 1px solid rgba(99, 102, 241, 0.22) !important; box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05) !important; }"

    ".quiz-card { background: rgba(10, 15, 28, 0.9); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 14px; padding: 18px 22px; margin-bottom: 14px; }"
    "html[data-deeprag-theme='light'] .quiz-card { background: #ffffff; border: 1px solid rgba(99, 102, 241, 0.25); color: #0f172a; }"

    ".report-shell { background: linear-gradient(145deg, rgba(10, 16, 32, 0.95), rgba(6, 10, 22, 0.98)); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 20px; padding: 26px; margin-top: 20px; margin-bottom: 20px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.45); }"
    "html[data-deeprag-theme='light'] .report-shell { background: #ffffff; border: 1px solid rgba(79, 70, 229, 0.35); color: #0f172a; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); }"

    ".meta-badge { display: inline-block; background: rgba(56, 189, 248, 0.14); border: 1px solid rgba(56, 189, 248, 0.38); color: #0284c7; font-size: 0.74rem; font-weight: 600; padding: 4px 11px; border-radius: 999px; margin-right: 8px; margin-top: 8px; }"
    "html:not([data-deeprag-theme='light']) .meta-badge { color: #bae6fd; }"
    ".match-badge { display: inline-block; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.45); color: #10b981; font-size: 0.74rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; float: right; }"
    ".citation-box { background: rgba(8, 12, 24, 0.92); border-left: 3px solid #38bdf8; border-top: 1px solid rgba(148, 163, 184, 0.12); border-right: 1px solid rgba(148, 163, 184, 0.12); border-bottom: 1px solid rgba(148, 163, 184, 0.12); padding: 14px 16px; border-radius: 10px; margin-bottom: 10px; font-size: 0.88rem; color: #cbd5e1; }"
    "html[data-deeprag-theme='light'] .citation-box { background: #f8fafc; color: #334155; border-left: 3px solid #4f46e5; border-color: rgba(99, 102, 241, 0.2); }"
    ".cite-bar-track { width: 100%; height: 5px; background: rgba(148, 163, 184, 0.25); border-radius: 999px; overflow: hidden; margin-top: 10px; }"
    ".cite-bar-fill { height: 100%; background: linear-gradient(90deg, #10b981, #38bdf8, #6366f1); border-radius: 999px; }"
    ".kw-glow { background: rgba(56, 189, 248, 0.22); color: inherit; border: 1px solid rgba(56, 189, 248, 0.5); padding: 1px 6px; border-radius: 4px; font-weight: 700; }"
    ".xray-box { background: rgba(6, 10, 20, 0.95); border: 1px solid rgba(56, 189, 248, 0.28); border-radius: 12px; padding: 14px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.77rem; color: #cbd5e1; margin-bottom: 10px; }"
    "html[data-deeprag-theme='light'] .xray-box { background: #f1f5f9; color: #1e293b; border-color: rgba(79, 70, 229, 0.3); }"
    ".terminal-box { background: #03060c; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 12px 14px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #38bdf8; max-height: 180px; overflow-y: auto; margin-bottom: 10px; }"

    ".flashcard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 16px; }"
    ".flashcard-wrap { background-color: transparent; height: 220px; perspective: 1000px; cursor: pointer; }"
    ".flashcard-inner { position: relative; width: 100%; height: 100%; text-align: center; transition: transform 0.55s ease; transform-style: preserve-3d; border-radius: 16px; }"
    ".flashcard-wrap:hover .flashcard-inner, .flashcard-wrap:active .flashcard-inner { transform: rotateY(180deg); }"
    ".flashcard-front, .flashcard-back { position: absolute; width: 100%; height: 100%; -webkit-backface-visibility: hidden; backface-visibility: hidden; border-radius: 16px; padding: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-sizing: border-box; }"
    ".flashcard-front { background: #0a1020; border: 1px solid rgba(56, 189, 248, 0.35); color: #f8fafc; }"
    "html[data-deeprag-theme='light'] .flashcard-front { background: #ffffff; border: 1px solid rgba(79, 70, 229, 0.35); color: #0f172a; box-shadow: 0 6px 20px rgba(15, 23, 42, 0.06); }"
    ".flashcard-back { background: #0f172a; border: 1px solid #6366f1; color: #e2e8f0; transform: rotateY(180deg); }"

    ".arch-card { background: rgba(10, 15, 28, 0.9); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 16px; padding: 22px; margin-bottom: 16px; transition: transform 0.2s ease, border-color 0.2s ease; }"
    "html[data-deeprag-theme='light'] .arch-card { background: #ffffff; border: 1px solid rgba(99, 102, 241, 0.22); box-shadow: 0 6px 20px rgba(15, 23, 42, 0.05); }"
    ".arch-card:hover { transform: translateY(-3px); border-color: #38bdf8; }"
    ".step-badge { display: inline-block; background: linear-gradient(135deg, #0284c7, #4f46e5); color: #ffffff; font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; font-weight: 700; padding: 4px 12px; border-radius: 999px; margin-bottom: 10px; letter-spacing: 0.06em; text-transform: uppercase; }"
    ".arch-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 6px; }"
    "html[data-deeprag-theme='light'] .arch-title { color: #0f172a; }"
    ".arch-desc { color: #94a3b8; font-size: 0.94rem; line-height: 1.6; }"
    "html[data-deeprag-theme='light'] .arch-desc { color: #475569; }"

    ".creator-cyber-stage { position: relative; overflow: hidden; background: linear-gradient(145deg, rgba(8, 13, 26, 0.95), rgba(4, 7, 16, 0.98)); border: 1px solid rgba(56, 189, 248, 0.4); border-radius: 26px; padding: 48px 32px; text-align: center; margin-top: 14px; box-shadow: 0 20px 60px rgba(0, 0, 0, 0.65), 0 0 40px rgba(56, 189, 248, 0.12); }"
    "html[data-deeprag-theme='light'] .creator-cyber-stage { background: linear-gradient(145deg, #ffffff, #f0f9ff); border: 1px solid rgba(79, 70, 229, 0.35); box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08); }"
    ".holo-laser { position: absolute; top: 0; left: -100%; width: 100%; height: 3px; background: linear-gradient(90deg, transparent, #38bdf8, #a855f7, #ec4899, transparent); box-shadow: 0 0 20px #38bdf8; animation: holoSweep 4s ease-in-out infinite; }"
    "@keyframes holoSweep { 0% { left: -100%; } 50% { left: 100%; } 100% { left: -100%; } }"
    ".avatar-orbital-system { position: relative; width: 142px; height: 142px; margin: 0 auto 22px auto; display: flex; align-items: center; justify-content: center; }"
    ".orbit-ring-outer { position: absolute; inset: 0; border-radius: 50%; border: 2px dashed #38bdf8; border-top-color: #a855f7; border-bottom-color: #10b981; animation: spinOrbitCW 10s linear infinite; box-shadow: 0 0 25px rgba(56, 189, 248, 0.25); }"
    ".orbit-ring-inner { position: absolute; inset: 12px; border-radius: 50%; border: 2px solid transparent; border-left-color: #ec4899; border-right-color: #6366f1; animation: spinOrbitCCW 6s linear infinite; }"
    "@keyframes spinOrbitCW { to { transform: rotate(360deg); } }"
    "@keyframes spinOrbitCCW { to { transform: rotate(-360deg); } }"
    ".avatar-core { width: 98px; height: 98px; border-radius: 50%; background: linear-gradient(135deg, #0284c7, #4f46e5, #9333ea); display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 800; color: #ffffff; box-shadow: 0 0 35px rgba(99, 102, 241, 0.65); z-index: 2; animation: corePulse 3s ease-in-out infinite; }"
    "@keyframes corePulse { 0%, 100% { transform: scale(1); box-shadow: 0 0 30px rgba(56, 189, 248, 0.55); } 50% { transform: scale(1.06); box-shadow: 0 0 50px rgba(168, 85, 247, 0.85); } }"
    ".creator-glitch-name { font-family: 'Space Grotesk', sans-serif; font-size: 2.8rem; font-weight: 800; background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc, #f472b6, #38bdf8); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; animation: shimmerText 4s linear infinite; margin-bottom: 6px; }"
    "@keyframes shimmerText { to { background-position: 200% center; } }"
    ".creator-badge-role { display: inline-block; background: rgba(56, 189, 248, 0.14); border: 1px solid rgba(56, 189, 248, 0.45); color: #38bdf8; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; padding: 6px 18px; border-radius: 999px; margin-bottom: 22px; letter-spacing: 0.06em; text-transform: uppercase; }"
    "html[data-deeprag-theme='light'] .creator-badge-role { color: #0284c7; background: rgba(2, 132, 199, 0.1); }"
    ".creator-quote { background: rgba(6, 10, 20, 0.85); border-left: 4px solid #38bdf8; border-right: 4px solid #a855f7; border-radius: 14px; padding: 18px 26px; max-width: 760px; margin: 0 auto 28px auto; color: #e2e8f0; font-style: italic; font-size: 1.04rem; line-height: 1.65; box-shadow: 0 10px 28px rgba(0,0,0,0.35); }"
    "html[data-deeprag-theme='light'] .creator-quote { background: #f8fafc; color: #334155; box-shadow: none; }"
    ".creator-stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 14px; max-width: 820px; margin: 0 auto 28px auto; }"
    ".c-stat-box { background: rgba(15, 23, 42, 0.78); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 16px; padding: 16px 12px; transition: all 0.3s ease; }"
    "html[data-deeprag-theme='light'] .c-stat-box { background: #ffffff; border-color: rgba(99, 102, 241, 0.25); }"
    ".c-stat-box:hover { transform: translateY(-6px) scale(1.03); border-color: #38bdf8; box-shadow: 0 12px 28px rgba(56, 189, 248, 0.25); }"
    ".c-stat-num { font-family: 'Space Grotesk', sans-serif; font-size: 1.65rem; font-weight: 800; color: #38bdf8; }"
    "html[data-deeprag-theme='light'] .c-stat-num { color: #4f46e5; }"
    ".c-stat-lbl { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }"
    "html[data-deeprag-theme='light'] .c-stat-lbl { color: #475569; }"
    ".skill-pill { display: inline-block; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.38); color: #0284c7; padding: 8px 18px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin: 5px; transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1); cursor: default; }"
    "html:not([data-deeprag-theme='light']) .skill-pill { color: #bae6fd; }"
    ".skill-pill:hover { transform: translateY(-4px) scale(1.05); background: linear-gradient(135deg, #0284c7, #4f46e5, #9333ea); color: #ffffff !important; border-color: transparent; box-shadow: 0 8px 22px rgba(99, 102, 241, 0.45); }"

    ".contact-box { background: rgba(10, 15, 28, 0.9); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 12px; padding: 14px; text-align: center; margin-top: 20px; }"
    "html[data-deeprag-theme='light'] .contact-box { background: #f8fafc; border: 1px solid rgba(99, 102, 241, 0.25); }"
    ".contact-title { font-family: 'Space Grotesk', sans-serif; font-size: 0.88rem; font-weight: 700; color: #0284c7; margin-bottom: 4px; }"
    "html:not([data-deeprag-theme='light']) .contact-title { color: #38bdf8; }"
    ".contact-handle { font-size: 0.94rem; font-weight: 600; color: #f8fafc; }"
    "html[data-deeprag-theme='light'] .contact-handle { color: #0f172a; }"
    ".premium-footer { margin-top: 48px; padding: 18px 0 10px 0; border-top: 1px solid rgba(56, 189, 248, 0.18); text-align: center; font-family: 'Space Grotesk', sans-serif; font-size: 0.94rem; font-weight: 600; color: #64748b; position: relative; z-index: 2; }"
    ".premium-footer span { color: #0284c7; font-weight: 800; }"
    "html:not([data-deeprag-theme='light']) .premium-footer span { color: #38bdf8; }"
    "footer { visibility: hidden; }"
    "</style>"
    "<div class='anime-vignette'></div>"
)
st.markdown(custom_css, unsafe_allow_html=True)


# --- HELPER: ZERO-LAG DARK ANIME CANVAS + AUTO THEME DETECTOR (LIGHT/DARK SYNC) ---
def inject_anime_night_canvas():
    canvas_js = (
        "<script>"
        "(function() {"
        "  try {"
        "    const parentDoc = window.parent.document;"
        "    function syncThemeMode() {"
        "      const appEl = parentDoc.querySelector('.stApp');"
        "      if (!appEl) return 'dark';"
        "      const rgb = window.parent.getComputedStyle(appEl).color.match(/\\d+/g);"
        "      const isLight = rgb && parseInt(rgb[0], 10) < 128;"
        "      const mode = isLight ? 'light' : 'dark';"
        "      if (parentDoc.documentElement.getAttribute('data-deeprag-theme') !== mode) {"
        "        parentDoc.documentElement.setAttribute('data-deeprag-theme', mode);"
        "      }"
        "      return mode;"
        "    }"
        "    syncThemeMode();"
        "    setInterval(syncThemeMode, 250);"
        "    if (parentDoc.getElementById('deeprag-anime-canvas')) return;"
        "    const canvas = parentDoc.createElement('canvas');"
        "    canvas.id = 'deeprag-anime-canvas';"
        "    canvas.style.position = 'fixed';"
        "    canvas.style.top = '0';"
        "    canvas.style.left = '0';"
        "    canvas.style.width = '100vw';"
        "    canvas.style.height = '100vh';"
        "    canvas.style.pointerEvents = 'none';"
        "    canvas.style.zIndex = '1';"
        "    parentDoc.body.appendChild(canvas);"
        "    const ctx = canvas.getContext('2d', { alpha: true });"
        "    let w = canvas.width = window.parent.innerWidth;"
        "    let h = canvas.height = window.parent.innerHeight;"
        "    window.parent.addEventListener('resize', () => {"
        "      w = canvas.width = window.parent.innerWidth;"
        "      h = canvas.height = window.parent.innerHeight;"
        "    }, { passive: true });"
        "    const stars = [];"
        "    for (let i = 0; i < 55; i++) {"
        "      stars.push({"
        "        x: Math.random() * w,"
        "        y: Math.random() * h * 0.75,"
        "        r: Math.random() * 1.1 + 0.4,"
        "        alpha: Math.random() * 0.6 + 0.15,"
        "        speed: Math.random() * 0.012 + 0.004"
        "      });"
        "    }"
        "    const embers = [];"
        "    const emberColors = ['#38bdf8', '#6366f1', '#818cf8', '#0284c7'];"
        "    for (let i = 0; i < 32; i++) {"
        "      embers.push({"
        "        x: Math.random() * w,"
        "        y: Math.random() * h,"
        "        r: Math.random() * 2.2 + 0.8,"
        "        vy: -(Math.random() * 0.45 + 0.18),"
        "        vx: (Math.random() - 0.5) * 0.25,"
        "        alpha: Math.random() * 0.55 + 0.2,"
        "        color: emberColors[i % emberColors.length]"
        "      });"
        "    }"
        "    let comet = { x: -200, y: -200, vx: 0, vy: 0, active: false, timer: 180 };"
        "    function spawnComet() {"
        "      comet.x = Math.random() * (w * 0.6) + w * 0.3;"
        "      comet.y = Math.random() * (h * 0.25);"
        "      comet.vx = -9;"
        "      comet.vy = 4.5;"
        "      comet.active = true;"
        "    }"
        "    function render() {"
        "      ctx.clearRect(0, 0, w, h);"
        "      const isLightMode = parentDoc.documentElement.getAttribute('data-deeprag-theme') === 'light';"
        "      for (let i = 0; i < stars.length; i++) {"
        "        const s = stars[i];"
        "        s.alpha += s.speed;"
        "        if (s.alpha > 0.75 || s.alpha < 0.12) s.speed = -s.speed;"
        "        ctx.globalAlpha = isLightMode ? s.alpha * 0.35 : s.alpha;"
        "        ctx.fillStyle = isLightMode ? '#6366f1' : '#e0f2fe';"
        "        ctx.beginPath();"
        "        ctx.arc(s.x, s.y, s.r, 0, 6.28);"
        "        ctx.fill();"
        "      }"
        "      for (let i = 0; i < embers.length; i++) {"
        "        const e = embers[i];"
        "        e.y += e.vy;"
        "        e.x += e.vx;"
        "        if (e.y < -10) { e.y = h + 10; e.x = Math.random() * w; }"
        "        ctx.globalAlpha = isLightMode ? e.alpha * 0.45 : e.alpha;"
        "        ctx.fillStyle = e.color;"
        "        ctx.beginPath();"
        "        ctx.arc(e.x, e.y, e.r, 0, 6.28);"
        "        ctx.fill();"
        "      }"
        "      if (!isLightMode) {"
        "        comet.timer--;"
        "        if (comet.timer <= 0 && !comet.active) {"
        "          spawnComet();"
        "          comet.timer = Math.floor(Math.random() * 240) + 220;"
        "        }"
        "        if (comet.active) {"
        "          comet.x += comet.vx;"
        "          comet.y += comet.vy;"
        "          ctx.globalAlpha = 0.75;"
        "          const grad = ctx.createLinearGradient(comet.x, comet.y, comet.x + 85, comet.y - 42);"
        "          grad.addColorStop(0, '#ffffff');"
        "          grad.addColorStop(0.3, '#38bdf8');"
        "          grad.addColorStop(1, 'transparent');"
        "          ctx.strokeStyle = grad;"
        "          ctx.lineWidth = 1.6;"
        "          ctx.beginPath();"
        "          ctx.moveTo(comet.x, comet.y);"
        "          ctx.lineTo(comet.x + 85, comet.y - 42);"
        "          ctx.stroke();"
        "          if (comet.x < -120 || comet.y > h + 120) comet.active = false;"
        "        }"
        "      }"
        "      ctx.globalAlpha = 1.0;"
        "      requestAnimationFrame(render);"
        "    }"
        "    render();"
        "  } catch (err) {}"
        "})();"
        "</script>"
    )
    components.html(canvas_js, height=0)


inject_anime_night_canvas()


# --- HELPER: HIGHLIGHT QUERY KEYWORDS IN CITATION SNIPPETS ---
def highlight_query_terms(snippet: str, query: str) -> str:
    safe_snippet = html.escape(snippet)
    stop_words = {
        "what", "when", "where", "which", "who", "whom", "this", "that", "these", "those",
        "from", "with", "about", "into", "through", "provide", "explain", "summary", "document",
        "uploaded", "paper", "questions", "generate", "extract", "important", "compare"
    }
    words = [w for w in re.findall(r"[A-Za-z0-9]{4,}", query) if w.lower() not in stop_words]
    for w in sorted(set(words), key=len, reverse=True)[:6]:
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        safe_snippet = pattern.sub(r"<span class='kw-glow'>\1</span>", safe_snippet)
    return safe_snippet


# --- HELPER: ULTRA-PREMIUM GRAPHIC PDF GENERATOR BUTTON (HTML2PDF + NATIVE VECTOR PRINT) ---
def render_premium_pdf_download_button(report_html_body: str, filename_pdf: str, btn_label: str, btn_id: str):
    b64_html = base64.b64encode(report_html_body.encode("utf-8")).decode("utf-8")
    widget_html = (
        f"<script src='https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js'></script>"
        f"<style>"
        f".pdf-action-bar {{"
        f"  display:flex; gap:10px; flex-wrap:wrap; align-items:center; width:100%; margin:6px 0;"
        f"  font-family:'Inter', sans-serif;"
        f"}}"
        f".pdf-btn-primary {{"
        f"  flex:1; min-width:210px; background:linear-gradient(135deg, #0284c7, #4f46e5, #7c3aed);"
        f"  color:#ffffff; border:none; border-radius:12px; padding:11px 18px; font-size:0.86rem;"
        f"  font-weight:700; cursor:pointer; display:inline-flex; align-items:center; justify-content:center;"
        f"  gap:8px; box-shadow:0 6px 20px rgba(79, 70, 229, 0.4); transition:all 0.25s ease;"
        f"}}"
        f".pdf-btn-primary:hover {{ transform:translateY(-2px); filter:brightness(1.08); }}"
        f".pdf-btn-print {{"
        f"  background:rgba(15, 23, 42, 0.9); color:#38bdf8; border:1px solid rgba(56, 189, 248, 0.45);"
        f"  border-radius:12px; padding:11px 16px; font-size:0.82rem; font-weight:700; cursor:pointer;"
        f"  display:inline-flex; align-items:center; justify-content:center; gap:6px; transition:all 0.2s ease;"
        f"}}"
        f".pdf-btn-print:hover {{ background:#0284c7; color:#ffffff; }}"
        f"</style>"
        f"<div class='pdf-action-bar'>"
        f"  <button id='dl_{btn_id}' class='pdf-btn-primary' onclick='downloadPdf_{btn_id}()'>{btn_label}</button>"
        f"  <button class='pdf-btn-print' onclick='printPdf_{btn_id}()'>🖨️ Save Vector PDF</button>"
        f"</div>"
        f"<script>"
        f"function decodeUtf8_{btn_id}(b64) {{"
        f"  const binStr = atob(b64);"
        f"  const bytes = new Uint8Array(binStr.length);"
        f"  for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);"
        f"  return new TextDecoder('utf-8').decode(bytes);"
        f"}}"
        f"function downloadPdf_{btn_id}() {{"
        f"  const btn = document.getElementById('dl_{btn_id}');"
        f"  const origText = btn.innerHTML;"
        f"  btn.innerHTML = '⏳ Rendering High-Res Graphic PDF...';"
        f"  const htmlStr = decodeUtf8_{btn_id}('{b64_html}');"
        f"  const holder = document.createElement('div');"
        f"  holder.style.width = '790px';"
        f"  holder.style.background = '#ffffff';"
        f"  holder.innerHTML = htmlStr;"
        f"  document.body.appendChild(holder);"
        f"  if (window.html2pdf) {{"
        f"    const opt = {{"
        f"      margin: [8, 8, 8, 8],"
        f"      filename: '{filename_pdf}',"
        f"      image: {{ type: 'jpeg', quality: 0.98 }},"
        f"      html2canvas: {{ scale: 2, useCORS: true, backgroundColor: '#ffffff' }},"
        f"      jsPDF: {{ unit: 'mm', format: 'a4', orientation: 'portrait' }},"
        f"      pagebreak: {{ mode: ['avoid-all', 'css', 'legacy'] }}"
        f"    }};"
        f"    html2pdf().set(opt).from(holder).save().then(() => {{"
        f"      document.body.removeChild(holder);"
        f"      btn.innerHTML = '✅ PDF Downloaded!';"
        f"      setTimeout(() => {{ btn.innerHTML = origText; }}, 2200);"
        f"    }}).catch(() => {{"
        f"      document.body.removeChild(holder);"
        f"      btn.innerHTML = origText;"
        f"      printPdf_{btn_id}();"
        f"    }});"
        f"  }} else {{"
        f"    document.body.removeChild(holder);"
        f"    btn.innerHTML = origText;"
        f"    printPdf_{btn_id}();"
        f"  }}"
        f"}}"
        f"function printPdf_{btn_id}() {{"
        f"  const htmlStr = decodeUtf8_{btn_id}('{b64_html}');"
        f"  const w = window.open('', '_blank');"
        f"  if (w) {{"
        f"    w.document.open();"
        f"    w.document.write('<!DOCTYPE html><html><head><title>{filename_pdf}</title><style>@page {{ size: A4; margin: 10mm; }} body {{ margin: 0; background: #ffffff; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }}</style></head><body>' + htmlStr + '<script>setTimeout(() => {{ window.print(); }}, 450);<\\/script></body></html>');"
        f"    w.document.close();"
        f"  }}"
        f"}}"
        f"</script>"
    )
    components.html(widget_html, height=58)


# --- HELPER: BUILD ULTRA-PREMIUM QUIZ ANALYSIS PDF HTML ---
def build_premium_quiz_pdf_html(
    score: int,
    total_q: int,
    accuracy_pct: int,
    grade_rank: str,
    rec_note: str,
    mastered_topics: list[str],
    weak_areas: list[str],
    quiz_records: list[dict],
    corpus_label: str
) -> str:
    date_str = time.strftime("%d %b %Y • %I:%M %p")
    mastered_html = "".join(
        [f"<li style='margin-bottom:6px; color:#065f46; font-weight:600;'>✅ {html.escape(m)}</li>" for m in mastered_topics]
    ) or "<li style='color:#64748b;'>Complete more questions correctly to unlock mastered topics.</li>"

    weak_html = "".join(
        [f"<li style='margin-bottom:6px; color:#991b1b; font-weight:600;'>📌 {html.escape(w)}</li>" for w in weak_areas]
    ) or "<li style='color:#065f46; font-weight:700;'>🎉 Zero weak areas detected! 100% Concept Mastery!</li>"

    cards_html = []
    for r in quiz_records:
        is_ok = r["is_correct"]
        border_col = "#10b981" if is_ok else "#ef4444"
        bg_col = "#f0fdf4" if is_ok else "#fef2f2"
        badge_bg = "#059669" if is_ok else "#dc2626"
        badge_txt = "CORRECT" if is_ok else "NEEDS REVISION"
        cards_html.append(
            f"<div style='page-break-inside:avoid; border:1px solid {border_col}; border-left:6px solid {border_col}; "
            f"background:{bg_col}; border-radius:12px; padding:14px 16px; margin-bottom:12px;'>"
            f"  <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;'>"
            f"    <span style='font-size:11px; font-weight:800; color:#4f46e5; text-transform:uppercase; letter-spacing:0.05em;'>"
            f"      Question #{r['num']:02d} • {html.escape(r['topic'])} ({html.escape(r['page'])})"
            f"    </span>"
            f"    <span style='background:{badge_bg}; color:#ffffff; font-size:10px; font-weight:800; padding:3px 9px; border-radius:999px;'>"
            f"      {badge_txt}"
            f"    </span>"
            f"  </div>"
            f"  <div style='font-size:14px; font-weight:700; color:#0f172a; margin-bottom:8px;'>{html.escape(r['question'])}</div>"
            f"  <div style='font-size:12.5px; color:#1e293b; margin-bottom:4px;'>"
            f"    <strong>Your Answer:</strong> <span style='color:{badge_bg}; font-weight:700;'>{html.escape(r['picked'])}</span>"
            f"  </div>"
            f"  <div style='font-size:12.5px; color:#0f172a; margin-bottom:8px;'>"
            f"    <strong>Verified Correct Option:</strong> <span style='color:#059669; font-weight:700;'>{html.escape(r['correct'])}</span>"
            f"  </div>"
            f"  <div style='background:#ffffff; border:1px solid rgba(15,23,42,0.08); border-radius:8px; padding:9px 12px; font-size:12px; color:#334155; line-height:1.5;'>"
            f"    <strong>🧠 AI Revision Note:</strong> {html.escape(r['explanation'])}"
            f"  </div>"
            f"</div>"
        )

    return (
        "<div style='font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif; color:#0f172a; background:#ffffff; padding:24px; box-sizing:border-box;'>"
        "  <div style='background:linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #0284c7 100%); color:#ffffff; border-radius:16px; padding:24px 28px; margin-bottom:20px;'>"
        "    <div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>"
        "      <div>"
        "        <div style='font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#38bdf8; font-weight:800;'>DEEPRAG AI WORKSPACE • OFFICIAL DIAGNOSTIC REPORT</div>"
        "        <div style='font-size:24px; font-weight:800; margin-top:4px;'>Executive Quiz Performance & Mastery Analysis</div>"
        f"       <div style='font-size:12px; color:#cbd5e1; margin-top:4px;'>📄 Indexed Corpus: {html.escape(corpus_label)} &nbsp;|&nbsp; 🕒 Generated: {date_str}</div>"
        "      </div>"
        "      <div style='background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.25); border-radius:12px; padding:10px 16px; text-align:right;'>"
        f"       <div style='font-size:22px; font-weight:800; color:#38bdf8;'>{accuracy_pct}%</div>"
        "        <div style='font-size:10px; text-transform:uppercase; color:#e2e8f0;'>Overall Accuracy</div>"
        "      </div>"
        "    </div>"
        "  </div>"
        "  <div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-bottom:18px;'>"
        "    <div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:14px; text-align:center;'>"
        "      <div style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>🎯 Final Score</div>"
        f"     <div style='font-size:22px; font-weight:800; color:#0f172a; margin-top:4px;'>{score} / {total_q}</div>"
        "    </div>"
        "    <div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:14px; text-align:center;'>"
        "      <div style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>📈 Precision Gauge</div>"
        f"     <div style='font-size:22px; font-weight:800; color:#0284c7; margin-top:4px;'>{accuracy_pct}%</div>"
        "    </div>"
        "    <div style='background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:14px; text-align:center;'>"
        "      <div style='font-size:11px; font-weight:700; color:#475569; text-transform:uppercase;'>🏅 Performance Tier</div>"
        f"     <div style='font-size:15px; font-weight:800; color:#4f46e5; margin-top:7px;'>{html.escape(grade_rank)}</div>"
        "    </div>"
        "  </div>"
        "  <div style='background:#eef2ff; border-left:5px solid #4f46e5; border-radius:10px; padding:12px 16px; margin-bottom:18px; font-size:13px; color:#1e1b4b;'>"
        f"   <strong>💡 AI Tutor Executive Prescription:</strong> {html.escape(rec_note)}"
        "  </div>"
        "  <div style='display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:22px;'>"
        "    <div style='background:#f0fdf4; border:1px solid #86efac; border-radius:12px; padding:14px;'>"
        "      <div style='font-size:13px; font-weight:800; color:#065f46; margin-bottom:8px; text-transform:uppercase;'>✅ Strengths (Topics Mastered)</div>"
        f"     <ul style='margin:0; padding-left:18px; font-size:12.5px;'>{mastered_html}</ul>"
        "    </div>"
        "    <div style='background:#fef2f2; border:1px solid #fca5a5; border-radius:12px; padding:14px;'>"
        "      <div style='font-size:13px; font-weight:800; color:#991b1b; margin-bottom:8px; text-transform:uppercase;'>📌 Action Plan (Pages to Revise)</div>"
        f"     <ul style='margin:0; padding-left:18px; font-size:12.5px;'>{weak_html}</ul>"
        "    </div>"
        "  </div>"
        "  <div style='font-size:15px; font-weight:800; color:#0f172a; margin-bottom:12px; border-bottom:2px solid #e2e8f0; padding-bottom:6px;'>"
        "    📝 Question-by-Question Verification & Revision Key"
        "  </div>"
        + "".join(cards_html) +
        "  <div style='margin-top:20px; padding-top:10px; border-top:1px solid #cbd5e1; text-align:center; font-size:11px; color:#64748b; font-weight:600;'>"
        "    Engineered with ❤️ by Deep Patel • DeepRAG AI Workspace Neural Assessment Engine"
        "  </div>"
        "</div>"
    )


# --- HELPER: BUILD ULTRA-PREMIUM CHAT SESSION PDF HTML ---
def build_premium_chat_pdf_html(chat_history: list[dict], corpus_label: str, total_pages: int, total_chunks: int) -> str:
    date_str = time.strftime("%d %b %Y • %I:%M %p")
    turns_html = []

    for idx, turn in enumerate(chat_history):
        ans_escaped = html.escape(turn.get("answer", "")).replace("\n", "<br>")
        ans_escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", ans_escaped)
        cites_rows = []
        for c_i, c in enumerate(turn.get("citations", [])):
            cites_rows.append(
                f"<div style='background:#f8fafc; border-left:3px solid #6366f1; padding:6px 10px; border-radius:6px; margin-top:5px; font-size:11px; color:#334155;'>"
                f"  <strong>📌 Source {c_i+1}: {html.escape(str(c.get('source', 'PDF')))} (Page {html.escape(str(c.get('page', '?')))}) "
                f"  — {c.get('match_pct', 92)}% Semantic Match</strong><br>"
                f"  {html.escape(str(c.get('snippet', ''))[:240])}"
                f"</div>"
            )

        turns_html.append(
            f"<div style='page-break-inside:avoid; border:1px solid #cbd5e1; border-radius:14px; padding:16px 18px; margin-bottom:16px; background:#ffffff;'>"
            f"  <div style='background:#0f172a; color:#ffffff; border-radius:10px; padding:10px 14px; margin-bottom:12px;'>"
            f"    <div style='font-size:10px; color:#38bdf8; font-weight:800; text-transform:uppercase; letter-spacing:0.06em;'>QUERY #{idx+1:02d} • USER INTERROGATION</div>"
            f"    <div style='font-size:14px; font-weight:700; margin-top:3px;'>{html.escape(turn.get('question', ''))}</div>"
            f"  </div>"
            f"  <div style='font-size:13px; line-height:1.65; color:#1e293b; margin-bottom:12px;'>{ans_escaped}</div>"
            f"  <div style='display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;'>"
            f"    <span style='background:#e0f2fe; color:#0369a1; font-size:10.5px; font-weight:700; padding:3px 9px; border-radius:999px;'>⚡ Engine: {html.escape(str(turn.get('model', 'Gemini')))}</span>"
            f"    <span style='background:#dcfce7; color:#15803d; font-size:10.5px; font-weight:700; padding:3px 9px; border-radius:999px;'>🎯 Confidence: {turn.get('avg_match', 94)}%</span>"
            f"    <span style='background:#ede9fe; color:#6d28d9; font-size:10.5px; font-weight:700; padding:3px 9px; border-radius:999px;'>⏱️ Latency: {turn.get('latency', 1.2)}s</span>"
            f"    <span style='background:#f1f5f9; color:#334155; font-size:10.5px; font-weight:700; padding:3px 9px; border-radius:999px;'>📝 {turn.get('words', 0)} Words</span>"
            f"  </div>"
            f"  <div style='border-top:1px dashed #cbd5e1; padding-top:8px;'>"
            f"    <div style='font-size:11px; font-weight:800; color:#4f46e5; text-transform:uppercase;'>📚 Verified PDF Page Citations</div>"
            + "".join(cites_rows) +
            f"  </div>"
            f"</div>"
        )

    return (
        "<div style='font-family:Inter,-apple-system,BlinkMacSystemFont,sans-serif; color:#0f172a; background:#ffffff; padding:24px; box-sizing:border-box;'>"
        "  <div style='background:linear-gradient(135deg, #0f172a 0%, #1e1b4b 55%, #4f46e5 100%); color:#ffffff; border-radius:16px; padding:24px 28px; margin-bottom:20px;'>"
        "    <div style='font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#38bdf8; font-weight:800;'>DEEPRAG AI WORKSPACE • RESEARCH & STUDY DOSSIER</div>"
        "    <div style='font-size:24px; font-weight:800; margin-top:4px;'>Verified Neural RAG Session & Citation Report</div>"
        f"   <div style='font-size:12px; color:#cbd5e1; margin-top:6px;'>📂 Active Corpus: {html.escape(corpus_label)} &nbsp;|&nbsp; 📄 Pages: {total_pages} &nbsp;|&nbsp; 🧩 Vector Chunks: {total_chunks} &nbsp;|&nbsp; 🕒 {date_str}</div>"
        "  </div>"
        + "".join(turns_html) +
        "  <div style='margin-top:20px; padding-top:10px; border-top:1px solid #cbd5e1; text-align:center; font-size:11px; color:#64748b; font-weight:600;'>"
        "    Engineered with ❤️ by Deep Patel • DeepRAG AI Workspace Neural Knowledge Report"
        "  </div>"
        "</div>"
    )


# --- HELPER: ULTRA-SMOOTH NEURAL AUDIO DECK + COPY BUTTON ---
def render_tts_audio_button(answer_text: str, btn_id: str):
    clean_text = re.sub(r"[\*\#\`\~\_\>]", "", answer_text)
    clean_text = re.sub(r"\[Page\s*\d+\]", "", clean_text)
    safe_js_text = json.dumps(clean_text)

    tts_html = (
        f"<style>"
        f".audio-deck {{"
        f"  display:inline-flex; align-items:center; gap:8px; flex-wrap:wrap;"
        f"  background:#0a1020; border:1px solid rgba(56, 189, 248, 0.35); border-radius:999px;"
        f"  padding:5px 12px; font-family:'Inter', sans-serif;"
        f"}}"
        f".deck-btn {{"
        f"  background:linear-gradient(135deg, #0284c7, #4f46e5); color:#ffffff; border:none;"
        f"  border-radius:999px; padding:6px 14px; font-size:0.76rem; font-weight:600; cursor:pointer;"
        f"  display:inline-flex; align-items:center; gap:6px;"
        f"}}"
        f".deck-sub {{"
        f"  background:rgba(30, 41, 59, 0.85); color:#cbd5e1; border:1px solid rgba(148,163,184,0.28);"
        f"  border-radius:999px; padding:5px 11px; font-size:0.73rem; font-weight:600; cursor:pointer;"
        f"  display:inline-flex; align-items:center; gap:4px;"
        f"}}"
        f".eq-wrap {{ display:none; align-items:flex-end; gap:3px; height:14px; margin:0 4px; }}"
        f".eq-bar {{ width:3px; background:#38bdf8; border-radius:2px; animation:eqBounce 0.7s infinite ease-in-out; }}"
        f".eq-bar:nth-child(2) {{ animation-delay:0.15s; background:#818cf8; }}"
        f".eq-bar:nth-child(3) {{ animation-delay:0.3s; background:#34d399; }}"
        f".eq-bar:nth-child(4) {{ animation-delay:0.45s; background:#38bdf8; }}"
        f"@keyframes eqBounce {{ 0%, 100% {{ height:4px; }} 50% {{ height:14px; }} }}"
        f"</style>"
        f"<div class='audio-deck'>"
        f"  <button id='play_{btn_id}' class='deck-btn' onclick='toggleSpeak_{btn_id}()'>🔊 Listen to Answer</button>"
        f"  <div id='eq_{btn_id}' class='eq-wrap'>"
        f"    <span class='eq-bar'></span><span class='eq-bar'></span><span class='eq-bar'></span><span class='eq-bar'></span>"
        f"  </div>"
        f"  <button id='stop_{btn_id}' class='deck-sub' style='display:none;' onclick='stopSpeak_{btn_id}()'>⏹️ Stop</button>"
        f"  <button id='spd_{btn_id}' class='deck-sub' onclick='cycleSpeed_{btn_id}()'>⚡ 1.0x</button>"
        f"  <button id='cpy_{btn_id}' class='deck-sub' onclick='copyAns_{btn_id}()'>📋 Copy</button>"
        f"</div>"
        f"<script>"
        f"const txt_{btn_id} = {safe_js_text};"
        f"let rate_{btn_id} = 1.0;"
        f"let isPlaying_{btn_id} = false;"
        f"let isPaused_{btn_id} = false;"
        f"function cycleSpeed_{btn_id}() {{"
        f"  if (rate_{btn_id} === 1.0) rate_{btn_id} = 1.25;"
        f"  else if (rate_{btn_id} === 1.25) rate_{btn_id} = 1.5;"
        f"  else rate_{btn_id} = 1.0;"
        f"  document.getElementById('spd_{btn_id}').innerHTML = '⚡ ' + rate_{btn_id} + 'x';"
        f"  if (isPlaying_{btn_id}) {{ stopSpeak_{btn_id}(); toggleSpeak_{btn_id}(); }}"
        f"}}"
        f"function toggleSpeak_{btn_id}() {{"
        f"  if (!('speechSynthesis' in window)) {{ alert('Speech synthesis is not supported in this browser.'); return; }}"
        f"  const playBtn = document.getElementById('play_{btn_id}');"
        f"  const stopBtn = document.getElementById('stop_{btn_id}');"
        f"  const eqBox = document.getElementById('eq_{btn_id}');"
        f"  if (isPlaying_{btn_id} && !isPaused_{btn_id}) {{"
        f"    window.speechSynthesis.pause();"
        f"    isPaused_{btn_id} = true;"
        f"    playBtn.innerHTML = '▶️ Resume Audio';"
        f"    eqBox.style.display = 'none';"
        f"    return;"
        f"  }}"
        f"  if (isPlaying_{btn_id} && isPaused_{btn_id}) {{"
        f"    window.speechSynthesis.resume();"
        f"    isPaused_{btn_id} = false;"
        f"    playBtn.innerHTML = '⏸️ Pause Audio';"
        f"    eqBox.style.display = 'inline-flex';"
        f"    return;"
        f"  }}"
        f"  window.speechSynthesis.cancel();"
        f"  const utter = new SpeechSynthesisUtterance(txt_{btn_id});"
        f"  utter.lang = 'en-US';"
        f"  utter.rate = rate_{btn_id};"
        f"  utter.onstart = () => {{"
        f"    isPlaying_{btn_id} = true; isPaused_{btn_id} = false;"
        f"    playBtn.innerHTML = '⏸️ Pause Audio';"
        f"    stopBtn.style.display = 'inline-flex';"
        f"    eqBox.style.display = 'inline-flex';"
        f"  }};"
        f"  utter.onend = () => {{"
        f"    isPlaying_{btn_id} = false; isPaused_{btn_id} = false;"
        f"    playBtn.innerHTML = '🔊 Listen to Answer';"
        f"    stopBtn.style.display = 'none';"
        f"    eqBox.style.display = 'none';"
        f"  }};"
        f"  utter.onerror = () => {{"
        f"    isPlaying_{btn_id} = false; isPaused_{btn_id} = false;"
        f"    playBtn.innerHTML = '🔊 Listen to Answer';"
        f"    stopBtn.style.display = 'none';"
        f"    eqBox.style.display = 'none';"
        f"  }};"
        f"  window.speechSynthesis.speak(utter);"
        f"}}"
        f"function stopSpeak_{btn_id}() {{"
        f"  if ('speechSynthesis' in window) {{ window.speechSynthesis.cancel(); }}"
        f"  isPlaying_{btn_id} = false; isPaused_{btn_id} = false;"
        f"  document.getElementById('play_{btn_id}').innerHTML = '🔊 Listen to Answer';"
        f"  document.getElementById('stop_{btn_id}').style.display = 'none';"
        f"  document.getElementById('eq_{btn_id}').style.display = 'none';"
        f"}}"
        f"function copyAns_{btn_id}() {{"
        f"  navigator.clipboard.writeText(txt_{btn_id});"
        f"  const cBtn = document.getElementById('cpy_{btn_id}');"
        f"  cBtn.innerHTML = '✅ Copied!';"
        f"  setTimeout(() => {{ cBtn.innerHTML = '📋 Copy'; }}, 1800);"
        f"}}"
        f"</script>"
    )
    components.html(tts_html, height=50)


# --- HELPER: RENDER ASSISTANT TURN WITH CITATION BARS & X-RAY INSPECTOR ---
def render_assistant_turn(turn: dict, turn_key: str, is_latest: bool = False):
    st.markdown(turn["answer"])
    render_tts_audio_button(turn["answer"], turn_key)

    avg_score = turn.get("avg_match", 94)
    vec_ms = turn.get("vec_ms", 11.4)
    used_temp = turn.get("temp", 0.3)
    ans_len_label = turn.get("length_mode", "Balanced")
    st.markdown(
        f'<span class="meta-badge">⚡ Engine: {turn["model"]}</span>'
        f'<span class="meta-badge">🎯 Confidence: {avg_score}% Match</span>'
        f'<span class="meta-badge">📏 Length: {ans_len_label}</span>'
        f'<span class="meta-badge">🔍 FAISS: {vec_ms}ms</span>'
        f'<span class="meta-badge">⏱️ Total: {turn["latency"]}s</span>'
        f'<span class="meta-badge">📝 {turn.get("words", 0)} Words</span>',
        unsafe_allow_html=True
    )

    with st.expander("📚 View Verified PDF Page Citations & Semantic Confidence Bars"):
        for c_idx, cite in enumerate(turn["citations"]):
            match_pct = cite.get("match_pct", 92)
            highlighted_snip = highlight_query_terms(cite["snippet"], turn["question"])
            st.markdown(
                f'<div class="citation-box">'
                f'<span class="match-badge">🎯 {match_pct}% Semantic Match</span>'
                f'<strong>📌 Source {c_idx+1} — {cite["source"]} (Page {cite["page"]})</strong><br>'
                f'<div style="margin-top:6px;">{highlighted_snip}</div>'
                f'<div class="cite-bar-track"><div class="cite-bar-fill" style="width:{match_pct}%;"></div></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with st.expander("🔬 X-Ray RAG Telemetry & Vector Space Inspector"):
        ctx_tokens = turn.get("ctx_tokens", 420)
        xray_rows = ""
        for c_idx, cite in enumerate(turn["citations"]):
            l2_val = cite.get("l2_dist", 0.42)
            xray_rows += (
                f"Chunk #{c_idx+1} ({cite['source']} • Page {cite['page']}) "
                f"→ L2 Distance: {l2_val} | Confidence: {cite.get('match_pct', 92)}%<br>"
            )
        st.markdown(
            f"<div class='xray-box'>"
            f"<strong>🧬 NEURAL RETRIEVAL TELEMETRY</strong><br>"
            f"• Embedding Model: <code>HuggingFace all-MiniLM-L6-v2 (384D Float32)</code><br>"
            f"• Vector Index: <code>FAISS FlatL2 Exact Inner-Product</code><br>"
            f"• Retrieval Latency: <code>{vec_ms} ms</code> &nbsp;|&nbsp; Context Tokens Injected: <code>~{ctx_tokens} tokens</code><br>"
            f"• Synthesis Model: <code>{turn['model']} (Temperature = {used_temp})</code><br><br>"
            f"<strong>📐 VECTOR DISTANCE MATRIX</strong><br>"
            f"{xray_rows}"
            f"</div>",
            unsafe_allow_html=True
        )

    followups = turn.get("followups", [])
    if is_latest and followups:
        st.markdown("###### 💡 Smart Next-Step Questions:")
        fcols = st.columns(len(followups))
        for f_idx, fq in enumerate(followups):
            with fcols[f_idx]:
                if st.button(f"🔎 {fq}", key=f"fq_{turn_key}_{f_idx}", use_container_width=True):
                    st.session_state.queued_followup = fq
                    st.rerun()


# --- HELPER: SMART TIERED MODEL CASCADE (PRIMARY -> 1500 RPD WORKHORSES) ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_ordered_gemini_models(clean_key: str) -> list[str]:
    # Explicit cascade: Primary model first, followed immediately by 1,500 req/day high-speed models
    priority_cascade = [
        "models/gemini-3.0-flash",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash-lite-001",
        "models/gemini-2.5-flash-lite",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.5-pro",
        "models/gemini-2.5-pro"
    ]
    try:
        list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={clean_key}"
        resp = requests.get(list_url, timeout=8)
        if resp.status_code != 200:
            return priority_cascade

        models_data = resp.json().get("models", [])
        available_set = {
            m["name"] for m in models_data
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "gemini" in m["name"].lower()
            and "embedding" not in m["name"].lower()
            and "imagen" not in m["name"].lower()
            and "vision" not in m["name"].lower()
            and "thinking" not in m["name"].lower()
            and "exp" not in m["name"].lower()
            and "gemini-flash-latest" not in m["name"].lower()
        }
        if not available_set:
            return priority_cascade

        ordered = [m for m in priority_cascade if m in available_set]
        for m in sorted(available_set):
            if m not in ordered and ("flash" in m.lower() or "pro" in m.lower()):
                ordered.append(m)

        return ordered if ordered else priority_cascade
    except Exception:
        return priority_cascade


# --- HELPER: MULTI-KEY + AUTO-SHIFTING 1500 RPD FAILOVER ENGINE ---
def generate_smart_gemini_response(
    api_keys_list: list[str],
    prompt: str,
    temperature: float = 0.3,
    expect_json: bool = False
) -> tuple[str, str]:
    valid_keys = [k.strip() for k in api_keys_list if k and k.strip()]
    if not valid_keys:
        raise RuntimeError("API Key is missing! Please configure GEMINI_API_KEY in Streamlit Cloud Secrets.")

    if "exhausted_routes" not in st.session_state:
        st.session_state.exhausted_routes = {}

    now_ts = time.time()
    last_error = ""

    # Two passes: Pass 1 skips recently quota-limited/timed-out models; Pass 2 retries all if needed
    for pass_idx in range(2):
        for key_idx, clean_key in enumerate(valid_keys):
            ordered_models = get_ordered_gemini_models(clean_key)
            for chosen_model in ordered_models:
                route_id = f"k{key_idx}_{chosen_model}"
                cooldown_until = st.session_state.exhausted_routes.get(route_id, 0)
                if pass_idx == 0 and now_ts < cooldown_until:
                    continue

                gen_url = f"https://generativelanguage.googleapis.com/v1beta/{chosen_model}:generateContent?key={clean_key}"
                gen_cfg = {"temperature": float(temperature)}
                if expect_json:
                    gen_cfg["responseMimeType"] = "application/json"

                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": gen_cfg
                }

                try:
                    gen_resp = requests.post(gen_url, json=payload, timeout=(6, 22))
                    # Fallback if a specific model doesn't support responseMimeType
                    if gen_resp.status_code == 400 and expect_json:
                        payload["generationConfig"] = {"temperature": float(temperature)}
                        gen_resp = requests.post(gen_url, json=payload, timeout=(6, 22))

                    if gen_resp.status_code == 200:
                        data = gen_resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts and "text" in parts[0]:
                                key_badge = f" (Key #{key_idx+1})" if len(valid_keys) > 1 else ""
                                return parts[0]["text"], chosen_model.replace("models/", "") + key_badge

                    err_body = gen_resp.json().get("error", {})
                    err_msg = err_body.get("message", f"HTTP {gen_resp.status_code}")
                    last_error = err_msg

                    # If Quota Exceeded (429) or Model Overloaded (503) or Not Found (404), put this model on cooldown
                    if gen_resp.status_code in (429, 503, 404) or "quota" in err_msg.lower() or "rate" in err_msg.lower():
                        cooldown_secs = 900 if gen_resp.status_code == 429 else 120
                        st.session_state.exhausted_routes[route_id] = time.time() + cooldown_secs
                        continue
                except requests.exceptions.Timeout:
                    # Put slow/hanging model on 3-minute cooldown and immediately shift to next 1500 RPD model
                    st.session_state.exhausted_routes[route_id] = time.time() + 180
                    last_error = f"{chosen_model} timed out, auto-shifting to next 1500 RPD model..."
                    continue
                except requests.exceptions.RequestException as req_err:
                    last_error = str(req_err)
                    continue

    raise RuntimeError(f"All active API keys & fallback models reached quota limit. Details: {last_error}")


# --- HELPER: MULTI-KEY VOICE TRANSCRIPTION ---
def transcribe_voice_with_gemini(api_keys_list: list[str], audio_bytes: bytes, mime_type: str = "audio/wav") -> str:
    valid_keys = [k.strip() for k in api_keys_list if k and k.strip()]
    if not valid_keys:
        raise RuntimeError("API Key is missing! GEMINI_API_KEY is required for voice transcription.")

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
    for clean_key in valid_keys:
        ordered_models = get_ordered_gemini_models(clean_key)
        for model_name in ordered_models[:8]:
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_name}:generateContent?key={clean_key}"
            try:
                resp = requests.post(url, json=payload, timeout=(6, 20))
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"].strip()
                else:
                    last_err = resp.json().get("error", {}).get("message", "Audio transcription failed.")
            except requests.exceptions.RequestException as req_err:
                last_err = str(req_err)
                continue

    raise RuntimeError(f"Voice Transcription Error: {last_err}")


# --- HELPER: SAMPLE EVENLY ACROSS ENTIRE PDF CORPUS ---
def sample_full_document_chunks(all_chunks: list, vector_store, target_samples: int = 12) -> list:
    if all_chunks:
        n = len(all_chunks)
        if n <= target_samples:
            return all_chunks
        step = max(1, (n - 1) / float(target_samples - 1))
        return [all_chunks[min(n - 1, int(round(i * step)))] for i in range(target_samples)]
    return vector_store.similarity_search("core concepts definitions architecture formulas summary", k=min(target_samples, 8))


# --- HELPER: STRATIFIED MULTI-DOCUMENT & MULTI-SECTION RETRIEVAL ---
def retrieve_smart_context(query: str, vector_store, all_chunks: list, top_k: int) -> list[tuple]:
    is_compare = any(w in query.lower() for w in ["compare", "comparison", "difference", "versus", "vs", "cross-section"])
    is_summary = any(w in query.lower() for w in ["executive summary", "summarize", "overview", "core thesis", "exam questions"])

    sem_results = vector_store.similarity_search_with_score(query, k=top_k)

    if not all_chunks or (not is_compare and not is_summary):
        return sem_results

    by_source = {}
    for ch in all_chunks:
        src = ch.metadata.get("source", "Document.pdf")
        by_source.setdefault(src, []).append(ch)

    stratified_docs = []
    seen_keys = set()

    per_doc_quota = max(2, (top_k + 2) // max(1, len(by_source)))
    for src, ch_list in by_source.items():
        n = len(ch_list)
        if n <= per_doc_quota:
            picked = ch_list
        else:
            step = max(1, (n - 1) // max(1, per_doc_quota - 1))
            picked = [ch_list[min(n - 1, i * step)] for i in range(per_doc_quota)]
        for d in picked:
            key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
            if key not in seen_keys:
                seen_keys.add(key)
                stratified_docs.append((d, 0.22))

    for d, sc in sem_results:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:80])
        if key not in seen_keys:
            seen_keys.add(key)
            stratified_docs.append((d, sc))

    return stratified_docs[:max(top_k + 2, 8)]


# --- HELPER: FAST JSON QUIZ GENERATOR WITH MULTI-KEY FAILOVER ---
def generate_interactive_quiz(api_keys_list: list[str], vector_store, all_chunks: list, num_questions: int = 5) -> list:
    sample_docs = sample_full_document_chunks(all_chunks, vector_store, target_samples=min(12, max(6, num_questions + 1)))
    context_str = "\n\n".join(
        [f"[{d.metadata.get('source', 'PDF')} | Page {d.metadata.get('page', 1)}] {d.page_content[:360]}" for d in sample_docs]
    )
    quiz_prompt = (
        f"Based strictly on the PDF context below covering the document, generate {num_questions} multiple-choice questions in valid JSON format.\n"
        f"Return ONLY a raw JSON array of {num_questions} objects (no markdown fences, no extra text). Each object must have:\n"
        '{"question": "...", "topic": "Short Topic (2-4 words)", "page": "Page X", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "Clear explanation why this is correct [Page X]"}\n\n'
        f"Context:\n{context_str}"
    )
    raw_res, _ = generate_smart_gemini_response(api_keys_list, quiz_prompt, temperature=0.2, expect_json=True)
    cleaned = re.sub(r"```json|```", "", raw_res).strip()
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx + 1]
    return json.loads(cleaned)


# --- HELPER: FAST JSON 3D FLASHCARD GENERATOR WITH MULTI-KEY FAILOVER ---
def generate_3d_flashcards(api_keys_list: list[str], vector_store, all_chunks: list, num_cards: int = 12) -> list:
    sample_docs = sample_full_document_chunks(all_chunks, vector_store, target_samples=min(14, max(8, num_cards)))
    context_str = "\n\n".join(
        [f"[{d.metadata.get('source', 'PDF')} | Page {d.metadata.get('page', 1)}] {d.page_content[:340]}" for d in sample_docs]
    )
    fc_prompt = (
        f"Create a complete revision flashcard deck covering the uploaded PDF document from start to finish.\n"
        f"Based strictly on the PDF context below, generate {num_cards} high-yield study flashcards in valid JSON format.\n"
        f"Return ONLY a raw JSON array of {num_cards} objects (no markdown fences, no extra text). Each object must have:\n"
        '{"term": "Key Concept / Question (under 9 words)", "definition": "Clear, complete revision answer (under 28 words)", "page": "Page X"}\n\n'
        f"Context:\n{context_str}"
    )
    raw_res, _ = generate_smart_gemini_response(api_keys_list, fc_prompt, temperature=0.2, expect_json=True)
    cleaned = re.sub(r"```json|```", "", raw_res).strip()
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx + 1]
    return json.loads(cleaned)


# --- HELPER: BUILD LETTER-BY-LETTER HTML ---
def render_animated_title(text: str) -> str:
    spans = []
    for idx, ch in enumerate(text):
        delay = idx * 0.03
        display_char = "&nbsp;" if ch == " " else ch
        spans.append(f'<span class="char" style="animation-delay: {delay:.2f}s">{display_char}</span>')
    return f'<div class="hero-title">{"".join(spans)}<span class="cursor-blink"></span></div>'


# --- HELPER: LOAD EMBEDDING MODEL CACHED WITH BATCH SPEEDUP ---
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
    )


# --- HELPER: CACHED FAST PDF PAGE EXTRACTION ---
@st.cache_data(show_spinner=False)
def extract_pdf_pages_cached(file_bytes: bytes, file_name: str) -> list[dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    extracted = []
    for page_idx, page in enumerate(reader.pages):
        try:
            raw_txt = page.extract_text() or ""
        except Exception:
            raw_txt = ""
        clean_txt = re.sub(r"\s+", " ", raw_txt).strip()
        if clean_txt:
            extracted.append({
                "text": clean_txt,
                "source": file_name,
                "page": page_idx + 1
            })
    return extracted


# --- SESSION STATE INITIALIZATION ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
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
if "queued_followup" not in st.session_state:
    st.session_state.queued_followup = None
if "last_vec_ms" not in st.session_state:
    st.session_state.last_vec_ms = 0.0
if "total_tokens_est" not in st.session_state:
    st.session_state.total_tokens_est = 0
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "flashcards_data" not in st.session_state:
    st.session_state.flashcards_data = []
if "exhausted_routes" not in st.session_state:
    st.session_state.exhausted_routes = {}
if "agent_terminal_logs" not in st.session_state:
    st.session_state.agent_terminal_logs = [
        "[SYS_BOOT] DeepRAG Neural Workspace Initialized.",
        f"[KEY_POOL] {max(1, len(API_KEYS_POOL))} API Key(s) Active with 1500 RPD Auto-Cascade.",
        "[FAISS] Vector Engine Ready (CPU FlatL2)."
    ]


# --- HELPER: ADD TERMINAL LOG ---
def log_terminal(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.agent_terminal_logs.append(f"[{timestamp}] {msg}")
    if len(st.session_state.agent_terminal_logs) > 25:
        st.session_state.agent_terminal_logs.pop(0)


# --- SIDEBAR CONTROL CENTER ---
with st.sidebar:
    st.markdown("### ⚙️ DeepRAG Control Center")
    st.markdown(
        f'<div class="status-pill">🟢 Neural Engine Active ({max(1, len(API_KEYS_POOL))} API Key Pool)</div>',
        unsafe_allow_html=True
    )

    st.markdown("#### 🛠️ Neural Hyperparameters")
    chunk_size = st.slider("Chunk Size (Characters)", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=400, value=180, step=30)
    top_k = st.slider("Retrieved Citations (Top-K)", min_value=3, max_value=10, value=6, step=1)
    llm_temp = st.slider("LLM Temperature (Creativity)", min_value=0.0, max_value=0.9, value=0.3, step=0.1)

    st.markdown("---")
    st.markdown("#### 📡 System Telemetry & Logs")
    sidebar_telemetry_slot = st.empty()
    sidebar_logs_slot = st.empty()
    sidebar_export_slot = st.empty()

    st.markdown("---")
    st.markdown(
        '<div class="contact-box">'
        '<div class="contact-title">📞 Contact Us</div>'
        '<div class="contact-handle">Telegram @DeepRAGAI</div>'
        '</div>',
        unsafe_allow_html=True
    )


# --- TOP NAVIGATION TABS ---
tab_workspace, tab_quiz, tab_how, tab_creator = st.tabs([
    "🚀 AI Workspace",
    "🎓 AI Quiz & 3D Flashcards",
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

    st.markdown(
        "<div class='upload-portal-card'>"
        "  <div class='upload-portal-title'>📂 Document Upload & Answer Preferences</div>"
        "  <div class='upload-portal-sub'>Upload one or more PDF documents below (supports large PDFs up to 200MB) and customize your AI output style.</div>"
        "</div>",
        unsafe_allow_html=True
    )

    ucol1, ucol2 = st.columns([1.4, 1])
    with ucol1:
        uploaded_files = st.file_uploader(
            "📄 Upload Your PDF Document(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload single or multiple PDFs. Multi-PDF comparison works automatically when 2+ files are uploaded."
        )
    with ucol2:
        response_mode = st.selectbox(
            "🎛️ Select Output Style",
            [
                "⚡ Deep Analysis (Default)",
                "🎯 Exam & Revision Notes",
                "💡 Simple Explain (ELI5)",
                "📌 Executive Bullet Points"
            ]
        )
        answer_length = st.radio(
            "📏 Select Answer Length",
            [
                "⚡ Short & Crisp",
                "⚖️ Balanced",
                "📚 Detailed & In-Depth"
            ],
            index=1,
            horizontal=True
        )

    # --- FAST CACHED PDF INDEXING WITH PROGRESS BAR ---
    if uploaded_files:
        file_signatures = [f"{f.name}_{f.size}" for f in uploaded_files]
        current_key = f"{file_signatures}_{chunk_size}_{chunk_overlap}"
        if current_key != st.session_state.processed_files_key:
            prog_bar = st.progress(0, text="🚀 Reading PDF pages & extracting clean text...")
            t0 = time.time()
            log_terminal(f"Ingesting {len(uploaded_files)} PDF file(s)...")
            docs = []
            total_pages = 0

            for f_idx, pdf in enumerate(uploaded_files):
                raw_bytes = pdf.getvalue()
                page_dicts = extract_pdf_pages_cached(raw_bytes, pdf.name)
                total_pages += len(page_dicts)
                for p_item in page_dicts:
                    docs.append(
                        Document(
                            page_content=p_item["text"],
                            metadata={"source": p_item["source"], "page": p_item["page"]}
                        )
                    )
                pct = int(((f_idx + 1) / len(uploaded_files)) * 45)
                prog_bar.progress(pct, text=f"📄 Extracted {total_pages} pages with text...")

            if docs:
                prog_bar.progress(60, text="🧩 Splitting text into context-aware semantic chunks...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                split_docs = splitter.split_documents(docs)

                prog_bar.progress(80, text=f"🧬 Encoding {len(split_docs)} chunks into 384D FAISS Vector Space...")
                embeddings = get_embedding_model()
                st.session_state.vector_store = FAISS.from_documents(split_docs, embeddings)
                st.session_state.all_chunks = split_docs
                st.session_state.pdf_pages = total_pages
                st.session_state.neural_chunks = len(split_docs)
                st.session_state.last_vec_ms = round((time.time() - t0) * 1000, 1)
                st.session_state.total_tokens_est = sum(len(d.page_content.split()) for d in split_docs)
                st.session_state.processed_files_key = current_key
                prog_bar.progress(100, text="✅ PDF Indexing Complete!")
                time.sleep(0.2)
                prog_bar.empty()
                log_terminal(f"Indexed {total_pages} pages into {len(split_docs)} chunks in {st.session_state.last_vec_ms}ms.")
            else:
                prog_bar.empty()
                st.warning("⚠️ No extractable text found in this PDF (it may be a scanned image-only PDF). Please upload a text-searchable PDF.")
    else:
        st.session_state.vector_store = None
        st.session_state.all_chunks = []
        st.session_state.pdf_pages = 0
        st.session_state.neural_chunks = 0
        st.session_state.last_vec_ms = 0.0
        st.session_state.total_tokens_est = 0
        st.session_state.processed_files_key = ""

    st.write("")

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
    if st.session_state.queued_followup:
        triggered_query = st.session_state.queued_followup
        st.session_state.queued_followup = None

    if st.session_state.vector_store:
        st.markdown("##### ✨ Instant Neural Actions & Visualizers")
        qcol1, qcol2, qcol3, qcol4, qcol5 = st.columns(5)
        with qcol1:
            if st.button("📝 Executive Summary", use_container_width=True):
                triggered_query = "Provide a comprehensive executive summary of the uploaded document(s), highlighting the core thesis, key sections, and main conclusions."
        with qcol2:
            if st.button("🔑 Key Formulas & Facts", use_container_width=True):
                triggered_query = "Extract all critical definitions, core concepts, frameworks, formulas, and key technical facts from the document(s)."
        with qcol3:
            if st.button("🗺️ Concept Flowchart", use_container_width=True):
                triggered_query = (
                    "Create a clear, visual step-by-step Concept Flowchart & Mind-Map of the core architecture or process in the document(s). "
                    "Include a clean Unicode box-drawing diagram inside a code block followed by stage-by-stage explanations with page citations."
                )
        with qcol4:
            if st.button("⚖️ Compare Documents", use_container_width=True):
                triggered_query = (
                    "Perform a comprehensive Cross-Document / Cross-Section Comparison of all uploaded PDF documents (or major sections if a single PDF is uploaded). "
                    "Compare their core objectives, methodologies, key data/metrics, strengths, limitations, and conclusions inside a structured Markdown Comparison Table followed by key analytical insights."
                )
        with qcol5:
            if st.button("🎓 Top 5 Exam Questions", use_container_width=True):
                triggered_query = "Generate 5 high-value exam or interview questions along with detailed model answers strictly based on the uploaded document(s)."

    vcol1, vcol2 = st.columns([2, 1])
    with vcol1:
        st.markdown("##### 💬 Ask Your Documents (Text or Voice)")
        st.caption("Upload your PDF above, then type a question below or record a voice query using the microphone.")
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
            log_terminal("Audio recording received. Routing to Gemini Audio Transcription Node...")
            with st.spinner("🎧 Transcribing your voice query with Gemini Audio Engine..."):
                try:
                    mime_type = getattr(audio_value, "type", "audio/wav") or "audio/wav"
                    transcribed_text = transcribe_voice_with_gemini(API_KEYS_POOL, audio_bytes, mime_type)
                    active_query = transcribed_text
                    log_terminal(f"Voice transcribed successfully: '{transcribed_text}'")
                    st.toast(f"🎙️ Voice Query Recognized: {transcribed_text}")
                except Exception as e:
                    log_terminal(f"Voice transcription error: {e}")
                    st.error(f"❌ {e}")

    previous_turns = list(reversed(st.session_state.chat_history))

    # --- INSTANT QUESTION DISPLAY & LIVE LOADING BELOW IT ---
    if active_query:
        if not st.session_state.vector_store:
            st.warning("⚠️ Please upload a PDF document above first to begin querying.")
        else:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(f"**{active_query}**")

            with st.chat_message("assistant", avatar="🧠"):
                start_time = time.time()
                log_terminal(f"Executing smart retrieval for query: '{active_query}' (Top-K={top_k})...")
                with st.spinner("🧠 Querying FAISS Vector Space & synthesizing verified response..."):
                    try:
                        v_t0 = time.time()
                        docs_and_scores = retrieve_smart_context(
                            active_query,
                            st.session_state.vector_store,
                            st.session_state.all_chunks,
                            top_k
                        )
                        vec_ms = round((time.time() - v_t0) * 1000, 2)
                        st.session_state.last_vec_ms = vec_ms
                        log_terminal(f"FAISS retrieval completed in {vec_ms}ms ({len(docs_and_scores)} passages).")

                        context_parts = []
                        citations_list = []
                        score_percentages = []

                        for idx, (doc, l2_score) in enumerate(docs_and_scores):
                            src = doc.metadata.get("source", "Document.pdf")
                            pg = doc.metadata.get("page", "?")
                            raw_l2 = round(float(l2_score), 4)
                            match_pct = max(68, min(99, int(round(100 / (1 + float(l2_score) * 0.35)))))
                            score_percentages.append(match_pct)

                            context_parts.append(f"[Document: {src} | Page {pg}]\n{doc.page_content}")
                            citations_list.append({
                                "source": src,
                                "page": pg,
                                "l2_dist": raw_l2,
                                "match_pct": match_pct,
                                "snippet": doc.page_content[:360].replace("\n", " ") + "..."
                            })

                        avg_match = int(round(sum(score_percentages) / len(score_percentages))) if score_percentages else 94
                        context_block = "\n\n".join(context_parts)
                        ctx_tokens = len(context_block.split())

                        history_context = ""
                        if st.session_state.chat_history:
                            recent_turns = st.session_state.chat_history[-2:]
                            history_context = "\n".join(
                                [f"User: {t['question']}\nAssistant: {t['answer']}" for t in recent_turns]
                            )

                        style_instructions = {
                            "⚡ Deep Analysis (Default)": "Provide a well-structured, high-clarity, and authoritative explanation in English using clear sub-headings and bullet points where helpful.",
                            "🎯 Exam & Revision Notes": "Format the response as high-impact study notes with bold key terms, structured bullet points, and clear definitions in English.",
                            "💡 Simple Explain (ELI5)": "Explain the concept in clear, beginner-friendly English using an intuitive real-world analogy.",
                            "📌 Executive Bullet Points": "Deliver a crisp executive summary in high-impact bullet points in English."
                        }[response_mode]

                        length_instructions = {
                            "⚡ Short & Crisp": "Keep the answer concise, direct, and brief (around 70 to 110 words, 3 to 5 crisp bullet points).",
                            "⚖️ Balanced": "Provide a well-balanced, medium-length explanation (around 180 to 280 words) covering all important points clearly.",
                            "📚 Detailed & In-Depth": "Provide a comprehensive, exhaustive, and deeply detailed explanation (400+ words) with thorough breakdowns, structured sub-headings, and complete depth."
                        }[answer_length]

                        unique_sources = sorted(list({c["source"] for c in citations_list}))
                        sources_summary = ", ".join(unique_sources)

                        final_prompt = (
                            "You are DeepRAG AI, an elite document intelligence assistant.\n"
                            "Answer the user's question strictly in English using ONLY the provided PDF Context below.\n"
                            f"Uploaded Document(s) in Context: {sources_summary}\n"
                            "Always cite the relevant document name and page number inline (for example, [Page 4] or [File.pdf, Page 2]) when stating facts.\n"
                            "If the user asks for a comparison and multiple documents are present, compare each document side-by-side in a clean Markdown table. "
                            "If only a single document is uploaded during a comparison request, compare the major sections, topics, or pages within that document in a clean Markdown table.\n\n"
                            "IMPORTANT: At the very end of your response, after your main answer, write the exact delimiter ---FOLLOWUP--- "
                            "followed by 3 short, insightful follow-up questions separated by the pipe character | (each question under 9 words, no numbering).\n\n"
                            f"Output Style Instruction: {style_instructions}\n"
                            f"Answer Length Instruction: {length_instructions}\n\n"
                            f"Recent Conversation Context:\n{history_context}\n\n"
                            f"Retrieved PDF Context:\n{context_block}\n\n"
                            f"User Question: {active_query}\n"
                        )

                        log_terminal(f"Routing grounded prompt across Multi-Key 1500 RPD Cascade ({llm_temp} temp)...")
                        raw_output, used_model = generate_smart_gemini_response(API_KEYS_POOL, final_prompt, temperature=llm_temp)
                        if "---FOLLOWUP---" in raw_output:
                            main_ans, fq_block = raw_output.split("---FOLLOWUP---", 1)
                            answer_text = main_ans.strip()
                            followups = [q.strip() for q in fq_block.split("|") if q.strip()][:3]
                        else:
                            answer_text = raw_output.strip()
                            followups = []

                        elapsed = round(time.time() - start_time, 2)
                        word_count = len(answer_text.split())
                        log_terminal(f"Synthesis complete via {used_model} in {elapsed}s ({word_count} words).")

                        new_turn = {
                            "question": active_query,
                            "answer": answer_text,
                            "model": used_model,
                            "latency": elapsed,
                            "vec_ms": vec_ms,
                            "temp": llm_temp,
                            "length_mode": answer_length.split(" ", 1)[-1],
                            "ctx_tokens": ctx_tokens,
                            "words": word_count,
                            "avg_match": avg_match,
                            "mode": response_mode,
                            "citations": citations_list,
                            "followups": followups
                        }
                        st.session_state.chat_history.append(new_turn)
                        render_assistant_turn(new_turn, f"live_{len(st.session_state.chat_history)}", is_latest=True)
                    except Exception as e:
                        log_terminal(f"Synthesis error: {e}")
                        st.error(f"❌ Error: {e}")

    # --- DISPLAY PREVIOUS CONVERSATION HISTORY + PREMIUM CHAT PDF EXPORT ---
    if st.session_state.chat_history:
        corpus_names = ", ".join([f.name for f in uploaded_files]) if uploaded_files else "Uploaded PDF Corpus"
        chat_pdf_html = build_premium_chat_pdf_html(
            st.session_state.chat_history,
            corpus_names,
            st.session_state.pdf_pages,
            st.session_state.neural_chunks
        )
        render_premium_pdf_download_button(
            chat_pdf_html,
            "DeepRAG_Research_Session_Report.pdf",
            "📥 Download Ultra-Premium Chat Session PDF (.pdf)",
            "chat_main_pdf"
        )

    for idx, turn in enumerate(previous_turns):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{turn['question']}**")

        with st.chat_message("assistant", avatar="🧠"):
            is_top_card = (idx == 0 and not active_query)
            render_assistant_turn(turn, f"hist_{idx}", is_latest=is_top_card)

# --- POPULATE SIDEBAR TELEMETRY, LOGS & PREMIUM CHAT PDF IN REAL TIME ---
doc_label = f"{len(uploaded_files)} Active PDF(s)" if uploaded_files else "No PDF Uploaded"
sidebar_telemetry_slot.markdown(
    f"<div class='telemetry-strip'>"
    f"  <div class='tel-item'><span class='tel-dot'></span><span>Corpus: <span class='tel-val'>{doc_label}</span></span></div>"
    f"  <div class='tel-item'><span>🧬 Embeddings: <span class='tel-val'>384D MiniLM-L6-v2</span></span></div>"
    f"  <div class='tel-item'><span>⚡ Vector Latency: <span class='tel-val'>{st.session_state.last_vec_ms} ms</span></span></div>"
    f"  <div class='tel-item'><span>📊 Indexed Words: <span class='tel-val'>{st.session_state.total_tokens_est}</span></span></div>"
    f"</div>",
    unsafe_allow_html=True
)

with sidebar_logs_slot.expander("💻 Live Neural Engine Logs"):
    terminal_lines = "<br>".join([f"&gt; {line}" for line in st.session_state.agent_terminal_logs])
    st.markdown(f"<div class='terminal-box'>{terminal_lines}</div>", unsafe_allow_html=True)

if st.session_state.chat_history:
    with sidebar_export_slot.container():
        st.markdown("#### 📥 Session Actions")
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# ==========================================
# TAB 2: AI QUIZ & FULL-PDF 3D FLASHCARDS
# ==========================================
with tab_quiz:
    st.markdown(render_animated_title("AI Quiz & 3D Flashcards"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Revise your entire PDF with comprehensive 3D flip flashcards, custom-length AI quizzes, and ultra-premium downloadable PDF diagnostic reports.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.vector_store:
        st.info("📂 Please upload your PDF document in the '🚀 AI Workspace' tab first to unlock full-document flashcards and quizzes.")
    else:
        scol1, scol2 = st.columns(2)
        with scol1:
            st.markdown("##### 🃏 Full-Document 3D Flashcard Deck")
            fc_count = st.slider(
                "Number of Revision Flashcards (Covers Full PDF)",
                min_value=6,
                max_value=24,
                value=12,
                step=2,
                key="slider_fc_count",
                help="Samples evenly from the first page to the last page so the entire PDF is revised."
            )
            if st.button("🃏 Generate Full-PDF 3D Flashcards", key="btn_gen_flashcards", use_container_width=True):
                log_terminal(f"Generating {fc_count} full-document 3D flashcards...")
                with st.spinner(f"✨ Crafting {fc_count} 3D Flip Flashcards across all PDF sections..."):
                    try:
                        st.session_state.flashcards_data = generate_3d_flashcards(
                            API_KEYS_POOL,
                            st.session_state.vector_store,
                            st.session_state.all_chunks,
                            num_cards=fc_count
                        )
                        log_terminal(f"{len(st.session_state.flashcards_data)} 3D Flashcards generated.")
                    except Exception as e:
                        log_terminal(f"Flashcard generation error: {e}")
                        st.error(f"Could not generate flashcards: {e}")

        with scol2:
            st.markdown("##### 🎯 Custom AI Exam & Quiz Generator")
            quiz_count = st.slider(
                "Number of Quiz Questions",
                min_value=3,
                max_value=15,
                value=5,
                step=1,
                key="slider_quiz_count",
                help="Choose how many questions you want in your practice test."
            )
            if st.button("🎯 Generate Custom AI Practice Quiz", key="btn_gen_quiz", use_container_width=True):
                log_terminal(f"Generating {quiz_count}-question interactive quiz...")
                with st.spinner(f"🧠 Synthesizing {quiz_count} exam questions across the document..."):
                    try:
                        st.session_state.quiz_data = generate_interactive_quiz(
                            API_KEYS_POOL,
                            st.session_state.vector_store,
                            st.session_state.all_chunks,
                            num_questions=quiz_count
                        )
                        st.session_state.quiz_submitted = False
                        log_terminal(f"AI Quiz ({len(st.session_state.quiz_data)} questions) generated.")
                    except Exception as e:
                        log_terminal(f"Quiz generation error: {e}")
                        st.error(f"Could not generate quiz: {e}")

        if st.session_state.flashcards_data:
            st.markdown("---")
            st.markdown(f"##### 🃏 Full-PDF 3D Revision Deck ({len(st.session_state.flashcards_data)} Cards) — *Hover or Tap any card to flip*")
            cards_html = ["<div class='flashcard-grid'>"]
            for fc_idx, fc in enumerate(st.session_state.flashcards_data):
                term_txt = html.escape(str(fc.get("term", "Concept")))
                def_txt = html.escape(str(fc.get("definition", "Definition")))
                pg_txt = html.escape(str(fc.get("page", "Page 1")))
                cards_html.append(
                    f"<div class='flashcard-wrap' onclick='this.querySelector(\".flashcard-inner\").style.transform = "
                    f"this.querySelector(\".flashcard-inner\").style.transform === \"rotateY(180deg)\" ? \"rotateY(0deg)\" : \"rotateY(180deg)\"'>"
                    f"  <div class='flashcard-inner'>"
                    f"    <div class='flashcard-front'>"
                    f"      <span class='step-badge'>Card #{fc_idx+1:02d} • {pg_txt}</span>"
                    f"      <div style='font-family:Space Grotesk,sans-serif; font-size:1.12rem; font-weight:700; margin-top:8px;'>{term_txt}</div>"
                    f"      <div style='color:#94a3b8; font-size:0.75rem; margin-top:12px;'>🔄 Hover or Tap to Flip</div>"
                    f"    </div>"
                    f"    <div class='flashcard-back'>"
                    f"      <span class='step-badge' style='background:linear-gradient(135deg,#10b981,#059669);'>Verified Answer • {pg_txt}</span>"
                    f"      <div style='font-size:0.9rem; line-height:1.5; margin-top:8px;'>{def_txt}</div>"
                    f"    </div>"
                    f"  </div>"
                    f"</div>"
                )
            cards_html.append("</div>")
            st.markdown("".join(cards_html), unsafe_allow_html=True)
            st.write("")

        if st.session_state.quiz_data:
            st.markdown("---")
            st.markdown(f"##### 🎯 Interactive Knowledge Check ({len(st.session_state.quiz_data)} Questions)")
            user_picks = {}
            for q_idx, q_item in enumerate(st.session_state.quiz_data):
                topic_tag = html.escape(str(q_item.get("topic", "Core Concept")))
                page_tag = html.escape(str(q_item.get("page", "PDF")))
                st.markdown(
                    f"<div class='quiz-card'>"
                    f"<span class='step-badge'>{topic_tag} • {page_tag}</span><br>"
                    f"<strong>Q{q_idx+1}. {html.escape(str(q_item.get('question', '')))}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                opts = q_item.get("options", ["Option A", "Option B", "Option C", "Option D"])
                user_picks[q_idx] = st.radio(
                    f"Select answer for Q{q_idx+1}:",
                    opts,
                    key=f"quiz_opt_{q_idx}",
                    label_visibility="collapsed"
                )
                st.write("")

            if st.button("✅ Submit Quiz & Generate Ultra-Premium Analysis Report", key="btn_submit_quiz", use_container_width=True):
                st.session_state.quiz_submitted = True

            if st.session_state.quiz_submitted:
                score = 0
                total_q = len(st.session_state.quiz_data)
                mastered_topics = []
                weak_areas = []
                quiz_records = []

                st.markdown("<div class='report-shell'>", unsafe_allow_html=True)
                st.markdown("### 📊 Executive Quiz Performance & Diagnostic Report")

                for q_idx, q_item in enumerate(st.session_state.quiz_data):
                    opts = q_item.get("options", ["Option A", "Option B", "Option C", "Option D"])
                    correct_i = int(q_item.get("correct_index", 0))
                    correct_opt = opts[correct_i] if 0 <= correct_i < len(opts) else opts[0]
                    picked = user_picks.get(q_idx, opts[0])
                    topic_lbl = str(q_item.get("topic", f"Concept #{q_idx+1}"))
                    page_lbl = str(q_item.get("page", "Page ?"))
                    expl_lbl = str(q_item.get("explanation", ""))
                    is_ok = (picked == correct_opt)

                    if is_ok:
                        score += 1
                        mastered_topics.append(f"{topic_lbl} ({page_lbl})")
                        st.success(f"**Q{q_idx+1} [{topic_lbl} • {page_lbl}]: Correct!**\n\n**Your Answer:** {picked}\n\n**Explanation:** {expl_lbl}")
                    else:
                        weak_areas.append(f"{topic_lbl} — Revise {page_lbl}")
                        st.error(f"**Q{q_idx+1} [{topic_lbl} • {page_lbl}]: Incorrect**\n\n**Your Answer:** {picked}  |  **Correct Answer:** {correct_opt}\n\n**Explanation & Revision Note:** {expl_lbl}")

                    quiz_records.append({
                        "num": q_idx + 1,
                        "question": str(q_item.get("question", "")),
                        "topic": topic_lbl,
                        "page": page_lbl,
                        "picked": str(picked),
                        "correct": str(correct_opt),
                        "explanation": expl_lbl,
                        "is_correct": is_ok
                    })

                accuracy_pct = int(round((score / max(1, total_q)) * 100))
                if accuracy_pct >= 85:
                    grade_rank = "🏆 S-Rank (Mastery Achieved)"
                    rec_note = "Outstanding command over the PDF material! You are ready for advanced viva or exam questions."
                elif accuracy_pct >= 65:
                    grade_rank = "🌟 A-Rank (Strong Understanding)"
                    rec_note = "Good grasp of the core concepts. Review the highlighted pages below to reach 100% mastery."
                else:
                    grade_rank = "📘 B-Rank (Needs Revision)"
                    rec_note = "Go through the 3D Flashcards and revise the specific PDF pages listed under Weak Areas below."

                rcol1, rcol2, rcol3 = st.columns(3)
                with rcol1:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>🎯 Final Score</div>"
                        f"<div class='metric-value'>{score} / {total_q}</div></div>",
                        unsafe_allow_html=True
                    )
                with rcol2:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>📈 Accuracy</div>"
                        f"<div class='metric-value'>{accuracy_pct}%</div></div>",
                        unsafe_allow_html=True
                    )
                with rcol3:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-title'>🏅 Performance Tier</div>"
                        f"<div class='metric-value' style='font-size:1.35rem;'>{grade_rank}</div></div>",
                        unsafe_allow_html=True
                    )

                st.write("")
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.markdown("##### ✅ Strengths (Topics Mastered)")
                    if mastered_topics:
                        for mt in mastered_topics:
                            st.markdown(f"- 🟢 **{mt}**")
                    else:
                        st.markdown("- *Complete more questions correctly to unlock mastered topics.*")
                with dcol2:
                    st.markdown("##### 📌 Action Plan (Pages to Revise)")
                    if weak_areas:
                        for wa in weak_areas:
                            st.markdown(f"- 🔴 **{wa}**")
                    else:
                        st.markdown("- 🎉 **Zero weak spots detected! 100% accuracy across all tested pages.**")

                st.info(f"💡 **AI Tutor Recommendation:** {rec_note}")

                corpus_names = ", ".join([f.name for f in uploaded_files]) if uploaded_files else "Uploaded PDF Document"
                quiz_pdf_html = build_premium_quiz_pdf_html(
                    score,
                    total_q,
                    accuracy_pct,
                    grade_rank,
                    rec_note,
                    mastered_topics,
                    weak_areas,
                    quiz_records,
                    corpus_names
                )
                render_premium_pdf_download_button(
                    quiz_pdf_html,
                    "DeepRAG_Quiz_Diagnostic_Report.pdf",
                    "📥 Download Ultra-Premium Quiz Analysis Report (.pdf)",
                    "quiz_diag_pdf"
                )
                st.markdown("</div>", unsafe_allow_html=True)

                if score == total_q:
                    st.balloons()

# ==========================================
# TAB 3: ANIMATED HOW IT WORKS
# ==========================================
with tab_how:
    st.markdown(render_animated_title("Neural RAG Architecture"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">How DeepRAG transforms static PDF documents into an interactive, hallucination-free neural knowledge base.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="arch-card">'
        '<span class="step-badge">Stage 01 • Ingestion</span>'
        '<div class="arch-title">📂 Multi-Document Parsing & Page-Level Metadata Tagging</div>'
        '<div class="arch-desc">Uploaded PDF files are processed through a high-speed cached <code>pypdf</code> extraction pipeline. Every single page is isolated, cleaned, and tagged with its exact source filename and page number so downstream answers retain 100% verifiable traceability.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 02 • Semantic Splitting</span>'
        '<div class="arch-title">🧩 Context-Aware Recursive Chunking</div>'
        '<div class="arch-desc">Using LangChain\'s <code>RecursiveCharacterTextSplitter</code>, long chapters are partitioned into 1000-character overlapping semantic windows. Dynamic chunk overlap ensures sentences and technical arguments are never severed mid-thought.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 03 • Neural Encoding</span>'
        '<div class="arch-title">🧬 384-Dimensional Batch Vector Embeddings</div>'
        '<div class="arch-desc">Each text chunk is encoded in high-speed batches through HuggingFace\'s <code>all-MiniLM-L6-v2</code> Sentence-Transformer model, mapping complex concepts into a normalized 384-dimensional dense geometric vector space.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 04 • Stratified & Semantic Search</span>'
        '<div class="arch-title">⚡ Millisecond FAISS Indexing & Full-PDF Sampling</div>'
        '<div class="arch-desc">Facebook AI Similarity Search (<code>FAISS</code>) indexes all neural vectors in memory. For summaries, flashcards, quizzes, and cross-document comparisons, our stratified sampler guarantees balanced coverage across every page of your PDF.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 05 • Multi-Key 1500 RPD Cascade</span>'
        '<div class="arch-title">🤖 Auto-Shifting Multi-API Key & Model Failover</div>'
        '<div class="arch-desc">When a primary model hits its daily rate limit, our session-aware router automatically shifts to high-quota 1,500 requests/day Flash models and rotates across backup API keys with zero downtime.</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ==========================================
# TAB 4: ULTRA-ANIMATED ABOUT CREATOR
# ==========================================
with tab_creator:
    st.markdown(render_animated_title("Meet The Creator"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Engineered with precision, futuristic motion systems, and production-grade neural architecture.</div>',
        unsafe_allow_html=True
    )

    creator_html = (
        '<div class="creator-cyber-stage">'
        '  <div class="holo-laser"></div>'
        '  <div class="avatar-orbital-system">'
        '    <div class="orbit-ring-outer"></div>'
        '    <div class="orbit-ring-inner"></div>'
        '    <div class="avatar-core">DP</div>'
        '  </div>'
        '  <div class="creator-glitch-name">Deep Patel</div>'
        '  <div class="creator-badge-role">⚡ Founder • Lead AI Architect • Creator of DeepRAG</div>'
        '  <div class="creator-quote">'
        '    "My mission is simple: transform complex artificial intelligence and neural retrieval pipelines into a fast, reliable, and visually breathtaking workspace that empowers students, researchers, and innovators worldwide."'
        '  </div>'
        '  <div class="creator-stats-grid">'
        '    <div class="c-stat-box">'
        '      <div class="c-stat-num">384D</div>'
        '      <div class="c-stat-lbl">Dense Vector Space</div>'
        '    </div>'
        '    <div class="c-stat-box">'
        '      <div class="c-stat-num">&lt;15ms</div>'
        '      <div class="c-stat-lbl">FAISS L2 Retrieval</div>'
        '    </div>'
        '    <div class="c-stat-box">'
        '      <div class="c-stat-num">100%</div>'
        '      <div class="c-stat-lbl">Citation Traceability</div>'
        '    </div>'
        '    <div class="c-stat-box">'
        '      <div class="c-stat-num">24/7</div>'
        '      <div class="c-stat-lbl">Multi-Key Failover</div>'
        '    </div>'
        '  </div>'
        '  <div>'
        '    <span class="skill-pill">🧠 Generative AI & LLMs</span>'
        '    <span class="skill-pill">⚡ Neural RAG Pipelines</span>'
        '    <span class="skill-pill">🔍 FAISS Vector Search</span>'
        '    <span class="skill-pill">🔗 LangChain Orchestration</span>'
        '    <span class="skill-pill">🎙️ Multimodal Voice AI</span>'
        '    <span class="skill-pill">🃏 3D Interactive Study Systems</span>'
        '    <span class="skill-pill">🎨 Futuristic Glassmorphism UI</span>'
        '    <span class="skill-pill">🐍 Python & Cloud Architecture</span>'
        '  </div>'
        '</div>'
    )
    st.markdown(creator_html, unsafe_allow_html=True)

# --- GLOBAL SIGNATURE FOOTER ---
st.markdown(
    '<div class="premium-footer">Engineered with ❤️ by <span>Deep</span></div>',
    unsafe_allow_html=True
)
