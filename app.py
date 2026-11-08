import streamlit as st
import streamlit.components.v1 as components
import requests
import base64
import time
import hashlib
import json
import re
import html
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

# --- SECURE API KEY RETRIEVAL ---
DEFAULT_API_KEY = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        API_KEY = DEFAULT_API_KEY
except Exception:
    API_KEY = DEFAULT_API_KEY

# --- ADAPTIVE DARK & LIGHT THEME CSS (ZERO-LAG ANIME DARK + FROSTED LIGHT) ---
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

    ".telemetry-strip { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; background: rgba(10, 15, 28, 0.88); border: 1px solid rgba(56, 189, 248, 0.22); border-radius: 14px; padding: 10px 18px; margin-bottom: 22px; font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: #cbd5e1; }"
    "html[data-deeprag-theme='light'] .telemetry-strip { background: rgba(255, 255, 255, 0.92); border: 1px solid rgba(99, 102, 241, 0.25); color: #334155; box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05); }"
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
    ".terminal-box { background: #03060c; border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 14px 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #38bdf8; max-height: 190px; overflow-y: auto; margin-bottom: 18px; }"
    ".flashcard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-top: 16px; }"
    ".flashcard-wrap { background-color: transparent; height: 210px; perspective: 1000px; cursor: pointer; }"
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
    ".creator-hero { background: rgba(10, 15, 28, 0.92); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 22px; padding: 38px 30px; text-align: center; margin-top: 12px; }"
    "html[data-deeprag-theme='light'] .creator-hero { background: #ffffff; border: 1px solid rgba(99, 102, 241, 0.28); box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06); }"
    ".avatar-ring { width: 92px; height: 92px; border-radius: 50%; margin: 0 auto 16px auto; background: linear-gradient(135deg, #0284c7, #4f46e5); display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 800; color: #ffffff; box-shadow: 0 0 24px rgba(56, 189, 248, 0.4); }"
    ".creator-name { font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px; }"
    "html[data-deeprag-theme='light'] .creator-name { color: #0f172a; }"
    ".creator-role { color: #0284c7; font-size: 1.02rem; font-weight: 600; margin-bottom: 20px; }"
    "html:not([data-deeprag-theme='light']) .creator-role { color: #7dd3fc; }"
    ".creator-quote { background: rgba(6, 10, 20, 0.85); border-left: 3px solid #38bdf8; border-radius: 10px; padding: 16px 22px; max-width: 700px; margin: 0 auto 24px auto; color: #e2e8f0; font-style: italic; font-size: 1rem; line-height: 1.6; }"
    "html[data-deeprag-theme='light'] .creator-quote { background: #f8fafc; color: #334155; border-left: 3px solid #4f46e5; }"
    ".skill-pill { display: inline-block; background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); color: #0284c7; padding: 6px 15px; border-radius: 999px; font-size: 0.84rem; font-weight: 600; margin: 4px; }"
    "html:not([data-deeprag-theme='light']) .skill-pill { color: #bae6fd; }"
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
        "uploaded", "paper", "questions", "generate", "extract", "important"
    }
    words = [w for w in re.findall(r"[A-Za-z0-9]{4,}", query) if w.lower() not in stop_words]
    for w in sorted(set(words), key=len, reverse=True)[:6]:
        pattern = re.compile(rf"\b({re.escape(w)})\b", re.IGNORECASE)
        safe_snippet = pattern.sub(r"<span class='kw-glow'>\1</span>", safe_snippet)
    return safe_snippet


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
def generate_smart_gemini_response(api_key: str, prompt: str, temperature: float = 0.3) -> tuple[str, str]:
    clean_key = api_key.strip()
    if not clean_key:
        raise RuntimeError("API Key is missing! Please configure GEMINI_API_KEY in Streamlit Cloud Secrets.")

    ordered_models = get_ordered_gemini_models(clean_key)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature)}
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


# --- HELPER: GENERATE INTERACTIVE QUIZ FROM VECTOR STORE ---
def generate_interactive_quiz(api_key: str, vector_store) -> list:
    sample_docs = vector_store.similarity_search("core concepts architecture definitions formulas summary", k=4)
    context_str = "\n\n".join([f"[Page {d.metadata.get('page', 1)}] {d.page_content}" for d in sample_docs])
    quiz_prompt = (
        "Based strictly on the PDF context below, generate 3 multiple-choice questions in valid JSON format.\n"
        "Return ONLY a raw JSON list of 3 objects (no markdown fences, no extra text). Each object must have:\n"
        '{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_index": 0, "explanation": "... [Page X]"}\n\n'
        f"Context:\n{context_str}"
    )
    raw_res, _ = generate_smart_gemini_response(api_key, quiz_prompt, temperature=0.2)
    cleaned = re.sub(r"```json|```", "", raw_res).strip()
    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx:end_idx + 1]
    return json.loads(cleaned)


# --- HELPER: GENERATE 3D FLIP FLASHCARDS FROM VECTOR STORE ---
def generate_3d_flashcards(api_key: str, vector_store) -> list:
    sample_docs = vector_store.similarity_search("important definitions key terms formulas architecture", k=4)
    context_str = "\n\n".join([f"[Page {d.metadata.get('page', 1)}] {d.page_content}" for d in sample_docs])
    fc_prompt = (
        "Based strictly on the PDF context below, generate 4 high-yield study flashcards in valid JSON format.\n"
        "Return ONLY a raw JSON list of 4 objects (no markdown fences, no extra text). Each object must have:\n"
        '{"term": "Short Concept or Question (under 8 words)", "definition": "Clear, concise explanation (under 28 words)", "page": "Page X"}\n\n'
        f"Context:\n{context_str}"
    )
    raw_res, _ = generate_smart_gemini_response(api_key, fc_prompt, temperature=0.2)
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
if "queued_followup" not in st.session_state:
    st.session_state.queued_followup = None
if "last_vec_ms" not in st.session_state:
    st.session_state.last_vec_ms = 0.0
if "total_tokens_est" not in st.session_state:
    st.session_state.total_tokens_est = 0
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "flashcards_data" not in st.session_state:
    st.session_state.flashcards_data = []
if "agent_terminal_logs" not in st.session_state:
    st.session_state.agent_terminal_logs = [
        "[SYS_BOOT] DeepRAG Neural Workspace Initialized.",
        "[FAISS] Vector Engine Ready (CPU FlatL2).",
        "[GEMINI] Auto-Adaptive Model Router connected to v1beta."
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
    st.markdown('<div class="status-pill">🟢 Neural Engine: Connected & Active</div>', unsafe_allow_html=True)

    st.markdown("#### 🛠️ Neural Hyperparameters")
    chunk_size = st.slider("Chunk Size (Characters)", min_value=200, max_value=1500, value=600, step=50)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=100, step=25)
    top_k = st.slider("Retrieved Citations (Top-K)", min_value=2, max_value=8, value=4, step=1)
    llm_temp = st.slider("LLM Temperature (Creativity)", min_value=0.0, max_value=0.9, value=0.3, step=0.1)

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

    # --- MAIN PAGE PDF UPLOAD PORTAL & ANSWER PREFERENCES ---
    st.markdown(
        "<div class='upload-portal-card'>"
        "  <div class='upload-portal-title'>📂 Document Upload & Answer Preferences</div>"
        "  <div class='upload-portal-sub'>Upload one or more PDF documents below and customize how you want your AI answers formatted.</div>"
        "</div>",
        unsafe_allow_html=True
    )

    ucol1, ucol2 = st.columns([1.4, 1])
    with ucol1:
        uploaded_files = st.file_uploader(
            "📄 Upload Your PDF Document(s)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload textbooks, notes, research papers, or manuals in PDF format."
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

    # --- AUTOMATIC PDF INDEXING WITH PAGE METADATA ---
    if uploaded_files:
        current_key = f"{[f.name for f in uploaded_files]}_{chunk_size}_{chunk_overlap}"
        if current_key != st.session_state.processed_files_key:
            with st.spinner("🧬 Indexing PDF pages into FAISS Neural Vector Space..."):
                t0 = time.time()
                log_terminal(f"Ingesting {len(uploaded_files)} PDF file(s)...")
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
                    st.session_state.last_vec_ms = round((time.time() - t0) * 1000, 1)
                    st.session_state.total_tokens_est = sum(len(d.page_content.split()) for d in split_docs)
                    st.session_state.processed_files_key = current_key
                    log_terminal(f"Indexed {total_pages} pages into {len(split_docs)} chunks successfully.")
                else:
                    st.warning("No extractable text found in the uploaded PDF. Please upload a text-based PDF document.")
    else:
        st.session_state.vector_store = None
        st.session_state.pdf_pages = 0
        st.session_state.neural_chunks = 0
        st.session_state.last_vec_ms = 0.0
        st.session_state.total_tokens_est = 0
        st.session_state.processed_files_key = ""

    doc_label = f"{len(uploaded_files)} Active PDF(s)" if uploaded_files else "No PDF Uploaded"
    st.markdown(
        f"<div class='telemetry-strip'>"
        f"  <div class='tel-item'><span class='tel-dot'></span><span>Corpus: <span class='tel-val'>{doc_label}</span></span></div>"
        f"  <div class='tel-item'><span>🧬 Embeddings: <span class='tel-val'>384D MiniLM-L6-v2</span></span></div>"
        f"  <div class='tel-item'><span>⚡ Vector Latency: <span class='tel-val'>{st.session_state.last_vec_ms} ms</span></span></div>"
        f"  <div class='tel-item'><span>📊 Indexed Words: <span class='tel-val'>{st.session_state.total_tokens_est}</span></span></div>"
        f"</div>",
        unsafe_allow_html=True
    )

    with st.expander("💻 Live Neural Engine Logs & System Telemetry"):
        terminal_lines = "<br>".join([f"&gt; {line}" for line in st.session_state.agent_terminal_logs])
        st.markdown(f"<div class='terminal-box'>{terminal_lines}</div>", unsafe_allow_html=True)

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
                triggered_query = "Provide a comprehensive executive summary of the uploaded document, highlighting its core thesis, key sections, and main conclusions."
        with qcol2:
            if st.button("🔑 Key Formulas & Facts", use_container_width=True):
                triggered_query = "Extract all critical definitions, core concepts, frameworks, formulas, and key technical facts from the document."
        with qcol3:
            if st.button("🗺️ Concept Flowchart", use_container_width=True):
                triggered_query = (
                    "Create a clear, visual step-by-step Concept Flowchart & Mind-Map of the core architecture or process in this document. "
                    "Include a clean Unicode box-drawing diagram inside a code block followed by stage-by-stage explanations with page citations."
                )
        with qcol4:
            if st.button("⚖️ Compare Documents", use_container_width=True):
                triggered_query = (
                    "Perform a structured Cross-Section / Multi-PDF Comparison of the key topics, methodologies, strengths, and conclusions across the indexed pages. "
                    "Present a clean Markdown Comparison Table followed by key analytical takeaways."
                )
        with qcol5:
            if st.button("🎓 Top 5 Exam Questions", use_container_width=True):
                triggered_query = "Generate 5 high-value exam or interview questions along with detailed model answers strictly based on this document."

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
                    transcribed_text = transcribe_voice_with_gemini(API_KEY, audio_bytes, mime_type)
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
                log_terminal(f"Executing similarity search for query: '{active_query}' (Top-K={top_k})...")
                with st.spinner("🧠 Querying FAISS Vector Space & synthesizing verified response..."):
                    try:
                        v_t0 = time.time()
                        docs_and_scores = st.session_state.vector_store.similarity_search_with_score(active_query, k=top_k)
                        vec_ms = round((time.time() - v_t0) * 1000, 2)
                        st.session_state.last_vec_ms = vec_ms
                        log_terminal(f"FAISS vector search completed in {vec_ms}ms.")

                        context_parts = []
                        citations_list = []
                        score_percentages = []

                        for idx, (doc, l2_score) in enumerate(docs_and_scores):
                            src = doc.metadata.get("source", "Document.pdf")
                            pg = doc.metadata.get("page", "?")
                            raw_l2 = round(float(l2_score), 4)
                            match_pct = max(65, min(99, int(round(100 / (1 + float(l2_score) * 0.35)))))
                            score_percentages.append(match_pct)

                            context_parts.append(f"[Source {idx+1}: {src}, Page {pg}]\n{doc.page_content}")
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
                            "⚡ Deep Analysis (Default)": "Provide a well-structured and authoritative explanation in English.",
                            "🎯 Exam & Revision Notes": "Format the response as high-impact study notes with bold key terms, structured bullet points, and clear definitions in English.",
                            "💡 Simple Explain (ELI5)": "Explain the concept in clear, beginner-friendly English using an intuitive real-world analogy.",
                            "📌 Executive Bullet Points": "Deliver a crisp executive summary in high-impact bullet points in English."
                        }[response_mode]

                        length_instructions = {
                            "⚡ Short & Crisp": "Keep the answer concise, direct, and brief (around 60 to 100 words, 3 to 5 crisp sentences or bullet points).",
                            "⚖️ Balanced": "Provide a balanced, medium-length explanation (around 150 to 250 words) covering the core points clearly.",
                            "📚 Detailed & In-Depth": "Provide a comprehensive, exhaustive, and deeply detailed explanation (350+ words) with thorough breakdowns, sub-headings, and complete technical depth."
                        }[answer_length]

                        final_prompt = (
                            "You are DeepRAG AI, an elite enterprise and academic document intelligence assistant.\n"
                            "Answer the user's question strictly in English using ONLY the provided PDF Context below.\n"
                            "If the answer cannot be found within the provided context, state clearly that the information is not present in the uploaded documents.\n"
                            "Always cite the relevant page numbers inline (for example, [Page 4]) when stating facts.\n\n"
                            "IMPORTANT: At the very end of your response, after your main answer, write the exact delimiter ---FOLLOWUP--- "
                            "followed by 3 short, insightful follow-up questions separated by the pipe character | (each question under 9 words, no numbering).\n\n"
                            f"Output Style Instruction: {style_instructions}\n"
                            f"Answer Length Instruction: {length_instructions}\n\n"
                            f"Recent Conversation Context:\n{history_context}\n\n"
                            f"Retrieved PDF Context:\n{context_block}\n\n"
                            f"User Question: {active_query}\n"
                        )

                        log_terminal(f"Routing grounded prompt to Gemini ({llm_temp} temp, {answer_length})...")
                        raw_output, used_model = generate_smart_gemini_response(API_KEY, final_prompt, temperature=llm_temp)
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

    # --- DISPLAY PREVIOUS CONVERSATION HISTORY ---
    for idx, turn in enumerate(previous_turns):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(f"**{turn['question']}**")

        with st.chat_message("assistant", avatar="🧠"):
            is_top_card = (idx == 0 and not active_query)
            render_assistant_turn(turn, f"hist_{idx}", is_latest=is_top_card)

# ==========================================
# TAB 2: AI QUIZ & 3D INTERACTIVE FLASHCARDS
# ==========================================
with tab_quiz:
    st.markdown(render_animated_title("AI Quiz & 3D Flashcards"), unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Master your uploaded documents with interactive 3D flip flashcards and AI-graded multiple-choice practice quizzes.</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.vector_store:
        st.info("📂 Please upload your PDF document in the '🚀 AI Workspace' tab first to generate flashcards and quizzes.")
    else:
        scol1, scol2 = st.columns(2)
        with scol1:
            if st.button("🃏 Generate 4 Interactive 3D Flashcards", use_container_width=True):
                log_terminal("Generating 3D interactive flashcards from vector store...")
                with st.spinner("✨ Crafting 3D Flip Flashcards from vector chunks..."):
                    try:
                        st.session_state.flashcards_data = generate_3d_flashcards(API_KEY, st.session_state.vector_store)
                        log_terminal("3D Flashcards generated successfully.")
                    except Exception as e:
                        log_terminal(f"Flashcard generation error: {e}")
                        st.error(f"Could not generate flashcards: {e}")
        with scol2:
            if st.button("🎯 Generate New 3-Question AI Quiz", use_container_width=True):
                log_terminal("Generating interactive multiple-choice quiz...")
                with st.spinner("🧠 Synthesizing quiz questions & verified explanations..."):
                    try:
                        st.session_state.quiz_data = generate_interactive_quiz(API_KEY, st.session_state.vector_store)
                        log_terminal("AI Quiz generated successfully.")
                    except Exception as e:
                        log_terminal(f"Quiz generation error: {e}")
                        st.error(f"Could not generate quiz: {e}")

        if st.session_state.flashcards_data:
            st.markdown("##### 🃏 Interactive 3D Study Flashcards *(Hover or Tap any card to flip)*")
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
                    f"      <span class='step-badge'>Flashcard 0{fc_idx+1} • {pg_txt}</span>"
                    f"      <div style='font-family:Space Grotesk,sans-serif; font-size:1.18rem; font-weight:700; margin-top:8px;'>{term_txt}</div>"
                    f"      <div style='color:#94a3b8; font-size:0.76rem; margin-top:14px;'>🔄 Hover or Tap to Reveal Answer</div>"
                    f"    </div>"
                    f"    <div class='flashcard-back'>"
                    f"      <span class='step-badge' style='background:linear-gradient(135deg,#10b981,#059669);'>Verified Answer • {pg_txt}</span>"
                    f"      <div style='font-size:0.92rem; line-height:1.5; margin-top:8px;'>{def_txt}</div>"
                    f"    </div>"
                    f"  </div>"
                    f"</div>"
                )
            cards_html.append("</div>")
            st.markdown("".join(cards_html), unsafe_allow_html=True)
            st.write("")

        if st.session_state.quiz_data:
            st.markdown("##### 🎯 Interactive Knowledge Check")
            user_picks = {}
            for q_idx, q_item in enumerate(st.session_state.quiz_data):
                st.markdown(
                    f"<div class='quiz-card'><strong>Q{q_idx+1}. {q_item.get('question', '')}</strong></div>",
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

            if st.button("✅ Submit & Grade My Quiz", use_container_width=True):
                score = 0
                total_q = len(st.session_state.quiz_data)
                for q_idx, q_item in enumerate(st.session_state.quiz_data):
                    opts = q_item.get("options", [])
                    correct_i = int(q_item.get("correct_index", 0))
                    correct_opt = opts[correct_i] if 0 <= correct_i < len(opts) else opts[0]
                    picked = user_picks.get(q_idx)
                    if picked == correct_opt:
                        score += 1
                        st.success(f"✔️ **Q{q_idx+1} Correct!** ({picked}) — {q_item.get('explanation', '')}")
                    else:
                        st.error(f"❌ **Q{q_idx+1} Incorrect.** Correct Answer: **{correct_opt}** — {q_item.get('explanation', '')}")

                st.markdown(f"### 🏆 Final Score: **{score} / {total_q}**")
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
        '<div class="arch-desc">Uploaded PDF files are processed through a high-speed <code>pypdf</code> extraction pipeline. Every single page is isolated, cleaned, and tagged with its exact source filename and page number so downstream answers retain 100% verifiable traceability.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 02 • Semantic Splitting</span>'
        '<div class="arch-title">🧩 Context-Aware Recursive Chunking</div>'
        '<div class="arch-desc">Using LangChain\'s <code>RecursiveCharacterTextSplitter</code>, long chapters are partitioned into overlapping semantic windows. Dynamic chunk overlap ensures sentences and technical arguments are never severed mid-thought.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 03 • Neural Encoding</span>'
        '<div class="arch-title">🧬 384-Dimensional Dense Vector Embeddings</div>'
        '<div class="arch-desc">Each text chunk is passed through HuggingFace\'s <code>all-MiniLM-L6-v2</code> Sentence-Transformer model, mapping complex concepts into a 384-dimensional dense geometric vector space where meaning is measured by spatial proximity.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 04 • Similarity Search</span>'
        '<div class="arch-title">⚡ Millisecond FAISS Vector Indexing & Confidence Scoring</div>'
        '<div class="arch-desc">Facebook AI Similarity Search (<code>FAISS</code>) indexes all neural vectors in memory. When you ask a question via text or voice, your query is embedded in real time and matched against the top-K passages with exact L2 semantic confidence percentages.</div>'
        '</div>'
        '<div class="arch-card">'
        '<span class="step-badge">Stage 05 • Grounded Synthesis</span>'
        '<div class="arch-title">🤖 Auto-Adaptive Gemini 3.0 Flash Synthesis</div>'
        '<div class="arch-desc">The retrieved passages, page numbers, and conversational memory are injected into a strict grounding prompt and routed to Google\'s <code>gemini-3.0-flash</code> engine—producing fast, accurate, citation-backed responses with zero hallucination.</div>'
        '</div>',
        unsafe_allow_html=True
    )

# ==========================================
# TAB 4: ANIMATED ABOUT CREATOR
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
