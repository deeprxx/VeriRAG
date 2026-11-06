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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #06080f;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 12% 12%, rgba(99, 102, 241, 0.16) 0%, transparent 38%),
                radial-gradient(circle at 88% 18%, rgba(168, 85, 247, 0.14) 0%, transparent 42%),
                radial-gradient(circle at 50% 88%, rgba(16, 185, 129, 0.08) 0%, transparent 45%),
                #06080f;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 24, 0.94) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.25);
    backdrop-filter: blur(20px);
}

/* Status Pill */
.status-pill {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.16), rgba(5, 150, 105, 0.08));
    border: 1px solid rgba(16, 185, 129, 0.45);
    color: #34d399;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.84rem;
    font-weight: 600;
    text-align: center;
    margin-bottom: 18px;
    box-shadow: 0 0 22px rgba(16, 185, 129, 0.15);
    animation: pulseGlow 3s infinite ease-in-out;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(16, 185, 129, 0.15); }
    50% { box-shadow: 0 0 28px rgba(16, 185, 129, 0.35); }
}

/* Animated Letter-by-Letter Title */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.9rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
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
    transform: translateY(14px) scale(0.95);
    animation: fadeUpChar 0.45s cubic-bezier(0.22, 1, 0.36, 1) forwards;
}

@keyframes fadeUpChar {
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.cursor-blink {
    display: inline-block;
    width: 4px;
    height: 40px;
    background: #ec4899;
    margin-left: 8px;
    border-radius: 2px;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0; }
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.04rem;
    margin-bottom: 26px;
    font-weight: 400;
    animation: fadeInSlide 0.8s ease forwards;
}

@keyframes fadeInSlide {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Metric Glass Cards */
.metric-card {
    background: linear-gradient(145deg, rgba(19, 25, 42, 0.88), rgba(12, 17, 30, 0.94));
    border: 1px solid rgba(99, 102, 241, 0.28);
    border-radius: 18px;
    padding: 22px 18px;
    text-align: center;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    animation: fadeInSlide 0.6s ease forwards;
}

.metric-card:hover {
    transform: translateY(-6px);
    border-color: rgba(168, 85, 247, 0.65);
    box-shadow: 0 16px 40px rgba(99, 102, 241, 0.25);
}

.metric-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Custom Tabs */
div[data-testid="stTabs"] button {
    background: rgba(30, 41, 59, 0.55) !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    margin-right: 10px !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stTabs"] button:hover {
    border-color: rgba(129, 140, 248, 0.5) !important;
    transform: translateY(-2px);
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #a855f7) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.45) !important;
}

/* Badges & Citations */
.meta-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.16);
    border: 1px solid rgba(99, 102, 241, 0.42);
    color: #c7d2fe;
    font-size: 0.76rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
    margin-right: 8px;
    margin-top: 8px;
}

.citation-box {
    background: rgba(15, 23, 42, 0.9);
    border-left: 4px solid #8b5cf6;
    border-top: 1px solid rgba(148, 163, 184, 0.12);
    border-right: 1px solid rgba(148, 163, 184, 0.12);
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
    padding: 14px 16px;
    border-radius: 10px;
    margin-bottom: 12px;
    font-size: 0.88rem;
    color: #cbd5e1;
    transition: all 0.25s ease;
}

.citation-box:hover {
    border-left-color: #ec4899;
    background: rgba(23, 33, 56, 0.95);
}

/* Animated Architecture Cards (How It Works Tab) */
.arch-card {
    background: linear-gradient(145deg, rgba(18, 24, 41, 0.9), rgba(11, 16, 28, 0.95));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    opacity: 0;
    transform: translateY(20px);
    animation: fadeUpCard 0.6s ease forwards;
}

@keyframes fadeUpCard {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.arch-card:hover {
    transform: translateY(-5px) scale(1.01);
    border-color: rgba(168, 85, 247, 0.65);
    box-shadow: 0 14px 36px rgba(99, 102, 241, 0.22);
}

.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: #ffffff;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.arch-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
}

.arch-desc {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Animated Creator Showcase Card (About Creator Tab) */
.creator-hero {
    background: linear-gradient(135deg, rgba(20, 27, 45, 0.92), rgba(12, 16, 29, 0.96));
    border: 1px solid rgba(139, 92, 246, 0.38);
    border-radius: 24px;
    padding: 40px 34px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 40px rgba(139, 92, 246, 0.14);
    animation: fadeUpCard 0.65s ease forwards;
    margin-top: 12px;
}

.avatar-ring {
    width: 96px;
    height: 96px;
    border-radius: 50%;
    margin: 0 auto 18px auto;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    box-shadow: 0 0 30px rgba(168, 85, 247, 0.5);
    animation: floatAvatar 3.5s ease-in-out infinite;
}

@keyframes floatAvatar {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
