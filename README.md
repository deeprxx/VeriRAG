<div align="center">

<!-- DYNAMIC ANIMATED HEADER WAVE & TITLE -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0284c7,50:4f46e5,100:7c3aed&height=230&section=header&text=DeepRAG%20AI%20Workspace&fontSize=56&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Enterprise%20Neural%20RAG%20%E2%80%A2%20Multimodal%20Voice%20AI%20%E2%80%A2%20384D%20FAISS%20Vector%20Engine&descAlignY=58&descSize=18" width="100%" alt="DeepRAG Header" />

<!-- ANIMATED TYPING SVG SUBTITLE -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=22&duration=3000&pause=800&color=38BDF8&center=true&vCenter=true&width=850&lines=%F0%9F%A7%A0+Zero-Hallucination+PDF+Intelligence+with+Exact+Page+Citations;%F0%9F%8E%99%EF%B8%8F+Multimodal+Voice+Interrogation+%26+Neural+Audio+Playback+Deck;%F0%9F%83%8F+Interactive+3D+Flip+Flashcards+%26+Auto-Graded+AI+Quizzes;%F0%9F%94%AC+Sub-15ms+FAISS+L2+Vector+Search+%26+X-Ray+Telemetry+Inspector" alt="Typing SVG" />
</a>

<br/>

<!-- TECH STACK & LIVE STATUS BADGES -->
[![Live App](https://img.shields.io/badge/🚀_Live_Workspace-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://rirag-ai.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-384D_Vector_Index-0081FB?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-Auto_Adaptive_Router-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-MiniLM_L6_v2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Telegram](https://img.shields.io/badge/Contact-@DeepRAGAI-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/DeepRAGAI)

</div>

---

## ⚡ Overview: Why DeepRAG AI Workspace?

Standard Large Language Models rely on static parametric memory, leading to **hallucinations** and an inability to verify facts across private documents. **DeepRAG AI Workspace** solves this by pairing **non-parametric dense vector retrieval (FAISS)** with **auto-adaptive Gemini synthesis** inside a zero-lag, adaptive Dark/Light glassmorphic interface.

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  🧠 DEEPRAG AI WORKSPACE                                         [🌑 Dark / ☀️ Light]   │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  [🚀 AI Workspace]   [🎓 AI Quiz & 3D Flashcards]   [⚡ How It Works]   [👨‍💻 Creator]     │
├──────────────────────────────────────────┬───────────────────────────────────────────────┤
│  📂 Upload Your PDF Document(s)          │  🎛️ Output Style: [⚡ Deep Analysis        ▼] │
│  [ + Drag & Drop Multiple PDFs Here ]    │  📏 Length: ( ) Short  (•) Balanced  ( ) Deep │
├──────────────────────────────────────────┴───────────────────────────────────────────────┤
│  📄 INDEXED PDF PAGES: 42       🧩 NEURAL VECTOR CHUNKS: 186      ⚡ ENGINE: FAISS Active │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  ✨ [📝 Summary]  [🔑 Formulas]  [🗺️ Concept Flowchart]  [⚖️ Compare]  [🎓 Top 5 Q&A]    │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  🧠 Verified Synthesis [Page 4] [Page 12]                                                │
│  ┌────────────────────────────────────────────────────────────────────────────────────┐  │
│  │ 🔊 [⏸️ Pause Audio]  ▂▅▇▃  [⏹️ Stop]  [⚡ 1.25x]  [📋 Copy]   🎯 96% Semantic Match │  │
│  └────────────────────────────────────────────────────────────────────────────────────┘  │
│  ▸ 📚 View Verified PDF Page Citations & Semantic Confidence Bars                        │
│  ▸ 🔬 X-Ray RAG Telemetry & Vector Space Inspector (L2 Distance Matrix)                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ 1. End-to-End Neural RAG Architecture

```mermaid
flowchart TB
    subgraph INGEST ["📂 STAGE 1: MULTI-DOCUMENT INGESTION & CHUNKING"]
        direction LR
        A["📄 Multi-PDF Upload Portal"] --> B["🔍 PyPDF Page Parser\nExtracts Text + Page Metadata"]
        B --> C["🧩 RecursiveCharacterTextSplitter\n600-Char Window • 100-Char Overlap"]
    end

    subgraph VECTOR ["🧬 STAGE 2: 384D NEURAL ENCODING & FAISS INDEXING"]
        direction LR
        C --> D["🧠 HuggingFace Sentence-Transformer\nall-MiniLM-L6-v2 Encoder"]
        D --> E[("⚡ FAISS FlatL2 Vector DB\n384D Float32 RAM Index")]
    end

    subgraph QUERY ["🎙️ STAGE 3: MULTIMODAL INTERROGATION & RETRIEVAL"]
        direction LR
        F["🧑‍💻 User Text Prompt\nor 🎙️ Voice Recording"] --> G["🎧 Gemini Audio Transcription\n& Query Vectorizer"]
        G --> H["📐 Exact L2 Distance Search\nTop-K Passages + Confidence %"]
        E --> H
    end

    subgraph SYNTH ["🤖 STAGE 4: GROUNDED SYNTHESIS & INTERACTIVE STUDY HUB"]
        direction LR
        H --> I["🛡️ Deterministic Grounding Prompt\nStyle + Length Controller"]
        I --> J["🚀 Auto-Adaptive Gemini Router\n3.0-Flash / 2.5-Flash / Pro"]
        J --> K["✅ Verified Answer + Audio Deck\n+ X-Ray Inspector + 3D Flashcards"]
    end

    style INGEST fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    style VECTOR fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc
    style QUERY fill:#091e1a,stroke:#10b981,stroke-width:2px,color:#f8fafc
    style SYNTH fill:#2e1065,stroke:#c084fc,stroke-width:2px,color:#f8fafc
```

---

## 🔄 2. Real-Time Execution Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as 🧑‍💻 User
    participant UI as 🎨 Streamlit Glass UI
    participant Audio as 🎙️ Gemini Voice Node
    participant Emb as 🧬 MiniLM-L6-v2 (384D)
    participant FAISS as ⚡ FAISS FlatL2 DB
    participant Router as 🤖 Adaptive Gemini Router

    User->>UI: Uploads PDF(s) & Sets Style + Answer Length
    UI->>Emb: Splits Pages (600 chars, 100 overlap) & Encodes Chunks
    Emb->>FAISS: Indexes 384D Float32 Vectors in RAM (<15ms)
    
    alt Voice Query Recorded
        User->>UI: Records Question via Microphone
        UI->>Audio: Sends Base64 Audio Stream
        Audio-->>UI: Returns Transcribed English Query
    else Text Query Typed
        User->>UI: Submits Question or Clicks Instant Visualizer
    end

    UI->>UI: Immediately Displays Question & Starts Live Spinner
    UI->>Emb: Encodes Query into 384D Vector E(q)
    Emb->>FAISS: Computes Exact L2 Euclidean Distance
    FAISS-->>UI: Returns Top-K Chunks + Page Metadata + Match %
    UI->>Router: Injects Context + History + Length Rules
    Router-->>UI: Synthesizes Grounded Answer + 3 Smart Follow-Ups
    UI-->>User: Renders Answer, Audio Deck, Citation Bars & X-Ray Matrix
```

---

## 🧩 3. Complete Feature Ecosystem

```mermaid
mindmap
  root((🧠 DeepRAG AI))
    📂 Document Intelligence
      Multi-PDF Simultaneous Ingestion
      Page-Level Metadata Traceability
      Context-Aware Sliding Window Overlap
    🎛️ Precision Controls
      4 Synthesis Modes: Deep, Exam, ELI5, Executive
      3 Answer Lengths: Short, Balanced, Detailed
      Dynamic Chunk Size, Overlap, Top-K & Temperature
    🎙️ Multimodal Audio Engine
      Voice-to-Text Query Interrogation
      Hindi/Hinglish to English Auto-Translation
      Neural Audio Deck with Live 4-Bar Equalizer
      1.0x / 1.25x / 1.5x Playback Speed & 1-Click Copy
    🎓 Interactive Study Hub
      3D CSS Flip Study Flashcards
      Auto-Graded Multiple-Choice AI Quizzes
      1-Click Concept Flowcharts & Cross-PDF Tables
    🔬 Developer X-Ray & UI
      FAISS L2 Distance & Token Telemetry
      Sidebar Live Neural Engine Terminal Logs
      Auto-Adaptive Dark Anime & Frosted Light Themes
```

---

## 📐 Mathematical Formulation: Dense Vector Retrieval

### 1. Semantic Vector Encoding
Every extracted document chunk $d_i \in \mathcal{D}$ and user query $q$ is projected into a 384-dimensional dense vector space using the `all-MiniLM-L6-v2` transformer encoder $E: \mathcal{T} \rightarrow \mathbb{R}^{384}$:

$$\mathbf{v}_q = E(q), \quad \mathbf{v}_{d_i} = E(d_i) \in \mathbb{R}^{384}$$

### 2. FAISS FlatL2 Distance & Cosine Similarity
FAISS indexes normalized 384D vectors and computes the squared Euclidean ($L_2$) distance across all chunks in RAM:

$$d_{L_2}(\mathbf{v}_q, \mathbf{v}_{d_i}) = \left\Vert{} \mathbf{v}_q - \mathbf{v}_{d_i} \right\Vert{}_2^2 = \sum_{k=1}^{384} \left( v_{q,k} - v_{d_i,k} \right)^2$$

### 3. Semantic Match Confidence Calibration
To provide transparent verification in the UI citation bars and X-Ray Inspector, raw $L_2$ distances are mapped to a bounded semantic confidence percentage:

$$\text{Confidence}(\%) = \text{clip}\left( \text{round}\left( \frac{100}{1 + 0.35 \cdot d_{L_2}(\mathbf{v}_q, \mathbf{v}_{d_i})} \right), \, 65, \, 99 \right)$$

---

## 🔥 Core Engineering Highlights

| Module | Feature | Technical Implementation |
| :--- | :--- | :--- |
| **📂 Main-Page Portal** | Intuitive PDF Upload & Length Control | Direct main-view uploader with `Short & Crisp`, `Balanced`, and `Detailed` radio controllers |
| **🌗 Adaptive Theme Engine** | Auto-Sync Dark & Light Modes | Real-time DOM computed-style observer switching between 60FPS Anime Night Sky & Frosted Light UI |
| **🎧 Neural Audio Deck** | Interactive Speech Synthesis | Browser `speechSynthesis` deck with Play/Pause/Resume, animated CSS soundwave bars & speed toggle |
| **🔬 X-Ray RAG Inspector** | Transparent Vector Telemetry | Displays per-chunk $L_2$ distances, retrieval latency (`ms`), injected token counts, and keyword highlights |
| **🃏 3D Study Hub** | Active Recall & Self-Testing | Structured JSON prompting generating 180° 3D CSS flip flashcards and auto-graded MCQ quizzes |
| **🛡️ Auto-Adaptive Router** | Zero-Downtime Gemini Routing | Dynamically queries `/v1beta/models` at runtime and falls back across active Flash/Pro models |

---

## 🚀 Local Installation & Quickstart

### 1. Clone the Repository
```bash
git clone [https://github.com/deeprxx/VeriRAG.git](https://github.com/deeprxx/VeriRAG.git)
cd VeriRAG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Your Gemini API Key
Create `.streamlit/secrets.toml` in your project root directory:
```toml
GEMINI_API_KEY = "your_google_gemini_api_key_here"
```

### 4. Launch the Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
VeriRAG/
├── app.py                  # Core application: Adaptive UI, FAISS RAG Engine, Voice AI & 3D Study Hub
├── requirements.txt        # Production Python dependencies for Streamlit Cloud
└── README.md               # System architecture, mathematical formulation & documentation
```

---

## 👨‍💻 About The Creator

<div align="center">

### **Deep Patel**
**Founder, Lead AI Architect & Creator of DeepRAG AI Workspace**

> *"My mission is simple: transform complex artificial intelligence and neural retrieval pipelines into a fast, reliable, and visually breathtaking workspace that empowers students, researchers, and innovators."*

[![Telegram](https://img.shields.io/badge/Telegram-@DeepRAGAI-26A5E4?style=for-the-badge&logo=telegram&logoColor=white)](https://t.me/DeepRAGAI)

</div>

<!-- DYNAMIC ANIMATED FOOTER WAVE -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0284c7,50:4f46e5,100:7c3aed&height=110&section=footer" width="100%" alt="Footer Wave" />
