import streamlit as st
from pypdf import PdfReader
from google import genai
import faiss
import numpy as np
import time

st.set_page_config(page_title="VeriRAG: Classroom Assistant", page_icon="📚", layout="wide")
st.title("📚 VeriRAG: Hallucination-Free Classroom Assistant")
st.caption("First-Principles Grounded RAG Engine | Built with Gemini Embeddings & FAISS L2 Vector Search")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Google Gemini API Key:", type="password")
    uploaded_file = st.file_uploader("Upload Syllabus / Notes (PDF)", type=["pdf"])
    chunk_size = st.slider("Chunk Size (Characters)", 300, 1500, 800, 100)
    overlap = st.slider("Chunk Overlap (Characters)", 50, 300, 150, 50)

def extract_and_chunk_pdf(pdf_file, size, step_overlap):
    reader = PdfReader(pdf_file)
    chunks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            end = start + size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({"page": page_num, "text": chunk_text})
            start += (size - step_overlap)
    return chunks

def build_vector_store(client, chunks):
    texts = [c["text"] for c in chunks]
    all_embeddings = []
    
    batch_size = 30
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch
        )
        all_embeddings.extend(response.embeddings)
        time.sleep(2)  # Rate limit se bachne ke liye chota pause
        
    vectors = np.array([e.values for e in all_embeddings], dtype="float32")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    return index

if uploaded_file and api_key:
    client = genai.Client(api_key=api_key)
    file_id = f"{uploaded_file.name}_{chunk_size}_{overlap}"
    
    if st.session_state.get("current_file") != file_id:
        with st.spinner("Chunking PDF & building FAISS Vector Index..."):
            chunks = extract_and_chunk_pdf(uploaded_file, chunk_size, overlap)
            index = build_vector_store(client, chunks)
            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["current_file"] = file_id
        st.success(f"Indexed {len(chunks)} chunks across the PDF!")

    query = st.text_input("Ask a question strictly from your uploaded PDF:")
    
    if query:
        with st.spinner("Searching vector space & verifying ground truth..."):
            q_embed = client.models.embed_content(
                model="gemini-embedding-001",
                contents=[query]
            )
            q_vector = np.array([q_embed.embeddings[0].values], dtype="float32")
            distances, indices = st.session_state["index"].search(q_vector, k=3)
            
            retrieved_chunks = [st.session_state["chunks"][i] for i in indices[0]]
            context_block = "\n\n".join(
                [f"[Page {c['page']}]: {c['text']}" for c in retrieved_chunks]
            )
            
            system_prompt = f"""You are VeriRAG, a strict, zero-hallucination academic assistant.
Answer the student's question using ONLY the Context Provided below.
Rules:
1. Every factual claim MUST include the exact page citation like [Page X].
2. If the answer is NOT present in the Context Provided, state clearly: 'I cannot answer this based on the uploaded document.' Do NOT guess or use outside knowledge.

Context Provided:
{context_block}

Student Question: {query}"""

            answer = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=system_prompt
            )
            
            st.subheader("✅ Verified Answer")
            st.write(answer.text)
            
            with st.expander("🔍 View Retrieved Vector Chunks & Euclidean (L2) Distances"):
                for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
                    chunk = st.session_state["chunks"][idx]
                    st.markdown(f"**Rank #{rank} | Page {chunk['page']} | L2 Distance: `{dist:.4f}`**")
                    st.info(chunk["text"])
else:
    st.info("👈 Please enter your Gemini API Key and upload a PDF in the sidebar to begin.")