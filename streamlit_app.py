import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- Configuration ---
FAISS_INDEX_PATH = 'circulars_faiss_ocr.index'
METADATA_PATH = 'circulars_faiss_ocr.index.meta.json'
CIRCULAR_DATA_PATH = 'circular-data.json'

SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'  # Use only one

TOP_K = 5

# --- Load Models and Data ---
@st.cache_resource
def load_sentence_model():
    """Load the sentence transformer embedding model."""
    try:
        return SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
    except Exception as e:
        st.error(f"Error loading sentence embedding model: {e}")
        return None

@st.cache_resource
def load_llm_pipeline():
    """Load a lightweight LLM pipeline (BART-base)."""
    try:
        model_id = "facebook/bart-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        return None

@st.cache_data
def load_faiss_index():
    """Load FAISS index from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index file not found: {FAISS_INDEX_PATH}")
        return None
    try:
        return faiss.read_index(FAISS_INDEX_PATH)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_data
def load_metadata():
    """Load chunk-level metadata."""
    if not os.path.exists(METADATA_PATH):
        st.error(f"Metadata file not found: {METADATA_PATH}")
        return []
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return []

@st.cache_data
def load_circular_details():
    """Load PDF title and URL mapping."""
    if not os.path.exists(CIRCULAR_DATA_PATH):
        st.error(f"Circular data file not found: {CIRCULAR_DATA_PATH}")
        return {}
    try:
        with open(CIRCULAR_DATA_PATH, 'r', encoding='utf-8') as f:
            all_circulars = json.load(f)

        circular_map = {}
        for circular in all_circulars:
            serial_no = circular.get('serial_no')
            if serial_no:
                pdf_id = f"{serial_no}_en"
                circular_map[pdf_id] = {
                    "title": circular.get("title", "Untitled Circular"),
                    "url": circular.get("english_pdf_link", "#")
                }
        return circular_map
    except Exception as e:
        st.error(f"Error loading circular details: {e}")
        return {}

# --- FAISS Search ---
def search_faiss(query_text, faiss_index, sentence_model, k=TOP_K):
    """Search top-k similar chunks using FAISS."""
    if not query_text or faiss_index is None or sentence_model is None:
        return [], []
    try:
        query_embedding = sentence_model.encode([query_text])
        distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), k)
        return distances[0], indices[0]
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return [], []

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("EPFO Circulars Question Answering App 💬")
st.markdown("""
Search through EPFO circulars indexed using FAISS.  
Enter your question to get relevant excerpts and a generated answer.
""")

# Load components
sentence_model = load_sentence_model()
llm_pipeline = load_llm_pipeline()
faiss_index = load_faiss_index()
metadata_chunks = load_metadata()
circular_details_map = load_circular_details()

if not all([sentence_model, llm_pipeline, faiss_index, metadata_chunks, circular_details_map]):
    st.error("Essential components failed to load. Check logs and file paths.")
    st.stop()

# --- Query Section ---
query = st.text_input("Enter your question:", placeholder="e.g., What are the new rules for PF withdrawal?")

if st.button("Search 🔍"):
    if not query.strip():
        st.warning("Please enter a question to search.")
    else:
        with st.spinner("Searching FAISS index..."):
            distances, indices = search_faiss(query, faiss_index, sentence_model)

        if not indices.size:
            st.warning("No results found. Try rephrasing.")
        else:
            st.subheader("🔍 Top Results")
            results_to_display = []
            for i, idx in enumerate(indices):
                if idx < 0 or idx >= len(metadata_chunks):
                    continue
                chunk = metadata_chunks[idx]
                text = chunk.get('text_to_embed', 'Text not available.')
                pdf_id = chunk.get('source_pdf_id', 'Unknown')
                page_num = chunk.get('source_page_num', 'N/A')
                info = circular_details_map.get(pdf_id, {})
                results_to_display.append({
                    "title": info.get("title", f"PDF ID: {pdf_id}"),
                    "url": info.get("url", "#"),
                    "page": page_num + 1 if isinstance(page_num, int) else page_num,
                    "text": text,
                    "distance": distances[i],
                    "chunk_id": chunk.get('chunk_id', 'N/A')
                })

            # Show results
            cols = st.columns(min(len(results_to_display), 3))
            for i, res in enumerate(results_to_display):
                col = cols[i % len(cols)]
                with col:
                    st.markdown(f"**{i+1}. {res['title']}**")
                    if res["url"] != "#":
                        st.markdown(f"[View PDF (Page {res['page']})]({res['url']}#page={res['page']})", unsafe_allow_html=True)
                    else:
                        st.markdown(f"*Page: {res['page']} (no link available)*")
                    st.markdown(f"<small>Score: {res['distance']:.4f} | Chunk ID: {res['chunk_id']}</small>", unsafe_allow_html=True)
                    with st.expander("Show relevant text"):
                        st.text_area("", value=res['text'], height=200, key=f"ta_{i}", label_visibility="collapsed")

            # Generate Answer
            context = "\n".join([res['text'] for res in results_to_display])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
            with st.spinner("Generating answer using LLM..."):
                try:
                    result = llm_pipeline(prompt)[0]['generated_text']
                    answer = result.split("Answer:")[-1].strip()
                    st.subheader("💡 LLM-Generated Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses:
- FAISS for vector search
- `all-mpnet-base-v2` for embeddings
- `facebook/bart-base` (CPU-friendly) for answer generation
""")
