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
SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/all-mpnet-base-v2'
TOP_K = 5

# --- Load Models and Data ---
@st.cache_resource
def load_sentence_model():
    try:
        model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
        return model
    except Exception as e:
        st.error(f"Error loading sentence model: {e}")
        return None

@st.cache_resource
def load_llm_pipeline():
    try:
        model_id = "google/gemma-2b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")  # For GPU
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        return llm
    except Exception as e:
        st.error(f"Error loading language model: {e}")
        return None

@st.cache_data
def load_faiss_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        st.error(f"FAISS index file not found at {FAISS_INDEX_PATH}")
        return None
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        return index
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

@st.cache_data
def load_metadata():
    if not os.path.exists(METADATA_PATH):
        st.error(f"Metadata file not found at {METADATA_PATH}")
        return []
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return []

@st.cache_data
def load_circular_details():
    if not os.path.exists(CIRCULAR_DATA_PATH):
        st.error(f"Circular data file not found at {CIRCULAR_DATA_PATH}")
        return {}
    try:
        with open(CIRCULAR_DATA_PATH, 'r', encoding='utf-8') as f:
            all_circulars = json.load(f)
        circular_map = {}
        for circular in all_circulars:
            serial_no = circular.get('serial_no')
            if serial_no:
                pdf_id_en = f"{serial_no}_en"
                circular_map[pdf_id_en] = {
                    "title": circular.get("title", "Title not available"),
                    "url": circular.get("english_pdf_link", "#")
                }
        return circular_map
    except Exception as e:
        st.error(f"Error loading circular details: {e}")
        return {}

# --- FAISS Search ---
def search_faiss(query_text, faiss_index, sentence_model, k=TOP_K):
    if not query_text or faiss_index is None or sentence_model is None:
        return [], []
    try:
        query_embedding = sentence_model.encode([query_text])
        distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), k)
        return distances[0], indices[0]
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return [], []

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("EPFO Circulars Question Answering App 💬")
st.markdown("""
This app allows you to search through EPFO circulars indexed using FAISS.
Enter your question to find relevant sections and receive an AI-generated answer.
""")

# Load resources
sentence_model = load_sentence_model()
llm_pipeline = load_llm_pipeline()
faiss_index = load_faiss_index()
metadata_chunks = load_metadata()
circular_details_map = load_circular_details()

if not sentence_model or not faiss_index or not metadata_chunks or not circular_details_map:
    st.error("One or more required resources failed to load. Check file paths and setup.")
    st.stop()

query = st.text_input("Enter your question:", placeholder="e.g., What are the new rules for PF withdrawal?")

if st.button("Search 🔍"):
    if query:
        with st.spinner("Searching for relevant circulars..."):
            distances, indices = search_faiss(query, faiss_index, sentence_model)

        if not indices.size:
            st.warning("No results found. Try rephrasing your query.")
        else:
            st.subheader(f"Top {len(indices)} Results:")
            results_to_display = []

            for i, idx in enumerate(indices):
                if idx < 0 or idx >= len(metadata_chunks):
                    continue
                chunk = metadata_chunks[idx]
                text_to_embed = chunk.get('text_to_embed', 'Text not available.')
                source_pdf_id = chunk.get('source_pdf_id', 'Unknown PDF ID')
                source_page_num = chunk.get('source_page_num', 'N/A')
                circular_info = circular_details_map.get(source_pdf_id, {})
                title = circular_info.get("title", f"Circular (ID: {source_pdf_id})")
                pdf_url = circular_info.get("url", "#")

                results_to_display.append({
                    "title": title,
                    "pdf_url": pdf_url,
                    "page_num": source_page_num + 1 if isinstance(source_page_num, int) else source_page_num,
                    "text": text_to_embed,
                    "distance": distances[i],
                    "chunk_id": chunk.get('chunk_id', 'N/A')
                })

            # Display results
            num_columns = min(len(results_to_display), 3)
            cols = st.columns(num_columns)

            for i, res in enumerate(results_to_display):
                col_to_use = cols[i % num_columns]
                with col_to_use:
                    container = st.container()
                    container.markdown(f"##### {i+1}. {res['title']}")
                    if res['pdf_url'] and res['pdf_url'] != "#":
                        container.markdown(f"📄 [View PDF (Page: {res['page_num']})]({res['pdf_url']}#page={res['page_num']})", unsafe_allow_html=True)
                    else:
                        container.markdown(f"*Page: {res['page_num']} (PDF link not available)*")
                    container.markdown(f"<small>Relevance Score (distance): {res['distance']:.4f} | Chunk ID: {res['chunk_id']}</small>", unsafe_allow_html=True)
                    with container.expander("Show relevant text", expanded=False):
                        st.text_area(label=f"Relevant text content {i+1}", value=res['text'], height=200, key=f"text_area_{i}_{res['chunk_id']}", label_visibility="collapsed")
                    container.markdown("---")

            # Generate answer
            if llm_pipeline:
                context = "\n".join([res['text'] for res in results_to_display])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
                with st.spinner("Generating answer using Gemma..."):
                    try:
                        response = llm_pipeline(prompt)[0]['generated_text']
                        answer = response.split("Answer:")[-1].strip()
                        st.subheader("💡 LLM-Generated Answer")
                        st.write(answer)
                    except Exception as e:
                        st.warning(f"LLM failed to generate a response: {e}")
            else:
                st.info("LLM not available or failed to load.")
    else:
        st.warning("Please enter a question to search.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info("""
This app uses:
- FAISS for vector similarity search
- SentenceTransformer for multilingual embeddings
- `google/gemma-2b-it` for answer generation
""")
