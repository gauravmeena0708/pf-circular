import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os

# --- Configuration ---
FAISS_INDEX_PATH = 'circulars_faiss_ocr.index'
METADATA_PATH = 'circulars_faiss_ocr.index.meta.json'
CIRCULAR_DATA_PATH = 'circular-data.json' # Source of PDF titles and links
SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
TOP_K = 5 # Number of results to display

# --- Load Models and Data ---
@st.cache_resource # For new Streamlit versions, replaces st.cache(allow_output_mutation=True) for models
def load_sentence_model():
    """Loads the Sentence Transformer model."""
    try:
        model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
        return model
    except Exception as e:
        st.error(f"Error loading sentence model: {e}")
        return None

@st.cache_data # For new Streamlit versions, replaces st.cache for data
def load_faiss_index():
    """Loads the FAISS index."""
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
    """Loads the metadata for text chunks."""
    if not os.path.exists(METADATA_PATH):
        st.error(f"Metadata file not found at {METADATA_PATH}")
        return []
    try:
        with open(METADATA_PATH, 'r', encoding='utf-8') as f:
            metadata_chunks = json.load(f)
        return metadata_chunks
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return []

@st.cache_data
def load_circular_details():
    """Loads circular details (titles, URLs) from circular-data.json."""
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
                # The FAISS metadata uses pdf_id like "49_en"
                pdf_id_en = f"{serial_no}_en"
                circular_map[pdf_id_en] = {
                    "title": circular.get("title", "Title not available"),
                    "url": circular.get("english_pdf_link", "#")
                }
        return circular_map
    except Exception as e:
        st.error(f"Error loading circular details: {e}")
        return {}

# --- Search Function ---
def search_faiss(query_text, faiss_index, sentence_model, k=TOP_K):
    """Encodes the query and searches the FAISS index."""
    if not query_text or faiss_index is None or sentence_model is None:
        return [], []
    try:
        query_embedding = sentence_model.encode([query_text])
        distances, indices = faiss_index.search(np.array(query_embedding).astype('float32'), k)
        return distances[0], indices[0] # Return for the single query
    except Exception as e:
        st.error(f"Error during FAISS search: {e}")
        return [], []

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title(" EPFO Circulars Question Answering App 💬")
st.markdown("""
This app allows you to search through EPFO circulars that have been processed and indexed using a FAISS vector database.
Enter your question below to find relevant sections from the circulars.
""")

# Load resources
sentence_model = load_sentence_model()
faiss_index = load_faiss_index()
metadata_chunks = load_metadata()
circular_details_map = load_circular_details()

if not sentence_model or not faiss_index or not metadata_chunks or not circular_details_map:
    st.error("One or more essential components (model, index, metadata) could not be loaded. Please check the file paths and ensure the files exist. See sidebar for setup instructions.")
    st.stop()

# User input
query = st.text_input("Enter your question:", placeholder="e.g., What are the new rules for PF withdrawal?")

if st.button("Search 🔍"):
    if query:
        with st.spinner("Searching for relevant circulars... ⏳"):
            distances, indices = search_faiss(query, faiss_index, sentence_model)

            if not indices.size: 
                st.warning("No results found for your query. Try rephrasing or check your search terms.")
            else:
                st.subheader(f"Top {len(indices)} Results:")
                
                results_to_display = []
                for i, idx in enumerate(indices):
                    if idx < 0 or idx >= len(metadata_chunks):
                        st.warning(f"Invalid index {idx} found from FAISS search. Skipping.")
                        continue
                    
                    chunk = metadata_chunks[idx]
                    text_to_embed = chunk.get('text_to_embed', 'Text not available.')
                    source_pdf_id = chunk.get('source_pdf_id', 'Unknown PDF ID')
                    source_page_num = chunk.get('source_page_num', 'N/A') # This is 0-indexed
                    
                    circular_info = circular_details_map.get(source_pdf_id, {})
                    title = circular_info.get("title", f"Circular (ID: {source_pdf_id})")
                    pdf_url = circular_info.get("url", "#")

                    results_to_display.append({
                        "title": title,
                        "pdf_url": pdf_url,
                        "page_num": source_page_num + 1 if isinstance(source_page_num, int) else source_page_num, # Display 1-indexed page
                        "text": text_to_embed,
                        "distance": distances[i],
                        "chunk_id": chunk.get('chunk_id', 'N/A')
                    })

                if results_to_display:
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
                                st.text_area("", value=res['text'], height=200, key=f"text_area_{i}_{res['chunk_id']}") # Ensure unique key
                            container.markdown("---")
                else:
                     st.warning("No valid results to display after processing.")
    else:
        st.warning("Please enter a question to search.")

st.sidebar.header("About")
st.sidebar.info("""
This application performs semantic search over EPFO circulars using a FAISS index and sentence embeddings.
- **FAISS Index**: `circulars_faiss_ocr.index`
- **Text Metadata**: `circulars_faiss_ocr.index.meta.json`
- **Circular Info**: `circular-data.json`
- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
""")
st.sidebar.header("Setup Instructions")
st.sidebar.markdown("""
1.  **File Placement**: Ensure the following files are in the same directory as this script, or update their paths in the script:
    * `circulars_faiss_ocr.index`
    * `circulars_faiss_ocr.index.meta.json`
    * `circular-data.json`
2.  **Python Libraries**: Install the required libraries:
    ```bash
    pip install streamlit faiss-cpu sentence-transformers numpy
    ```
    (Use `faiss-gpu` if you have a compatible GPU and CUDA installed).
3.  **Run the App**:
    ```bash
    streamlit run your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of this Python file).
""")
