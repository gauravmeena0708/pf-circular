# create_faiss_index_ocr_incremental.py

import json
import os
import requests
import logging
import time
import shutil

# PDF, Image, OCR, Table Detection
import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import cv2 # OpenCV for image manipulation
import easyocr

# Transformers for table detection and sentence embeddings
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

# --- Configuration ---
JSON_FILE_PATH = 'circular-data.json' # Source of all PDF links
PDF_DOWNLOAD_DIR = 'downloaded_pdfs_ocr'
FAISS_INDEX_PATH = 'circulars_faiss_ocr.index'
METADATA_PATH = FAISS_INDEX_PATH + ".meta.json"
PROCESSED_PDFS_LOG_PATH = 'processed_pdfs_log.json' # Stores IDs of processed PDFs
TEMP_IMAGE_DIR = 'temp_pdf_images_incremental'

PDF_TO_IMAGE_DPI = 200
OCR_LANGUAGES = ['en']

TABLE_DETECTION_MODEL = "microsoft/table-transformer-detection"
SENTENCE_EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
MODEL_DEVICE = "cpu" # "cuda" if GPU is available

MAX_TEXT_BLOCK_LENGTH_FOR_EMBEDDING = 500
BATCH_SIZE = 10 # Number of new PDFs to process per run

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
# --- End Configuration ---

def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def load_processed_pdf_ids(log_path):
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {log_path}. Starting with an empty set of processed IDs.")
            return set()
        except Exception as e:
            logger.error(f"Error loading processed PDF IDs from {log_path}: {e}. Starting fresh.")
            return set()
    return set()

def save_processed_pdf_ids(ids_set, log_path):
    ensure_dir(os.path.dirname(log_path) or '.')
    try:
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(list(ids_set), f, indent=4)
        logger.info(f"Saved {len(ids_set)} processed PDF IDs to {log_path}")
    except Exception as e:
        logger.error(f"Error saving processed PDF IDs to {log_path}: {e}")

def load_existing_metadata(metadata_path):
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing metadata from {metadata_path}: {e}")
            return []
    return []

def save_combined_metadata(metadata_list, metadata_path):
    ensure_dir(os.path.dirname(metadata_path) or '.')
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4)
        logger.info(f"Saved combined metadata for {len(metadata_list)} chunks to {metadata_path}")
    except Exception as e:
        logger.error(f"Error saving metadata to {metadata_path}: {e}")


# (Keep convert_pdf_page_to_image, download_pdf, extract_structured_content_from_pdf_page, 
#  process_text_for_embedding, and load_english_pdf_links functions from the previous 
#  `create_faiss_index_ocr.py` script. They are largely unchanged in their core logic,
#  but ensure they use the `logger` instance defined in this script.)

# --- Re-include necessary helper functions here (ensure they use the global logger) ---
# Placeholder for brevity, assume these are copied from the previous script:
# def convert_pdf_page_to_image(...): ...
# def download_pdf(...): ...
# def extract_structured_content_from_pdf_page(...): ... (ensure it uses global config vars)
# def process_text_for_embedding(...): ... (ensure it uses global config vars)
# def load_english_pdf_links(...): ...
# --- Make sure the actual functions are included when you use this script ---
# --- For the purpose of this response, I will paste them directly ---

def convert_pdf_page_to_image(pdf_path, page_num, dpi=PDF_TO_IMAGE_DPI, output_folder=TEMP_IMAGE_DIR):
    """Converts a single page of a PDF to a PIL Image, saves it, and returns the path."""
    ensure_dir(output_folder)
    pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0] # e.g., "49_en"
    page_prefix = f"{pdf_basename}_page_{page_num + 1}" # e.g., "49_en_page_1"

    try:
        # convert_from_path will save images in output_folder.
        # It will use `page_prefix` as the base for the filename.
        # It returns a list of PIL.Image objects.
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num + 1, # pdf2image uses 1-based indexing for pages
            last_page=page_num + 1,
            output_folder=output_folder,
            fmt='png', # Save as PNG
            thread_count=1, # Better for predictable filenames with output_file
            output_file=page_prefix, # This will be the prefix of the saved file(s)
            paths_only=False # We will get PIL objects, but files are saved
        )

        if images:
            created_image_files = [f for f in os.listdir(output_folder) if f.startswith(page_prefix) and f.endswith(".png")]
            
            if created_image_files:
                actual_image_path = os.path.join(output_folder, created_image_files[0])
                logger.debug(f"Page {page_num + 1} of PDF '{os.path.basename(pdf_path)}' converted to image: {actual_image_path}")
                return actual_image_path # Return the actual string path of the saved image
            else:
                explicit_image_path = os.path.join(output_folder, f"{page_prefix}_explicit.png")
                images[0].save(explicit_image_path, 'PNG')
                logger.warning(f"Could not find image via listdir for prefix {page_prefix}. Saved explicitly to: {explicit_image_path}")
                return explicit_image_path
        else:
            logger.warning(f"pdf2image returned no images for page {page_num + 1} of PDF '{os.path.basename(pdf_path)}'.")
            return None

    except Exception as e:
        # Log the full traceback for PDF conversion errors
        logger.error(f"Error converting page {page_num + 1} of PDF '{os.path.basename(pdf_path)}' to image: {e}", exc_info=True)
    return None

def load_english_pdf_links(json_path):
    links_info = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # data is a list of dictionaries
        
        if not isinstance(data, list):
            logger.error(f"Error: Expected a list from {json_path}, but got {type(data)}. Please check JSON_FILE_PATH.")
            return []

        for item_content in data: # Iterate directly over the list
            if not isinstance(item_content, dict):
                logger.warning(f"Skipping an item as it's not a dictionary: {item_content}")
                continue

            # Use 'serial_no' as the base for the ID. Add a fallback if 'serial_no' is missing.
            item_id_base = item_content.get('serial_no', f"item_{len(links_info)}") 
            
            eng_link = item_content.get('english_pdf_link')
            
            # Ensure eng_link is not null and not an empty string before processing
            if eng_link and isinstance(eng_link, str) and eng_link.strip():
                links_info.append({"id": f"{item_id_base}_en", "url": eng_link, "language": "english"})
            # You can add similar handling for 'hindi_pdf_link' if needed in the future,
            # just ensure to create a distinct 'id' (e.g., f"{item_id_base}_hi")
            
        logger.info(f"Loaded {len(links_info)} English PDF links to process from {json_path}")
    except FileNotFoundError:
        logger.error(f"Error: {json_path} not found.")
    except json.JSONDecodeError:
        logger.error(f"Error: Could not decode JSON from {json_path}.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading PDF links: {e}")
    return links_info

def download_pdf(pdf_info, download_dir):
    ensure_dir(download_dir)
    safe_file_id = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in pdf_info['id'])
    file_name = f"{safe_file_id}.pdf"
    file_path = os.path.join(download_dir, file_name)
    url = pdf_info['url']
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        logger.info(f"PDF '{file_name}' already downloaded: {file_path}")
        return file_path
    logger.info(f"Downloading PDF from {url} to {file_path}")
    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded {url} to {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading {url}: {e}")
    return None

def extract_structured_content_from_pdf_page(image_path, page_num, pdf_id, table_detector, ocr_reader):
    page_content_blocks = []
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Could not open image {image_path} for page {page_num} of {pdf_id}: {e}")
        return page_content_blocks
    table_detections = []
    try:
        table_detections = table_detector(pil_image)
    except Exception as e:
        logger.error(f"Table detection error page {page_num} of {pdf_id}: {e}")
    table_boxes_pil = []
    for detection in table_detections:
        if detection['label'].lower() in ['table', 'table rotated']:
            box = detection['box']
            x0, y0, x1, y1 = max(0, int(box['xmin'])), max(0, int(box['ymin'])), min(pil_image.width, int(box['xmax'])), min(pil_image.height, int(box['ymax']))
            if x1 > x0 and y1 > y0: table_boxes_pil.append((x0, y0, x1, y1))
    table_boxes_pil.sort(key=lambda b: b[1])
    img_np_rgb = np.array(pil_image)
    non_table_mask = np.ones(img_np_rgb.shape[:2], dtype=np.uint8) * 255
    for x0, y0, x1, y1 in table_boxes_pil: non_table_mask[y0:y1, x0:x1] = 0
    non_table_img_for_ocr_np = img_np_rgb.copy()
    non_table_img_for_ocr_np[non_table_mask == 0] = [255, 255, 255]
    try:
        non_table_ocr_results = ocr_reader.readtext(non_table_img_for_ocr_np, paragraph=True, detail=1)
        for (bbox, text, conf) in non_table_ocr_results:
            page_content_blocks.append({"type": "text", "text": text, "bbox_pil": [int(c) for pt in bbox for c in pt], "y_start": int(bbox[0][1]), "page_num": page_num, "pdf_id": pdf_id})
    except Exception as e:
        logger.error(f"Non-table OCR error page {page_num} of {pdf_id}: {e}")
    for i, (x0_tbl, y0_tbl, x1_tbl, y1_tbl) in enumerate(table_boxes_pil):
        try:
            table_pil_image_crop = pil_image.crop((x0_tbl, y0_tbl, x1_tbl, y1_tbl))
            table_ocr_results = ocr_reader.readtext(np.array(table_pil_image_crop), paragraph=False, detail=1)
            cell_data = sorted([{'text': tc, 'y': int(bc[0][1]), 'x': int(bc[0][0])} for (bc, tc, cc) in table_ocr_results], key=lambda c: (c['y'], c['x']))
            linearized_table_text = " ".join([cell['text'] for cell in cell_data])
            if linearized_table_text.strip():
                page_content_blocks.append({"type": "table", "text": linearized_table_text, "bbox_pil": [x0_tbl, y0_tbl, x1_tbl, y1_tbl], "y_start": y0_tbl, "page_num": page_num, "pdf_id": pdf_id, "table_id_on_page": i})
        except Exception as e:
            logger.error(f"Table OCR error, table {i} page {page_num} of {pdf_id}: {e}")
    page_content_blocks.sort(key=lambda x: x['y_start'])
    return page_content_blocks

def process_text_for_embedding(all_extracted_blocks):
    final_chunks_for_embedding = []
    for block_idx, block in enumerate(all_extracted_blocks):
        text = block['text']
        words = text.split()
        # Using block_idx for unique part of chunk_id if sub-chunking happens for the same block type
        block_base_id = f"{block['pdf_id']}_pg{block['page_num']}_{block['type']}_{block.get('table_id_on_page', f'txt{block_idx}')}"

        if len(words) > MAX_TEXT_BLOCK_LENGTH_FOR_EMBEDDING:
            for i in range(0, len(words), MAX_TEXT_BLOCK_LENGTH_FOR_EMBEDDING):
                sub_chunk_text = " ".join(words[i : i + MAX_TEXT_BLOCK_LENGTH_FOR_EMBEDDING])
                final_chunks_for_embedding.append({
                    "chunk_id": f"{block_base_id}_sc{i//MAX_TEXT_BLOCK_LENGTH_FOR_EMBEDDING}",
                    "text_to_embed": sub_chunk_text, "source_pdf_id": block['pdf_id'],
                    "source_page_num": block['page_num'], "source_block_type": block['type'],
                    "source_bbox_pil": block['bbox_pil'], "original_text_block_ref": block_base_id
                })
        else:
            final_chunks_for_embedding.append({
                "chunk_id": block_base_id,
                "text_to_embed": text, "source_pdf_id": block['pdf_id'],
                "source_page_num": block['page_num'], "source_block_type": block['type'],
                "source_bbox_pil": block['bbox_pil']
            })
    logger.info(f"Processed {len(all_extracted_blocks)} blocks into {len(final_chunks_for_embedding)} final chunks for embedding.")
    return final_chunks_for_embedding


def main_incremental():
    logger.info("Starting INCREMENTAL FAISS index creation with OCR...")
    ensure_dir(PDF_DOWNLOAD_DIR)
    ensure_dir(TEMP_IMAGE_DIR)

    # 1. Load state and existing data
    processed_pdf_ids = load_processed_pdf_ids(PROCESSED_PDFS_LOG_PATH)
    logger.info(f"Loaded {len(processed_pdf_ids)} already processed PDF IDs.")
    
    existing_metadata_chunks = load_existing_metadata(METADATA_PATH)
    logger.info(f"Loaded {len(existing_metadata_chunks)} existing metadata chunks.")

    faiss_index = None
    dimension = -1
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            dimension = faiss_index.d
            logger.info(f"Loaded existing FAISS index with {faiss_index.ntotal} vectors, dimension {dimension}.")
            # Consistency check:
            if faiss_index.ntotal != len(existing_metadata_chunks):
                logger.warning(f"Mismatch! FAISS index has {faiss_index.ntotal} vectors, metadata has {len(existing_metadata_chunks)} chunks. "
                               f"This might lead to issues. Consider re-indexing if problems occur.")
        except Exception as e:
            logger.error(f"Error loading existing FAISS index from {FAISS_INDEX_PATH}: {e}. Will create a new one.")
            faiss_index = None # Ensure it's reset
            existing_metadata_chunks = [] # Reset metadata if index is bad

    # 2. Load Models
    logger.info(f"Loading Sentence Transformer: {SENTENCE_EMBEDDING_MODEL}")
    sentence_model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL, device=MODEL_DEVICE)
    if dimension == -1 and hasattr(sentence_model, 'get_sentence_embedding_dimension'):
         dimension = sentence_model.get_sentence_embedding_dimension()
    elif dimension == -1: # Fallback, encode a dummy sentence
        try:
            dummy_emb = sentence_model.encode("test")
            dimension = dummy_emb.shape[0]
            logger.info(f"Determined embedding dimension: {dimension}")
        except Exception as e:
            logger.error(f"Could not determine embedding dimension: {e}. Exiting.")
            return


    logger.info(f"Loading Table Detection model: {TABLE_DETECTION_MODEL}")
    table_detector = hf_pipeline("object-detection", model=TABLE_DETECTION_MODEL, device=0 if MODEL_DEVICE=="cuda" else -1)
    logger.info(f"Loading OCR reader: {OCR_LANGUAGES}")
    ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=(MODEL_DEVICE == "cuda"))

    # 3. Determine PDFs to process in this batch
    all_pdf_infos_from_json = load_english_pdf_links(JSON_FILE_PATH)
    unprocessed_pdf_infos = [pdf for pdf in all_pdf_infos_from_json if pdf['id'] not in processed_pdf_ids]
    
    if not unprocessed_pdf_infos:
        logger.info("No new PDFs to process. All PDFs from JSON are already in the processed log.")
        # Clean up empty temp image dir if it was created
        if os.path.exists(TEMP_IMAGE_DIR) and not os.listdir(TEMP_IMAGE_DIR):
            try:
                os.rmdir(TEMP_IMAGE_DIR)
            except OSError: pass # Ignore if not empty or other error
        return

    batch_to_process = unprocessed_pdf_infos[:BATCH_SIZE]
    logger.info(f"Processing batch of {len(batch_to_process)} new PDFs out of {len(unprocessed_pdf_infos)} remaining unprocessed PDFs.")

    newly_processed_in_this_batch_ids = set()
    new_metadata_chunks_this_batch = []
    
    # 4. Process this batch
    for pdf_info_item in batch_to_process:
        pdf_id = pdf_info_item['id']
        logger.info(f"--- Processing PDF for batch: {pdf_id} ---")
        downloaded_pdf_path = download_pdf(pdf_info_item, PDF_DOWNLOAD_DIR)
        if not downloaded_pdf_path:
            logger.warning(f"Skipping PDF {pdf_id} (batch) due to download failure.")
            continue

        current_pdf_content_blocks = []
        try:
            pdf_document = fitz.open(downloaded_pdf_path)
            num_pages = len(pdf_document)
        except Exception as e:
            logger.error(f"Failed to open PDF {downloaded_pdf_path}: {e}")
            continue
        
        pdf_temp_image_folder = os.path.join(TEMP_IMAGE_DIR, "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in pdf_id))
        ensure_dir(pdf_temp_image_folder)

        for page_num in range(num_pages):
            logger.info(f"Page {page_num + 1}/{num_pages} of PDF {pdf_id}")
            page_image_path = convert_pdf_page_to_image(downloaded_pdf_path, page_num, output_folder=pdf_temp_image_folder)
            if page_image_path and os.path.exists(page_image_path):
                page_blocks = extract_structured_content_from_pdf_page(page_image_path, page_num, pdf_id, table_detector, ocr_reader)
                current_pdf_content_blocks.extend(page_blocks)
                try: os.remove(page_image_path)
                except OSError as e: logger.warning(f"Could not remove temp page image {page_image_path}: {e}")
        pdf_document.close()
        try: # Attempt to remove the pdf-specific temp image folder if empty
            if os.path.exists(pdf_temp_image_folder) and not os.listdir(pdf_temp_image_folder):
                os.rmdir(pdf_temp_image_folder)
        except OSError as e: logger.warning(f"Could not remove {pdf_temp_image_folder}: {e}")


        if current_pdf_content_blocks:
            pdf_chunks_for_embedding = process_text_for_embedding(current_pdf_content_blocks)
            new_metadata_chunks_this_batch.extend(pdf_chunks_for_embedding)
            newly_processed_in_this_batch_ids.add(pdf_id)
            logger.info(f"Successfully processed PDF {pdf_id}, got {len(pdf_chunks_for_embedding)} chunks.")
        else:
            logger.warning(f"No content blocks extracted for PDF {pdf_id}. It might be empty or failed processing.")
            # Still mark as processed to avoid retrying problematic PDFs indefinitely,
            # OR implement a retry/error log for manual inspection. For now, mark as processed.
            newly_processed_in_this_batch_ids.add(pdf_id)


    # 5. Add new embeddings to FAISS index
    if new_metadata_chunks_this_batch:
        texts_to_embed = [chunk['text_to_embed'] for chunk in new_metadata_chunks_this_batch]
        logger.info(f"Generating embeddings for {len(texts_to_embed)} new text chunks from this batch...")
        new_embeddings = sentence_model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        new_embeddings_np = new_embeddings.astype('float32')

        if new_embeddings_np.shape[0] > 0 : # Check if any embeddings were generated
            if dimension != -1 and new_embeddings_np.shape[1] != dimension:
                logger.error(f"CRITICAL: New embeddings dimension ({new_embeddings_np.shape[1]}) "
                               f"does not match existing/expected dimension ({dimension}). Aborting add to index.")
            else:
                if faiss_index is None: # First time creating the index
                    if dimension == -1 : dimension = new_embeddings_np.shape[1] # Should have been set
                    logger.info(f"Creating new FAISS index with dimension {dimension}.")
                    faiss_index = faiss.IndexFlatL2(dimension)
                
                faiss_index.add(new_embeddings_np)
                logger.info(f"Added {new_embeddings_np.shape[0]} new vectors to FAISS index. Total vectors: {faiss_index.ntotal}")
                
                # Update and save all data
                combined_metadata = existing_metadata_chunks + new_metadata_chunks_this_batch # Order matters
                
                faiss.write_index(faiss_index, FAISS_INDEX_PATH)
                logger.info(f"FAISS index saved to {FAISS_INDEX_PATH}")
                
                save_combined_metadata(combined_metadata, METADATA_PATH) # Save combined
                
                processed_pdf_ids.update(newly_processed_in_this_batch_ids)
                save_processed_pdf_ids(processed_pdf_ids, PROCESSED_PDFS_LOG_PATH)
        else:
            logger.info("No new embeddings generated in this batch (perhaps no text was extracted).")
            # Still save processed IDs for PDFs that yielded no text to avoid reprocessing them
            if newly_processed_in_this_batch_ids:
                 processed_pdf_ids.update(newly_processed_in_this_batch_ids)
                 save_processed_pdf_ids(processed_pdf_ids, PROCESSED_PDFS_LOG_PATH)


    else:
        logger.info("No new PDF content processed in this batch to add to index.")
        # If some PDFs were attempted but yielded no chunks, their IDs might be in newly_processed_in_this_batch_ids
        if newly_processed_in_this_batch_ids: # e.g. problematic PDFs marked as processed
             processed_pdf_ids.update(newly_processed_in_this_batch_ids)
             save_processed_pdf_ids(processed_pdf_ids, PROCESSED_PDFS_LOG_PATH)


    # Cleanup overall temp image directory if it's empty
    if os.path.exists(TEMP_IMAGE_DIR) and not os.listdir(TEMP_IMAGE_DIR):
        try:
            shutil.rmtree(TEMP_IMAGE_DIR)
            logger.info(f"Cleaned up empty temporary image directory: {TEMP_IMAGE_DIR}")
        except Exception as e:
            logger.warning(f"Could not remove temporary image directory {TEMP_IMAGE_DIR}: {e}")

    logger.info("Incremental FAISS index update run completed.")
    remaining_unprocessed_count = len(all_pdf_infos_from_json) - len(load_processed_pdf_ids(PROCESSED_PDFS_LOG_PATH))
    logger.info(f"Estimated remaining PDFs to process in future runs: {remaining_unprocessed_count}")


if __name__ == '__main__':
    start_time = time.time()
    main_incremental()
    end_time = time.time()
    logger.info(f"Total execution time for this incremental run: {end_time - start_time:.2f} seconds.")
