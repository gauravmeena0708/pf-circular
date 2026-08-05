import io
import re
import time
from bs4 import BeautifulSoup as bs, NavigableString
import requests
from urllib.parse import urljoin
import json
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
except ImportError as e:
    print(f"Warning: OCR dependencies not fully available ({e}).")
from datetime import datetime, timezone
import os
import argparse # For command-line arguments

# --- Configuration ---
# If tesseract is not in your PATH, you might need to specify its location
# For example, on Windows:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

CIRCULAR_DATA_FILE = "circular-data.json" # Not used directly anymore for output
INDEX_DATA_FILE = "index-data.json" # Not used directly anymore for output
MAX_URLS_TO_INDEX_PER_RUN = 50

HEADERS = {
    'Host': 'www.epfindia.gov.in',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0',
    'Accept': 'text/html, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://www.epfindia.gov.in/site_en/Contact_office_wise.php?id=MHEM',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

def generate_year_params():
    """Dynamically generates financial year parameters up to the current date."""
    current_date = datetime.now(timezone.utc)
    current_year = current_date.year
    current_month = current_date.month
    
    # Financial year in India starts in April
    if current_month >= 4:
        latest_start_year = current_year
    else:
        latest_start_year = current_year - 1
        
    params = []
    # Oldest explicitly listed year before 'Old Circulars' is 2009-2010
    for year in range(latest_start_year, 2008, -1):
        params.append(f"yr={year}-{year+1}")
        
    params.append("yr=Old+Circulars")
    return params

YEAR_PARAMS = generate_year_params()

# --- Utility Functions ---
def load_json_file(filepath):
    """Loads a JSON file if it exists, otherwise returns an empty dictionary."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not decode JSON from {filepath} ({e}). Starting fresh.")
            return {}
    return {}

def save_json_file(data, filepath):
    """Saves data to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data saved to {filepath}")

# --- Text Processing & PDF Extraction Functions ---
def clean_extracted_text(text):
    """Normalizes extracted text by removing control characters, fixing hyphenations, and cleaning whitespace."""
    if not text:
        return ""
    # Strip non-printable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # Fix broken line-end hyphens (e.g., "communica-\ntion" -> "communication")
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    # Collapse multiple horizontal spaces and tabs
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize excessive vertical linebreaks (3+ newlines to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(lines)
    return text.strip()

def extract_pdf_text_from_url(pdf_url, max_pages=2, retries=3):
    """
    Downloads a PDF from a URL with retries and rate limiting, extracts text from up to `max_pages` pages
    using direct PyMuPDF text extraction first, falling back to OCR if needed.
    Returns a metadata dictionary with extracted text or None if error.
    """
    if not pdf_url:
        return None
    
    pdf_bytes = None
    for attempt in range(retries):
        try:
            time.sleep(0.3)  # Rate limiting delay
            print(f"    Downloading PDF (attempt {attempt+1}/{retries}): {pdf_url}")
            pdf_response = requests.get(pdf_url, headers=HEADERS, timeout=35)
            pdf_response.raise_for_status()
            pdf_bytes = pdf_response.content
            break
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                print(f"    Connection warning ({e}), retrying in {(attempt+1)*2}s...")
                time.sleep((attempt + 1) * 2)
            else:
                print(f"    Failed downloading PDF after {retries} attempts: {pdf_url} ({e})")
                return None

    try:
        print(f"    Opening PDF with PyMuPDF...")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(pdf_document)
        if total_pages == 0:
            print(f"    PDF is empty: {pdf_url}")
            pdf_document.close()
            return None

        pages_to_process = min(total_pages, max_pages)
        page_texts = []
        extraction_methods = []

        for page_num in range(pages_to_process):
            page = pdf_document.load_page(page_num)
            direct_text = page.get_text("text").strip()
            
            # Use direct text if it yields >= 50 characters of content
            if len(direct_text) >= 50:
                page_texts.append(direct_text)
                extraction_methods.append("direct_text")
            else:
                # Scanned/image page fallback to Tesseract OCR
                try:
                    pix = page.get_pixmap(dpi=300)
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img, lang='eng').strip()
                    if ocr_text:
                        page_texts.append(ocr_text)
                        extraction_methods.append("ocr")
                    elif direct_text:
                        page_texts.append(direct_text)
                        extraction_methods.append("direct_text")
                except pytesseract.TesseractNotFoundError:
                    if direct_text:
                        page_texts.append(direct_text)
                        extraction_methods.append("direct_text")
                    else:
                        print("    Tesseract OCR not found.")
                        pdf_document.close()
                        return {
                            "ocr_content": "OCR_ERROR: Tesseract not found",
                            "method": "failed",
                            "total_pages": total_pages,
                            "pages_indexed": 0,
                            "indexed_at": datetime.now(timezone.utc).isoformat()
                        }

        pdf_document.close()

        raw_combined = "\n\n".join(page_texts)
        cleaned_text = clean_extracted_text(raw_combined)

        if "direct_text" in extraction_methods and "ocr" in extraction_methods:
            final_method = "hybrid"
        elif "ocr" in extraction_methods:
            final_method = "ocr"
        else:
            final_method = "direct_text"

        print(f"    Extraction successful for {pdf_url.split('/')[-1]} ({pages_to_process}/{total_pages} pages, method: {final_method}).")

        return {
            "ocr_content": cleaned_text,
            "method": final_method,
            "total_pages": total_pages,
            "pages_indexed": pages_to_process,
            "indexed_at": datetime.now(timezone.utc).isoformat()
        }

    except requests.exceptions.RequestException as e:
        print(f"    Error downloading PDF {pdf_url}: {e}")
        return None
    except Exception as e:
        print(f"    Error processing PDF {pdf_url}: {e}")
        return None

# --- Main Data Fetching Logic ---
def fetch_circular_metadata():
    """Fetches circular metadata from EPFO website and saves to circular-data.json."""
    parsed_circulars_data = []
    print("Starting to fetch circular metadata...")

    for q_param in YEAR_PARAMS:
        current_page_url = f'https://www.epfindia.gov.in/site_en/get_cir_content.php?{q_param}'
        print(f"  Requesting URL: {current_page_url}")

        try:
            r = requests.get(current_page_url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            soup = bs(r.text, 'html.parser')
            print(f"  Status Code: {r.status_code} for {q_param}")

            table_rows = soup.find_all('tr')
            data_rows = table_rows[1:] if table_rows and table_rows[0].find('th') else table_rows

            for row_idx, row in enumerate(data_rows):
                cells = row.find_all('td')
                if len(cells) < 4:
                    # print(f"    Skipping row {row_idx+1} in {q_param} due to insufficient cells ({len(cells)}).")
                    continue

                serial_no = cells[0].get_text(strip=True)
                subject_cell = cells[1]
                title_parts = [content.strip() for content in subject_cell.contents if isinstance(content, NavigableString) and content.strip()]
                title = " ".join(title_parts).split('Circular No.')[0].split('No.')[0].strip() # Basic title cleaning

                circular_no_date_raw = ""
                after_first_br = False
                temp_circular_parts = []
                for content in subject_cell.contents:
                    if content.name == 'br':
                        if not after_first_br:
                            after_first_br = True
                            continue
                        else: # Second br or end of relevant part
                            break
                    if after_first_br:
                        if isinstance(content, NavigableString):
                            text_content = content.strip()
                            if text_content:
                                temp_circular_parts.append(text_content)
                        elif content.name == 'a' and temp_circular_parts: # Link after some text
                            break
                circular_no_date_raw = " ".join(filter(None, temp_circular_parts))


                circular_no = ""
                date_of_circular = ""
                delimiter_dated = " dated "
                delimiter_date = " date " # some entries use "date" instead of "dated"
                
                actual_delimiter = None
                if delimiter_dated in circular_no_date_raw.lower(): # Check lower case
                    actual_delimiter = delimiter_dated
                elif delimiter_date in circular_no_date_raw.lower():
                    actual_delimiter = delimiter_date

                if actual_delimiter:
                    # Find the actual delimiter with original casing for split
                    delimiter_pos = circular_no_date_raw.lower().find(actual_delimiter)
                    original_delimiter = circular_no_date_raw[delimiter_pos : delimiter_pos + len(actual_delimiter)]
                    
                    parts = circular_no_date_raw.split(original_delimiter, 1)
                    circular_no = parts[0].strip()
                    if len(parts) > 1:
                        date_of_circular = parts[1].strip()
                else:
                    circular_no = circular_no_date_raw.strip()
                
                # Further clean title from circular number if any residue
                if circular_no and title.endswith(circular_no): # simple check
                    title = title[:-len(circular_no)].strip()


                def get_pdf_link(cell, base_url):
                    link_tag = cell.find('a')
                    if link_tag and link_tag.has_attr('href'):
                        relative_link = link_tag['href']
                        return urljoin(base_url, relative_link)
                    return None

                hindi_pdf_link = get_pdf_link(cells[2], current_page_url)
                english_pdf_link = get_pdf_link(cells[3], current_page_url)

                circular_data = {
                    "serial_no": serial_no,
                    "title": title,
                    "circular_no": circular_no,
                    "date": date_of_circular,
                    "hindi_pdf_link": hindi_pdf_link,
                    "english_pdf_link": english_pdf_link
                }
                parsed_circulars_data.append(circular_data)
            print(f"  Successfully processed {q_param}")

        except requests.exceptions.RequestException as e:
            print(f"  Error fetching {q_param}: {e}")
        except Exception as e:
            print(f"  An error occurred during parsing for {q_param}: {e}")

    if parsed_circulars_data:
        import os
        os.makedirs('data', exist_ok=True)
        years = {}
        for c in parsed_circulars_data:
            url = c.get('english_pdf_link') or c.get('hindi_pdf_link') or ''
            m = re.search(r'/Y(\d{4}-\d{4})/', url)
            year = m.group(1) if m else 'Unknown'
            if year not in years: years[year] = []
            years[year].append(c)
        
        sorted_years = sorted(list(years.keys()), reverse=True)
        if 'Unknown' in sorted_years:
            sorted_years.remove('Unknown')
            sorted_years.append('Unknown')
        metadata = {'years': sorted_years}
        save_json_file(metadata, 'data/metadata.json')
        
        for year, data in years.items():
            save_json_file(data, f'data/circulars-{year}.json')
    else:
        print("No circular metadata was extracted.")
    print("Finished fetching circular metadata.")

# --- Indexing Logic ---
def update_pdf_index(max_urls=50):
    """
    Reads circular metadata files from data/, extracts text (direct + OCR fallback)
    for English PDF links up to 2 pages, and updates data/index-{year}.json.
    Limits processing to max_urls new URLs per execution (or unlimited if None/0).
    """
    print("\nStarting PDF indexing process...")
    if not os.path.exists('data/metadata.json'):
        print("No metadata found. Run fetch first.")
        return

    metadata = load_json_file('data/metadata.json')
    newly_indexed_count = 0
    processed_urls_in_this_run = 0
    limit = max_urls if max_urls and max_urls > 0 else float('inf')

    for year in metadata.get('years', []):
        if processed_urls_in_this_run >= limit:
            break
        circulars_file = f'data/circulars-{year}.json'
        index_file = f'data/index-{year}.json'
        circulars = load_json_file(circulars_file)
        if not circulars:
            continue

        indexed_data = load_json_file(index_file) or {}
        year_updated = False

        for circular_entry in circulars:
            if processed_urls_in_this_run >= limit:
                break

            pdf_url = circular_entry.get("english_pdf_link")
            if not pdf_url:
                continue

            if pdf_url in indexed_data:
                continue  # Skip if already indexed

            print(f"  Processing new URL for indexing: {pdf_url.split('/')[-1]} (S.No {circular_entry.get('serial_no', 'N/A')})")
            result = extract_pdf_text_from_url(pdf_url, max_pages=2)
            processed_urls_in_this_run += 1

            if result is not None:
                indexed_data[pdf_url] = result
                if not result.get("ocr_content", "").startswith("OCR_ERROR"):
                    newly_indexed_count += 1
                year_updated = True
                print(f"    Added to index. Total newly indexed in this run: {newly_indexed_count}")
            else:
                print(f"    Extraction returned None. Skipping for now: {pdf_url.split('/')[-1]}")

        if year_updated:
            save_json_file(indexed_data, index_file)

    print(f"Finished PDF indexing process. Successfully indexed {newly_indexed_count} new PDFs across processed batches.")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch EPFO circular data and/or update PDF index.")
    parser.add_argument(
        "--action",
        choices=['fetch', 'index', 'all'],
        default='all',
        help="Specify action: 'fetch' metadata, 'index' PDFs, or 'all' (default)."
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=50,
        help="Maximum number of new PDFs to process for indexing in this run (0 for unlimited)."
    )
    args = parser.parse_args()

    if args.action == 'fetch' or args.action == 'all':
        fetch_circular_metadata()

    if args.action == 'index' or args.action == 'all':
        update_pdf_index(max_urls=args.max_urls)

    print("\nScript finished.")
