# EPFO Circular Search

## Project Overview
This project aims to fetch, process, and provide search capabilities for EPFO (Employees' Provident Fund Organisation) circulars. It achieves this through two main methods: a basic OCR-indexed search for quick keyword-based retrieval and an advanced RAG-based Q&A system for more nuanced semantic search and question answering.

## Features
- Automated fetching of circulars from the EPFO website.
- OCR processing of circulars for text extraction.
- Basic keyword search functionality via a generated HTML page.
- Advanced semantic search and RAG-based question answering using FAISS and Sentence Transformers.
- Incremental updates for FAISS index.
- Streamlit web interface for the RAG system.
- GitHub Actions for automating fetching and indexing processes.

## How it Works

The project operates through two primary workflows:

**Workflow 1: Basic OCR Search**

- The `fetch.py` script is responsible for:
    - Fetching metadata for circulars (such as title, circular number, date, and PDF links) directly from the EPFO website.
    - Storing this collected metadata in a JSON file named `circular-data.json`.
    - Performing Optical Character Recognition (OCR) on the first page of English language PDF circulars to extract raw text.
    - Saving this extracted OCR text into `index-data.json`.
- The `index.html` file then utilizes both `circular-data.json` (for metadata) and `index-data.json` (for searchable text) to offer a straightforward keyword search functionality that runs directly in any web browser.

**Workflow 2: Advanced RAG-based Question Answering**

- The `update_faiss_index.py` script handles the more complex processing:
    - It processes the English PDF circulars that are listed in `circular-data.json`.
    - It performs detailed OCR across the entire content of these PDFs. Future enhancements may include table detection and extraction.
    - Text extracted from the circulars is then used to generate sentence embeddings using sophisticated Sentence Transformer models.
    - These embeddings are used to create and incrementally update a FAISS vector index, stored in `circulars_faiss_ocr.index`. This index allows for highly efficient similarity searches.
    - Metadata corresponding to the text chunks stored in the FAISS index is saved in `circulars_faiss_ocr.index.meta.json`.
    - To ensure efficient incremental updates, the script maintains a log of already processed PDFs in `processed_pdfs_log.json`, preventing redundant processing.
- The `streamlit_app.py` script launches an interactive web application:
    - This interface allows users to pose questions in natural language.
    - The user's question is encoded into an embedding using the same Sentence Transformer model that processed the circulars.
    - The FAISS index (`circulars_faiss_ocr.index`) is then queried to find the most relevant text chunks from the circulars based on semantic similarity.
    - The retrieved information is presented to the user, forming the basis of the answer (this process is known as Retrieval Augmented Generation - RAG).

## Repository Structure

-   `fetch.py`: Python script for fetching circular metadata and performing basic OCR indexing.
-   `fetch2.py`: Alternative/older script for fetching circular metadata.
-   `update_faiss_index.py`: Python script for creating and updating the FAISS index with detailed PDF content. It also creates `downloaded_pdfs_ocr/` to store downloaded PDFs and `temp_pdf_images_incremental/` for temporary image files during OCR.
-   `streamlit_app.py`: Python script for the Streamlit-based RAG question-answering application.
-   `index.html`: HTML file for basic keyword search on circulars.
-   `requirements.txt`: Lists Python dependencies for the project.
-   `circular-data.json`: Stores metadata of fetched circulars (e.g., title, PDF links).
-   `index-data.json`: Stores OCR text from the first page of circulars for basic search.
-   `circulars_faiss_ocr.index`: FAISS vector index file for advanced search.
-   `circulars_faiss_ocr.index.meta.json`: Metadata for text chunks stored in the FAISS index.
-   `processed_pdfs_log.json`: Logs PDF files that have been processed by `update_faiss_index.py`.
-   `.github/workflows/`: Contains GitHub Actions workflow configurations (e.g., `fetch.yaml`, `update-index.yaml`).

## Setup and Installation

### 1. Prerequisites

*   **Python**: Python 3.8+ is recommended.
*   **Tesseract OCR**: Required by `fetch.py` for OCR.
    *   It must be installed on your system and ideally added to your system's PATH.
    *   Installation guides: [Tesseract OCR Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
*   **Poppler**: Required by `pdf2image` (used in `update_faiss_index.py`) for converting PDFs to images, especially on Linux and macOS.
    *   On Debian/Ubuntu: `sudo apt-get install poppler-utils`
    *   On macOS (using Homebrew): `brew install poppler`
    *   For Windows, you might need to download Poppler binaries and add them to your PATH. Refer to `pdf2image` documentation for details.

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/epfo-circular-search.git # Replace with the actual project URL
cd epfo-circular-search
```

### 3. Install Python Dependencies

It's highly recommended to use a virtual environment:

```bash
python -m venv venv
# On Windows
# venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Tesseract Configuration (Optional)

If Tesseract OCR is installed but not added to your system's PATH, `fetch.py` might not be able to find it. In such cases, you may need to specify the path to the Tesseract executable directly in the `fetch.py` script by setting `pytesseract.pytesseract.tesseract_cmd`. For example:

```python
# In fetch.py, if Tesseract is not in PATH
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'/path/to/your/tesseract' # Example for Linux/macOS
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Example for Windows
```

## Usage

Make sure you have activated your virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) before running the scripts.

### 1. Fetching Circular Data and Basic OCR Index (`fetch.py`)

This script fetches circular metadata from the EPFO website and can perform OCR on the first page of English PDFs for basic search.

*   **Fetch only metadata**:
    ```bash
    python fetch.py --action fetch
    ```
    This updates `circular-data.json` with the latest circulars.

*   **Process local PDFs for OCR indexing**:
    ```bash
    python fetch.py --action index
    ```
    This reads `circular-data.json`, performs OCR on new English PDFs (up to `MAX_URLS_TO_INDEX_PER_RUN` defined in the script), and updates `index-data.json`.

*   **Fetch metadata and then index**:
    ```bash
    python fetch.py --action all
    ```
    Or simply:
    ```bash
    python fetch.py
    ```
    This performs both actions sequentially. The `index-data.json` is used by `index.html` for its search functionality.

### 2. Creating/Updating the FAISS Index (`update_faiss_index.py`)

This script processes English PDFs listed in `circular-data.json`, performs detailed OCR on their entire content, generates sentence embeddings, and creates/updates the FAISS vector index for advanced semantic search.

```bash
python update_faiss_index.py
```

*   The script is **incremental**: it keeps track of processed PDFs in `processed_pdfs_log.json` and will only process new or updated circulars.
*   It processes PDFs in batches (controlled by the `BATCH_SIZE` variable within the script).
*   The generated FAISS index is saved as `circulars_faiss_ocr.index`, and its metadata is stored in `circulars_faiss_ocr.index.meta.json`.
*   **Note**: This script can be time-consuming, especially during the initial run when all available circulars are processed. Subsequent runs will be faster.

### 3. Running the Streamlit RAG Application (`streamlit_app.py`)

This application provides a web interface for the RAG-based question-answering system.

```bash
streamlit run streamlit_app.py
```

*   After running the command, Streamlit will typically provide a local URL (e.g., `http://localhost:8501`).
*   Open this URL in your web browser to interact with the application, ask questions about EPFO circulars, and get answers based on the indexed content.

### 4. Using the Basic HTML Search (`index.html`)

This provides a simpler, offline keyword search functionality.

*   Simply open the `index.html` file in your web browser (e.g., by double-clicking it or using "File > Open" in your browser).
*   It uses the metadata from `circular-data.json` and the first-page OCR text from `index-data.json` to allow you to search for circulars by keywords found in their titles or first page content.

## Key Data Files

-   `circular-data.json`: Contains metadata about EPFO circulars fetched from the official website. This includes titles, circular numbers, dates, and direct links to PDF documents (both English and Hindi if available). Generated by `fetch.py` (or `fetch2.py`).
-   `index-data.json`: Stores OCR-extracted text from the first page of English circular PDFs, along with the URL of the PDF and an indexing timestamp. Used by `index.html` for basic search. Generated by `fetch.py`.
-   `circulars_faiss_ocr.index`: The FAISS vector index file. It stores numerical vector embeddings of text chunks extracted from the circulars, enabling efficient semantic similarity searches. Generated by `update_faiss_index.py`.
-   `circulars_faiss_ocr.index.meta.json`: Contains metadata associated with each vector in the FAISS index. This includes the original text chunk, its source PDF identifier (which links back to an entry in `circular-data.json`), page number, and other relevant details necessary for retrieving the context for search results. Generated by `update_faiss_index.py`.
-   `processed_pdfs_log.json`: A JSON file that logs the IDs (usually URLs or unique identifiers corresponding to entries in `circular-data.json`) of PDFs that have already been processed by `update_faiss_index.py`. This allows the script to incrementally process only new or unprocessed circulars in subsequent runs, saving time and resources.

## Automation (GitHub Actions)

The project includes GitHub Actions workflows defined in the `.github/workflows/` directory to automate certain tasks:

-   `.github/workflows/fetch.yaml`: This workflow likely automates the process of fetching the latest circular metadata from the EPFO website using `fetch.py --action fetch`, ensuring `circular-data.json` is kept up-to-date.
-   `.github/workflows/update-index.yaml`: This workflow probably automates the basic OCR indexing process. It likely runs `fetch.py --action index` (or `fetch.py --action all` after a fetch step) to update `index-data.json` based on the contents of `circular-data.json`.
-   `.github/workflows/update-feiss-index.yaml`: This workflow (note the typo in filename, likely intended to be `update-faiss-index.yaml`) is responsible for automating the creation and update of the FAISS index. It would run `update_faiss_index.py` to process PDFs, generate embeddings, and update `circulars_faiss_ocr.index`, `circulars_faiss_ocr.index.meta.json`, and `processed_pdfs_log.json`.

These workflows help in keeping the data files current, potentially running on a schedule (e.g., daily) or triggered by changes to the codebase. This ensures that the search indexes and underlying data reflect the latest available circulars with minimal manual intervention.

## Contributing

Contributions to this project are welcome. Please follow these general guidelines:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Test your changes thoroughly.
5.  Submit a pull request with a clear description of your changes.

## License

This project is currently unlicensed. Please add license information here.
