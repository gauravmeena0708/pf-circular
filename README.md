# EPFO Circular Fetcher and Indexer

This project fetches circulars from the Employees' Provident Fund Organisation (EPFO) India website. It extracts metadata, downloads PDF circulars, extracts text from the first two pages using direct PDF text with an OCR fallback, and builds a static client-side search index.

## Features

*   **Circular Metadata Fetching:** Automatically scrapes the official EPFO website to gather metadata for circulars, including title, circular number, date, and direct PDF download links.
*   **PDF Downloading:** Downloads the circulars in PDF format from the links obtained.
*   **PDF Text Processing:** Extracts text from up to two pages and uses Tesseract OCR when a page has no usable text layer.
*   **Text Indexing:** Creates year-sharded indexes of extracted PDF text, mapped to the original PDF URLs.
*   **Static Search Assets:** Builds a compact all-years catalog and token posting buckets for fast client-side search on GitHub Pages without loading the complete OCR corpus.
*   **Command-Line Interface:** Provides a script (`fetch.py`) to control fetching and indexing processes.

## How it Works

The project operates in two main stages:

1.  **Fetching Circular Metadata (`fetch_circular_metadata()` in `fetch.py`):**
    *   The script iterates through predefined year parameters to query the EPFO circulars page.
    *   It parses the HTML response to extract details for each circular, such as serial number, title, circular number, date, and links to Hindi and English PDFs.
    *   This extracted metadata is stored in `circular-data.json`.

2.  **PDF Indexing (`update_pdf_index()` in `fetch.py`):**
    *   The script reads `circular-data.json` for entries with English PDF links.
    *   For each new PDF, it downloads the file.
    *   Up to two pages are processed using PyMuPDF direct text extraction.
    *   Pytesseract is used as a fallback for scanned pages without usable embedded text.
    *   The extracted text and an indexing timestamp are stored in `index-data.json`, keyed by the PDF URL.
    *   To avoid overwhelming the system or the source server, the indexing process is limited by `MAX_URLS_TO_INDEX_PER_RUN` for each execution.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   Tesseract OCR: This project relies on Tesseract OCR for extracting text from PDF images. You must install it separately and ensure that the `tesseract` command is available in your system's PATH.
        *   **Windows:** Download and run the installer from the [official Tesseract releases page](https://github.com/UB-Mannheim/tesseract/wiki). The common `C:\Program Files\Tesseract-OCR\tesseract.exe` installation is detected automatically; other locations should be added to PATH.
        *   **Linux (Ubuntu/Debian):** `sudo apt-get install tesseract-ocr libtesseract-dev tesseract-ocr-eng`
        *   **macOS:** `brew install tesseract tesseract-lang`

2.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
    *(Replace `<repository-url>` and `<repository-directory>` with the actual URL and folder name)*

3.  **Install Python Dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The primary way to interact with this project is through the `fetch.py` script.

1.  **Fetching Circular Metadata:**
    To fetch the latest circular metadata from the EPFO website and save it in year-sharded `data/circulars-*.json` files:
    ```bash
    python fetch.py --action fetch
    ```

2.  **Indexing PDF Content:**
    To process the English PDFs, extract text, and update the year-sharded `data/index-*.json` files:
    ```bash
    python fetch.py --action index
    ```
    This will process up to `MAX_URLS_TO_INDEX_PER_RUN` (currently 50) new PDFs per run. Run it multiple times if you have more PDFs to index.

3.  **Fetch and Index (All Actions):**
    To perform both fetching and indexing in a single run:
    ```bash
    python fetch.py --action all
    ```
    This is equivalent to running `fetch` then `index`.

4.  **Rebuild GitHub Pages Search Assets:**
    ```bash
    python fetch.py --action search
    ```
    Run this after manually changing any `circulars-*.json` or `index-*.json` file. The `fetch`, `index`, and `all` actions rebuild these assets automatically.

5.  **Rebuild Topic Explorer Assets:**
    ```bash
    python fetch.py --action topics
    ```
    This classifies the current catalog and generates the compact topic, timeline, comparison, language, confidence, and seasonality summaries used by `explore.html`. The `all` action runs this step automatically.

## Policy Intelligence Hub

Open `explore.html` through a local web server or GitHub Pages. It provides:

* **Policy Milestone Chronicle**: 18-year chronological timeline across 6 regulatory eras with founding circular links.
* **Cross-Domain Knowledge Graph**: D3 force-directed co-occurrence network reflecting real secondary topic overlaps and linkage intensities.
* **Signal vs. Noise Filtering**: One-click toggles between `All Circulars` (9,534), `Public Policy & Schemes` (3,597), and `Internal Administration` (5,937).
* **Division Activity Matrix**: Heatmap cross-tabulating 18 financial years against key issuing wings (WSU, Pension, Compliance, Legal, HRM, etc.).
* **Dynamic Topic Atlas**: Zoomable treemap and hierarchical explorer with instant subtopic breakdowns.
* **Document Drawer**: Contextual slide-over drawer with instant search, metadata badges, and deep reader links into `index.html?fy=...&doc=...`.

## Data Files

The project uses the following JSON files to store data:

*   **`data/circulars-YYYY-YYYY.json`**: Year-sharded metadata fetched from EPFO, including titles, numbers, dates, and PDF links.
*   **`data/index-YYYY-YYYY.json`**: Year-sharded reproduced text, loaded lazily when a user opens a reproduction or runs an exact phrase search that needs verification.
*   **`data/search/catalog.json`**: Compact display metadata for every year. This is loaded once by the browser.
*   **`data/search/postings-*.json`**: First-character token buckets used for exact all-word matching. Only buckets needed by the query are downloaded.

## Key Dependencies

This project relies on several key Python libraries:

*   **`requests`**: For making HTTP requests to the EPFO website.
*   **`BeautifulSoup4` (`bs4`)**: For parsing HTML content scraped from the website.
*   **`PyMuPDF` (`fitz`)**: For opening, reading, and rendering PDF documents.
*   **`Pillow` (`PIL`)**: Used for image manipulation, specifically to handle images generated from PDF pages before OCR.
*   **`pytesseract`**: A Python wrapper for Google's Tesseract OCR Engine.

For a complete list of dependencies and their versions, please refer to `requirements.txt`.

## Contributing

Contributions to this project are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch for your changes (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Create a new Pull Request.

Please ensure your code follows the existing style and that you provide clear commit messages and a description of your changes in the pull request.


