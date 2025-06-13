# EPFO Circular Fetcher and Indexer

This project is designed to fetch circulars from the Employees' Provident Fund Organisation (EPFO) India website. It extracts metadata, downloads PDF circulars, performs Optical Character Recognition (OCR) on their first page, and builds an index of the extracted text to facilitate searching and information retrieval.

## Features

*   **Circular Metadata Fetching:** Automatically scrapes the official EPFO website to gather metadata for circulars, including title, circular number, date, and direct PDF download links.
*   **PDF Downloading:** Downloads the circulars in PDF format from the links obtained.
*   **OCR Processing:** Performs OCR on the first page of downloaded English PDF circulars using Tesseract OCR to extract textual content.
*   **Text Indexing:** Creates a local index (`index-data.json`) of the OCRed text, mapping it to the respective PDF URLs for quick lookups.
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
    *   The first page of the PDF is then processed using PyMuPDF to render it as an image.
    *   Pytesseract is used to perform OCR on this image to extract text.
    *   The extracted text and an indexing timestamp are stored in `index-data.json`, keyed by the PDF URL.
    *   To avoid overwhelming the system or the source server, the indexing process is limited by `MAX_URLS_TO_INDEX_PER_RUN` for each execution.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   Tesseract OCR: This project relies on Tesseract OCR for extracting text from PDF images. You must install it separately and ensure that the `tesseract` command is available in your system's PATH.
        *   **Windows:** Download and run the installer from the [official Tesseract releases page](https://github.com/UB-Mannheim/tesseract/wiki). During installation, make sure to add Tesseract to your PATH. If you encounter issues, you might need to specify the path to `tesseract.exe` directly in the `fetch.py` script by uncommenting and setting the `pytesseract.pytesseract.tesseract_cmd` variable (e.g., `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`).
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
    To fetch the latest circular metadata from the EPFO website and save it to `circular-data.json`:
    ```bash
    python fetch.py --action fetch
    ```

2.  **Indexing PDF Content:**
    To process the English PDFs linked in `circular-data.json`, perform OCR, and update `index-data.json`:
    ```bash
    python fetch.py --action index
    ```
    This will process up to `MAX_URLS_TO_INDEX_PER_RUN` (currently 500) new PDFs per run. Run it multiple times if you have more PDFs to index.

3.  **Fetch and Index (All Actions):**
    To perform both fetching and indexing in a single run:
    ```bash
    python fetch.py --action all
    ```
    This is equivalent to running `fetch` then `index`.

## Data Files

The project uses the following JSON files to store data:

*   **`circular-data.json`**: This file stores the metadata fetched from the EPFO website. Each entry includes details like the circular's title, number, date, and direct links to PDF versions (both English and Hindi, if available).
*   **`index-data.json`**: This file acts as an index for the OCR-processed PDFs. It contains a mapping where keys are the URLs of the English PDF circulars, and values are objects containing the extracted text from the first page (`ocr_content`) and the timestamp of when it was indexed (`indexed_at`).

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

## License

This project is currently not licensed. You may want to add a license file (e.g., MIT, Apache 2.0) to define how others can use, modify, and distribute the code.
