import io
from bs4 import BeautifulSoup as bs, NavigableString
import requests
from urllib.parse import urljoin # For resolving relative URLs
import json
import fitz  # PyMuPDF
import pytesseract
from PIL import Image # Pillow

# If tesseract is not in your PATH, you might need to specify its location
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Corrected headers
headers1 = {
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

# This list will store the extracted data from the table
parsed_circulars_data = []
f = [
    "yr=2025-2026",
    "yr=2024-2025",
    "yr=2023-2024",
    "yr=2022-2023",
    "yr=2021-2022",
    "yr=2020-2021",
    "yr=2019-2020",
    "yr=2018-2019",
    "yr=2017-2018",
    "yr=2016-2017",
    "yr=2015-2016",
    "yr=2014-2015",
    "yr=2013-2014",
    "yr=2012-2013",
    "yr=2011-2012",
    "yr=2010-2011",
    "yr=2009-2010",
    "yr=Old+Circulars",
]

def get_first_page_ocr_text(pdf_url):
    """
    Downloads a PDF from a URL, extracts the first page,
    and performs OCR on it.
    Returns the OCRed text or None if an error occurs.
    """
    if not pdf_url:
        return None
    try:
        print(f"  Downloading PDF: {pdf_url}")
        pdf_response = requests.get(pdf_url, headers=headers1, timeout=30) # Increased timeout for PDF downloads
        pdf_response.raise_for_status()
        pdf_bytes = pdf_response.content

        print(f"  Opening PDF with PyMuPDF...")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(pdf_document) > 0:
            first_page = pdf_document.load_page(0)  # Load the first page (index 0)
            
            # Render page to an image (pixmap)
            # Increase DPI for better OCR results, e.g., 300
            pix = first_page.get_pixmap(dpi=300) 
            img_bytes = pix.tobytes("png") # Convert to PNG bytes
            
            img = Image.open(io.BytesIO(img_bytes))
            
            print(f"  Performing OCR on the first page...")
            # Specify language if needed, e.g., lang='eng'
            ocr_text = pytesseract.image_to_string(img, lang='eng') 
            pdf_document.close()
            print(f"  OCR successful.")
            return ocr_text.strip()
        else:
            print(f"  PDF is empty: {pdf_url}")
            pdf_document.close()
            return None
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading PDF {pdf_url}: {e}")
        return None
    except pytesseract.TesseractNotFoundError:
        print("  Tesseract OCR not found. Please ensure it's installed and in your PATH.")
        # You might want to raise this error or handle it to stop the script
        return "OCR_ERROR: Tesseract not found"
    except Exception as e:
        print(f"  Error processing PDF {pdf_url} for OCR: {e}")
        return None


for q_param in f:
    current_page_url = f'https://www.epfindia.gov.in/site_en/get_cir_content.php?{q_param}'
    print(f"Requesting URL: {current_page_url}")

    try:
        r = requests.get(current_page_url, headers=headers1, timeout=10)
        r.raise_for_status()
        soup = bs(r.text, 'html.parser')
        print(f"Status Code: {r.status_code}")

        table_rows = soup.find_all('tr')
        data_rows = []
        if table_rows and table_rows[0].find('th'):
            data_rows = table_rows[1:]
        else:
            data_rows = table_rows

        for row_idx, row in enumerate(data_rows): # Added index for detailed logging
            cells = row.find_all('td')
            if len(cells) < 4:
                print(f"  Skipping row {row_idx+1} in {q_param} due to insufficient cells ({len(cells)}).")
                continue

            serial_no = cells[0].get_text(strip=True)
            subject_cell = cells[1]
            title = ""
            circular_no_date_raw = ""

            title_parts = []
            for content in subject_cell.contents:
                if content.name == 'br':
                    break
                if isinstance(content, NavigableString):
                    title_parts.append(content.strip())
            title = " ".join(filter(None, title_parts))

            circular_parts_raw = []
            after_first_br = False
            for content in subject_cell.contents:
                if content.name == 'br':
                    if not after_first_br:
                        after_first_br = True
                        continue
                    else:
                        break
                if after_first_br:
                    if isinstance(content, NavigableString):
                        text_content = content.strip()
                        if text_content:
                            circular_parts_raw.append(text_content)
                    elif content.name == 'a' and not circular_parts_raw:
                        pass
                    elif content.name == 'a' and circular_parts_raw:
                        break
            circular_no_date_raw = " ".join(filter(None, circular_parts_raw))

            circular_no = ""
            date_of_circular = ""
            delimiter = " dated "
            if delimiter in circular_no_date_raw:
                parts = circular_no_date_raw.split(delimiter, 1)
                circular_no = parts[0].strip()
                if len(parts) > 1:
                    date_of_circular = parts[1].strip()
            else:
                circular_no = circular_no_date_raw.strip()

            def get_pdf_link(cell, base_url):
                link_tag = cell.find('a')
                if link_tag and link_tag.has_attr('href'):
                    relative_link = link_tag['href']
                    # Ensure the link is absolute
                    return urljoin(base_url, relative_link)
                return None

            # We are not processing hindi_pdf_link as per requirement
            # hindi_pdf_link = get_pdf_link(cells[2], current_page_url)
            english_pdf_link = get_pdf_link(cells[3], current_page_url)
            
            first_page_ocr_content = None
            if english_pdf_link:
                print(f"Processing English PDF for S.No: {serial_no}, Title: {title[:30]}...")
                first_page_ocr_content = get_first_page_ocr_text(english_pdf_link)
            else:
                print(f"  No English PDF link found for S.No: {serial_no}, Title: {title[:30]}...")


            circular_data = {
                "serial_no": serial_no,
                "title": title,
                "circular_no": circular_no,
                "date": date_of_circular,
                "hindi_pdf_link": None, # Setting to None as we are not processing it
                "english_pdf_link": english_pdf_link,
                "english_first_page_ocr_content": first_page_ocr_content # New field
            }
            parsed_circulars_data.append(circular_data)

        print(f"Successfully fetched and parsed content for {q_param}")

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error for {q_param}: {errh}")
        if 'r' in locals() and r is not None:
             print(f"Response Text: {r.text}")
        else:
            print("No response object available.")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting for {q_param}: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error for {q_param}: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"OOps: Something Else for {q_param}: {err}")
    except Exception as e:
        print(f"An error occurred during parsing for {q_param}: {e}")

if parsed_circulars_data:
    output_filename = "circular-data.json"
    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_circulars_data, json_file, ensure_ascii=False, indent=4)
    print(f"\n--- Extracted Circulars Data saved to {output_filename} ---")
else:
    print("No circular data was extracted to save.")

print("Done")
