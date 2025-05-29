import io
from bs4 import BeautifulSoup as bs, NavigableString
import requests
from urllib.parse import urljoin # For resolving relative URLs
import json # Added for JSON output

# Corrected headers
headers1 = {
    'Host': 'www.epfindia.gov.in', # Corrected Host
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:55.0) Gecko/20100101 Firefox/55.0',
    'Accept': 'text/html, */*; q=0.01',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br', # Added br for broader compatibility
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'https://www.epfindia.gov.in/site_en/Contact_office_wise.php?id=MHEM', # Corrected Referer domain and scheme
    'Connection': 'keep-alive', # Added based on typical browser requests
    'Upgrade-Insecure-Requests': '1' # Added based on typical browser requests
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

for q_param in f:
    # Construct the full URL with https and the correct domain
    current_page_url = f'https://www.epfindia.gov.in/site_en/get_cir_content.php?{q_param}'
    print(f"Requesting URL: {current_page_url}")

    try:
        r = requests.get(current_page_url, headers=headers1, timeout=10) # Added timeout
        r.raise_for_status()  # This will raise an HTTPError for bad responses (4XX or 5XX)
        
        soup = bs(r.text, 'html.parser')
        print(f"Status Code: {r.status_code}")
        
        # --- Start of Parsing Logic ---
        table_rows = soup.find_all('tr')
        
        data_rows = []
        if table_rows and table_rows[0].find('th'):
            data_rows = table_rows[1:]
        else:
            data_rows = table_rows

        for row in data_rows:
            cells = row.find_all('td')
            if len(cells) < 4:
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

            # Split circular_no_date_raw into circular_no and date
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
                date_of_circular = "" 

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
            
        print(f"Successfully fetched and parsed content for {q_param}")
        # Remove break if you want to process all years, otherwise it processes only the first one
        break 

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error for {q_param}: {errh}")
        print(f"Response Text: {r.text if 'r' in locals() else 'No response object'}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting for {q_param}: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error for {q_param}: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"OOps: Something Else for {q_param}: {err}")
    except Exception as e:
        print(f"An error occurred during parsing for {q_param}: {e}")

# Save the extracted data to a JSON file
if parsed_circulars_data:
    output_filename = "circular-data.json"
    with open(output_filename, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_circulars_data, json_file, ensure_ascii=False, indent=4)
    print(f"\n--- Extracted Circulars Data saved to {output_filename} ---")
else:
    print("No circular data was extracted to save.")

print("Done")
