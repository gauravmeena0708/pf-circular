import json
import re

with open('fetch.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace file paths
content = content.replace('CIRCULAR_DATA_FILE = "circular-data.json"', 'CIRCULAR_DATA_FILE = "circular-data.json" # Not used directly anymore for output')
content = content.replace('INDEX_DATA_FILE = "index-data.json"', 'INDEX_DATA_FILE = "index-data.json" # Not used directly anymore for output')

# Replace writing logic in fetch action
fetch_write_old = '''    if parsed_circulars_data:
        save_json_file(parsed_circulars_data, CIRCULAR_DATA_FILE)
    else:
        print("No circular metadata was extracted.")'''

fetch_write_new = '''    if parsed_circulars_data:
        import os
        os.makedirs('data', exist_ok=True)
        years = {}
        for c in parsed_circulars_data:
            url = c.get('english_pdf_link') or c.get('hindi_pdf_link') or ''
            m = re.search(r'/Y(\d{4}-\d{4})/', url)
            year = m.group(1) if m else 'Unknown'
            if year not in years: years[year] = []
            years[year].append(c)
        
        metadata = {'years': sorted(list(years.keys()), reverse=True)}
        save_json_file(metadata, 'data/metadata.json')
        
        for year, data in years.items():
            save_json_file(data, f'data/circulars-{year}.json')
    else:
        print("No circular metadata was extracted.")'''
content = content.replace(fetch_write_old, fetch_write_new)

# Replace indexing logic
index_write_old = '''    print("\\nStarting PDF indexing process...")
    circulars = load_json_file(CIRCULAR_DATA_FILE)
    if not circulars:
        print("No circular data found in {CIRCULAR_DATA_FILE}. Cannot proceed with indexing.")
        return

    indexed_data = load_json_file(INDEX_DATA_FILE)
    newly_indexed_count = 0
    processed_urls_in_this_run = 0

    for circular_entry in circulars:
        if processed_urls_in_this_run >= MAX_URLS_TO_INDEX_PER_RUN:
            print(f"Reached maximum of {MAX_URLS_TO_INDEX_PER_RUN} URLs for this indexing run.")
            break'''

index_write_new = '''    print("\\nStarting PDF indexing process...")
    import os
    if not os.path.exists('data/metadata.json'):
        print("No metadata found. Run fetch first.")
        return
    
    metadata = load_json_file('data/metadata.json')
    newly_indexed_count = 0
    processed_urls_in_this_run = 0

    for year in metadata.get('years', []):
        if processed_urls_in_this_run >= MAX_URLS_TO_INDEX_PER_RUN: break
        circulars = load_json_file(f'data/circulars-{year}.json')
        if not circulars: continue
        
        indexed_data = load_json_file(f'data/index-{year}.json') or {}

        for circular_entry in circulars:
            if processed_urls_in_this_run >= MAX_URLS_TO_INDEX_PER_RUN: break'''

content = content.replace(index_write_old, index_write_new)

index_save_old = '''    if newly_indexed_count > 0:
        save_json_file(indexed_data, INDEX_DATA_FILE)
        print(f"Finished PDF indexing process. Successfully indexed {newly_indexed_count} new PDFs.")
    else:
        print("Finished PDF indexing process. No new PDFs needed indexing.")'''

index_save_new = '''            # We need to save per year inside the loop now.
            if ocr_text is not None:
                indexed_data[pdf_url] = {
                    "ocr_content": ocr_text,
                    "indexed_at": datetime.now(timezone.utc).isoformat()
                }
                newly_indexed_count += 1
                print(f"    Added to index. Total newly indexed in this run so far: {newly_indexed_count}")
                save_json_file(indexed_data, f'data/index-{year}.json')

    print(f"Finished PDF indexing process. Successfully indexed {newly_indexed_count} new PDFs.")'''

content = re.sub(r'            if ocr_text is not None:.*?print\("Finished PDF indexing process.*?new PDFs needed indexing\."\)', index_save_new, content, flags=re.DOTALL)

with open('fetch.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("fetch.py updated for chunking.")
