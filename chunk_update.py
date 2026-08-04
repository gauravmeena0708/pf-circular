import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Add 'Load Older Circulars' button CSS
css_add = '''
        .load-more-btn { display: block; margin: 20px auto; padding: 10px 20px; border: 1px solid var(--accent); background: var(--surface); color: var(--accent); border-radius: 6px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .load-more-btn:hover { background: var(--accent-soft); }
        .load-more-btn:disabled { opacity: 0.5; cursor: not-allowed; }
'''
content = content.replace('</style>', css_add + '</style>')

# Add button to HTML
btn_html = '''
            <div class="pagination-controls" id="paginationControls"></div>
            <button id="loadOlderBtn" class="load-more-btn" style="display: none;">Load Historical Data (Older Years)</button>
'''
content = content.replace('<div class="pagination-controls" id="paginationControls"></div>', btn_html)

# Update loadAllData JS
js_replace_target = '''        async function loadAllData() {
            try {
                const [circularResponse, indexResponse] = await Promise.all([
                    fetch('circular-data.json'),
                    fetch('index-data.json').catch(err => {
                        console.warn('index-data.json unavailable; full-text search disabled.', err);
                        return { ok: false, json: () => Promise.resolve({}) };
                    })
                ]);

                if (!circularResponse.ok) {
                    throw new Error(HTTP  for circular-data.json);
                }
                allCircularData = await circularResponse.json();
                allIndexData = indexResponse.ok ? await indexResponse.json() : {};

                updateStats();
                populateFilters();'''

js_new = '''        let availableYears = [];
        let loadedYears = [];

        async function loadYearData(year) {
            const [circularResponse, indexResponse] = await Promise.all([
                fetch(data/circulars-.json),
                fetch(data/index-.json).catch(() => ({ ok: false, json: () => Promise.resolve({}) }))
            ]);
            
            if (circularResponse.ok) {
                const cData = await circularResponse.json();
                allCircularData = allCircularData.concat(cData);
            }
            if (indexResponse.ok) {
                const iData = await indexResponse.json();
                Object.assign(allIndexData, iData);
            }
            loadedYears.push(year);
            window.fuseInstance = null; // reset fuse index
        }

        async function loadAllData() {
            try {
                const metaResp = await fetch('data/metadata.json');
                const meta = await metaResp.json();
                availableYears = meta.years;
                
                // Load latest year by default
                await loadYearData(availableYears[0]);

                if (availableYears.length > 1) {
                    const btn = document.getElementById('loadOlderBtn');
                    btn.style.display = 'block';
                    btn.onclick = async () => {
                        btn.disabled = true;
                        btn.textContent = 'Loading...';
                        for (let i = 1; i < availableYears.length; i++) {
                            await loadYearData(availableYears[i]);
                        }
                        btn.style.display = 'none';
                        updateStats();
                        populateFilters();
                        performSearch();
                    };
                }

                updateStats();
                populateFilters();'''

content = content.replace(js_replace_target, js_new)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Chunking implemented.")
