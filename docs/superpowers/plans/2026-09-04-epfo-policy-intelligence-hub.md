# EPFO Policy Intelligence Hub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the unified EPFO Policy Intelligence Hub in `explore.html` powered by an enhanced data pipeline in `classify.py` with real-data network graph, milestone timeline, division matrix, signal-vs-noise filter, and recovered classifications.

**Architecture:** Upgrade `classify.py` to classify ~1,900 previously unclassified circulars, tag circulars with tier (`policy` vs `admin`) and division, and export `network.json` and `milestones.json`. Refactor `explore.html` and `explore.js` into a 4-view hub (Policy Atlas, Real-Data Network, Milestone Chronicle, Division Matrix) with global filters and a rich document drawer. Update `graph-demo.html` to integrate with the real data engine.

**Tech Stack:** Python 3 (offline data pipeline), Vanilla JavaScript (ES6+), D3.js v7 (vector visualizations, treemap, force-directed graph), HTML5, CSS3 (custom properties, responsive layout).

**Spec:** `docs/superpowers/specs/2026-09-04-epfo-policy-intelligence-hub-design.md`

## Global Constraints

- **Zero heavy runtime frameworks**: Use standard HTML5, CSS3, and vanilla JS with D3.js (v7) via CDN.
- **Payload optimization**: Initial page payload must remain under 100 KB gzipped. Heavy indexes (`assignments.json`, `catalog.json`) load on-demand.
- **Backward compatibility**: Deep links (`index.html?fy=...&doc=...`) must continue functioning seamlessly.
- **Accessibility**: Keyboard navigation across views, WCAG AA color contrast (> 4.5:1), and ARIA tabs.

---

### Task 1: Data Pipeline Enhancement (`classify.py`) & Automated Tests

**Files:**
- Modify: `classify.py`
- Create: `tests/test_classify.py`

**Interfaces:**
- Consumes: `data/search/catalog.json`, `data/index-*.json`
- Produces: Enhanced `classify.py` with:
  - `classify_document(title, cno, filename, ocr)` returning `domain, subtopic, conf, secondaries, tier, division`
  - `generate_network_data(assignments, docs)` emitting `data/topics/network.json`
  - `generate_milestones_data(docs)` emitting `data/topics/milestones.json`

- [ ] **Step 1: Write failing tests for the enhanced classifier and data generation**

Create `tests/test_classify.py`:
```python
import json
import pytest
from classify import classify_document, extract_division, determine_tier, generate_network_data, generate_milestones_data

def test_recovery_of_unclassified_interest_credit():
    title = "Prompt Interest credit @8.25% for 2025-2026 in CITES 2.01 - Regarding"
    cno = "No:WSU/5(1)2005/Annual Accts/2026-2027/E-789534/23"
    domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
    assert domain == "finance_accounts_invest"
    assert subtopic == "interest_rate"
    assert tier == "policy"
    assert division == "WSU"

def test_recovery_of_unclassified_amnesty_compliance():
    title = "Launch of AMNESTY, 2026 for regularization of exemption status of Provident Fund Trusts"
    cno = "No.: Exemption/AMNESTY-2026/[E.III/10(58)/2025]"
    domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
    assert domain in ["exempted_establishments", "compliance_recovery"]
    assert tier == "policy"
    assert division == "Exemption"

def test_internal_admin_tier_tagging():
    title = "Final Seniority list in the cadre of Section Officer as on 31.08.2024"
    cno = "No. HRM-IV/28(6)2018/SO/SeniorityList /254"
    domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
    assert tier == "admin"
    assert division == "HRM"

def test_network_and_milestone_schema():
    mock_assignments = [
        [0, "pension_eps", "higher_pension", 3, "2022-2023", ["legal_litigation"], "policy", "Pension"],
        [1, "compliance_recovery", "7a_and_quasi_judicial", 3, "2023-2024", ["legal_litigation"], "policy", "Compliance"],
        [2, "hr_personnel_cadre", "promotions_seniority_dpc", 3, "2024-2025", [], "admin", "HRM"],
    ]
    mock_docs = [
        ["0", "Higher Pension SC Order", "P-1", "04/11/2022", None, "http://pdf", "2022-2023", 1],
        ["1", "7A Inquiry Guidelines", "C-1", "01/01/2023", None, "http://pdf", "2023-2024", 1],
        ["2", "Seniority List", "H-1", "01/01/2024", None, "http://pdf", "2024-2025", 1],
    ]
    net = generate_network_data(mock_assignments, mock_docs)
    assert "nodes" in net and "links" in net
    assert any(n["id"] == "pension_eps" for n in net["nodes"])

    ms = generate_milestones_data(mock_docs)
    assert "eras" in ms and "milestones" in ms
    assert len(ms["milestones"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_classify.py -v`  
Expected: FAIL with `ImportError` or missing functions.

- [ ] **Step 3: Implement enhanced keywords, tier tagging, division extraction, network & milestone generators in `classify.py`**

Update `classify.py`:
- Add regex and keyword extensions for:
  - `finance_accounts_invest`: `interest credit`, `annual accts`, `reconciliation of remittances`, `audit manual`, `fixed assets`, `depreciation`, `budget circular`.
  - `compliance_recovery`: `amnesty`, `vishwas`, `ibc`, `insolvency`, `resolution plan`, `damages under section`, `125 and 128`, `code on social security`, `coss`.
  - `it_digital_services`: `samadhan setu`, `issue tracker`, `scheduled downtime`, `e-office`, `samadhan`.
  - `pension_eps`: `dearness relief to central government pensioners`, `pensioners`, `pohw`, `higher wages`.
  - `exempted_establishments`: `regularization of exemption`, `provident fund trusts`.
  - `governance_cbt`: `calendar of sittings`, `parliament`, `monsoon session`, `central board`.
- Implement `extract_division(cno, title)`:
  - Detect `WSU`, `Pension`, `Compliance`, `CAIU`, `Finance`, `BSC`, `Audit`, `Investment`, `Exemption`, `Legal`, `IS`, `NDC`, `HRM`, `HRD`, `PDUNASS`, `NATRSS`, `CSD`, `Coordination`.
- Implement `determine_tier(domain, subtopic, title)`:
  - Returns `'policy'` for citizen/member/employer schemes (Pension, Compliance, Exemptions, Finance/Interest, IT/Portals, Social Security, Legal rulings).
  - Returns `'admin'` for internal cadre items (Promotions, Seniority, DPC, Transfers, APAR, Exams, Sports, Rajbhasha inspections).
- Implement `generate_network_data(assignments, docs)`:
  - Aggregate co-occurrences of primary domain with secondary domains and statutory flags.
  - Return `{ "nodes": [...], "links": [...] }`.
- Implement `generate_milestones_data(docs)`:
  - Define the 6 historical eras and 15 milestone events with dates, summary descriptions, and matching circular IDs.
- Write outputs to `data/topics/network.json`, `data/topics/milestones.json`, and updated `taxonomy.json`, `assignments.json`, `summary.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_classify.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add classify.py tests/test_classify.py
git commit -m "feat: enhance classify.py with unclassified recovery, tiering, and network/milestone generators"
```

---

### Task 2: Execute Classification Pipeline & Validate Data Assets

**Files:**
- Modify: `data/topics/taxonomy.json`
- Modify: `data/topics/assignments.json`
- Modify: `data/topics/summary.json`
- Create: `data/topics/network.json`
- Create: `data/topics/milestones.json`

**Interfaces:**
- Input: `python classify.py`
- Output: Enriched topic files with recovery of ~1,900 documents, tier flags, division fields, and network links.

- [ ] **Step 1: Execute `classify.py` to regenerate all topic assets**

Run: `python classify.py`  
Expected: Process all 9,534 documents, print reduction in unclassified circulars, and write `network.json` and `milestones.json`.

- [ ] **Step 2: Run verification script to audit generated datasets**

Run validation script:
```bash
python -c "
import json
with open('data/topics/assignments.json', 'r', encoding='utf-8') as f:
    a = json.load(f)
rows = a['rows']
unclass = sum(1 for r in rows if r[1] == 'unclassified')
print(f'Unclassified count: {unclass} ({unclass*100/len(rows):.1f}%)')
with open('data/topics/network.json', 'r', encoding='utf-8') as f:
    net = json.load(f)
print(f'Network nodes: {len(net[\"nodes\"])}, links: {len(net[\"links\"])}')
with open('data/topics/milestones.json', 'r', encoding='utf-8') as f:
    ms = json.load(f)
print(f'Milestones: {len(ms[\"milestones\"])}')
assert unclass < 800, 'Unclassified rate still too high'
assert len(net['links']) > 15, 'Network links insufficient'
assert len(ms['milestones']) >= 12, 'Milestones missing'
print('ALL CHECKS PASSED!')
"
```
Expected: `ALL CHECKS PASSED!` with unclassified count drastically reduced.

- [ ] **Step 3: Commit updated data files**

```bash
git add data/topics/
git commit -m "data: reclassify circulars, recover unclassified, and generate network and milestone assets"
```

---

### Task 3: Rebuild HTML Layout & Styles in `explore.html`

**Files:**
- Modify: `explore.html`

**Interfaces:**
- Consumes: Clean top navigation, CSS variables, D3.js v7 CDN
- Produces: Semantic HTML shell containing:
  - Global Controls Bar: Signal-vs-Noise toggle (`All`, `Public Policy`, `Internal Admin`), Year select, Division select, dynamic stats pills.
  - 4 View Tab Panels: `panelAtlas`, `panelNetwork`, `panelMilestones`, `panelMatrix`.
  - Side Inspector / Drawer shell with live filter and deep links.

- [ ] **Step 1: Update CSS styling in `explore.html`**

Add modern, high-contrast styles for:
- `.signal-switch`: Segmented pill control for Policy vs Admin vs All.
- `.view-switcher`: 4-tab pill navigation.
- `#panelNetwork`: Flex layout with D3 SVG container + `.network-sidebar` detail inspector.
- `#panelMilestones`: Horizontal/vertical era blocks with timeline milestone cards, tag pills, and summary drawer triggers.
- `#panelMatrix`: Scrollable heatmap table/grid with color intensity ramp and hover states.
- Responsive styles for mobile (< 768px).

- [ ] **Step 2: Update HTML markup in `explore.html`**

Restructure `<main class="workspace">`:
- Replace existing 6 panels with 4 streamlined panels:
  - `<section id="panelAtlas" class="view-panel active">`
  - `<section id="panelNetwork" class="view-panel">`
  - `<section id="panelMilestones" class="view-panel">`
  - `<section id="panelMatrix" class="view-panel">`
- Add network control bar: link strength range slider (`#strengthRange`), topic search select (`#networkFocusSelect`), reset view button.
- Update Document Drawer markup to support division pills, tier pills, and bilingual status.

- [ ] **Step 3: Manually test HTML rendering in browser/dev server**

Verify:
- Page loads with zero console errors.
- Tab buttons toggle `active` class.
- Controls bar displays all dropdowns and segment buttons.

- [ ] **Step 4: Commit markup and styling changes**

```bash
git add explore.html
git commit -m "feat(ui): rebuild explore.html markup and styles for 4-view Unified Hub"
```

---

### Task 4: Implement Unified Hub Logic & Visualizations in `explore.js`

**Files:**
- Modify: `explore.js`

**Interfaces:**
- Consumes: `taxonomy.json`, `summary.json`, `network.json`, `milestones.json`, lazy `assignments.json`, lazy `catalog.json`
- Produces: Interactive client-side application:
  - View navigation and URL hash syncing (`#atlas`, `#network`, `#milestones`, `#matrix`).
  - Tier filtering (`all`, `policy`, `admin`).
  - D3 Treemap with breadcrumb drilldown.
  - D3 Force-directed network graph with drag, zoom, edge selection, and neighbor highlighting.
  - Interactive Milestone Chronicle timeline with founding circular popups.
  - Division-by-year Heatmap Matrix with cell click interactions.
  - Unified slide-over document drawer with live search and deep links.

- [ ] **Step 1: Implement State & Data Loading in `explore.js`**

Implement:
- `state = { taxonomy, summary, network, milestones, assignments, catalog, activeView, activeTier, selectedFY, selectedDivision, networkSelection }`
- Async loader: fetch small files concurrently (`taxonomy.json`, `summary.json`, `network.json`, `milestones.json`).
- On-demand loader for `assignments.json` and `catalog.json`.
- Event listeners for Signal toggle, Year filter, Division filter, and View tabs.

- [ ] **Step 2: Implement Policy Atlas (Treemap) with Tier Filtering**

Implement:
- Dynamic treemap computation using `d3.treemap()`.
- Filter nodes based on `activeTier` (hide admin items when in `policy` mode).
- Click domain to zoom smoothly into subtopics; update breadcrumbs.
- Click subtopic to open pre-filtered document drawer.

- [ ] **Step 3: Implement Real-Data Policy Network (D3 Force Graph)**

Implement:
- Force simulation: `d3.forceSimulation(nodes)` with charge, center, link, and collision forces.
- Edge width scale based on true shared circular count.
- Link strength slider filter: dynamically hide links below threshold.
- Interactive click handlers:
  - Node click: Highlights neighbors, dims unrelated nodes/links, displays connected topics and circular counts in the sidebar.
  - Edge click: Highlights connecting topics, displays why they connect and exact shared circulars.
  - "Inspect Circulars in Drawer" button.

- [ ] **Step 4: Implement Milestone Chronicle (Policy Timeline)**

Implement:
- Group milestones by the 6 historical eras.
- Render era cards with dates, badges, background descriptions, and policy impact points.
- Click "View Circulars" on any milestone to open founding circulars in the drawer.

- [ ] **Step 5: Implement Division & Regulatory Matrix (Heatmap)**

Implement:
- Grid of Issuing Divisions (rows) vs Financial Years (columns).
- Cell color intensity scaled by circular volume (`d3.interpolateBlues`).
- Tooltip on cell hover showing exact circular count and primary topics.
- Cell click opens drawer filtered to that division and year.

- [ ] **Step 6: Implement Unified Document Drawer**

Implement:
- Drawer slide-in animation.
- Instant search filter by title, circular number, or date.
- Render rich document card: Monospace circular number, date, domain badge, division pill, bilingual indicator, official PDF link, and `index.html?fy=...&doc=...` reader link.

- [ ] **Step 7: Commit JavaScript changes**

```bash
git add explore.js
git commit -m "feat(ui): implement Unified Hub view renderers, network simulation, and drawer logic"
```

---

### Task 5: Update `graph-demo.html` & Cross-Linking

**Files:**
- Modify: `graph-demo.html`

**Interfaces:**
- Consumes: `data/topics/network.json`
- Produces: Updated standalone page with real-data integration and prominent banner linking to `explore.html#network`.

- [ ] **Step 1: Update `graph-demo.html` to load real `network.json`**

Replace hardcoded mock arrays with `fetch('data/topics/network.json')`. Wire up the node and link rendering to the real data.

- [ ] **Step 2: Add header banner linking to the Unified Hub**

Add top banner:
```html
<div class="hub-banner">
  <span>Looking for the complete EPFO Policy Intelligence Hub? Explore Treemap, Milestones & Division Matrix in the new explorer.</span>
  <a href="explore.html#network" class="btn-hub">Open Unified Hub →</a>
</div>
```

- [ ] **Step 3: Commit changes**

```bash
git add graph-demo.html
git commit -m "feat: wire graph-demo.html to real network data and add link banner to Unified Hub"
```

---

### Task 6: End-to-End Verification & Responsive Audit

**Files:**
- Test: Local HTTP server verification across all views.

- [ ] **Step 1: Start local HTTP server and verify data fetches**

Run: `python -m http.server 8000` (or verify via headless browser probe).  
Check console for any 404s, CORS issues, or NaN coordinates in D3 simulation.

- [ ] **Step 2: Verify all 4 view transitions**

1. Policy Atlas: Zoom into Pension -> Higher Pension -> Click opens drawer with 15+ circulars.
2. Policy Network: Adjust threshold slider, click `Pension` node, click `Pension <-> Legal` edge, open shared circulars in drawer.
3. Milestone Chronicle: Click `Supreme Court Higher Pension Ruling (Nov 2022)`, verify founding circular appears in drawer.
4. Division Matrix: Click `WSU` in `2024-2025`, verify member portal and joint declaration circulars open.
5. Signal-vs-Noise toggle: Toggle `Public Policy` -> total count updates to ~3,800. Toggle `Internal Admin` -> count updates to ~5,700.

- [ ] **Step 3: Verify document deep links**

Click "Open in Search Ledger" on a card -> opens `index.html?doc=...` correctly.

- [ ] **Step 4: Final commit and clean up**

```bash
git status
git commit -am "chore: complete verification of EPFO Policy Intelligence Hub"
```
