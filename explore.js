(() => {
    'use strict';

    const state = {
        taxonomy: null,
        summary: null,
        assignments: null,
        catalog: null,
        assignmentById: null,
        activeView: 'atlas',
        zoomDomain: null,
        evolutionMetric: 'count',
        trendIds: [],
        drawerRows: [],
        lastFocused: null,
    };

    const viewNames = ['atlas', 'evolution', 'compare', 'trends', 'quality'];
    const tabElements = Object.fromEntries(viewNames.map(name => [
        name,
        document.getElementById(`tab${name[0].toUpperCase()}${name.slice(1)}`),
    ]));
    const panelElements = Object.fromEntries(viewNames.map(name => [
        name,
        document.getElementById(`panel${name[0].toUpperCase()}${name.slice(1)}`),
    ]));

    const yearFilter = document.getElementById('yearFilterSelect');
    const domainFilter = document.getElementById('domainFilterSelect');
    const evolutionDomain = document.getElementById('evolutionDomainSelect');
    const compareYearA = document.getElementById('compareYearA');
    const compareYearB = document.getElementById('compareYearB');
    const qualityYear = document.getElementById('qualityYearSelect');
    const treemapContainer = document.getElementById('treemapContainer');
    const heatmapContainer = document.getElementById('heatmapContainer');
    const compareChart = document.getElementById('compareChart');
    const compareSummary = document.getElementById('compareSummary');
    const trendChart = document.getElementById('trendChart');
    const trendBreakdown = document.getElementById('trendBreakdown');
    const trendStatus = document.getElementById('trendStatus');
    const confidenceChart = document.getElementById('confidenceChart');
    const languageChart = document.getElementById('languageChart');
    const seasonalityChart = document.getElementById('seasonalityChart');
    const breadcrumb = document.getElementById('breadcrumbNav');
    const atlasTitle = document.getElementById('atlasTitle');
    const atlasSubtitle = document.getElementById('atlasSubtitle');
    const tooltip = document.getElementById('explorerTooltip');
    const drawer = document.getElementById('documentDrawer');
    const drawerOverlay = document.getElementById('drawerOverlay');
    const drawerTitle = document.getElementById('drawerTitle');
    const drawerSubtitle = document.getElementById('drawerSubtitle');
    const drawerList = document.getElementById('drawerList');
    const drawerSearch = document.getElementById('drawerSearchInput');
    const drawerClose = document.getElementById('drawerCloseBtn');

    const confidenceLabels = {
        '9': 'Manual override',
        '3': 'High confidence',
        '2': 'Moderate confidence',
        '1': 'Low confidence',
        '0': 'No confident match',
    };
    const languageLabels = {
        both: 'English + Hindi',
        english: 'English only',
        hindi: 'Hindi only',
        none: 'No PDF link',
    };
    const monthLabels = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar'];
    const financialMonthNumbers = ['04', '05', '06', '07', '08', '09', '10', '11', '12', '01', '02', '03'];
    let resizeTimer = null;

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>"']/g, character => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;',
        })[character]);
    }

    function formatNumber(value, digits = 0) {
        return Number(value || 0).toLocaleString('en-IN', {
            minimumFractionDigits: digits,
            maximumFractionDigits: digits,
        });
    }

    function knownYears() {
        return (state.summary?.financial_years || []).filter(year => year !== 'Unknown');
    }

    function yearTotal(year) {
        if (!year) return state.summary.total_documents;
        return state.summary.year_totals?.[year]
            ?? Object.values(state.summary.timeline?.[year] || {}).reduce((sum, count) => sum + count, 0);
    }

    function showTooltip(event, html) {
        tooltip.innerHTML = html;
        tooltip.style.opacity = '1';
        moveTooltip(event);
    }

    function moveTooltip(event) {
        const bounds = event.currentTarget?.getBoundingClientRect?.();
        const pageX = Number.isFinite(event.pageX) ? event.pageX : (bounds ? bounds.left + bounds.width / 2 + window.scrollX : 0);
        const pageY = Number.isFinite(event.pageY) ? event.pageY : (bounds ? bounds.top + bounds.height / 2 + window.scrollY : 0);
        tooltip.style.left = `${pageX + 12}px`;
        tooltip.style.top = `${pageY + 12}px`;
    }

    function hideTooltip() {
        tooltip.style.opacity = '0';
    }

    function hexToRgba(hex, alpha) {
        const clean = String(hex || '#2563eb').replace('#', '');
        const value = Number.parseInt(clean.length === 3
            ? clean.split('').map(char => char + char).join('')
            : clean, 16);
        return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
    }

    function activateOnKeyboard(event, action) {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            action();
        }
    }

    async function fetchJson(path) {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`Unable to load ${path}`);
        return response.json();
    }

    async function init() {
        try {
            [state.taxonomy, state.summary] = await Promise.all([
                fetchJson('data/topics/taxonomy.json'),
                fetchJson('data/topics/summary.json'),
            ]);
            populateControls();
            bindEvents();
            updateArchiveStats();
            renderAtlas();
        } catch (error) {
            console.error(error);
            treemapContainer.innerHTML = '<div class="empty-drawer-state">The topic assets could not be loaded. Run the classification pipeline and reload this page.</div>';
        }
    }

    function populateControls() {
        const years = state.summary.financial_years || [];
        const reversedYears = years.slice().reverse();
        [yearFilter, qualityYear].forEach(select => {
            reversedYears.forEach(year => select.add(new Option(`FY ${year}`, year)));
        });

        const comparisonYears = knownYears();
        comparisonYears.forEach(year => {
            compareYearA.add(new Option(`FY ${year}`, year));
            compareYearB.add(new Option(`FY ${year}`, year));
        });
        compareYearA.value = comparisonYears.includes('2014-2015') ? '2014-2015' : comparisonYears[0];
        compareYearB.value = comparisonYears.includes('2024-2025') ? '2024-2025' : comparisonYears.at(-1);

        for (const [domainId, domain] of Object.entries(state.taxonomy)) {
            const label = `${domain.name} (${formatNumber(domain.count)})`;
            domainFilter.add(new Option(label, domainId));
            evolutionDomain.add(new Option(label, domainId));
        }
    }

    function bindEvents() {
        for (const name of viewNames) {
            tabElements[name].addEventListener('click', () => switchView(name));
        }
        yearFilter.addEventListener('change', renderAtlas);
        domainFilter.addEventListener('change', () => {
            state.zoomDomain = domainFilter.value || null;
            renderAtlas();
        });
        evolutionDomain.addEventListener('change', renderHeatmap);
        document.getElementById('metricCount').addEventListener('click', () => setEvolutionMetric('count'));
        document.getElementById('metricShare').addEventListener('click', () => setEvolutionMetric('share'));
        compareYearA.addEventListener('change', renderComparison);
        compareYearB.addEventListener('change', renderComparison);
        qualityYear.addEventListener('change', renderQuality);
        document.getElementById('trendForm').addEventListener('submit', runTrendSearch);
        document.getElementById('reviewLowConfidence').addEventListener('click', reviewLowConfidence);
        drawerClose.addEventListener('click', closeDrawer);
        drawerOverlay.addEventListener('click', closeDrawer);
        document.addEventListener('keydown', handleGlobalKeydown);
        window.addEventListener('resize', () => {
            window.clearTimeout(resizeTimer);
            resizeTimer = window.setTimeout(() => {
                if (state.activeView === 'atlas') renderAtlas();
                if (state.activeView === 'trends' && state.trendIds.length) renderTrendResults(state.trendIds);
            }, 140);
        });
    }

    function switchView(name) {
        state.activeView = name;
        for (const viewName of viewNames) {
            const active = viewName === name;
            tabElements[viewName].classList.toggle('active', active);
            tabElements[viewName].setAttribute('aria-selected', String(active));
            panelElements[viewName].classList.toggle('active', active);
        }
        if (name === 'evolution') renderHeatmap();
        if (name === 'compare') renderComparison();
        if (name === 'quality') renderQuality();
    }

    function updateArchiveStats() {
        document.getElementById('statTotalDocs').textContent = formatNumber(state.summary.total_documents);
        document.getElementById('statTotalDomains').textContent = Object.keys(state.taxonomy).length;
    }

    function atlasCount(domainId, subtopicId = null) {
        const year = yearFilter.value;
        if (!year) {
            return subtopicId
                ? state.taxonomy[domainId].subtopics[subtopicId].count
                : state.taxonomy[domainId].count;
        }
        return subtopicId
            ? state.summary.subtopic_timeline?.[year]?.[domainId]?.[subtopicId] || 0
            : state.summary.timeline?.[year]?.[domainId] || 0;
    }

    function renderAtlas() {
        if (!state.taxonomy) return;
        const year = yearFilter.value;
        const domainId = state.zoomDomain;
        const scopeTotal = yearTotal(year);
        const entries = domainId
            ? Object.entries(state.taxonomy[domainId].subtopics).map(([subtopicId, subtopic]) => ({
                id: subtopicId,
                domainId,
                name: subtopic.name,
                count: atlasCount(domainId, subtopicId),
                color: state.taxonomy[domainId].color,
                type: 'subtopic',
            }))
            : Object.entries(state.taxonomy).map(([id, domain]) => ({
                id,
                name: domain.name,
                count: atlasCount(id),
                color: domain.color,
                type: 'domain',
            }));
        const visible = entries.filter(entry => entry.count > 0).sort((a, b) => b.count - a.count);

        if (domainId) {
            atlasTitle.textContent = `${state.taxonomy[domainId].name} · Subtopic Breakdown`;
            atlasSubtitle.textContent = year
                ? `Exact subtopic counts for FY ${year}. Select a tile to inspect the matching circulars.`
                : 'Select a tile to inspect the matching circulars.';
        } else {
            atlasTitle.textContent = year ? `Topic Breakdown · FY ${year}` : 'EPFO Policy Domain Breakdown';
            atlasSubtitle.textContent = 'Select a topic to zoom into its subtopics.';
        }
        updateBreadcrumb();

        if (!visible.length) {
            treemapContainer.innerHTML = '<div class="empty-drawer-state">No circulars match this topic and year.</div>';
            return;
        }
        if (treemapContainer.clientWidth < 560) {
            renderAtlasList(visible, scopeTotal);
            return;
        }

        treemapContainer.innerHTML = '';
        const width = treemapContainer.clientWidth || 1000;
        const height = 580;
        const root = d3.hierarchy({ children: visible }).sum(item => item.count).sort((a, b) => b.value - a.value);
        d3.treemap().size([width, height]).paddingInner(4).round(true)(root);
        const svg = d3.select(treemapContainer).append('svg')
            .attr('width', width)
            .attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('role', 'img')
            .attr('aria-label', `${atlasTitle.textContent}. ${visible.length} selectable topic areas.`);
        const nodes = svg.selectAll('g').data(root.leaves()).enter().append('g')
            .attr('class', 'treemap-node')
            .attr('transform', item => `translate(${item.x0},${item.y0})`)
            .attr('role', 'button')
            .attr('tabindex', 0)
            .attr('aria-label', item => `${item.data.name}, ${formatNumber(item.data.count)} circulars`)
            .on('click', (_event, item) => activateAtlasItem(item.data))
            .on('keydown', (event, item) => activateOnKeyboard(event, () => activateAtlasItem(item.data)))
            .on('mouseenter focus', (event, item) => {
                const share = scopeTotal ? 100 * item.data.count / scopeTotal : 0;
                showTooltip(event, `<strong>${escapeHtml(item.data.name)}</strong><br>${formatNumber(item.data.count)} circulars · ${formatNumber(share, 1)}% of selected scope`);
            })
            .on('mousemove', moveTooltip)
            .on('mouseleave blur', hideTooltip);
        nodes.append('rect')
            .attr('width', item => Math.max(0, item.x1 - item.x0))
            .attr('height', item => Math.max(0, item.y1 - item.y0))
            .attr('fill', item => item.data.color)
            .attr('rx', 4);
        nodes.append('text').attr('class', 'node-title').attr('x', 10).attr('y', 22)
            .text(item => fitLabel(item.data.name, item.x1 - item.x0, item.y1 - item.y0));
        nodes.append('text').attr('class', 'node-count').attr('x', 10).attr('y', 40)
            .text(item => (item.x1 - item.x0) >= 75 && (item.y1 - item.y0) >= 50
                ? `${formatNumber(item.data.count)} circulars` : '');
    }

    function fitLabel(label, width, height) {
        if (width < 75 || height < 38) return '';
        const max = Math.max(6, Math.floor(width / 8.3));
        return label.length > max ? `${label.slice(0, max - 1)}…` : label;
    }

    function renderAtlasList(entries, scopeTotal) {
        treemapContainer.innerHTML = `<div class="trend-breakdown"><div class="breakdown-list">${entries.map(entry => {
            const share = scopeTotal ? 100 * entry.count / scopeTotal : 0;
            return `<button type="button" class="breakdown-item" data-atlas-type="${entry.type}" data-domain="${escapeHtml(entry.domainId || entry.id)}" data-subtopic="${entry.type === 'subtopic' ? escapeHtml(entry.id) : ''}">
                <span>${escapeHtml(entry.name)}</span>
                <span class="mini-track"><span class="mini-fill" style="display:block;width:${Math.max(1, share)}%;background:${entry.color}"></span></span>
                <strong>${formatNumber(entry.count)}</strong>
            </button>`;
        }).join('')}</div></div>`;
        treemapContainer.querySelectorAll('[data-atlas-type]').forEach(button => button.addEventListener('click', () => {
            activateAtlasItem({
                type: button.dataset.atlasType,
                id: button.dataset.subtopic || button.dataset.domain,
                domainId: button.dataset.domain,
                name: button.querySelector('span').textContent,
            });
        }));
    }

    function activateAtlasItem(item) {
        if (item.type === 'domain') {
            state.zoomDomain = item.id;
            domainFilter.value = item.id;
            renderAtlas();
        } else {
            openDrawer(item.domainId, item.id, item.name, yearFilter.value || null);
        }
    }

    function updateBreadcrumb() {
        breadcrumb.innerHTML = '';
        const rootButton = document.createElement('button');
        rootButton.type = 'button';
        rootButton.className = state.zoomDomain ? 'breadcrumb-item' : 'breadcrumb-item current';
        rootButton.textContent = `All topics (${formatNumber(yearTotal(yearFilter.value))})`;
        rootButton.disabled = !state.zoomDomain;
        rootButton.addEventListener('click', () => {
            state.zoomDomain = null;
            domainFilter.value = '';
            renderAtlas();
        });
        breadcrumb.appendChild(rootButton);
        if (state.zoomDomain) {
            const separator = document.createElement('span');
            separator.className = 'breadcrumb-sep';
            separator.textContent = '›';
            breadcrumb.appendChild(separator);
            const current = document.createElement('span');
            current.className = 'breadcrumb-item current';
            current.textContent = state.taxonomy[state.zoomDomain].name;
            breadcrumb.appendChild(current);
        }
    }

    function setEvolutionMetric(metric) {
        state.evolutionMetric = metric;
        for (const name of ['Count', 'Share']) {
            const button = document.getElementById(`metric${name}`);
            const active = name.toLowerCase() === metric;
            button.classList.toggle('active', active);
            button.setAttribute('aria-pressed', String(active));
        }
        renderHeatmap();
    }

    function renderHeatmap() {
        if (!state.summary) return;
        heatmapContainer.innerHTML = '';
        const years = state.summary.financial_years || [];
        const domainIds = evolutionDomain.value ? [evolutionDomain.value] : Object.keys(state.taxonomy);
        const values = [];
        for (const domainId of domainIds) {
            for (const year of years) {
                const count = state.summary.timeline?.[year]?.[domainId] || 0;
                values.push(state.evolutionMetric === 'share' ? (yearTotal(year) ? 100 * count / yearTotal(year) : 0) : count);
            }
        }
        const maximum = Math.max(...values, 1);
        const table = document.createElement('table');
        table.className = 'heatmap-grid';
        const head = document.createElement('thead');
        const headRow = document.createElement('tr');
        headRow.innerHTML = '<th class="col-topic-name">Policy topic</th>';
        for (const year of years) {
            const cell = document.createElement('th');
            cell.scope = 'col';
            cell.textContent = year === 'Unknown' ? 'Unknown' : year.replace('20', "'").replace('-20', "–'");
            headRow.appendChild(cell);
        }
        head.appendChild(headRow);
        table.appendChild(head);
        const body = document.createElement('tbody');
        for (const domainId of domainIds) {
            const domain = state.taxonomy[domainId];
            const row = document.createElement('tr');
            const header = document.createElement('th');
            header.scope = 'row';
            header.innerHTML = `<div class="heatmap-row-header"><span class="domain-dot" style="background:${domain.color}"></span><span>${escapeHtml(domain.name)}</span></div>`;
            row.appendChild(header);
            for (const year of years) {
                const count = state.summary.timeline?.[year]?.[domainId] || 0;
                const share = yearTotal(year) ? 100 * count / yearTotal(year) : 0;
                const value = state.evolutionMetric === 'share' ? share : count;
                const cell = document.createElement('td');
                cell.className = count ? 'heatmap-cell' : 'heatmap-cell empty';
                if (count) {
                    const intensity = 0.12 + 0.88 * Math.sqrt(value / maximum);
                    cell.style.background = hexToRgba(domain.color, intensity);
                    cell.style.color = intensity > 0.58 ? '#ffffff' : '#0f172a';
                    const button = document.createElement('button');
                    button.type = 'button';
                    button.className = 'heatmap-button';
                    button.textContent = state.evolutionMetric === 'share' ? `${formatNumber(share, 1)}%` : formatNumber(count);
                    button.setAttribute('aria-label', `${domain.name}, FY ${year}: ${formatNumber(count)} circulars, ${formatNumber(share, 1)} percent of that year`);
                    button.addEventListener('mouseenter', event => showTooltip(event, `<strong>${escapeHtml(domain.name)}</strong><br>FY ${escapeHtml(year)} · ${formatNumber(count)} circulars · ${formatNumber(share, 1)}% of year`));
                    button.addEventListener('mousemove', moveTooltip);
                    button.addEventListener('mouseleave', hideTooltip);
                    button.addEventListener('click', () => openDrawer(domainId, null, `${domain.name} · FY ${year}`, year));
                    cell.appendChild(button);
                } else {
                    cell.textContent = '—';
                    cell.setAttribute('aria-label', `${domain.name}, FY ${year}: no circulars`);
                }
                row.appendChild(cell);
            }
            body.appendChild(row);
        }
        table.appendChild(body);
        heatmapContainer.appendChild(table);
    }

    function renderComparison() {
        if (!state.summary) return;
        const yearA = compareYearA.value;
        const yearB = compareYearB.value;
        const totalA = yearTotal(yearA);
        const totalB = yearTotal(yearB);
        const rows = Object.entries(state.taxonomy).map(([domainId, domain]) => {
            const countA = state.summary.timeline?.[yearA]?.[domainId] || 0;
            const countB = state.summary.timeline?.[yearB]?.[domainId] || 0;
            const shareA = totalA ? 100 * countA / totalA : 0;
            const shareB = totalB ? 100 * countB / totalB : 0;
            return { domainId, domain, countA, countB, shareA, shareB, delta: shareB - shareA };
        }).sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
        const biggest = rows[0];
        compareSummary.innerHTML = `
            <div class="summary-item"><span>FY ${escapeHtml(yearA)}</span><strong>${formatNumber(totalA)} circulars</strong></div>
            <div class="summary-item"><span>FY ${escapeHtml(yearB)}</span><strong>${formatNumber(totalB)} circulars</strong></div>
            <div class="summary-item"><span>Largest share change</span><strong>${escapeHtml(biggest.domain.name)} · ${biggest.delta >= 0 ? '+' : ''}${formatNumber(biggest.delta, 1)} pp</strong></div>`;
        const maxShare = Math.max(...rows.flatMap(row => [row.shareA, row.shareB]), 1);
        compareChart.innerHTML = rows.map(row => {
            const deltaClass = row.delta > 0.05 ? 'delta-positive' : row.delta < -0.05 ? 'delta-negative' : 'delta-flat';
            return `<div class="comparison-row">
                <button type="button" class="comparison-topic" data-domain="${row.domainId}" aria-label="Open ${escapeHtml(row.domain.name)} circulars for FY ${escapeHtml(yearB)}">${escapeHtml(row.domain.name)}</button>
                <div class="comparison-bars" aria-label="${escapeHtml(row.domain.name)}: ${formatNumber(row.shareA, 1)} percent in ${escapeHtml(yearA)}, ${formatNumber(row.shareB, 1)} percent in ${escapeHtml(yearB)}">
                    <div class="bar-track"><div class="bar-fill" style="width:${100 * row.shareA / maxShare}%;background:${hexToRgba(row.domain.color, 0.42)}"></div></div>
                    <div class="bar-track"><div class="bar-fill" style="width:${100 * row.shareB / maxShare}%;background:${row.domain.color}"></div></div>
                </div>
                <div class="comparison-value ${deltaClass}">${row.delta >= 0 ? '+' : ''}${formatNumber(row.delta, 1)} pp<br><span class="text-muted">${formatNumber(row.countA)} → ${formatNumber(row.countB)}</span></div>
            </div>`;
        }).join('');
        compareChart.querySelectorAll('[data-domain]').forEach(button => button.addEventListener('click', () => {
            const domainId = button.dataset.domain;
            openDrawer(domainId, null, `${state.taxonomy[domainId].name} · FY ${yearB}`, yearB);
        }));
    }

    function normalizeTokens(query) {
        return String(query || '').normalize('NFKC').toLocaleLowerCase('en')
            .replace(/[^\p{L}\p{M}\p{N}]+/gu, ' ').trim().split(/\s+/).filter(token => token.length >= 2);
    }

    function tokenVariants(token) {
        const variants = new Set([token]);
        if (!/^[a-z]+$/.test(token) || token.length < 4) return [...variants];
        if (token.endsWith('ies') && token.length > 4) variants.add(`${token.slice(0, -3)}y`);
        else if (token.endsWith('s') && !token.endsWith('ss')) variants.add(token.slice(0, -1));
        else if (token.endsWith('y')) variants.add(`${token.slice(0, -1)}ies`);
        else variants.add(`${token}s`);
        return [...variants];
    }

    function postingBucket(token) {
        return /^[a-z0-9]$/.test(token[0]) ? token[0] : 'other';
    }

    async function runTrendSearch(event) {
        event.preventDefault();
        const query = document.getElementById('trendQuery').value.trim();
        const tokens = normalizeTokens(query);
        if (!tokens.length) {
            trendStatus.textContent = 'Enter at least one word containing two or more characters.';
            return;
        }
        trendStatus.textContent = 'Finding matching circulars…';
        trendChart.innerHTML = '';
        trendBreakdown.innerHTML = '';
        try {
            const buckets = [...new Set(tokens.flatMap(token => tokenVariants(token).map(postingBucket)))];
            const bucketData = Object.fromEntries(await Promise.all(buckets.map(async bucket => [
                bucket,
                await fetchJson(`data/search/postings-${bucket}.json`),
            ])));
            const tokenSets = tokens.map(token => {
                const ids = new Set();
                for (const variant of tokenVariants(token)) {
                    for (const id of bucketData[postingBucket(variant)]?.[variant] || []) ids.add(id);
                }
                return ids;
            });
            let result = [...tokenSets[0]];
            for (const set of tokenSets.slice(1)) result = result.filter(id => set.has(id));
            state.trendIds = result;
            await ensureDetailData();
            renderTrendResults(result, query);
        } catch (error) {
            console.error(error);
            trendStatus.textContent = 'The trend could not be calculated. Reload the page and try again.';
        }
    }

    function renderTrendResults(ids, query = document.getElementById('trendQuery').value.trim()) {
        const idSet = new Set(ids);
        const years = knownYears();
        const byYear = Object.fromEntries(years.map(year => [year, 0]));
        const byDomain = Object.fromEntries(Object.keys(state.taxonomy).map(domainId => [domainId, 0]));
        for (const id of ids) {
            const assignment = state.assignmentById.get(id);
            if (!assignment) continue;
            if (assignment[4] !== 'Unknown') byYear[assignment[4]] = (byYear[assignment[4]] || 0) + 1;
            byDomain[assignment[1]] = (byDomain[assignment[1]] || 0) + 1;
        }
        trendStatus.innerHTML = `<strong>${formatNumber(ids.length)}</strong> circulars contain every word in “${escapeHtml(query)}”. Select a point or topic to inspect them.`;
        if (!ids.length) {
            trendChart.innerHTML = '<div class="empty-drawer-state">No indexed circulars contain all of those terms.</div>';
            return;
        }
        drawTrendLine(years.map(year => ({ year, count: byYear[year] || 0 })), idSet, query);
        const domains = Object.entries(byDomain).filter(([, count]) => count > 0)
            .sort((a, b) => b[1] - a[1]);
        const maximum = Math.max(...domains.map(([, count]) => count), 1);
        trendBreakdown.innerHTML = `<div class="breakdown-list">${domains.map(([domainId, count]) => `
            <button type="button" class="breakdown-item" data-trend-domain="${domainId}">
                <span>${escapeHtml(state.taxonomy[domainId].name)}</span>
                <span class="mini-track"><span class="mini-fill" style="display:block;width:${100 * count / maximum}%;background:${state.taxonomy[domainId].color}"></span></span>
                <strong>${formatNumber(count)}</strong>
            </button>`).join('')}</div>`;
        trendBreakdown.querySelectorAll('[data-trend-domain]').forEach(button => button.addEventListener('click', () => {
            const domainId = button.dataset.trendDomain;
            const rows = ids.map(id => state.assignmentById.get(id)).filter(row => row?.[1] === domainId);
            openDrawerRows(rows, `${query} · ${state.taxonomy[domainId].name}`);
        }));
    }

    function drawTrendLine(points, idSet, query) {
        trendChart.innerHTML = '';
        const width = Math.max(320, trendChart.clientWidth - 40);
        const height = 280;
        const margin = { top: 18, right: 20, bottom: 55, left: 54 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const x = d3.scalePoint().domain(points.map(point => point.year)).range([0, innerWidth]).padding(0.25);
        const y = d3.scaleLinear().domain([0, d3.max(points, point => point.count) || 1]).nice().range([innerHeight, 0]);
        const svg = d3.select(trendChart).append('svg').attr('width', width).attr('height', height)
            .attr('viewBox', `0 0 ${width} ${height}`).attr('role', 'img')
            .attr('aria-label', `Yearly trend for ${query}`);
        const group = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
        const tickStep = width < 620 ? 3 : 2;
        group.append('g').attr('class', 'chart-axis').attr('transform', `translate(0,${innerHeight})`)
            .call(d3.axisBottom(x).tickValues(points.filter((_, index) => index % tickStep === 0).map(point => point.year)).tickFormat(year => year.slice(2, 4)))
            .selectAll('text').attr('transform', 'rotate(-35)').style('text-anchor', 'end');
        group.append('g').attr('class', 'chart-axis').call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('d')));
        group.append('path').datum(points).attr('class', 'trend-line')
            .attr('d', d3.line().x(point => x(point.year)).y(point => y(point.count)));
        group.selectAll('circle').data(points).enter().append('circle').attr('class', 'trend-point')
            .attr('cx', point => x(point.year)).attr('cy', point => y(point.count)).attr('r', 5)
            .attr('role', 'button').attr('tabindex', 0)
            .attr('aria-label', point => `FY ${point.year}, ${point.count} matching circulars`)
            .on('click', (_event, point) => openTrendYear(point.year, idSet, query))
            .on('keydown', (event, point) => activateOnKeyboard(event, () => openTrendYear(point.year, idSet, query)))
            .on('mouseenter focus', (event, point) => showTooltip(event, `<strong>FY ${point.year}</strong><br>${formatNumber(point.count)} matching circulars`))
            .on('mousemove', moveTooltip).on('mouseleave blur', hideTooltip);
        svg.append('text').attr('class', 'chart-label').attr('x', margin.left + innerWidth / 2).attr('y', height - 5)
            .attr('text-anchor', 'middle').text('Financial year');
        svg.append('text').attr('class', 'chart-label').attr('transform', 'rotate(-90)').attr('x', -(margin.top + innerHeight / 2)).attr('y', 13)
            .attr('text-anchor', 'middle').text('Matching circulars');
    }

    function openTrendYear(year, idSet, query) {
        const rows = [...idSet].map(id => state.assignmentById.get(id)).filter(row => row?.[4] === year);
        openDrawerRows(rows, `${query} · FY ${year}`);
    }

    function renderQuality() {
        if (!state.summary) return;
        const year = qualityYear.value;
        const confidence = year
            ? state.summary.confidence_timeline?.[year] || {}
            : state.summary.confidence_counts || {};
        const language = year
            ? state.summary.language_timeline?.[year] || {}
            : state.summary.language_counts || {};
        renderQualityRows(confidenceChart, confidence, confidenceLabels, 'confidence');
        renderQualityRows(languageChart, language, languageLabels, 'language');
        renderSeasonality(year);
    }

    function renderQualityRows(container, counts, labels, kind) {
        const orderedKeys = kind === 'confidence' ? ['3', '2', '1', '0', '9'] : Object.keys(labels);
        const entries = orderedKeys.map(key => [key, labels[key], counts[key] || 0]);
        const total = entries.reduce((sum, entry) => sum + entry[2], 0);
        const maximum = Math.max(...entries.map(entry => entry[2]), 1);
        container.innerHTML = entries.map(([key, label, count]) => `
            <button type="button" class="quality-row" data-quality-kind="${kind}" data-quality-key="${key}">
                <span>${escapeHtml(label)}</span>
                <span class="mini-track"><span class="mini-fill" style="display:block;width:${100 * count / maximum}%"></span></span>
                <strong>${formatNumber(count)} <span class="text-muted">${total ? formatNumber(100 * count / total, 1) : '0.0'}%</span></strong>
            </button>`).join('');
        container.querySelectorAll('[data-quality-kind]').forEach(button => button.addEventListener('click', async () => {
            await ensureDetailData();
            const selectedYear = qualityYear.value;
            let rows = state.assignments.rows.filter(row => !selectedYear || row[4] === selectedYear);
            if (button.dataset.qualityKind === 'confidence') {
                rows = rows.filter(row => String(row[3]) === button.dataset.qualityKey);
            } else {
                rows = rows.filter(row => languageForDocument(state.catalog.documents[row[0]]) === button.dataset.qualityKey);
            }
            openDrawerRows(rows, `${button.querySelector('span').textContent}${selectedYear ? ` · FY ${selectedYear}` : ''}`);
        }));
    }

    function renderSeasonality(year) {
        const counts = Object.fromEntries(financialMonthNumbers.map(month => [month, 0]));
        if (year) {
            for (const [month, count] of Object.entries(state.summary.month_timeline?.[year] || {})) counts[month] += count;
        } else {
            for (const yearCounts of Object.values(state.summary.month_timeline || {})) {
                for (const [month, count] of Object.entries(yearCounts)) counts[month] += count;
            }
        }
        const maximum = Math.max(...Object.values(counts), 1);
        seasonalityChart.innerHTML = `<div class="seasonality-bars">${financialMonthNumbers.map((month, index) => `
            <div class="month-bar" aria-label="${monthLabels[index]}: ${formatNumber(counts[month])} circulars">
                <strong>${formatNumber(counts[month])}</strong>
                <span class="month-bar-fill" style="height:${Math.max(1, 130 * counts[month] / maximum)}px"></span>
                <span>${monthLabels[index]}</span>
            </div>`).join('')}</div>`;
    }

    function languageForDocument(documentRow) {
        if (documentRow[4] && documentRow[5]) return 'both';
        if (documentRow[5]) return 'english';
        if (documentRow[4]) return 'hindi';
        return 'none';
    }

    async function reviewLowConfidence() {
        await ensureDetailData();
        const year = qualityYear.value;
        const rows = state.assignments.rows.filter(row => (row[3] === 0 || row[3] === 1) && (!year || row[4] === year));
        openDrawerRows(rows, `Low-confidence classifications${year ? ` · FY ${year}` : ''}`);
    }

    async function ensureDetailData() {
        if (state.assignments && state.catalog) return;
        [state.assignments, state.catalog] = await Promise.all([
            state.assignments || fetchJson('data/topics/assignments.json'),
            state.catalog || fetchJson('data/search/catalog.json'),
        ]);
        state.assignmentById = new Map(state.assignments.rows.map(row => [row[0], row]));
    }

    async function openDrawer(domainId, subtopicId = null, title = 'Matching circulars', financialYear = null) {
        await ensureDetailData();
        const rows = state.assignments.rows.filter(row => row[1] === domainId
            && (!subtopicId || row[2] === subtopicId)
            && (!financialYear || row[4] === financialYear));
        openDrawerRows(rows, title);
    }

    function openDrawerRows(rows, title) {
        state.drawerRows = rows;
        state.lastFocused = document.activeElement;
        drawerTitle.textContent = title;
        drawerSubtitle.textContent = `${formatNumber(rows.length)} circulars found`;
        drawerSearch.value = '';
        renderDrawerItems(rows);
        drawerOverlay.classList.add('open');
        drawer.classList.add('open');
        drawer.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        drawerClose.focus();
        drawerSearch.oninput = () => filterDrawerRows();
    }

    function filterDrawerRows() {
        const query = drawerSearch.value.toLocaleLowerCase('en').trim();
        if (!query) return renderDrawerItems(state.drawerRows);
        const filtered = state.drawerRows.filter(row => {
            const documentRow = state.catalog.documents[row[0]] || [];
            return `${documentRow[1] || ''} ${documentRow[2] || ''}`.toLocaleLowerCase('en').includes(query);
        });
        renderDrawerItems(filtered);
    }

    function renderDrawerItems(rows) {
        drawerSubtitle.textContent = `${formatNumber(rows.length)} circulars found`;
        if (!rows.length) {
            drawerList.innerHTML = '<div class="empty-drawer-state">No matching circulars in this filter.</div>';
            return;
        }
        drawerList.innerHTML = rows.slice(0, 100).map(row => {
            const documentRow = state.catalog.documents[row[0]] || [];
            const title = documentRow[1] || 'Untitled circular';
            const circularNumber = documentRow[2] || '—';
            const date = documentRow[3] || '—';
            const hindiLink = documentRow[4];
            const englishLink = documentRow[5];
            const financialYear = documentRow[6] || 'Unknown';
            const ocrSource = documentRow[7];
            const primaryLink = englishLink || hindiLink || '';
            const filename = primaryLink ? primaryLink.split('/').pop().split('?')[0].split('#')[0] : '';
            const domain = state.taxonomy[row[1]]?.name || row[1];
            const confidence = confidenceLabels[String(row[3])] || 'Classified';
            return `<article class="drawer-circular-card">
                <div class="card-meta-line"><span class="card-cno">${escapeHtml(circularNumber)}</span><span>${escapeHtml(date)} · FY ${escapeHtml(financialYear)}</span></div>
                <div class="card-title">${escapeHtml(title)}</div>
                <div class="card-meta-line"><span>${escapeHtml(domain)}</span><span>${escapeHtml(confidence)}</span></div>
                <div class="card-actions">
                    ${ocrSource > 0 && filename ? `<a href="index.html?fy=${encodeURIComponent(financialYear)}&doc=${encodeURIComponent(filename)}" target="_blank" class="card-read-btn">Read paper view</a>` : ''}
                    ${englishLink ? `<a href="${escapeHtml(englishLink)}" target="_blank" rel="noopener noreferrer" class="card-pdf-link">EN ↗</a>` : ''}
                    ${hindiLink ? `<a href="${escapeHtml(hindiLink)}" target="_blank" rel="noopener noreferrer" class="card-pdf-link">HI ↗</a>` : ''}
                </div>
            </article>`;
        }).join('') + (rows.length > 100
            ? `<div class="empty-drawer-state">Showing the first 100 of ${formatNumber(rows.length)}. Filter by title or circular number to narrow the list.</div>` : '');
    }

    function closeDrawer() {
        drawerOverlay.classList.remove('open');
        drawer.classList.remove('open');
        drawer.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        if (state.lastFocused instanceof HTMLElement) state.lastFocused.focus();
    }

    function handleGlobalKeydown(event) {
        if (event.key === 'Escape' && drawer.classList.contains('open')) closeDrawer();
        if (event.key === 'Tab' && drawer.classList.contains('open')) {
            const focusable = [...drawer.querySelectorAll('button, input, a[href], select')].filter(element => !element.disabled);
            if (!focusable.length) return;
            const first = focusable[0];
            const last = focusable.at(-1);
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
