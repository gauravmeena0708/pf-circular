/**
 * explore.js - EPFO Policy Intelligence Hub Controller
 * Unifies Policy Atlas, Real-Data Network, Milestone Chronicle, and Division Matrix.
 */
(() => {
    'use strict';

    // Application State
    const state = {
        taxonomy: null,
        summary: null,
        network: null,
        milestones: null,
        assignments: null,
        catalog: null,
        activeView: 'atlas',        // 'atlas' | 'network' | 'milestones' | 'matrix'
        activeTier: 'policy',       // 'all' | 'policy' | 'admin'
        selectedFY: '',             // '' or 'YYYY-YYYY'
        selectedDivision: '',       // '' or division name
        zoomDomain: null,           // for Treemap drilldown
        networkThreshold: 4,        // min link strength
        networkFocus: '',           // focused node ID
        networkSelection: null,     // current selection in network
        simulation: null,           // D3 simulation instance
        drawerRows: [],             // active circulars in drawer
    };

    // DOM References
    const elements = {
        tabs: {
            atlas: document.getElementById('tabAtlas'),
            network: document.getElementById('tabNetwork'),
            milestones: document.getElementById('tabMilestones'),
            matrix: document.getElementById('tabMatrix'),
        },
        panels: {
            atlas: document.getElementById('panelAtlas'),
            network: document.getElementById('panelNetwork'),
            milestones: document.getElementById('panelMilestones'),
            matrix: document.getElementById('panelMatrix'),
        },
        signalButtons: document.querySelectorAll('.signal-btn'),
        yearSelect: document.getElementById('yearFilterSelect'),
        divisionSelect: document.getElementById('divisionFilterSelect'),
        statVisibleDocs: document.getElementById('statVisibleDocs'),
        statVisibleDomains: document.getElementById('statVisibleDomains'),
        statBilingualPct: document.getElementById('statBilingualPct'),
        // Atlas
        treemapContainer: document.getElementById('treemapContainer'),
        breadcrumbNav: document.getElementById('breadcrumbNav'),
        atlasTitle: document.getElementById('atlasTitle'),
        atlasSubtitle: document.getElementById('atlasSubtitle'),
        // Network
        networkContainer: document.getElementById('networkContainer'),
        networkSidebar: document.getElementById('networkSidebar'),
        networkStrengthRange: document.getElementById('networkStrengthRange'),
        networkStrengthOutput: document.getElementById('networkStrengthOutput'),
        networkFocusSelect: document.getElementById('networkFocusSelect'),
        networkResetBtn: document.getElementById('networkResetBtn'),
        // Milestones
        eraButtonsBar: document.getElementById('eraButtonsBar'),
        milestonesContainer: document.getElementById('milestonesContainer'),
        // Matrix
        matrixContainer: document.getElementById('matrixContainer'),
        // Drawer
        drawer: document.getElementById('documentDrawer'),
        drawerOverlay: document.getElementById('drawerOverlay'),
        drawerTitle: document.getElementById('drawerTitle'),
        drawerSubtitle: document.getElementById('drawerSubtitle'),
        drawerSearch: document.getElementById('drawerSearchInput'),
        drawerList: document.getElementById('drawerList'),
        drawerCloseBtn: document.getElementById('drawerCloseBtn'),
        tooltip: document.getElementById('explorerTooltip'),
    };

    // --- Data Loading ---

    async function init() {
        try {
            // Load compact initial assets in parallel
            const [taxResp, sumResp, netResp, msResp] = await Promise.all([
                fetch('data/topics/taxonomy.json').then(r => r.json()),
                fetch('data/topics/summary.json').then(r => r.json()),
                fetch('data/topics/network.json').then(r => r.json()),
                fetch('data/topics/milestones.json').then(r => r.json()),
            ]);

            state.taxonomy = taxResp;
            state.summary = sumResp;
            state.network = netResp;
            state.milestones = msResp;

            // Populate filter selects
            populateFilters();

            // Setup event listeners
            setupEventListeners();

            // Handle URL Hash if specified
            const initialHash = window.location.hash.replace('#', '').toLowerCase();
            if (['atlas', 'network', 'milestones', 'matrix'].includes(initialHash)) {
                switchView(initialHash);
            } else {
                renderActiveView();
            }

            updateMetrics();
        } catch (err) {
            console.error('Failed to initialize Policy Intelligence Hub:', err);
        }
    }

    async function ensureAssignmentsAndCatalog() {
        if (!state.assignments || !state.catalog) {
            const [assignResp, catResp] = await Promise.all([
                state.assignments ? Promise.resolve(null) : fetch('data/topics/assignments.json').then(r => r.json()),
                state.catalog ? Promise.resolve(null) : fetch('data/search/catalog.json').then(r => r.json()),
            ]);
            if (assignResp) state.assignments = assignResp;
            if (catResp) state.catalog = catResp;
        }
    }

    // --- Filters & UI Setup ---

    function populateFilters() {
        // Financial years
        if (elements.yearSelect) {
            const fys = (state.summary.financial_years || []).filter(y => y !== 'Unknown');
            elements.yearSelect.innerHTML = '<option value="">All Financial Years (2009–2027)</option>';
            fys.forEach(fy => {
                const opt = document.createElement('option');
                opt.value = fy;
                opt.textContent = `FY ${fy}`;
                elements.yearSelect.appendChild(opt);
            });
        }

        // Divisions
        if (elements.divisionSelect) {
            const divs = Object.keys(state.summary.divisions || {}).sort();
            elements.divisionSelect.innerHTML = '<option value="">All Divisions / Wings</option>';
            divs.forEach(div => {
                const opt = document.createElement('option');
                opt.value = div;
                opt.textContent = `${div} (${state.summary.divisions[div].toLocaleString()})`;
                elements.divisionSelect.appendChild(opt);
            });
        }

        // Network Focus Select
        if (elements.networkFocusSelect) {
            const nodes = state.network?.nodes || [];
            elements.networkFocusSelect.innerHTML = '<option value="">All connected nodes</option>';
            nodes.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n.id;
                opt.textContent = n.name;
                elements.networkFocusSelect.appendChild(opt);
            });
        }
    }

    function setupEventListeners() {
        // View tabs
        Object.entries(elements.tabs).forEach(([viewName, btn]) => {
            if (btn) btn.addEventListener('click', () => switchView(viewName));
        });

        // Signal tier switch
        elements.signalButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                elements.signalButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                state.activeTier = btn.dataset.tier;
                updateMetrics();
                renderActiveView();
            });
        });

        // FY Select
        if (elements.yearSelect) {
            elements.yearSelect.addEventListener('change', (e) => {
                state.selectedFY = e.target.value;
                updateMetrics();
                renderActiveView();
            });
        }

        // Division Select
        if (elements.divisionSelect) {
            elements.divisionSelect.addEventListener('change', (e) => {
                state.selectedDivision = e.target.value;
                updateMetrics();
                renderActiveView();
            });
        }

        // Network Slider
        if (elements.networkStrengthRange) {
            elements.networkStrengthRange.addEventListener('input', (e) => {
                state.networkThreshold = parseInt(e.target.value, 10);
                if (elements.networkStrengthOutput) {
                    elements.networkStrengthOutput.innerHTML = `<b>${state.networkThreshold}+</b>`;
                }
                if (state.activeView === 'network') renderNetwork();
            });
        }

        // Network Focus Select
        if (elements.networkFocusSelect) {
            elements.networkFocusSelect.addEventListener('change', (e) => {
                state.networkFocus = e.target.value;
                if (state.activeView === 'network') {
                    if (state.networkFocus) selectNetworkNode(state.networkFocus);
                    else clearNetworkSelection();
                }
            });
        }

        // Network Reset
        if (elements.networkResetBtn) {
            elements.networkResetBtn.addEventListener('click', () => {
                state.networkFocus = '';
                if (elements.networkFocusSelect) elements.networkFocusSelect.value = '';
                clearNetworkSelection();
            });
        }

        // Drawer close
        if (elements.drawerCloseBtn) elements.drawerCloseBtn.addEventListener('click', closeDrawer);
        if (elements.drawerOverlay) elements.drawerOverlay.addEventListener('click', closeDrawer);
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && elements.drawer && elements.drawer.classList.contains('open')) {
                closeDrawer();
            }
        });

        // Resize handler
        window.addEventListener('resize', debounce(() => {
            if (state.activeView === 'atlas') renderTreemap();
            if (state.activeView === 'network') renderNetwork();
        }, 150));
    }

    function switchView(viewName) {
        state.activeView = viewName;
        window.location.hash = viewName;

        Object.entries(elements.tabs).forEach(([name, btn]) => {
            const isActive = name === viewName;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });

        Object.entries(elements.panels).forEach(([name, panel]) => {
            panel.classList.toggle('active', name === viewName);
        });

        renderActiveView();
    }

    function renderActiveView() {
        switch (state.activeView) {
            case 'atlas':
                renderTreemap();
                break;
            case 'network':
                renderNetwork();
                break;
            case 'milestones':
                renderMilestones();
                break;
            case 'matrix':
                renderMatrix();
                break;
        }
    }

    function updateMetrics() {
        const { summary, activeTier, selectedFY, selectedDivision } = state;
        if (!summary) return;

        let total = summary.total_documents;
        if (activeTier === 'policy') total = summary.tiers?.policy || 3597;
        else if (activeTier === 'admin') total = summary.tiers?.admin || 5937;

        if (selectedFY) {
            const fyTotal = summary.year_totals?.[selectedFY] || 0;
            const fyTiers = summary.fy_tiers?.[selectedFY] || {};
            if (activeTier === 'policy') total = fyTiers.policy || 0;
            else if (activeTier === 'admin') total = fyTiers.admin || 0;
            else total = fyTotal;
        }

        if (selectedDivision) {
            total = summary.divisions?.[selectedDivision] || total;
        }

        elements.statVisibleDocs.textContent = total.toLocaleString('en-IN');
        elements.statVisibleDomains.textContent = Object.keys(summary.domains || {}).length;

        const lang = summary.language_counts || {};
        const both = lang.both || 0;
        const sumAll = summary.total_documents || 1;
        elements.statBilingualPct.textContent = `${Math.round((both / sumAll) * 100)}%`;
    }

    // --- View 1: Policy Atlas (Treemap) ---

    function renderTreemap() {
        const container = elements.treemapContainer;
        container.innerHTML = '';

        const width = container.clientWidth || 960;
        const height = container.clientHeight || 580;

        // Build hierarchical dataset based on active filters
        const rootData = { name: 'EPFO Policy Taxonomy', children: [] };
        const { taxonomy, summary, activeTier, selectedFY, zoomDomain } = state;

        Object.entries(taxonomy).forEach(([dId, dInfo]) => {
            if (dId === 'unclassified' && activeTier === 'policy') return;

            // Tier filtering
            const isPolicyDomain = ['pension_eps', 'compliance_recovery', 'exempted_establishments', 'finance_accounts_invest', 'it_digital_services', 'social_security_schemes', 'citizen_services_rti'].includes(dId);
            if (activeTier === 'policy' && !isPolicyDomain) return;
            if (activeTier === 'admin' && isPolicyDomain) return;

            if (zoomDomain && zoomDomain !== dId) return;

            const domainNode = {
                id: dId,
                name: dInfo.name,
                color: dInfo.color,
                children: []
            };

            Object.entries(dInfo.subtopics).forEach(([sId, sInfo]) => {
                let count = sInfo.count || 0;
                if (selectedFY && summary.subtopic_timeline?.[selectedFY]?.[dId]?.[sId] !== undefined) {
                    count = summary.subtopic_timeline[selectedFY][dId][sId];
                }
                if (count > 0) {
                    domainNode.children.push({
                        id: sId,
                        domainId: dId,
                        name: sInfo.name,
                        value: count,
                        color: dInfo.color
                    });
                }
            });

            if (domainNode.children.length > 0) {
                rootData.children.push(domainNode);
            }
        });

        // Update Breadcrumb
        renderBreadcrumb();

        if (rootData.children.length === 0) {
            container.innerHTML = '<div class="empty-state">No circulars match the selected tier and financial year.</div>';
            return;
        }

        const hierarchy = d3.hierarchy(rootData)
            .sum(d => d.value)
            .sort((a, b) => b.value - a.value);

        d3.treemap()
            .size([width, height])
            .paddingInner(3)
            .paddingOuter(4)
            .round(true)(hierarchy);

        const svg = d3.select(container)
            .append('svg')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('width', '100%')
            .attr('height', '100%');

        const leaves = hierarchy.leaves();

        const nodes = svg.selectAll('g')
            .data(leaves)
            .enter()
            .append('g')
            .attr('class', 'treemap-node')
            .attr('transform', d => `translate(${d.x0},${d.y0})`)
            .on('click', (event, d) => {
                if (!state.zoomDomain) {
                    // Zoom into domain
                    state.zoomDomain = d.data.domainId;
                    renderTreemap();
                } else {
                    // Open drawer for subtopic
                    openDrawer({
                        domain: d.data.domainId,
                        subtopic: d.data.id,
                        title: `${d.data.name}`
                    });
                }
            })
            .on('mouseenter', (event, d) => {
                showTooltip(event, `<strong>${d.data.name}</strong><br>${d.value.toLocaleString('en-IN')} circulars`);
            })
            .on('mousemove', moveTooltip)
            .on('mouseleave', hideTooltip);

        nodes.append('rect')
            .attr('width', d => Math.max(0, d.x1 - d.x0))
            .attr('height', d => Math.max(0, d.y1 - d.y0))
            .attr('fill', d => d.data.color)
            .attr('rx', 4);

        nodes.append('text')
            .attr('x', 6)
            .attr('y', 16)
            .attr('fill', '#ffffff')
            .attr('font-size', '11px')
            .attr('font-weight', '700')
            .each(function(d) {
                const rectWidth = d.x1 - d.x0;
                const rectHeight = d.y1 - d.y0;
                if (rectWidth < 45 || rectHeight < 28) return;

                const textEl = d3.select(this);
                textEl.text(d.data.name);

                // Add count line
                if (rectHeight >= 42 && rectWidth >= 55) {
                    textEl.append('tspan')
                        .attr('x', 6)
                        .attr('dy', 14)
                        .attr('font-weight', '500')
                        .attr('fill', 'rgba(255,255,255,0.85)')
                        .text(`${d.value.toLocaleString('en-IN')}`);
                }
            });
    }

    function renderBreadcrumb() {
        const nav = elements.breadcrumbNav;
        nav.innerHTML = '';

        const rootBtn = document.createElement('button');
        rootBtn.type = 'button';
        rootBtn.className = 'breadcrumb-item';
        rootBtn.textContent = 'All Policy Domains';
        rootBtn.onclick = () => {
            state.zoomDomain = null;
            renderTreemap();
        };
        nav.appendChild(rootBtn);

        if (state.zoomDomain) {
            const sep = document.createElement('span');
            sep.textContent = '›';
            sep.style.color = 'var(--muted)';
            nav.appendChild(sep);

            const domainBtn = document.createElement('span');
            domainBtn.className = 'breadcrumb-item current';
            domainBtn.textContent = state.taxonomy[state.zoomDomain]?.name || state.zoomDomain;
            nav.appendChild(domainBtn);
        }
    }

    // --- View 2: Policy Network (D3 Force Graph) ---

    function renderNetwork() {
        const container = elements.networkContainer;
        container.innerHTML = '';

        if (!state.network) return;

        const width = container.clientWidth || 700;
        const height = container.clientHeight || 560;

        const { nodes: rawNodes, links: rawLinks } = state.network;
        const threshold = state.networkThreshold;

        // Filter links by threshold
        const visibleLinks = rawLinks
            .filter(l => l.value >= threshold)
            .map(l => ({ ...l, source: l.source, target: l.target }));

        const activeNodeIds = new Set();
        visibleLinks.forEach(l => {
            activeNodeIds.add(typeof l.source === 'object' ? l.source.id : l.source);
            activeNodeIds.add(typeof l.target === 'object' ? l.target.id : l.target);
        });

        // Filter nodes
        const visibleNodes = rawNodes
            .filter(n => activeNodeIds.has(n.id) || n.id === state.networkFocus)
            .map(n => ({ ...n }));

        if (visibleNodes.length === 0 || visibleLinks.length === 0) {
            container.innerHTML = '<div class="sidebar-placeholder">No relationships meet this connection threshold. Try lowering the minimum shared circulars slider.</div>';
            return;
        }

        const svg = d3.select(container)
            .append('svg')
            .attr('viewBox', `0 0 ${width} ${height}`)
            .attr('width', '100%')
            .attr('height', '100%');

        const zoomLayer = svg.append('g');

        const zoom = d3.zoom()
            .scaleExtent([0.4, 3])
            .on('zoom', (event) => zoomLayer.attr('transform', event.transform));

        svg.call(zoom).on('dblclick.zoom', null);

        // Scales
        const maxVal = d3.max(visibleLinks, d => d.value) || 1;
        const linkWidthScale = d3.scaleSqrt().domain([threshold, maxVal]).range([1.5, 7]);
        const nodeRadius = d => d.is_hub ? 16 : 22;

        // Links
        const linkGroup = zoomLayer.append('g').attr('class', 'links');
        const hitLinkGroup = zoomLayer.append('g').attr('class', 'hit-links');

        const linkElements = linkGroup.selectAll('line')
            .data(visibleLinks)
            .enter()
            .append('line')
            .attr('class', 'network-link')
            .attr('stroke-width', d => linkWidthScale(d.value));

        hitLinkGroup.selectAll('line')
            .data(visibleLinks)
            .enter()
            .append('line')
            .attr('class', 'network-hit-link')
            .on('click', (event, d) => {
                event.stopPropagation();
                selectNetworkLink(d);
            })
            .on('mouseenter', (event, d) => {
                const srcName = getNodeName(d.source);
                const tgtName = getNodeName(d.target);
                showTooltip(event, `<strong>${srcName} ↔ ${tgtName}</strong><br>${d.value} shared circulars`);
            })
            .on('mousemove', moveTooltip)
            .on('mouseleave', hideTooltip);

        // Nodes
        const nodeGroup = zoomLayer.append('g').attr('class', 'nodes');
        const nodeElements = nodeGroup.selectAll('g')
            .data(visibleNodes)
            .enter()
            .append('g')
            .attr('class', 'network-node')
            .on('click', (event, d) => {
                event.stopPropagation();
                selectNetworkNode(d.id);
            })
            .on('mouseenter', (event, d) => {
                showTooltip(event, `<strong>${d.name}</strong><br>Click to view connected topics & circulars`);
            })
            .on('mousemove', moveTooltip)
            .on('mouseleave', hideTooltip)
            .call(d3.drag()
                .on('start', dragStarted)
                .on('drag', dragged)
                .on('end', dragEnded));

        nodeElements.append('circle')
            .attr('r', nodeRadius)
            .attr('fill', d => d.color);

        nodeElements.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', d => nodeRadius(d) + 14)
            .text(d => d.name.length > 20 ? d.name.slice(0, 18) + '…' : d.name);

        svg.on('click', clearNetworkSelection);

        // Simulation
        if (state.simulation) state.simulation.stop();

        state.simulation = d3.forceSimulation(visibleNodes)
            .force('link', d3.forceLink(visibleLinks).id(d => d.id).distance(d => 120 - Math.min(40, d.value)).strength(0.6))
            .force('charge', d3.forceManyBody().strength(-350))
            .force('center', d3.forceCenter(width / 2, height / 2))
            .force('collide', d3.forceCollide().radius(d => nodeRadius(d) + 20))
            .on('tick', () => {
                visibleNodes.forEach(d => {
                    const r = nodeRadius(d) + 4;
                    d.x = Math.max(r, Math.min(width - r, d.x));
                    d.y = Math.max(r, Math.min(height - r, d.y));
                });

                linkElements
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                hitLinkGroup.selectAll('line')
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                nodeElements.attr('transform', d => `translate(${d.x},${d.y})`);
            });

        function dragStarted(event, d) {
            if (!event.active) state.simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }

        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }

        function dragEnded(event, d) {
            if (!event.active) state.simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
    }

    function selectNetworkNode(nodeId) {
        state.networkSelection = { type: 'node', id: nodeId };
        elements.networkFocusSelect.value = nodeId;

        const node = state.network.nodes.find(n => n.id === nodeId);
        if (!node) return;

        // Find connected links and neighbors
        const neighborIds = new Set([nodeId]);
        const connectedLinks = state.network.links.filter(l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            if (sId === nodeId) { neighborIds.add(tId); return true; }
            if (tId === nodeId) { neighborIds.add(sId); return true; }
            return false;
        });

        // Update visuals
        d3.selectAll('.network-node').classed('active', d => d.id === nodeId).classed('dimmed', d => !neighborIds.has(d.id));
        d3.selectAll('.network-link').classed('active', l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            return sId === nodeId || tId === nodeId;
        }).classed('dimmed', l => {
            const sId = typeof l.source === 'object' ? l.source.id : l.source;
            const tId = typeof l.target === 'object' ? l.target.id : l.target;
            return sId !== nodeId && tId !== nodeId;
        });

        // Update Sidebar
        renderNodeSidebar(node, connectedLinks);
    }

    function selectNetworkLink(link) {
        const sId = typeof link.source === 'object' ? link.source.id : link.source;
        const tId = typeof link.target === 'object' ? link.target.id : link.target;
        state.networkSelection = { type: 'link', source: sId, target: tId };

        d3.selectAll('.network-node').classed('active', d => d.id === sId || d.id === tId).classed('dimmed', d => d.id !== sId && d.id !== tId);
        d3.selectAll('.network-link').classed('active', l => {
            const curS = typeof l.source === 'object' ? l.source.id : l.source;
            const curT = typeof l.target === 'object' ? l.target.id : l.target;
            return (curS === sId && curT === tId) || (curS === tId && curT === sId);
        }).classed('dimmed', l => {
            const curS = typeof l.source === 'object' ? l.source.id : l.source;
            const curT = typeof l.target === 'object' ? l.target.id : l.target;
            return !((curS === sId && curT === tId) || (curS === tId && curT === sId));
        });

        renderLinkSidebar(link);
    }

    function clearNetworkSelection() {
        state.networkSelection = null;
        d3.selectAll('.network-node').classed('active', false).classed('dimmed', false);
        d3.selectAll('.network-link').classed('active', false).classed('dimmed', false);
        elements.networkSidebar.innerHTML = `
            <div class="sidebar-placeholder">
                Select any topic node to view its strongest relationships, or select a link line to inspect the shared circulars connecting them.
            </div>`;
    }

    function renderNodeSidebar(node, links) {
        const sortedLinks = [...links].sort((a, b) => b.value - a.value);

        elements.networkSidebar.innerHTML = `
            <div class="sidebar-title">${escapeHtml(node.name)}</div>
            <div class="sidebar-meta-row">
                <span class="tag-pill ${node.tier === 'policy' ? 'tag-policy' : 'tag-admin'}">${escapeHtml(node.tier || 'Topic')}</span>
                <span class="tag-pill tag-division">${links.length} Connected Links</span>
            </div>
            <div class="sidebar-section-heading">Strongest Cross-Domain Connections</div>
            <div class="connection-chip-list">
                ${sortedLinks.slice(0, 8).map(l => {
                    const otherId = (typeof l.source === 'object' ? l.source.id : l.source) === node.id ? (typeof l.target === 'object' ? l.target.id : l.target) : (typeof l.source === 'object' ? l.source.id : l.source);
                    const otherName = getNodeName(otherId);
                    return `
                        <div class="connection-chip" data-other="${otherId}">
                            <span>${escapeHtml(otherName)}</span>
                            <strong>${l.value} shared</strong>
                        </div>`;
                }).join('')}
            </div>
            <button type="button" id="btnOpenNodeDrawer" class="nav-btn nav-btn-primary" style="margin-top:12px;justify-content:center;">
                Inspect Circulars in Drawer →
            </button>
        `;

        elements.networkSidebar.querySelectorAll('.connection-chip').forEach(chip => {
            chip.addEventListener('click', () => {
                const otherId = chip.dataset.other;
                const link = links.find(l => {
                    const s = typeof l.source === 'object' ? l.source.id : l.source;
                    const t = typeof l.target === 'object' ? l.target.id : l.target;
                    return (s === node.id && t === otherId) || (t === node.id && s === otherId);
                });
                if (link) selectNetworkLink(link);
            });
        });

        document.getElementById('btnOpenNodeDrawer').addEventListener('click', () => {
            openDrawer({
                domain: node.id.startsWith('hub_') ? null : node.id,
                search: node.is_hub ? node.name : null,
                title: node.name
            });
        });
    }

    function renderLinkSidebar(link) {
        const sId = typeof link.source === 'object' ? link.source.id : link.source;
        const tId = typeof link.target === 'object' ? link.target.id : link.target;
        const sName = getNodeName(sId);
        const tName = getNodeName(tId);

        elements.networkSidebar.innerHTML = `
            <div class="sidebar-title">${escapeHtml(sName)} ↔ ${escapeHtml(tName)}</div>
            <div class="sidebar-meta-row">
                <span class="tag-pill tag-policy">${link.value} Shared Circulars</span>
                <span class="tag-pill tag-division">Co-occurrence</span>
            </div>
            <p style="font-size:11.5px;color:var(--muted);line-height:1.55;margin-top:4px;">
                These circulars bridge both policy domains through shared secondary tags, cross-referenced statutory provisions, or joint administrative directions.
            </p>
            <button type="button" id="btnOpenLinkDrawer" class="nav-btn nav-btn-primary" style="margin-top:10px;justify-content:center;">
                View All ${link.value} Bridging Circulars →
            </button>
        `;

        document.getElementById('btnOpenLinkDrawer').addEventListener('click', () => {
            openDrawer({
                ids: link.example_ids || [],
                title: `${sName} ↔ ${tName}`
            });
        });
    }

    function getNodeName(nodeOrId) {
        const id = typeof nodeOrId === 'object' ? nodeOrId.id : nodeOrId;
        const n = state.network?.nodes?.find(item => item.id === id);
        return n ? n.name : id;
    }

    // --- View 3: Milestone Chronicle ---

    function renderMilestones() {
        const { milestones: msData } = state;
        if (!msData) return;

        const { eras, milestones } = msData;

        // Render Quick Jump Buttons
        elements.eraButtonsBar.innerHTML = eras.map((era, i) => `
            <button type="button" class="era-pill-btn ${i === 0 ? 'active' : ''}" data-era="${era.id}">
                ${escapeHtml(era.key)}: ${escapeHtml(era.title)}
            </button>
        `).join('');

        elements.eraButtonsBar.querySelectorAll('.era-pill-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                elements.eraButtonsBar.querySelectorAll('.era-pill-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const target = document.getElementById(`era_section_${btn.dataset.era}`);
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });

        // Render Timeline Sections
        elements.milestonesContainer.innerHTML = eras.map(era => {
            const eraMilestones = milestones.filter(m => m.era === era.id);
            return `
                <section class="era-section" id="era_section_${era.id}">
                    <div class="era-header">
                        <div>
                            <h3>${escapeHtml(era.title)} (${escapeHtml(era.key)})</h3>
                            <span>${escapeHtml(era.desc)}</span>
                        </div>
                        <span class="stat-pill"><b>${eraMilestones.length}</b> Major Landmarks</span>
                    </div>
                    <div class="milestone-cards-grid">
                        ${eraMilestones.map(m => `
                            <article class="milestone-card">
                                <div class="milestone-meta-row">
                                    <span class="milestone-badge">${escapeHtml(m.category)}</span>
                                    <span class="milestone-date">${escapeHtml(m.date)} · FY ${escapeHtml(m.year)}</span>
                                </div>
                                <h4 class="milestone-title">${escapeHtml(m.title)}</h4>
                                <p class="milestone-desc">${escapeHtml(m.summary)}</p>
                                <div class="milestone-impact">
                                    <strong>Impact:</strong> ${escapeHtml(m.impact)}
                                </div>
                                <button type="button" class="milestone-action-btn" data-milestone-id="${m.id}">
                                    <span>📄</span> Inspect Founding Circulars (${m.circular_ids?.length || 0})
                                </button>
                            </article>
                        `).join('')}
                    </div>
                </section>
            `;
        }).join('');

        // Wire Milestone Action Buttons
        elements.milestonesContainer.querySelectorAll('.milestone-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const mId = btn.dataset.milestoneId;
                const m = milestones.find(item => item.id === mId);
                if (m) {
                    openDrawer({
                        ids: m.circular_ids || [],
                        title: `Founding Circulars: ${m.title}`
                    });
                }
            });
        });
    }

    // --- View 4: Division Matrix (Heatmap) ---

    function renderMatrix() {
        const { summary } = state;
        if (!summary) return;

        const container = elements.matrixContainer;
        container.innerHTML = '';

        const fys = (summary.financial_years || []).filter(y => y !== 'Unknown');
        const divisions = Object.keys(summary.divisions || {}).sort((a, b) => summary.divisions[b] - summary.divisions[a]);

        // Find max count for color ramp
        let maxVal = 1;
        divisions.forEach(div => {
            fys.forEach(fy => {
                const count = summary.fy_divisions?.[fy]?.[div] || 0;
                if (count > maxVal) maxVal = count;
            });
        });

        const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxVal]);

        const table = document.createElement('table');
        table.className = 'matrix-table';

        // Table Header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th class="row-label">Issuing Wing / Division</th>' +
            fys.map(fy => `<th>${escapeHtml(fy.replace('20', '’').replace('-20', '–'))}</th>`).join('');
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Table Body
        const tbody = document.createElement('tbody');
        divisions.forEach(div => {
            const tr = document.createElement('tr');
            const th = document.createElement('th');
            th.className = 'row-label';
            th.textContent = `${div}`;
            tr.appendChild(th);

            fys.forEach(fy => {
                const count = summary.fy_divisions?.[fy]?.[div] || 0;
                const td = document.createElement('td');
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'matrix-cell-btn';
                btn.textContent = count > 0 ? count : '—';

                if (count > 0) {
                    btn.style.backgroundColor = colorScale(count);
                    btn.style.color = count > maxVal * 0.45 ? '#ffffff' : 'var(--ink)';
                    btn.addEventListener('click', () => {
                        openDrawer({
                            division: div,
                            fy: fy,
                            title: `${div} Circulars (FY ${fy})`
                        });
                    });
                    btn.addEventListener('mouseenter', (e) => {
                        showTooltip(e, `<strong>${escapeHtml(div)}</strong><br>FY ${escapeHtml(fy)}: ${count} circulars`);
                    });
                    btn.addEventListener('mousemove', moveTooltip);
                    btn.addEventListener('mouseleave', hideTooltip);
                } else {
                    btn.style.backgroundColor = '#f8fafc';
                    btn.style.color = 'var(--line-strong)';
                    btn.style.cursor = 'default';
                }

                td.appendChild(btn);
                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        container.appendChild(table);
    }

    // --- Slide-Over Document Drawer ---

    async function openDrawer(params = {}) {
        const { domain, subtopic, division, fy, ids, search, title } = params;

        elements.drawerTitle.textContent = title || 'Matching Circulars';
        elements.drawerSubtitle.textContent = 'Loading records from archive catalog...';
        elements.drawerList.innerHTML = '<div class="empty-state">Loading documents...</div>';
        elements.drawerOverlay.classList.add('open');
        elements.drawer.classList.add('open');
        elements.drawer.setAttribute('aria-hidden', 'false');

        // Lazy load catalog & assignments if not ready
        await ensureAssignmentsAndCatalog();

        let matchingRows = [];
        const { rows } = state.assignments;

        if (ids && Array.isArray(ids)) {
            const idSet = new Set(ids);
            matchingRows = rows.filter(r => idSet.has(r[0]));
        } else {
            matchingRows = rows.filter(r => {
                // r = [id, domain, subtopic, conf, fy, secondaries, tier, division]
                if (domain && r[1] !== domain) return false;
                if (subtopic && r[2] !== subtopic) return false;
                if (division && r[7] !== division) return false;
                if (fy && r[4] !== fy) return false;
                if (state.activeTier !== 'all' && r[6] !== state.activeTier) return false;
                return true;
            });
        }

        state.drawerRows = matchingRows;
        elements.drawerSubtitle.textContent = `${matchingRows.length.toLocaleString('en-IN')} circulars found`;

        renderDrawerCards(matchingRows);

        // Setup instant drawer search
        elements.drawerSearch.value = search || '';
        elements.drawerSearch.oninput = (e) => {
            const q = e.target.value.toLowerCase().trim();
            if (!q) {
                renderDrawerCards(matchingRows);
            } else {
                const filtered = matchingRows.filter(r => {
                    const doc = state.catalog.documents[r[0]];
                    const t = (doc[1] || '').toLowerCase();
                    const c = (doc[2] || '').toLowerCase();
                    const d = (doc[3] || '').toLowerCase();
                    return t.includes(q) || c.includes(q) || d.includes(q);
                });
                renderDrawerCards(filtered);
            }
        };

        if (search) {
            elements.drawerSearch.dispatchEvent(new Event('input'));
        }
    }

    function renderDrawerCards(rows) {
        const list = elements.drawerList;
        list.innerHTML = '';

        if (rows.length === 0) {
            list.innerHTML = '<div class="empty-state">No circulars match the active search or filters.</div>';
            return;
        }

        const maxVisible = 100;
        rows.slice(0, maxVisible).forEach(r => {
            const doc = state.catalog.documents[r[0]];
            if (!doc) return;

            // doc: [serial_no, title, circular_no, date, hindi_pdf_link, english_pdf_link, year, ocr_source]
            const title = doc[1] || 'Untitled Circular';
            const cno = doc[2] || '—';
            const date = doc[3] || '—';
            const hindiLink = doc[4];
            const englishLink = doc[5];
            const fy = doc[6] || '';
            const ocrSource = doc[7];
            const primaryLink = englishLink || hindiLink || '';
            const filename = primaryLink ? primaryLink.split('/').pop().split('?')[0] : '';

            const tier = r[6] || 'policy';
            const division = r[7] || 'Head Office';
            const domainName = state.taxonomy[r[1]]?.name || r[1];

            const card = document.createElement('article');
            card.className = 'circular-card';
            card.innerHTML = `
                <div class="card-top-line">
                    <span class="card-cno">${escapeHtml(cno)}</span>
                    <span>${escapeHtml(date)} · FY ${escapeHtml(fy)}</span>
                </div>
                <h4 class="card-title">${escapeHtml(title)}</h4>
                <div class="card-tags">
                    <span class="tag-pill ${tier === 'policy' ? 'tag-policy' : 'tag-admin'}">${escapeHtml(tier)}</span>
                    <span class="tag-pill tag-division">${escapeHtml(domainName)}</span>
                    <span class="tag-pill tag-division">${escapeHtml(division)}</span>
                    ${hindiLink && englishLink ? '<span class="tag-pill tag-lang-both">EN + HI</span>' : (englishLink ? '<span class="tag-pill tag-lang-single">EN</span>' : '<span class="tag-pill tag-lang-single">HI</span>')}
                </div>
                <div class="card-actions">
                    ${ocrSource > 0 ? `
                        <a href="index.html?fy=${encodeURIComponent(fy)}&doc=${encodeURIComponent(filename)}" target="_blank" class="card-read-btn" title="Open full text ledger view">
                            📄 Read Paper View
                        </a>
                    ` : ''}
                    ${englishLink ? `<a href="${escapeHtml(englishLink)}" target="_blank" rel="noopener noreferrer" class="card-pdf-link">PDF (EN) ↗</a>` : ''}
                    ${hindiLink ? `<a href="${escapeHtml(hindiLink)}" target="_blank" rel="noopener noreferrer" class="card-pdf-link">PDF (HI) ↗</a>` : ''}
                </div>
            `;
            list.appendChild(card);
        });

        if (rows.length > maxVisible) {
            const note = document.createElement('div');
            note.style.cssText = 'padding:14px;text-align:center;color:var(--muted);font-size:12px;font-weight:600;';
            note.textContent = `Showing first ${maxVisible} of ${rows.length.toLocaleString('en-IN')} circulars. Use the search bar to refine.`;
            list.appendChild(note);
        }
    }

    function closeDrawer() {
        elements.drawerOverlay.classList.remove('open');
        elements.drawer.classList.remove('open');
        elements.drawer.setAttribute('aria-hidden', 'true');
    }

    // --- Tooltips & Helpers ---

    function showTooltip(event, htmlContent) {
        elements.tooltip.innerHTML = htmlContent;
        elements.tooltip.style.opacity = '1';
        moveTooltip(event);
    }

    function moveTooltip(event) {
        const x = event.pageX + 12;
        const y = event.pageY + 12;
        elements.tooltip.style.left = `${x}px`;
        elements.tooltip.style.top = `${y}px`;
    }

    function hideTooltip() {
        elements.tooltip.style.opacity = '0';
    }

    function escapeHtml(str) {
        return String(str || '').replace(/[&<>"']/g, m => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
        })[m]);
    }

    function debounce(func, wait) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Start on DOM ready
    document.addEventListener('DOMContentLoaded', init);

})();
