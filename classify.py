#!/usr/bin/env python3
"""
classify.py - Classify EPFO circulars into a structured policy taxonomy.

Implements the specification in plan.md:
- Weighted evidence: Title (x5), Circular No (x3), Filename (x3), OCR text (x1)
- Bilingual matching (English + Hindi keywords)
- Generates compact, optimized data files for explore.html:
  - data/topics/taxonomy.json
  - data/topics/assignments.json
  - data/topics/summary.json
  - data/topics/review.json
"""

import os
import sys
import json
import re
import html
import unicodedata
from collections import defaultdict

# --- Taxonomy Definition ---

TAXONOMY = {
    "pension_eps": {
        "name": "Pension & EPS-95",
        "color": "#059669", # Emerald
        "icon": "user-check",
        "description": "EPS-95 scheme, higher pension, pension calculations, PPOs, and pensioner welfare",
        "subtopics": {
            "higher_pension": {
                "name": "Higher Pension (SC Judgment)",
                "keywords": [
                    "higher pension", "higher wages", "sc judgment", "supreme court judgment on pension",
                    "04.11.2022", "04-11-2022", "para 11(3)", "para 11(4)", "joint option", "joint options",
                    "validation of option", "validation of joint option", "pohw", "para 44",
                    "pension on higher wages", "pension on higher", "उच्चतर वेतन", "उच्च पेंशन"
                ]
            },
            "digital_life_certificate": {
                "name": "Life Certificate & Jeevan Pramaan",
                "keywords": [
                    "jeevan pramaan", "life certificate", "dlc", "biometric", "face authentication",
                    "जीवन प्रमाण", "जीवन प्रमाण-पत्र"
                ]
            },
            "ppo_and_processing": {
                "name": "PPO & Pension Processing",
                "keywords": [
                    "ppo", "pension payment order", "commutation", "family pension", "widow pension",
                    "pensioner", "annuity", "pension disbursement", "disbursement of pension",
                    "पेंशन", "पीपीओ", "पेंशनभोगी"
                ]
            },
            "general_pension": {
                "name": "General EPS-95 Policies",
                "keywords": [
                    "eps", "eps-95", "employees pension scheme", "pension fund", "table-b", "table-d",
                    "पेंशन योजना", "कर्मचारी पेंशन योजना"
                ]
            }
        }
    },
    "compliance_recovery": {
        "name": "Compliance & Recovery",
        "color": "#dc2626", # Red
        "icon": "shield-alert",
        "description": "Establishment coverage, 7A/14B proceedings, default enforcement, inspections and recovery",
        "subtopics": {
            "7a_and_quasi_judicial": {
                "name": "Section 7A, 7Q & 14B Quasi-Judicial",
                "keywords": [
                    "7a", "7q", "14b", "7b", "7c", "damages", "levy of damages", "penal damages",
                    "quasi judicial", "inquiry under section 7a", "7a inquiry", "determination of dues",
                    "धारा 7क", "धारा 14ख", "हर्जाना"
                ]
            },
            "recovery_and_defaulters": {
                "name": "Recovery & Defaulter Actions",
                "keywords": [
                    "recovery", "recovery officer", "recovery certificate", "defaulter", "arrear",
                    "attachment of bank", "auction", "cp-1", "warrant of arrest", "recovery of dues",
                    "वसूली", "बकाया"
                ]
            },
            "coverage_and_inspection": {
                "name": "Coverage, Inspection & Compliance",
                "keywords": [
                    "coverage", "inspection", "compliance", "non-compliance", "survey", "factory",
                    "unorganized", "enforcement officer", "shram suvidha", "compliance monitoring",
                    "कवरेज", "निरीक्षण", "अनुपालन"
                ]
            }
        }
    },
    "exempted_establishments": {
        "name": "Exempted Establishments",
        "color": "#d97706", # Amber
        "icon": "building",
        "description": "Section 17 exemptions, private PF trusts, surrender of exemption, and trust compliance",
        "subtopics": {
            "trust_compliance_audit": {
                "name": "Trust Compliance & Audits",
                "keywords": [
                    "exempted", "exemption", "private trust", "board of trustees", "trustee",
                    "performance of exempted", "monitoring of exempted", "exempted establishment",
                    "छूट प्राप्त", "छूट", "ट्रस्ट"
                ]
            },
            "surrender_and_cancellation": {
                "name": "Surrender & Cancellation of Exemption",
                "keywords": [
                    "surrender of exemption", "cancellation of exemption", "transfer of past accumulations",
                    "relaxation under section", "revocation of exemption"
                ]
            }
        }
    },
    "finance_accounts_invest": {
        "name": "Finance, Accounts & Investment",
        "color": "#2563eb", # Blue
        "icon": "trending-up",
        "description": "Interest rates, budget, investments, banking, balance sheets, and CAG audit",
        "subtopics": {
            "interest_rate": {
                "name": "Interest Rate Declarations",
                "keywords": [
                    "rate of interest", "interest rate", "declaration of rate of interest", "credit of interest",
                    "ब्याज दर", "ब्याज"
                ]
            },
            "investment_and_portfolio": {
                "name": "Investments & Banking",
                "keywords": [
                    "investment", "portfolio manager", "etf", "exchange traded fund", "g-sec", "securities",
                    "banking arrangements", "sbi", "bank reconciliation", "banking transaction", "remittance",
                    "निवेश", "बैंकिंग"
                ]
            },
            "budget_accounts_audit": {
                "name": "Budget, Balance Sheet & Audit",
                "keywords": [
                    "budget", "revised estimates", "budget estimate", "balance sheet", "annual accounts",
                    "cag audit", "internal audit", "audit para", "reconciliation of accounts", "accounting procedure",
                    "बजट", "लेखा", "ऑडिट", "लेखापरीक्षा"
                ]
            }
        }
    },
    "it_digital_services": {
        "name": "IT, UAN & Digital Services",
        "color": "#7c3aed", # Purple
        "icon": "cpu",
        "description": "Universal Account Number (UAN), member portals, e-office, APIs, software, and cybersecurity",
        "subtopics": {
            "uan_and_kyc": {
                "name": "UAN, KYC & Online Portals",
                "keywords": [
                    "uan", "universal account number", "kyc", "aadhaar seeding", "unified portal",
                    "member portal", "employer portal", "ecr", "electronic challan", "online transfer",
                    "यूएएन", "केवाईसी", "आधार"
                ]
            },
            "eoffice_and_software": {
                "name": "Software, E-Office & Downtime",
                "keywords": [
                    "e-office", "eoffice", "ndc", "national data centre", "application software",
                    "portal downtime", "server", "cyber security", "it infrastructure", "data centre",
                    "सॉफ्टवेयर", "ई-ऑफिस"
                ]
            }
        }
    },
    "hr_personnel_cadre": {
        "name": "HR, Personnel & Cadre",
        "color": "#4f46e5", # Indigo
        "icon": "users",
        "description": "Promotions, DPC, seniority lists, transfers, recruitment, APAR, SPARROW, and vigilance",
        "subtopics": {
            "promotions_seniority_dpc": {
                "name": "Promotions, DPC & Seniority",
                "keywords": [
                    "promotion", "dpc", "departmental promotion", "seniority list", "gradation list",
                    "confirmation", "probation", "regularisation", "mcp", "macp", "cadre restructuring",
                    "पदोन्नति", "वरिष्ठता सूची", "डीपीसी"
                ]
            },
            "transfers_and_postings": {
                "name": "Transfers & Postings",
                "keywords": [
                    "transfer of staff", "transfer order", "posting order", "annual general transfer",
                    "agt", "inter regional transfer", "irt", "rotational transfer", "posting",
                    "relieving", "joining report", "additional charge", "charge handover",
                    "transfer policy", "deputation", "स्थानांतरण", "तैनाती"
                ]
            },
            "apar_sparrow": {
                "name": "APAR & SPARROW Appraisal",
                "keywords": [
                    "apar", "sparrow", "annual performance", "annual confidential report", "acr",
                    "timelines for apar", "समीक्षा", "अपार"
                ]
            },
            "recruitment_exams": {
                "name": "Recruitment & Departmental Exams",
                "keywords": [
                    "recruitment", "direct recruitment", "vacancy", "exam", "upsc", "nta", "examination",
                    "skill test", "merit list", "advertisement", "भर्ती", "परीक्षा"
                ]
            },
            "vigilance_and_discipline": {
                "name": "Vigilance & Disciplinary",
                "keywords": [
                    "vigilance", "disciplinary", "charge sheet", "inquiry officer", "suspension", "penalty",
                    "major penalty", "minor penalty", "cvc", "सतर्कता", "अनुशासनात्मक"
                ]
            },
            "benefits_allowances_leave": {
                "name": "Salaries, Allowances & Leave",
                "keywords": [
                    "dearness allowance", "dearness relief", "da", "dr", "hra", "medical reimbursement",
                    "cghs", "bonus", "productivity linked bonus", "ltc", "leave", "pay commission",
                    "महंगाई भत्ता", "छुट्टी", "बोनस"
                ]
            }
        }
    },
    "legal_litigation": {
        "name": "Legal & Court Matters",
        "color": "#0284c7", # Sky
        "icon": "scale",
        "description": "Supreme Court, High Court, CAT cases, panel advocates, and legal dispute monitoring",
        "subtopics": {
            "court_cases_and_orders": {
                "name": "Court Judgments & Writs",
                "keywords": [
                    "supreme court", "high court", "cat", "central administrative tribunal", "slp",
                    "writ petition", "civil appeal", "contempt", "judgment", "order of court", "stay order",
                    "उच्च न्यायालय", "सर्वोच्च न्यायालय", "अदालत"
                ]
            },
            "advocates_and_legal_fees": {
                "name": "Panel Advocates & Legal Fees",
                "keywords": [
                    "panel advocate", "standing counsel", "legal fee", "fee payable to advocate",
                    "briefing", "advocate bill", "panel of advocates", "अधिवक्ता", "वकील"
                ]
            }
        }
    },
    "citizen_services_rti": {
        "name": "Citizen Care & RTI",
        "color": "#ea580c", # Orange
        "icon": "message-circle",
        "description": "Right to Information (RTI), CPGRAMS, EPFiGMS grievances, and member assistance",
        "subtopics": {
            "rti_act": {
                "name": "RTI Applications & Appeals",
                "keywords": [
                    "rti", "right to information", "cpio", "first appellate authority", "cic",
                    "central information commission", "section 6(3)", "rti application",
                    "सूचना का अधिकार", "आरटीआई"
                ]
            },
            "grievances_epfigms": {
                "name": "Grievance Portals (EPFiGMS / CPGRAMS)",
                "keywords": [
                    "grievance", "epfigms", "cpgrams", "complaint", "bhavishya nidhi adalat",
                    "redressal", "citizen charter", "pending grievances", "शिकायत", "अदालत"
                ]
            }
        }
    },
    "training_research": {
        "name": "Training & Academics",
        "color": "#0891b2", # Cyan
        "icon": "book-open",
        "description": "PDUNASS, NATRSS, ZTI training calendars, workshops, seminars, and capacity building",
        "subtopics": {
            "pdunass_natrss": {
                "name": "PDUNASS & ZTI Programs",
                "keywords": [
                    "pdunass", "natrss", "zti", "training", "workshop", "seminar", "induction training",
                    "refresher course", "training calendar", "pandit deendayal", "प्रशिक्षण", "कार्यशाला"
                ]
            }
        }
    },
    "official_language": {
        "name": "Official Language (Rajbhasha)",
        "color": "#be185d", # Pink
        "icon": "globe",
        "description": "Hindi Pakhwada, Rajbhasha inspections, Hindi Diwas, workshops, and quarterly reports",
        "subtopics": {
            "rajbhasha_and_events": {
                "name": "Hindi Pakhwada & Rajbhasha",
                "keywords": [
                    "rajbhasha", "hindi", "hindi pakhwada", "hindi diwas", "rajbhasha sammelan",
                    "hindi workshop", "quarterly progress report", "qpr", "राजभाषा", "हिंदी पखवाड़ा",
                    "हिंदी दिवस", "तिमाही प्रगति रिपोर्ट"
                ]
            }
        }
    },
    "admin_procurement_facilities": {
        "name": "Administration & Procurement",
        "color": "#64748b", # Slate
        "icon": "tool",
        "description": "Vehicles, buildings, GeM procurement, office premises, logistics, tenders, and Swachhata",
        "subtopics": {
            "vehicles_and_transport": {
                "name": "Hiring of Vehicles & Transport",
                "keywords": [
                    "vehicle", "hiring of vehicle", "hiring vehicle", "car", "transport", "driver",
                    "वाहन", "गाड़ी"
                ]
            },
            "buildings_and_premises": {
                "name": "Buildings, Infrastructure & Leases",
                "keywords": [
                    "building", "premises", "office accommodation", "cpwd", "construction", "rent",
                    "lease", "land", "repair", "भवनों", "परिसर"
                ]
            },
            "procurement_and_tenders": {
                "name": "GeM, Tenders & General Admin",
                "keywords": [
                    "procurement", "gem", "tender", "e-procurement", "stationery", "furniture",
                    "swachh bharat", "swachhata", "security agency", "housekeeping", "निविदा", "खरीद"
                ]
            }
        }
    },
    "social_security_schemes": {
        "name": "Schemes & Campaigns",
        "color": "#0d9488", # Teal
        "icon": "award",
        "description": "International Workers (SSA), EEC, PMRPY, ABRY schemes, and EDLI insurance",
        "subtopics": {
            "international_workers_ssa": {
                "name": "International Workers & SSA",
                "keywords": [
                    "international worker", "social security agreement", "ssa", "certificate of coverage",
                    "coc", "bilateral agreement", "प्रवासी"
                ]
            },
            "special_schemes_campaigns": {
                "name": "EEC, PMRPY, ABRY & Special Relief",
                "keywords": [
                    "pmrpy", "abry", "atmanirbhar", "enrolment campaign", "eec", "covid-19 advance",
                    "corona", "pandemic", "edli", "insurance scheme", "अभियान"
                ]
            }
        }
    },
    "governance_cbt": {
        "name": "Governance & CBT",
        "color": "#854d0e", # Brown
        "icon": "landmark",
        "description": "Central Board of Trustees (CBT), Executive Committee, Parliament questions, and annual reports",
        "subtopics": {
            "cbt_and_committees": {
                "name": "CBT & Executive Committee",
                "keywords": [
                    "central board of trustees", "cbt", "executive committee", "agenda of cbt", "minutes of cbt",
                    "annual report of epfo", "parliament question", "lok sabha", "rajya sabha",
                    "केंद्रीय न्यासी बोर्ड", "संसद"
                ]
            }
        }
    },
    "unclassified": {
        "name": "Other / Unclassified",
        "color": "#94a3b8", # Light slate
        "icon": "help-circle",
        "description": "General circulars, miscellaneous notices, and multi-topic announcements",
        "subtopics": {
            "general": {
                "name": "Miscellaneous",
                "keywords": []
            }
        }
    }
}


def normalize_text(text):
    """Normalize text for consistent keyword matching."""
    if not text:
        return ""
    text = html.unescape(str(text))
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    return text


def score_document(title, circular_no, filename, ocr_text):
    """
    Score a document across all taxonomy domains and subtopics using weighted evidence:
      Title:         x5
      Circular No:   x3
      Filename:      x3
      Extracted OCR: x1
    """
    norm_title = normalize_text(title)
    norm_cno = normalize_text(circular_no)
    norm_fname = normalize_text(filename)
    norm_ocr = normalize_text(ocr_text)[:12000] # Cap OCR at first 12k chars for speed & signal

    domain_scores = defaultdict(float)
    subtopic_matches = defaultdict(lambda: defaultdict(float))
    matched_keywords = defaultdict(set)

    for domain_id, domain_info in TAXONOMY.items():
        if domain_id == "unclassified":
            continue

        for subtopic_id, subtopic_info in domain_info["subtopics"].items():
            for kw in subtopic_info["keywords"]:
                kw_norm = normalize_text(kw)
                # Word boundary check for short Latin keywords
                if re.match(r'^[a-z0-9\s\-]+$', kw_norm):
                    escaped = re.escape(kw_norm)
                    pattern = rf'(?:^|[\s,.\-—–/()"\']){escaped}(?:$|[\s,.\-—–/()"\'])'
                else:
                    pattern = re.escape(kw_norm)

                # 1. Title match (5x weight)
                if re.search(pattern, norm_title):
                    pts = 5.0
                    domain_scores[domain_id] += pts
                    subtopic_matches[domain_id][subtopic_id] += pts
                    matched_keywords[domain_id].add(f"title:{kw}")

                # 2. Circular No match (3x weight)
                if re.search(pattern, norm_cno):
                    pts = 3.0
                    domain_scores[domain_id] += pts
                    subtopic_matches[domain_id][subtopic_id] += pts
                    matched_keywords[domain_id].add(f"cno:{kw}")

                # 3. Filename match (3x weight)
                if re.search(pattern, norm_fname):
                    pts = 3.0
                    domain_scores[domain_id] += pts
                    subtopic_matches[domain_id][subtopic_id] += pts
                    matched_keywords[domain_id].add(f"fname:{kw}")

                # 4. OCR text match (1x weight)
                if norm_ocr and re.search(pattern, norm_ocr):
                    pts = 1.0
                    domain_scores[domain_id] += pts
                    subtopic_matches[domain_id][subtopic_id] += pts
                    matched_keywords[domain_id].add(f"ocr:{kw}")

    if not domain_scores:
        return "unclassified", "general", "none", 0.0, []

    # Sort domains by score
    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
    top_domain, top_score = sorted_domains[0]
    second_score = sorted_domains[1][1] if len(sorted_domains) > 1 else 0.0

    # Pick best subtopic
    best_subtopics = subtopic_matches[top_domain]
    if best_subtopics:
        best_subtopic = max(best_subtopics.items(), key=lambda x: x[1])[0]
    else:
        best_subtopic = list(TAXONOMY[top_domain]["subtopics"].keys())[0]

    # Secondary tags
    secondary_domains = [d for d, s in sorted_domains[1:3] if s >= 4.0 and s >= top_score * 0.5]

    # Determine confidence
    if top_score >= 8.0 and (top_score - second_score >= 3.0):
        confidence = "high"
    elif top_score >= 3.0 and (top_score - second_score >= 1.0):
        confidence = "medium"
    elif top_score >= 2.0:
        confidence = "low"
    else:
        return "unclassified", "general", "low", top_score, []

    return top_domain, best_subtopic, confidence, top_score, secondary_domains


def run_classification():
    print("Starting classification pipeline...")

    # 1. Load catalog
    catalog_path = os.path.join('data', 'search', 'catalog.json')
    if not os.path.exists(catalog_path):
        print(f"Error: {catalog_path} not found. Run fetch.py --action search first.")
        sys.exit(1)

    with open(catalog_path, 'r', encoding='utf-8') as f:
        catalog_data = json.load(f)
    documents = catalog_data.get('documents', [])
    print(f"Loaded {len(documents)} circulars from catalog.")

    # 2. Load all index-YYYY-YYYY.json files for OCR text lookup
    all_ocr_data = {}
    data_dir = 'data'
    for fn in os.listdir(data_dir):
        if fn.startswith('index-') and fn.endswith('.json'):
            fp = os.path.join(data_dir, fn)
            try:
                with open(fp, 'r', encoding='utf-8') as f:
                    idx = json.load(f)
                    all_ocr_data.update(idx)
            except Exception as e:
                print(f"Warning: could not read {fn}: {e}")
    print(f"Loaded OCR cache with {len(all_ocr_data)} document texts.")

    # 3. Load overrides if any
    overrides_path = os.path.join('data', 'topic-overrides.json')
    overrides = {}
    if os.path.exists(overrides_path):
        with open(overrides_path, 'r', encoding='utf-8') as f:
            overrides = json.load(f)
        print(f"Loaded {len(overrides)} manual overrides.")

    # 4. Classify each document
    assignments = []
    domain_counts = defaultdict(int)
    subtopic_counts = defaultdict(lambda: defaultdict(int))
    fy_domain_counts = defaultdict(lambda: defaultdict(int))
    fy_subtopic_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    confidence_counts = defaultdict(int)
    fy_confidence_counts = defaultdict(lambda: defaultdict(int))
    language_counts = defaultdict(int)
    fy_language_counts = defaultdict(lambda: defaultdict(int))
    month_counts = defaultdict(lambda: defaultdict(int))
    review_list = []

    for doc_id, doc in enumerate(documents):
        # [serial_no, title, circular_no, date, hindi_pdf_link, english_pdf_link, year, ocr_source]
        title = doc[1] or ''
        circular_no = doc[2] or ''
        date = doc[3] or ''
        hindi_link = doc[4]
        english_link = doc[5]
        fy = doc[6] or 'Unknown'
        ocr_source = doc[7]

        # Determine OCR link and text
        primary_link = english_link if ocr_source == 1 else (hindi_link if ocr_source == 2 else (english_link or hindi_link))
        ocr_text = ''
        if primary_link and primary_link in all_ocr_data:
            ocr_text = all_ocr_data[primary_link].get('ocr_content', '') or ''
            if ocr_text.startswith('OCR_ERROR:'):
                ocr_text = ''

        filename = os.path.basename(primary_link.split('?')[0]) if primary_link else ''

        # Check manual override
        override_key = f"{fy}:{circular_no}" if circular_no else f"{fy}:{title[:30]}"
        if override_key in overrides:
            ov = overrides[override_key]
            domain = ov.get("domain", "unclassified")
            subtopic = ov.get("subtopic", "general")
            confidence = "override"
            secondaries = ov.get("secondary_topics", [])
        else:
            domain, subtopic, confidence, score, secondaries = score_document(
                title=title,
                circular_no=circular_no,
                filename=filename,
                ocr_text=ocr_text
            )

        conf_code = {"high": 3, "medium": 2, "low": 1, "none": 0, "override": 9}.get(confidence, 1)
        assignments.append([doc_id, domain, subtopic, conf_code, fy, secondaries])

        domain_counts[domain] += 1
        subtopic_counts[domain][subtopic] += 1
        fy_domain_counts[fy][domain] += 1
        fy_subtopic_counts[fy][domain][subtopic] += 1
        confidence_counts[conf_code] += 1
        fy_confidence_counts[fy][conf_code] += 1

        if hindi_link and english_link:
            language = "both"
        elif english_link:
            language = "english"
        elif hindi_link:
            language = "hindi"
        else:
            language = "none"
        language_counts[language] += 1
        fy_language_counts[fy][language] += 1

        date_match = re.match(r'^\d{1,2}/(\d{1,2})/\d{4}$', date)
        if date_match:
            month = f"{int(date_match.group(1)):02d}"
            month_counts[fy][month] += 1

        if domain == "unclassified" or confidence == "low":
            review_list.append({
                "id": doc_id,
                "title": title,
                "circular_no": circular_no,
                "fy": fy,
                "confidence": confidence,
                "assigned_domain": domain
            })

    print(f"Classification complete.")
    print("Domain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (c / len(documents)) * 100
        print(f"  {d:<30}: {c:>5} ({pct:>5.1f}%)")

    # 5. Build output files
    out_dir = os.path.join('data', 'topics')
    os.makedirs(out_dir, exist_ok=True)

    # 5a. taxonomy.json
    taxonomy_export = {}
    for d_id, d_data in TAXONOMY.items():
        taxonomy_export[d_id] = {
            "name": d_data["name"],
            "color": d_data["color"],
            "icon": d_data.get("icon", "folder"),
            "description": d_data["description"],
            "count": domain_counts[d_id],
            "subtopics": {
                s_id: {
                    "name": s_data["name"],
                    "count": subtopic_counts[d_id][s_id]
                }
                for s_id, s_data in d_data["subtopics"].items()
            }
        }
    with open(os.path.join(out_dir, 'taxonomy.json'), 'w', encoding='utf-8') as f:
        json.dump(taxonomy_export, f, ensure_ascii=False, indent=2)

    # 5b. assignments.json (compact)
    with open(os.path.join(out_dir, 'assignments.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "version": 2,
            "total_documents": len(assignments),
            "columns": ["id", "domain", "subtopic", "conf", "fy", "secondary_domains"],
            "rows": assignments
        }, f, ensure_ascii=False, separators=(',', ':'))

    # 5c. summary.json (precomputed counts by financial year)
    all_fys = sorted(list(fy_domain_counts.keys()))
    summary_export = {
        "version": 2,
        "total_documents": len(documents),
        "domains": {
            d_id: {
                "name": TAXONOMY[d_id]["name"],
                "color": TAXONOMY[d_id]["color"],
                "total": domain_counts[d_id],
                "subtopics": dict(subtopic_counts[d_id])
            }
            for d_id in TAXONOMY
        },
        "financial_years": all_fys,
        "year_totals": {
            fy: sum(fy_domain_counts[fy].values())
            for fy in all_fys
        },
        "timeline": {
            fy: {d: fy_domain_counts[fy][d] for d in TAXONOMY if fy_domain_counts[fy][d] > 0}
            for fy in all_fys
        },
        "subtopic_timeline": {
            fy: {
                d: {
                    s: count
                    for s, count in fy_subtopic_counts[fy][d].items()
                    if count > 0
                }
                for d in TAXONOMY
                if fy_subtopic_counts[fy][d]
            }
            for fy in all_fys
        },
        "confidence_counts": {str(code): confidence_counts[code] for code in sorted(confidence_counts)},
        "confidence_timeline": {
            fy: {str(code): count for code, count in sorted(fy_confidence_counts[fy].items())}
            for fy in all_fys
        },
        "language_counts": dict(language_counts),
        "language_timeline": {
            fy: dict(fy_language_counts[fy])
            for fy in all_fys
        },
        "month_timeline": {
            fy: dict(month_counts[fy])
            for fy in all_fys
        }
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_export, f, ensure_ascii=False, indent=2)

    # 5d. review.json
    with open(os.path.join(out_dir, 'review.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "unclassified_count": domain_counts["unclassified"],
            "review_count": len(review_list),
            "sample": review_list[:100]
        }, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated topic assets in {out_dir}:")
    print(f"  - taxonomy.json    ({os.path.getsize(os.path.join(out_dir, 'taxonomy.json')) // 1024} KB)")
    print(f"  - assignments.json ({os.path.getsize(os.path.join(out_dir, 'assignments.json')) // 1024} KB)")
    print(f"  - summary.json     ({os.path.getsize(os.path.join(out_dir, 'summary.json')) // 1024} KB)")
    print(f"  - review.json      ({os.path.getsize(os.path.join(out_dir, 'review.json')) // 1024} KB)")


if __name__ == "__main__":
    run_classification()
