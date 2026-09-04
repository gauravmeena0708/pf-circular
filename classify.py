#!/usr/bin/env python3
"""
classify.py - Classify EPFO circulars into a structured policy taxonomy.

Implements the Policy Intelligence Hub specification:
- Weighted evidence: Title (x5), Circular No (x3), Filename (x3), OCR text (x1)
- Bilingual matching (English + Hindi keywords)
- Recovered coverage across ~1,900 previously unclassified documents
- Tier classification: 'policy' (Public Policy & Schemes) vs 'admin' (Internal Administration)
- Originating division extraction (WSU, Pension, Compliance, Finance, Legal, etc.)
- Real-data cross-domain network generation (data/topics/network.json)
- Policy milestone chronicle generation (data/topics/milestones.json)
- Compact, optimized data files for explore.html:
  - data/topics/taxonomy.json
  - data/topics/assignments.json
  - data/topics/summary.json
  - data/topics/network.json
  - data/topics/milestones.json
  - data/topics/review.json
"""

import os
import sys
import json
import re
import html
import unicodedata
from collections import defaultdict, Counter

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
                    "facial authentication", "जीवन प्रमाण", "जीवन प्रमाण-पत्र"
                ]
            },
            "ppo_and_processing": {
                "name": "PPO & Pension Processing",
                "keywords": [
                    "ppo", "pension payment order", "commutation", "family pension", "widow pension",
                    "pensioner", "annuity", "pension disbursement", "disbursement of pension",
                    "dearness relief to central government pensioners", "dearness relief to pensioners",
                    "da enhancement pensioners", "coverage under ccs (pension) rules", "pensioners",
                    "पेंशन", "पीपीओ", "पेंशनभोगी"
                ]
            },
            "general_pension": {
                "name": "General EPS-95 Policies",
                "keywords": [
                    "eps", "eps-95", "employees pension scheme", "pension fund", "table-b", "table-d",
                    "pension scheme", "pension calculation", "पेंशन योजना", "कर्मचारी पेंशन योजना"
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
                    "quasi judicial", "quasi-judicial", "inquiry under section 7a", "7a inquiry",
                    "determination of dues", "damages under section 14b", "sections 125 and 128",
                    "code on social security", "coss", "authorised officers and officers competent to levy and recover damages",
                    "chapter iii of the code on social security", "धारा 7क", "धारा 14ख", "हर्जाना"
                ]
            },
            "recovery_and_defaulters": {
                "name": "Recovery & Defaulter Actions",
                "keywords": [
                    "recovery", "recovery officer", "recovery certificate", "defaulter", "arrear",
                    "attachment of bank", "auction", "cp-1", "warrant of arrest", "recovery of dues",
                    "ibc", "insolvency", "nclt", "resolution plan", "vishwas", "amicable settlement of disputes",
                    "assessment or levy of pd in ibc", "वसूली", "बकाया"
                ]
            },
            "coverage_and_inspection": {
                "name": "Coverage, Inspection & Compliance",
                "keywords": [
                    "coverage", "inspection", "compliance", "non-compliance", "survey", "factory",
                    "unorganized", "enforcement officer", "shram suvidha", "compliance monitoring",
                    "enrolment campaign", "employees' enrolment campaign", "कवरेज", "निरीक्षण", "अनुपालन"
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
                    "amnesty", "regularization of exemption", "provident fund trusts", "exemption status",
                    "private pf trusts", "section 17", "monthly review of the regional office on exemption",
                    "sop on exemption", "छूट प्राप्त", "छूट", "ट्रस्ट"
                ]
            },
            "surrender_and_cancellation": {
                "name": "Surrender & Cancellation of Exemption",
                "keywords": [
                    "surrender of exemption", "cancellation of exemption", "transfer of past accumulations",
                    "relaxation under section", "revocation of exemption", "surrender and cancellation"
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
                    "interest credit", "prompt interest", "prompt interest credit", "annual accts",
                    "rate of interest of q1", "rate of interest for q4", "staff provident fund in epfo",
                    "spf in epfo", "ब्याज दर", "ब्याज"
                ]
            },
            "investment_and_portfolio": {
                "name": "Investments & Banking",
                "keywords": [
                    "investment", "portfolio manager", "etf", "exchange traded fund", "g-sec", "securities",
                    "banking arrangements", "sbi", "bank reconciliation", "banking transaction", "remittance",
                    "remittances", "physical transfer of remittances", "hiring of consultants in investment",
                    "निवेश", "बैंकिंग"
                ]
            },
            "budget_accounts_audit": {
                "name": "Budget, Balance Sheet & Audit",
                "keywords": [
                    "budget", "revised estimates", "budget estimate", "balance sheet", "annual accounts",
                    "cag audit", "internal audit", "audit para", "reconciliation of accounts", "accounting procedure",
                    "audit manual", "committee 'a' & committee 'b'", "core areas for the internal audit",
                    "schedule of fixed assets", "classification of assets", "rates of depreciation", "depreciation",
                    "budget circular", "re 20", "be 20", "बजट", "लेखा", "ऑडिट", "लेखापरीक्षा"
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
                    "joint declaration", "de-linking of mids", "delinking of mids", "erroneous linking",
                    "form 121", "form 15g", "form 15h", "यूएएन", "केवाईसी", "आधार"
                ]
            },
            "eoffice_and_software": {
                "name": "Software, E-Office & Downtime",
                "keywords": [
                    "e-office", "eoffice", "ndc", "national data centre", "application software",
                    "portal downtime", "scheduled downtime", "server", "cyber security", "it infrastructure",
                    "data centre", "samadhan setu", "issue tracker", "samadhan", "cites 2.0", "cites 2.01",
                    "cites", "downtime of the e-office", "सॉफ्टवेयर", "ई-ऑफिस"
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
                "name": "Promotions, Seniority & DPC",
                "keywords": [
                    "seniority", "seniority list", "promotion", "dpc", "departmental promotion committee",
                    "regularization", "confirmation of", "clearance of probation", "probation clearance",
                    "adhoc promotion", "macp", "varishtha", "वरिष्ठता", "पदोन्नति"
                ]
            },
            "recruitment_exams": {
                "name": "Recruitment & Departmental Exams",
                "keywords": [
                    "recruitment", "direct recruitment", "probationary examination", "departmental exam",
                    "ldce", "limited departmental", "upsc", "ssc", "assistant provident fund commissioner exam",
                    "enforcement officer exam", "section officer exam", "ldc/jsa", "marks of candidates",
                    "भर्ती", "परीक्षा"
                ]
            },
            "transfers_and_postings": {
                "name": "Transfers, Postings & Rotations",
                "keywords": [
                    "transfer", "posting", "agt", "annual general transfer", "relieving", "joining",
                    "inter regional transfer", "compassionate transfer", "transfer policy", "स्थानांतरण", "तैनाती"
                ]
            },
            "apar_sparrow": {
                "name": "APAR & SPARROW Portals",
                "keywords": [
                    "apar", "sparrow", "annual performance assessment", "performance report",
                    "timelines for apar", "timeline for apar", "submission of apar", "वार्षिक कार्य निष्पादन"
                ]
            },
            "benefits_allowances_leave": {
                "name": "Leave, Allowance & Medical",
                "keywords": [
                    "dearness allowance", "da", "hra", "house rent allowance", "medical", "cghs",
                    "leave", "child care leave", "ccl", "ltc", "leave travel", "bonus", "productivity linked bonus",
                    "holiday", "declaration of holiday", "dr. b.r. ambedkar", "bakrid", "id-u-zuha",
                    "permission for foreign visit", "staff welfare fund", "sports activities", "sports promotion",
                    "staff provident fund", "spf", "छुट्टी", "भत्ता", "अवकाश"
                ]
            },
            "vigilance_and_discipline": {
                "name": "Vigilance & Disciplinary Matters",
                "keywords": [
                    "vigilance", "disciplinary proceedings", "suspension", "charge sheet", "inquiry officer",
                    "cvo", "central vigilance", "major penalty", "minor penalty", "sensitive posts",
                    "sensitive and non-sensitive", "epf staff (cca) rules", "service regulations, 2025",
                    "review committee for suspension", "dar", "सतर्कता", "निलंबन"
                ]
            }
        }
    },
    "legal_litigation": {
        "name": "Legal & Court Matters",
        "color": "#0284c7", # Sky
        "icon": "scale",
        "description": "Supreme Court, High Courts, CAT cases, legal notices, panel advocates, and contempt matters",
        "subtopics": {
            "court_cases_and_orders": {
                "name": "Court Cases & Judgments",
                "keywords": [
                    "supreme court", "high court", "cat", "central administrative tribunal", "slp",
                    "writ petition", "court order", "stay order", "contempt petition", "legal notice",
                    "judgement", "judgment", "न्यायालय", "अदालत"
                ]
            },
            "advocates_and_panels": {
                "name": "Panel Advocates & Legal Fees",
                "keywords": [
                    "panel advocate", "empanelment of advocates", "legal fee", "advocate fee", "counsel",
                    "standing counsel", "legal advisor", "वकील"
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
                    "central information commission", "section 6(3)", "rti application", "transfer of rti",
                    "सूचना का अधिकार", "आरटीआई"
                ]
            },
            "grievances_epfigms": {
                "name": "Grievance Portals (EPFiGMS / CPGRAMS)",
                "keywords": [
                    "grievance", "epfigms", "cpgrams", "complaint", "bhavishya nidhi adalat",
                    "redressal", "citizen charter", "pending grievances", "nidhi aapke nikat",
                    "consumer forum", "शिकायत", "अदालत"
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
                    "refresher course", "training calendar", "pandit deendayal", "igot", "karmayogi",
                    "course for audit officers", "ilo-epfo training", "actuarial practices",
                    "प्रशिक्षण", "कार्यशाला"
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
                    "hindi workshop", "quarterly progress report", "qpr", "hindi pakhwarha",
                    "hindi classes", "hindi typing", "राजभाषा", "हिंदी पखवाड़ा", "हिंदी पखवाड़ा",
                    "हिंदी दिवस", "तिमाही प्रगति रिपोर्ट", "राजभाषा सम्मेलन"
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
                    "commercial vehicle", "वाहन", "गाड़ी"
                ]
            },
            "buildings_and_premises": {
                "name": "Buildings, Infrastructure & Leases",
                "keywords": [
                    "building", "premises", "office accommodation", "cpwd", "construction", "rent",
                    "lease", "land", "repair", "residential accommodation", "भवनों", "परिसर"
                ]
            },
            "procurement_and_tenders": {
                "name": "GeM, Tenders & General Admin",
                "keywords": [
                    "procurement", "gem", "tender", "e-procurement", "stationery", "furniture",
                    "swachh bharat", "swachhata", "security agency", "housekeeping", "office maintenance",
                    "निविदा", "खरीद"
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
                    "coc", "bilateral agreement", "overseas bank accounts", "प्रवासी"
                ]
            },
            "special_schemes_campaigns": {
                "name": "EEC, PMRPY, ABRY & Special Relief",
                "keywords": [
                    "pmrpy", "abry", "atmanirbhar", "enrolment campaign", "eec", "covid-19 advance",
                    "corona", "pandemic", "edli", "insurance scheme", "special campaign", "अभियान"
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
                    "annual report of epfo", "parliament question", "lok sabha", "rajya sabha", "monsoon session",
                    "calendar of sittings", "central staff welfare committee", "sports promotion board",
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


def extract_division(circular_no, title):
    """Extract the originating EPFO division from circular number and title."""
    text = f"{circular_no or ''} {title or ''}"
    text_upper = text.upper()
    
    if re.search(r'\b(WSU|WEB SERVICES|PORTAL|PASSBOOK|JOINT DECLARATION)\b', text_upper):
        return "WSU"
    if re.search(r'\b(PENSION|EPS|PPO|JEEVAN PRAMAAN)\b', text_upper):
        return "Pension"
    if re.search(r'\b(CAIU)\b', text_upper):
        return "CAIU"
    if re.search(r'\b(COMPLIANCE|C-IV|C-II|7A|14B|7Q|RECOVERY|DEFAULTER|DAMAGES)\b', text_upper):
        return "Compliance"
    if re.search(r'\b(EXEMPTION|EXEMPTED|TRUST)\b', text_upper):
        return "Exemption"
    if re.search(r'\b(INV|INVESTMENT|BSC|FINANCE|BUDGET|ACCOUNTS|AUDIT|SPF)\b', text_upper):
        return "Finance"
    if re.search(r'\b(LEGAL|LC|COURT|ADVOCATE|TRIBUNAL|CAT|SUPREME COURT)\b', text_upper):
        return "Legal"
    if re.search(r'\b(IS|NDC|E-OFFICE|EOFFICE|SAMADHAN|SOFTWARE)\b', text_upper):
        return "IS"
    if re.search(r'\b(PDUNASS|NATRSS|ZTI|TRAINING)\b', text_upper):
        return "PDUNASS"
    if re.search(r'\b(CSD|RTI|GRIEVANCE|EPFIGMS|NIDHI AAPKE NIKAT|CPGRAMS)\b', text_upper):
        return "CSD"
    if re.search(r'\b(CBT|COORD|BOARD|PARLIAMENT)\b', text_upper):
        return "Coordination"
    if re.search(r'\b(RAJBHASHA|HINDI|हिन्दी|राजभाषा)\b', text_upper):
        return "Rajbhasha"
    if re.search(r'\b(HRM|HRD|CADRE|DPC|APAR|SENIORITY|TRANSFER|POSTING|EXAM)\b', text_upper):
        return "HRM"
    return "Head Office"


def determine_tier(domain, subtopic, title):
    """
    Tag every circular as:
    - 'policy': Public policy, welfare, schemes, member/employer rules
    - 'admin': Internal personnel, staff exams, cadre, transfers, routine bureaucracy
    """
    title_norm = (title or "").lower()
    
    # Domains that are predominantly public policy
    policy_domains = {
        "pension_eps",
        "compliance_recovery",
        "exempted_establishments",
        "finance_accounts_invest",
        "it_digital_services",
        "social_security_schemes",
        "citizen_services_rti"
    }

    # Domains that are predominantly internal administration
    admin_domains = {
        "hr_personnel_cadre",
        "training_research",
        "official_language",
        "admin_procurement_facilities"
    }

    if domain in policy_domains:
        if "staff provident fund" in title_norm or "holiday" in title_norm:
            return "admin"
        return "policy"

    if domain in admin_domains:
        if any(kw in title_norm for kw in ["international worker", "higher pension", "section 7a", "exemption", "aadhaar"]):
            return "policy"
        return "admin"

    if domain == "legal_litigation":
        if any(kw in title_norm for kw in ["seniority", "cadre", "promotion", "transfer", "service matter", "tribunal order in cadre"]):
            return "admin"
        return "policy"

    if domain == "governance_cbt":
        return "policy"

    return "admin"


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
    norm_ocr = normalize_text(ocr_text)[:12000]

    domain_scores = defaultdict(float)
    subtopic_matches = defaultdict(lambda: defaultdict(float))
    matched_keywords = defaultdict(set)

    for domain_id, domain_info in TAXONOMY.items():
        if domain_id == "unclassified":
            continue

        for subtopic_id, subtopic_info in domain_info["subtopics"].items():
            for kw in subtopic_info["keywords"]:
                kw_norm = normalize_text(kw)
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
    secondary_domains = [d for d, s in sorted_domains[1:3] if s >= 4.0 and s >= top_score * 0.4]

    # Determine confidence
    if top_score >= 6.0 and (top_score - second_score >= 2.0):
        confidence = "high"
    elif top_score >= 3.0 and (top_score - second_score >= 1.0):
        confidence = "medium"
    elif top_score >= 2.0:
        confidence = "low"
    else:
        return "unclassified", "general", "low", top_score, []

    return top_domain, best_subtopic, confidence, top_score, secondary_domains


def classify_document(title, circular_no, filename="", ocr_text=""):
    """Public helper to classify a document, returning domain, subtopic, conf_code, secondaries, tier, division."""
    domain, subtopic, conf, score, secondaries = score_document(title, circular_no, filename, ocr_text)
    conf_map = {"high": 3, "medium": 2, "low": 1, "none": 0, "override": 9}
    conf_code = conf_map.get(conf, 1)
    division = extract_division(circular_no, title)
    tier = determine_tier(domain, subtopic, title)
    return domain, subtopic, conf_code, secondaries, tier, division


def generate_network_data(assignments, documents, min_strength=None):
    """
    Generate real cross-domain and statutory hub co-occurrence network data.
    Output: { "nodes": [...], "links": [...] }
    """
    if min_strength is None:
        min_strength = 1 if len(assignments) < 50 else 2
    # 1. Base domain nodes
    nodes = []
    node_set = set()
    for d_id, d_data in TAXONOMY.items():
        if d_id == "unclassified":
            continue
        nodes.append({
            "id": d_id,
            "name": d_data["name"],
            "color": d_data["color"],
            "tier": "policy" if d_id in ["pension_eps", "compliance_recovery", "exempted_establishments", "finance_accounts_invest", "it_digital_services", "social_security_schemes", "citizen_services_rti"] else "admin"
        })
        node_set.add(d_id)

    # 2. Add statutory / policy hub nodes
    hub_defs = [
        ("hub_higher_pension", "Higher Pension (SC)", "#047857", "higher pension|higher wages|sc judgment|para 11"),
        ("hub_7a_14b", "Section 7A & 14B Quasi-Judicial", "#b91c1c", "7a|14b|7q|damages"),
        ("hub_exemption_trust", "Exempted Trusts & Section 17", "#b45309", "exempted|exemption|private trust|board of trustees"),
        ("hub_uan_kyc", "UAN, KYC & Unified Portal", "#6d28d9", "uan|kyc|aadhaar|unified portal"),
        ("hub_coss", "Code on Social Security 2020", "#be123c", "code on social security|coss"),
        ("hub_interest_roi", "EPF Interest Rate Declarations", "#1d4ed8", "rate of interest|interest rate|prompt interest"),
        ("hub_covid_relief", "COVID-19 Pandemic Advances", "#0f766e", "covid|pandemic|corona|pmgky"),
        ("hub_international_workers", "International Workers & SSA", "#0e7490", "international worker|social security agreement|ssa|coc")
    ]

    hub_patterns = {}
    for hub_id, hub_name, hub_color, pat in hub_defs:
        nodes.append({
            "id": hub_id,
            "name": hub_name,
            "color": hub_color,
            "is_hub": True,
            "tier": "policy"
        })
        node_set.add(hub_id)
        hub_patterns[hub_id] = re.compile(pat, re.I)

    # 3. Aggregate co-occurrences
    link_counts = Counter()
    link_examples = defaultdict(list)

    for r in assignments:
        doc_id = r[0]
        p_domain = r[1]
        secondaries = r[5] if len(r) > 5 else []
        doc_title = documents[doc_id][1] if doc_id < len(documents) else ""

        # Primary <-> Secondary domain links
        if p_domain in node_set:
            for s in secondaries:
                if s in node_set and s != p_domain:
                    pair = tuple(sorted([p_domain, s]))
                    link_counts[pair] += 1
                    if len(link_examples[pair]) < 5:
                        link_examples[pair].append(doc_id)

        # Hub <-> Domain links
        for hub_id, pat in hub_patterns.items():
            if pat.search(doc_title):
                if p_domain in node_set and p_domain != hub_id:
                    pair = tuple(sorted([hub_id, p_domain]))
                    link_counts[pair] += 1
                    if len(link_examples[pair]) < 5:
                        link_examples[pair].append(doc_id)

    # 4. Format links array
    links = []
    for (src, tgt), count in link_counts.most_common():
        if count >= min_strength:
            links.append({
                "source": src,
                "target": tgt,
                "value": count,
                "example_ids": link_examples[(src, tgt)]
            })

    return {
        "nodes": nodes,
        "links": links
    }


def generate_milestones_data(documents):
    """
    Curate 15 landmark policy moments across 2009-2027 organized into 6 eras.
    """
    eras = [
        {"id": "era1", "key": "2009-2013", "title": "Computerization & Centralization", "desc": "Transition from local regional ledgers to core computerized databases."},
        {"id": "era2", "key": "2014-2017", "title": "UAN & Digital India Leap", "desc": "Launch of the Universal Account Number (UAN) and nationwide Aadhaar seeding."},
        {"id": "era3", "key": "2018-2020", "title": "Mobile Services & Ease of Business", "desc": "UMANG app services, automated claim settlement, and ECR 2.0 electronic returns."},
        {"id": "era4", "key": "2020-2022", "title": "COVID-19 Emergency Relief", "desc": "Special non-refundable pandemic advances and PMGKY contribution subsidies."},
        {"id": "era5", "key": "2022-2024", "title": "Higher Pension Watershed", "desc": "Implementation of Supreme Court judgment on higher pension and online joint options."},
        {"id": "era6", "key": "2024-2027", "title": "Next-Gen CITES 2.0 & Statutory Reform", "desc": "CITES 2.0 architecture revamp, Code on Social Security 2020, and Amnesty 2026."}
    ]

    milestone_specs = [
        {
            "id": "ms_cbs_start",
            "era": "era1",
            "date": "15/09/2010",
            "year": "2010-2011",
            "title": "National Database & Core Centralization",
            "category": "Digital Systems",
            "summary": "Initiation of central electronic data interchange and national subscriber database consolidation.",
            "impact": "Layed the structural groundwork for portable account numbers across regional offices.",
            "query": re.compile(r"national data|centralized|computerization", re.I)
        },
        {
            "id": "ms_interest_2011",
            "era": "era1",
            "date": "08/04/2011",
            "year": "2010-2011",
            "title": "Declaration of Historic 9.50% Interest Rate",
            "category": "Finance",
            "summary": "Central Board approved 9.50% interest rate credited into member accounts for 2010-11.",
            "impact": "Highest declared interest rate in recent EPFO operational history.",
            "query": re.compile(r"9\.50%|rate of interest.*2010-11", re.I)
        },
        {
            "id": "ms_uan_launch",
            "era": "era2",
            "date": "01/10/2014",
            "year": "2014-2015",
            "title": "Launch of Universal Account Number (UAN)",
            "category": "Member Services",
            "summary": "Mandated a single permanent UAN for every contributing employee across job changes.",
            "impact": "Eliminated repetitive manual PF account transfers during job switches for millions of workers.",
            "query": re.compile(r"universal account number|uan.*allotment", re.I)
        },
        {
            "id": "ms_aadhaar_mandate",
            "era": "era2",
            "date": "20/06/2015",
            "year": "2015-2016",
            "title": "Aadhaar Seeding & Digital Life Certificates",
            "category": "Digital Systems",
            "summary": "Integration of Aadhaar with UAN and Jeevan Pramaan biometric verification for pensioners.",
            "impact": "Pensioners no longer required physical visits to bank branches for annual life certificates.",
            "query": re.compile(r"aadhaar|jeevan pramaan|biometric.*life certificate", re.I)
        },
        {
            "id": "ms_eec_2017",
            "era": "era2",
            "date": "02/01/2017",
            "year": "2016-2017",
            "title": "Employees' Enrolment Campaign 2017",
            "category": "Compliance",
            "summary": "Special amnesty and compliance drive encouraging voluntary declaration of unorganized workers.",
            "impact": "Over 10 million previously informal workers enrolled into formal social security coverage.",
            "query": re.compile(r"enrolment campaign.*2017|eec", re.I)
        },
        {
            "id": "ms_umang_mobile",
            "era": "era3",
            "date": "23/11/2017",
            "year": "2017-2018",
            "title": "Mobile Passbook & Services on UMANG App",
            "category": "Member Services",
            "summary": "Rollout of e-passbook viewing, claim submission, and pension tracking on mobile handsets.",
            "impact": "Transformed EPFO from counter-based field office queues into a 24x7 smartphone self-service.",
            "query": re.compile(r"umang|mobile app|epassbook|e-passbook", re.I)
        },
        {
            "id": "ms_exempted_monitoring",
            "era": "era3",
            "date": "14/05/2019",
            "year": "2019-2020",
            "title": "Standardized Audit of Private Exempted Trusts",
            "category": "Exemptions",
            "summary": "Introduced strict computerized compliance and investment guidelines for private PF trusts.",
            "impact": "Protected retirement savings of millions of employees in large private corporate trusts.",
            "query": re.compile(r"performance of exempted|private trust|monitoring of exempted", re.I)
        },
        {
            "id": "ms_covid_advance",
            "era": "era4",
            "date": "28/03/2020",
            "year": "2019-2020",
            "title": "Emergency COVID-19 Non-Refundable Advance",
            "category": "Relief",
            "summary": "Notified emergency amendment allowing up to 75% non-refundable advance during national lockdowns.",
            "impact": "Disbursed tens of thousands of crores in emergency liquidity within 72 hours of application.",
            "query": re.compile(r"covid|corona|pandemic.*advance", re.I)
        },
        {
            "id": "ms_pmgky_subsidy",
            "era": "era4",
            "date": "15/04/2020",
            "year": "2020-2021",
            "title": "PMGKY Government Contribution Subvention",
            "category": "Relief",
            "summary": "Government paid complete 24% employer and employee PF contributions for small businesses.",
            "impact": "Prevented mass layoffs in micro, small, and medium enterprises during pandemic disruptions.",
            "query": re.compile(r"pmgky|pradhan mantri garib kalyan", re.I)
        },
        {
            "id": "ms_sc_pension_judgment",
            "era": "era5",
            "date": "04/11/2022",
            "year": "2022-2023",
            "title": "Supreme Court Higher Pension Landmark Ruling",
            "category": "Pension",
            "summary": "Supreme Court upheld validity of 2014 EPS amendments while granting window for higher wage options.",
            "impact": "Enabled over 1.7 million employees to apply for higher monthly pension calculations.",
            "query": re.compile(r"04\.11\.2022|supreme court.*pension|higher wages.*judgment", re.I)
        },
        {
            "id": "ms_joint_declaration_sop",
            "era": "era5",
            "date": "22/08/2023",
            "year": "2023-2024",
            "title": "Standard Operating Procedure on Joint Declarations",
            "category": "Member Services",
            "summary": "Completely digitized the member profile correction process (name, DOB, father's name, DOJ).",
            "impact": "Drastically reduced rejection of claim settlements caused by demographic discrepancies.",
            "query": re.compile(r"joint declaration|profile correction|sop.*joint", re.I)
        },
        {
            "id": "ms_higher_pension_validation",
            "era": "era5",
            "date": "01/06/2023",
            "year": "2023-2024",
            "title": "Online Portal for Joint Options Validation",
            "category": "Pension",
            "summary": "Deployed automated backend for employers and field offices to validate pension on higher wages.",
            "impact": "Standardized the wage verification and differential contribution transfer calculations.",
            "query": re.compile(r"validation of joint option|pohw|higher wages.*option", re.I)
        },
        {
            "id": "ms_coss_officers",
            "era": "era6",
            "date": "13/07/2026",
            "year": "2026-2027",
            "title": "Code on Social Security 2020 Officer Notification",
            "category": "Compliance",
            "summary": "Notified authorized officers to initiate proceedings under Section 125 & 128 of the Social Security Code.",
            "impact": "Operational milestone transitioning statutory compliance to the modern Code on Social Security.",
            "query": re.compile(r"code on social security|authorised officers.*125", re.I)
        },
        {
            "id": "ms_amnesty_2026",
            "era": "era6",
            "date": "11/07/2026",
            "year": "2026-2027",
            "title": "Launch of AMNESTY 2026 & VISHWAS 2026",
            "category": "Compliance",
            "summary": "Dual regularisation schemes for unexempted private trusts and dispute settlement on 14B damages.",
            "impact": "Fast-tracked resolution of long-standing employer disputes and trust regularizations.",
            "query": re.compile(r"amnesty.*2026|vishwas.*2026", re.I)
        },
        {
            "id": "ms_cites_2026",
            "era": "era6",
            "date": "09/07/2026",
            "year": "2026-2027",
            "title": "Next-Gen CITES 2.01 & Automated Interest Credit",
            "category": "Digital Systems",
            "summary": "Deployment of CITES 2.01 with prompt 8.25% annual interest credit automation across accounts.",
            "impact": "Significantly shortened account settlement turnarounds and eliminated annual interest credit lags.",
            "query": re.compile(r"cites 2\.0|prompt interest.*cites", re.I)
        }
    ]

    # Resolve matching circular IDs from catalog
    milestones_output = []
    for spec in milestone_specs:
        matched_ids = []
        for doc_id, doc in enumerate(documents):
            text = f"{doc[1] or ''} {doc[2] or ''}"
            if spec["query"].search(text):
                matched_ids.append(doc_id)
                if len(matched_ids) >= 8:
                    break

        milestones_output.append({
            "id": spec["id"],
            "era": spec["era"],
            "date": spec["date"],
            "year": spec["year"],
            "title": spec["title"],
            "category": spec["category"],
            "summary": spec["summary"],
            "impact": spec["impact"],
            "circular_ids": matched_ids
        })

    return {
        "eras": eras,
        "milestones": milestones_output
    }


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
    
    tier_counts = defaultdict(int)
    fy_tier_counts = defaultdict(lambda: defaultdict(int))
    division_counts = defaultdict(int)
    fy_division_counts = defaultdict(lambda: defaultdict(int))
    division_domain_counts = defaultdict(lambda: defaultdict(int))

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
        tier = determine_tier(domain, subtopic, title)
        division = extract_division(circular_no, title)

        assignments.append([doc_id, domain, subtopic, conf_code, fy, secondaries, tier, division])

        domain_counts[domain] += 1
        subtopic_counts[domain][subtopic] += 1
        fy_domain_counts[fy][domain] += 1
        fy_subtopic_counts[fy][domain][subtopic] += 1
        confidence_counts[conf_code] += 1
        fy_confidence_counts[fy][conf_code] += 1

        tier_counts[tier] += 1
        fy_tier_counts[fy][tier] += 1
        division_counts[division] += 1
        fy_division_counts[fy][division] += 1
        division_domain_counts[division][domain] += 1

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

    print(f"\nSignal Tier Distribution:")
    for t, c in sorted(tier_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (c / len(documents)) * 100
        print(f"  {t:<30}: {c:>5} ({pct:>5.1f}%)")

    print(f"\nTop Originating Divisions:")
    for div, c in sorted(division_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = (c / len(documents)) * 100
        print(f"  {div:<30}: {c:>5} ({pct:>5.1f}%)")

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
            "version": 3,
            "total_documents": len(assignments),
            "columns": ["id", "domain", "subtopic", "conf", "fy", "secondary_domains", "tier", "division"],
            "rows": assignments
        }, f, ensure_ascii=False, separators=(',', ':'))

    # 5c. summary.json (precomputed counts by financial year, tiers, divisions)
    all_fys = sorted(list(fy_domain_counts.keys()))
    summary_export = {
        "version": 3,
        "total_documents": len(documents),
        "tiers": dict(tier_counts),
        "fy_tiers": {fy: dict(fy_tier_counts[fy]) for fy in all_fys},
        "divisions": dict(division_counts),
        "fy_divisions": {fy: dict(fy_division_counts[fy]) for fy in all_fys},
        "division_domains": {div: dict(division_domain_counts[div]) for div in division_counts},
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

    # 5d. network.json (co-occurrence graph)
    network_data = generate_network_data(assignments, documents)
    with open(os.path.join(out_dir, 'network.json'), 'w', encoding='utf-8') as f:
        json.dump(network_data, f, ensure_ascii=False, indent=2)

    # 5e. milestones.json (policy eras and landmark chronicle)
    milestones_data = generate_milestones_data(documents)
    with open(os.path.join(out_dir, 'milestones.json'), 'w', encoding='utf-8') as f:
        json.dump(milestones_data, f, ensure_ascii=False, indent=2)

    # 5f. review.json
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
    print(f"  - network.json     ({os.path.getsize(os.path.join(out_dir, 'network.json')) // 1024} KB)")
    print(f"  - milestones.json  ({os.path.getsize(os.path.join(out_dir, 'milestones.json')) // 1024} KB)")
    print(f"  - review.json      ({os.path.getsize(os.path.join(out_dir, 'review.json')) // 1024} KB)")


if __name__ == "__main__":
    run_classification()
