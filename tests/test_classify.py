import os
import sys
import unittest

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classify import (
    classify_document,
    extract_division,
    determine_tier,
    generate_network_data,
    generate_milestones_data,
)


class TestClassify(unittest.TestCase):
    def test_recovery_of_unclassified_interest_credit(self):
        title = "Prompt Interest credit @8.25% for 2025-2026 in CITES 2.01 - Regarding"
        cno = "No:WSU/5(1)2005/Annual Accts/2026-2027/E-789534/23"
        domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
        self.assertEqual(domain, "finance_accounts_invest")
        self.assertEqual(subtopic, "interest_rate")
        self.assertEqual(tier, "policy")
        self.assertEqual(division, "WSU")

    def test_recovery_of_unclassified_amnesty_compliance(self):
        title = "Launch of AMNESTY, 2026 for regularization of exemption status of Provident Fund Trusts recognized under Income Tax Act"
        cno = "No.: Exemption/AMNESTY-2026/[E.III/10(58)/2025]"
        domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
        self.assertIn(domain, ["exempted_establishments", "compliance_recovery"])
        self.assertEqual(tier, "policy")
        self.assertEqual(division, "Exemption")

    def test_recovery_of_audit_manual(self):
        title = "Modifications in the Audit Manual-Reconstitution of Committee 'A' & Committee 'B'"
        cno = "No. Audit/ 1(07)2023/QuarterlymeetingCPFC/-Part(1)/18"
        domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
        self.assertEqual(domain, "finance_accounts_invest")
        self.assertEqual(tier, "policy")
        self.assertEqual(division, "Finance")

    def test_internal_admin_tier_tagging(self):
        title = "Final Seniority list in the cadre of Section Officer as on 31.08.2024"
        cno = "No. HRM-IV/28(6)2018/SO/SeniorityList /254"
        domain, subtopic, conf, secondaries, tier, division = classify_document(title, cno, "", "")
        self.assertEqual(tier, "admin")
        self.assertEqual(division, "HRM")

    def test_network_and_milestone_schema(self):
        mock_assignments = [
            [0, "pension_eps", "higher_pension", 3, "2022-2023", ["legal_litigation"], "policy", "Pension"],
            [1, "compliance_recovery", "7a_and_quasi_judicial", 3, "2023-2024", ["legal_litigation"], "policy", "Compliance"],
            [2, "hr_personnel_cadre", "promotions_seniority_dpc", 3, "2024-2025", [], "admin", "HRM"],
            [3, "it_digital_services", "uan_and_kyc", 3, "2024-2025", ["citizen_services_rti"], "policy", "WSU"],
        ]
        mock_docs = [
            ["0", "Higher Pension SC Order", "P-1", "04/11/2022", None, "http://pdf", "2022-2023", 1],
            ["1", "7A Inquiry Guidelines", "C-1", "01/01/2023", None, "http://pdf", "2023-2024", 1],
            ["2", "Seniority List", "H-1", "01/01/2024", None, "http://pdf", "2024-2025", 1],
            ["3", "UAN Activation Portal Update", "W-1", "15/02/2025", None, "http://pdf", "2024-2025", 1],
        ]
        net = generate_network_data(mock_assignments, mock_docs)
        self.assertIn("nodes", net)
        self.assertIn("links", net)
        self.assertTrue(any(n["id"] == "pension_eps" for n in net["nodes"]))
        self.assertTrue(any(n["id"] == "legal_litigation" for n in net["nodes"]))
        self.assertGreater(len(net["links"]), 0)

        ms = generate_milestones_data(mock_docs)
        self.assertIn("eras", ms)
        self.assertIn("milestones", ms)
        self.assertGreater(len(ms["milestones"]), 0)
        self.assertIn("impact", ms["milestones"][0])


if __name__ == '__main__':
    unittest.main()
