"""Έλεγχοι της σημασίας των ετικετών, χωρίς εκπαίδευση νέων μοντέλων."""
import unittest

import pandas as pd

from prepare_data import prepare_data


def case(**changes):
    row = dict(case_id=1, sc_type="MI SC", delay_minutes=60,
               alternative_provided="YES", airline="TEST", airline_type="LEGACY",
               pap_request="REBOOKING", solution="REJECTED",
               rejection_reason="NOT ELIGIBLE", full_refund_allowed="YES",
               rebooking_allowed="NO")
    row.update(changes)
    return row


class LabelTests(unittest.TestCase):
    def prepare(self, **changes):
        raw = pd.DataFrame([case(**changes)])
        before = raw.copy(deep=True)
        data, audit = prepare_data(raw)
        pd.testing.assert_frame_equal(raw, before)
        return data.iloc[0], audit

    def test_rejected_rebooking_preserves_refund(self):
        row, _ = self.prepare()
        self.assertEqual(row.full_refund_allowed, "YES")
        self.assertEqual(row.rebooking_allowed, "NO")

    def test_rejected_refund_preserves_rebooking(self):
        row, _ = self.prepare(pap_request="REFUND", full_refund_allowed="NO",
                              rebooking_allowed="YES")
        self.assertEqual(row.full_refund_allowed, "NO")
        self.assertEqual(row.rebooking_allowed, "YES")

    def test_no_availability_is_not_no_entitlement(self):
        row, _ = self.prepare(rejection_reason="NO AVAIL", rebooking_allowed="YES")
        self.assertEqual(row.rebooking_allowed, "YES")
        row, _ = self.prepare(rejection_reason="NO AVAIL", rebooking_allowed=None)
        self.assertTrue(pd.isna(row.rebooking_allowed))

    def test_rejected_without_reason_does_not_fill_no(self):
        row, _ = self.prepare(rejection_reason=None, rebooking_allowed=None)
        self.assertTrue(pd.isna(row.rebooking_allowed))

    def test_confirmed_ineligibility_fills_only_requested_target(self):
        row, _ = self.prepare(rebooking_allowed=None, full_refund_allowed=None)
        self.assertEqual(row.rebooking_allowed, "NO")
        self.assertTrue(pd.isna(row.full_refund_allowed))

    def test_successful_rebooking_fills_only_rebooking(self):
        row, _ = self.prepare(solution="REBOOKING", rejection_reason=None,
                              rebooking_allowed=None, full_refund_allowed=None)
        self.assertEqual(row.rebooking_allowed, "YES")
        self.assertTrue(pd.isna(row.full_refund_allowed))

    def test_refund_is_not_automatically_full_refund(self):
        row, audit = self.prepare(solution="REFUND", pap_request="REFUND",
                                  rejection_reason=None, full_refund_allowed="NO")
        self.assertEqual(row.full_refund_allowed, "NO")
        self.assertIn("REVIEW", set(audit.severity))
        row, _ = self.prepare(solution="REFUND", full_refund_allowed=None)
        self.assertTrue(pd.isna(row.full_refund_allowed))
        row, _ = self.prepare(solution="FULL REFUND", full_refund_allowed=None)
        self.assertEqual(row.full_refund_allowed, "YES")

    def test_conflict_excludes_only_affected_target(self):
        row, audit = self.prepare(rebooking_allowed="YES")
        self.assertTrue(pd.isna(row.rebooking_allowed))
        self.assertEqual(row.rebooking_allowed_source, "conflict")
        self.assertEqual(row.full_refund_allowed, "YES")
        self.assertIn("CONFLICT", set(audit.severity))

    def test_cancellation_delay_only_without_alternative(self):
        row, _ = self.prepare(sc_type="CR SC", alternative_provided="NO", delay_minutes=None)
        self.assertEqual(row.delay_minutes, 0)
        row, _ = self.prepare(sc_type="CR SC", alternative_provided="YES", delay_minutes=None)
        self.assertTrue(pd.isna(row.delay_minutes))
        row, _ = self.prepare(sc_type="CR SC", alternative_provided="YES", delay_minutes=300)
        self.assertEqual(row.delay_minutes, 300)

    def test_ignored_columns_do_not_change_labels(self):
        a, _ = self.prepare(rerouting_allowed="YES", codeshare_allowed="YES", policy_exception="YES")
        b, _ = self.prepare(rerouting_allowed="NO", codeshare_allowed="NO", policy_exception="NO")
        for key in ("full_refund_allowed", "rebooking_allowed"):
            self.assertEqual(a[key], b[key])

    def test_normalization_and_invalid_target(self):
        row, _ = self.prepare(pap_request=" rebooking ", solution=" rejected ",
                              rejection_reason=" not eligible ", rebooking_allowed=None)
        self.assertEqual(row.rebooking_allowed, "NO")
        row, _ = self.prepare(rebooking_allowed="MAYBE")
        self.assertTrue(pd.isna(row.rebooking_allowed))
        self.assertEqual(row.rebooking_allowed_source, "conflict")

    def test_real_cases(self):
        # Σταθερά παραδείγματα από το αρχικό CSV. Το editable dataset μπορεί να διορθωθεί.
        rows = [case(case_id=i) for i in [86, 29]]
        rows += [case(case_id=i, rejection_reason="NO AVAIL", rebooking_allowed="YES")
                 for i in [24, 25]]
        rows += [case(case_id=33, rebooking_allowed="YES"),
                 case(case_id=110, pap_request="REFUND", solution="REFUND",
                      rejection_reason=None, full_refund_allowed="NO"),
                 case(case_id=14, solution="REBOOKING", rejection_reason=None,
                      full_refund_allowed=None, rebooking_allowed=None),
                 case(case_id=9, pap_request="REFUND", solution="REFUND",
                      rejection_reason=None, rebooking_allowed=None)]
        data, _ = prepare_data(pd.DataFrame(rows))
        data = data.set_index("case_id")
        for case_id in [86, 29]:
            self.assertEqual(data.at[case_id, "full_refund_allowed"], "YES")
            self.assertEqual(data.at[case_id, "rebooking_allowed"], "NO")
        for case_id in [24, 25]:
            self.assertEqual(data.at[case_id, "rebooking_allowed"], "YES")
        self.assertTrue(pd.isna(data.at[33, "rebooking_allowed"]))
        self.assertEqual(data.at[33, "full_refund_allowed"], "YES")
        self.assertEqual(data.at[110, "full_refund_allowed"], "NO")
        self.assertEqual(data.at[14, "rebooking_allowed"], "YES")
        self.assertTrue(pd.isna(data.at[14, "full_refund_allowed"]))
        self.assertTrue(pd.isna(data.at[9, "rebooking_allowed"]))


if __name__ == "__main__":
    unittest.main()
