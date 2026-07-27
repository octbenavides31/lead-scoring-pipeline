"""Tests for the scoring pipeline. Run: python -m unittest discover -s tests"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer import (  # noqa: E402
    TIER_A_MIN,
    TIER_B_MIN,
    Features,
    LlmScorer,
    RuleScorer,
    assign_tier,
    extract_features,
    rank,
    render_html,
)

TODAY = date(2026, 7, 25)


def features(**kwargs) -> Features:
    base = {
        "distress_signals": ("none",),
        "is_absentee": False,
        "equity_ratio": 0.0,
        "days_since_event": 9999,
    }
    base.update(kwargs)
    return Features(**base)


class TestExtractFeatures(unittest.TestCase):
    def test_absentee_when_addresses_differ(self) -> None:
        f = extract_features(
            {"situs_address": "1 A St", "mailing_address": "2 B St"}, today=TODAY)
        self.assertTrue(f.is_absentee)

    def test_not_absentee_when_addresses_match_ignoring_case_and_space(self) -> None:
        f = extract_features(
            {"situs_address": " 1 A St ", "mailing_address": "1 a st"}, today=TODAY)
        self.assertFalse(f.is_absentee)

    def test_missing_mailing_address_is_not_treated_as_absentee(self) -> None:
        f = extract_features({"situs_address": "1 A St", "mailing_address": ""}, today=TODAY)
        self.assertFalse(f.is_absentee)

    def test_equity_ratio(self) -> None:
        f = extract_features({"assessed_value": 200000, "amount_owed": 50000}, today=TODAY)
        self.assertAlmostEqual(f.equity_ratio, 0.75)

    def test_equity_is_clamped_and_never_divides_by_zero(self) -> None:
        self.assertEqual(extract_features({"assessed_value": 0}, today=TODAY).equity_ratio, 0.0)
        underwater = extract_features(
            {"assessed_value": 100000, "amount_owed": 150000}, today=TODAY)
        self.assertEqual(underwater.equity_ratio, 0.0)

    def test_missing_event_date_sorts_last(self) -> None:
        self.assertEqual(extract_features({}, today=TODAY).days_since_event, 9999)

    def test_days_since_event(self) -> None:
        f = extract_features({"event_date": "2026-07-10"}, today=TODAY)
        self.assertEqual(f.days_since_event, 15)


class TestRuleScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = RuleScorer()

    def test_no_signals_scores_zero(self) -> None:
        score, reasons = self.scorer.score(features())
        self.assertEqual(score, 0)
        self.assertEqual(reasons, ("no scoring signals",))

    def test_uses_the_strongest_distress_signal_not_the_sum(self) -> None:
        both, _ = self.scorer.score(features(distress_signals=("tax_delinquent", "code_violation")))
        worst, _ = self.scorer.score(features(distress_signals=("tax_delinquent",)))
        self.assertEqual(both, worst, "signals must not stack past the category weight")

    def test_components_add_up(self) -> None:
        score, reasons = self.scorer.score(features(
            distress_signals=("foreclosure",), is_absentee=True,
            equity_ratio=1.0, days_since_event=10))
        self.assertEqual(score, 100)
        self.assertEqual(len(reasons), 4)

    def test_score_is_capped_at_100(self) -> None:
        score, _ = self.scorer.score(features(
            distress_signals=("foreclosure",), is_absentee=True,
            equity_ratio=1.0, days_since_event=0))
        self.assertLessEqual(score, 100)

    def test_recency_bands(self) -> None:
        fresh, _ = self.scorer.score(features(days_since_event=30))
        mid, _ = self.scorer.score(features(days_since_event=90))
        stale, _ = self.scorer.score(features(days_since_event=91))
        self.assertEqual((fresh, mid, stale), (15, 7, 0))

    def test_every_reason_is_attributable(self) -> None:
        _, reasons = self.scorer.score(features(is_absentee=True, equity_ratio=0.5))
        self.assertTrue(all("+" in r for r in reasons))


class TestTiers(unittest.TestCase):
    def test_boundaries_are_inclusive_at_the_bottom(self) -> None:
        self.assertEqual(assign_tier(TIER_A_MIN), "A")
        self.assertEqual(assign_tier(TIER_A_MIN - 1), "B")
        self.assertEqual(assign_tier(TIER_B_MIN), "B")
        self.assertEqual(assign_tier(TIER_B_MIN - 1), "C")

    def test_extremes(self) -> None:
        self.assertEqual(assign_tier(100), "A")
        self.assertEqual(assign_tier(0), "C")


class TestRank(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {"parcel_id": "B-2", "situs_address": "1 A St", "mailing_address": "1 A St",
             "distress_signals": ["none"]},
            {"parcel_id": "B-1", "situs_address": "2 B St", "mailing_address": "2 B St",
             "distress_signals": ["none"]},
            {"parcel_id": "A-1", "situs_address": "3 C St", "mailing_address": "9 Z St",
             "distress_signals": ["foreclosure"], "assessed_value": 100, "amount_owed": 0,
             "event_date": "2026-07-20"},
        ]

    def test_sorted_by_score_descending(self) -> None:
        leads = rank(self.records, today=TODAY)
        self.assertEqual(leads[0].parcel_id, "A-1")
        self.assertEqual([l.score for l in leads], sorted((l.score for l in leads), reverse=True))

    def test_ties_break_on_parcel_id_for_stable_output(self) -> None:
        leads = rank(self.records, today=TODAY)
        tied = [l.parcel_id for l in leads if l.score == 0]
        self.assertEqual(tied, ["B-1", "B-2"])


class TestReport(unittest.TestCase):
    def test_escapes_html_in_record_fields(self) -> None:
        leads = rank([{"parcel_id": "<script>alert(1)</script>",
                       "situs_address": "1 & 2 St", "distress_signals": ["none"]}], today=TODAY)
        out = render_html(leads, "2026-07-25")
        self.assertNotIn("<script>alert(1)</script>", out)
        self.assertIn("&amp;", out)


class TestLlmScorerSeam(unittest.TestCase):
    def test_is_deliberately_unimplemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            LlmScorer().score(features())


if __name__ == "__main__":
    unittest.main()
