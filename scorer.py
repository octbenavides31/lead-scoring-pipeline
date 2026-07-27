"""Turn normalized property records into a ranked, tiered lead list.

The scorer is deliberately deterministic and rule-based. That is not a placeholder
for "real" ML - it is the baseline you need in order to evaluate anything fancier.
See Scorer / LlmScorer below and the README section "Why the baseline is the point."

Input records use the schema produced by the public-records-scraper repo.
"""
from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Protocol

FIXTURES = Path(__file__).parent / "fixtures"

# Weights sum to 100 so a score reads as a percentage of the maximum.
WEIGHTS = {"distress": 45, "absentee": 20, "equity": 20, "recency": 15}

DISTRESS_POINTS = {
    "foreclosure": 45,
    "tax_delinquent": 35,
    "probate": 30,
    "code_violation": 20,
    "none": 0,
}

TIER_A_MIN = 70
TIER_B_MIN = 45


@dataclass(frozen=True)
class Features:
    """Everything the scorer is allowed to look at. Extracted once, then frozen."""

    distress_signals: tuple[str, ...]
    is_absentee: bool
    equity_ratio: float          # 0.0 - 1.0
    days_since_event: int

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "distress_signals": list(self.distress_signals)}


@dataclass(frozen=True)
class ScoredLead:
    parcel_id: str
    situs_address: str
    score: int
    tier: str
    reasons: tuple[str, ...]
    features: Features

    def as_dict(self) -> dict[str, Any]:
        return {
            "parcel_id": self.parcel_id,
            "situs_address": self.situs_address,
            "score": self.score,
            "tier": self.tier,
            "reasons": list(self.reasons),
            "features": self.features.as_dict(),
        }


def extract_features(record: dict[str, Any], today: date | None = None) -> Features:
    """Pull scoring inputs out of a raw record. Pure, so it is trivially testable."""
    today = today or date.today()

    signals = tuple(record.get("distress_signals") or ["none"])

    situs = (record.get("situs_address") or "").strip().lower()
    mailing = (record.get("mailing_address") or "").strip().lower()
    is_absentee = bool(situs and mailing and situs != mailing)

    assessed = float(record.get("assessed_value") or 0)
    owed = float(record.get("amount_owed") or 0)
    equity_ratio = 0.0 if assessed <= 0 else max(0.0, min(1.0, (assessed - owed) / assessed))

    event = record.get("event_date")
    if event:
        days = (today - date.fromisoformat(event)).days
    else:
        days = 9999
    return Features(signals, is_absentee, equity_ratio, max(0, days))


class Scorer(Protocol):
    """Any scorer must return (score 0-100, human-readable reasons)."""

    def score(self, features: Features) -> tuple[int, tuple[str, ...]]:
        ...


class RuleScorer:
    """Transparent, deterministic baseline.

    Every point is attributable to a named reason, which matters twice: a caller
    working the list can see why a lead ranked, and any replacement scorer has a
    fixed reference to be measured against.
    """

    def score(self, features: Features) -> tuple[int, tuple[str, ...]]:
        reasons: list[str] = []

        best = max((DISTRESS_POINTS.get(s, 0) for s in features.distress_signals), default=0)
        distress = min(WEIGHTS["distress"], best)
        if distress:
            top = max(features.distress_signals, key=lambda s: DISTRESS_POINTS.get(s, 0))
            reasons.append(f"{top.replace('_', ' ')} (+{distress})")

        absentee = WEIGHTS["absentee"] if features.is_absentee else 0
        if absentee:
            reasons.append(f"absentee owner (+{absentee})")

        equity = round(WEIGHTS["equity"] * features.equity_ratio)
        if equity:
            reasons.append(f"{features.equity_ratio:.0%} estimated equity (+{equity})")

        if features.days_since_event <= 30:
            recency = WEIGHTS["recency"]
        elif features.days_since_event <= 90:
            recency = WEIGHTS["recency"] // 2
        else:
            recency = 0
        if recency:
            reasons.append(f"event {features.days_since_event}d ago (+{recency})")

        total = min(100, distress + absentee + equity + recency)
        return total, tuple(reasons or ("no scoring signals",))


class LlmScorer:
    """Interface sketch for an LLM-based scorer. Intentionally not implemented.

    The point of the seam is that a replacement must satisfy the same contract:
    a 0-100 score plus attributable reasons. Before trusting one in production you
    would run it against the labeled set alongside RuleScorer and compare rank
    correlation and tier agreement - not just eyeball a few outputs. See the README.
    """

    def score(self, features: Features) -> tuple[int, tuple[str, ...]]:
        raise NotImplementedError(
            "Deliberately unimplemented. Validate against RuleScorer before adding one."
        )


def assign_tier(score: int) -> str:
    """A >= 70, B >= 45, else C. Boundaries are inclusive at the bottom."""
    if score >= TIER_A_MIN:
        return "A"
    if score >= TIER_B_MIN:
        return "B"
    return "C"


def rank(records: Iterable[dict[str, Any]], scorer: Scorer | None = None,
         today: date | None = None) -> list[ScoredLead]:
    scorer = scorer or RuleScorer()
    leads: list[ScoredLead] = []
    for record in records:
        features = extract_features(record, today=today)
        score, reasons = scorer.score(features)
        leads.append(ScoredLead(
            parcel_id=record.get("parcel_id", ""),
            situs_address=record.get("situs_address", ""),
            score=score,
            tier=assign_tier(score),
            reasons=reasons,
            features=features,
        ))
    # Ties break on parcel_id so output is stable run to run - diffs stay meaningful.
    return sorted(leads, key=lambda l: (-l.score, l.parcel_id))


def render_html(leads: list[ScoredLead], generated_on: str) -> str:
    """Self-contained HTML report. No external assets, opens anywhere."""
    counts = {t: sum(1 for l in leads if l.tier == t) for t in ("A", "B", "C")}
    rows = "\n".join(
        f"    <tr class='t{l.tier}'><td>{l.tier}</td><td>{l.score}</td>"
        f"<td>{html.escape(l.parcel_id)}</td><td>{html.escape(l.situs_address)}</td>"
        f"<td>{html.escape(', '.join(l.reasons))}</td></tr>"
        for l in leads
    )
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Lead Report</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#16161a}}
 h1{{margin:0 0 .25rem}} p.meta{{color:#666;margin:0 0 1.5rem}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #e4e2df;vertical-align:top}}
 th{{background:#f5f4f2;font-size:13px;letter-spacing:.03em;text-transform:uppercase}}
 .tA td:first-child{{color:#116b43;font-weight:700}}
 .tB td:first-child{{color:#8a6d1f;font-weight:700}}
 .tC td:first-child{{color:#888}}
</style></head><body>
<h1>Lead Report</h1>
<p class="meta">{len(leads)} leads &middot; Tier A: {counts['A']} &middot;
 Tier B: {counts['B']} &middot; Tier C: {counts['C']} &middot; generated {generated_on}</p>
<table>
  <thead><tr><th>Tier</th><th>Score</th><th>Parcel</th><th>Address</th><th>Why</th></tr></thead>
  <tbody>
{rows}
  </tbody>
</table>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Score and rank property leads.")
    parser.add_argument("--records", type=Path, default=FIXTURES / "records.json")
    parser.add_argument("--json-out", type=Path, default=Path("ranked.json"))
    parser.add_argument("--html-out", type=Path, default=Path("report.html"))
    args = parser.parse_args()

    records = json.loads(args.records.read_text(encoding="utf-8"))
    leads = rank(records)
    today = date.today().isoformat()

    args.json_out.write_text(
        json.dumps([l.as_dict() for l in leads], indent=2), encoding="utf-8")
    args.html_out.write_text(render_html(leads, today), encoding="utf-8")

    counts = {t: sum(1 for l in leads if l.tier == t) for t in ("A", "B", "C")}
    print(f"{len(leads)} leads scored -> {args.json_out}, {args.html_out}")
    print(f"  Tier A: {counts['A']}   Tier B: {counts['B']}   Tier C: {counts['C']}")
    for lead in leads[:3]:
        print(f"  {lead.tier} {lead.score:>3}  {lead.parcel_id}  {', '.join(lead.reasons)}")


if __name__ == "__main__":
    main()
