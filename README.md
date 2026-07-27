# lead-scoring-pipeline

[![tests](https://github.com/octbenavides31/lead-scoring-pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/octbenavides31/lead-scoring-pipeline/actions/workflows/tests.yml)

Takes normalized property records, extracts features, scores and tiers them, and emits ranked JSON plus a self-contained HTML report.

```bash
python scorer.py
python -m unittest discover -s tests
```

No dependencies, no network. Consumes the schema produced by [`public-records-scraper`](../public-records-scraper).

```
$ python scorer.py
8 leads scored -> ranked.json, report.html
  Tier A: 3   Tier B: 1   Tier C: 4
  A  93  SMP-000104  foreclosure (+45), absentee owner (+20), 67% estimated equity (+13), event 9d ago (+15)
  A  88  SMP-000108  tax delinquent (+35), absentee owner (+20), 92% estimated equity (+18), event 26d ago (+15)
  A  86  SMP-000102  tax delinquent (+35), absentee owner (+20), 81% estimated equity (+16), event 17d ago (+15)
```

*(Your exact scores will differ slightly, because recency is a live feature and the same fixture ages as days pass. That is deliberate. A lead list that doesn't decay is lying to you.)*

## Why the baseline is the point

The scorer is a transparent rule engine, and that is a deliberate choice rather than a placeholder for something smarter.

Every point it assigns is attributable to a named reason. That does two jobs. The person actually working the list can see *why* a lead ranked where it did. "Foreclosure, absentee, 67% equity" tells you how to open the call, where a bare `0.87` tells you nothing. And more importantly, it gives you a fixed reference to measure any replacement against.

`LlmScorer` exists as an interface with the same contract, score plus attributable reasons, and it deliberately raises `NotImplementedError`. The seam is the design. The implementation is the part you have to earn.

**Before I'd trust an LLM scorer in production**, I'd want the following against a held-out labeled set: rank correlation with the baseline (does it broadly agree, and where doesn't it?), tier-agreement rate (how often does a lead move tiers, and is that movement defensible?), stability under re-runs at temperature 0, cost and latency per thousand records, and a manual read of every case where the two scorers disagree by more than one tier. Those disagreements are where you find out whether the model learned something real or just learned to be confidently vague.

If it can't beat a rule engine that took an afternoon to write, it isn't worth the cost, the latency, or the debugging surface. Most of the time the honest answer is a hybrid: rules for the signals you can define, and a model only for the genuinely fuzzy ones.

## Scoring model

| Component | Max | How it's earned |
|---|---|---|
| Distress | 45 | Strongest signal only. Foreclosure 45, tax delinquent 35, probate 30, code violation 20 |
| Absentee | 20 | Mailing address differs from the property address |
| Equity | 20 | Scaled by estimated equity ratio, clamped to 0 through 1 |
| Recency | 15 | Full within 30 days, half within 90, zero after |

Tiers: **A** ≥ 70, **B** ≥ 45, **C** below.

Two details that matter more than they look:

**Distress signals don't stack.** A property with both a tax delinquency and a code violation scores the same as the tax delinquency alone. Stacking would let three minor flags outrank an active foreclosure, which is exactly backwards.

**Ties break on parcel ID.** Output is stable across runs, so a diff between two days shows real movement instead of reordering noise.

## What this demonstrates

Feature extraction as pure functions, deterministic baseline before ML, interface seams for swappable components, explainable scoring, evaluation criteria for model replacement, boundary-condition testing, HTML escaping in generated reports, and stable sort ordering for diffable output.

## Layout

```
scorer.py                  # feature extraction, rule scorer, tiering, HTML report
fixtures/records.json      # 8 synthetic records
tests/test_scorer.py       # 19 tests, including tier boundaries and clamping
```

All data is invented. Any resemblance to a real parcel, person, or address is coincidental.

## License

MIT
