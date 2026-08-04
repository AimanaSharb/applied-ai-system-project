#!/usr/bin/env python3
"""
Evaluation harness for the natural-language parsing step.

Scores two things per case:
  correctness  did the parse land on an acceptable genre, mood, and count?
               Each case allows a set of values, since more than one mapping is
               often defensible.
  consistency  each case runs twice; a parse that changes between runs fails.

    python tests/eval_harness.py
    python tests/eval_harness.py --provider gemini --runs 3

Exits 0 if every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import nl_recommender as nl  # noqa: E402
from src.llm import NLRecommenderError, Provider, build_provider  # noqa: E402


@dataclass(frozen=True)
class EvalCase:
    request: str
    genres: Set[str]
    moods: Set[str]
    k: Optional[int] = None
    note: str = ""


EVAL_CASES: Sequence[EvalCase] = (
    EvalCase(
        request="I want something chill for studying",
        genres={"lofi", "ambient"},
        moods={"chill", "focused", "relaxed"},
        note="activity -> mood inference",
    ),
    EvalCase(
        request="high energy music for my workout",
        genres={"pop", "rock"},
        moods={"intense"},
        note="activity -> high energy",
    ),
    EvalCase(
        request="some jazz to relax to",
        genres={"jazz"},
        moods={"relaxed", "chill"},
        note="explicit genre",
    ),
    EvalCase(
        request="upbeat pop to start my morning",
        genres={"pop", "indie pop"},
        moods={"happy"},
        note="explicit genre + implied mood",
    ),
    EvalCase(
        request="moody synth music for a long night drive",
        genres={"synthwave"},
        moods={"moody"},
        note="genre named loosely ('synth')",
    ),
    EvalCase(
        request="give me 3 ambient tracks for deep focus",
        genres={"ambient", "lofi"},
        moods={"focused", "chill"},
        k=3,
        note="explicit count must be honoured",
    ),
    EvalCase(
        request="loud aggressive rock, nothing gentle",
        genres={"rock"},
        moods={"intense"},
        note="negation should not flip the mood",
    ),
    EvalCase(
        request="happy indie songs for a rooftop party",
        genres={"indie pop", "pop"},
        moods={"happy"},
        note="two-word genre",
    ),
)


@dataclass
class CaseResult:
    case: EvalCase
    parses: List[nl.ParsedRequest] = field(default_factory=list)
    error: str = ""

    @property
    def first(self) -> Optional[nl.ParsedRequest]:
        return self.parses[0] if self.parses else None

    @property
    def genre_ok(self) -> bool:
        return bool(self.first) and self.first.genre in self.case.genres

    @property
    def mood_ok(self) -> bool:
        return bool(self.first) and self.first.mood in self.case.moods

    @property
    def k_ok(self) -> bool:
        if self.case.k is None:
            return True
        return bool(self.first) and self.first.k == self.case.k

    @property
    def consistent(self) -> bool:
        """Did every run agree on genre, mood and k?"""
        if len(self.parses) < 2:
            return False
        signatures = {(p.genre, p.mood, p.k) for p in self.parses}
        return len(signatures) == 1

    @property
    def passed(self) -> bool:
        return (
            not self.error
            and self.genre_ok
            and self.mood_ok
            and self.k_ok
            and self.consistent
        )

    def observed(self) -> str:
        if self.error:
            return "ERROR"
        return " | ".join(f"{p.genre}/{p.mood}/{p.k}" for p in self.parses)


def evaluate_case(
    case: EvalCase, catalog: nl.Catalog, provider: Provider, runs: int = 2
) -> CaseResult:
    """Parse one case `runs` times and collect the results."""
    result = CaseResult(case=case)
    for _ in range(runs):
        try:
            result.parses.append(
                nl.parse_request(case.request, catalog, provider=provider)
            )
        except NLRecommenderError as exc:
            result.error = str(exc)
            break
    return result


def evaluate_all(
    catalog: nl.Catalog,
    provider: Provider,
    runs: int = 2,
    cases: Sequence[EvalCase] = EVAL_CASES,
) -> List[CaseResult]:
    return [evaluate_case(c, catalog, provider, runs=runs) for c in cases]


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def render_report(results: Sequence[CaseResult], runs: int, provider_label: str) -> str:
    """Build the pass/fail summary table and total score."""
    request_w = max(len(r.case.request) for r in results)
    request_w = min(max(request_w, 24), 46)
    observed_w = max(20, max(len(r.observed()) for r in results))

    header = (
        f"{'REQUEST':<{request_w}}  {'GENRE':<6} {'MOOD':<6} {'K':<6} "
        f"{'CONSIST':<8} {'RESULT':<7} OBSERVED (genre/mood/k per run)"
    )
    lines = [
        "=" * len(header),
        f"  PARSER EVALUATION — {len(results)} cases x {runs} runs — {provider_label}",
        "=" * len(header),
        header,
        "-" * len(header),
    ]

    for r in results:
        request = r.case.request
        if len(request) > request_w:
            request = request[: request_w - 1] + "…"
        lines.append(
            f"{request:<{request_w}}  "
            f"{_mark(r.genre_ok):<6} "
            f"{_mark(r.mood_ok):<6} "
            f"{('n/a' if r.case.k is None else _mark(r.k_ok)):<6} "
            f"{_mark(r.consistent):<8} "
            f"{_mark(r.passed):<7} "
            f"{r.observed():<{observed_w}}"
        )
        if r.error:
            lines.append(f"{'':<{request_w}}  -> error: {r.error}")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    checks = {
        "genre correct": sum(1 for r in results if r.genre_ok),
        "mood correct": sum(1 for r in results if r.mood_ok),
        "count correct": sum(1 for r in results if r.k_ok),
        "consistent across runs": sum(1 for r in results if r.consistent),
    }

    lines.append("-" * len(header))
    for name, count in checks.items():
        lines.append(f"  {name:<24} {count}/{total}")
    pct = (passed / total * 100) if total else 0.0
    lines.append("-" * len(header))
    lines.append(f"  TOTAL SCORE: {passed}/{total} cases fully passed ({pct:.0f}%)")
    lines.append("=" * len(header))

    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("\nFailing cases:")
        for r in failures:
            reasons = []
            if r.error:
                reasons.append(r.error)
            else:
                if not r.genre_ok:
                    reasons.append(
                        f"genre {r.first.genre!r} not in {sorted(r.case.genres)}"
                    )
                if not r.mood_ok:
                    reasons.append(
                        f"mood {r.first.mood!r} not in {sorted(r.case.moods)}"
                    )
                if not r.k_ok:
                    reasons.append(f"k={r.first.k}, expected {r.case.k}")
                if not r.consistent:
                    reasons.append("runs disagreed with each other")
            lines.append(f"  - {r.case.request!r}: {'; '.join(reasons)}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--provider", choices=["anthropic", "gemini"])
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="runs per case, 2 or more measures consistency (default: 2)",
    )
    args = parser.parse_args()

    if args.runs < 1:
        print("--runs must be at least 1", file=sys.stderr)
        return 2

    try:
        catalog = nl.Catalog.load("data/songs.csv")
        provider = build_provider(args.provider)
    except NLRecommenderError as exc:
        print(f"\nCannot run the harness: {exc}\n", file=sys.stderr)
        print(
            "The harness makes real model calls. For the offline logic tests, run:"
            "\n    pytest\n",
            file=sys.stderr,
        )
        return 1

    calls = len(EVAL_CASES) * args.runs
    print(f"Running {len(EVAL_CASES)} cases x {args.runs} runs = {calls} model calls...")

    results = evaluate_all(catalog, provider, runs=args.runs)
    print()
    print(render_report(results, args.runs, provider.describe()))

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
