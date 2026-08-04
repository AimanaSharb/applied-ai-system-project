"""
Agentic self-check: a second model pass that audits the retrieved songs.

The verifier picks songs by index, so it can only filter or reorder what
retrieval already found - it cannot add a song. It also never returns an empty
list; if it rejects everything the original ranking is kept and the user is
told the matches are weak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

try:
    from llm import Provider, build_provider
    import nl_recommender as nl
except ModuleNotFoundError:
    from src.llm import Provider, build_provider
    from src import nl_recommender as nl


VERIFY_SYSTEM = """You are auditing a music recommender's output for mismatches.

You are given a listener's original request, the structured search that was run
(genre, mood), and the numbered songs that were retrieved. Judge each song.

A song is a match if it plausibly satisfies the listener's request. Judge it on
the request's intent, not on exact field equality: a song can match on mood
alone even if its genre differs. Mark a song as NOT a match only when it would
genuinely disappoint the listener — for example an intense workout track for a
request about falling asleep.

Reply with a single JSON object and nothing else:
{
  "assessments": [
    {"index": 1, "matches": true, "reason": "one short clause"}
  ],
  "summary": "one sentence on the overall quality of the list"
}

Include exactly one assessment per retrieved song, using the numbers shown.
Do not mention any song that is not in the list."""


@dataclass
class Assessment:
    index: int
    title: str
    matches: bool
    reason: str


@dataclass
class Verdict:
    passed: bool
    assessments: List[Assessment] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    summary: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_model_output: str = ""

    @property
    def matched_count(self) -> int:
        return sum(1 for a in self.assessments if a.matches)


def _parse_verdict(
    raw: str, recs: Sequence["nl.Recommendation"]
) -> Tuple[List[Assessment], str, List[str]]:
    """Validate the verifier's JSON against the retrieved songs."""
    warnings: List[str] = []
    try:
        data = nl._extract_json(raw)
    except json.JSONDecodeError:
        warnings.append(
            "The verification step did not return valid JSON, so the original "
            "ranking was kept unchanged."
        )
        return [], "", warnings

    raw_assessments = data.get("assessments")
    if not isinstance(raw_assessments, list):
        warnings.append(
            "The verification step returned no assessments, so the original "
            "ranking was kept unchanged."
        )
        return [], str(data.get("summary") or ""), warnings

    assessments: List[Assessment] = []
    seen = set()
    for item in raw_assessments:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        # An index must point at a song we actually retrieved.
        if not 1 <= index <= len(recs) or index in seen:
            warnings.append(
                f"The verification step referred to song #{item.get('index')}, "
                "which was not in the retrieved list; that judgement was ignored."
            )
            continue
        seen.add(index)

        matches = item.get("matches", True)
        if not isinstance(matches, bool):
            matches = str(matches).strip().lower() not in {"false", "no", "0"}

        assessments.append(
            Assessment(
                index=index,
                title=recs[index - 1].song.title,
                matches=matches,
                reason=str(item.get("reason") or "").strip(),
            )
        )

    # A song the verifier skipped gets the benefit of the doubt.
    for i in range(1, len(recs) + 1):
        if i not in seen:
            assessments.append(
                Assessment(
                    index=i,
                    title=recs[i - 1].song.title,
                    matches=True,
                    reason="not assessed; kept by default",
                )
            )

    assessments.sort(key=lambda a: a.index)
    return assessments, str(data.get("summary") or ""), warnings


def verify_recommendations(
    request_text: str,
    parsed: "nl.ParsedRequest",
    recs: Sequence["nl.Recommendation"],
    catalog: "nl.Catalog",
    provider: Optional[Provider] = None,
) -> Tuple[Verdict, List["nl.Recommendation"]]:
    """Audit recs and return (verdict, filtered and reordered recs)."""
    recs = list(recs)
    if not recs:
        return Verdict(passed=True, summary="Nothing to verify."), recs

    provider = provider or build_provider()
    user = (
        f"Listener's original request: {request_text.strip()}\n"
        f"Structured search that was run: genre={parsed.genre}, mood={parsed.mood}\n\n"
        f"Retrieved songs:\n{nl.format_catalog_context(recs)}"
    )

    try:
        raw = provider.complete(
            system=VERIFY_SYSTEM, user=user, max_tokens=800, effort="low"
        )
    except nl.APIUnavailableError as exc:
        # A failed audit should not sink an otherwise good answer.
        return (
            Verdict(
                passed=True,
                summary="Verification skipped.",
                warnings=[f"Could not run the verification step ({exc})."],
            ),
            recs,
        )

    assessments, summary, warnings = _parse_verdict(raw, recs)

    if not assessments:
        return (
            Verdict(
                passed=True,
                summary=summary,
                warnings=warnings,
                raw_model_output=raw,
            ),
            recs,
        )

    kept = [recs[a.index - 1] for a in assessments if a.matches]
    dropped = [a for a in assessments if not a.matches]

    if not kept:
        # Never return an empty list. Be honest instead.
        warnings.append(
            "The verification step judged none of the retrieved songs to be a "
            "strong match — showing the closest available options anyway."
        )
        return (
            Verdict(
                passed=False,
                assessments=assessments,
                summary=summary,
                warnings=warnings,
                raw_model_output=raw,
            ),
            recs,
        )

    removed_titles = [a.title for a in dropped]
    if removed_titles:
        warnings.append(
            "The verification step removed "
            + ", ".join(f"'{t}'" for t in removed_titles)
            + " as a poor fit for your request."
        )

    return (
        Verdict(
            passed=not dropped,
            assessments=assessments,
            removed=removed_titles,
            summary=summary,
            warnings=warnings,
            raw_model_output=raw,
        ),
        kept,
    )
