"""
Writes each pipeline step's reasoning to logs/agent_trace.md.

Best effort: if the log cannot be written, the run continues.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional, Sequence

DEFAULT_PATH = os.path.join("logs", "agent_trace.md")


class AgentTrace:
    """Appends one Markdown section per request."""

    def __init__(self, path: str = DEFAULT_PATH, provider_label: str = "unknown"):
        self.path = path
        self.provider_label = provider_label
        self.enabled = True
        self._step = 0

        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            if not os.path.exists(self.path):
                self._raw(
                    "# Agent Trace\n\n"
                    "Step-by-step reasoning log for the natural-language "
                    "recommender. Newest entries are appended at the bottom.\n"
                )
        except OSError:
            self.enabled = False

    def _raw(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            self.enabled = False

    def _write_step(self, name: str, lines: Sequence[str]) -> None:
        self._step += 1
        body = "\n".join(f"- {line}" for line in lines)
        self._raw(f"\n### Step {self._step} — {name}\n\n{body}\n")

    def start(self, request: str) -> None:
        self._step = 0
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        self._raw(
            f"\n---\n\n## Request: {request.strip()!r}\n\n"
            f"*{stamp} · provider: {self.provider_label}*\n"
        )

    def log_parse(self, request: str, parsed) -> None:
        lines = [
            f"**Input:** {request.strip()!r}",
            f"**Extracted:** genre=`{parsed.genre}`, mood=`{parsed.mood}`, k=`{parsed.k}`",
            f"**Is a music request:** {parsed.is_music_request}",
        ]
        if parsed.reasoning:
            lines.append(f"**Model's reasoning:** {parsed.reasoning}")
        for warning in parsed.warnings:
            lines.append(f"**Guardrail:** {warning}")
        if not parsed.warnings:
            lines.append(
                "**Guardrail:** all values valid in songs.csv; no correction needed."
            )
        self._write_step("Parse + guardrail validation", lines)

    def log_rejected(self, request: str, parsed) -> None:
        self._write_step(
            "Rejected",
            [
                "The parse step flagged this as not a music request.",
                "Pipeline stopped before retrieval; no answer was generated.",
            ],
        )

    def log_retrieval(self, parsed, recs) -> None:
        lines = [
            f"**Query:** genre=`{parsed.genre}`, mood=`{parsed.mood}`, top-{parsed.k}",
            f"**Retrieved {len(recs)} song(s) from data/songs.csv** "
            "(deterministic — scored by the rule-based Recommender):",
        ]
        lines += [
            f"  {i}. {r.song.title} — {r.song.artist} "
            f"(`{r.song.genre}`/`{r.song.mood}`, score {r.score:.2f})"
            for i, r in enumerate(recs, start=1)
        ]
        self._write_step("Retrieval", lines)

    def log_verification(self, verdict, recs) -> None:
        lines = [
            "**Verdict:** "
            + ("pass — all retrieved songs match" if verdict.passed else "issues found")
        ]
        if verdict.summary:
            lines.append(f"**Model's summary:** {verdict.summary}")
        for a in verdict.assessments:
            mark = "match" if a.matches else "MISMATCH"
            lines.append(
                f"  {a.index}. {a.title} — {mark}: {a.reason or 'no reason given'}"
            )
        if verdict.removed:
            lines.append(f"**Filtered out:** {', '.join(verdict.removed)}")
        for warning in verdict.warnings:
            lines.append(f"**Guardrail:** {warning}")
        lines.append(
            f"**Final list ({len(recs)}):** "
            + (", ".join(r.song.title for r in recs) or "none")
        )
        self._write_step("Verification (agentic self-check)", lines)

    def log_answer(self, answer: str) -> None:
        quoted = "\n".join(f"  > {line}" for line in answer.strip().splitlines())
        self._write_step("Grounded answer shown to user", [f"\n{quoted}"])

    def log_error(self, message: str) -> None:
        self._write_step("Error (handled)", [f"{message}"])


def build_trace(
    path: str = DEFAULT_PATH, provider_label: str = "unknown", enabled: bool = True
) -> Optional[AgentTrace]:
    """Return a trace logger, or None if tracing is off."""
    return AgentTrace(path=path, provider_label=provider_label) if enabled else None
