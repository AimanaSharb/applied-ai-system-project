# AI Interactions Log

> **Stretch features only.** Sections below cover the stretch features actually attempted.

---

## Agentic Workflow (SF8)

**What task did you give the agent?**

Extend the rule-based Music Recommender Simulation with a natural-language front
end: a RAG pipeline, an agentic self-check, reliability guardrails, an evaluation
harness, an architecture diagram, and updated documentation. The agent worked
incrementally, committing after each section.

**Prompts used (key ones):**

- The initial six-part spec: create `src/nl_recommender.py` that takes free text,
  calls the API to parse it into `{genre, mood, k}`, retrieves from
  `data/songs.csv` using the existing `Recommender`, and generates a response
  grounded only in the retrieved songs; integrate as an interactive mode with
  `--classic` as a fallback flag.
- *"After generating recommendations, add a second verification step where the
  model reviews whether the recommended songs actually match the parsed request;
  if any don't match, it re-ranks or filters them. Log each step's reasoning to a
  file `logs/agent_trace.md`."*
- *"Input validation: handle empty input, non-music requests, and API failures
  gracefully with clear messages (never crash). Output guardrail: validate that
  the LLM's JSON parses correctly and that genre/mood values exist in songs.csv;
  if not, fall back to closest valid values and tell the user."*
- *"Work step by step, run the tests after each major change, and commit with
  clear messages after each numbered section. Ask me before making design
  decisions that deviate from this plan."*
- A mid-task scope check — *"are you going according to the rubric, I kind of
  think you are going too much"* — which produced a written file-by-file
  accounting of what had been built versus what was specified.

**What did the agent generate or change?**

New: `src/nl_recommender.py` (RAG pipeline + guardrails), `src/agent_check.py`
(self-check), `src/trace.py` (reasoning log), `src/llm.py` (provider seam),
`tests/eval_harness.py`, `tests/test_nl_recommender.py`,
`tests/test_harness_and_trace.py`, `diagrams/architecture.mmd`, `conftest.py`.

Rewritten: `src/main.py`, `README.md`, `model_card.md`, `requirements.txt`.

**Unchanged on purpose:** `src/recommender.py`, `data/songs.csv`, and
`tests/test_recommender.py`. The original scoring rule still decides which songs
are returned; the AI layer only translates at the edges.

**What did you verify or fix manually?**

- **Two design decisions were escalated rather than assumed.** The agent stopped
  when it found no API key available and asked how to handle sample outputs, and
  stopped again when the suggested fix (switching to Gemini) would have deviated
  from the Anthropic spec in the original prompt. The resolution was the provider
  seam in `src/llm.py`: Anthropic stays the default and documented path, with
  Gemini selectable via `LLM_PROVIDER`.
- **SDK APIs were verified, not recalled.** `google-genai`'s client signature,
  config fields, and error classes were introspected from the installed package
  before any code was written against them.
- **Every guardrail branch was executed and its output inspected** — near-miss
  genre, unknown genre, malformed JSON, out-of-range `k`, gibberish input, empty
  input, verifier rejecting everything, verifier citing a non-existent index, and
  a simulated API outage.
- **Test-suite plumbing was a real bug found by testing.** `from src...` imports
  only worked under `python -m pytest` because that form adds the CWD to
  `sys.path`; bare `pytest` would have failed. Fixed with `conftest.py` and
  package `__init__.py` files, then verified both invocations.
- **Sample outputs are labelled honestly.** No API key was available, so
  model-written prose in the README is marked `[illustrative]` while every
  deterministic figure (scores, rankings, guardrail messages) is real captured
  output. The agent declined to present invented transcripts as real runs.

**Reasoning trace:** every pipeline step logs its own reasoning to
**[`logs/agent_trace.md`](logs/agent_trace.md)** — see the *Reasoning trace*
section of the [README](README.md#reasoning-trace) and step 4 of
[`diagrams/architecture.mmd`](diagrams/architecture.mmd). One Markdown section
per request records the parse step's extraction and reasoning, which guardrails
fired, what retrieval returned with scores, the verifier's per-song verdicts, and
the final answer. Written by `src/trace.py`; disable with `--no-trace`.

---

## Design Pattern (SF10)

**Which design pattern did you use?**

**Strategy**, in `src/llm.py`. `Provider` declares one operation — system prompt +
user message → text — and `AnthropicProvider`, `GeminiProvider`, and
`FakeProvider` implement it interchangeably. `build_provider()` is a small factory
that selects one from `LLM_PROVIDER` or an explicit `--provider` flag.

**How did AI help you brainstorm or implement it?**

The pattern was not in the original plan. It emerged from a constraint: the
Anthropic API has no free tier, but the assignment spec named Anthropic, so
neither "stay Anthropic-only" nor "switch to Gemini" was fully satisfactory. The
agent proposed the seam as the option that satisfied both, and flagged the
trade-off (roughly 200 extra lines) before writing it.

The second, larger benefit was not the motivation but became the more useful
outcome: `FakeProvider` is the same pattern, so the entire pipeline — including
every guardrail and the eval harness's own scoring — is testable offline with no
API key and no network. All 59 tests run in about a second.

**How does the pattern appear in your final code?**

```python
class Provider:
    def complete(self, system: str, user: str, max_tokens: int, effort: str) -> str:
        raise NotImplementedError

class AnthropicProvider(Provider): ...   # claude-sonnet-4-6
class GeminiProvider(Provider): ...      # gemini-2.5-flash
class FakeProvider(Provider): ...        # scripted replies, used by every test
```

Every call site takes an optional `provider` argument and defaults to
`build_provider()`, so production code names no vendor:

```python
parsed = nl.parse_request(text, catalog, provider=provider)
verdict, recs = agent_check.verify_recommendations(..., provider=provider)
answer = nl.generate_answer(text, recs, provider=provider)
```

The `effort` parameter shows the seam doing real work rather than just renaming
one API as another: it maps to Anthropic's `output_config.effort` directly, and to
a `thinking_budget` token count on Gemini.
