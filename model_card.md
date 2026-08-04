# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeMatch 2.0** — a rule-based music recommender with a language-model front
end. It is a **hybrid**, and the split matters for everything below:

| Component | Type | Decides |
|---|---|---|
| Retrieval and ranking | Deterministic Python (`src/recommender.py`) | **Which songs are returned, in what order** |
| Request understanding | LLM (`claude-sonnet-4-6` by default) | How free text maps to a `{genre, mood, k}` query |
| Self-check | LLM | Which retrieved songs to *filter out* (by index only) |
| Response writing | LLM | The prose the user reads |

The model never selects songs from the catalog. It selects a *query*, may *veto*
results, and *describes* them. VibeMatch 1.0 (rule-based only) is preserved and
still runnable via `--classic`.

---

## 2. Intended Use

Recommend songs from a fixed 10-song catalog in response to a free-text request
("something chill for studying"), returning a ranked list with a match score, a
template reason per song, and a short natural-language summary.

- **Audience:** a **classroom / learning project**, not a production system. It
  exists to show how RAG, guardrails, and an agentic self-check compose around an
  existing deterministic system.
- **Assumption:** the listener's intent can be usefully compressed into one genre
  and one mood drawn from the catalog's vocabulary. This is a real limitation, not
  a neutral simplification — see §6.
- **Out of scope:** music discovery beyond these 10 songs, personalisation across
  sessions, audio analysis, and any use where a wrong recommendation carries
  consequences.

---

## 3. LLM Usage

### Which model, and where

Default **`claude-sonnet-4-6`** via the Anthropic Messages API. An alternative
**`gemini-2.5-flash`** backend is selectable with `LLM_PROVIDER=gemini`. Both sit
behind one interface in `src/llm.py` (system prompt + user message → text), so no
pipeline code is provider-specific.

Three calls per request:

| # | Step | Prompt contains | Returns | Effort |
|---|---|---|---|---|
| 1 | **Parse** | The catalog's real genre and mood lists, read from `songs.csv` | JSON `{is_music_request, genre, mood, k, reasoning}` | low |
| 2 | **Verify** | The request + the retrieved songs only | JSON per-song match verdicts, keyed by index | low |
| 3 | **Generate** | The request + the retrieved songs only | Prose + one line per song | medium |

Sonnet 4.6 does not support the API's JSON-schema-constrained output mode, so
calls 1 and 2 ask for JSON in the prompt and the response is parsed and validated
in Python. That validation layer is not a workaround to be removed later — it is
where the guardrails live, and it would be needed regardless.

### How hallucination is constrained

Three mechanisms, in decreasing strength:

1. **The model cannot choose songs.** Retrieval is deterministic. This is
   structural — no prompt can bypass it.
2. **The self-check works by index.** The verifier picks from `1..n`; Python maps
   integers back to real objects. It cannot name a song that was not retrieved.
   Out-of-range indices are discarded with a warning.
3. **The generation prompt is grounded.** The model is given only retrieved rows
   and instructed not to mention anything else, invent release years, or oversell
   a weak fit.

**Mechanism 3 is the weak one and is honestly the system's main residual risk.**
Grounding of the final prose is enforced by instruction, not by code. Nothing
programmatically diffs the generated text against the catalog, so a determined
hallucination in the prose could reach the user even though the *ranked list*
beside it is guaranteed correct.

### Vocabulary guardrail

Model-supplied `genre`/`mood` values are checked against values that actually
occur in `songs.csv`. An exact match passes; a near miss (`difflib` ratio ≥ 0.6,
e.g. `jaz` → `jazz`) is snapped; anything else falls back to a valid default.
Every substitution is surfaced to the user. `k` is clamped to 1–10.

### Non-determinism, and how it is measured

The same sentence can parse differently on different runs. `tests/eval_harness.py`
therefore runs each of 8 cases **twice** and scores consistency alongside
correctness: a case that is acceptable once but different the next time is marked
failed. A single-run evaluation cannot see this, and it is the property most
likely to make the system feel unreliable in use.

### Cost, latency, privacy

Three calls per request where the base project made zero. Request text is sent to
a third-party API — no user data is stored by this project beyond
`logs/agent_trace.md`, which is written locally and contains the request text.

---

## 4. Data

`data/songs.csv`, **10 songs, unchanged from the base project.**

- **Columns:** id, title, artist, genre, mood, energy, tempo_bpm, valence,
  danceability, acousticness.
- **Genres (7):** pop, indie pop, lofi, rock, ambient, jazz, synthwave.
- **Moods (6):** happy, chill, intense, relaxed, moody, focused.
- **Missing:** language, lyrics, release year, artist popularity, cultural
  context. Whole families of music — hip-hop, classical, country, and most
  non-Western genres — are absent.

The extension adds no data, but it does derive one thing from it: `target_energy`
is the **mean energy of songs sharing the requested mood** rather than a
hardcoded number, so a "chill" request aims at the energy chill songs actually
have (0.35) rather than a guess.

**The catalog is also the guardrail's vocabulary.** This has a consequence worth
stating plainly: the system's *understanding* is bounded by the dataset's
narrowness, not just its recommendations. A hip-hop request cannot fail loudly —
it gets silently mapped to whichever of these 7 genres is nearest.

---

## 5. Strengths

- **Free-text input is a genuine capability change.** The base project required
  editing source to change the request.
- **Ranking stayed reproducible.** Because retrieval is deterministic, the same
  parsed query always yields the same ranked list with the same scores — verified
  by a test. Explanations remain traceable to the scoring rule.
- **Failure is handled rather than propagated.** Empty input, non-music input,
  malformed JSON, unknown vocabulary, out-of-range counts, and every API error
  class produce a clear message. No unhandled exception reaches the user.
- **It is auditable.** `logs/agent_trace.md` records each step's reasoning, so a
  surprising recommendation can be traced to the step that caused it.
- **Testable without the vendor.** All 59 tests run offline against a
  `FakeProvider`, so guardrail behaviour is verified deterministically rather than
  by hoping a live model misbehaves on cue.

---

## 6. Limitations and Bias

**Inherited from the base project**

- Scoring ignores tempo, valence, and danceability.
- Genre's 2.0 weight can override a better mood match.
- Underrepresented tastes get thin lists; absent genres can never be recommended.
- No popularity or novelty signal.

**Introduced by the LLM layer**

- **Non-determinism.** Identical input can produce a different parse, so the
  system is not reproducible end-to-end even though retrieval is.
- **The one-genre-one-mood bottleneck** discards most of what a request can
  express. Comparative or referential requests ("like early Radiohead but
  quieter") have nowhere to land.
- **Silent-substitution bias.** The guardrail always produces *a* valid answer.
  For a request the catalog cannot serve, "closest valid value" is arbitrary — the
  first alphabetically — so users outside the catalog's taste range get a
  confident, disclosed, but essentially meaningless answer.
- **Prose grounding is unverified in code** (§3).
- **The self-check shares the first pass's blind spots.** The same model family
  auditing its own output will not catch errors that stem from a misreading both
  passes share. It catches obvious mismatches, not subtle ones, and its
  independence is limited by construction.
- **Anglophone-intent bias.** Prompts, activity inference ("studying" → calm), and
  the vocabulary all assume English and Western listening conventions.

**Compounding effect:** a narrow catalog plus a model that always returns
*something* means the system's confidence is roughly constant regardless of
whether it can actually serve the request. Users whose taste the dataset does not
represent get answers that look exactly as authoritative as good ones. The
warnings are the only signal, and they are easy to skim past.

---

## 7. Evaluation

| What | How | Result |
|---|---|---|
| Original scoring logic | `tests/test_recommender.py` (unchanged) | 2/2 passing |
| Parsing, guardrails, retrieval, self-check, trace, harness scoring | `tests/test_nl_recommender.py`, `tests/test_harness_and_trace.py` — all model calls mocked | 59/59 passing offline |
| Parser correctness **and** run-to-run consistency | `tests/eval_harness.py` — 8 NL queries × 2 runs, set-valued expectations | Requires an API key; not yet run against a live model |
| Guardrail branches | One test per branch: bad JSON, near-miss vocabulary, unknown vocabulary, `k` out of range, non-music input, empty input, out-of-range verifier index, verifier rejecting everything, API failure | All verified |

**Set-valued expectations, deliberately.** Each harness case lists acceptable
genres/moods rather than one right answer, because more than one mapping is often
defensible — "chill for studying" is reasonably lofi *or* ambient. Scoring a
single value would measure taste rather than correctness.

**Honest gap:** the live-model numbers in the harness table are not yet filled
in — the machine this was built on had no API key, so the README labels
model-written prose as illustrative. Every deterministic figure quoted (scores,
rankings, guardrail messages) is real captured output.

**What surprised me:** how much of the work was error handling. The RAG happy path
is roughly 60 lines; the guardrails, fallbacks, and their tests are several times
that. The other surprise was the consistency dimension — the parser looked fine
until I ran each query twice and had to decide whether "right once, different
next time" counts as correct. It does not.

---

## 8. Future Work

- **Verify grounding in code.** Diff generated prose against retrieved titles and
  artists, and flag or regenerate on a mismatch. This closes the one hallucination
  gap not already structurally blocked.
- **Multi-genre / multi-mood queries** with weighted rather than exact matching,
  so "pop" can partially match "indie pop" instead of scoring zero.
- **Use the unused columns** — tempo, valence, danceability — in scoring.
- **Cache parses** for repeated requests to cut cost and mask non-determinism.
- **A much larger catalog**, which would relieve the vocabulary bottleneck that
  currently causes the silent-substitution problem.
- **An independent verifier** — a different model, or a rule-based check — so the
  self-check does not inherit the first pass's blind spots.

---

## 9. Personal Reflection

The base project taught that a recommender is a scoring rule and the intelligence
lives in the weights. Adding a language model to it taught something narrower and
more useful: **the model is best used at the edges, not the centre.**

The tempting design was to hand the whole job to the model — ask it for songs
directly. That is much less code, and it fails in the ways that matter: it invents
songs that are not in the catalog, cannot explain a ranking, and produces a
different answer every time. Keeping retrieval deterministic and confining the
model to translation is what makes the output defensible.

The guardrails were where the real learning sat. Writing the fallback for "the
model returned a genre that does not exist" made it concrete that a prompt listing
valid values does not constrain the output to them — the prompt is a request, not
a type system. And the fix is not just to avoid crashing: it is to substitute *and
tell the user you did*. The same instinct produced the rule that the self-check
may never return an empty list. Somewhere in the middle of that it became clear
that most of this extension is not the AI feature at all, it is the handling of
the AI feature being wrong.

On bias, the sharpest thing I noticed is that this system cannot fail loudly. A
request it has no business answering still gets a confident, well-formatted,
plausibly-explained list — because the guardrail's job is to always produce
*something* valid. In the base project, a badly-served user at least got obviously
weak scores. Now they get fluent prose over the same weak scores. Fluency made the
output better to read and, in that specific way, worse to trust.
