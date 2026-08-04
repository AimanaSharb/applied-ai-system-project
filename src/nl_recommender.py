"""
Natural-language recommendations.

Free text -> LLM parses it to {genre, mood, k} -> guardrails validate those
values against songs.csv -> the existing Recommender ranks the real catalog ->
the LLM writes an answer using only the retrieved songs.

The model picks the query; the rule-based recommender picks the songs.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

try:
    from recommender import Recommender, Song, UserProfile, load_songs
    from llm import APIUnavailableError, NLRecommenderError, Provider, build_provider
except ModuleNotFoundError:
    from src.recommender import Recommender, Song, UserProfile, load_songs
    from src.llm import (
        APIUnavailableError,
        NLRecommenderError,
        Provider,
        build_provider,
    )

__all__ = [
    "APIUnavailableError",
    "Catalog",
    "NLRecommenderError",
    "NLResult",
    "ParsedRequest",
    "Recommendation",
    "build_provider",
    "format_catalog_context",
    "generate_answer",
    "parse_request",
    "recommend_from_text",
    "retrieve",
    "validate_parsed",
]

DEFAULT_K = 5
MAX_K = 10
DEFAULT_CSV = "data/songs.csv"


@dataclass
class Catalog:
    """The songs plus the genre/mood values that are actually valid in them."""

    songs: List[Song]
    genres: List[str]
    moods: List[str]
    mood_energy: Dict[str, float]

    @classmethod
    def load(cls, csv_path: str = DEFAULT_CSV) -> "Catalog":
        try:
            rows = load_songs(csv_path)
        except FileNotFoundError as exc:
            raise NLRecommenderError(f"Song catalog not found at {csv_path}.") from exc
        if not rows:
            raise NLRecommenderError(f"No songs found in {csv_path}.")

        songs = [
            Song(
                id=row["id"],
                title=row["title"],
                artist=row["artist"],
                genre=row["genre"],
                mood=row["mood"],
                energy=row["energy"],
                tempo_bpm=row["tempo_bpm"],
                valence=row["valence"],
                danceability=row["danceability"],
                acousticness=row["acousticness"],
            )
            for row in rows
        ]

        genres = sorted({s.genre for s in songs})
        moods = sorted({s.mood for s in songs})

        # Average the energy of each mood instead of hardcoding a target.
        mood_energy = {
            mood: sum(s.energy for s in songs if s.mood == mood)
            / sum(1 for s in songs if s.mood == mood)
            for mood in moods
        }
        return cls(songs=songs, genres=genres, moods=moods, mood_energy=mood_energy)


@dataclass
class ParsedRequest:
    """A request after parsing and validation."""

    genre: str
    mood: str
    k: int
    is_music_request: bool = True
    reasoning: str = ""
    warnings: List[str] = field(default_factory=list)
    raw_model_output: str = ""


PARSE_SYSTEM = """You translate a listener's free-text request into structured search \
fields for a small music catalog.

Reply with a single JSON object and nothing else. Use exactly these keys:
  "is_music_request": true if the text expresses any musical or listening intent \
(a mood, an activity to soundtrack, a genre, an occasion). false only if the text \
is unrelated to music or is meaningless.
  "genre": one value from the allowed genres, or null if the request implies no genre.
  "mood": one value from the allowed moods, or null if the request implies no mood.
  "k": how many songs to return, as an integer from 1 to {max_k}. Use {default_k} \
unless the listener asks for a specific number.
  "reasoning": one short sentence explaining the mapping.

Allowed genres: {genres}
Allowed moods: {moods}

Map the listener's intent, not their literal words: studying or focus implies a \
calm, low-energy mood; a workout implies an intense one. Only use values from the \
allowed lists."""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        braced = re.search(r"\{.*\}", text, re.DOTALL)
        if braced:
            text = braced.group(0)

    data = json.loads(text)
    if not isinstance(data, dict):
        raise json.JSONDecodeError("top-level JSON value is not an object", text, 0)
    return data


def _snap_to_vocabulary(
    value: Optional[str],
    allowed: Sequence[str],
    field_name: str,
    fallback: str,
    warnings: List[str],
) -> str:
    """Force a model value to one that exists in the catalog."""
    if value is None or not str(value).strip():
        warnings.append(
            f"No {field_name} was implied by your request, so I used '{fallback}'."
        )
        return fallback

    candidate = str(value).strip().lower()
    lookup = {a.lower(): a for a in allowed}
    if candidate in lookup:
        return lookup[candidate]

    close = difflib.get_close_matches(candidate, list(lookup), n=1, cutoff=0.6)
    if close:
        chosen = lookup[close[0]]
        warnings.append(
            f"'{value}' is not a {field_name} in the catalog — using the closest "
            f"match, '{chosen}'."
        )
        return chosen

    warnings.append(
        f"'{value}' is not a {field_name} in the catalog and has no close match — "
        f"falling back to '{fallback}'."
    )
    return fallback


def _coerce_k(value: object, warnings: List[str]) -> int:
    try:
        k = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        warnings.append(f"Could not read a song count from '{value}'; using {DEFAULT_K}.")
        return DEFAULT_K

    if k < 1:
        warnings.append(f"Asked for {k} songs; raising it to 1.")
        return 1
    if k > MAX_K:
        warnings.append(f"Asked for {k} songs; the catalog only supports {MAX_K}.")
        return MAX_K
    return k


def validate_parsed(data: dict, catalog: Catalog) -> ParsedRequest:
    """Validate parsed JSON. Never raises on a bad value."""
    warnings: List[str] = []

    is_music = data.get("is_music_request", True)
    if not isinstance(is_music, bool):
        is_music = str(is_music).strip().lower() not in {"false", "no", "0", "none", ""}

    genre = _snap_to_vocabulary(
        data.get("genre"), catalog.genres, "genre", catalog.genres[0], warnings
    )
    mood = _snap_to_vocabulary(
        data.get("mood"), catalog.moods, "mood", catalog.moods[0], warnings
    )
    k = _coerce_k(data.get("k", DEFAULT_K), warnings)

    return ParsedRequest(
        genre=genre,
        mood=mood,
        k=k,
        is_music_request=is_music,
        reasoning=str(data.get("reasoning") or ""),
        warnings=warnings,
    )


def parse_request(
    text: str, catalog: Catalog, provider: Optional[Provider] = None
) -> ParsedRequest:
    """Turn free text into a validated ParsedRequest."""
    if not text or not text.strip():
        raise NLRecommenderError(
            "I did not catch a request — tell me a mood, an activity, or a genre."
        )

    provider = provider or build_provider()
    system = PARSE_SYSTEM.format(
        genres=", ".join(catalog.genres),
        moods=", ".join(catalog.moods),
        max_k=MAX_K,
        default_k=DEFAULT_K,
    )
    raw = provider.complete(
        system=system, user=text.strip(), max_tokens=400, effort="low"
    )

    try:
        data = _extract_json(raw)
    except json.JSONDecodeError:
        # Unusable JSON must not crash the app.
        return ParsedRequest(
            genre=catalog.genres[0],
            mood=catalog.moods[0],
            k=DEFAULT_K,
            warnings=[
                "The model's reply was not valid JSON, so I fell back to a default "
                f"search ({catalog.genres[0]} / {catalog.moods[0]})."
            ],
            raw_model_output=raw,
        )

    parsed = validate_parsed(data, catalog)
    parsed.raw_model_output = raw
    return parsed


@dataclass
class Recommendation:
    song: Song
    score: float
    explanation: str


def retrieve(parsed: ParsedRequest, catalog: Catalog) -> List[Recommendation]:
    """Rank the catalog with the existing rule-based Recommender."""
    target_energy = catalog.mood_energy.get(parsed.mood, 0.5)
    profile = UserProfile(
        favorite_genre=parsed.genre,
        favorite_mood=parsed.mood,
        target_energy=target_energy,
        likes_acoustic=target_energy < 0.5,
    )
    recommender = Recommender(catalog.songs)
    return [
        Recommendation(
            song=song,
            score=recommender._score(profile, song),
            explanation=recommender.explain_recommendation(profile, song),
        )
        for song in recommender.recommend(profile, k=parsed.k)
    ]


def format_catalog_context(recs: Sequence[Recommendation]) -> str:
    """Render retrieved songs as the only facts the model may use."""
    return "\n".join(
        f'{i}. "{r.song.title}" by {r.song.artist} — genre: {r.song.genre}, '
        f"mood: {r.song.mood}, energy: {r.song.energy}, "
        f"tempo: {r.song.tempo_bpm} bpm, acousticness: {r.song.acousticness} "
        f"(match score {r.score:.2f})"
        for i, r in enumerate(recs, start=1)
    )


GENERATE_SYSTEM = """You are a music recommender writing a short reply to a listener.

You will be given the listener's request and a numbered list of retrieved songs.
Ground every factual claim ONLY in that list:
- Never mention a song, artist, genre, or mood that is not in the list.
- Never invent release years, chart positions, lyrics, or popularity.
- If a song is a weak fit for the request, say so plainly rather than overselling it.

Write 2-4 sentences of friendly prose, then one line per song in the form
"- Title by Artist: reason". Keep it concise and do not use headings."""


def generate_answer(
    request_text: str,
    recs: Sequence[Recommendation],
    provider: Optional[Provider] = None,
) -> str:
    """Write the reply, grounded in the retrieved songs."""
    if not recs:
        return "I could not find any songs in the catalog for that request."

    provider = provider or build_provider()
    user = (
        f"Listener's request: {request_text.strip()}\n\n"
        "Retrieved songs (the only songs you may mention):\n"
        f"{format_catalog_context(recs)}"
    )
    return provider.complete(
        system=GENERATE_SYSTEM, user=user, max_tokens=700, effort="medium"
    )


@dataclass
class NLResult:
    """Everything one request produced."""

    request: str
    parsed: ParsedRequest
    recommendations: List[Recommendation]
    answer: str
    warnings: List[str] = field(default_factory=list)
    verdict: Optional[object] = None


def recommend_from_text(
    text: str,
    catalog: Optional[Catalog] = None,
    provider: Optional[Provider] = None,
    csv_path: str = DEFAULT_CSV,
    verify: bool = True,
    trace=None,
) -> NLResult:
    """Run parse -> guardrail -> retrieve -> verify -> generate."""
    # Imported here to avoid a circular import.
    try:
        from agent_check import verify_recommendations
    except ModuleNotFoundError:
        from src.agent_check import verify_recommendations

    catalog = catalog or Catalog.load(csv_path)
    provider = provider or build_provider()
    if trace:
        trace.start(text)

    parsed = parse_request(text, catalog, provider=provider)
    warnings = list(parsed.warnings)
    if trace:
        trace.log_parse(text, parsed)

    if not parsed.is_music_request:
        if trace:
            trace.log_rejected(text, parsed)
        raise NLRecommenderError(
            "That does not look like a music request. Try describing a mood, an "
            'activity, or a genre — for example "something chill for studying".'
        )

    recs = retrieve(parsed, catalog)
    if trace:
        trace.log_retrieval(parsed, recs)

    verdict = None
    if verify:
        verdict, recs = verify_recommendations(
            text, parsed, recs, catalog, provider=provider
        )
        warnings.extend(verdict.warnings)
        if trace:
            trace.log_verification(verdict, recs)

    answer = generate_answer(text, recs, provider=provider)
    if trace:
        trace.log_answer(answer)

    return NLResult(
        request=text.strip(),
        parsed=parsed,
        recommendations=recs,
        answer=answer,
        warnings=warnings,
        verdict=verdict,
    )
