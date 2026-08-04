"""Tests for the AI layer. All model calls are mocked, so this runs offline."""

import json

import pytest

from src import agent_check, nl_recommender as nl
from src.llm import (
    APIUnavailableError,
    FakeProvider,
    NLRecommenderError,
    build_provider,
)

GENERATED = "Here are some picks.\n- Some Song by Someone: it fits."


@pytest.fixture
def catalog():
    return nl.Catalog.load("data/songs.csv")


def parse_reply(genre="lofi", mood="chill", k=3, is_music=True, reasoning="ok"):
    return json.dumps(
        {
            "is_music_request": is_music,
            "genre": genre,
            "mood": mood,
            "k": k,
            "reasoning": reasoning,
        }
    )


def verify_reply(assessments, summary="fine"):
    return json.dumps({"assessments": assessments, "summary": summary})


def all_match(n):
    return verify_reply([{"index": i, "matches": True, "reason": "fits"} for i in range(1, n + 1)])


def test_catalog_vocabulary_comes_from_the_csv(catalog):
    assert "lofi" in catalog.genres
    assert "chill" in catalog.moods
    # Nothing invented, nothing missing.
    assert set(catalog.genres) == {s.genre for s in catalog.songs}
    assert set(catalog.moods) == {s.mood for s in catalog.songs}


def test_catalog_derives_target_energy_per_mood(catalog):
    # Chill songs really are lower energy than intense ones in this catalog.
    assert catalog.mood_energy["chill"] < catalog.mood_energy["intense"]


def test_missing_csv_raises_a_handled_error():
    with pytest.raises(NLRecommenderError):
        nl.Catalog.load("data/does_not_exist.csv")


def test_parse_extracts_structured_fields(catalog):
    provider = FakeProvider([parse_reply("jazz", "relaxed", 2)])
    parsed = nl.parse_request("some jazz to unwind", catalog, provider=provider)

    assert (parsed.genre, parsed.mood, parsed.k) == ("jazz", "relaxed", 2)
    assert parsed.warnings == []
    # The prompt must advertise the real catalog vocabulary to the model.
    assert "jazz" in provider.calls[0]["system"]


def test_parse_tolerates_a_markdown_code_fence(catalog):
    fenced = f"```json\n{parse_reply('pop', 'happy', 4)}\n```"
    parsed = nl.parse_request("happy pop", catalog, provider=FakeProvider([fenced]))
    assert (parsed.genre, parsed.mood, parsed.k) == ("pop", "happy", 4)


def test_parse_tolerates_prose_around_the_json(catalog):
    noisy = f"Sure! Here you go:\n{parse_reply('rock', 'intense', 1)}\nHope that helps."
    parsed = nl.parse_request("hard rock", catalog, provider=FakeProvider([noisy]))
    assert (parsed.genre, parsed.mood) == ("rock", "intense")


def test_empty_input_is_rejected_without_calling_the_model(catalog):
    provider = FakeProvider([])  # any call would raise AssertionError
    for text in ["", "   ", "\n\t"]:
        with pytest.raises(NLRecommenderError):
            nl.parse_request(text, catalog, provider=provider)
    assert provider.calls == []


def test_unparseable_json_falls_back_instead_of_crashing(catalog):
    parsed = nl.parse_request(
        "anything", catalog, provider=FakeProvider(["I'm not JSON at all."])
    )
    assert parsed.genre in catalog.genres
    assert parsed.mood in catalog.moods
    assert any("not valid JSON" in w for w in parsed.warnings)


def test_near_miss_genre_snaps_to_the_closest_valid_value(catalog):
    parsed = nl.validate_parsed({"genre": "jaz", "mood": "chill", "k": 3}, catalog)
    assert parsed.genre == "jazz"
    assert any("closest match" in w for w in parsed.warnings)


def test_unknown_genre_falls_back_and_warns(catalog):
    parsed = nl.validate_parsed({"genre": "reggaeton", "mood": "chill", "k": 3}, catalog)
    assert parsed.genre in catalog.genres
    assert any("no close match" in w for w in parsed.warnings)


def test_unknown_mood_falls_back_and_warns(catalog):
    parsed = nl.validate_parsed({"genre": "pop", "mood": "euphoric", "k": 3}, catalog)
    assert parsed.mood in catalog.moods
    assert parsed.warnings


def test_null_fields_are_filled_with_defaults(catalog):
    parsed = nl.validate_parsed({"genre": None, "mood": None}, catalog)
    assert parsed.genre in catalog.genres
    assert parsed.mood in catalog.moods
    assert parsed.k == nl.DEFAULT_K


@pytest.mark.parametrize(
    "given,expected",
    [(99, nl.MAX_K), (0, 1), (-4, 1), ("not a number", nl.DEFAULT_K), (None, nl.DEFAULT_K), (3, 3)],
)
def test_k_is_clamped_to_the_catalog(catalog, given, expected):
    parsed = nl.validate_parsed({"genre": "pop", "mood": "happy", "k": given}, catalog)
    assert parsed.k == expected


def test_string_boolean_for_is_music_request_is_coerced(catalog):
    assert nl.validate_parsed({"is_music_request": "false"}, catalog).is_music_request is False
    assert nl.validate_parsed({"is_music_request": "true"}, catalog).is_music_request is True


def test_retrieval_returns_only_real_catalog_songs(catalog):
    parsed = nl.ParsedRequest(genre="lofi", mood="chill", k=3)
    recs = nl.retrieve(parsed, catalog)

    titles = {s.title for s in catalog.songs}
    assert len(recs) == 3
    assert all(r.song.title in titles for r in recs)


def test_retrieval_ranks_the_exact_match_first(catalog):
    parsed = nl.ParsedRequest(genre="jazz", mood="relaxed", k=3)
    recs = nl.retrieve(parsed, catalog)
    assert recs[0].song.genre == "jazz"
    assert recs[0].song.mood == "relaxed"


def test_retrieval_scores_are_descending(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=5), catalog)
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_is_deterministic(catalog):
    parsed = nl.ParsedRequest(genre="lofi", mood="chill", k=4)
    first = [r.song.id for r in nl.retrieve(parsed, catalog)]
    second = [r.song.id for r in nl.retrieve(parsed, catalog)]
    assert first == second


def test_context_block_lists_every_retrieved_song(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="lofi", mood="chill", k=3), catalog)
    context = nl.format_catalog_context(recs)
    for rec in recs:
        assert rec.song.title in context
        assert rec.song.artist in context


def test_generation_is_given_only_retrieved_songs(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="jazz", mood="relaxed", k=2), catalog)
    provider = FakeProvider([GENERATED])
    nl.generate_answer("some jazz", recs, provider=provider)

    prompt = provider.calls[0]["user"]
    retrieved = {r.song.title for r in recs}
    for song in catalog.songs:
        if song.title not in retrieved:
            assert song.title not in prompt


def test_generation_with_no_results_does_not_call_the_model():
    provider = FakeProvider([])
    answer = nl.generate_answer("anything", [], provider=provider)
    assert "could not find" in answer.lower()
    assert provider.calls == []


def test_verifier_filters_the_song_it_rejects(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="lofi", mood="chill", k=3), catalog)
    reply = verify_reply(
        [
            {"index": 1, "matches": True, "reason": "spot on"},
            {"index": 2, "matches": True, "reason": "good"},
            {"index": 3, "matches": False, "reason": "too energetic"},
        ]
    )
    verdict, kept = agent_check.verify_recommendations(
        "chill lofi", nl.ParsedRequest("lofi", "chill", 3), recs, catalog,
        provider=FakeProvider([reply]),
    )

    assert len(kept) == 2
    assert recs[2].song.title in verdict.removed
    assert recs[2].song not in [r.song for r in kept]
    assert verdict.passed is False


def test_verifier_passing_everything_keeps_the_original_order(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=3), catalog)
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 3), recs, catalog,
        provider=FakeProvider([all_match(3)]),
    )
    assert verdict.passed is True
    assert [r.song.id for r in kept] == [r.song.id for r in recs]


def test_verifier_rejecting_everything_never_returns_an_empty_list(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=2), catalog)
    reply = verify_reply([{"index": i, "matches": False, "reason": "no"} for i in (1, 2)])
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 2), recs, catalog,
        provider=FakeProvider([reply]),
    )
    assert len(kept) == len(recs)  # falls back rather than returning nothing
    assert verdict.passed is False
    assert any("closest available" in w for w in verdict.warnings)


def test_verifier_cannot_reference_a_song_outside_the_retrieved_list(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=2), catalog)
    reply = verify_reply([{"index": 99, "matches": False, "reason": "phantom"}])
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 2), recs, catalog,
        provider=FakeProvider([reply]),
    )
    assert len(kept) == len(recs)
    assert any("not in the retrieved list" in w for w in verdict.warnings)


def test_verifier_skipping_a_song_keeps_it_by_default(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=3), catalog)
    reply = verify_reply([{"index": 1, "matches": True, "reason": "fits"}])
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 3), recs, catalog,
        provider=FakeProvider([reply]),
    )
    assert len(kept) == 3
    assert len(verdict.assessments) == 3


def test_verifier_bad_json_keeps_the_original_ranking(catalog):
    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=3), catalog)
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 3), recs, catalog,
        provider=FakeProvider(["not json"]),
    )
    assert [r.song.id for r in kept] == [r.song.id for r in recs]
    assert any("valid JSON" in w for w in verdict.warnings)


def test_verifier_api_failure_does_not_sink_the_answer(catalog):
    class Broken(FakeProvider):
        def complete(self, **kwargs):
            raise APIUnavailableError("simulated outage")

    recs = nl.retrieve(nl.ParsedRequest(genre="pop", mood="happy", k=2), catalog)
    verdict, kept = agent_check.verify_recommendations(
        "happy pop", nl.ParsedRequest("pop", "happy", 2), recs, catalog,
        provider=Broken([]),
    )
    assert len(kept) == len(recs)
    assert verdict.passed is True
    assert any("simulated outage" in w for w in verdict.warnings)


def test_full_pipeline_makes_three_calls_in_order(catalog):
    provider = FakeProvider([parse_reply("lofi", "chill", 3), all_match(3), GENERATED])
    result = nl.recommend_from_text(
        "something chill for studying", catalog=catalog, provider=provider
    )

    assert len(provider.calls) == 3  # parse, verify, generate
    assert "structured search fields" in provider.calls[0]["system"]
    assert "auditing" in provider.calls[1]["system"]
    assert "Ground every factual claim" in provider.calls[2]["system"]
    assert result.answer == GENERATED
    assert len(result.recommendations) == 3


def test_pipeline_can_skip_verification(catalog):
    provider = FakeProvider([parse_reply("lofi", "chill", 2), GENERATED])
    result = nl.recommend_from_text(
        "chill music", catalog=catalog, provider=provider, verify=False
    )
    assert len(provider.calls) == 2
    assert result.verdict is None


def test_non_music_request_stops_before_retrieval(catalog):
    provider = FakeProvider([parse_reply(None, None, 5, is_music=False)])
    with pytest.raises(NLRecommenderError, match="does not look like a music request"):
        nl.recommend_from_text("asdkjfh qwerty", catalog=catalog, provider=provider)
    assert len(provider.calls) == 1  # never reached generation


def test_api_failure_surfaces_as_a_handled_error(catalog):
    class Broken(FakeProvider):
        def complete(self, **kwargs):
            raise APIUnavailableError("the API is down")

    with pytest.raises(NLRecommenderError, match="the API is down"):
        nl.recommend_from_text("chill music", catalog=catalog, provider=Broken([]))


def test_guardrail_warnings_reach_the_result(catalog):
    provider = FakeProvider([parse_reply("reggaeton", "chill", 2), all_match(2), GENERATED])
    result = nl.recommend_from_text("some reggaeton", catalog=catalog, provider=provider)
    assert result.warnings
    assert result.parsed.genre in catalog.genres


def test_unknown_provider_is_rejected():
    with pytest.raises(NLRecommenderError, match="Unknown LLM_PROVIDER"):
        build_provider("not-a-provider")


def test_anthropic_provider_without_a_key_explains_the_options(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(APIUnavailableError) as exc:
        build_provider("anthropic")
    assert "ANTHROPIC_API_KEY" in str(exc.value)
    assert "--classic" in str(exc.value)


def test_gemini_provider_without_a_key_explains_the_options(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(APIUnavailableError) as exc:
        build_provider("gemini")
    assert "GEMINI_API_KEY" in str(exc.value)


def test_provider_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(APIUnavailableError, match="ANTHROPIC_API_KEY"):
        build_provider()
