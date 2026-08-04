"""Tests for the trace logger and the eval harness's scoring, driven offline."""

import json

import pytest

from src import nl_recommender as nl, trace as trace_mod
from src.llm import FakeProvider
from tests import eval_harness as eh


@pytest.fixture
def catalog():
    return nl.Catalog.load("data/songs.csv")


def reply(genre, mood, k):
    return json.dumps({"is_music_request": True, "genre": genre, "mood": mood, "k": k})


def test_trace_records_every_pipeline_step(tmp_path, catalog):
    path = tmp_path / "nested" / "agent_trace.md"
    trace = trace_mod.build_trace(path=str(path), provider_label="fake (test)")
    provider = FakeProvider(
        [
            reply("lofi", "chill", 3),
            json.dumps(
                {
                    "assessments": [
                        {"index": 1, "matches": True, "reason": "fits"},
                        {"index": 2, "matches": True, "reason": "fits"},
                        {"index": 3, "matches": False, "reason": "too upbeat"},
                    ],
                    "summary": "two solid picks",
                }
            ),
            "Some grounded prose.",
        ]
    )
    nl.recommend_from_text(
        "chill study music", catalog=catalog, provider=provider, trace=trace
    )

    text = path.read_text(encoding="utf-8")
    assert "## Request: 'chill study music'" in text
    assert "Parse + guardrail validation" in text
    assert "Retrieval" in text
    assert "Verification (agentic self-check)" in text
    assert "Grounded answer shown to user" in text
    assert "two solid picks" in text
    assert "MISMATCH" in text
    assert "fake (test)" in text


def test_trace_records_a_rejected_non_music_request(tmp_path, catalog):
    path = tmp_path / "agent_trace.md"
    trace = trace_mod.build_trace(path=str(path))
    provider = FakeProvider([json.dumps({"is_music_request": False, "k": 5})])

    with pytest.raises(nl.NLRecommenderError):
        nl.recommend_from_text("qwerty asdf", catalog=catalog, provider=provider, trace=trace)

    text = path.read_text(encoding="utf-8")
    assert "Rejected" in text
    assert "not a music request" in text


def test_trace_appends_rather_than_overwriting(tmp_path, catalog):
    path = tmp_path / "agent_trace.md"
    for text in ["first request", "second request"]:
        trace = trace_mod.build_trace(path=str(path))
        provider = FakeProvider([reply("pop", "happy", 1), '{"assessments":[]}', "prose"])
        nl.recommend_from_text(text, catalog=catalog, provider=provider, trace=trace)

    contents = path.read_text(encoding="utf-8")
    assert "first request" in contents and "second request" in contents
    assert contents.count("## Request:") == 2


def test_trace_can_be_disabled():
    assert trace_mod.build_trace(enabled=False) is None


def test_unwritable_trace_path_disables_logging_without_raising(tmp_path):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("i am a file")
    trace = trace_mod.build_trace(path=str(blocker / "trace.md"))
    assert trace.enabled is False
    trace.log_answer("this must not raise")  # no exception


def test_eval_set_has_enough_cases_and_valid_expectations(catalog):
    assert 6 <= len(eh.EVAL_CASES) <= 8
    for case in eh.EVAL_CASES:
        assert case.genres, f"{case.request} has no expected genres"
        assert case.moods, f"{case.request} has no expected moods"
        # Expectations must be reachable in the real catalog.
        assert case.genres <= set(catalog.genres), case.request
        assert case.moods <= set(catalog.moods), case.request
        if case.k is not None:
            assert 1 <= case.k <= nl.MAX_K


def test_case_passes_when_correct_and_consistent(catalog):
    case = eh.EvalCase(request="jazz to relax", genres={"jazz"}, moods={"relaxed"})
    provider = FakeProvider([reply("jazz", "relaxed", 5)] * 2)
    result = eh.evaluate_case(case, catalog, provider, runs=2)

    assert result.genre_ok and result.mood_ok and result.k_ok
    assert result.consistent
    assert result.passed


def test_case_fails_when_the_genre_is_wrong(catalog):
    case = eh.EvalCase(request="jazz to relax", genres={"jazz"}, moods={"relaxed"})
    provider = FakeProvider([reply("rock", "relaxed", 5)] * 2)
    result = eh.evaluate_case(case, catalog, provider, runs=2)

    assert result.genre_ok is False
    assert result.consistent is True  # consistently wrong is still inconsistent-free
    assert result.passed is False


def test_inconsistent_runs_fail_even_when_each_run_is_acceptable(catalog):
    """The property a single-run eval cannot detect."""
    case = eh.EvalCase(
        request="chill study music", genres={"lofi", "ambient"}, moods={"chill"}
    )
    provider = FakeProvider([reply("lofi", "chill", 5), reply("ambient", "chill", 5)])
    result = eh.evaluate_case(case, catalog, provider, runs=2)

    assert result.genre_ok is True  # first run is acceptable
    assert result.consistent is False  # but the two runs disagreed
    assert result.passed is False


def test_explicit_count_is_checked(catalog):
    case = eh.EvalCase(request="3 ambient tracks", genres={"ambient"}, moods={"chill"}, k=3)
    good = eh.evaluate_case(case, catalog, FakeProvider([reply("ambient", "chill", 3)] * 2), runs=2)
    bad = eh.evaluate_case(case, catalog, FakeProvider([reply("ambient", "chill", 5)] * 2), runs=2)

    assert good.k_ok and good.passed
    assert bad.k_ok is False and bad.passed is False


def test_count_check_is_skipped_when_the_case_names_no_count(catalog):
    case = eh.EvalCase(request="ambient", genres={"ambient"}, moods={"chill"})
    result = eh.evaluate_case(case, catalog, FakeProvider([reply("ambient", "chill", 7)] * 2), runs=2)
    assert result.k_ok is True and result.passed


def test_single_run_cannot_claim_consistency(catalog):
    case = eh.EvalCase(request="ambient", genres={"ambient"}, moods={"chill"})
    result = eh.evaluate_case(case, catalog, FakeProvider([reply("ambient", "chill", 5)]), runs=1)
    assert result.consistent is False


def test_report_shows_the_table_and_total_score(catalog):
    cases = [
        eh.EvalCase(request="jazz to relax", genres={"jazz"}, moods={"relaxed"}),
        eh.EvalCase(request="hard rock", genres={"rock"}, moods={"intense"}),
    ]
    results = [
        eh.evaluate_case(cases[0], catalog, FakeProvider([reply("jazz", "relaxed", 5)] * 2), runs=2),
        eh.evaluate_case(cases[1], catalog, FakeProvider([reply("pop", "happy", 5)] * 2), runs=2),
    ]
    report = eh.render_report(results, runs=2, provider_label="fake")

    assert "PARSER EVALUATION" in report
    assert "TOTAL SCORE: 1/2" in report
    assert "50%" in report
    assert "Failing cases:" in report
    assert "hard rock" in report
    assert "genre correct" in report
    assert "consistent across runs" in report


def test_report_handles_an_all_passing_run(catalog):
    case = eh.EvalCase(request="jazz", genres={"jazz"}, moods={"relaxed"})
    results = [eh.evaluate_case(case, catalog, FakeProvider([reply("jazz", "relaxed", 5)] * 2), runs=2)]
    report = eh.render_report(results, runs=2, provider_label="fake")

    assert "TOTAL SCORE: 1/1" in report
    assert "100%" in report
    assert "Failing cases:" not in report


def test_evaluate_all_covers_every_case(catalog):
    # Two runs per case, always a valid reply.
    provider = FakeProvider(lambda system, user: reply("lofi", "chill", 3))
    results = eh.evaluate_all(catalog, provider, runs=2)
    assert len(results) == len(eh.EVAL_CASES)
    assert all(len(r.parses) == 2 for r in results)
