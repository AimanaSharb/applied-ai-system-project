"""
Command line runner for the Music Recommender Simulation.

Modes:
  python -m src.main                       AI mode - ask in plain English.
  python -m src.main --ask "TEXT"          AI mode, one question, then exit.
  python -m src.main --classic             Original rule-based mode, no API key.

The AI model is reached through src/llm.py, so --provider selects the backend.
"""

import argparse
import sys

try:  # run as: python src/main.py
    from recommender import load_songs, recommend_songs
    import nl_recommender as nl
    import trace as trace_mod
except ModuleNotFoundError:  # run as: python -m src.main
    from src.recommender import load_songs, recommend_songs
    from src import nl_recommender as nl
    from src import trace as trace_mod

CSV_PATH = "data/songs.csv"
RULE = "=" * 60


def run_classic() -> None:
    """The original, non-AI recommender with a fixed profile."""
    songs = load_songs(CSV_PATH)
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    recommendations = recommend_songs(user_prefs, songs, k=5)

    print(RULE)
    print("  MUSIC RECOMMENDER (classic mode)")
    print(
        f"  Profile: genre={user_prefs['genre']}, "
        f"mood={user_prefs['mood']}, energy={user_prefs['energy']}"
    )
    print(f"  Loaded {len(songs)} songs")
    print(RULE)
    print(f"\nTop {len(recommendations)} recommendations:\n")

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        print(f"{rank}. {song['title']} - {song['artist']}  (score: {score:.2f})")
        print(f"   Because: {explanation}")
        print()


def print_result(result) -> None:
    """Render one natural-language result."""
    p = result.parsed
    print(
        f"\nInterpreted as: genre={p.genre}, mood={p.mood}, k={p.k}"
        + (f"  ({p.reasoning})" if p.reasoning else "")
    )
    for warning in result.warnings:
        print(f"  ! {warning}")

    print(f"\nTop {len(result.recommendations)} matches:\n")
    for rank, rec in enumerate(result.recommendations, start=1):
        print(f"{rank}. {rec.song.title} - {rec.song.artist}  (score: {rec.score:.2f})")
        print(f"   Because: {rec.explanation}")
    print(f"\n{result.answer}\n")


def handle_one(text: str, catalog, provider, trace, verify: bool = True) -> bool:
    """Process one request. Returns True on success, False on a handled error."""
    try:
        result = nl.recommend_from_text(
            text, catalog=catalog, provider=provider, verify=verify, trace=trace
        )
    except nl.NLRecommenderError as exc:
        # Empty input, non-music requests, and API failures all land here.
        if trace:
            trace.log_error(str(exc))
        print(f"\n  {exc}")
        return False

    print_result(result)
    return True


def run_ai(args) -> int:
    """Interactive natural-language mode. Returns a process exit code."""
    try:
        catalog = nl.Catalog.load(CSV_PATH)
        provider = nl.build_provider(args.provider)
    except nl.NLRecommenderError as exc:
        print(f"\nCannot start AI mode: {exc}\n", file=sys.stderr)
        return 1

    trace = trace_mod.build_trace(
        provider_label=provider.describe(), enabled=not args.no_trace
    )

    print(RULE)
    print("  MUSIC RECOMMENDER (AI mode)")
    print(f"  model: {provider.describe()}")
    print(f"  {len(catalog.songs)} songs | genres: {', '.join(catalog.genres)}")
    print(f"  moods: {', '.join(catalog.moods)}")
    if trace and trace.enabled:
        print(f"  reasoning trace: {trace.path}")
    print(RULE)

    verify = not args.no_verify
    if args.ask:
        return 0 if handle_one(args.ask, catalog, provider, trace, verify) else 1

    print("Describe what you want to hear. Type 'quit' to exit.")
    while True:
        try:
            text = input("\nWhat are you in the mood for? ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0

        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            return 0
        handle_one(text, catalog, provider, trace, verify)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music Recommender Simulation - rule-based retrieval with an "
        "AI natural-language front end."
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="run the original rule-based recommender (no API key required)",
    )
    parser.add_argument(
        "--ask",
        metavar="TEXT",
        help="answer a single natural-language request and exit",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "gemini"],
        help="which model backend to use (default: $LLM_PROVIDER, else anthropic)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the agentic self-check step",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="do not write logs/agent_trace.md",
    )
    args = parser.parse_args()

    if args.classic:
        run_classic()
        return

    sys.exit(run_ai(args))


if __name__ == "__main__":
    main()
