import typer

app = typer.Typer(help="ctxrot — LLM context analytics for DSPy")


@app.callback(invoke_without_command=True)
def default(
    ctx: typer.Context,
    db: str = typer.Option("ctxrot.db", "--db", "-d", help="Path to database file"),
    session: str | None = typer.Option(
        None, "--session", "-s", help="Session ID to display"
    ),
) -> None:
    """Launch the TUI dashboard (default command)."""
    if ctx.invoked_subcommand is None:
        _run_dashboard(db, session)


@app.command()
def dashboard(
    db: str = typer.Option("ctxrot.db", "--db", "-d"),
    session: str | None = typer.Option(None, "--session", "-s"),
) -> None:
    """Launch the TUI dashboard."""
    _run_dashboard(db, session)


@app.command()
def reset(
    db: str = typer.Option("ctxrot.db", "--db", "-d"),
) -> None:
    """Truncate all tables in the database."""
    from ctxrot.storage import CtxRotStore

    store = CtxRotStore(db)
    store.truncate_all()
    store.close()
    typer.echo(f"All tables truncated in {db}")


@app.command()
def analyze(
    db: str = typer.Option("ctxrot.db", "--db", "-d"),
    session: str | None = typer.Option(None, "--session", "-s"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Analyze a session for context rot (repetition detection)."""
    import json

    from ctxrot.analysis import analyze_session
    from ctxrot.storage import CtxRotStore

    store = CtxRotStore(db, read_only=True)
    sid = session or store.get_latest_session_id()
    if not sid:
        typer.echo("No sessions found.")
        raise typer.Exit(1)

    result = analyze_session(store, sid)
    store.close()

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    summary = result["summary"]
    typer.echo(f"\nContext Rot Analysis — session {sid[:8]}")
    typer.echo("=" * 50)

    terminal_state = summary.get("terminal_state")
    if terminal_state:
        typer.echo(f"Terminal state: {terminal_state}")

    if not result["has_content"]:
        typer.echo(
            "\n  No content data available."
            "\n  Re-run with store_content=True in CtxRotCallback"
            "\n  to enable repetition detection."
        )
        typer.echo("\n  Showing efficiency ratios only:\n")
    else:
        if summary["repetition_detected"]:
            typer.echo(
                f"\n  REPETITION DETECTED at iteration #{summary['onset_iteration']}"
                f"  (max overlap: {summary['max_repetition']:.2f})"
            )
        else:
            typer.echo(
                f"\n  Repetition: LOW (max overlap: {summary['max_repetition']:.2f})"
            )

        typer.echo("\n  Per-iteration repetition:")
        for row in result["repetition"]:
            tag = " <<<" if row["ngram_jaccard"] > 0.4 else ""
            typer.echo(
                f"    #{row['seq']:>2d}  ngram={row['ngram_jaccard']:.2f}"
                f"  seq={row['sequence_similarity']:.2f}"
                f"  cumul={row['cumulative_max']:.2f}{tag}"
            )

    if result["efficiency"]:
        typer.echo("\n  Efficiency (completion/prompt ratio):")
        for row in result["efficiency"]:
            typer.echo(
                f"    #{row['seq']:>2d}  {row['efficiency_ratio']:.4f}"
                f"  ({row['completion_tokens']:,} / {row['prompt_tokens']:,})"
            )

    typer.echo("")


@app.command("deep-analyze")
def deep_analyze(
    db: str = typer.Option("ctxrot.db", "--db", "-d"),
    session: str | None = typer.Option(None, "--session", "-s"),
    query: str = typer.Option(
        "Perform a comprehensive context rot analysis.",
        "--query",
        "-q",
        help="Focus area or specific question for the analysis",
    ),
    main_model: str = typer.Option(
        "openai/gpt-5.4",
        "--model",
        "-m",
        help="Main LM for RLM reasoning",
    ),
    sub_model: str = typer.Option(
        "openai/gpt-5.4-mini",
        "--sub-model",
        help="Sub LM for semantic analysis (llm_query calls)",
    ),
    max_iterations: int = typer.Option(
        15, "--max-iters", help="Max RLM REPL iterations"
    ),
    max_llm_calls: int = typer.Option(30, "--max-calls", help="Max sub-LLM calls"),
    api_key: str | None = typer.Option(
        None, "--api-key", help="API key (or set OPENAI_API_KEY env var / .env)"
    ),
    api_base: str | None = typer.Option(
        None, "--api-base", help="API base URL (or set OPENAI_API_BASE env var / .env)"
    ),
    env_file: str | None = typer.Option(
        ".env", "--env-file", help="Path to .env file for API credentials"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Output full result as JSON"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show RLM reasoning steps"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip cost warning confirmation"
    ),
) -> None:
    """Run AI-powered deep analysis on a session using DSPy RLM."""
    import json

    from ctxrot.storage import CtxRotStore

    store = CtxRotStore(db, read_only=True)
    sid = session or store.get_latest_session_id()
    if not sid:
        typer.echo("No sessions found.")
        raise typer.Exit(1)

    if not yes:
        session_info = store.get_session(sid)
        summary = store.get_session_summary(sid)
        session_data = session_info or {}
        model_name = session_data.get("model", "unknown")
        typer.echo(
            f"\n  Deep analysis of session {sid[:8]} ({model_name})"
            f"\n  {summary['total_calls']} LM calls recorded"
            f"\n"
            f"\n  This will use:"
            f"\n    Main LM: {main_model} (up to {max_iterations} calls)"
            f"\n    Sub  LM: {sub_model} (up to {max_llm_calls} calls)"
            f"\n"
            f"\n  Estimated cost: $0.10 - $2.00 depending on session size"
        )
        if not typer.confirm("\n  Proceed?", default=True):
            raise typer.Exit(0)

    try:
        from rich.console import Console

        from ctxrot.deep_analysis import run_deep_analysis

        console = Console()
        with console.status(f"Running deep analysis on session {sid[:8]}..."):
            result = run_deep_analysis(
                store=store,
                session_id=sid,
                query=query,
                main_model=main_model,
                sub_model=sub_model,
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                verbose=verbose,
                api_key=api_key,
                api_base=api_base,
                env_file=env_file,
            )
    except Exception as e:
        typer.echo(f"\n  Error: {e}", err=True)
        raise typer.Exit(1) from None
    finally:
        store.close()

    if json_output:
        typer.echo(json.dumps(result, indent=2, default=str))
        return

    typer.echo(result["report"])

    missing = result.get("missing_sections", [])
    if missing:
        typer.echo(
            f"\n  Warning: Report is missing sections: {', '.join(missing)}",
            err=True,
        )

    typer.echo(f"\n  ---" f"\n  RLM used {len(result['trajectory'])} REPL iterations.")


@app.command()
def export(
    db: str = typer.Option("ctxrot.db", "--db", "-d"),
    session: list[str] = typer.Option(
        [],
        "--session",
        "-s",
        help="Session ID (repeatable for multiple IDs). Bypasses filters.",
    ),
    all_: bool = typer.Option(
        False, "--all", help="Export every session in the DB"
    ),
    since: str | None = typer.Option(
        None,
        "--since",
        help="Filter: sessions started at/after this ISO datetime",
    ),
    until: str | None = typer.Option(
        None,
        "--until",
        help="Filter: sessions started at/before this ISO datetime",
    ),
    only_errored: bool = typer.Option(
        False,
        "--only-errored",
        help="Filter: only sessions with terminal_state='errored'",
    ),
    only_completed: bool = typer.Option(
        False,
        "--only-completed",
        help="Filter: only sessions with terminal_state='completed'",
    ),
    fmt: str = typer.Option(
        "opentraces",
        "--format",
        "-f",
        help="Output format: 'opentraces' (TraceRecord shape) or 'ctxrot' (native)",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (stdout if omitted)",
    ),
) -> None:
    """Export sessions to JSONL (opentraces TraceRecord shape by default)."""
    import sys

    from ctxrot.export import resolve_session_ids, stream_export
    from ctxrot.storage import CtxRotStore

    if only_errored and only_completed:
        typer.echo(
            "Error: --only-errored and --only-completed are mutually exclusive.",
            err=True,
        )
        raise typer.Exit(2)

    if fmt not in ("opentraces", "ctxrot"):
        typer.echo(
            f"Error: --format must be 'opentraces' or 'ctxrot', got {fmt!r}.",
            err=True,
        )
        raise typer.Exit(2)

    typer.echo(
        "WARNING: export includes raw prompts, completions, and tool outputs "
        "when available. Review for secrets/PII before sharing. Redaction is "
        "not yet available.",
        err=True,
    )

    store = CtxRotStore(db, read_only=True)
    try:
        ids = resolve_session_ids(
            store,
            sessions=session,
            all_=all_,
            since=since,
            until=until,
            only_errored=only_errored,
            only_completed=only_completed,
        )

        def _progress(n: int) -> None:
            typer.echo(f"  exported {n} sessions...", err=True)

        if output is None:
            count = stream_export(
                store,
                ids,
                sys.stdout,
                fmt=fmt,
            )
        else:
            with open(output, "w", encoding="utf-8") as f:
                count = stream_export(
                    store,
                    ids,
                    f,
                    fmt=fmt,
                    progress=_progress,
                )
    finally:
        store.close()

    typer.echo(f"  done: {count} session(s) exported.", err=True)


@app.command()
def tail() -> None:
    """Stream LM calls in real-time (coming soon)."""
    typer.echo("Coming soon")


@app.command()
def summary() -> None:
    """Print one-shot session stats (coming soon)."""
    typer.echo("Coming soon")


def _run_dashboard(db: str, session: str | None) -> None:
    from ctxrot.tui.app import CtxRotApp

    app = CtxRotApp(db_path=db, session_id=session)
    app.run()
