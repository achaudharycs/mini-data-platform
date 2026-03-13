"""CLI entry point — interactive REPL for the data agent."""

from __future__ import annotations

import readline  # noqa: F401 — enables up/down arrow history in input()

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from agent.config import Config
from agent.context.discovery import ProjectContext
from agent.core.agent import Agent, LogEntry
from agent.core.llm import AnthropicClient
from agent.core.session import ConversationManager, SchemaCache
from agent.display.formatter import ResultPager, print_error, print_info, print_sql
from agent.warehouse.duckdb import DuckDBWarehouse

app = typer.Typer(
    name="agent",
    help="Interactive data agent that answers questions by querying the warehouse.",
    add_completion=False,
)
console = Console()


@app.command()
def repl(
    show_sql: bool = typer.Option(False, "--show-sql", help="Show SQL queries executed by the agent."),
    verbose: bool = typer.Option(False, "--verbose", help="Show all tool calls and thinking live."),
) -> None:
    """Start the interactive data agent REPL."""
    config = Config.from_env()

    if not config.anthropic_api_key:
        print_error("ANTHROPIC_API_KEY environment variable is not set.")
        raise typer.Exit(1)

    if not config.warehouse_path.exists():
        print_error(f"Warehouse not found at {config.warehouse_path}. Run setup.sh first.")
        raise typer.Exit(1)

    # Initialize components — schema cache and conversation persist across questions
    warehouse = DuckDBWarehouse(config.warehouse_path)
    schema_cache = SchemaCache(warehouse)
    conversation = ConversationManager()
    context = ProjectContext(config.warehouse_path.parent.parent)
    llm = AnthropicClient(api_key=config.anthropic_api_key, model=config.model)
    agent = Agent(
        llm=llm,
        warehouse=warehouse,
        context=context,
        max_iterations=config.max_iterations,
        verbose=verbose,
        page_size=config.page_size,
        schema_cache=schema_cache,
        conversation=conversation,
    )

    console.print("\n[bold]Mini Data Platform Agent[/bold]")
    console.print(
        "Ask questions about your data. "
        "Type [bold]help[/bold] to see available commands.\n"
    )

    try:
        _run_loop(agent, show_sql=show_sql, page_size=config.page_size)
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye![/dim]")
    finally:
        schema_cache.close()


def _run_loop(agent: Agent, *, show_sql: bool, page_size: int) -> None:
    """Main REPL loop."""
    pagers: dict[int, ResultPager] = {}

    while True:
        try:
            question = console.input("[bold green]>[/bold green] ").strip()
        except EOFError:
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print_info("Goodbye!")
            break
        if question.lower() == "help":
            _print_help()
            continue
        if question.lower() == "log":
            _print_log(agent.log)
            continue
        if question.lower().startswith("more"):
            _handle_more(question, pagers)
            continue

        try:
            console.print()  # blank line before answer
            agent.stream_answer(question)

            # Set up pagers for all displayed results that exceed one page
            pagers = {}
            pageable = []
            for i, result in enumerate(agent.displayed_results, 1):
                if result.row_count > page_size:
                    pagers[i] = ResultPager(result, page_size=page_size, start_offset=page_size)
                    pageable.append((i, result.row_count))

            if len(pageable) == 1:
                _, count = pageable[0]
                console.print(
                    f"\n[dim]{count} rows available. "
                    f"Type [bold]more[/bold] to see next page.[/dim]"
                )
            elif len(pageable) > 1:
                parts = [f"#{i} ({count} rows)" for i, count in pageable]
                console.print(
                    f"\n[dim]Pageable results: {', '.join(parts)}. "
                    f"Type [bold]more[/bold] or [bold]more #N[/bold] to page.[/dim]"
                )

            if show_sql and agent.last_sql:
                console.print()
                print_sql(agent.last_sql)

            console.print()  # blank line after answer

        except Exception as e:
            print_error(str(e))


def _handle_more(question: str, pagers: dict[int, ResultPager]) -> None:
    """Handle the 'more' command with optional table number."""
    if not pagers:
        console.print("[dim]No more results to show.[/dim]")
        return

    # Parse optional table number: "more", "more 1", "more #2"
    parts = question.split()
    table_num: int | None = None
    if len(parts) > 1:
        try:
            table_num = int(parts[1].lstrip("#"))
        except ValueError:
            console.print("[dim]Usage: more or more #N[/dim]")
            return

    if table_num is not None:
        pager = pagers.get(table_num)
        if not pager:
            console.print(f"[dim]No pageable result #{table_num}.[/dim]")
            return
    elif len(pagers) == 1:
        pager = next(iter(pagers.values()))
    else:
        # Multiple pagers, no number specified — show the first with remaining rows
        pager = None
        for p in pagers.values():
            if p.has_more:
                pager = p
                break
        if not pager:
            console.print("[dim]No more results to show.[/dim]")
            return

    if pager.has_more:
        pager.show_next_page()
    else:
        console.print("[dim]No more rows to display.[/dim]")

    # Show remaining pages across all tables
    _print_pager_status(pagers)


def _print_pager_status(pagers: dict[int, ResultPager]) -> None:
    """Show which tables still have pages remaining."""
    remaining = [(i, p.remaining) for i, p in pagers.items() if p.has_more]
    if not remaining:
        return
    if len(remaining) == 1:
        i, rows = remaining[0]
        label = f"more #{i}" if len(pagers) > 1 else "more"
        console.print(f"[dim]{rows} rows remaining. Type [bold]{label}[/bold] to continue.[/dim]")
    else:
        parts = [f"#{i} ({rows} rows left)" for i, rows in remaining]
        console.print(
            f"[dim]Remaining: {', '.join(parts)}. "
            f"Type [bold]more #N[/bold] to continue.[/dim]"
        )


def _print_help() -> None:
    """Print available commands."""
    console.print()
    console.print("[bold]Available commands:[/bold]")
    console.print("  [bold]help[/bold]   — Show this help message")
    console.print("  [bold]log[/bold]    — Show the agent's activity log from the last question")
    console.print("  [bold]more[/bold]   — Page through results (use [bold]more #N[/bold] to pick a specific table)")
    console.print("  [bold]quit[/bold]   — Exit the agent")
    console.print()


def _print_log(log: list[LogEntry]) -> None:
    """Print the full activity log from the last question."""
    if not log:
        console.print("[dim]No activity log yet. Ask a question first.[/dim]")
        return

    console.print()
    console.print("[bold]Agent Activity Log[/bold]")
    console.print()

    for entry in log:
        if entry.kind == "thinking":
            console.print(
                Panel(
                    entry.content,
                    title="[yellow]Thinking[/yellow]",
                    border_style="yellow",
                    padding=(0, 1),
                )
            )
        elif entry.kind == "tool_call":
            text = Text()
            text.append("→ ", style="bold cyan")
            text.append(entry.content, style="cyan")
            console.print(text)
        elif entry.kind == "tool_result":
            content = entry.content
            lines = content.split("\n")
            if len(lines) > 20:
                content = "\n".join(lines[:20]) + f"\n... ({len(lines) - 20} more lines)"
            console.print(
                Panel(
                    content,
                    title="[green]Result[/green]",
                    border_style="dim",
                    padding=(0, 1),
                )
            )
        elif entry.kind == "error":
            console.print(
                Panel(
                    entry.content,
                    title="[red]Error[/red]",
                    border_style="red",
                    padding=(0, 1),
                )
            )

    console.print()


if __name__ == "__main__":
    app()
