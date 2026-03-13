"""Rich display formatting for agent output."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from agent.warehouse.base import QueryResult

console = Console()


def print_result_table(result: QueryResult, title: str | None = None) -> None:
    """Render a QueryResult as a Rich table."""
    table = Table(title=title, show_lines=False, pad_edge=True)
    for col in result.columns:
        table.add_column(col, overflow="fold")
    for row in result.rows:
        table.add_row(*(str(v) if v is not None else "NULL" for v in row))
    console.print(table)
    if result.truncated:
        console.print(
            f"[dim]Showing {result.row_count} rows (more available — type 'more' to see next page)[/dim]"
        )


def print_sql(sql: str) -> None:
    """Render SQL with syntax highlighting."""
    console.print(Syntax(sql.strip(), "sql", theme="monokai", padding=1))


def print_markdown(text: str) -> None:
    """Render markdown text."""
    console.print(Markdown(text))


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[dim]{message}[/dim]")


def print_tool_call(name: str, inputs: dict) -> None:
    """Print a tool call for verbose mode."""
    args = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    console.print(f"[dim]→ {name}({args})[/dim]")


class ResultPager:
    """Caches a full query result and pages through it."""

    def __init__(self, result: QueryResult, page_size: int = 100, start_offset: int = 0) -> None:
        self._result = result
        self._page_size = page_size
        self._offset = start_offset

    @property
    def has_more(self) -> bool:
        return self._offset < len(self._result.rows)

    @property
    def remaining(self) -> int:
        """Number of rows not yet displayed."""
        return max(0, len(self._result.rows) - self._offset)

    def show_next_page(self) -> bool:
        """Display the next page. Returns True if more pages remain."""
        if not self.has_more:
            console.print("[dim]No more rows to display.[/dim]")
            return False

        end = min(self._offset + self._page_size, len(self._result.rows))
        page_rows = self._result.rows[self._offset : end]

        page_result = QueryResult(
            columns=self._result.columns,
            rows=page_rows,
            row_count=len(page_rows),
            truncated=end < len(self._result.rows),
        )
        print_result_table(
            page_result,
            title=f"Rows {self._offset + 1}–{end} of {len(self._result.rows)}",
        )
        self._offset = end
        return self.has_more
