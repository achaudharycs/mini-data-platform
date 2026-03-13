"""Agent loop: messages → tool calls → iterate → final answer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.status import Status

from agent.context.discovery import ProjectContext
from agent.core.llm import LLMClient, LLMResponse
from agent.core.session import ConversationManager, SchemaCache
from agent.core.tools import TOOL_DEFINITIONS, dispatch_tool
from agent.display.formatter import print_result_table, print_tool_call
from agent.warehouse.base import QueryResult, Warehouse

_BASE_SYSTEM_PROMPT = """\
You are a data analyst agent with access to a SQL data warehouse. \
Your job is to answer ad-hoc data questions by exploring the warehouse schema and writing SQL queries.

Follow this workflow:
1. ALWAYS start by calling list_schemas to see what schemas exist.
2. Call list_tables on schemas to discover what data is available. \
Prefer schemas with the fewest, widest tables — they are usually the most query-ready.
3. Call describe_table to verify exact column names and types before writing any SQL.
4. Write SQL queries using run_query. Perform aggregations in SQL, not by processing raw rows.
5. If a query fails, read the error message carefully and retry with corrections (up to 3 retries).

Guidelines:
- Always verify column names with describe_table before querying.
- For ambiguous questions, state your assumptions clearly.
- Do NOT include SQL queries in your final answer. The user can view them separately with --show-sql.
- Format numbers with appropriate precision (e.g. currency with 2 decimals).
- If available, use get_dbt_model or get_dag_info to understand how tables are built.
- Your output is displayed in a CLI terminal. Write plain text — no markdown headers, no bold/italic markup. Use simple formatting: dashes for lists, indentation for structure, blank lines for separation.
- Query results from run_query are already displayed to the user as a table. Do NOT repeat, reformat, or re-describe the raw data in your response. Instead, give a brief summary (row count, key observations) and the SQL query used. Let the data speak for itself.
"""

_SCHEMA_CONTEXT_HEADER = """
You already have knowledge of the warehouse schema from prior exploration. \
You can skip introspection tools (list_schemas, list_tables, describe_table) \
and go straight to run_query unless the user asks about a table you haven't seen.

{schema_summary}
"""


@dataclass
class LogEntry:
    """A single entry in the agent's activity log."""

    kind: str  # "thinking", "tool_call", "tool_result", "error"
    content: str


class Agent:
    """Tool-use agent that answers data questions by querying the warehouse."""

    def __init__(
        self,
        llm: LLMClient,
        warehouse: Warehouse,
        context: ProjectContext,
        *,
        max_iterations: int = 15,
        verbose: bool = False,
        page_size: int = 50,
        schema_cache: SchemaCache | None = None,
        conversation: ConversationManager | None = None,
    ) -> None:
        self._llm = llm
        self._warehouse = warehouse
        self._context = context
        self._max_iterations = max_iterations
        self._verbose = verbose
        self._page_size = page_size
        self._last_sql: str | None = None
        self._last_result: QueryResult | None = None
        self._displayed_results: list[QueryResult] = []
        self._log: list[LogEntry] = []
        self._console = Console()
        self._schema_cache = schema_cache or SchemaCache(warehouse)
        self._conversation = conversation or ConversationManager()

    @property
    def last_sql(self) -> str | None:
        """The last SQL query executed during the most recent run."""
        return self._last_sql

    @property
    def last_result(self) -> QueryResult | None:
        """The full QueryResult from the last SQL query (up to 10k rows)."""
        return self._last_result

    @property
    def displayed_results(self) -> list[QueryResult]:
        """All query results displayed as tables during the last run."""
        return self._displayed_results

    @property
    def log(self) -> list[LogEntry]:
        """Activity log from the most recent run."""
        return self._log

    @property
    def schema_cache(self) -> SchemaCache:
        return self._schema_cache

    @property
    def conversation(self) -> ConversationManager:
        return self._conversation

    def _build_system_prompt(self) -> str:
        """Assemble the system prompt with optional schema context and history."""
        parts = [_BASE_SYSTEM_PROMPT]

        # Inject cached schema knowledge so LLM can skip introspection
        schema_summary = self._schema_cache.build_schema_summary()
        if schema_summary:
            parts.append(_SCHEMA_CONTEXT_HEADER.format(schema_summary=schema_summary))

        # Inject conversation summary for older turns
        history_block = self._conversation.build_context_block()
        if history_block:
            parts.append(history_block)

        return "\n".join(parts)

    def _build_messages(self, question: str) -> list[dict[str, Any]]:
        """Build the message list with conversation history + current question."""
        messages = self._conversation.build_history_messages()
        messages.append({"role": "user", "content": question})
        return messages

    def run(self, question: str) -> str:
        """Run the agent loop for a question and return the final text answer."""
        self._last_sql = None
        self._log = []
        self._last_result = None
        self._displayed_results = []
        system = self._build_system_prompt()
        messages = self._build_messages(question)

        for iteration in range(self._max_iterations):
            response: LLMResponse = self._llm.send(
                messages=messages,
                system=system,
                tools=TOOL_DEFINITIONS,
            )

            if not response.tool_calls:
                self._conversation.add_turn(question, response.text)
                return response.text

            if response.text:
                self._log.append(LogEntry(kind="thinking", content=response.text))

            messages, tool_results = self._handle_tool_calls(response, messages)

        return "I've reached the maximum number of iterations. Please try rephrasing your question or breaking it into smaller parts."

    def stream_answer(self, question: str) -> str:
        """Run the agent loop with a status spinner, then stream the final answer.

        In default mode: tool-call iterations show a spinner with the
        current tool name. The final answer streams token-by-token.
        All activity is recorded in the log for later review.

        In verbose mode: everything streams live (thinking + tool calls + answer).
        """
        self._last_sql = None
        self._last_result = None
        self._displayed_results = []
        self._log = []

        if self._verbose:
            return self._stream_verbose(question)
        return self._stream_with_spinner(question)

    def _stream_verbose(self, question: str) -> str:
        """Verbose mode: stream all text live, show tool calls inline."""
        system = self._build_system_prompt()
        messages = self._build_messages(question)

        for iteration in range(self._max_iterations):
            response: LLMResponse = self._llm.send(
                messages=messages,
                system=system,
                tools=TOOL_DEFINITIONS,
                stream_text=True,
            )

            if not response.tool_calls:
                self._conversation.add_turn(question, response.text)
                return response.text

            if response.text:
                self._log.append(LogEntry(kind="thinking", content=response.text))

            messages, _ = self._handle_tool_calls(
                response, messages, show_tool_calls=True
            )

        return "I've reached the maximum number of iterations. Please try rephrasing your question or breaking it into smaller parts."

    def _stream_with_spinner(self, question: str) -> str:
        """Default mode: spinner during tool calls, stream final answer."""
        system = self._build_system_prompt()
        messages = self._build_messages(question)

        with Status(
            "Thinking...", console=self._console, spinner="dots"
        ) as status:
            for iteration in range(self._max_iterations):
                response: LLMResponse = self._llm.send(
                    messages=messages,
                    system=system,
                    tools=TOOL_DEFINITIONS,
                )

                if not response.tool_calls:
                    status.stop()
                    streamed = self._llm.send(
                        messages=messages,
                        system=system,
                        tools=TOOL_DEFINITIONS,
                        stream_text=True,
                    )
                    self._conversation.add_turn(question, streamed.text)
                    return streamed.text

                if response.text:
                    self._log.append(
                        LogEntry(kind="thinking", content=response.text)
                    )

                # Execute tools, updating spinner with current tool
                tool_results: list[dict[str, Any]] = []
                for tc in response.tool_calls:
                    status.update(_tool_active_label(tc.name, tc.input))

                    self._log.append(
                        LogEntry(kind="tool_call", content=_format_tool_call(tc.name, tc.input))
                    )

                    if tc.name == "run_query" and "sql" in tc.input:
                        self._last_sql = tc.input["sql"]
                        # Cache full result for user pagination
                        try:
                            self._last_result = self._schema_cache.execute_query(
                                tc.input["sql"], max_rows=10_000
                            )
                        except Exception:
                            self._last_result = None

                    try:
                        result_text = dispatch_tool(
                            tc.name, tc.input, self._schema_cache, self._context
                        )
                        is_error = False
                    except Exception as e:
                        result_text = f"Error: {e}"
                        is_error = True

                    # Display query results to the user (first page only)
                    if tc.name == "run_query" and not is_error and self._last_result and self._last_result.rows:
                        self._displayed_results.append(self._last_result)
                        table_num = len(self._displayed_results)
                        title = _table_title_from_sql(tc.input.get("sql", ""), table_num)
                        status.stop()
                        display_result = _first_page(self._last_result, self._page_size)
                        print_result_table(display_result, title=title)
                        status.update("Thinking...")
                        status.start()

                    self._log.append(
                        LogEntry(kind="error" if is_error else "tool_result", content=result_text)
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": result_text,
                            "is_error": is_error,
                        }
                    )

                # Append to message history
                assistant_content: list[dict[str, Any]] = []
                if response.text:
                    assistant_content.append({"type": "text", "text": response.text})
                for tc in response.tool_calls:
                    assistant_content.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.input,
                        }
                    )
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

        return "I've reached the maximum number of iterations. Please try rephrasing your question or breaking it into smaller parts."

    def _handle_tool_calls(
        self,
        response: LLMResponse,
        messages: list[dict[str, Any]],
        *,
        show_tool_calls: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Process tool calls from a response and append to messages.

        Returns the updated messages list and the tool results.
        """
        assistant_content: list[dict[str, Any]] = []
        if response.text:
            assistant_content.append({"type": "text", "text": response.text})
        for tc in response.tool_calls:
            assistant_content.append(
                {
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.input,
                }
            )
        messages.append({"role": "assistant", "content": assistant_content})

        tool_results: list[dict[str, Any]] = []
        for tc in response.tool_calls:
            if show_tool_calls:
                print_tool_call(tc.name, tc.input)

            self._log.append(
                LogEntry(
                    kind="tool_call",
                    content=_format_tool_call(tc.name, tc.input),
                )
            )

            if tc.name == "run_query" and "sql" in tc.input:
                self._last_sql = tc.input["sql"]
                try:
                    self._last_result = self._schema_cache.execute_query(
                        tc.input["sql"], max_rows=10_000
                    )
                except Exception:
                    self._last_result = None

            try:
                result_text = dispatch_tool(
                    tc.name, tc.input, self._schema_cache, self._context
                )
                is_error = False
            except Exception as e:
                result_text = f"Error: {e}"
                is_error = True

            # Display query results to the user (first page only)
            if tc.name == "run_query" and not is_error and self._last_result and self._last_result.rows:
                self._displayed_results.append(self._last_result)
                table_num = len(self._displayed_results)
                title = _table_title_from_sql(tc.input.get("sql", ""), table_num)
                display_result = _first_page(self._last_result, self._page_size)
                print_result_table(display_result, title=title)

            self._log.append(
                LogEntry(
                    kind="error" if is_error else "tool_result",
                    content=result_text,
                )
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                    "is_error": is_error,
                }
            )

        messages.append({"role": "user", "content": tool_results})
        return messages, tool_results


_TOOL_ACTIVE_LABELS = {
    "list_schemas": "Exploring schemas...",
    "list_tables": "Listing tables in {schema}...",
    "describe_table": "Inspecting {schema}.{table}...",
    "run_query": "Running SQL query...",
    "get_dbt_model": "Reading dbt model {name}...",
    "get_dag_info": "Reading DAG {name}...",
}


def _tool_active_label(name: str, inputs: dict[str, Any]) -> str:
    """Human-readable label for an in-progress tool call."""
    template = _TOOL_ACTIVE_LABELS.get(name, name)
    try:
        return template.format(**inputs)
    except KeyError:
        return template


def _format_tool_call(name: str, inputs: dict[str, Any]) -> str:
    """Format a tool call for the log."""
    args = ", ".join(f"{k}={v!r}" for k, v in inputs.items())
    return f"{name}({args})"


import re

_FROM_RE = re.compile(r"\bFROM\s+([\w.]+)", re.IGNORECASE)


def _table_title_from_sql(sql: str, table_num: int) -> str:
    """Derive a human-readable table title from a SQL query."""
    match = _FROM_RE.search(sql)
    if match:
        table_name = match.group(1).split(".")[-1]  # strip schema prefix
        return f"#{table_num} — {table_name}"
    return f"#{table_num}"


def _first_page(result: QueryResult, page_size: int) -> QueryResult:
    """Return the first page of a query result for display."""
    if len(result.rows) <= page_size:
        return result
    return QueryResult(
        columns=result.columns,
        rows=result.rows[:page_size],
        row_count=page_size,
        truncated=True,
    )
