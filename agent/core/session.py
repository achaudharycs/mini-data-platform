"""Session context: schema cache + conversation history for multi-turn sessions."""

from __future__ import annotations

from typing import Any

from agent.warehouse.base import Warehouse


class SchemaCache:
    """Memoizes warehouse introspection results within a session.

    Caches list_schemas, list_tables, and describe_table results so repeated
    calls return instantly. Also builds a schema summary string for injection
    into the LLM system prompt, letting it skip introspection entirely on
    follow-up questions.
    """

    def __init__(self, warehouse: Warehouse) -> None:
        self._warehouse = warehouse
        self._schemas: list[str] | None = None
        self._tables: dict[str, list[dict[str, Any]]] = {}  # schema -> table info
        self._columns: dict[tuple[str, str], list[dict[str, Any]]] = {}  # (schema, table) -> columns

    def list_schemas(self) -> list[str]:
        if self._schemas is None:
            self._schemas = self._warehouse.list_schemas()
        return self._schemas

    def list_tables(self, schema: str) -> list:
        if schema not in self._tables:
            self._tables[schema] = self._warehouse.list_tables(schema)
        return self._tables[schema]

    def describe_table(self, schema: str, table: str) -> list:
        key = (schema, table)
        if key not in self._columns:
            self._columns[key] = self._warehouse.describe_table(schema, table)
        return self._columns[key]

    def execute_query(self, sql: str, max_rows: int = 100):
        """Queries are never cached — always hit the warehouse."""
        return self._warehouse.execute_query(sql, max_rows=max_rows)

    def close(self) -> None:
        self._warehouse.close()

    @property
    def has_cached_schema(self) -> bool:
        """True if any schema introspection has been cached."""
        return self._schemas is not None

    def build_schema_summary(self) -> str:
        """Build a text summary of all cached schema knowledge.

        Returns an empty string if nothing has been cached yet.
        """
        if self._schemas is None:
            return ""

        lines: list[str] = ["Known schema context (from prior exploration):"]
        lines.append(f"  Schemas: {', '.join(self._schemas)}")

        for schema, tables in self._tables.items():
            table_parts = []
            for t in tables:
                count = f" ({t.row_count:,} rows)" if t.row_count is not None else ""
                table_parts.append(f"{t.table_name}{count}")
            lines.append(f"  {schema} tables: {', '.join(table_parts)}")

            for t in tables:
                key = (schema, t.table_name)
                if key in self._columns:
                    col_names = [c.name for c in self._columns[key]]
                    lines.append(f"    {t.table_name} columns: {', '.join(col_names)}")

        return "\n".join(lines)


class ConversationManager:
    """Manages conversation history with a sliding window.

    Keeps the last `window_size` turns (question + answer pairs) in full.
    Older turns are compressed into a running summary to keep the token
    count bounded.
    """

    def __init__(self, *, window_size: int = 3) -> None:
        self._window_size = window_size
        self._turns: list[dict[str, str]] = []  # [{"question": ..., "answer": ...}, ...]
        self._summary: str = ""

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def recent_turns(self) -> list[dict[str, str]]:
        """The most recent turns kept in full."""
        return self._turns[-self._window_size :]

    def add_turn(self, question: str, answer: str) -> None:
        """Record a completed question-answer turn.

        If we exceed the window size, the oldest turn gets folded into
        the running summary.
        """
        self._turns.append({"question": question, "answer": answer})

        # Compress older turns into summary
        while len(self._turns) > self._window_size:
            old = self._turns.pop(0)
            entry = f"Q: {_truncate(old['question'], 80)} -> A: {_truncate(old['answer'], 150)}"
            if self._summary:
                self._summary += "\n" + entry
            else:
                self._summary = entry

    def build_history_messages(self) -> list[dict[str, Any]]:
        """Build message list from recent turns for the LLM context.

        Returns alternating user/assistant messages from the sliding window.
        Does NOT include the current question — the caller appends that.
        """
        messages: list[dict[str, Any]] = []
        for turn in self.recent_turns:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        return messages

    def build_context_block(self) -> str:
        """Build a text block summarizing conversation history.

        Returned as a string suitable for injection into the system prompt.
        """
        if not self._turns and not self._summary:
            return ""

        parts: list[str] = []
        if self._summary:
            parts.append(f"Conversation history (summarized):\n{self._summary}")
        return "\n".join(parts)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters, adding ellipsis if needed."""
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
