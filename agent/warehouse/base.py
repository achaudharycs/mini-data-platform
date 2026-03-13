"""Warehouse protocol and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool = True
    sample_values: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TableInfo:
    schema_name: str
    table_name: str
    table_type: str  # "TABLE" or "VIEW"
    row_count: int | None = None


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[list[object]]
    row_count: int
    truncated: bool = False  # True if more rows exist beyond what was returned


@runtime_checkable
class Warehouse(Protocol):
    """Protocol for warehouse backends. Implementations must be read-only."""

    def list_schemas(self) -> list[str]: ...

    def list_tables(self, schema: str) -> list[TableInfo]: ...

    def describe_table(self, schema: str, table: str) -> list[ColumnInfo]: ...

    def execute_query(self, sql: str, max_rows: int = 100) -> QueryResult: ...

    def close(self) -> None: ...
