"""DuckDB implementation of the Warehouse protocol."""

from __future__ import annotations

from pathlib import Path

import duckdb

from agent.warehouse.base import ColumnInfo, QueryResult, TableInfo


class DuckDBWarehouse:
    """Read-only DuckDB warehouse backed by a local file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._conn = duckdb.connect(str(self._path), read_only=True)

    def list_schemas(self) -> list[str]:
        rows = self._conn.execute(
            """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
            ORDER BY schema_name
            """
        ).fetchall()
        return [r[0] for r in rows]

    def list_tables(self, schema: str) -> list[TableInfo]:
        tables = self._conn.execute(
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = ?
            ORDER BY table_name
            """,
            [schema],
        ).fetchall()

        result: list[TableInfo] = []
        for table_name, table_type in tables:
            try:
                count_row = self._conn.execute(
                    f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'
                ).fetchone()
                row_count = count_row[0] if count_row else None
            except duckdb.Error:
                row_count = None

            result.append(
                TableInfo(
                    schema_name=schema,
                    table_name=table_name,
                    table_type=table_type,
                    row_count=row_count,
                )
            )
        return result

    def describe_table(self, schema: str, table: str) -> list[ColumnInfo]:
        columns = self._conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()

        result: list[ColumnInfo] = []
        for col_name, data_type, is_nullable in columns:
            # Fetch up to 5 distinct non-null sample values
            try:
                samples = self._conn.execute(
                    f"""
                    SELECT DISTINCT "{col_name}"
                    FROM "{schema}"."{table}"
                    WHERE "{col_name}" IS NOT NULL
                    LIMIT 5
                    """
                ).fetchall()
                sample_values = [str(s[0]) for s in samples]
            except duckdb.Error:
                sample_values = []

            result.append(
                ColumnInfo(
                    name=col_name,
                    data_type=data_type,
                    is_nullable=is_nullable == "YES",
                    sample_values=sample_values,
                )
            )
        return result

    def execute_query(self, sql: str, max_rows: int = 100) -> QueryResult:
        rel = self._conn.execute(sql)
        description = rel.description
        columns = [col[0] for col in description] if description else []

        # Fetch max_rows + 1 to detect truncation
        rows_raw = rel.fetchmany(max_rows + 1)
        truncated = len(rows_raw) > max_rows
        rows = [list(r) for r in rows_raw[:max_rows]]

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    def close(self) -> None:
        self._conn.close()
