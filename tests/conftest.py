"""Shared test fixtures — in-memory DuckDB with test data."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import duckdb
import pytest

from agent.warehouse.duckdb import DuckDBWarehouse


@pytest.fixture()
def raw_conn() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """In-memory DuckDB connection with test schemas and data."""
    conn = duckdb.connect(":memory:")

    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    conn.execute("CREATE SCHEMA IF NOT EXISTS staging")
    conn.execute("CREATE SCHEMA IF NOT EXISTS marts")

    # Raw products
    conn.execute("""
        CREATE TABLE raw.products (
            product_id INTEGER,
            product_name VARCHAR,
            brand VARCHAR,
            category VARCHAR,
            price DOUBLE,
            cost DOUBLE
        )
    """)
    conn.execute("""
        INSERT INTO raw.products VALUES
        (1, 'Widget A', 'Acme', 'Widgets', 29.99, 10.00),
        (2, 'Widget B', 'Acme', 'Widgets', 49.99, 20.00),
        (3, 'Gadget X', 'TechCo', 'Gadgets', 99.99, 40.00),
        (4, 'Gadget Y', 'TechCo', 'Gadgets', 149.99, 60.00),
        (5, 'Doohickey', NULL, 'Misc', 9.99, 5.00)
    """)

    # Raw users
    conn.execute("""
        CREATE TABLE raw.users (
            user_id INTEGER,
            email VARCHAR,
            first_name VARCHAR,
            last_name VARCHAR,
            city VARCHAR,
            state VARCHAR,
            age INTEGER,
            customer_segment VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO raw.users VALUES
        (1, 'alice@test.com', 'Alice', 'Smith', 'Denver', 'CO', 30, 'Premium'),
        (2, 'bob@test.com', 'Bob', 'Jones', 'Austin', 'TX', 25, 'Standard'),
        (3, 'carol@test.com', 'Carol', 'Lee', 'Denver', 'CO', 45, 'Premium'),
        (4, 'dave@test.com', 'Dave', 'Kim', 'Austin', 'TX', 35, 'Standard'),
        (5, 'eve@test.com', 'Eve', 'Park', NULL, NULL, NULL, 'Standard')
    """)

    # Marts fact table (denormalized)
    conn.execute("""
        CREATE TABLE marts.fct_orders (
            transaction_id INTEGER,
            transaction_date DATE,
            user_id INTEGER,
            customer_email VARCHAR,
            customer_first_name VARCHAR,
            customer_state VARCHAR,
            customer_segment VARCHAR,
            product_id INTEGER,
            product_name VARCHAR,
            brand VARCHAR,
            category VARCHAR,
            quantity INTEGER,
            unit_price DOUBLE,
            total DOUBLE,
            discount DOUBLE,
            is_discounted BOOLEAN
        )
    """)
    conn.execute("""
        INSERT INTO marts.fct_orders VALUES
        (1, '2024-01-15', 1, 'alice@test.com', 'Alice', 'CO', 'Premium', 1, 'Widget A', 'Acme', 'Widgets', 2, 29.99, 59.98, 0, false),
        (2, '2024-01-16', 2, 'bob@test.com', 'Bob', 'TX', 'Standard', 3, 'Gadget X', 'TechCo', 'Gadgets', 1, 99.99, 99.99, 5.00, true),
        (3, '2024-02-01', 1, 'alice@test.com', 'Alice', 'CO', 'Premium', 2, 'Widget B', 'Acme', 'Widgets', 3, 49.99, 149.97, 0, false),
        (4, '2024-02-10', 3, 'carol@test.com', 'Carol', 'CO', 'Premium', 4, 'Gadget Y', 'TechCo', 'Gadgets', 1, 149.99, 149.99, 10.00, true),
        (5, '2024-03-01', 4, 'dave@test.com', 'Dave', 'TX', 'Standard', 1, 'Widget A', 'Acme', 'Widgets', 5, 29.99, 149.95, 0, false),
        (6, '2024-03-15', 2, 'bob@test.com', 'Bob', 'TX', 'Standard', 5, 'Doohickey', NULL, 'Misc', 10, 9.99, 99.90, 0, false),
        (7, '2024-03-20', 5, 'eve@test.com', 'Eve', NULL, 'Standard', 3, 'Gadget X', 'TechCo', 'Gadgets', 2, 99.99, 199.98, 15.00, true),
        (8, '2024-04-01', 1, 'alice@test.com', 'Alice', 'CO', 'Premium', 1, 'Widget A', 'Acme', 'Widgets', 1, 29.99, 29.99, 0, false)
    """)

    # Marts dim_customers
    conn.execute("""
        CREATE VIEW marts.dim_customers AS
        SELECT * FROM raw.users
    """)

    # Marts dim_products
    conn.execute("""
        CREATE VIEW marts.dim_products AS
        SELECT *, price - cost AS margin FROM raw.products
    """)

    # Staging view
    conn.execute("""
        CREATE VIEW staging.stg_products AS
        SELECT * FROM raw.products
    """)

    yield conn
    conn.close()


class InMemoryDuckDBWarehouse:
    """Warehouse backed by an existing in-memory DuckDB connection (for tests)."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

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

    def list_tables(self, schema: str):
        from agent.warehouse.base import TableInfo

        tables = self._conn.execute(
            """
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = ?
            ORDER BY table_name
            """,
            [schema],
        ).fetchall()

        result = []
        for table_name, table_type in tables:
            try:
                count_row = self._conn.execute(
                    f'SELECT COUNT(*) FROM "{schema}"."{table_name}"'
                ).fetchone()
                row_count = count_row[0] if count_row else None
            except Exception:
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

    def describe_table(self, schema: str, table: str):
        from agent.warehouse.base import ColumnInfo

        columns = self._conn.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()

        result = []
        for col_name, data_type, is_nullable in columns:
            try:
                samples = self._conn.execute(
                    f'SELECT DISTINCT "{col_name}" FROM "{schema}"."{table}" WHERE "{col_name}" IS NOT NULL LIMIT 5'
                ).fetchall()
                sample_values = [str(s[0]) for s in samples]
            except Exception:
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

    def execute_query(self, sql: str, max_rows: int = 100):
        from agent.warehouse.base import QueryResult

        rel = self._conn.execute(sql)
        description = rel.description
        columns = [col[0] for col in description] if description else []
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
        pass  # Don't close the shared connection


@pytest.fixture()
def warehouse(raw_conn) -> InMemoryDuckDBWarehouse:
    """In-memory warehouse backed by the test DuckDB connection."""
    return InMemoryDuckDBWarehouse(raw_conn)


@pytest.fixture()
def project_root() -> Path:
    """Return the actual project root for context discovery tests."""
    return Path(__file__).parent.parent
