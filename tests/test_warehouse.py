"""Tests for warehouse introspection and query execution."""

from __future__ import annotations

import pytest


class TestListSchemas:
    def test_returns_expected_schemas(self, warehouse):
        schemas = warehouse.list_schemas()
        assert "raw" in schemas
        assert "marts" in schemas
        assert "staging" in schemas

    def test_excludes_system_schemas(self, warehouse):
        schemas = warehouse.list_schemas()
        assert "information_schema" not in schemas
        assert "pg_catalog" not in schemas


class TestListTables:
    def test_marts_tables(self, warehouse):
        tables = warehouse.list_tables("marts")
        names = {t.table_name for t in tables}
        assert "fct_orders" in names
        assert "dim_customers" in names
        assert "dim_products" in names

    def test_row_counts(self, warehouse):
        tables = warehouse.list_tables("marts")
        fct = next(t for t in tables if t.table_name == "fct_orders")
        assert fct.row_count == 8

    def test_table_types(self, warehouse):
        tables = warehouse.list_tables("marts")
        fct = next(t for t in tables if t.table_name == "fct_orders")
        assert fct.table_type in ("BASE TABLE", "TABLE", "LOCAL TEMPORARY")
        dim = next(t for t in tables if t.table_name == "dim_customers")
        assert dim.table_type == "VIEW"

    def test_empty_schema(self, warehouse):
        # main schema should exist but may be empty in our test setup
        # Just verify it doesn't error
        warehouse.list_tables("main")

    def test_nonexistent_schema(self, warehouse):
        tables = warehouse.list_tables("does_not_exist")
        assert tables == []


class TestDescribeTable:
    def test_column_names(self, warehouse):
        cols = warehouse.describe_table("raw", "products")
        col_names = [c.name for c in cols]
        assert "product_id" in col_names
        assert "product_name" in col_names
        assert "price" in col_names

    def test_column_types(self, warehouse):
        cols = warehouse.describe_table("raw", "products")
        price_col = next(c for c in cols if c.name == "price")
        assert "DOUBLE" in price_col.data_type.upper() or "FLOAT" in price_col.data_type.upper()

    def test_sample_values(self, warehouse):
        cols = warehouse.describe_table("raw", "products")
        name_col = next(c for c in cols if c.name == "product_name")
        assert len(name_col.sample_values) > 0
        assert "Widget A" in name_col.sample_values

    def test_nullable_detection(self, warehouse):
        cols = warehouse.describe_table("raw", "products")
        brand_col = next(c for c in cols if c.name == "brand")
        # brand has NULL values in test data, column should be nullable
        assert brand_col.is_nullable

    def test_nonexistent_table(self, warehouse):
        cols = warehouse.describe_table("raw", "nonexistent")
        assert cols == []


class TestExecuteQuery:
    def test_simple_select(self, warehouse):
        result = warehouse.execute_query("SELECT * FROM raw.products")
        assert result.row_count == 5
        assert "product_id" in result.columns

    def test_aggregation(self, warehouse):
        result = warehouse.execute_query(
            "SELECT category, COUNT(*) as cnt FROM raw.products GROUP BY category ORDER BY cnt DESC"
        )
        assert result.row_count > 0
        assert "category" in result.columns

    def test_truncation(self, warehouse):
        result = warehouse.execute_query("SELECT * FROM marts.fct_orders", max_rows=3)
        assert result.row_count == 3
        assert result.truncated is True

    def test_no_truncation(self, warehouse):
        result = warehouse.execute_query("SELECT * FROM raw.products", max_rows=100)
        assert result.truncated is False

    def test_empty_result(self, warehouse):
        result = warehouse.execute_query(
            "SELECT * FROM raw.products WHERE product_id = -1"
        )
        assert result.row_count == 0
        assert result.rows == []

    def test_invalid_sql_raises(self, warehouse):
        with pytest.raises(Exception):
            warehouse.execute_query("SELECT * FROM nonexistent_table")

    def test_join_query(self, warehouse):
        result = warehouse.execute_query("""
            SELECT p.product_name, SUM(o.total) as revenue
            FROM marts.fct_orders o
            JOIN raw.products p ON o.product_id = p.product_id
            GROUP BY p.product_name
            ORDER BY revenue DESC
        """)
        assert result.row_count > 0
        assert result.columns == ["product_name", "revenue"]
