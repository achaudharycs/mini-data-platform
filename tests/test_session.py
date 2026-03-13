"""Tests for schema caching and conversation management."""

from __future__ import annotations

from typing import Any

import pytest

from agent.core.session import ConversationManager, SchemaCache
from agent.warehouse.base import ColumnInfo, QueryResult, TableInfo


class SpyWarehouse:
    """Warehouse that counts how many times each method is called."""

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {
            "list_schemas": 0,
            "list_tables": 0,
            "describe_table": 0,
            "execute_query": 0,
        }

    def list_schemas(self) -> list[str]:
        self.call_counts["list_schemas"] += 1
        return ["raw", "staging", "marts"]

    def list_tables(self, schema: str) -> list[TableInfo]:
        self.call_counts["list_tables"] += 1
        if schema == "marts":
            return [
                TableInfo(schema_name="marts", table_name="fct_orders", table_type="TABLE", row_count=100),
                TableInfo(schema_name="marts", table_name="dim_customers", table_type="VIEW", row_count=50),
            ]
        return []

    def describe_table(self, schema: str, table: str) -> list[ColumnInfo]:
        self.call_counts["describe_table"] += 1
        if table == "fct_orders":
            return [
                ColumnInfo(name="order_id", data_type="INTEGER", is_nullable=False),
                ColumnInfo(name="total", data_type="DOUBLE", is_nullable=True),
            ]
        return []

    def execute_query(self, sql: str, max_rows: int = 100) -> QueryResult:
        self.call_counts["execute_query"] += 1
        return QueryResult(columns=["x"], rows=[[1]], row_count=1)

    def close(self) -> None:
        pass


class TestSchemaCache:
    def test_list_schemas_cached(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        result1 = cache.list_schemas()
        result2 = cache.list_schemas()
        assert result1 == result2 == ["raw", "staging", "marts"]
        assert spy.call_counts["list_schemas"] == 1

    def test_list_tables_cached_per_schema(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        cache.list_tables("marts")
        cache.list_tables("marts")
        cache.list_tables("raw")
        assert spy.call_counts["list_tables"] == 2  # marts once, raw once

    def test_describe_table_cached(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        cache.describe_table("marts", "fct_orders")
        cache.describe_table("marts", "fct_orders")
        assert spy.call_counts["describe_table"] == 1

    def test_execute_query_not_cached(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        cache.execute_query("SELECT 1")
        cache.execute_query("SELECT 1")
        assert spy.call_counts["execute_query"] == 2

    def test_has_cached_schema(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        assert not cache.has_cached_schema
        cache.list_schemas()
        assert cache.has_cached_schema

    def test_build_schema_summary_empty(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        assert cache.build_schema_summary() == ""

    def test_build_schema_summary_with_data(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        cache.list_schemas()
        cache.list_tables("marts")
        cache.describe_table("marts", "fct_orders")

        summary = cache.build_schema_summary()
        assert "raw, staging, marts" in summary
        assert "fct_orders" in summary
        assert "dim_customers" in summary
        assert "order_id" in summary
        assert "total" in summary

    def test_build_schema_summary_tables_without_columns(self):
        spy = SpyWarehouse()
        cache = SchemaCache(spy)
        cache.list_schemas()
        cache.list_tables("marts")

        summary = cache.build_schema_summary()
        assert "fct_orders" in summary
        # Columns shouldn't appear since describe_table wasn't called
        assert "order_id" not in summary


class TestConversationManager:
    def test_empty_history(self):
        conv = ConversationManager()
        assert conv.turn_count == 0
        assert conv.build_history_messages() == []
        assert conv.build_context_block() == ""

    def test_add_turns_within_window(self):
        conv = ConversationManager(window_size=3)
        conv.add_turn("How many orders?", "There are 100 orders.")
        conv.add_turn("Break down by month", "January: 30, February: 70.")

        assert conv.turn_count == 2
        messages = conv.build_history_messages()
        assert len(messages) == 4  # 2 turns * 2 messages each
        assert messages[0] == {"role": "user", "content": "How many orders?"}
        assert messages[1] == {"role": "assistant", "content": "There are 100 orders."}
        assert messages[2] == {"role": "user", "content": "Break down by month"}
        assert messages[3] == {"role": "assistant", "content": "January: 30, February: 70."}

    def test_summary_created_beyond_window(self):
        conv = ConversationManager(window_size=2)
        conv.add_turn("Q1", "A1")
        conv.add_turn("Q2", "A2")
        conv.add_turn("Q3", "A3")

        # Q1 should be in summary, Q2+Q3 in window
        assert conv.turn_count == 2
        assert "Q1" in conv.summary
        assert "A1" in conv.summary

        messages = conv.build_history_messages()
        assert len(messages) == 4  # Q2+A2, Q3+A3
        assert messages[0]["content"] == "Q2"

    def test_summary_accumulates(self):
        conv = ConversationManager(window_size=1)
        conv.add_turn("Q1", "A1")
        conv.add_turn("Q2", "A2")
        conv.add_turn("Q3", "A3")

        # Q1 and Q2 should be in summary
        assert "Q1" in conv.summary
        assert "Q2" in conv.summary
        # Only Q3 in window
        messages = conv.build_history_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Q3"

    def test_context_block_includes_summary(self):
        conv = ConversationManager(window_size=1)
        conv.add_turn("How many users?", "5000 users.")
        conv.add_turn("Top states?", "Texas, Colorado.")

        block = conv.build_context_block()
        assert "How many users?" in block
        assert "5000 users" in block

    def test_context_block_empty_when_no_summary(self):
        conv = ConversationManager(window_size=3)
        conv.add_turn("Q1", "A1")
        # Still within window, no summary yet
        assert conv.build_context_block() == ""

    def test_long_answers_truncated_in_summary(self):
        conv = ConversationManager(window_size=1)
        long_answer = "x" * 300
        conv.add_turn("Q1", long_answer)
        conv.add_turn("Q2", "A2")

        # The summary should contain a truncated version
        assert len(conv.summary) < 300
        assert "..." in conv.summary


class TestSchemaCacheWithRealWarehouse:
    """Integration tests using the in-memory DuckDB fixture."""

    def test_cached_introspection(self, warehouse):
        cache = SchemaCache(warehouse)
        schemas1 = cache.list_schemas()
        schemas2 = cache.list_schemas()
        assert schemas1 == schemas2

        tables1 = cache.list_tables("marts")
        tables2 = cache.list_tables("marts")
        assert tables1 == tables2

    def test_schema_summary_with_real_data(self, warehouse):
        cache = SchemaCache(warehouse)
        cache.list_schemas()
        cache.list_tables("marts")
        cache.describe_table("marts", "fct_orders")

        summary = cache.build_schema_summary()
        assert "marts" in summary
        assert "fct_orders" in summary
        assert "transaction_id" in summary
