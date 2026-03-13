"""Tests for tool dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.context.discovery import ProjectContext
from agent.core.tools import TOOL_DEFINITIONS, dispatch_tool


@pytest.fixture()
def context(project_root) -> ProjectContext:
    return ProjectContext(project_root)


class TestToolDefinitions:
    def test_all_tools_have_required_fields(self):
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_expected_tools_exist(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == {
            "list_schemas",
            "list_tables",
            "describe_table",
            "run_query",
            "get_dbt_model",
            "get_dag_info",
        }


class TestDispatchListSchemas:
    def test_returns_schemas(self, warehouse, context):
        result = dispatch_tool("list_schemas", {}, warehouse, context)
        assert "raw" in result
        assert "marts" in result


class TestDispatchListTables:
    def test_returns_tables(self, warehouse, context):
        result = dispatch_tool("list_tables", {"schema": "marts"}, warehouse, context)
        assert "fct_orders" in result
        assert "dim_customers" in result

    def test_empty_schema(self, warehouse, context):
        result = dispatch_tool("list_tables", {"schema": "nonexistent"}, warehouse, context)
        assert "No tables found" in result


class TestDispatchDescribeTable:
    def test_returns_columns(self, warehouse, context):
        result = dispatch_tool(
            "describe_table", {"schema": "raw", "table": "products"}, warehouse, context
        )
        assert "product_id" in result
        assert "product_name" in result

    def test_nonexistent_table(self, warehouse, context):
        result = dispatch_tool(
            "describe_table", {"schema": "raw", "table": "nope"}, warehouse, context
        )
        assert "No columns found" in result


class TestDispatchRunQuery:
    def test_returns_formatted_results(self, warehouse, context):
        result = dispatch_tool(
            "run_query",
            {"sql": "SELECT product_name, price FROM raw.products ORDER BY price DESC LIMIT 2"},
            warehouse,
            context,
        )
        assert "product_name" in result
        assert "price" in result
        assert "2 row(s)" in result

    def test_empty_result(self, warehouse, context):
        result = dispatch_tool(
            "run_query",
            {"sql": "SELECT * FROM raw.products WHERE product_id = -1"},
            warehouse,
            context,
        )
        assert "0 rows" in result

    def test_sql_error_propagates(self, warehouse, context):
        with pytest.raises(Exception):
            dispatch_tool(
                "run_query",
                {"sql": "SELECT * FROM fake_table"},
                warehouse,
                context,
            )


class TestDispatchDbtModel:
    def test_existing_model(self, warehouse, context):
        result = dispatch_tool("get_dbt_model", {"name": "fct_orders"}, warehouse, context)
        assert "transaction_id" in result
        assert "ref(" in result

    def test_nonexistent_model(self, warehouse, context):
        result = dispatch_tool("get_dbt_model", {"name": "nonexistent_model"}, warehouse, context)
        assert "not found" in result
        assert "Available models" in result


class TestDispatchDagInfo:
    def test_existing_dag(self, warehouse, context):
        result = dispatch_tool("get_dag_info", {"name": "run_dbt"}, warehouse, context)
        assert "dbt" in result.lower()

    def test_nonexistent_dag(self, warehouse, context):
        result = dispatch_tool("get_dag_info", {"name": "nonexistent_dag"}, warehouse, context)
        assert "not found" in result
        assert "Available DAGs" in result


class TestUnknownTool:
    def test_unknown_tool(self, warehouse, context):
        result = dispatch_tool("fake_tool", {}, warehouse, context)
        assert "Unknown tool" in result
