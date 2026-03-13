"""Tool definitions and dispatch for the agent."""

from __future__ import annotations

from typing import Any

from agent.context.discovery import ProjectContext
from agent.warehouse.base import Warehouse

# -- Tool JSON schemas for Claude tool-use format --

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_schemas",
        "description": (
            "List all schemas (namespaces) in the data warehouse. "
            "Call this first to understand what data layers are available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_tables",
        "description": (
            "List all tables and views in a specific schema, including row counts. "
            "Use this to discover what data is available before querying."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "The schema name to list tables from (e.g. 'marts', 'staging', 'raw').",
                }
            },
            "required": ["schema"],
        },
    },
    {
        "name": "describe_table",
        "description": (
            "Get column names, data types, and sample values for a table. "
            "Always call this before writing SQL to verify exact column names and types."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "schema": {
                    "type": "string",
                    "description": "The schema name.",
                },
                "table": {
                    "type": "string",
                    "description": "The table name.",
                },
            },
            "required": ["schema", "table"],
        },
    },
    {
        "name": "run_query",
        "description": (
            "Execute a read-only SQL query against the warehouse and return results. "
            "Returns at most 100 rows. Write aggregations in SQL rather than processing raw rows. "
            "If the query fails, read the error message and retry with corrections."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute. Must be read-only (SELECT only).",
                }
            },
            "required": ["sql"],
        },
    },
    {
        "name": "get_dbt_model",
        "description": (
            "Read the SQL source of a dbt model to understand how a table is built. "
            "Useful for understanding column derivations, joins, and business logic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The dbt model name (e.g. 'fct_orders', 'stg_transactions').",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_dag_info",
        "description": (
            "Read the source of an Airflow DAG file to understand data pipeline logic, "
            "scheduling, and data loading processes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The DAG filename (e.g. 'ingest_products', 'run_dbt').",
                }
            },
            "required": ["name"],
        },
    },
]


def dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    warehouse: Warehouse,
    context: ProjectContext,
) -> str:
    """Execute a tool call and return a string result for the LLM."""
    if tool_name == "list_schemas":
        schemas = warehouse.list_schemas()
        return f"Available schemas: {', '.join(schemas)}"

    if tool_name == "list_tables":
        tables = warehouse.list_tables(tool_input["schema"])
        if not tables:
            return f"No tables found in schema '{tool_input['schema']}'."
        lines = [f"Tables in '{tool_input['schema']}':\n"]
        for t in tables:
            count = f" ({t.row_count:,} rows)" if t.row_count is not None else ""
            lines.append(f"  - {t.table_name} [{t.table_type}]{count}")
        return "\n".join(lines)

    if tool_name == "describe_table":
        columns = warehouse.describe_table(tool_input["schema"], tool_input["table"])
        if not columns:
            return f"No columns found for '{tool_input['schema']}.{tool_input['table']}'. Check schema/table name."
        lines = [f"Columns in {tool_input['schema']}.{tool_input['table']}:\n"]
        for c in columns:
            nullable = "nullable" if c.is_nullable else "not null"
            samples = f" — samples: {', '.join(c.sample_values)}" if c.sample_values else ""
            lines.append(f"  - {c.name} ({c.data_type}, {nullable}){samples}")
        return "\n".join(lines)

    if tool_name == "run_query":
        result = warehouse.execute_query(tool_input["sql"])
        if not result.columns:
            return "Query returned no columns."
        if not result.rows:
            return f"Query returned 0 rows. Columns: {', '.join(result.columns)}"

        # Format as a text table
        lines = [" | ".join(result.columns)]
        lines.append("-+-".join("-" * max(len(c), 8) for c in result.columns))
        for row in result.rows:
            lines.append(" | ".join(_fmt(v) for v in row))

        if result.truncated:
            lines.append(f"\n... results truncated (showing {result.row_count} rows, more available)")
        else:
            lines.append(f"\n{result.row_count} row(s)")

        return "\n".join(lines)

    if tool_name == "get_dbt_model":
        return context.get_dbt_model(tool_input["name"])

    if tool_name == "get_dag_info":
        return context.get_dag_info(tool_input["name"])

    return f"Unknown tool: {tool_name}"


def _fmt(value: object) -> str:
    if value is None:
        return "NULL"
    return str(value)
