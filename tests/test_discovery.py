"""Tests for project context discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.context.discovery import ProjectContext


@pytest.fixture()
def context(project_root) -> ProjectContext:
    return ProjectContext(project_root)


class TestListDbtModels:
    def test_finds_staging_models(self, context):
        models = context.list_dbt_models()
        assert "stg_products" in models
        assert "stg_users" in models
        assert "stg_transactions" in models

    def test_finds_marts_models(self, context):
        models = context.list_dbt_models()
        assert "fct_orders" in models
        assert "dim_customers" in models
        assert "dim_products" in models

    def test_excludes_underscore_files(self, context):
        models = context.list_dbt_models()
        assert "_sources" not in models

    def test_sorted(self, context):
        models = context.list_dbt_models()
        assert models == sorted(models)


class TestGetDbtModel:
    def test_existing_model(self, context):
        content = context.get_dbt_model("fct_orders")
        assert "transaction_id" in content
        assert "ref(" in content

    def test_nonexistent_model(self, context):
        result = context.get_dbt_model("nonexistent")
        assert "not found" in result
        assert "Available models" in result

    def test_staging_model(self, context):
        content = context.get_dbt_model("stg_products")
        assert "source(" in content or "ref(" in content or "product_id" in content


class TestListDags:
    def test_finds_dags(self, context):
        dags = context.list_dags()
        assert "ingest_products" in dags
        assert "run_dbt" in dags

    def test_excludes_init(self, context):
        dags = context.list_dags()
        assert "__init__" not in dags

    def test_sorted(self, context):
        dags = context.list_dags()
        assert dags == sorted(dags)


class TestGetDagInfo:
    def test_existing_dag(self, context):
        content = context.get_dag_info("run_dbt")
        assert "dbt" in content.lower()

    def test_nonexistent_dag(self, context):
        result = context.get_dag_info("nonexistent")
        assert "not found" in result
        assert "Available DAGs" in result


class TestNonexistentProject:
    def test_no_dbt_models(self):
        ctx = ProjectContext("/tmp/nonexistent_project_12345")
        assert ctx.list_dbt_models() == []

    def test_no_dags(self):
        ctx = ProjectContext("/tmp/nonexistent_project_12345")
        assert ctx.list_dags() == []
