"""CLI smoke tests via typer CliRunner."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agent.cli import app

runner = CliRunner()


class TestCLIStartup:
    def test_missing_api_key(self):
        """CLI exits with error when ANTHROPIC_API_KEY is missing."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = runner.invoke(app, [], input="quit\n")
            assert result.exit_code != 0 or "ANTHROPIC_API_KEY" in result.output

    def test_missing_warehouse(self, tmp_path):
        """CLI exits with error when warehouse file doesn't exist."""
        with patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "test-key", "WAREHOUSE_PATH": str(tmp_path / "nonexistent.duckdb")},
            clear=False,
        ):
            result = runner.invoke(app, [], input="quit\n")
            assert result.exit_code != 0 or "not found" in result.output.lower() or "Warehouse" in result.output


class TestCLIHelp:
    def test_help_flag(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "data agent" in result.output.lower() or "warehouse" in result.output.lower()
