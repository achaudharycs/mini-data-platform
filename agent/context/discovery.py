"""Project context discovery — scan for dbt models and Airflow DAGs."""

from __future__ import annotations

from pathlib import Path


class ProjectContext:
    """Provides read access to dbt models and Airflow DAGs in the project."""

    def __init__(self, project_root: str | Path) -> None:
        self._root = Path(project_root)
        self._dbt_dir = self._root / "dbt_project" / "models"
        self._dag_dir = self._root / "airflow" / "dags"

    def list_dbt_models(self) -> list[str]:
        """Return names of all dbt model SQL files found in the project."""
        if not self._dbt_dir.exists():
            return []
        return sorted(
            p.stem for p in self._dbt_dir.rglob("*.sql") if not p.stem.startswith("_")
        )

    def get_dbt_model(self, name: str) -> str:
        """Read the SQL source of a dbt model by name."""
        matches = list(self._dbt_dir.rglob(f"{name}.sql"))
        if not matches:
            available = self.list_dbt_models()
            return f"Model '{name}' not found. Available models: {', '.join(available)}"
        return matches[0].read_text()

    def list_dags(self) -> list[str]:
        """Return names of all Airflow DAG Python files."""
        if not self._dag_dir.exists():
            return []
        return sorted(
            p.stem
            for p in self._dag_dir.glob("*.py")
            if not p.stem.startswith("_") and p.stem != "__init__"
        )

    def get_dag_info(self, name: str) -> str:
        """Read the source of an Airflow DAG file by name."""
        # Accept with or without .py suffix
        dag_file = self._dag_dir / f"{name}.py"
        if not dag_file.exists():
            available = self.list_dags()
            return f"DAG '{name}' not found. Available DAGs: {', '.join(available)}"
        return dag_file.read_text()
