# Mini Data Platform

A synthetic data platform with a CLI agent that answers ad-hoc data questions by autonomously exploring and querying a DuckDB warehouse.

## CLI Agent

### How It Works

The agent uses Claude's tool-use API to autonomously explore the warehouse and answer natural language questions. Rather than hardcoding knowledge about this specific data platform, the agent discovers everything at runtime:

1. **Schema introspection** — On the first question, the agent calls `list_schemas`, `list_tables`, and `describe_table` to discover what data exists. These results are cached in a `SchemaCache` and injected into subsequent system prompts, so follow-up questions skip introspection entirely and go straight to SQL.
2. **SQL generation** — Based on what it discovers, the agent writes and executes SQL queries. If a query fails, it reads the error and retries with corrections (up to 3 attempts). Query results are displayed inline as Rich tables.
3. **Conversation memory** — A `ConversationManager` keeps the last 3 Q&A turns in full and compresses older turns into a running summary. This lets the agent handle follow-ups like "break that down by month" or "now show me just Texas" without re-explaining context.
4. **Project context** — The agent can also read dbt model SQL and Airflow DAG source code to understand how tables are built, what joins and business logic exist, and how data flows through the pipeline.

This means the same agent works on any data platform — swap the DuckDB file and it adapts. The system prompt contains no hardcoded schema knowledge; everything is discovered through tools and cached for the session.

### Architecture

```
User Question
     │
     ▼
┌─────────────┐
│ Session      │
│ Context      │
│              │
│ SchemaCache ─┤── Memoized introspection (injected into system prompt)
│ Conversation │── Sliding window + summary (recent turns + compressed history)
└──────┬──────┘
       │
       ▼
┌─────────┐     ┌──────────────┐     ┌───────────┐
│  Agent   │────▶│  LLM (Claude)│────▶│   Tools   │
│  Loop    │◀────│  tool-use API│     │           │
└─────────┘     └──────────────┘     │ list_schemas    │
                                      │ list_tables     │
                                      │ describe_table  │
                                      │ run_query       │
                                      │ get_dbt_model   │
                                      │ get_dag_info    │
                                      └───────────┘
                                            │
                                      ┌─────┴─────┐
                                      │ Warehouse  │  ◀── Protocol (pluggable)
                                      │ (DuckDB)   │
                                      └───────────┘
```

**Key design decisions:**

- **Raw Anthropic SDK with tool-use** - Instead of a framework like LangChain — less abstraction, more control over the agent loop, easier to debug and test.
- **`Warehouse` Protocol** — A Python Protocol that any backend can implement (`list_schemas`, `list_tables`, `describe_table`, `execute_query`). The DuckDB implementation is one concrete backend; swapping to Snowflake or Postgres means writing a new class with the same interface.
- **Session context with caching** — `SchemaCache` memoizes introspection at the warehouse level and injects a schema summary into the system prompt. Follow-up questions skip 2-3 tool calls entirely. At scale, this cache could be shared across users on the same warehouse (e.g. via Redis with TTL).
- **Bounded conversation history** — `ConversationManager` uses a sliding window (last 3 turns in full) plus a compressed summary for older turns. This keeps token costs bounded (~3-4k tokens regardless of session length) while enabling follow-up questions. At scale, summarization could be offloaded to a cheaper model.
- **Inline result tables** — Query results are rendered as Rich tables directly in the terminal as they execute, with titles derived from the SQL source table. The LLM's text response provides a brief summary rather than repeating the data.
- **Multi-table pagination** — When queries return more than 50 rows, users can page through results with `more`. Multiple query results are numbered (`#1 — fct_orders`, `#2 — dim_customers`) and can be paged independently with `more #N`. Remaining row counts are shown after each page.
- **Read-only connection** — The DuckDB connection is opened in read-only mode. The agent can only SELECT, never modify data.
- **No hardcoded schema knowledge** — The system prompt is fully generic — it tells the agent to prefer schemas with fewer, wider tables (a heuristic for analytics-ready data) but knows nothing about specific schemas, tables, or columns. Everything is discovered through tools.
- **15-iteration cap** — Prevents runaway tool-call loops. In practice, most questions resolve in 3-6 iterations. Follow-up questions with cached schema often resolve in 1-2.

### Running the Agent

```bash
# 1. Set up the data platform (if not already done)
./setup.sh

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-...

# 3. Start the agent
uv run agent

# Options:
uv run agent --help        # Show all available flags
uv run agent --show-sql    # Display SQL queries the agent executes
uv run agent --verbose     # Show each tool call as it happens
```

### REPL Commands

| Command | Description |
|---|---|
| `help` | Show available commands |
| `log` | Show the agent's activity log from the last question |
| `more` | Page through results when a query returns many rows |
| `more #N` | Page through a specific result table by number |
| `quit` | Exit the agent |

Arrow keys (up/down) cycle through previous inputs.

### Configuration

| Env Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `WAREHOUSE_PATH` | `warehouse/data.duckdb` | Path to the DuckDB database file |
| `AGENT_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |

### Example Questions

- "How much in sales did we do last quarter?"
- "Which two products are most frequently bought together?"
- "What's our average customer lifetime value?"
- "Are there any anomalies with how we sell products?"
- "What are the top 5 states by revenue, and what's the average order size in each?"
- "How does the Premium customer segment compare to Standard in terms of repeat purchases?"
- "Show me all orders from Q1 2024, all customers who signed up in 2024, and all products with their profit margins" (multi-table output)
- "Break that down by month" (follow-up using conversation history)

### Testing

All tests run against an in-memory DuckDB instance with no API key required:

```bash
uv run pytest tests/ -v    # 80 tests
```

### Where I'd Go Next

**Query cost guardrails** — Before executing a query, run `EXPLAIN` and check estimated row scans. Warn the user or refuse if a query would scan an unreasonable amount of data. More important on real warehouses (Snowflake, BigQuery) where queries have dollar costs.

**LLM-powered conversation summarization** — Currently older turns are compressed with simple truncation. Using a fast/cheap model (e.g. Haiku) to generate semantic summaries would preserve more context with fewer tokens. At 10M users, this could run async after each turn.

**Shared schema cache** — The current `SchemaCache` is per-session. For multi-user deployments against the same warehouse, a shared cache (Redis/Memcached with TTL) would eliminate redundant introspection across all users.

**More warehouse backends** — The `Warehouse` Protocol makes this straightforward. A Postgres or Snowflake implementation would mostly differ in connection setup and `information_schema` dialect quirks.

**Semantic layer integration** — For platforms with a dbt semantic layer or metrics definitions, the agent could use those to write more business-correct queries instead of raw SQL against tables.

---

## Quick Setup

Run the setup script to initialize everything:

```bash
./setup.sh
```

This will:
1. Generate synthetic data
2. Initialize Airflow and load data into DuckDB
3. Run dbt transformations

Then view the dashboards:

```bash
cd evidence
npm install       # First time only
npm run sources   # Build data sources
npm run dev       # Start dev server
# Open http://localhost:3000
```

---

## Manual Setup (Advanced)

<details>
<summary>Click to expand manual setup steps</summary>

### 1. Install dependencies

```bash
uv sync
```

### 2. Generate synthetic data

```bash
uv run python scripts/generate_all.py
```

### 3. Initialize Airflow

First, update `airflow/airflow.cfg` to use an absolute path for the database:

```bash
cd airflow
# Update sql_alchemy_conn in airflow.cfg to:
# sql_alchemy_conn = sqlite:////absolute/path/to/your/mini-data-platform/airflow/airflow.db

export AIRFLOW_HOME=$(pwd)
uv run airflow db migrate
```

### 4. Run ingestion DAGs

```bash
# From airflow/ directory
export AIRFLOW_HOME=$(pwd)
uv run python dags/ingest_products.py
uv run python dags/ingest_users.py
uv run python dags/ingest_transactions.py
uv run python dags/ingest_campaigns.py
uv run python dags/ingest_pageviews.py
```

### 5. Run dbt transformations

```bash
# From airflow/ directory
export AIRFLOW_HOME=$(pwd)
uv run python dags/run_dbt.py

# Or run dbt directly
cd ../dbt_project
uv run dbt build --profiles-dir .
```

</details>

## Project Structure

```sh
mini-data-platform/
├── agent/                # CLI agent
│   ├── cli.py           # Typer REPL entry point
│   ├── config.py        # Config from env vars
│   ├── core/
│   │   ├── agent.py     # Agent loop (messages → tool calls → answer)
│   │   ├── llm.py       # LLM client protocol + Anthropic implementation
│   │   ├── session.py   # SchemaCache + ConversationManager
│   │   └── tools.py     # Tool definitions + dispatch
│   ├── warehouse/
│   │   ├── base.py      # Warehouse protocol + dataclasses
│   │   └── duckdb.py    # DuckDB implementation
│   ├── context/
│   │   └── discovery.py # dbt model + DAG file discovery
│   └── display/
│       └── formatter.py # Rich tables, SQL syntax, pagination
├── tests/               # 80 tests (no API key needed)
├── sources/              # Raw source data (CSV files)
│   ├── postgres/         # Sales, products, users
│   ├── salesforce/       # Marketing campaigns
│   └── analytics/        # Page view events
├── airflow/
│   ├── dags/            # Airflow DAGs for ingestion and transformation
│   │   ├── ingest_*.py  # Load data from sources → raw schema
│   │   ├── run_dbt.py   # Run dbt staging → marts pipeline
│   │   └── build_evidence.py  # Build Evidence dashboards
│   └── utils/           # Shared utilities
├── warehouse/           # DuckDB database (data.duckdb)
├── dbt_project/         # dbt transformations
│   └── models/
│       ├── staging/     # Clean raw data (5 models)
│       └── marts/       # Analytics-ready tables (3 models)
├── evidence/            # Evidence BI dashboards
│   ├── pages/           # Dashboard pages (index, sales, products, customers)
│   └── sources/         # SQL queries and connection
└── scripts/             # Data generation scripts
```

## Data Pipeline

### Raw Layer (`raw` schema)

- Loaded by Airflow ingestion DAGs
- 5 tables: products, users, transactions, campaigns, pageviews

### Staging Layer (`staging` schema)

- Created by dbt
- 5 views: stg_products, stg_users, stg_transactions, stg_campaigns, stg_pageviews

### Marts Layer (`marts` schema)

- Created by dbt
- Denormalized tables for analysis
- 3 tables:
  - `dim_products`: Current product catalog (62 products)
  - `dim_customers`: Current customer info (5,000 customers)
  - `fct_orders`: Order line items with dimensions (35,980 rows)

## Data Volumes

- **Raw**: ~93K total rows across 5 tables
- **Staging**: Same as raw (views)
- **Marts**: 5,062 dimension rows + 35,980 fact rows
- **Database Size**: ~5-10 MB (DuckDB)

## Evidence Dashboards

The project includes interactive dashboards built with Evidence:

### Available Dashboards

1. **Overview** (`/`) - Key metrics, revenue trends, category performance
2. **Sales** (`/sales`) - Daily/monthly sales, country analysis, recent orders
3. **Products** (`/products`) - Product performance, category trends, price analysis
4. **Customers** (`/customers`) - Customer segments, lifetime value, acquisition trends

### Running Evidence

```bash
cd evidence
npm install       # First time only
npm run sources   # Build data sources
npm run dev       # Start dev server
```

Then open http://localhost:3000 to view dashboards.

**Note**: Evidence connects to the DuckDB warehouse at `../warehouse/data.duckdb` and queries the `marts` schema through pass-through SQL files (`fct_orders.sql`, `dim_customers.sql`, `dim_products.sql`).

### Building Evidence (Static Site)

```bash
# Using Airflow DAG
cd airflow
uv run python dags/build_evidence.py

# Or build directly
cd evidence
npm run build
```
