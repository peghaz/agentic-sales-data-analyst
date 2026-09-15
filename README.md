# Sales Data Analyst (DB Agent)

Sales Data Analyst is a Streamlit chat application for exploring PostgreSQL
sales data in plain language. It uses an OpenAI-compatible model to plan
read-only SQL queries, validates every query, and presents business findings
with KPI cards, charts, and downloadable tables.

The setup instructions below are for the developer or operator running the
application. Business users only need the Streamlit URL produced at startup.

## Quick start

This recommended path runs PostgreSQL locally with Docker and connects to an
existing OpenAI-compatible or vLLM model endpoint. Run every command from the
repository root, where `manage.py` and `gui.py` are located.

### 1. Check the prerequisites

- Docker with Docker Compose
- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv)
- A chat-completions model endpoint with function/tool calling enabled

### 2. Configure the environment

```bash
cp .env.example .env
```

Open `.env` and set the PostgreSQL container values, the matching database-agent
connection, the model name, and the model endpoint. A complete model endpoint
takes precedence over `LLM_HOST` and `LLM_PORT`:

```env
POSTGRES_DB=dbagent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=choose-a-password
POSTGRES_HOST=localhost
POSTGRES_PORT=5656

DB_AGENT_DATABASE_HOST=localhost
DB_AGENT_DATABASE_PORT=5656
DB_AGENT_DATABASE_NAME=dbagent
DB_AGENT_DATABASE_USER=postgres
DB_AGENT_DATABASE_PASSWORD=choose-a-password

LLM_MODEL_NAME=your-served-model-name
LLM_API_MODE=chat
LLM_API_BASE_URL=http://192.168.1.50:8000/v1
LLM_API_KEY=EMPTY
```

The two database passwords must match for this local setup. Use the real API
key instead of `EMPTY` when the model endpoint requires one. See
[`.env.example`](.env.example) for every available setting.

### 3. Install dependencies and start PostgreSQL

```bash
uv sync
docker compose up -d db
docker compose ps
```

Wait until `dbagent-db` reports `healthy`.

### 4. Initialize and populate the database

```bash
uv run manage.py migrate
uv run manage.py load_csv_data --path data
```

The bundled CSV files provide the sample sales data used by the showcase
questions. The loader is idempotent, so running it again will not duplicate
existing records.

### 5. Verify the model and agent

```bash
uv run manage.py llm_ask "Reply with: model ready"
uv run manage.py db_ask "How many active customers are in the system?"
```

The first command checks the configured model endpoint. The second checks the
complete schema-inspection, tool-calling, SQL-validation, and database flow.

### 6. Start the Streamlit chat

```bash
uv run streamlit run gui.py
```

Streamlit prints addresses similar to:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.1.20:8501
```

Open the local URL on the same computer. Share the network URL with business
users on the same reachable network, subject to the host firewall and network
policy. Keep the terminal running while the app is in use and press `Ctrl+C` to
stop it.

Restart Streamlit after changing `.env`. **Clear chat** starts a new
conversation and rebuilds the agent runtime, but it does not reload environment
variables already loaded by the running Python process.

## Readiness checklist

Before sharing the application, confirm that:

- `docker compose ps` reports PostgreSQL as healthy.
- `uv run manage.py migrate` completes without pending migration errors.
- `uv run manage.py load_csv_data --path data` completes successfully.
- `uv run manage.py llm_ask "Reply with: model ready"` returns a response.
- The basic `db_ask` command returns a customer count.
- `uv run streamlit run gui.py` prints a reachable URL.

## Model endpoint options

### Remote model endpoint

For a model running on another machine, use either a complete URL:

```env
LLM_API_BASE_URL=http://192.168.1.50:8000/v1
```

or a host and port:

```env
LLM_HOST=192.168.1.50
LLM_PORT=8000
```

`LLM_API_BASE_URL` takes precedence when both forms are set. Verify a vLLM
endpoint directly when troubleshooting connectivity:

```bash
curl http://192.168.1.50:8000/health
curl http://192.168.1.50:8000/v1/models
```

With a remote endpoint, start only the local database; there is no need to run
the Compose `llm` service.

### Local vLLM endpoint

Running vLLM locally requires an NVIDIA GPU supported by vLLM, the NVIDIA
Container Toolkit, enough GPU memory for the selected model, and an absolute
host path to the model files. Configure the model-related values in `.env`, then
start the service:

```env
LLM_HOST_MODEL_PATH=/absolute/path/to/your/model
LLM_CONTAINER_MODEL_PATH=/models/llm
LLM_MODEL_NAME=your-served-model-name
LLM_HOST=127.0.0.1
LLM_PORT=8000
LLM_GPU_DEVICE_ID=0
```

```bash
docker compose up -d llm
docker compose ps
docker compose logs -f llm
```

The database agent requires automatic tool choice and a parser compatible with
the selected model. A vLLM launch normally needs:

```text
--enable-auto-tool-choice
--tool-call-parser <parser-compatible-with-your-model>
```

Consult the vLLM documentation for the parser supported by the exact model and
vLLM version. The current `docker-compose.yml` does not append these two flags;
setting `LLM_ENABLE_AUTO_TOOL_CHOICE` or `LLM_TOOL_CALL_PARSER` in `.env` alone
does not change the Compose command. Until the Compose service is updated for
the selected model, treat it as a base vLLM launcher rather than a turnkey
tool-calling configuration.

## Using Sales Data Analyst

The chat answers in business language and shows the data behind each answer.
SQL, model metadata, and execution details remain collapsed under **How this
was calculated**.

Follow-up questions reuse the current conversation. For example, after a
monthly comparison, ask “Which shop drove the change?” or “Now show only GBP.”
Conversation context is held in memory for the current Streamlit session only;
old conversations are not saved.

Try these showcase questions:

- Give me an executive sales summary for the latest 12 months in the data: revenue by currency, orders, active customers, average order value, top shop, and top product.
- Compare monthly revenue and order volume by shop for the latest 12 months, including month-over-month change and keeping currencies separate.
- Rank the top 10 customers by lifetime spend, showing order count, average order value, last purchase date, and keeping currencies separate.
- Build a customer retention view by signup month: customers acquired and how many purchased again within 30, 60, and 90 days.
- Find high-value customers at risk: at least 5 paid or shipped orders, but no purchase in the 90 days before the latest order in the dataset.
- Which product categories deliver the highest estimated gross profit and margin percentage, using product cost and line-item sales and keeping currencies separate?
- Find the product pairs most frequently bought together, with pair count and combined sales by currency.
- Compare payment failure rates by payment method and shop, including attempts, failed payments, and failed amount by currency.
- Compare carrier performance by destination country: shipment count, average and 90th-percentile delivery time, return rate, and shipping cost by currency.
- Show cancellation and refund rates by shop and month, with affected order value by currency.

Monetary analyses keep currencies separate unless conversion data is available.

If an analysis fails, the chat shows a short error ID and a collapsed diagnostic.
The Streamlit terminal prints a Rich traceback with the same ID so an operator
can correlate the report. Tracebacks omit frame-local values. Keep terminal
logs access-controlled because errors returned by external services may include
their own details.

## Command-line usage

Use `llm_ask` for a plain model prompt:

```bash
uv run manage.py llm_ask "List three shop names."
uv run manage.py llm_ask --system "You are a data analyst assistant." "Summarize the top five customers."
```

Use `db_ask` for a grounded database question:

```bash
uv run manage.py db_ask "How many active customers are in the system?"
uv run manage.py db_ask "Top five products by total purchase amount this month."
```

`db_ask` refreshes the allowed PostgreSQL schema, asks the model to call the
read-only SQL tool, validates the query, applies row and timeout limits, and
returns the answer with an execution trace.

## Configuration reference

Important database-agent settings include:

- `DB_AGENT_DATABASE_URL`
- `DB_AGENT_DATABASE_HOST`, `DB_AGENT_DATABASE_PORT`, `DB_AGENT_DATABASE_NAME`
- `DB_AGENT_DATABASE_USER`, `DB_AGENT_DATABASE_PASSWORD`
- `DB_AGENT_ALLOWED_SCHEMAS`, `DB_AGENT_ALLOWED_TABLES`
- `DB_AGENT_MAX_ROWS`, `DB_AGENT_MAX_RESULT_CHARS`
- `DB_AGENT_STATEMENT_TIMEOUT_MS`
- `DB_AGENT_MAX_TOOL_CALLS` — maximum SQL executions per question
- `DB_AGENT_QUERY_RETRIES`
- `DB_AGENT_MODEL_MAX_TOKENS` — output budget per model turn; default `4096`
- `DB_AGENT_ENABLE_THINKING` — Qwen/vLLM thinking mode; default `true`

Important model-client settings include:

- `LLM_MODEL_NAME`
- `LLM_API_MODE` — must be `chat` for tool calling
- `LLM_API_BASE_URL`, or `LLM_HOST` and `LLM_PORT`
- `LLM_API_KEY`
- `LLM_REQUEST_TIMEOUT_SECONDS`

If a response reaches its output limit before returning a tool call, raise
`DB_AGENT_MODEL_MAX_TOKENS` to `8192`. For lower latency and more direct tool
calls, set `DB_AGENT_ENABLE_THINKING=false`. Restart Streamlit after changing
either value.

## Architecture and conversation behavior

The agent is implemented as a LangGraph workflow. It refreshes the allowed
schema, asks the model, validates and executes one read-only SQL call at a time,
and then produces a business-facing answer. An in-memory checkpointer provides
session-scoped follow-ups.

The Streamlit chat reuses one thread ID for follow-ups. The `db_ask` command is
single-turn. Programmatic callers can use:

```python
agent.ask(question, thread_id="session-id")
agent.ask(question, thread_id="session-id", force_query=True)
```

`force_query=True` requires a fresh database read when retrying an answer.

## Tests

The offline suite does not require PostgreSQL or the model server:

```bash
uv run manage.py test
```

Tests live in the root `tests/` package. The GUI integration tests drive the
real LangGraph while mocking the database and model boundaries.

## Sample data management

The loader reads the CSV files under `data/` and uses Rich output with progress
bars. Loading without `--truncate` is idempotent:

```bash
uv run manage.py load_csv_data --path data
```

Use `--truncate` only when you intentionally want a clean reload:

```bash
uv run manage.py load_csv_data --path data --truncate
```

Expected files:

- `product_categories.csv`
- `suppliers.csv`
- `shops.csv`
- `customers.csv`
- `customer_addresses.csv`
- `products.csv`
- `purchases.csv`
- `purchase_items.csv`
- `payments.csv`
- `shipments.csv`

## Resetting and stopping services

Clear application data and reset PostgreSQL counters:

```bash
uv run manage.py clear_database --yes
```

Add `--app-only` to truncate only `customer_service` tables. This operation is
destructive and is intended for local or experimental data.

Stop the Compose services without removing the database volume:

```bash
docker compose down
```

Remove the PostgreSQL volume as well:

```bash
docker compose down -v
```

The final command permanently deletes the Compose-managed database volume.
