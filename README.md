# DB Agent

This project provides a PostgreSQL-backed Django data model for a customer + purchases domain (shops, customers, products, purchases, payments, shipments, etc.) and a Django management command to load synthetic CSV fixtures.

## 1) Requirements

- Docker and Docker Compose
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (or adjust commands to your Python toolchain)

## 2) Configure environment variables

The project expects environment variables in `.env`. Keep secrets and host-specific values there.

```bash
cp .env.example .env
```

Default file template:

```env
POSTGRES_DB=dbagent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=changeme
POSTGRES_HOST=localhost
POSTGRES_PORT=5656
LLM_HOST_MODEL_PATH=/absolute/path/to/your/llm/model
LLM_CONTAINER_MODEL_PATH=/models/llm
LLM_MODEL_NAME=qwen-3.8-27b
LLM_API_MODE=chat
LLM_DTYPE=bfloat16
LLM_GPU_MEMORY_UTILIZATION=0.6
LLM_HOST=127.0.0.1
LLM_PORT=8000
# LLM_API_BASE_URL=http://192.168.1.50:8000/v1
LLM_TENSOR_PARALLEL_SIZE=1
LLM_API_KEY=EMPTY
LLM_REQUEST_TIMEOUT_SECONDS=60
LLM_GPU_DEVICE_ID=0
```

`config/settings.py` uses `python-dotenv` to load `.env`.

For an LLM running on another machine, set its IPv4 address or hostname with
`LLM_HOST`. `LLM_PORT` is used to build `http://<host>:<port>/v1`:

```env
LLM_HOST=192.168.1.50
LLM_PORT=8000
```

Set `LLM_API_BASE_URL` when you need a complete endpoint, such as HTTPS, IPv6,
a proxy, or a nonstandard API path. It takes precedence over `LLM_HOST` and
`LLM_PORT`:

```env
LLM_API_BASE_URL=https://llm.example.com/v1
```

`LLM_API_MODE` controls which OpenAI-compatible endpoint is used:

- `chat` calls `/v1/chat/completions` with system and user messages. This is the
  default for chat-native models like Qwen, Llama Chat, and similar.
- `completion` calls `/v1/completions` with a raw prompt. Use it for models or
  endpoints that do not define a chat template.

## 3) Start services with Docker Compose

Start both DB and LLM containers together:

```bash
docker compose up -d
```

## 4) Install Python dependencies

```bash
uv sync
```

## 5) Run migrations

```bash
python manage.py migrate
```

## 6) Start LLM service only (optional)

Bring up just the LLM service:

```bash
docker compose up -d llm
```

Verify it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

If startup reports a networking error like `network <id> not found`, reset stale compose state and recreate both containers:

```bash
docker compose down --remove-orphans
docker network prune -f
docker compose up -d --force-recreate
```

### 6b) Manual LLM smoke check

Run a prompt directly from Django once the LLM service is up:

```bash
python manage.py llm_ask "List three shop names."
python manage.py llm_ask --system "You are a data analyst assistant." "Summarize the top 5 customers by total purchases."
```

`llm_ask` is useful for plain LLM prompts. For database questions, use `db_ask`:

```bash
uv run manage.py db_ask "How many active customers are in the system?"
uv run manage.py db_ask "Top 5 products by total purchase amount this month."
```

Run the Streamlit chat interface locally:

```bash
uv run streamlit run gui.py
```

The **Sales Data Analyst** chat uses the same environment, schema checks, and
read-only SQL safeguards as `db_ask`. Answers lead with business findings, then
show relevant KPI cards, charts, and downloadable tables. The SQL and run
details are available under **How this was calculated**.

Follow-up questions use the current chat as context. For example, after a
monthly sales comparison, ask "Which shop drove the change?" or "Now show only
GBP." Chat context is held in memory for the current Streamlit session only;
**Clear chat** starts a new conversation, and old conversations are not saved.

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

Monetary showcase questions keep currencies separate so unrelated values are not
combined into misleading totals.

`db_ask` flow:

- Refreshes schema from PostgreSQL every run.
- Injects schema context into system prompt for grounding.
- Uses a read-only `execute_readonly_sql(sql, purpose)` tool.
- Validates SQL and then executes with caps from `DB_AGENT_*` env values.
- Returns final answer plus SQL execution trace.

The agent loop is implemented as a LangGraph workflow: refresh the allowed
schema, ask the model, validate and execute each read-only SQL tool call, then
produce a final answer. Its in-memory checkpointer enables session-scoped
follow-ups. The OpenAI-compatible client remains the model adapter, and the
existing SQL validator and database limits remain the execution boundary.
`db_ask` remains a single-question command. Programmatic callers can use `agent.ask(question,
thread_id="session-id")` for follow-ups; `force_query=True` requires a fresh
database read when re-running an answer.

For vLLM deployments, tool-calling must be enabled in the server configuration:

```bash
--enable-auto-tool-choice
--tool-call-parser qwen
```

You can adjust behavior with:

- `DB_AGENT_DATABASE_URL`
- `DB_AGENT_ALLOWED_SCHEMAS`
- `DB_AGENT_ALLOWED_TABLES`
- `DB_AGENT_MAX_ROWS`
- `DB_AGENT_MAX_RESULT_CHARS`
- `DB_AGENT_STATEMENT_TIMEOUT_MS`
- `DB_AGENT_MAX_TOOL_CALLS` — maximum SQL tool executions; the final answer turn is not counted
- `DB_AGENT_QUERY_RETRIES`

To switch GPU, set:

- `LLM_GPU_DEVICE_ID=0` for GPU 0
- `LLM_GPU_DEVICE_ID=1` for GPU 1

## 7) Load synthetic data from `data/*.csv`

The command uses:

- `rich` for structured console output
- `tqdm` for row-level progress bars

Use `--truncate` when you want a clean reload:

```bash
python manage.py load_csv_data --path data --truncate
```

Load again without truncating (idempotent inserts via `get_or_create`) if you want to append/refresh:

```bash
python manage.py load_csv_data --path data
```

Expected file inputs:

- `data/product_categories.csv`
- `data/suppliers.csv`
- `data/shops.csv`
- `data/customers.csv`
- `data/customer_addresses.csv`
- `data/products.csv`
- `data/purchases.csv`
- `data/purchase_items.csv`
- `data/payments.csv`
- `data/shipments.csv`

## 8) Reset data in-place

The DB-reset command clears rows and resets PostgreSQL auto-increment counters.

```bash
python manage.py clear_database --yes
```

Helpful options:

- `--app-only`: truncate only `customer_service` tables.
- `--yes`: skip interactive confirmation.

Example:

```bash
python manage.py clear_database --app-only --yes
```

This command is destructive and intended for local/experiment data resets.

## 9) Stop and reset DB (optional)

```bash
docker compose down
docker compose down -v   # removes postgres volume (destructive)
```
