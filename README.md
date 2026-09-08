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
```

`config/settings.py` uses `python-dotenv` to load `.env`.

## 3) Start PostgreSQL with Docker Compose (DB only)

`docker-compose.yml` is intentionally limited to PostgreSQL.

```bash
docker compose up -d
```

Verify the container:

```bash
docker compose ps
```

## 4) Install Python dependencies

```bash
uv sync
```

## 5) Run migrations

```bash
python manage.py migrate
```

## 6) Load synthetic data from `data/*.csv`

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

## 7) Reset data in-place

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

## 8) Stop and reset DB (optional)

```bash
docker compose down
docker compose down -v   # removes postgres volume (destructive)
```
