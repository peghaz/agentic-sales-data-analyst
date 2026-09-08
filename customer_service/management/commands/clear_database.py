from django.apps import apps
from django.core.management.base import BaseCommand, CommandError
from django.db import connection, transaction
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

EXCLUDED_TABLES = {"django_migrations"}


class Command(BaseCommand):
    help = "Clear database rows and reset auto-increment sequences."

    def __init__(self):
        super().__init__()
        self.console = Console()

    def add_arguments(self, parser):
        parser.add_argument(
            "--yes",
            action="store_true",
            help="Run without interactive confirmation.",
        )
        parser.add_argument(
            "--app-only",
            action="store_true",
            help="Only truncate tables under the customer_service app.",
        )

    def handle(self, *args, **options):
        if connection.vendor != "postgresql":
            raise CommandError("This command supports PostgreSQL only.")

        tables = self._collect_tables(app_only=options["app_only"])
        if not tables:
            self.stdout.write(self.style.WARNING("No tables matched the selected scope."))
            return

        self._show_plan(tables, options["app_only"])

        if not options["yes"] and not self._confirm():
            self.stdout.write(self.style.WARNING("Aborted by user."))
            return

        with transaction.atomic():
            self._truncate_tables(tables)

        self.console.print(
            Panel.fit(
                f"Deleted all rows from {len(tables)} tables and reset indexes.",
                title="[bold green]Database reset complete",
                border_style="green",
            )
        )

    def _collect_tables(self, app_only: bool) -> list[str]:
        if app_only:
            app = apps.get_app_config("customer_service")
            tables = {model._meta.db_table for model in app.get_models()}
        else:
            tables = {
                table
                for table in connection.introspection.table_names()
                if table not in EXCLUDED_TABLES
            }

        return sorted(tables)

    def _show_plan(self, tables: list[str], app_only: bool) -> None:
        scope = "customer_service app tables only" if app_only else "all non-system tables"
        summary = "\n".join([
            f"[bold]Scope[/]: {scope}",
            f"[bold]Tables[/]: {len(tables)}",
            "[dim]First tables: " + ", ".join(tables[:5]) + ("..." if len(tables) > 5 else ""),
            "[bold red]WARNING[/]: This operation is destructive and cannot be undone.",
        ])
        self.console.print(
            Panel.fit(
                summary,
                title="[red]Database Reset Plan",
                border_style="red",
            )
        )

    def _confirm(self) -> bool:
        response = input("Type 'yes' to proceed: ").strip().lower()
        return response == "yes"

    def _truncate_tables(self, tables: list[str]) -> None:
        with connection.cursor() as cursor:
            quoted = ", ".join(connection.ops.quote_name(table) for table in tables)
            for _ in tqdm(range(1), desc="Resetting tables", unit="operation", leave=False):
                cursor.execute(f"TRUNCATE TABLE {quoted} RESTART IDENTITY CASCADE;")
