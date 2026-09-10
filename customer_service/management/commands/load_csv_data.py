import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction
from rich.console import Console
from rich.panel import Panel
from tqdm import tqdm

from customer_service.models import (
    Customer,
    CustomerAddress,
    Payment,
    Product,
    ProductCategory,
    Purchase,
    PurchaseItem,
    Shipment,
    Shop,
    Supplier,
)


class Command(BaseCommand):
    help = "Load synthetic CSV data into customer_service models."

    def __init__(self):
        super().__init__()
        self.console = Console()
        self._stage_summary: dict[str, tuple[int, int, int]] = {}

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            default="data",
            help="Directory containing CSV files (default: data)",
        )
        parser.add_argument(
            "--truncate",
            action="store_true",
            help="Delete existing rows before loading",
        )

    def handle(self, *args, **options):
        data_path = Path(options["path"])
        self.console.print(
            Panel.fit(
                "Loading CSV fixtures from customer_service dataset",
                title="[bold cyan]Database Import",
            )
        )
        with transaction.atomic():
            if options["truncate"]:
                self._truncate()

            categories = self._load_product_categories(
                data_path / "product_categories.csv",
            )
            suppliers = self._load_suppliers(data_path / "suppliers.csv")
            shops = self._load_shops(data_path / "shops.csv")
            customers = self._load_customers(data_path / "customers.csv")
            addresses = self._load_addresses(
                data_path / "customer_addresses.csv",
                customers,
            )
            products = self._load_products(
                data_path / "products.csv",
                categories,
                suppliers,
            )
            purchases = self._load_purchases(
                data_path / "purchases.csv",
                customers,
                shops,
                addresses,
            )
            self._load_purchase_items(
                data_path / "purchase_items.csv",
                purchases,
                products,
            )
            self._load_payments(data_path / "payments.csv", purchases)
            self._load_shipments(data_path / "shipments.csv", purchases)

        self._print_summary()
        self.stdout.write(self.style.SUCCESS("CSV data loaded successfully."))

    @staticmethod
    def _to_bool(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _to_decimal(value: str) -> Decimal:
        return Decimal(str(value or "0"))

    @staticmethod
    def _to_datetime(value: str) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(str(value))

    @staticmethod
    def _read_rows(file_path: Path) -> list[dict[str, str]]:
        with file_path.open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _truncate(self) -> None:
        self.console.print(Panel.fit("[yellow]Truncating existing rows[/yellow]"))
        rows = []
        for model in [
            Shipment,
            Payment,
            PurchaseItem,
            Purchase,
            CustomerAddress,
            Product,
            ProductCategory,
            Supplier,
            Shop,
            Customer,
        ]:
            deleted_count, _ = model.objects.all().delete()
            rows.append(f"{model.__name__}: {deleted_count}")
        self.console.print(
            Panel.fit(
                "\n".join(rows),
                title="[yellow]Truncate Summary",
                border_style="yellow",
            )
        )

    def _load_product_categories(self, file_path: Path):
        rows = self._read_rows(file_path)
        by_slug = {}
        created = 0
        skipped = 0

        self.console.print(
            Panel.fit("Loading product categories", title="[blue]Step 1")
        )
        for row in tqdm(rows, desc="product_categories.csv", unit="rows", leave=False):
            slug = row["slug"]
            parent = by_slug.get(row["parent_slug"]) if row["parent_slug"] else None
            category, was_created = ProductCategory.objects.get_or_create(
                slug=slug,
                defaults={"name": row["name"], "parent": parent},
            )
            if category.parent != parent:
                category.parent = parent
                category.save(update_fields=["parent"])
            by_slug[slug] = category
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Product Categories"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Product Categories",
            file_path.name,
            len(rows),
            created,
            skipped,
        )
        return by_slug

    def _load_suppliers(self, file_path: Path):
        rows = self._read_rows(file_path)
        by_code = {}
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading suppliers", title="[blue]Step 2"))
        for row in tqdm(rows, desc="suppliers.csv", unit="rows", leave=False):
            supplier, was_created = Supplier.objects.get_or_create(
                code=row["code"],
                defaults={
                    "name": row["name"],
                    "contact_email": row["contact_email"],
                    "phone": row["phone"],
                    "is_active": self._to_bool(row["is_active"]),
                },
            )
            by_code[row["code"]] = supplier
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Suppliers"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Suppliers",
            file_path.name,
            len(rows),
            created,
            skipped,
        )
        return by_code

    def _load_shops(self, file_path: Path):
        rows = self._read_rows(file_path)
        by_code = {}
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading shops", title="[blue]Step 3"))
        for row in tqdm(rows, desc="shops.csv", unit="rows", leave=False):
            shop, was_created = Shop.objects.get_or_create(
                code=row["code"],
                defaults={
                    "name": row["name"],
                    "email": row["email"],
                    "phone": row["phone"],
                    "timezone": row["timezone"],
                    "is_active": self._to_bool(row["is_active"]),
                },
            )
            by_code[row["code"]] = shop
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Shops"] = (len(rows), created, skipped)
        self._print_stage_detail("Shops", file_path.name, len(rows), created, skipped)
        return by_code

    def _load_customers(self, file_path: Path):
        rows = self._read_rows(file_path)
        by_email = {}
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading customers", title="[blue]Step 4"))
        for row in tqdm(rows, desc="customers.csv", unit="rows", leave=False):
            customer, was_created = Customer.objects.get_or_create(
                email=row["email"],
                defaults={
                    "first_name": row["first_name"],
                    "last_name": row["last_name"],
                    "phone": row["phone"],
                    "is_active": self._to_bool(row["is_active"]),
                    "signup_at": self._to_datetime(row["signup_at"]),
                },
            )
            by_email[row["email"]] = customer
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Customers"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Customers",
            file_path.name,
            len(rows),
            created,
            skipped,
        )
        return by_email

    def _load_addresses(
        self,
        file_path: Path,
        customers_by_email: dict[str, Customer],
    ):
        rows = self._read_rows(file_path)
        by_code = {}
        created = 0
        skipped = 0

        self.console.print(
            Panel.fit("Loading customer addresses", title="[blue]Step 5")
        )
        for row in tqdm(rows, desc="customer_addresses.csv", unit="rows", leave=False):
            customer = customers_by_email[row["customer_email"]]
            address, was_created = CustomerAddress.objects.get_or_create(
                external_code=row["code"],
                defaults={
                    "customer": customer,
                    "label": row["label"],
                    "line1": row["line1"],
                    "line2": row["line2"],
                    "city": row["city"],
                    "state": row["state"],
                    "country": row["country"],
                    "postal_code": row["postal_code"],
                    "is_default": self._to_bool(row["is_default"]),
                },
            )
            by_code[row["code"]] = address
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Customer Addresses"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Customer Addresses",
            file_path.name,
            len(rows),
            created,
            skipped,
        )
        return by_code

    def _load_products(
        self,
        file_path: Path,
        categories_by_slug: dict[str, ProductCategory],
        suppliers_by_code: dict[str, Supplier],
    ):
        rows = self._read_rows(file_path)
        by_sku = {}
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading products", title="[blue]Step 6"))
        for row in tqdm(rows, desc="products.csv", unit="rows", leave=False):
            category = categories_by_slug[row["category_slug"]]
            supplier = suppliers_by_code[row["supplier_code"]]
            product, was_created = Product.objects.get_or_create(
                sku=row["sku"],
                defaults={
                    "name": row["name"],
                    "category": category,
                    "supplier": supplier,
                    "unit_cost": self._to_decimal(row["unit_cost"]),
                    "unit_price": self._to_decimal(row["unit_price"]),
                    "is_active": self._to_bool(row["is_active"]),
                },
            )
            by_sku[row["sku"]] = product
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Products"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Products", file_path.name, len(rows), created, skipped
        )
        return by_sku

    def _load_purchases(
        self,
        file_path: Path,
        customers_by_email: dict[str, Customer],
        shops_by_code: dict[str, Shop],
        addresses_by_code: dict[str, CustomerAddress],
    ):
        rows = self._read_rows(file_path)
        by_order = {}
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading purchases", title="[blue]Step 7"))
        for row in tqdm(rows, desc="purchases.csv", unit="rows", leave=False):
            customer = customers_by_email[row["customer_email"]]
            shop = shops_by_code[row["shop_code"]]
            address_code = row.get("shipping_address_code", "")
            shipping_address = (
                addresses_by_code.get(address_code) if address_code else None
            )
            purchase, was_created = Purchase.objects.get_or_create(
                order_number=row["order_number"],
                defaults={
                    "customer": customer,
                    "shop": shop,
                    "shipping_address": shipping_address,
                    "status": row["status"],
                    "ordered_at": self._to_datetime(row["ordered_at"]),
                    "currency": row["currency"],
                    "subtotal": self._to_decimal(row["subtotal"]),
                    "discount_total": self._to_decimal(row["discount_total"]),
                    "tax_total": self._to_decimal(row["tax_total"]),
                    "total": self._to_decimal(row["total"]),
                },
            )
            by_order[row["order_number"]] = purchase
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Purchases"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Purchases",
            file_path.name,
            len(rows),
            created,
            skipped,
        )
        return by_order

    def _load_purchase_items(
        self,
        file_path: Path,
        purchases_by_order: dict[str, Purchase],
        products_by_sku: dict[str, Product],
    ):
        rows = self._read_rows(file_path)
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading purchase items", title="[blue]Step 8"))
        for row in tqdm(rows, desc="purchase_items.csv", unit="rows", leave=False):
            purchase = purchases_by_order[row["order_number"]]
            product = products_by_sku[row["product_sku"]]
            _, was_created = PurchaseItem.objects.get_or_create(
                purchase=purchase,
                product=product,
                defaults={
                    "quantity": int(row["quantity"]),
                    "unit_price": self._to_decimal(row["unit_price"]),
                    "line_discount": self._to_decimal(row["line_discount"]),
                    "line_total": self._to_decimal(row["line_total"]),
                },
            )
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Purchase Items"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Purchase Items",
            file_path.name,
            len(rows),
            created,
            skipped,
        )

    def _load_payments(self, file_path: Path, purchases_by_order: dict[str, Purchase]):
        rows = self._read_rows(file_path)
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading payments", title="[blue]Step 9"))
        for row in tqdm(rows, desc="payments.csv", unit="rows", leave=False):
            purchase = purchases_by_order[row["order_number"]]
            _, was_created = Payment.objects.get_or_create(
                provider_reference=row["provider_reference"],
                defaults={
                    "purchase": purchase,
                    "method": row["method"],
                    "status": row["status"],
                    "amount": self._to_decimal(row["amount"]),
                    "paid_at": self._to_datetime(row["paid_at"]),
                },
            )
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Payments"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Payments",
            file_path.name,
            len(rows),
            created,
            skipped,
        )

    def _load_shipments(self, file_path: Path, purchases_by_order: dict[str, Purchase]):
        rows = self._read_rows(file_path)
        created = 0
        skipped = 0

        self.console.print(Panel.fit("Loading shipments", title="[blue]Step 10"))
        for row in tqdm(rows, desc="shipments.csv", unit="rows", leave=False):
            purchase = purchases_by_order[row["order_number"]]
            _, was_created = Shipment.objects.get_or_create(
                purchase=purchase,
                defaults={
                    "carrier": row["carrier"],
                    "tracking_number": row["tracking_number"] or None,
                    "status": row["status"],
                    "shipped_at": self._to_datetime(row["shipped_at"]),
                    "delivered_at": self._to_datetime(row["delivered_at"]),
                    "cost": self._to_decimal(row["cost"]),
                },
            )
            if was_created:
                created += 1
            else:
                skipped += 1

        self._stage_summary["Shipments"] = (len(rows), created, skipped)
        self._print_stage_detail(
            "Shipments",
            file_path.name,
            len(rows),
            created,
            skipped,
        )

    def _print_stage_detail(
        self,
        title: str,
        file_name: str,
        processed: int,
        created: int,
        skipped: int,
    ) -> None:
        self.console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]Source[/]: {file_name}",
                        f"[green]Created[/]: {created}",
                        f"[yellow]Skipped[/]: {skipped}",
                        f"[cyan]Processed[/]: {processed}",
                    ]
                ),
                title=f"[blue]{title}",
                border_style="blue",
            )
        )

    def _print_summary(self) -> None:
        summary_lines = ["[bold]Stage totals[/bold]"]
        for stage, (processed, created, skipped) in self._stage_summary.items():
            summary_lines.append(
                f"{stage}: processed={processed} created={created} skipped={skipped}"
            )
        self.console.print(
            Panel.fit(
                "\n".join(summary_lines),
                title="[bold green]Load Summary",
                border_style="green",
            )
        )
