from django.db import models


class TimeStampedModel(models.Model):
    """Shared creation and update timestamps."""

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class Customer(TimeStampedModel):
    """A customer in the commerce system."""

    first_name = models.CharField(max_length=120)
    last_name = models.CharField(max_length=120)
    email = models.EmailField(unique=True, db_index=True)
    phone = models.CharField(max_length=30, unique=True, null=True, blank=True)
    is_active = models.BooleanField(default=True, db_index=True)
    signup_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.first_name} {self.last_name}"

    class Meta:
        ordering = ["last_name", "first_name"]
        indexes = [models.Index(fields=["is_active"])]


class CustomerAddress(TimeStampedModel):
    """Customer mailing and shipping addresses."""

    external_code = models.CharField(max_length=64, unique=True)

    class AddressLabel(models.TextChoices):
        HOME = "home", "Home"
        WORK = "work", "Work"
        OTHER = "other", "Other"

    customer = models.ForeignKey(Customer, on_delete=models.CASCADE, related_name="addresses")
    label = models.CharField(max_length=16, choices=AddressLabel.choices, default=AddressLabel.HOME)
    line1 = models.CharField(max_length=255)
    line2 = models.CharField(max_length=255, blank=True)
    city = models.CharField(max_length=120)
    state = models.CharField(max_length=120)
    country = models.CharField(max_length=120)
    postal_code = models.CharField(max_length=20)
    is_default = models.BooleanField(default=False)

    def __str__(self) -> str:
        return f"{self.customer} - {self.label} ({self.city})"

    class Meta:
        indexes = [
            models.Index(fields=["customer", "is_default"]),
            models.Index(fields=["country", "city"]),
        ]


class Shop(TimeStampedModel):
    """Storefront entity where purchases are placed."""

    name = models.CharField(max_length=120, unique=True)
    code = models.CharField(max_length=32, unique=True)
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=30, blank=True)
    timezone = models.CharField(max_length=40, default="UTC")
    is_active = models.BooleanField(default=True, db_index=True)

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"

    class Meta:
        indexes = [
            models.Index(fields=["code"]),
            models.Index(fields=["is_active"]),
        ]


class Supplier(TimeStampedModel):
    """External suppliers for products."""

    name = models.CharField(max_length=160)
    code = models.CharField(max_length=32, unique=True)
    contact_email = models.EmailField(blank=True)
    phone = models.CharField(max_length=30, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.name

    class Meta:
        ordering = ["name"]
        indexes = [models.Index(fields=["code"])]


class ProductCategory(TimeStampedModel):
    """Product taxonomy."""

    name = models.CharField(max_length=120, unique=True)
    slug = models.SlugField(unique=True)
    parent = models.ForeignKey("self", null=True, blank=True, on_delete=models.SET_NULL, related_name="children")

    def __str__(self) -> str:
        return self.name

    class Meta:
        indexes = [models.Index(fields=["slug"])]


class Product(TimeStampedModel):
    """Sellable product record."""

    name = models.CharField(max_length=200)
    sku = models.CharField(max_length=64, unique=True, db_index=True)
    category = models.ForeignKey(ProductCategory, on_delete=models.PROTECT, related_name="products")
    supplier = models.ForeignKey(Supplier, on_delete=models.PROTECT, related_name="products")
    unit_cost = models.DecimalField(max_digits=12, decimal_places=2)
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    is_active = models.BooleanField(default=True, db_index=True)

    def __str__(self) -> str:
        return f"{self.sku} - {self.name}"

    class Meta:
        indexes = [
            models.Index(fields=["category", "is_active"]),
            models.Index(fields=["supplier"]),
        ]


class Purchase(TimeStampedModel):
    """Purchase order placed by a customer at a shop."""

    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        PAID = "paid", "Paid"
        SHIPPED = "shipped", "Shipped"
        CANCELLED = "cancelled", "Cancelled"
        REFUNDED = "refunded", "Refunded"

    order_number = models.CharField(max_length=64, unique=True, db_index=True)
    customer = models.ForeignKey(Customer, on_delete=models.PROTECT, related_name="purchases")
    shop = models.ForeignKey(Shop, on_delete=models.PROTECT, related_name="purchases")
    shipping_address = models.ForeignKey(
        CustomerAddress,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="purchases",
    )
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.DRAFT, db_index=True)
    ordered_at = models.DateTimeField(db_index=True)
    currency = models.CharField(max_length=3, default="USD")
    subtotal = models.DecimalField(max_digits=12, decimal_places=2)
    discount_total = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    tax_total = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    total = models.DecimalField(max_digits=12, decimal_places=2)

    def __str__(self) -> str:
        return self.order_number

    class Meta:
        indexes = [
            models.Index(fields=["customer", "ordered_at"]),
            models.Index(fields=["shop", "status"]),
            models.Index(fields=["ordered_at", "status"]),
        ]


class PurchaseItem(TimeStampedModel):
    """Line items that belong to a purchase."""

    purchase = models.ForeignKey(Purchase, on_delete=models.PROTECT, related_name="items")
    product = models.ForeignKey(Product, on_delete=models.PROTECT, related_name="purchase_items")
    quantity = models.PositiveIntegerField(default=1)
    unit_price = models.DecimalField(max_digits=12, decimal_places=2)
    line_discount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    line_total = models.DecimalField(max_digits=12, decimal_places=2)

    def __str__(self) -> str:
        return f"{self.purchase.order_number} - {self.product.sku}"

    class Meta:
        indexes = [
            models.Index(fields=["purchase"]),
            models.Index(fields=["product"]),
        ]
        constraints = [
            models.UniqueConstraint(fields=["purchase", "product"], name="unique_purchase_product"),
        ]


class Payment(TimeStampedModel):
    """Payment attempts / settlements for purchases."""

    class Method(models.TextChoices):
        CARD = "card", "Card"
        CASH = "cash", "Cash"
        BANK = "bank_transfer", "Bank Transfer"
        WIRE = "wire", "Wire"
        WALLET = "wallet", "Wallet"

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        AUTHORIZED = "authorized", "Authorized"
        CAPTURED = "captured", "Captured"
        FAILED = "failed", "Failed"
        REFUNDED = "refunded", "Refunded"

    purchase = models.ForeignKey(Purchase, on_delete=models.PROTECT, related_name="payments")
    method = models.CharField(max_length=24, choices=Method.choices)
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.PENDING, db_index=True)
    amount = models.DecimalField(max_digits=12, decimal_places=2)
    provider_reference = models.CharField(max_length=128, unique=True, null=True, blank=True)
    paid_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.purchase.order_number} - {self.amount} {self.method}"

    class Meta:
        indexes = [
            models.Index(fields=["purchase", "status"]),
            models.Index(fields=["provider_reference"]),
        ]


class Shipment(TimeStampedModel):
    """Shipping information for a purchase."""

    class Status(models.TextChoices):
        READY = "ready", "Ready"
        IN_TRANSIT = "in_transit", "In Transit"
        DELIVERED = "delivered", "Delivered"
        CANCELLED = "cancelled", "Cancelled"
        RETURNED = "returned", "Returned"

    purchase = models.ForeignKey(Purchase, on_delete=models.PROTECT, related_name="shipments")
    carrier = models.CharField(max_length=120, blank=True)
    tracking_number = models.CharField(max_length=128, unique=True, null=True, blank=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.READY, db_index=True)
    shipped_at = models.DateTimeField(null=True, blank=True)
    delivered_at = models.DateTimeField(null=True, blank=True)
    cost = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    def __str__(self) -> str:
        return f"{self.purchase.order_number} - {self.status}"

    class Meta:
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["tracking_number"]),
        ]
