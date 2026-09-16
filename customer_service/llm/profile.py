"""Load validated, human-editable domain profiles from Markdown files."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_CONFIG_ROOT = Path(__file__).with_name("config")
_PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_SECTIONS = (
    "Page icon",
    "Page caption",
    "Welcome message",
    "Chat placeholder",
    "Analysis status",
    "Data source label",
)
_PRESENTATION_SECTIONS = (
    "Metric keywords",
    "Time keywords",
    "Category keywords",
)


class DomainProfileError(ValueError):
    """Raised when a selected domain profile is missing or malformed."""


@dataclass(frozen=True)
class PromptCategory:
    """One labeled group of example questions."""

    label: str
    prompts: tuple[str, ...]


@dataclass(frozen=True)
class PresentationHints:
    """Column-name hints used for deterministic metrics and charts."""

    metric_words: tuple[str, ...]
    time_words: tuple[str, ...]
    category_words: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseSourceProfile:
    """One named database and its allowed analytical surface."""

    name: str
    description: str
    allowed_schemas: tuple[str, ...]
    allowed_tables: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseRelationship:
    """Declared semantic relationship between columns in two databases."""

    name: str
    left: str
    right: str
    cardinality: str
    description: str


@dataclass(frozen=True)
class SourceProfile:
    """Machine-readable database topology for a domain profile."""

    databases: tuple[DatabaseSourceProfile, ...]
    relationships: tuple[DatabaseRelationship, ...]

    def database(self, name: str) -> DatabaseSourceProfile | None:
        return next((source for source in self.databases if source.name == name), None)


@dataclass(frozen=True)
class DomainProfile:
    """Runtime content and presentation configuration for one analytical domain."""

    name: str
    app_name: str
    page_icon: str
    page_caption: str
    welcome_message: str
    chat_placeholder: str
    analysis_status: str
    data_source_label: str
    agent_instructions: str
    example_categories: tuple[PromptCategory, ...]
    presentation: PresentationHints
    sources: SourceProfile


def _read(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise DomainProfileError(f"Could not read domain profile file: {path}") from exc
    if not content:
        raise DomainProfileError(f"Domain profile file is empty: {path}")
    return content


def _sections(text: str, path: Path) -> tuple[str | None, dict[str, str]]:
    title: str | None = None
    current: str | None = None
    bodies: dict[str, list[str]] = {}

    for line in text.splitlines():
        if line.startswith("# "):
            if title is not None:
                raise DomainProfileError(f"Duplicate H1 title in {path}")
            title = line[2:].strip()
            if not title:
                raise DomainProfileError(f"Empty H1 title in {path}")
            current = None
            continue
        if line.startswith("## "):
            current = line[3:].strip()
            if not current:
                raise DomainProfileError(f"Empty H2 section name in {path}")
            if current in bodies:
                raise DomainProfileError(f"Duplicate section {current!r} in {path}")
            bodies[current] = []
            continue
        if current is not None:
            bodies[current].append(line)

    return title, {name: "\n".join(lines).strip() for name, lines in bodies.items()}


def _require_sections(
    sections: dict[str, str], required: tuple[str, ...], path: Path
) -> None:
    missing = [name for name in required if not sections.get(name)]
    unknown = sorted(set(sections) - set(required))
    if missing:
        raise DomainProfileError(
            f"Missing required section(s) in {path}: {', '.join(missing)}"
        )
    if unknown:
        raise DomainProfileError(f"Unknown section(s) in {path}: {', '.join(unknown)}")


def _bullet_values(body: str, section: str, path: Path) -> tuple[str, ...]:
    values: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("- "):
            raise DomainProfileError(
                f"Section {section!r} in {path} must contain only bullet items."
            )
        value = stripped[2:].strip().lower()
        if not value:
            raise DomainProfileError(f"Empty bullet in section {section!r} of {path}")
        if value not in values:
            values.append(value)
    if not values:
        raise DomainProfileError(f"Section {section!r} in {path} has no values.")
    return tuple(values)


def _parse_examples(text: str, path: Path) -> tuple[PromptCategory, ...]:
    _, sections = _sections(text, path)
    if not sections:
        raise DomainProfileError(f"No example categories found in {path}")

    categories: list[PromptCategory] = []
    for label, body in sections.items():
        prompts: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                prompt = stripped[2:].strip()
                if not prompt:
                    raise DomainProfileError(
                        f"Empty example prompt in section {label!r} of {path}"
                    )
                prompts.append(prompt)
            elif prompts:
                prompts[-1] = f"{prompts[-1]} {stripped}"
            else:
                raise DomainProfileError(
                    f"Example section {label!r} in {path} must start with a bullet."
                )
        if not prompts:
            raise DomainProfileError(
                f"Example section {label!r} in {path} has no prompts."
            )
        categories.append(PromptCategory(label=label, prompts=tuple(prompts)))
    return tuple(categories)


def _string_list(value: object, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise DomainProfileError(f"{field} in {path} must be a list of strings.")
    values = tuple(dict.fromkeys(item.strip() for item in value))
    return values


def _required_string(payload: dict[str, object], field: str, path: Path) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DomainProfileError(f"{field} in {path} must be a non-empty string.")
    return value.strip()


def _reject_unknown_keys(
    payload: dict[str, object],
    allowed: set[str],
    path: Path,
    context: str,
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise DomainProfileError(
            f"Unknown field(s) in {context} of {path}: {', '.join(unknown)}"
        )


def _load_sources(path: Path) -> SourceProfile:
    try:
        payload = tomllib.loads(_read(path))
    except tomllib.TOMLDecodeError as exc:
        raise DomainProfileError(f"Invalid TOML in {path}: {exc}") from exc
    _reject_unknown_keys(payload, {"databases", "relationships"}, path, "root")

    raw_databases = payload.get("databases")
    if not isinstance(raw_databases, list) or not raw_databases:
        raise DomainProfileError(
            f"{path} must define at least one [[databases]] entry."
        )

    databases: list[DatabaseSourceProfile] = []
    names: set[str] = set()
    for raw_source in raw_databases:
        if not isinstance(raw_source, dict):
            raise DomainProfileError(
                f"Each [[databases]] entry in {path} must be a table."
            )
        _reject_unknown_keys(
            raw_source,
            {"name", "description", "allowed_schemas", "allowed_tables"},
            path,
            "[[databases]] entry",
        )
        name = _required_string(raw_source, "name", path)
        if not _PROFILE_NAME.fullmatch(name):
            raise DomainProfileError(
                f"Database name {name!r} in {path} must contain only lowercase "
                "letters, numbers, hyphens, or underscores."
            )
        if name in names:
            raise DomainProfileError(f"Duplicate database name {name!r} in {path}.")
        names.add(name)
        schemas = _string_list(
            raw_source.get("allowed_schemas"), "allowed_schemas", path
        )
        if not schemas:
            raise DomainProfileError(
                f"Database {name!r} in {path} must allow at least one schema."
            )
        raw_tables = raw_source.get("allowed_tables", [])
        tables = _string_list(raw_tables, "allowed_tables", path) if raw_tables else ()
        databases.append(
            DatabaseSourceProfile(
                name=name,
                description=_required_string(raw_source, "description", path),
                allowed_schemas=schemas,
                allowed_tables=tables,
            )
        )

    relationships: list[DatabaseRelationship] = []
    relationship_names: set[str] = set()
    raw_relationships = payload.get("relationships", [])
    if not isinstance(raw_relationships, list):
        raise DomainProfileError(f"relationships in {path} must be an array of tables.")
    for raw_relationship in raw_relationships:
        if not isinstance(raw_relationship, dict):
            raise DomainProfileError(
                f"Each [[relationships]] entry in {path} must be a table."
            )
        _reject_unknown_keys(
            raw_relationship,
            {"name", "left", "right", "cardinality", "description"},
            path,
            "[[relationships]] entry",
        )
        name = _required_string(raw_relationship, "name", path)
        if name in relationship_names:
            raise DomainProfileError(f"Duplicate relationship name {name!r} in {path}.")
        relationship_names.add(name)
        left = _required_string(raw_relationship, "left", path)
        right = _required_string(raw_relationship, "right", path)
        for endpoint in (left, right):
            parts = endpoint.split(".")
            if len(parts) != 4 or parts[0] not in names or not all(parts):
                raise DomainProfileError(
                    f"Relationship endpoint {endpoint!r} in {path} must be "
                    "database.schema.table.column and reference a declared database."
                )
        cardinality = _required_string(raw_relationship, "cardinality", path)
        if cardinality not in {
            "one-to-one",
            "one-to-many",
            "many-to-one",
            "many-to-many",
        }:
            raise DomainProfileError(
                f"Unsupported cardinality {cardinality!r} in {path}."
            )
        relationships.append(
            DatabaseRelationship(
                name=name,
                left=left,
                right=right,
                cardinality=cardinality,
                description=_required_string(raw_relationship, "description", path),
            )
        )
    return SourceProfile(tuple(databases), tuple(relationships))


def _load_profile(name: str, root: Path) -> DomainProfile:
    if not _PROFILE_NAME.fullmatch(name):
        raise DomainProfileError(
            "DB_AGENT_PROFILE must contain only lowercase letters, numbers, "
            "hyphens, or underscores."
        )

    directory = root / name
    if not directory.is_dir():
        raise DomainProfileError(f"Domain profile {name!r} was not found under {root}.")

    profile_path = directory / "profile.md"
    title, profile_sections = _sections(_read(profile_path), profile_path)
    if not title:
        raise DomainProfileError(f"Missing H1 application name in {profile_path}")
    _require_sections(profile_sections, _PROFILE_SECTIONS, profile_path)

    examples_path = directory / "examples.md"
    presentation_path = directory / "presentation.md"
    _, presentation_sections = _sections(_read(presentation_path), presentation_path)
    _require_sections(presentation_sections, _PRESENTATION_SECTIONS, presentation_path)

    return DomainProfile(
        name=name,
        app_name=title,
        page_icon=profile_sections["Page icon"],
        page_caption=profile_sections["Page caption"],
        welcome_message=profile_sections["Welcome message"],
        chat_placeholder=profile_sections["Chat placeholder"],
        analysis_status=profile_sections["Analysis status"],
        data_source_label=profile_sections["Data source label"],
        agent_instructions=_read(directory / "agent.md"),
        example_categories=_parse_examples(_read(examples_path), examples_path),
        presentation=PresentationHints(
            metric_words=_bullet_values(
                presentation_sections["Metric keywords"],
                "Metric keywords",
                presentation_path,
            ),
            time_words=_bullet_values(
                presentation_sections["Time keywords"],
                "Time keywords",
                presentation_path,
            ),
            category_words=_bullet_values(
                presentation_sections["Category keywords"],
                "Category keywords",
                presentation_path,
            ),
        ),
        sources=_load_sources(directory / "sources.toml"),
    )


@lru_cache(maxsize=16)
def _load_cached(name: str) -> DomainProfile:
    return _load_profile(name, _CONFIG_ROOT)


def load_domain_profile(
    name: str = "sales", *, config_root: Path | None = None
) -> DomainProfile:
    """Load a named profile, optionally from a test-specific configuration root."""

    normalized = name.strip() or "sales"
    if config_root is not None:
        return _load_profile(normalized, Path(config_root))
    return _load_cached(normalized)


def clear_domain_profile_cache() -> None:
    """Clear cached profiles for tests and controlled reloads."""

    _load_cached.cache_clear()
