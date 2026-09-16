"""Load validated, human-editable domain profiles from Markdown files."""

from __future__ import annotations

import re
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
