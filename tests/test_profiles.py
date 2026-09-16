"""Markdown domain-profile loading and validation tests."""

from pathlib import Path
from tempfile import TemporaryDirectory

from django.test import SimpleTestCase

from customer_service.llm.profile import DomainProfileError, load_domain_profile


def _write_profile(root: Path, *, name: str = "animals") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "profile.md").write_text(
        """# Animal Population Analyst

## Page icon
🐾
## Page caption
Population findings across species and habitats.
## Welcome message
Ask about animal populations and habitats.
## Chat placeholder
Ask a question about wildlife data
## Analysis status
Analyzing wildlife data...
## Data source label
wildlife data
""",
        encoding="utf-8",
    )
    (directory / "agent.md").write_text(
        "You are an animal population analyst speaking to conservation teams.",
        encoding="utf-8",
    )
    (directory / "examples.md").write_text(
        """## Population

- Compare annual population by species.
- Which habitats show the fastest decline?
""",
        encoding="utf-8",
    )
    (directory / "presentation.md").write_text(
        """## Metric keywords
- population
- count
## Time keywords
- year
## Category keywords
- species
- habitat
""",
        encoding="utf-8",
    )
    (directory / "sources.toml").write_text(
        """[[databases]]
name = "animals"
description = "Wildlife population observations."
allowed_schemas = ["public"]
allowed_tables = ["observations", "species"]
""",
        encoding="utf-8",
    )
    return directory


class DomainProfileTests(SimpleTestCase):
    def test_default_sales_profile_preserves_current_experience(self):
        profile = load_domain_profile("sales")

        self.assertEqual(profile.app_name, "Sales Data Analyst")
        self.assertEqual(profile.page_icon, "📊")
        self.assertIn("sales data analyst", profile.agent_instructions)
        self.assertEqual(profile.example_categories[0].label, "Executive")
        self.assertIn("revenue", profile.presentation.metric_words)
        self.assertEqual(profile.sources.databases[0].name, "dbagent")

    def test_alternate_profile_loads_utf8_content_and_wrapped_examples(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = _write_profile(root)
            (directory / "examples.md").write_text(
                """## Population

- Compare annual population by species,
  habitat, and conservation status.
""",
                encoding="utf-8",
            )

            profile = load_domain_profile("animals", config_root=root)

        self.assertEqual(profile.app_name, "Animal Population Analyst")
        self.assertEqual(profile.page_icon, "🐾")
        self.assertEqual(profile.data_source_label, "wildlife data")
        self.assertEqual(
            profile.example_categories[0].prompts,
            (
                "Compare annual population by species, habitat, and conservation status.",
            ),
        )
        self.assertEqual(profile.presentation.category_words, ("species", "habitat"))
        self.assertEqual(
            profile.sources.databases[0].allowed_tables, ("observations", "species")
        )

    def test_rejects_unsafe_or_unknown_profile_names(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(
                DomainProfileError, "DB_AGENT_PROFILE must contain only"
            ):
                load_domain_profile("../sales", config_root=root)
            with self.assertRaisesRegex(DomainProfileError, "was not found"):
                load_domain_profile("missing", config_root=root)

    def test_rejects_missing_required_profile_section(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = _write_profile(root)
            (directory / "profile.md").write_text(
                """# Animal Population Analyst

## Page icon
🐾
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DomainProfileError, "Missing required section"):
                load_domain_profile("animals", config_root=root)

    def test_rejects_malformed_example_and_presentation_lists(self):
        scenarios = (
            ("examples.md", "## Population\nCompare annual population.\n", "start"),
            (
                "presentation.md",
                """## Metric keywords
population
## Time keywords
- year
## Category keywords
- species
""",
                "bullet items",
            ),
        )
        for filename, content, error in scenarios:
            with self.subTest(filename=filename), TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                directory = _write_profile(root)
                (directory / filename).write_text(content, encoding="utf-8")

                with self.assertRaisesRegex(DomainProfileError, error):
                    load_domain_profile("animals", config_root=root)

    def test_rejects_relationship_to_an_undeclared_database(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = _write_profile(root)
            (directory / "sources.toml").write_text(
                """[[databases]]
name = "animals"
description = "Wildlife population observations."
allowed_schemas = ["public"]

[[relationships]]
name = "unknown_source"
left = "animals.public.observations.species_id"
right = "habitats.public.species.id"
cardinality = "many-to-one"
description = "Invalid undeclared source."
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DomainProfileError, "declared database"):
                load_domain_profile("animals", config_root=root)

    def test_rejects_unknown_source_configuration_fields(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            directory = _write_profile(root)
            (directory / "sources.toml").write_text(
                """[[databases]]
name = "animals"
description = "Wildlife population observations."
allowed_schemas = ["public"]
allow_tables = ["observations"]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DomainProfileError, "allow_tables"):
                load_domain_profile("animals", config_root=root)
