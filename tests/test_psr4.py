"""Tests for PSR-4 namespace resolution and PHP import graph fixes."""

import json
import pytest
from pathlib import Path

from jcodemunch_mcp.parser.imports import (
    build_psr4_map,
    resolve_php_namespace,
    resolve_specifier,
)


# ---------------------------------------------------------------------------
# build_psr4_map
# ---------------------------------------------------------------------------

class TestBuildPsr4Map:
    def test_basic_autoload(self, tmp_path):
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({
            "autoload": {
                "psr-4": {
                    "App\\": "app/",
                    "Database\\Factories\\": "database/factories/",
                }
            }
        }))
        m = build_psr4_map(str(tmp_path))
        assert m["App\\"] == "app/"
        assert m["Database\\Factories\\"] == "database/factories/"

    def test_autoload_dev_included(self, tmp_path):
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({
            "autoload": {"psr-4": {"App\\": "app/"}},
            "autoload-dev": {"psr-4": {"Tests\\": "tests/"}},
        }))
        m = build_psr4_map(str(tmp_path))
        assert m["App\\"] == "app/"
        assert m["Tests\\"] == "tests/"

    def test_missing_composer_returns_empty(self, tmp_path):
        assert build_psr4_map(str(tmp_path)) == {}

    def test_empty_source_root_returns_empty(self):
        assert build_psr4_map("") == {}

    def test_array_paths_uses_first(self, tmp_path):
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({
            "autoload": {"psr-4": {"App\\": ["app/", "src/"]}}
        }))
        m = build_psr4_map(str(tmp_path))
        assert m["App\\"] == "app/"

    def test_cached_on_second_call(self, tmp_path):
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"autoload": {"psr-4": {"App\\": "app/"}}}))
        m1 = build_psr4_map(str(tmp_path))
        m2 = build_psr4_map(str(tmp_path))
        assert m1 is m2  # same object from cache

    def test_no_psr4_section_returns_empty(self, tmp_path):
        composer = tmp_path / "composer.json"
        composer.write_text(json.dumps({"autoload": {"classmap": ["src/"]}}))
        assert build_psr4_map(str(tmp_path)) == {}


# ---------------------------------------------------------------------------
# resolve_php_namespace
# ---------------------------------------------------------------------------

class TestResolvePhpNamespace:
    PSR4 = {"App\\": "app/", "Tests\\": "tests/"}

    def test_simple_model(self):
        source = {"app/Models/User.php"}
        result = resolve_php_namespace("App\\Models\\User", self.PSR4, source)
        assert result == "app/Models/User.php"

    def test_nested_namespace(self):
        source = {"app/Http/Controllers/Api/UserController.php"}
        result = resolve_php_namespace(
            "App\\Http\\Controllers\\Api\\UserController", self.PSR4, source
        )
        assert result == "app/Http/Controllers/Api/UserController.php"

    def test_tests_prefix(self):
        source = {"tests/Feature/UserTest.php"}
        result = resolve_php_namespace("Tests\\Feature\\UserTest", self.PSR4, source)
        assert result == "tests/Feature/UserTest.php"

    def test_no_match_returns_none(self):
        source = {"app/Models/User.php"}
        result = resolve_php_namespace("Vendor\\Package\\Foo", self.PSR4, source)
        assert result is None

    def test_file_not_in_source_files_returns_none(self):
        source = {"app/Models/Post.php"}  # User.php not here
        result = resolve_php_namespace("App\\Models\\User", self.PSR4, source)
        assert result is None

    def test_longest_prefix_wins(self):
        # More specific prefix should win over less specific
        psr4 = {"App\\": "app/", "App\\Models\\": "models/"}
        source = {"models/User.php"}
        result = resolve_php_namespace("App\\Models\\User", psr4, source)
        assert result == "models/User.php"


# ---------------------------------------------------------------------------
# resolve_specifier with psr4_map
# ---------------------------------------------------------------------------

class TestResolveSpecifierPsr4:
    PSR4 = {"App\\": "app/", "Tests\\": "tests/"}

    def test_php_use_statement_resolved(self):
        source = {"app/Models/User.php", "app/Http/Controllers/UserController.php"}
        result = resolve_specifier(
            "App\\Models\\User",
            "app/Http/Controllers/UserController.php",
            source,
            psr4_map=self.PSR4,
        )
        assert result == "app/Models/User.php"

    def test_no_psr4_map_returns_none(self):
        source = {"app/Models/User.php"}
        result = resolve_specifier(
            "App\\Models\\User",
            "app/Http/Controllers/UserController.php",
            source,
            psr4_map=None,
        )
        assert result is None

    def test_relative_import_still_works(self):
        source = {"src/a.py", "src/b.py"}
        result = resolve_specifier("./b", "src/a.py", source)
        assert result == "src/b.py"

    def test_php_require_path_still_works(self):
        source = {"helpers.php"}
        result = resolve_specifier("helpers.php", "app/helpers.php", source)
        assert result == "helpers.php"

    def test_backslash_specifier_not_tried_for_sql_stem(self):
        """Backslash specifiers should not fall through to SQL stem matching."""
        source = {"app/Models/User.php"}
        # No psr4_map — should return None, not "User.php" via stem match
        result = resolve_specifier("App\\Models\\User", "some.php", source)
        assert result is None


# ---------------------------------------------------------------------------
# Integration: find_importers with PSR-4
# ---------------------------------------------------------------------------

class TestFindImportersPsr4Integration:
    """End-to-end: index a fake PHP project, run find_importers."""

    def test_php_importer_resolved_via_psr4(self, tmp_path):
        """find_importers returns correct results when PSR-4 map is present."""
        import json as _json
        from jcodemunch_mcp.storage.index_store import CodeIndex

        # Build a minimal CodeIndex mimicking a Laravel project
        source_files = [
            "app/Models/User.php",
            "app/Http/Controllers/UserController.php",
        ]
        imports = {
            "app/Http/Controllers/UserController.php": [
                {"specifier": "App\\Models\\User", "names": ["User"]},
            ]
        }
        # composer.json so psr4_map auto-loads
        (tmp_path / "composer.json").write_text(_json.dumps({
            "autoload": {"psr-4": {"App\\": "app/"}}
        }))

        index = CodeIndex(
            repo="local/test",
            owner="local",
            name="test",
            indexed_at="2026-04-02T00:00:00",
            source_files=source_files,
            languages={"php": 2},
            symbols=[],
            imports=imports,
            source_root=str(tmp_path),
        )

        # psr4_map should be auto-loaded
        assert index.psr4_map.get("App\\") == "app/"

        # Simulate what find_importers does
        from jcodemunch_mcp.parser.imports import resolve_specifier

        source_set = frozenset(source_files)
        importers = []
        for src_file, file_imports in index.imports.items():
            for imp in file_imports:
                resolved = resolve_specifier(
                    imp["specifier"], src_file, source_set,
                    index.alias_map, index.psr4_map,
                )
                if resolved == "app/Models/User.php":
                    importers.append(src_file)

        assert importers == ["app/Http/Controllers/UserController.php"]
