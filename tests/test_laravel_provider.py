"""Tests for the Laravel context provider."""

import json
from pathlib import Path

import pytest

from jcodemunch_mcp.parser.context.laravel import (
    LaravelContextProvider,
    _guess_table,
    _parse_migration,
    _parse_model,
    _parse_routes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_laravel_project(tmp_path: Path, laravel_version: str = "^11.0") -> None:
    """Create a minimal Laravel project skeleton."""
    _write(tmp_path / "artisan", "#!/usr/bin/env php\n<?php // artisan\n")
    _write(tmp_path / "composer.json", json.dumps({
        "require": {"laravel/framework": laravel_version},
        "autoload": {"psr-4": {"App\\": "app/"}},
    }))


# ---------------------------------------------------------------------------
# _guess_table
# ---------------------------------------------------------------------------

class TestGuessTable:
    def test_regular_noun(self):
        assert _guess_table("User") == "users"

    def test_y_suffix(self):
        assert _guess_table("Category") == "categories"

    def test_already_plural_pattern(self):
        assert _guess_table("Status") == "statuses"

    def test_camel_case(self):
        assert _guess_table("BlogPost") == "blog_posts"


# ---------------------------------------------------------------------------
# _parse_migration
# ---------------------------------------------------------------------------

MIGRATION_PHP = """<?php
use Illuminate\\Database\\Migrations\\Migration;
use Illuminate\\Database\\Schema\\Blueprint;
use Illuminate\\Support\\Facades\\Schema;

return new class extends Migration {
    public function up(): void
    {
        Schema::create('users', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->string('email')->unique();
            $table->foreignId('team_id')->constrained();
            $table->timestamp('email_verified_at')->nullable();
            $table->timestamps();
        });
    }
};
"""


class TestParseMigration:
    def test_parses_table_name(self):
        result = _parse_migration(MIGRATION_PHP)
        assert result is not None
        table_name, columns = result
        assert table_name == "users"

    def test_parses_column_names(self):
        _, columns = _parse_migration(MIGRATION_PHP)
        assert "name" in columns
        assert "email" in columns

    def test_unique_modifier(self):
        _, columns = _parse_migration(MIGRATION_PHP)
        assert "unique" in columns["email"]

    def test_nullable_modifier(self):
        _, columns = _parse_migration(MIGRATION_PHP)
        assert "nullable" in columns["email_verified_at"]

    def test_skips_timestamps_helper(self):
        _, columns = _parse_migration(MIGRATION_PHP)
        # timestamps() is a helper, not a column name
        assert "timestamps" not in columns

    def test_no_schema_returns_none(self):
        assert _parse_migration("<?php echo 'hello';") is None


# ---------------------------------------------------------------------------
# _parse_model
# ---------------------------------------------------------------------------

MODEL_PHP = """<?php
namespace App\\Models;

class User extends Model
{
    protected $fillable = ['name', 'email', 'password'];
    protected $casts = ['email_verified_at' => 'datetime'];

    public function posts()
    {
        return $this->hasMany(Post::class);
    }

    public function team()
    {
        return $this->belongsTo(Team::class);
    }

    public function scopeActive($query)
    {
        return $query->where('active', true);
    }
}
"""


class TestParseModel:
    def test_relationships(self):
        meta = _parse_model(MODEL_PHP)
        assert any("hasMany" in r for r in meta["relationships"])
        assert any("belongsTo" in r for r in meta["relationships"])

    def test_fillable(self):
        meta = _parse_model(MODEL_PHP)
        assert "name" in meta["fillable"]
        assert "email" in meta["fillable"]

    def test_scopes(self):
        meta = _parse_model(MODEL_PHP)
        assert "Active" in meta["scopes"]

    def test_no_relationships(self):
        meta = _parse_model("<?php class Foo extends Model {}")
        assert meta["relationships"] == []


# ---------------------------------------------------------------------------
# _parse_routes
# ---------------------------------------------------------------------------

ROUTES_PHP = """<?php
use App\\Http\\Controllers\\UserController;
use Illuminate\\Support\\Facades\\Route;

Route::get('/users', [UserController::class, 'index'])->name('users.index');
Route::post('/users', [UserController::class, 'store'])->name('users.store');
Route::get('/users/{user}', [UserController::class, 'show'])->name('users.show');
"""


class TestParseRoutes:
    def test_parses_array_style_routes(self, tmp_path):
        routes_dir = tmp_path / "routes"
        routes_dir.mkdir()
        (routes_dir / "api.php").write_text(ROUTES_PHP)
        routes = _parse_routes(routes_dir)
        assert len(routes) == 3
        verbs = [r["verb"] for r in routes]
        assert "GET" in verbs
        assert "POST" in verbs

    def test_route_names_extracted(self, tmp_path):
        routes_dir = tmp_path / "routes"
        routes_dir.mkdir()
        (routes_dir / "api.php").write_text(ROUTES_PHP)
        routes = _parse_routes(routes_dir)
        names = [r["name"] for r in routes]
        assert "users.index" in names
        assert "users.store" in names

    def test_controller_extracted(self, tmp_path):
        routes_dir = tmp_path / "routes"
        routes_dir.mkdir()
        (routes_dir / "api.php").write_text(ROUTES_PHP)
        routes = _parse_routes(routes_dir)
        assert all(r["controller"] == "UserController" for r in routes)

    def test_missing_routes_dir_returns_empty(self, tmp_path):
        routes = _parse_routes(tmp_path / "routes")
        assert routes == []


# ---------------------------------------------------------------------------
# LaravelContextProvider
# ---------------------------------------------------------------------------

class TestLaravelContextProvider:
    def test_detect_laravel_project(self, tmp_path):
        _make_laravel_project(tmp_path)
        provider = LaravelContextProvider()
        assert provider.detect(tmp_path) is True

    def test_detect_no_artisan(self, tmp_path):
        # No artisan file
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"laravel/framework": "^11.0"}
        }))
        provider = LaravelContextProvider()
        assert provider.detect(tmp_path) is False

    def test_detect_wrong_framework(self, tmp_path):
        _write(tmp_path / "artisan", "#!/usr/bin/env php\n")
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"symfony/symfony": "^7.0"}
        }))
        provider = LaravelContextProvider()
        assert provider.detect(tmp_path) is False

    def test_load_model_enrichment(self, tmp_path):
        _make_laravel_project(tmp_path)
        _write(tmp_path / "app" / "Models" / "User.php", MODEL_PHP)

        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("app/Models/User.php")
        assert ctx is not None
        assert "users" in ctx.description or "User" in ctx.description
        assert "eloquent-model" in ctx.tags

    def test_load_migration_columns(self, tmp_path):
        _make_laravel_project(tmp_path)
        _write(tmp_path / "database" / "migrations" / "2024_01_01_create_users_table.php",
               MIGRATION_PHP)

        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        meta = provider.get_metadata()
        assert "laravel_columns" in meta
        assert "users" in meta["laravel_columns"]
        assert "email" in meta["laravel_columns"]["users"]

    def test_load_route_enriches_controller(self, tmp_path):
        _make_laravel_project(tmp_path)
        _write(tmp_path / "routes" / "api.php", ROUTES_PHP)
        _write(tmp_path / "app" / "Http" / "Controllers" / "UserController.php",
               "<?php namespace App\\Http\\Controllers; class UserController {}")

        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("app/Http/Controllers/UserController.php")
        assert ctx is not None
        assert "controller" in ctx.tags

    def test_stats(self, tmp_path):
        _make_laravel_project(tmp_path)
        _write(tmp_path / "app" / "Models" / "User.php", MODEL_PHP)
        _write(tmp_path / "database" / "migrations" / "2024_01_01_create_users_table.php",
               MIGRATION_PHP)

        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        stats = provider.stats()
        assert stats["models"] >= 1
        assert stats["migration_tables"] >= 1

    def test_blade_context_returned(self, tmp_path):
        _make_laravel_project(tmp_path)
        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("resources/views/users/index.blade.php")
        assert ctx is not None
        assert "blade" in ctx.tags

    def test_non_laravel_file_returns_none(self, tmp_path):
        _make_laravel_project(tmp_path)
        provider = LaravelContextProvider()
        assert provider.detect(tmp_path)
        provider.load(tmp_path)

        ctx = provider.get_file_context("src/something.go")
        assert ctx is None
