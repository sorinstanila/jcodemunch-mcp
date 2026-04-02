"""Laravel context provider — detects Laravel projects and enriches symbols with framework metadata.

When a Laravel project is detected (via artisan + composer.json containing laravel/framework),
this provider:
1. Parses routes/*.php for HTTP method, URI, controller, and route name
2. Parses app/Models/*.php for Eloquent relationships, fillable, casts, and scopes
3. Parses database/migrations/*.php for table column definitions (exposed via search_columns)
4. Parses app/Providers/EventServiceProvider.php for event→listener mappings
5. Improves Blade parsing: <x-component> tags and dot-notation view resolution
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .base import ContextProvider, FileContext, register_provider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Route definitions: Route::get('/uri', [Controller::class, 'method'])
# Pattern 1: [ClassName::class, 'method']  Pattern 2: 'ClassName@method'
_ROUTE_ARRAY = re.compile(
    r"""Route\s*::\s*(?P<verb>get|post|put|patch|delete|any|resource|apiResource)\s*"""
    r"""\(\s*['"](?P<uri>[^'"]+)['"]\s*,\s*"""
    r"""\[\s*(?P<class>[\w\\]+)::class\s*,\s*['"](?P<action>\w+)['"]\s*\]""",
    re.MULTILINE,
)
_ROUTE_OLDSTYLE = re.compile(
    r"""Route\s*::\s*(?P<verb>get|post|put|patch|delete|any|resource|apiResource)\s*"""
    r"""\(\s*['"](?P<uri>[^'"]+)['"]\s*,\s*['"](?P<oldstyle>[\w\\@]+)['"]""",
    re.MULTILINE,
)

# Route name chaining: ->name('route.name')
_ROUTE_NAME = re.compile(r"""->name\s*\(\s*['"](?P<name>[^'"]+)['"]\s*\)""")

# Eloquent relationship methods
_ELOQUENT_RELATION = re.compile(
    r"""\$this\s*->\s*(?P<type>hasMany|hasOne|belongsTo|belongsToMany|hasManyThrough|"""
    r"""hasOneThrough|morphMany|morphOne|morphTo|morphToMany|morphedByMany)\s*"""
    r"""\(\s*(?P<model>[\w\\]+)::class""",
    re.MULTILINE,
)

# $fillable, $guarded, $casts, $table
_PHP_PROPERTY_ARRAY = re.compile(
    r"""(?:protected|public|private)\s+(?:static\s+)?\$(?P<prop>fillable|guarded|casts|table|with|hidden)\s*=\s*"""
    r"""(?:\[(?P<arr>[^\]]*)\]|['"'](?P<str>[^'"]+)['"'])\s*;""",
    re.MULTILINE | re.DOTALL,
)

# Local scope methods: public function scopeXxx($query)
_SCOPE_METHOD = re.compile(r"""function\s+scope(?P<name>[A-Z]\w*)\s*\(""", re.MULTILINE)

# Migration Schema::create / Schema::table
_SCHEMA_CREATE = re.compile(
    r"""Schema\s*::\s*(?:create|table)\s*\(\s*['"](?P<table>[^'"]+)['"]""",
    re.MULTILINE,
)

# Column definitions inside Blueprint callback: $table->type('name')
_COLUMN_DEF = re.compile(
    r"""\$table\s*->\s*(?P<type>\w+)\s*\(\s*['"](?P<name>[^'"]+)['"]""",
    re.MULTILINE,
)

# Event::listen or $listen array entries
_EVENT_LISTEN_ARRAY = re.compile(
    r"""['"'](?P<event>[\w\\]+)['"']\s*=>\s*\[([^\]]+)\]""",
    re.MULTILINE | re.DOTALL,
)
_LISTENER_ENTRY = re.compile(r"""['"'](?P<listener>[\w\\]+::class|[\w\\]+)['"']""")

# Blade: <x-component> tag syntax
_BLADE_X_COMPONENT = re.compile(r"""<x-(?P<name>[\w\-.]+)""", re.MULTILINE)

# Blade dot-notation: @extends('layouts.app'), @include('partials.header'), view('users.index')
_BLADE_DOTREF = re.compile(
    r"""(?:@extends|@include(?:When|First)?|@component|view)\s*\(\s*['"](?P<ref>[^'"]+)['"]""",
    re.MULTILINE,
)

# Non-column Blueprint calls to skip (modifiers, not column definitions)
_COLUMN_SKIP_TYPES = frozenset({
    "index", "unique", "primary", "foreign", "foreignId", "dropColumn",
    "dropIndex", "dropPrimary", "dropForeign", "renameColumn", "timestamps",
    "softDeletes", "rememberToken", "engine", "charset", "collation",
    "comment", "after", "constrained", "cascadeOnDelete", "nullOnDelete",
    "restrictOnDelete", "references", "on",
})


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _read_composer_require(folder_path: Path) -> str:
    """Return the raw content of composer.json require section for framework detection."""
    try:
        data = json.loads((folder_path / "composer.json").read_text("utf-8", errors="replace"))
        requires = list(data.get("require", {}).keys()) + list(data.get("require-dev", {}).keys())
        return " ".join(requires)
    except Exception:
        return ""


def _read_php(path: Path) -> str:
    try:
        return path.read_text("utf-8", errors="replace")
    except Exception:
        return ""


def _parse_routes(routes_dir: Path) -> list[dict]:
    """Parse all route files and return a list of route dicts."""
    routes: list[dict] = []
    if not routes_dir.is_dir():
        return routes

    for route_file in sorted(routes_dir.glob("*.php")):
        content = _read_php(route_file)

        for m in _ROUTE_ARRAY.finditer(content):
            remainder = content[m.end():m.end() + 200]
            name_m = _ROUTE_NAME.search(remainder)
            routes.append({
                "verb": m.group("verb").upper(),
                "uri": m.group("uri"),
                "controller": m.group("class").rsplit("\\", 1)[-1],
                "controller_fqn": m.group("class"),
                "action": m.group("action"),
                "name": name_m.group("name") if name_m else "",
                "file": route_file.name,
            })

        for m in _ROUTE_OLDSTYLE.finditer(content):
            old_style = m.group("oldstyle")
            if "@" in old_style:
                parts = old_style.split("@", 1)
                controller, action = parts[0], parts[1]
            else:
                controller, action = old_style, ""
            remainder = content[m.end():m.end() + 200]
            name_m = _ROUTE_NAME.search(remainder)
            routes.append({
                "verb": m.group("verb").upper(),
                "uri": m.group("uri"),
                "controller": controller.rsplit("\\", 1)[-1],
                "controller_fqn": controller,
                "action": action,
                "name": name_m.group("name") if name_m else "",
                "file": route_file.name,
            })

    return routes


def _parse_model(content: str) -> dict:
    """Extract Eloquent model metadata from PHP source."""
    relationships: list[str] = []
    for m in _ELOQUENT_RELATION.finditer(content):
        model_name = m.group("model").rsplit("\\", 1)[-1]
        relationships.append(f"{m.group('type')}({model_name})")

    scopes: list[str] = [m.group("name") for m in _SCOPE_METHOD.finditer(content)]

    props: dict[str, str] = {}
    for m in _PHP_PROPERTY_ARRAY.finditer(content):
        prop = m.group("prop")
        arr_content = m.group("arr") or ""
        str_val = m.group("str") or ""
        if str_val:
            props[prop] = str_val
        elif arr_content:
            items = re.findall(r"""['"]([\w]+)['"]""", arr_content)
            props[prop] = ", ".join(items[:10])

    return {
        "relationships": relationships,
        "scopes": scopes,
        "fillable": props.get("fillable", ""),
        "table": props.get("table", ""),
        "casts": props.get("casts", ""),
    }


def _parse_migration(content: str) -> Optional[tuple[str, dict[str, str]]]:
    """Extract table name and column definitions from a migration file.

    Returns (table_name, {col_name: col_description}) or None.
    """
    m = _SCHEMA_CREATE.search(content)
    if not m:
        return None
    table_name = m.group("table")

    columns: dict[str, str] = {}
    for cm in _COLUMN_DEF.finditer(content):
        col_type = cm.group("type")
        col_name = cm.group("name")
        if col_type in _COLUMN_SKIP_TYPES:
            continue
        # Check for common modifiers in the short trailing text
        trailing = content[cm.end():cm.end() + 120]
        mods: list[str] = []
        if "->nullable()" in trailing:
            mods.append("nullable")
        if "->unique()" in trailing:
            mods.append("unique")
        if "->unsigned()" in trailing:
            mods.append("unsigned")
        desc = col_type
        if mods:
            desc += ", " + ", ".join(mods)
        columns[col_name] = desc

    return table_name, columns


def _parse_events(content: str) -> list[str]:
    """Extract event→listener mappings from EventServiceProvider."""
    mappings: list[str] = []
    for m in _EVENT_LISTEN_ARRAY.finditer(content):
        event = m.group("event").rsplit("\\", 1)[-1]
        listeners_block = m.group(2)
        listeners = [lm.group("listener").rsplit("\\", 1)[-1].replace("::class", "")
                     for lm in _LISTENER_ENTRY.finditer(listeners_block)]
        if listeners:
            mappings.append(f"{event} → {', '.join(listeners)}")
    return mappings


# ---------------------------------------------------------------------------
# Context Provider
# ---------------------------------------------------------------------------

@register_provider
class LaravelContextProvider(ContextProvider):
    """Context provider for Laravel projects.

    Detects artisan + laravel/framework in composer.json, then parses:
    - routes/*.php  → HTTP method, URI, controller, route name
    - app/Models/*  → Eloquent relationships, fillable, scopes
    - database/migrations/* → table column definitions (for search_columns)
    - app/Providers/EventServiceProvider.php → event→listener mappings
    """

    def __init__(self) -> None:
        self._folder: Optional[Path] = None
        # file stem → FileContext
        self._file_contexts: dict[str, FileContext] = {}
        # route lookup: controller class name → list of route summaries
        self._controller_routes: dict[str, list[str]] = {}
        # migration columns: table_name → {col_name: col_description}
        self._table_columns: dict[str, dict[str, str]] = {}
        # blade view resolution: relative folder path
        self._views_path: Optional[Path] = None

    @property
    def name(self) -> str:
        return "laravel"

    def detect(self, folder_path: Path) -> bool:
        if not (folder_path / "artisan").exists():
            return False
        requires = _read_composer_require(folder_path)
        if "laravel/framework" not in requires:
            return False
        self._folder = folder_path
        return True

    def load(self, folder_path: Path) -> None:
        if self._folder is None:
            self._folder = folder_path

        routes = _parse_routes(folder_path / "routes")
        logger.info("Laravel: parsed %d routes", len(routes))

        # Build controller → routes lookup
        for route in routes:
            ctrl = route["controller"]
            if ctrl:
                summary = f"{route['verb']} {route['uri']}"
                if route["name"]:
                    summary += f" ({route['name']})"
                self._controller_routes.setdefault(ctrl, []).append(summary)

        # Parse Eloquent models
        models_dir = folder_path / "app" / "Models"
        model_count = 0
        if models_dir.is_dir():
            for php_file in sorted(models_dir.glob("*.php")):
                content = _read_php(php_file)
                meta = _parse_model(content)
                stem = php_file.stem

                # Determine table name: explicit $table or pluralized stem (simple heuristic)
                table_name = meta["table"] or _guess_table(stem)

                props: dict[str, str] = {"table": table_name}
                if meta["relationships"]:
                    props["relationships"] = ", ".join(meta["relationships"][:8])
                if meta["scopes"]:
                    props["scopes"] = ", ".join(meta["scopes"][:6])
                if meta["fillable"]:
                    props["fillable"] = meta["fillable"]

                route_strs = self._controller_routes.get(stem + "Controller", [])
                if route_strs:
                    props["routes"] = "; ".join(route_strs[:4])

                self._file_contexts[stem] = FileContext(
                    description=f"Eloquent model for `{table_name}` table",
                    tags=["eloquent-model", f"{table_name}-table"],
                    properties=props,
                )
                model_count += 1

        logger.info("Laravel: parsed %d models", model_count)

        # Parse migrations for column metadata
        migrations_dir = folder_path / "database" / "migrations"
        migration_count = 0
        if migrations_dir.is_dir():
            for php_file in sorted(migrations_dir.glob("*.php")):
                content = _read_php(php_file)
                result = _parse_migration(content)
                if result:
                    table_name, columns = result
                    if columns:
                        self._table_columns[table_name] = columns
                        migration_count += 1

        logger.info("Laravel: parsed %d migration tables", migration_count)

        # Parse EventServiceProvider
        esp_path = folder_path / "app" / "Providers" / "EventServiceProvider.php"
        if esp_path.exists():
            content = _read_php(esp_path)
            events = _parse_events(content)
            if events:
                self._file_contexts["EventServiceProvider"] = FileContext(
                    description="Laravel event→listener registry",
                    tags=["event-provider", "events"],
                    properties={"event_mappings": "; ".join(events[:10])},
                )
                logger.info("Laravel: parsed %d event mappings", len(events))

        # Parse controllers: enrich with route info
        controllers_dir = folder_path / "app" / "Http" / "Controllers"
        if controllers_dir.is_dir():
            for php_file in sorted(controllers_dir.rglob("*.php")):
                stem = php_file.stem
                route_strs = self._controller_routes.get(stem, [])
                if route_strs:
                    self._file_contexts[stem] = FileContext(
                        description=f"Laravel controller: handles {', '.join(route_strs[:3])}",
                        tags=["controller", "http"],
                        properties={"routes": "; ".join(route_strs[:6])},
                    )

        # Store views path for Blade resolution
        self._views_path = folder_path / "resources" / "views"

    def get_file_context(self, file_path: str) -> Optional[FileContext]:
        stem = Path(file_path).stem
        ctx = self._file_contexts.get(stem)
        if ctx:
            return ctx

        # Blade: enrich with component usage info
        if file_path.endswith(".blade.php") or ".blade." in file_path:
            return self._blade_context(file_path)

        return None

    def _blade_context(self, file_path: str) -> Optional[FileContext]:
        """Produce context for Blade template files."""
        stem = Path(file_path).stem.replace(".blade", "")
        return FileContext(
            description=f"Blade template: {stem}",
            tags=["blade", "template"],
        )

    def get_metadata(self) -> dict:
        """Expose migration column metadata for search_columns."""
        if not self._table_columns:
            return {}
        return {"laravel_columns": self._table_columns}

    def stats(self) -> dict:
        return {
            "models": sum(1 for ctx in self._file_contexts.values()
                         if "eloquent-model" in ctx.tags),
            "controllers_with_routes": len(self._controller_routes),
            "migration_tables": len(self._table_columns),
        }


def _guess_table(model_stem: str) -> str:
    """Simple pluralization heuristic for model class name → table name."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", model_stem).lower()
    if name.endswith("y") and not name.endswith(("ay", "ey", "iy", "oy", "uy")):
        return name[:-1] + "ies"
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return name + "es"
    return name + "s"
