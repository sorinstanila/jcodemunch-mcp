"""Framework profile detection for zero-config indexing.

A FrameworkProfile captures conventions auto-detected from marker files.
Profiles affect **indexing behavior** (what to ignore, what counts as an
entry point, what architectural layers exist) — separate from Context
Providers which affect symbol enrichment.

Detection is cheap (file existence checks only) and runs before directory
discovery so detected ignore patterns are applied during the initial scan.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Layer:
    name: str
    paths: list[str]


@dataclass
class FrameworkProfile:
    name: str
    ignore_patterns: list[str] = field(default_factory=list)
    entry_point_patterns: list[str] = field(default_factory=list)
    layer_definitions: list[Layer] = field(default_factory=list)
    high_value_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

_LARAVEL = FrameworkProfile(
    name="laravel",
    ignore_patterns=[
        "vendor/", "node_modules/", "storage/logs/", "storage/framework/",
        "bootstrap/cache/", ".phpunit.cache/", "*.log", "*.cache",
    ],
    entry_point_patterns=[
        "routes/*.php",
        "app/Console/Commands/*.php",
        "app/Providers/*.php",
        "database/seeders/*.php",
    ],
    layer_definitions=[
        Layer("routes",      ["routes/"]),
        Layer("controllers", ["app/Http/Controllers/"]),
        Layer("requests",    ["app/Http/Requests/"]),
        Layer("services",    ["app/Services/"]),
        Layer("models",      ["app/Models/"]),
        Layer("migrations",  ["database/migrations/"]),
    ],
    high_value_paths=[
        "app/Models/", "app/Http/Controllers/", "routes/", "config/",
    ],
)

_NUXT = FrameworkProfile(
    name="nuxt",
    ignore_patterns=[
        "node_modules/", ".nuxt/", ".output/", "dist/", ".nitro/",
    ],
    entry_point_patterns=[
        "pages/**/*.vue",
        "server/api/**/*.ts",
        "plugins/**/*.ts",
        "middleware/**/*.ts",
    ],
    layer_definitions=[
        Layer("pages",       ["pages/"]),
        Layer("components",  ["components/"]),
        Layer("composables", ["composables/"]),
        Layer("stores",      ["stores/"]),
        Layer("server",      ["server/"]),
        Layer("plugins",     ["plugins/"]),
    ],
    high_value_paths=["pages/", "composables/", "server/api/"],
)

_NEXT = FrameworkProfile(
    name="next",
    ignore_patterns=[
        "node_modules/", ".next/", "out/", "dist/",
    ],
    entry_point_patterns=[
        "app/**/page.tsx",
        "app/**/route.ts",
        "app/layout.tsx",
        "middleware.ts",
    ],
    layer_definitions=[
        Layer("pages",      ["app/"]),
        Layer("components", ["components/"]),
        Layer("lib",        ["lib/"]),
        Layer("api",        ["app/api/"]),
    ],
    high_value_paths=["app/", "lib/", "components/"],
)

_VUE_SPA = FrameworkProfile(
    name="vue-spa",
    ignore_patterns=["node_modules/", "dist/"],
    entry_point_patterns=["src/main.ts", "src/main.js", "src/App.vue"],
    layer_definitions=[
        Layer("views",      ["src/views/"]),
        Layer("components", ["src/components/"]),
        Layer("stores",     ["src/stores/", "src/store/"]),
    ],
    high_value_paths=["src/"],
)

_REACT_SPA = FrameworkProfile(
    name="react-spa",
    ignore_patterns=["node_modules/", "dist/", "build/"],
    entry_point_patterns=["src/index.tsx", "src/index.jsx", "src/App.tsx", "src/App.jsx"],
    layer_definitions=[
        Layer("components", ["src/components/"]),
        Layer("pages",      ["src/pages/"]),
        Layer("hooks",      ["src/hooks/"]),
    ],
    high_value_paths=["src/"],
)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------

def _has_file(folder: Path, *parts: str) -> bool:
    return (folder / Path(*parts)).exists()


def _composer_requires(folder: Path) -> str:
    try:
        data = json.loads((folder / "composer.json").read_text("utf-8", errors="replace"))
        return " ".join(list(data.get("require", {}).keys()) + list(data.get("require-dev", {}).keys()))
    except Exception:
        return ""


def detect_framework(folder_path: Path) -> Optional[FrameworkProfile]:
    """Detect the primary framework in *folder_path* and return a FrameworkProfile.

    Returns ``None`` if no known framework is detected.  Detection is done via
    cheap file-existence checks in priority order — only the first match is
    returned.
    """
    # Laravel (must precede generic PHP)
    if _has_file(folder_path, "artisan"):
        requires = _composer_requires(folder_path)
        if "laravel/framework" in requires:
            logger.info("Framework profile detected: laravel")
            return _LARAVEL

    # Nuxt.js
    if _has_file(folder_path, "nuxt.config.ts") or _has_file(folder_path, "nuxt.config.js"):
        logger.info("Framework profile detected: nuxt")
        return _NUXT

    # Next.js
    if (
        _has_file(folder_path, "next.config.js")
        or _has_file(folder_path, "next.config.ts")
        or _has_file(folder_path, "next.config.mjs")
    ):
        logger.info("Framework profile detected: next")
        return _NEXT

    # Vue SPA (vite + App.vue, no Nuxt marker)
    if _has_file(folder_path, "vite.config.ts") or _has_file(folder_path, "vite.config.js"):
        if _has_file(folder_path, "src", "App.vue"):
            logger.info("Framework profile detected: vue-spa")
            return _VUE_SPA
        if _has_file(folder_path, "src", "App.tsx") or _has_file(folder_path, "src", "App.jsx"):
            logger.info("Framework profile detected: react-spa")
            return _REACT_SPA

    return None


def profile_to_meta(profile: FrameworkProfile) -> dict:
    """Serialize a FrameworkProfile for storage in context_metadata."""
    return {
        "framework_profile": {
            "name": profile.name,
            "entry_point_patterns": profile.entry_point_patterns,
            "layer_definitions": [{"name": l.name, "paths": l.paths} for l in profile.layer_definitions],
            "high_value_paths": profile.high_value_paths,
        }
    }
