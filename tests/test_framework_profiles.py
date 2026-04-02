"""Tests for framework profile detection."""

import json
from pathlib import Path

import pytest

from jcodemunch_mcp.parser.context.framework_profiles import (
    detect_framework,
    profile_to_meta,
)


def _write(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


class TestDetectFramework:
    def test_laravel_detected(self, tmp_path):
        _write(tmp_path / "artisan", "#!/usr/bin/env php")
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"laravel/framework": "^11.0"}
        }))
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "laravel"

    def test_laravel_ignore_patterns(self, tmp_path):
        _write(tmp_path / "artisan", "")
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"laravel/framework": "^11.0"}
        }))
        profile = detect_framework(tmp_path)
        assert "vendor/" in profile.ignore_patterns
        assert "node_modules/" in profile.ignore_patterns

    def test_nuxt_detected_ts(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "export default defineNuxtConfig({})")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "nuxt"

    def test_nuxt_detected_js(self, tmp_path):
        _write(tmp_path / "nuxt.config.js", "module.exports = {}")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "nuxt"

    def test_nuxt_ignore_patterns(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "")
        profile = detect_framework(tmp_path)
        assert ".nuxt/" in profile.ignore_patterns
        assert "node_modules/" in profile.ignore_patterns

    def test_next_detected(self, tmp_path):
        _write(tmp_path / "next.config.js", "module.exports = {}")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "next"

    def test_next_mjs(self, tmp_path):
        _write(tmp_path / "next.config.mjs", "export default {}")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "next"

    def test_vue_spa_detected(self, tmp_path):
        _write(tmp_path / "vite.config.ts", "")
        _write(tmp_path / "src" / "App.vue", "<template></template>")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "vue-spa"

    def test_react_spa_detected(self, tmp_path):
        _write(tmp_path / "vite.config.ts", "")
        _write(tmp_path / "src" / "App.tsx", "export default function App() {}")
        profile = detect_framework(tmp_path)
        assert profile is not None
        assert profile.name == "react-spa"

    def test_no_framework_returns_none(self, tmp_path):
        profile = detect_framework(tmp_path)
        assert profile is None

    def test_laravel_takes_precedence_over_nuxt(self, tmp_path):
        # If somehow both markers exist, Laravel wins (checked first)
        _write(tmp_path / "artisan", "")
        (tmp_path / "composer.json").write_text(json.dumps({
            "require": {"laravel/framework": "^11.0"}
        }))
        _write(tmp_path / "nuxt.config.ts", "")
        profile = detect_framework(tmp_path)
        assert profile.name == "laravel"

    def test_nuxt_not_confused_with_next(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "")
        _write(tmp_path / "next.config.js", "")
        profile = detect_framework(tmp_path)
        # Nuxt is checked first
        assert profile.name == "nuxt"


class TestProfileToMeta:
    def test_meta_contains_name(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "")
        profile = detect_framework(tmp_path)
        meta = profile_to_meta(profile)
        assert "framework_profile" in meta
        assert meta["framework_profile"]["name"] == "nuxt"

    def test_meta_contains_entry_points(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "")
        profile = detect_framework(tmp_path)
        meta = profile_to_meta(profile)
        assert len(meta["framework_profile"]["entry_point_patterns"]) > 0

    def test_meta_contains_layers(self, tmp_path):
        _write(tmp_path / "nuxt.config.ts", "")
        profile = detect_framework(tmp_path)
        meta = profile_to_meta(profile)
        layers = meta["framework_profile"]["layer_definitions"]
        assert any(l["name"] == "pages" for l in layers)
