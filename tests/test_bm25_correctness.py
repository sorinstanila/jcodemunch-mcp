"""Tests for BM25 correctness — T10 (avgdl inflation) and T11 (canonical corpus)."""

import math

from jcodemunch_mcp.tools.search_symbols import (
    _sym_tokens,
    _compute_bm25,
    _identity_score,
    _bm25_breakdown,
    _FIELD_REPS,
)


# ---------------------------------------------------------------------------
# T10: _dl uses unique token count, not repeated bag
# ---------------------------------------------------------------------------

class TestBM25DlUnique:

    def test_dl_is_unique_token_count(self):
        """_dl must equal len(set(tokens)), not len(tokens)."""
        sym = {"name": "parse_file", "signature": "", "summary": "", "docstring": "", "keywords": []}
        tokens = _sym_tokens(sym)
        # tokens = _tokenize("parse_file") * 3 = ["parse", "file"] * 3 → 6 items
        assert len(tokens) == 6
        assert sym["_dl"] == 2  # unique: {"parse", "file"}

    def test_dl_not_inflated_by_field_reps(self):
        """Repeated tokens from field weights must not inflate _dl."""
        sym = {
            "name": "foo",
            "signature": "foo(x)",
            "summary": "foo does something",
            "docstring": "",
            "keywords": [],
        }
        tokens = _sym_tokens(sym)
        # Raw repeated bag has more entries than unique tokens
        assert len(tokens) > sym["_dl"]
        assert sym["_dl"] == len(set(tokens))

    def test_dl_matches_set_of_tokens(self):
        """_dl must always equal len(set(_tokens)) for any symbol."""
        sym = {
            "name": "process_data",
            "signature": "process_data(df: DataFrame) -> dict",
            "summary": "process a data frame",
            "docstring": "Process the input data and return results.",
            "keywords": ["data", "process"],
        }
        tokens = _sym_tokens(sym)
        assert sym["_dl"] == len(set(tokens))

    def test_dl_no_inflation_compared_to_pre_t10(self):
        """_dl must be strictly less than len(tokens) when tokens have repeats."""
        sym = {"name": "handler", "signature": "handler(req)", "summary": "", "docstring": "", "keywords": []}
        tokens = _sym_tokens(sym)
        # name tokens appear 3×, signature tokens appear 2× — repeated bag > unique
        if len(set(tokens)) < len(tokens):
            assert sym["_dl"] < len(tokens), "_dl should not equal the inflated bag length"

    def test_dl_consistent_on_second_call(self):
        """Fast-path (_tokens cached) does not reset _dl to wrong value."""
        sym = {"name": "connect", "signature": "connect(host, port)", "summary": "", "docstring": "", "keywords": []}
        _sym_tokens(sym)                    # first call — computes everything
        dl_first = sym["_dl"]
        _sym_tokens(sym)                    # second call — fast path
        # _dl is not recomputed on fast path, but should stay correct
        assert sym["_dl"] == dl_first == len(set(sym["_tokens"]))


# ---------------------------------------------------------------------------
# T10 + T11: _compute_bm25 uses canonical unique-token avgdl
# ---------------------------------------------------------------------------

class TestComputeBM25Canonical:

    def _make_sym(self, name: str, summary: str = "") -> dict:
        return {"name": name, "signature": "", "summary": summary, "docstring": "", "keywords": []}

    def test_avgdl_not_inflated(self):
        """avgdl must be computed from unique token counts, not repeated bag lengths."""
        syms = [self._make_sym("parse"), self._make_sym("render"), self._make_sym("connect")]
        _, avgdl, _ = _compute_bm25(syms)
        # Each symbol: name token × 3 field reps → unique tokens per symbol
        # avgdl should reflect unique count, NOT the 3× repeated bag length.
        # With jCore tokenizer, stems/expansions may add tokens (e.g. "render" -> ["render", "rend"]).
        # Key invariant: avgdl < 3.0 (not inflated by field reps)
        assert avgdl < 3.0, f"avgdl inflated by field reps: {avgdl}"

    def test_compute_bm25_rewrites_dl(self):
        """_compute_bm25 must overwrite stale _dl values on retained symbols (T11)."""
        sym = self._make_sym("alpha")
        # Manually inject a stale _dl (simulating a symbol cached before T10)
        _sym_tokens(sym)
        sym["_dl"] = 999  # stale value from old len(tokens) formula
        _compute_bm25([sym])
        assert sym["_dl"] != 999, "_compute_bm25 must rewrite stale _dl"
        assert sym["_dl"] == len(set(sym["_tokens"]))

    def test_bm25_corpus_size_equals_symbol_count(self):
        """After _compute_bm25, inverted index covers exactly N symbols (no doubling)."""
        syms = [self._make_sym(f"func_{i}") for i in range(5)]
        idf, avgdl, inverted = _compute_bm25(syms)
        # Every symbol index 0-4 must appear in the inverted index
        all_indices = set()
        for idxs in inverted.values():
            all_indices.update(idxs)
        assert all_indices == set(range(5)), "Inverted index must cover exactly the 5 symbols"

    def test_dl_consistent_after_compute(self):
        """All symbols must have _dl == len(set(_tokens)) after _compute_bm25 runs."""
        syms = [
            self._make_sym("alpha", summary="first function"),
            self._make_sym("beta", summary="second function here"),
            self._make_sym("gamma"),
        ]
        _compute_bm25(syms)
        for sym in syms:
            assert sym["_dl"] == len(set(sym["_tokens"])), (
                f"Symbol {sym['name']}: _dl={sym['_dl']} != {len(set(sym['_tokens']))}"
            )

    def test_idf_formula_correct(self):
        """idf uses Robertson formula: log((N - df + 0.5) / (df + 0.5) + 1)."""
        # 2 symbols both containing "shared", 1 symbol with "unique"
        sym_a = {"name": "shared_alpha", "signature": "", "summary": "", "docstring": "", "keywords": []}
        sym_b = {"name": "shared_beta", "signature": "", "summary": "", "docstring": "", "keywords": []}
        sym_c = {"name": "unique_func", "signature": "", "summary": "", "docstring": "", "keywords": []}
        idf, _, _ = _compute_bm25([sym_a, sym_b, sym_c])
        N = 3
        # "shared" appears in 2 docs
        expected_shared = math.log((N - 2 + 0.5) / (2 + 0.5) + 1.0)
        assert abs(idf.get("shared", 0) - expected_shared) < 1e-9
        # "unique" appears in 1 doc
        expected_unique = math.log((N - 1 + 0.5) / (1 + 0.5) + 1.0)
        assert abs(idf.get("unique", 0) - expected_unique) < 1e-9

    def test_retained_symbol_stale_dl_corrected(self):
        """Simulates a retained symbol with pre-T10 stale _dl surviving into a BM25 rebuild."""
        sym = {"name": "process", "signature": "process(x, y)", "summary": "", "docstring": "", "keywords": []}
        # Pre-T10: _tokens cached, _tf cached, _dl set to inflated len(tokens)
        tokens = _sym_tokens(sym)
        inflated_dl = len(tokens)
        sym["_dl"] = inflated_dl  # inject stale pre-T10 value
        # Simulate a BM25 rebuild (e.g., after deferred summarize or server restart)
        _compute_bm25([sym])
        # T11: _compute_bm25 must have corrected _dl
        canonical_dl = len(set(sym["_tokens"]))
        assert sym["_dl"] == canonical_dl, (
            f"After rebuild, _dl={sym['_dl']} should be canonical {canonical_dl}, "
            f"not inflated {inflated_dl}"
        )


# ---------------------------------------------------------------------------
# Identity channel tests (Gap 3 partial)
# ---------------------------------------------------------------------------

class TestIdentityChannel:
    """Tests for the _identity_score function that replaces the 50.0 hack."""

    def test_exact_name_match(self):
        sym = {"name": "process_data", "id": "src/utils.py::process_data"}
        assert _identity_score(sym, "process_data") == 50.0

    def test_exact_id_match(self):
        sym = {"name": "process_data", "id": "src/utils.py::process_data"}
        assert _identity_score(sym, "src/utils.py::process_data") == 50.0

    def test_name_prefix_match(self):
        sym = {"name": "process_data", "id": "src/utils.py::process_data"}
        score = _identity_score(sym, "process")
        assert score == 30.0

    def test_id_segment_match(self):
        sym = {"name": "IndexStore", "id": "src/storage/index_store.py::IndexStore"}
        score = _identity_score(sym, "index_store")
        # "index_store" appears in the ID "src/storage/index_store.py::indexstore"
        assert score == 20.0

    def test_no_match(self):
        sym = {"name": "process_data", "id": "src/utils.py::process_data"}
        assert _identity_score(sym, "completely_unrelated") == 0.0

    def test_empty_query(self):
        sym = {"name": "process_data", "id": "src/utils.py::process_data"}
        assert _identity_score(sym, "") == 0.0

    def test_case_insensitive(self):
        sym = {"name": "ProcessData", "id": "src/utils.py::ProcessData"}
        assert _identity_score(sym, "processdata") == 50.0

    def test_raw_query_exact_snake_case_match(self):
        sym = {"name": "build_ui", "id": "src/ui.py::build_ui"}
        assert _identity_score(sym, "build ui", raw_query="build_ui") == 50.0

    def test_raw_query_exact_camel_case_match(self):
        sym = {"name": "ProcessData", "id": "src/utils.py::ProcessData"}
        assert _identity_score(sym, "process data", raw_query="ProcessData") == 50.0

    def test_exact_beats_prefix(self):
        """Exact match (50) must score higher than prefix match (30)."""
        sym = {"name": "get_symbol", "id": "src/tools.py::get_symbol"}
        exact = _identity_score(sym, "get_symbol")
        prefix = _identity_score(sym, "get_sym")
        assert exact > prefix

    def test_prefix_beats_segment(self):
        """Prefix match (30) must score higher than segment match (20)."""
        sym = {"name": "get_symbol_source", "id": "src/tools/get_symbol.py::get_symbol_source"}
        prefix = _identity_score(sym, "get_symbol")
        segment = _identity_score({"name": "IndexStore", "id": "src/tools/get_symbol.py::IndexStore"}, "get_symbol")
        assert prefix > segment

    def test_breakdown_includes_identity(self):
        """Debug breakdown should include 'identity' and 'identity_type' fields."""
        sym = {
            "name": "helper",
            "signature": "helper()",
            "summary": "",
            "docstring": "",
            "keywords": [],
        }
        _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym])
        breakdown = _bm25_breakdown(sym, ["helper"], idf, avgdl)
        assert "identity" in breakdown
        assert "identity_type" in breakdown
        assert breakdown["identity"] == 50.0
        assert breakdown["identity_type"] == "exact"

    def test_breakdown_no_match_identity_type(self):
        """When no identity match, identity_type should be 'none'."""
        sym = {
            "name": "helper",
            "signature": "helper()",
            "summary": "",
            "docstring": "",
            "keywords": [],
        }
        _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym])
        breakdown = _bm25_breakdown(sym, ["unrelated"], idf, avgdl)
        assert breakdown["identity"] == 0.0
        assert breakdown["identity_type"] == "none"

    def test_breakdown_uses_raw_query_for_snake_case_exact_match(self):
        sym = {
            "name": "_build_left_pane_cache",
            "signature": "_build_left_pane_cache()",
            "summary": "",
            "docstring": "",
            "keywords": [],
        }
        _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym])
        breakdown = _bm25_breakdown(sym, ["build", "left", "pane", "cache"], idf, avgdl, raw_query="_build_left_pane_cache")
        assert breakdown["identity"] == 50.0
        assert breakdown["identity_type"] == "exact"


# ---------------------------------------------------------------------------
# Identity channel on BM25 path (raw_query regression tests)
# ---------------------------------------------------------------------------

class TestIdentityChannelBM25Path:
    """Tests that BM25 path preserves exact snake_case/camelCase matching via raw_query."""

    def _make_sym(self, name: str, summary: str = "") -> dict:
        return {"name": name, "signature": "", "summary": summary, "docstring": "", "keywords": []}

    def test_bm25_score_exact_snake_case_match(self):
        """BM25 path should use raw_query for exact snake_case identity match."""
        from jcodemunch_mcp.tools.search_symbols import _bm25_score, _tokenize

        sym = self._make_sym("_build_left_pane_cache")
        _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym])

        # Tokenize the raw query as the BM25 path would
        raw_query = "_build_left_pane_cache"
        query_terms = _tokenize(raw_query)
        # The bug: _identity_score gets " ".join(query_terms) which loses underscore structure
        # After fix: _identity_score should also receive raw_query="_build_left_pane_cache"
        score = _bm25_score(sym, query_terms, idf, avgdl, raw_query=raw_query)
        # Without raw_query fix, identity score is 0 (no exact match)
        # With raw_query fix, identity score should be 50 (exact name match)
        assert score >= 50.0, f"Expected exact identity match (50+), got {score}"

    def test_bm25_score_exact_camel_case_match(self):
        """BM25 path should use raw_query for exact camelCase identity match."""
        from jcodemunch_mcp.tools.search_symbols import _bm25_score, _tokenize

        sym = self._make_sym("renderRow")
        _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym])

        raw_query = "renderRow"
        query_terms = _tokenize(raw_query)
        score = _bm25_score(sym, query_terms, idf, avgdl, raw_query=raw_query)
        # After fix: should get exact match bonus
        assert score >= 50.0, f"Expected exact identity match (50+), got {score}"

    def test_exact_name_beats_partial_in_bm25(self):
        """Exact name match should score higher than partial match in BM25 path."""
        from jcodemunch_mcp.tools.search_symbols import _bm25_score, _tokenize

        sym_exact = self._make_sym("build_ui")
        sym_partial = self._make_sym("set_ui")
        for sym in [sym_exact, sym_partial]:
            _sym_tokens(sym)
        idf, avgdl, _ = _compute_bm25([sym_exact, sym_partial])

        raw_query = "build_ui"
        query_terms = _tokenize(raw_query)
        exact_score = _bm25_score(sym_exact, query_terms, idf, avgdl, raw_query=raw_query)
        partial_score = _bm25_score(sym_partial, query_terms, idf, avgdl, raw_query=raw_query)
        # After fix: exact match should outrank non-matching
        assert exact_score > partial_score, (
            f"Exact 'build_ui' ({exact_score}) should beat 'set_ui' ({partial_score})"
        )
