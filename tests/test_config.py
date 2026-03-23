"""Tests for JSONC config parsing."""

import pytest

from src.jcodemunch_mcp.config import _strip_jsonc


class TestJSONCParser:
    """Test JSONC comment stripping."""

    def test_strips_line_comments(self):
        """Should strip // comments to end of line."""
        text = '{"key": "value" // this is a comment\n}'
        result = _strip_jsonc(text)
        assert result == '{"key": "value" \n}'

    def test_strips_line_comment_no_trailing_newline(self):
        """Should strip // comment at end of file."""
        text = '{"key": "value"} // comment'
        result = _strip_jsonc(text)
        assert result == '{"key": "value"} '

    def test_strips_block_comments(self):
        """Should strip /* */ block comments."""
        text = '{"key" /* comment */: "value"}'
        result = _strip_jsonc(text)
        assert result == '{"key" : "value"}'

    def test_strips_multiline_block_comments(self):
        """Should strip multiline /* */ comments."""
        text = '''{
    "key": "value" /* this is
    a multiline
    comment */
}'''
        result = _strip_jsonc(text)
        assert '"key"' in result
        assert 'this is' not in result

    def test_preserves_strings_with_comment_chars(self):
        """Should not strip // or /* inside quoted strings."""
        text = '{"url": "http://example.com", "note": "use /* here*/"}'
        result = _strip_jsonc(text)
        assert result == text  # Should be unchanged
