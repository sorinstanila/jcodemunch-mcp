"""Unit tests for the SessionJournal get_context method with sort options."""

import pytest
from src.jcodemunch_mcp.tools.session_journal import get_journal


def reset_journal_for_test():
    """Reset the journal singleton for testing."""
    from src.jcodemunch_mcp.tools import session_journal
    with session_journal._journal_lock:
        session_journal._journal = None  # Reset singleton to fresh state


def test_get_context_sort_by_frequency():
    """Test that get_context sorts all components by frequency when sort_by='frequency'."""
    reset_journal_for_test()
    journal = get_journal()
    
    # Record activities to test frequency sorting
    
    # For files: add with different read counts
    journal.record_read('least_read.py', 'get_file_outline')
    journal.record_read('most_read.py', 'get_file_outline') 
    journal.record_read('most_read.py', 'get_file_content')  # 2nd read
    journal.record_read('most_read.py', 'get_file_content')  # 3rd read - most frequent
    journal.record_read('moderately_read.py', 'get_file_outline')
    journal.record_read('moderately_read.py', 'get_file_content')  # 2nd read
    
    # For queries: add with different frequencies
    journal.record_search('least_searched', 5)
    journal.record_search('most_searched', 2)  # Search first
    journal.record_search('most_searched', 3)  # Search again - increases frequency to 2
    journal.record_search('moderately_searched', 4)
    
    # For edits: add with different edit counts
    journal.record_edit('least_edited.py')
    journal.record_edit('most_edited.py')
    journal.record_edit('most_edited.py')  # 2nd edit
    journal.record_edit('most_edited.py')  # 3rd edit - most frequent  
    journal.record_edit('moderately_edited.py')
    journal.record_edit('moderately_edited.py')  # 2nd edit

    # Test frequency-based sorting
    context = journal.get_context(
        max_files=10, 
        max_queries=10, 
        max_edits=10, 
        sort_by='frequency'
    )
    
    # Verify files are sorted by read count (descending)
    files_accessed = context["files_accessed"]
    # Filter our test files only
    test_files = [f for f in files_accessed if f["file"] in ["most_read.py", "moderately_read.py", "least_read.py"]]
    assert len(test_files) == 3
    assert test_files[0]["file"] == "most_read.py"
    assert test_files[0]["reads"] == 3
    assert test_files[1]["file"] == "moderately_read.py"
    assert test_files[1]["reads"] == 2
    assert test_files[2]["file"] == "least_read.py"
    assert test_files[2]["reads"] == 1
    
    # Verify queries are sorted by count frequency (descending) 
    recent_searches = context["recent_searches"]
    test_searches = [s for s in recent_searches if s["query"] in ["most_searched", "moderately_searched", "least_searched"]]
    assert len(test_searches) == 3
    assert test_searches[0]["query"] == "most_searched"
    assert test_searches[0]["count"] == 2  # Most searched (2 times)
    
    # Verify edits are sorted by edit count (descending)
    files_edited = context["files_edited"]
    test_edits = [e for e in files_edited if e["file"] in ["most_edited.py", "moderately_edited.py", "least_edited.py"]]
    assert len(test_edits) == 3
    assert test_edits[0]["file"] == "most_edited.py"
    assert test_edits[0]["edits"] == 3
    assert test_edits[1]["file"] == "moderately_edited.py"
    assert test_edits[1]["edits"] == 2
    assert test_edits[2]["file"] == "least_edited.py"
    assert test_edits[2]["edits"] == 1


def test_get_context_sort_by_timestamp():
    """Test that get_context sorts by timestamp when sort_by='timestamp'."""
    reset_journal_for_test()
    journal = get_journal()
    
    # Record activities in a specific order to test timestamp sorting
    journal.record_read('first_read.py', 'get_file_outline') 
    journal.record_read('second_read.py', 'get_file_content')  # Later timestamp
    
    # Test default behavior (timestamp sorting) 
    context_default = journal.get_context(max_files=10, max_queries=10, max_edits=10)
    
    # Test explicit timestamp sorting
    context_timestamp = journal.get_context(
        max_files=10, 
        max_queries=10, 
        max_edits=10, 
        sort_by='timestamp'
    )
    
    # Both should have the same number of files
    assert len(context_default["files_accessed"]) == len(context_timestamp["files_accessed"])
    
    # Verify that both behave the same way for timestamp sorting
    assert context_default["files_accessed"] == context_timestamp["files_accessed"]


def test_get_context_default_sort_is_timestamp():
    """Test that default sorting is by timestamp."""
    reset_journal_for_test()
    journal = get_journal()
    
    # Record activities
    journal.record_read('first_read.py', 'get_file_outline')
    journal.record_read('second_read.py', 'get_file_content')
    
    # Compare default vs explicit timestamp sorting
    context_default = journal.get_context(max_files=10, max_queries=10, max_edits=10)
    context_timestamp = journal.get_context(
        max_files=10, 
        max_queries=10, 
        max_edits=10, 
        sort_by='timestamp'
    )
    
    # Results should be equivalent when both using timestamp sort
    assert len(context_default["files_accessed"]) == len(context_timestamp["files_accessed"])