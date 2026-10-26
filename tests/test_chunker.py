from runbook_ai.chunker import split_into_chunks, load_runbooks

SAMPLE = """# Guide
Intro text.

## Step one
Do A.

## Step two
Do B.
"""


def test_splits_by_heading():
    chunks = split_into_chunks(SAMPLE, source="guide.md")
    assert [c.section for c in chunks] == ["Overview", "Step one", "Step two"]


def test_keeps_source_and_text():
    chunks = split_into_chunks(SAMPLE, source="guide.md")
    assert chunks[1].source == "guide.md"
    assert chunks[1].text == "Do A."


def test_empty_text_gives_no_chunks():
    assert split_into_chunks("", source="x.md") == []


def test_loads_real_runbooks():
    chunks = load_runbooks("data/runbooks")
    assert len(chunks) >= 9