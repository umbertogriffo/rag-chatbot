from document_loader.format import Format
from document_loader.text_splitter import RecursiveCharacterTextSplitter, create_recursive_text_splitter


def test_recursive_character_text_splitter_keep_separators() -> None:
    split_tags = [",", "."]
    query = "Apple,banana,orange and tomato."
    # keep_separator True
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator=True,
    )
    result = splitter.split_text(query)
    assert result == ["Apple", ",banana", ",orange and tomato", "."]

    # keep_separator False
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=0,
        separators=split_tags,
        keep_separator=False,
    )
    result = splitter.split_text(query)
    assert result == ["Apple", "banana", "orange and tomato"]


def test_iterative_text_splitter() -> None:
    """Test iterative text splitter."""
    text = """Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.

Bye!\n\n-H."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=1)
    output = splitter.split_text(text)
    expected_output = [
        "Hi.",
        "I'm",
        "Harrison.",
        "How? Are?",
        "You?",
        "Okay then",
        "f f f f.",
        "This is a",
        "weird",
        "text to",
        "write,",
        "but gotta",
        "test the",
        "splitting",
        "gggg",
        "some how.",
        "Bye!",
        "-H.",
    ]
    assert output == expected_output


def test_markdown_splitter() -> None:
    splitter = create_recursive_text_splitter(format=Format.MARKDOWN.value, chunk_size=16, chunk_overlap=0)
    code = """
# Sample Document

## Section

This is the content of the section.

## Lists

- Item 1
- Item 2
- Item 3

### Horizontal lines

***********
____________
-------------------

#### Code blocks
```
This is a code block

# sample code
a = 1
b = 2
```
    """
    chunks = splitter.split_text(code)
    assert chunks == [
        "# Sample",
        "Document",
        "## Section",
        "This is the",
        "content of the",
        "section.",
        "## Lists",
        "- Item 1",
        "- Item 2",
        "- Item 3",
        "### Horizontal",
        "lines",
        "***********",
        "____________",
        "---------------",
        "----",
        "#### Code",
        "blocks",
        "```",
        "This is a code",
        "block",
        "# sample code",
        "a = 1\nb = 2",
        "```",
    ]
    # Special test for special characters
    code = "harry\n***\nbabylon is"
    chunks = splitter.split_text(code)
    assert chunks == ["harry\n***", "babylon is"]
