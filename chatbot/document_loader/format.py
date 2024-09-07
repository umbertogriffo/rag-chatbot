from enum import Enum


class Format(Enum):
    MARKDOWN = "markdown"
    HTML = "html"


SUPPORTED_FORMATS = {
    Format.MARKDOWN.value: [
        # First, try to split along Markdown headings (starting with level 2)
        "\n#{1,6} ",
        # Note the alternative syntax for headings (below) is not handled here
        # Heading level 2
        # ---------------
        # End of code block
        "```\n",
        # Horizontal lines
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        # Note that this splitter doesn't handle horizontal lines defined
        # by *three or more* of ***, ---, or ___, but this is not handled
        "\n\n",
        "\n",
        " ",
        "",
    ],
    Format.HTML.value: [
        # First, try to split along HTML tags
        "<body",
        "<div",
        "<p",
        "<br",
        "<li",
        "<h1",
        "<h2",
        "<h3",
        "<h4",
        "<h5",
        "<h6",
        "<span",
        "<table",
        "<tr",
        "<td",
        "<th",
        "<ul",
        "<ol",
        "<header",
        "<footer",
        "<nav",
        # Head
        "<head",
        "<style",
        "<script",
        "<meta",
        "<title",
        "",
    ],
}


def get_separators(format: str):
    separators = SUPPORTED_FORMATS.get(format)

    # validate input
    if separators is None:
        raise KeyError(format + " is a not supported format")

    return separators
